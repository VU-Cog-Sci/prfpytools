import os
import numpy as np
import imageio
import nibabel as nb

opj = os.path.join

from prfpy.timecourse import sgfilter_predictions


def create_dm_from_screenshots(screenshot_path,
                               n_pix=40,
                               ):

    image_list = os.listdir(screenshot_path)

    # there is one more MR image than screenshot
    design_matrix = np.zeros((n_pix, n_pix, 1+len(image_list)))
    for image_file in image_list:
        # assuming last three numbers before .png are the screenshot number
        img_number = int(image_file[-7:-4])-1
        # subtract one to start from zero
        img = imageio.imread(os.path.join(screenshot_path, image_file))
        # make it square
        if img.shape[0] != img.shape[1]:
            offset = int((img.shape[1]-img.shape[0])/2)
            img = img[:, offset:(offset+img.shape[0])]

        # downsample
        downsampling_constant = int(img.shape[1]/n_pix)
        downsampled_img = img[::downsampling_constant, ::downsampling_constant]

        # binarize image into dm matrix
        # assumes: standard RGB255 format; only colors present in image are black, white, grey, red, green.
        design_matrix[:, :, img_number][np.where(((downsampled_img[:, :, 0] == 0) & (
            downsampled_img[:, :, 1] == 0)) | ((downsampled_img[:, :, 0] == 255) & (downsampled_img[:, :, 1] == 255)))] = 1
    
    
    return design_matrix

def prepare_surface_data(subj,
                         task_names,
                         discard_volumes,
                         window_length,
                         late_iso_dict,
                         data_path,
                         fitting_space):

    tc_dict = {}
    tc_full_iso_dict ={}
    tc_full_iso_nonzerovar_dict = {}
    for hemi in ['L', 'R']:
        tc_dict[hemi] = []
        for task_name in task_names:
            data_ses1 = nb.load(opj(data_path, 'fmriprep/'+subj+'/ses-1/func/'+subj+'_ses-1_task-'+task_name+'_run-1_space-'+fitting_space+'_hemi-'+hemi+'.func.gii'))
            data_ses2 = nb.load(opj(data_path, 'fmriprep/'+subj+'/ses-2/func/'+subj+'_ses-2_task-'+task_name+'_run-1_space-'+fitting_space+'_hemi-'+hemi+'.func.gii'))
                   
            tc_ses_1 = sgfilter_predictions(np.array([arr.data for arr in data_ses1.darrays]).T[...,discard_volumes:],
                                             window_length=window_length)
    
            tc_ses_2 = sgfilter_predictions(np.array([arr.data for arr in data_ses2.darrays]).T[...,discard_volumes:],
                                             window_length=window_length)
    
            tc_dict[hemi].append((tc_ses_1+tc_ses_2)/2.0)
        
        #when scanning sub-001 i mistakenly set the length of the 4F scan to 147, while it should have been 145
        #therefore, there are two extra images at the end to discard in that time series.
        #from sub-002 onwards, this was corrected.
        if subj == 'sub-001':
            tc_dict[hemi][3] = tc_dict[hemi][3][...,:-2]
            
        tc_full=np.concatenate(tuple(tc_dict[hemi]), axis=-1)
        
        #shift timeseries so they have the same average value in proper baseline periods across conditions
        iso_full = np.mean(tc_full[...,late_iso_dict['periods']], axis=-1)
        
        for i,task in enumerate(task_names):
            iso_diff = iso_full - np.mean(tc_full[...,late_iso_dict[task_name]], axis=-1)
            tc_dict[hemi][i] += iso_diff[...,np.newaxis]
           
        tc_full_iso_dict[hemi]=np.concatenate(tuple(tc_dict[hemi]), axis=-1)
        
    tc_full_iso = np.concatenate((tc_full_iso_dict['L'], tc_full_iso_dict['R']), axis=0)
    
    tc_full_iso_nonzerovar_dict['orig_data_shape'] = {'R_shape':tc_full_iso_dict['R'].shape, 'L_shape':tc_full_iso_dict['L'].shape}
    tc_full_iso_nonzerovar_dict['indices'] = np.where(np.var(tc_full_iso, axis=-1)>0)[0]
    tc_full_iso_nonzerovar_dict['tc'] = tc_full_iso[np.where(np.var(tc_full_iso, axis=-1)>0)]    
    
    return tc_full_iso_nonzerovar_dict


def prepare_volume_data(subj,
                         task_names,
                         discard_volumes,
                         window_length,
                         late_iso_dict,
                         data_path,
                         fitting_space):
    
    #############preparing the data (VOLUME FITTING)
    #create a single brain mask in epi space 
    for i,task_name in enumerate(task_names):
        mask_ses_1 = nb.load(opj(data_path,'fmriprep/'+subj+'/ses-1/func/'+subj+'_ses-1_task-'+task_name+'_run-1_space-'+fitting_space+'_desc-brain_mask.nii.gz')).get_data().astype(bool)
        mask_ses_2 = nb.load(opj(data_path, 'fmriprep/'+subj+'/ses-2/func/'+subj+'_ses-2_task-'+task_name+'_run-1_space-'+fitting_space+'_desc-brain_mask.nii.gz')).get_data().astype(bool)
        
        if i==0:
            final_mask = np.ones_like(mask_ses_1).astype(bool)
            
        final_mask = final_mask & mask_ses_1 & mask_ses_2
        
    
    tc_list = []
    tc_full_iso_nonzerovar_dict = {}
    
    for task_name in task_names:
        data_ses_1 = nb.load(opj(data_path, 'fmriprep/'+subj+'/ses-1/func/'+subj+'_ses-1_task-'+task_name+'_run-1_space-'+fitting_space+'_desc-preproc_bold.nii.gz'))
        data_ses_2 = nb.load(opj(data_path, 'fmriprep/'+subj+'/ses-2/func/'+subj+'_ses-2_task-'+task_name+'_run-1_space-'+fitting_space+'_desc-preproc_bold.nii.gz'))
           
        tc_ses_1 = sgfilter_predictions(data_ses_1.get_data()[...,discard_volumes:],
                                                window_length=window_length)
        tc_ses_2 = sgfilter_predictions(data_ses_2.get_data()[...,discard_volumes:],
                                                window_length=window_length)
            
        tc_list.append((tc_ses_1+tc_ses_2)/2.0)
        

    #when scanning sub-001 i mistakenly set the length of the 4F-task scan to 147, while it should have been 145
    #therefore, there are two extra images at the end to discard in that time series.
    #from sub-002 onwards, this was corrected.
    if subj == 'sub-001':
        tc_list[3] = tc_list[3][...,:-2]
        
    
    tc_full=np.concatenate(tuple(tc_list), axis=-1)
    
    
    #shift timeseries so they have the same average value in baseline periods across conditions
    iso_full = np.mean(tc_full[...,late_iso_dict['periods']], axis=-1)
    
    for i,task in enumerate(task_names):
        iso_diff = iso_full - np.mean(tc_full[...,late_iso_dict[task_name]], axis=-1)
        tc_list[i] += iso_diff[...,np.newaxis]
    
    
    tc_full_iso=np.concatenate(tuple(tc_list), axis=-1)
                

    #exclude timecourses with zero variance
    tc_full_iso_nonzerovar_dict['orig_data_shape'] = tc_full_iso.shape
    tc_full_iso_nonzerovar_dict['indices'] = final_mask & np.where(np.var(tc_full_iso, axis=-1)>0)[0]
    tc_full_iso_nonzerovar_dict['tc'] = tc_full_iso[final_mask & np.where(np.var(tc_full_iso, axis=-1)>0)]

    return tc_full_iso_nonzerovar_dict