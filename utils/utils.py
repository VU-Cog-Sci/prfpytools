import os
import numpy as np
import imageio
import nibabel as nb
from joblib import Parallel, delayed
import time

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
#        import matplotlib.pyplot as pl
#        fig = pl.figure()
#        pl.imshow(downsampled_img)
#        fig.close()
        
        
        if downsampled_img[:,:,0].shape != design_matrix[...,0].shape:
            print("please choose a n_pix value that is a divisor of "+str(img.shape[0]))

        # binarize image into dm matrix
        # assumes: standard RGB255 format; only colors present in image are black, white, grey, red, green.
        design_matrix[:, :, img_number][np.where(((downsampled_img[:, :, 0] == 0) & (
            downsampled_img[:, :, 1] == 0)) | ((downsampled_img[:, :, 0] == 255) & (downsampled_img[:, :, 1] == 255)))] = 1
    
        design_matrix[:, :, img_number][np.where(((downsampled_img[:, :, 0] == downsampled_img[:, :, 1]) & (
            downsampled_img[:, :, 1] == downsampled_img[:, :, 2]) & (downsampled_img[:,:,0] != 127) ))] = 1
    
    #clipping edges
    design_matrix[:2,:,:] = 0
    design_matrix[-2:,:,:] = 0
    design_matrix[:,0,:] = 0
    design_matrix[:,-1,:] = 0
    print("Design matrix completed")
    
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
        
        for i,task_name in enumerate(task_names):
            iso_diff = iso_full - np.mean(tc_full[...,late_iso_dict[task_name]], axis=-1)
            tc_dict[hemi][i] += iso_diff[...,np.newaxis]
           
        tc_full_iso_dict[hemi]=np.concatenate(tuple(tc_dict[hemi]), axis=-1)
        
    tc_full_iso = np.concatenate((tc_full_iso_dict['L'], tc_full_iso_dict['R']), axis=0)
    
    tc_full_iso_nonzerovar_dict['orig_data_shape'] = {'R_shape':tc_full_iso_dict['R'].shape, 'L_shape':tc_full_iso_dict['L'].shape}
    tc_full_iso_nonzerovar_dict['nonzerovar_mask'] = np.var(tc_full_iso, axis=-1)>0
    tc_full_iso_nonzerovar_dict['tc'] = tc_full_iso[tc_full_iso_nonzerovar_dict['nonzerovar_mask']]    
    
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
        data_ses_1 = nb.load(opj(data_path, 'fmriprep/'+subj+'/ses-1/func/'+subj+'_ses-1_task-'+task_name+'_run-1_space-'+fitting_space+'_desc-preproc_bold.nii.gz')).get_data()[...,discard_volumes:]
        data_ses_2 = nb.load(opj(data_path, 'fmriprep/'+subj+'/ses-2/func/'+subj+'_ses-2_task-'+task_name+'_run-1_space-'+fitting_space+'_desc-preproc_bold.nii.gz')).get_data()[...,discard_volumes:]
        
        data_ses_1 = np.reshape(data_ses_1,(-1, data_ses_1.shape[-1]))
        data_ses_2 = np.reshape(data_ses_2,(-1, data_ses_2.shape[-1]))
        
        #parallel fit is slower?
#        start = time.time()
#        tc_ses_1 = Parallel(n_jobs=6, verbose=True)(
#            delayed(sgfilter_predictions)(data_ses_1[vox,:],
#                                                window_length=window_length)
#            for vox in range(data_ses_1.shape[0]))
#        tc_ses_1 = np.array(tc_ses_1)    
#
#        print(time.time()-start)
#        start=time.time()
#        tc_ses_2 = sgfilter_predictions(data_ses_2,
#                                                window_length=window_length)
#        print(time.time()-start)
        tc_ses_1 = sgfilter_predictions(data_ses_1,
                                                window_length=window_length) 
        tc_ses_2 = sgfilter_predictions(data_ses_2,
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
    
    for i,task_name in enumerate(task_names):
        iso_diff = iso_full - np.mean(tc_full[...,late_iso_dict[task_name]], axis=-1)
        tc_list[i] += iso_diff[...,np.newaxis]
    
    
    tc_full_iso=np.concatenate(tuple(tc_list), axis=-1)
                

    #exclude timecourses with zero variance
    tc_full_iso_nonzerovar_dict['brain_mask_shape'] = final_mask.shape
    tc_full_iso_nonzerovar_dict['nonzerovar_brain_mask'] = np.ravel(final_mask) & (np.var(tc_full_iso, axis=-1)>0)
    tc_full_iso_nonzerovar_dict['tc'] = tc_full_iso[tc_full_iso_nonzerovar_dict['nonzerovar_brain_mask']]

    return tc_full_iso_nonzerovar_dict