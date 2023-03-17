import os
import numpy as np
import h5py
import nibabel as nb
from collections import defaultdict as dd
from pathlib import Path
import matplotlib.image as mpimg
from scipy.stats import zscore

opj = os.path.join

from prfpy.timecourse import filter_predictions
from prfpy.stimulus import PRFStimulus2D


def roi_mask(roi, array):
    array_2 = np.zeros(array.shape).astype('bool')
    array_2[roi] = True
    masked_array = array * array_2
    return masked_array

def inverse_roi_mask(roi, array):
    array_2 = np.ones(array.shape).astype('bool')
    array_2[roi] = False
    masked_array = array * array_2
    return masked_array

def create_dm_from_screenshots(screenshot_path,
                               n_pix=40,
                               dm_edges_clipping=[0,0,0,0]
                               ):

    image_list = os.listdir(screenshot_path)

    # there is one more MR image than screenshot
    design_matrix = np.zeros((n_pix, n_pix, 1+len(image_list)))
    for image_file in image_list:
        
        # assuming last three numbers before .png are the screenshot number
        img_number = int(image_file[-7:-4])-1
        # subtract one to start from zero
        img = (255*mpimg.imread(os.path.join(screenshot_path, image_file))).astype('int')
        # make it square
        if img.shape[0] != img.shape[1]:
            offset = int((img.shape[1]-img.shape[0])/2)
            img = img[:, offset:(offset+img.shape[0])]

        assert img.shape[0]%n_pix == 0, f"please choose a n_pix value that is a divisor of {str(img.shape[0])}"
        # downsample
        downsampling_constant = int(img.shape[0]/n_pix)
        downsampled_img = img[::downsampling_constant, ::downsampling_constant]
#        import matplotlib.pyplot as pl
#        fig = pl.figure()
#        pl.imshow(downsampled_img)
#        fig.close()
        

        # binarize image into dm matrix
        # assumes: standard RGB255 format; only colors present in image are black, white, grey, red, green.
        design_matrix[:, :, img_number][np.where(((downsampled_img[:, :, 0] == 0) & (
            downsampled_img[:, :, 1] == 0)) | ((downsampled_img[:, :, 0] == 255) & (downsampled_img[:, :, 1] == 255)))] = 1
    
        design_matrix[:, :, img_number][np.where(((downsampled_img[:, :, 0] == downsampled_img[:, :, 1]) & (
            downsampled_img[:, :, 1] == downsampled_img[:, :, 2]) & (downsampled_img[:,:,0] != 127) ))] = 1
    
    #clipping edges
    #top, bottom, left, right
    design_matrix[:dm_edges_clipping[0],:,:] = 0
    design_matrix[(design_matrix.shape[0]-dm_edges_clipping[1]):,:,:] = 0
    design_matrix[:,:dm_edges_clipping[2],:] = 0
    design_matrix[:,(design_matrix.shape[0]-dm_edges_clipping[3]):,:] = 0
    print("Design matrix completed")
    
    return design_matrix

def create_full_stim(screenshot_paths,
                n_pix,
                discard_volumes,
                dm_edges_clipping,
                screen_size_cm,
                screen_distance_cm,
                TR,
                task_names,
                normalize_integral_dx):
    dm_list = []

    for i, task_name in enumerate(task_names):
        # create stimulus
        if task_name in screenshot_paths[i]:
            #this is for hcp-format design matrix
            if screenshot_paths[i].endswith('hdf5'):
                with h5py.File(screenshot_paths[i], 'r') as f:
                    dm_task = np.array(f.get('stim')).T
                    dm_task /= dm_task.max()
                    
                    #assert dm_task.shape[0]%n_pix == 0, f"please choose a n_pix value that is a divisor of {str(dm_task.shape[0])}"
                    if dm_task.shape[0]%n_pix != 0:
                        print(f"warning: n_pix is not a divisor of original DM size. The true downsampled size is: {dm_task[::int(dm_task.shape[0]/n_pix),0,0].shape[0]}")
                        
                dm_list.append(dm_task[::int(dm_task.shape[0]/n_pix),::int(dm_task.shape[0]/n_pix),:])
                
            #this is for screenshots
            else:
                
                dm_list.append(create_dm_from_screenshots(screenshot_paths[i],
                                                  n_pix,
                                                  dm_edges_clipping)[..., discard_volumes:])
    
    
    task_lengths = [dm.shape[-1] for dm in dm_list]    
    
    dm_full = np.concatenate(tuple(dm_list), axis=-1)

    # late-empty DM periods (for calculation of BOLD baseline)
    shifted_dm = np.zeros_like(dm_full)
    
    # use timepoints where bar was gone from at least 7 TRs (this is a heuristic approximation)
    shifted_dm[..., 7:] = dm_full[..., :-7]
    
    late_iso_dict = {}
    late_iso_dict['periods'] = np.where((np.sum(dm_full, axis=(0, 1)) == 0) & (
        np.sum(shifted_dm, axis=(0, 1)) == 0))[0]
    
    start=0
    for i, task_name in enumerate(task_names):
        stop=start+task_lengths[i]
        if task_name not in screenshot_paths[i]:
            print("WARNING: check that screenshot paths and task names are in the same order")
        late_iso_dict[task_name] = late_iso_dict['periods'][np.where((late_iso_dict['periods']>=start) & (late_iso_dict['periods']<stop))]-start
            
        start+=task_lengths[i]

    prf_stim = PRFStimulus2D(screen_size_cm=screen_size_cm,
                             screen_distance_cm=screen_distance_cm,
                             design_matrix=dm_full,
                             TR=TR,
                             task_lengths=task_lengths,
                             task_names=task_names,
                             late_iso_dict=late_iso_dict,
                             normalize_integral_dx=normalize_integral_dx)


    return prf_stim


def prepare_data(subj,
                 prf_stim,
                 test_prf_stim,
                 
                 discard_volumes,
                 min_percent_var,
                 fix_bold_baseline,
                 
                 filter_type,
                 
                 filter_params,                

                 data_path,
                 fitting_space,
                 data_scaling,
                 roi_idx,
                 save_raw_timecourse,
                 
                 crossvalidate,
                 fit_runs,
                 fit_task,
                 save_noise_ceiling,

                 session = 'ses-*',
                 pybest = False):

    if fitting_space == 'fsaverage' or fitting_space == 'fsnative':
        hemis = ['L', 'R']
    elif fitting_space == 'HCP':
        hemis = ['LR']
        
    tc_dict = dd(lambda:dd(dict))
    raw_tcs = False
    
    for hemi in hemis:
        for task_name in prf_stim.task_names:
            tc_task = []
            if fitting_space == 'fsaverage' or fitting_space == 'fsnative':
                if pybest:
                    tc_paths = sorted(Path(opj(data_path,'pybest',subj,'unzscored')).glob(f"{subj}_{session}_task-{task_name}*run-*_space-{fitting_space}_hemi-{hemi}*bold.npy"))
                else:                   
                    tc_paths = sorted(Path(opj(data_path,'fmriprep',subj)).glob(opj('**',f"{subj}_{session}_task-{task_name}*run-*_space-{fitting_space}_hemi-{hemi}*.func.gii")))
            elif fitting_space == 'HCP': 
                tc_paths = sorted(Path(opj(data_path,subj)).glob(opj('**',f"tfMRI_RET{task_name}*_7T_*_Atlas_1.6mm_MSMAll_hp2000_clean.dtseries.nii")))
            
            print(f"For task {task_name}, session {session}, hemisphere {hemi}, of subject {subj}, a total of {len(tc_paths)} runs were found.")
            
            if fit_runs is not None and (len(fit_runs)>len(tc_paths) or np.any(np.array(fit_runs)>=len(tc_paths))):
                print(f"{fit_runs} fit_runs requested but only {len(tc_paths)} runs were found.")
                raise ValueError

            if fit_runs is None:
                #if CV over tasks, or if no CV, use all runs
                fit_runs = np.arange(len(tc_paths))
            
                
            for tc_path in [tc_paths[run] for run in fit_runs]:
                
                if fitting_space == 'fsaverage' or fitting_space == 'fsnative':
                    if pybest:
                        tc_run_data = np.load(str(tc_path)).T[...,discard_volumes:]
                    else:
                        tc_run = nb.load(str(tc_path))
                        tc_run_data = np.array([arr.data for arr in tc_run.darrays]).T[...,discard_volumes:]
                elif fitting_space == 'HCP':
                    #cortex only HCP data
                    tc_run = nb.load(str(tc_path))
                    tc_run_data = np.array(tc_run.get_data()).T[:118584,discard_volumes:]
                        
                #no need to pass further args, only filtering 1 condition
                if data_scaling in ["zsc", "z-score"]:
                    tc_task.append(zscore(filter_predictions(tc_run_data,
                                                 filter_type=filter_type,
                                                 filter_params=filter_params), axis=0))
                elif data_scaling in ["psc", "percent_signal_change"]:
                    tc_task.append(filter_predictions(tc_run_data,
                             filter_type=filter_type,
                             filter_params=filter_params))
                    tc_task[-1] *= (100/np.mean(tc_task[-1], axis=-1))[...,np.newaxis]
                else:
                    print("Using raw data")
                    raw_tcs = True
                    tc_task.append(filter_predictions(tc_run_data,
                             filter_type=filter_type,
                             filter_params=filter_params))                        

            
            tc_dict[hemi][task_name]['timecourse'] = np.mean(tc_task, axis=0)
            tc_dict[hemi][task_name]['baseline'] = np.median(tc_dict[hemi][task_name]['timecourse'][...,prf_stim.late_iso_dict[task_name]],
                                               axis=-1)
            

        #this part needs updating
        if crossvalidate:
            for task_name in test_prf_stim.task_names:                          
                tc_task = []
                if fitting_space == 'fsaverage' or fitting_space == 'fsnative':
                    tc_paths = sorted(Path(opj(data_path,'fmriprep',subj)).glob(opj('**',subj+'_ses-*_task-'+task_name+'_run-*_space-'+fitting_space+'_hemi-'+hemi+'*.func.gii')))
                elif fitting_space == 'HCP': 
                    tc_paths = sorted(Path(opj(data_path,subj)).glob(opj('**',f"tfMRI_RET{task_name}*_7T_*_Atlas_1.6mm_MSMAll_hp2000_clean.dtseries.nii")))
                
                print("For task "+task_name+", hemisphere "+hemi+" of subject "+subj+", a total of "+str(len(tc_paths))+" runs were found.")
                
                if fit_task is not None:
                    #if CV is over tasks, can use all runs for test data as well
                    cv_runs = np.arange(len(tc_paths))
                else:
                    cv_runs = [run for run in np.arange(len(tc_paths)) if run not in fit_runs]

                for tc_path in [tc_paths[run] for run in cv_runs]:
                    tc_run = nb.load(str(tc_path))
                    #no need to pass further args, only filtering 1 condition
                    if fitting_space == 'fsaverage' or fitting_space == 'fsnative':
                        tc_run_data = np.array([arr.data for arr in tc_run.darrays]).T[...,discard_volumes:]
                    elif fitting_space == 'HCP':
                        #cortex only HCP data
                        tc_run_data = np.array(tc_run.get_data()).T[:118584,discard_volumes:]                    
                    
                    if data_scaling in ["zsc", "z-score"]:
                        tc_task.append(zscore(filter_predictions(tc_run_data,
                                                     filter_type=filter_type,
                                                     filter_params=filter_params), axis=0))
                    elif data_scaling in ["psc", "percent_signal_change"]:
                        tc_task.append(filter_predictions(tc_run_data,
                                 filter_type=filter_type,
                                 filter_params=filter_params))
                        tc_task[-1] *= (100/np.mean(tc_task[-1], axis=-1))[...,np.newaxis]
                    else:
                        print("Using raw data")
                        raw_tcs = True
                        tc_task.append(filter_predictions(tc_run_data,
                                 filter_type=filter_type,
                                 filter_params=filter_params))   
    
                    #when scanning sub-001 i mistakenly set the length of the 4F scan to 147, while it should have been 145
                    #therefore, there are two extra images at the end to discard in that time series.
                    #from sub-002 onwards, this was corrected.
                    if subj == 'sub-001' and task_name=='4F':
                        tc_task[-1] = tc_task[-1][...,:-2]

                
                tc_dict[hemi][task_name]['timecourse_test'] = np.mean(tc_task, axis=0)
                tc_dict[hemi][task_name]['baseline_test'] = np.median(tc_dict[hemi][task_name]['timecourse_test'][...,test_prf_stim.late_iso_dict[task_name]],
                                                   axis=-1)

    
        #shift timeseries so they have the same average value in proper baseline periods across conditions
        tc_dict[hemi]['median_baseline'] = np.median([tc_dict[hemi][task_name]['baseline'] for task_name in prf_stim.task_names], axis=0)

        for task_name in prf_stim.task_names:
            iso_diff = tc_dict[hemi]['median_baseline'] - tc_dict[hemi][task_name]['baseline']
            tc_dict[hemi][task_name]['timecourse'] += iso_diff[...,np.newaxis]
           
        tc_dict[hemi]['full_iso']=np.concatenate(tuple([tc_dict[hemi][task_name]['timecourse'] for task_name in prf_stim.task_names]), axis=-1)

        if crossvalidate:
            tc_dict[hemi]['median_baseline_test'] = np.median([tc_dict[hemi][task_name]['baseline_test'] for task_name in test_prf_stim.task_names], axis=0)
    
            for task_name in test_prf_stim.task_names:
                iso_diff_test = tc_dict[hemi]['median_baseline_test'] - tc_dict[hemi][task_name]['baseline_test']
                tc_dict[hemi][task_name]['timecourse_test'] += iso_diff_test[...,np.newaxis]
               
            tc_dict[hemi]['full_iso_test']=np.concatenate(tuple([tc_dict[hemi][task_name]['timecourse_test'] for task_name in test_prf_stim.task_names]), axis=-1)


    tc_full_iso = np.concatenate(tuple([tc_dict[hemi]['full_iso'] for hemi in hemis]), axis=0)
    iso_full = np.concatenate(tuple([tc_dict[hemi]['median_baseline'] for hemi in hemis]), axis=0)
    tc_mean = tc_full_iso.mean(-1)

    if crossvalidate:
        tc_full_iso_test = np.concatenate(tuple([tc_dict[hemi]['full_iso_test'] for hemi in hemis]), axis=0)
        iso_full_test = np.concatenate(tuple([tc_dict[hemi]['median_baseline_test'] for hemi in hemis]), axis=0)
        tc_mean_test = tc_full_iso_test.mean(-1)

    #masking flat or nearly flat timecourses
    if crossvalidate:
        nonlow_var = (np.abs(tc_full_iso - tc_mean[...,np.newaxis]).max(-1) > (tc_mean*min_percent_var/100)) \
             * (np.abs(tc_full_iso_test - tc_mean_test[...,np.newaxis]).max(-1) > (tc_mean_test*min_percent_var/100)) #\
             #* (tc_mean>0) * (tc_mean_test>0)
    else:
        nonlow_var = (np.abs(tc_full_iso - tc_mean[...,np.newaxis]).max(-1) > (tc_mean*min_percent_var/100)) #\
             #* (tc_mean>0) 
             
    if roi_idx is not None:
        mask = roi_mask(roi_idx, nonlow_var)
    else:
        mask = nonlow_var
        
    tc_dict_combined = dict()
    tc_dict_combined['mask'] = mask


    tc_full_iso_nonzerovar = tc_full_iso[mask]
    if crossvalidate:
        tc_full_iso_nonzerovar_test = tc_full_iso_test[mask]
            
    if fix_bold_baseline:
        tc_full_iso_nonzerovar -= iso_full[mask][...,np.newaxis]
        if crossvalidate:
            tc_full_iso_nonzerovar_test -= iso_full_test[mask][...,np.newaxis]
                       
            
    if save_raw_timecourse and raw_tcs == True:
        np.save(opj(data_path.replace('scratch-shared', 'home'),'prfpy',subj+"_timecourse-raw_space-"+fitting_space+".npy"),tc_full_iso[mask])
        np.save(opj(data_path.replace('scratch-shared', 'home'),'prfpy',subj+"_mask-raw_space-"+fitting_space+".npy"),mask)
        
        if crossvalidate:
            np.save(opj(data_path.replace('scratch-shared', 'home'),'prfpy',subj+"_timecourse-test-raw_space-"+fitting_space+".npy"),tc_full_iso_test[mask])
            
    if save_noise_ceiling:
        noise_ceiling = 1-np.sum((tc_full_iso_nonzerovar_test-tc_full_iso_nonzerovar)**2, axis=-1)/(tc_full_iso_nonzerovar_test.shape[-1]*tc_full_iso_nonzerovar_test.var(-1))
        np.save(opj(data_path,'prfpy',f"{subj}_noise-ceiling_space-{fitting_space}.npy"),noise_ceiling)
            
    order = np.random.permutation(tc_full_iso_nonzerovar.shape[0])

    tc_dict_combined['order'] = order

    tc_dict_combined['tc'] = tc_full_iso_nonzerovar[order]
    if crossvalidate:
        tc_dict_combined['tc_test'] = tc_full_iso_nonzerovar_test[order]
        
    return tc_dict_combined


            
                                
        
