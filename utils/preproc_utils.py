import os
import numpy as np

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
    #top, bottom, left, right
    design_matrix[:dm_edges_clipping[0],:,:] = 0
    design_matrix[-dm_edges_clipping[1]:,:,:] = 0
    design_matrix[:,:dm_edges_clipping[2],:] = 0
    design_matrix[:,-dm_edges_clipping[3]:,:] = 0
    print("Design matrix completed")
    
    return design_matrix

def create_full_stim(screenshot_paths,
                n_pix,
                discard_volumes,
                baseline_volumes_begin_end,
                dm_edges_clipping,
                screen_size_cm,
                screen_distance_cm,
                TR,
                task_names):
    dm_list = []

    for i, task_name in enumerate(task_names):
        # create stimulus
        if task_name in screenshot_paths[i]:
            dm_list.append(create_dm_from_screenshots(screenshot_paths[i],
                                                  n_pix,
                                                  dm_edges_clipping)[..., discard_volumes:])
    
    
    task_lengths = [dm.shape[-1] for dm in dm_list]    
    
    dm_full = np.concatenate(tuple(dm_list), axis=-1)

    # late-empty DM periods (for calculation of BOLD baseline)
    shifted_dm = np.zeros_like(dm_full)
    
    # number of TRs in which activity may linger (hrf)
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

    # late_iso_dict = {}
    # for i, task_name in enumerate(task_names):
    #     #to estimate baseline across conditions
    #     late_iso_dict[task_name] = np.concatenate((np.arange(baseline_volumes_begin_end[0]),np.arange(task_lengths[i]-baseline_volumes_begin_end[1], task_lengths[i])))

    prf_stim = PRFStimulus2D(screen_size_cm=screen_size_cm,
                             screen_distance_cm=screen_distance_cm,
                             design_matrix=dm_full,
                             TR=TR,
                             task_lengths=task_lengths,
                             task_names=task_names,
                             late_iso_dict=late_iso_dict)


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
                 save_noise_ceiling):

    if fitting_space == 'fsaverage' or fitting_space == 'fsnative':
        
        tc_dict = dd(lambda:dd(dict))
        for hemi in ['L', 'R']:
            for task_name in prf_stim.task_names:
                tc_task = []
                tc_paths = sorted(Path(opj(data_path,'fmriprep',subj)).glob(opj('**',subj+'_ses-*_task-'+task_name+'_run-*_space-'+fitting_space+'_hemi-'+hemi+'*.func.gii')))

                print("For task "+task_name+", hemisphere "+hemi+" of subject "+subj+", a total of "+str(len(tc_paths))+" runs were found.")
                
                if fit_runs is not None and (len(fit_runs)>len(tc_paths) or np.any(np.array(fit_runs)>=len(tc_paths))):
                    print(f"{fit_runs} fit_runs requested but only {len(tc_paths)} runs were found.")
                    raise ValueError

                if fit_runs is None:
                    #if CV over tasks, or if no CV, use all runs
                    fit_runs = np.arange(len(tc_paths))
                
                    
                for tc_path in [tc_paths[run] for run in fit_runs]:
                    tc_run = nb.load(str(tc_path))
                    #no need to pass further args, only filtering 1 condition
                    tc_task.append(filter_predictions(np.array([arr.data for arr in tc_run.darrays]).T[...,discard_volumes:],
                                                     filter_type=filter_type,
                                                     filter_params=filter_params))
    
                    #when scanning sub-001 i mistakenly set the length of the 4F scan to 147, while it should have been 145
                    #therefore, there are two extra images at the end to discard in that time series.
                    #from sub-002 onwards, this was corrected.
                    if subj == 'sub-001' and task_name=='4F':
                        tc_task[-1] = tc_task[-1][...,:-2]

                
                tc_dict[hemi][task_name]['timecourse'] = np.mean(tc_task, axis=0)
                tc_dict[hemi][task_name]['baseline'] = np.median(tc_dict[hemi][task_name]['timecourse'][...,prf_stim.late_iso_dict[task_name]],
                                                   axis=-1)
   
            if crossvalidate:
                for task_name in test_prf_stim.task_names:                          
                    tc_task = []
                    tc_paths = sorted(Path(opj(data_path,'fmriprep',subj)).glob(opj('**',subj+'_ses-*_task-'+task_name+'_run-*_space-'+fitting_space+'_hemi-'+hemi+'*.func.gii')))
    
                    print("For task "+task_name+", hemisphere "+hemi+" of subject "+subj+", a total of "+str(len(tc_paths))+" runs were found.")
    
                    if fit_task is not None:
                        #if CV is over tasks, can use all runs for test data as well
                        cv_runs = np.arange(len(tc_paths))
                    else:
                        cv_runs = [run for run in np.arange(len(tc_paths)) if run not in fit_runs]
    
                    for tc_path in [tc_paths[run] for run in cv_runs]:
                        tc_run = nb.load(str(tc_path))
                        #no need to pass further args, only filtering 1 condition
                        tc_task.append(filter_predictions(np.array([arr.data for arr in tc_run.darrays]).T[...,discard_volumes:],
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


        tc_full_iso = np.concatenate((tc_dict['L']['full_iso'], tc_dict['R']['full_iso']), axis=0)
        iso_full = np.concatenate((tc_dict['L']['median_baseline'], tc_dict['R']['median_baseline']), axis=0)
        tc_mean = tc_full_iso.mean(-1)

        if crossvalidate:
            tc_full_iso_test = np.concatenate((tc_dict['L']['full_iso_test'], tc_dict['R']['full_iso_test']), axis=0)
            iso_full_test = np.concatenate((tc_dict['L']['median_baseline_test'], tc_dict['R']['median_baseline_test']), axis=0)
            tc_mean_test = tc_full_iso_test.mean(-1)

        #masking flat or nearly flat timecourses
        if crossvalidate:
            nonlow_var = (np.abs(tc_full_iso - tc_mean[...,np.newaxis]).max(-1) > (tc_mean*min_percent_var/100)) \
                 * (np.abs(tc_full_iso_test - tc_mean_test[...,np.newaxis]).max(-1) > (tc_mean_test*min_percent_var/100)) \
                 * (tc_mean>0) * (tc_mean_test>0)
        else:
            nonlow_var = (np.abs(tc_full_iso - tc_mean[...,np.newaxis]).max(-1) > (tc_mean*min_percent_var/100)) \
                 * (tc_mean>0) 
                 
        if roi_idx is not None:
            mask = roi_mask(roi_idx, nonlow_var)
        else:
            mask = nonlow_var
            
        tc_dict_combined = dict()
        tc_dict_combined['mask'] = mask

        #conversion to +- of % of mean
        if data_scaling in ["psc", "percent_signal_change"]:
            tc_full_iso_nonzerovar = 100*(tc_full_iso[mask] / tc_mean[mask,np.newaxis])
            iso_full = 100*(iso_full/tc_mean)
            if crossvalidate:
                tc_full_iso_nonzerovar_test = 100*(tc_full_iso_test[mask] / tc_mean_test[mask,np.newaxis])
                iso_full_test = 100*(iso_full_test/tc_mean_test)
        elif data_scaling in ["zsc", "z-score"]:
            tc_full_iso_nonzerovar = zscore(tc_full_iso[mask], axis=-1)
            if crossvalidate:
                tc_full_iso_nonzerovar_test = zscore(tc_full_iso_test[mask], axis=-1)   
        elif data_scaling == None:
            tc_full_iso_nonzerovar = tc_full_iso[mask]
            if crossvalidate:
                tc_full_iso_nonzerovar_test = tc_full_iso_test[mask]
        else:
            print("Warning: data scaling option not recognized. Using raw data.")
            tc_full_iso_nonzerovar = tc_full_iso[mask]
            if crossvalidate:
                tc_full_iso_nonzerovar_test = tc_full_iso_test[mask]
                
        if fix_bold_baseline:
            tc_full_iso_nonzerovar += (tc_full_iso_nonzerovar.mean(-1)-iso_full[mask])[...,np.newaxis]
            if crossvalidate:
                tc_full_iso_nonzerovar_test += (tc_full_iso_nonzerovar_test.mean(-1)-iso_full_test[mask])[...,np.newaxis]
                           
                
        if save_raw_timecourse:
            np.save(opj(data_path,'prfpy',subj+"_timecourse-raw_space-"+fitting_space+".npy"),tc_full_iso[mask])
            if crossvalidate:
                np.save(opj(data_path,'prfpy',subj+"_timecourse-test-raw_space-"+fitting_space+".npy"),tc_full_iso_test[mask])
                
        if save_noise_ceiling:
            noise_ceiling = 1-np.sum((tc_full_iso_nonzerovar_test-tc_full_iso_nonzerovar)**2, axis=-1)/(tc_full_iso_nonzerovar_test.shape[-1]*tc_full_iso_nonzerovar_test.var(-1))
            np.save(opj(data_path,'prfpy',f"{subj}_noise-ceiling_space-{fitting_space}.npy"),noise_ceiling)
                
        order = np.random.permutation(tc_full_iso_nonzerovar.shape[0])

        tc_dict_combined['order'] = order

        tc_dict_combined['tc'] = tc_full_iso_nonzerovar[order]
        if crossvalidate:
            tc_dict_combined['tc_test'] = tc_full_iso_nonzerovar_test[order]
            
        return tc_dict_combined

    else:

        #############preparing the data (VOLUME FITTING) (NOT UP TO DATE)
        tc_dict=dd(dict)

        #create a single brain mask in BOLD space
        brain_masks = []
        for task_name in prf_stim.task_names:

            mask_paths = sorted(Path(opj(data_path,'fmriprep',subj)).glob(opj('**',subj+'_ses-*_task-'+task_name+'_run-*_space-'+fitting_space+'_desc-brain_mask.nii.gz')))
    
            for mask_path in mask_paths:
                brain_masks.append(nb.load(str(mask_path)).get_data().astype(bool))
    
        combined_brain_mask = np.ones_like(brain_masks[0]).astype(bool)
        for brain_mask in brain_masks:
            combined_brain_mask *= brain_mask
            

        for task_name in prf_stim.task_names:
            tc_task = []
            tc_paths = sorted(Path(opj(data_path,'fmriprep',subj)).glob(opj('**',subj+'_ses-*_task-'+task_name+'_run-*_space-'+fitting_space+'_desc-preproc_bold.nii.gz')))

            print("For task "+task_name+", of subject "+subj+", a total of "+str(len(tc_paths))+" runs were found.")

            if fit_runs is not None and fit_runs>=len(tc_paths):
                print(f"{fit_runs} fit_runs requested but only {len(tc_paths)} runs were found.")
                raise ValueError

            if not crossvalidate or fit_task is not None:
                #if CV over tasks, or if no CV, use all runs
                fit_runs = len(tc_paths)

            for tc_path in tc_paths[:fit_runs]:
                tc_run = nb.load(str(tc_path)).get_data()[...,discard_volumes:]
    
                tc_task.append(sgfilter_predictions(np.reshape(tc_run,(-1, tc_run.shape[-1])),
                                                     window_length=window_length,
                                                     polyorder=polyorder,
                                                     highpass=highpass,
                                                     add_mean=add_mean))
    
                #when scanning sub-001 i mistakenly set the length of the 4F scan to 147, while it should have been 145
                #therefore, there are two extra images at the end to discard in that time series.
                #from sub-002 onwards, this was corrected.
                if subj == 'sub-001' and task_name=='4F':
                    tc_task[-1] = tc_task[-1][...,:-2]
    
    
            tc_dict[task_name]['timecourse'] = np.median(tc_task, axis=0)
            tc_dict[task_name]['baseline'] = np.median(tc_dict[task_name]['timecourse'][...,prf_stim.late_iso_dict[task_name]],
                                                   axis=-1)

        if crossvalidate:
            for task_name in test_prf_stim.task_names:                
                tc_task = []
                tc_paths = sorted(Path(opj(data_path,'fmriprep',subj)).glob(opj('**',subj+'_ses-*_task-'+task_name+'_run-*_space-'+fitting_space+'_desc-preproc_bold.nii.gz')))
    
                print("For task "+task_name+", of subject "+subj+", a total of "+str(len(tc_paths))+" runs were found.")

                if fit_task is not None:
                    #if CV is over tasks, can use all runs for test data
                    fit_runs = 0
                    
                for tc_path in tc_paths[fit_runs:]:
                    tc_run = nb.load(str(tc_path)).get_data()[...,discard_volumes:]
        
                    tc_task.append(sgfilter_predictions(np.reshape(tc_run,(-1, tc_run.shape[-1])),
                                                         window_length=window_length,
                                                         polyorder=polyorder,
                                                         highpass=highpass,
                                                         add_mean=add_mean))
    
                    #when scanning sub-001 i mistakenly set the length of the 4F scan to 147, while it should have been 145
                    #therefore, there are two extra images at the end to discard in that time series.
                    #from sub-002 onwards, this was corrected.
                    if subj == 'sub-001' and task_name=='4F':
                        tc_task[-1] = tc_task[-1][...,:-2]                
                    
                    
                tc_dict[task_name]['timecourse_test'] = np.median(tc_task, axis=0)
                tc_dict[task_name]['baseline_test'] = np.median(tc_dict[task_name]['timecourse_test'][...,test_prf_stim.late_iso_dict[task_name]],
                                               axis=-1)

        #shift timeseries so they have the same average value in proper baseline periods across conditions
        iso_full = np.median([tc_dict[task_name]['baseline'] for task_name in prf_stim.task_names], axis=0)
    
        for task_name in prf_stim.task_names:
            iso_diff = iso_full - tc_dict[task_name]['baseline']
            tc_dict[task_name]['timecourse'] += iso_diff[...,np.newaxis]

        tc_full_iso=np.concatenate(tuple([tc_dict[task_name]['timecourse'] for task_name in prf_stim.task_names]), axis=-1)
        tc_mean = tc_full_iso.mean(-1)

        if crossvalidate:
            iso_full_test = np.median([tc_dict[task_name]['baseline_test'] for task_name in test_prf_stim.task_names], axis=0)
    
            for task_name in test_prf_stim.task_names:
                iso_diff_test = iso_full_test - tc_dict[task_name]['baseline_test']
                tc_dict[task_name]['timecourse_test'] += iso_diff_test[...,np.newaxis]
               
            tc_full_iso_test=np.concatenate(tuple([tc_dict[task_name]['timecourse_test'] for task_name in test_prf_stim.task_names]), axis=-1)
            tc_mean_test = tc_full_iso_test.mean(-1)

        #mask flat or nearly flat timecourses
        nonlow_var = np.reshape(combined_brain_mask, tc_full_iso.shape[0]) * \
            (np.abs(tc_full_iso - tc_mean[...,np.newaxis]).max(-1) > (tc_mean*min_percent_var/100)) \
                * (tc_mean>0) * (tc_mean_test>0)

        if roi_idx is not None:
            mask = roi_mask(roi_idx, nonlow_var)
        else:
            mask = nonlow_var
            
        tc_dict_combined = dict()
        
        tc_dict_combined['mask'] = np.reshape(mask, combined_brain_mask.shape)

        #conversion to +- of % of mean
        if data_scaling in ["psc", "percent_signal_change"]:
            tc_full_iso_nonzerovar = 100*(tc_full_iso[mask]/ iso_full[mask,np.newaxis])
            if crossvalidate:
                tc_full_iso_nonzerovar_test = 100*(tc_full_iso_test[mask] / iso_full_test[mask,np.newaxis])
        elif data_scaling == None:
            tc_full_iso_nonzerovar = tc_full_iso[mask]
            if crossvalidate:
                tc_full_iso_nonzerovar_test = tc_full_iso_test[mask]
        else:
            print("Warning: data scaling option not recognized. Using raw data.")
            tc_full_iso_nonzerovar = tc_full_iso[mask]
            if crossvalidate:
                tc_full_iso_nonzerovar_test = tc_full_iso_test[mask]

        if save_raw_timecourse:
            np.save(opj(data_path,'prfpy',subj+"_timecourse-raw_space-"+fitting_space+".npy"),tc_full_iso[mask])
            if crossvalidate:
                np.save(opj(data_path,'prfpy',subj+"_timecourse-test-raw_space-"+fitting_space+".npy"),tc_full_iso_test[mask])
                
        order = np.random.permutation(tc_full_iso_nonzerovar.shape[0])

        tc_dict_combined['order'] = order

        tc_dict_combined['tc'] = tc_full_iso_nonzerovar[order]
        if crossvalidate:
            tc_dict_combined['tc_test'] = tc_full_iso_nonzerovar_test[order]

        return tc_dict_combined



            
                                
        
