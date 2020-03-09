import os
import numpy as np
import scipy as sp
import matplotlib.pyplot as pl
from matplotlib import colors
import cortex
import nibabel as nb
from collections import defaultdict as dd
from pathlib import Path
import matplotlib.image as mpimg
import time
from scipy.stats import sem, ks_2samp

opj = os.path.join

from prfpy.timecourse import sgfilter_predictions
from prfpy.stimulus import PRFStimulus2D
from statsmodels.stats import weightstats
from sklearn.linear_model import LinearRegression
from nibabel.freesurfer.io import read_morph_data, write_morph_data


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
        late_iso_dict[task_name] = late_iso_dict['periods'][np.where((late_iso_dict['periods']>=start) & (late_iso_dict['periods']<stop))]
            
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
                 
                 window_length,
                 polyorder,
                 highpass,
                 add_mean,                

                 data_path,
                 fitting_space,
                 data_scaling,
                 roi_idx,
                 save_raw_timecourse,
                 
                 crossvalidate,
                 fit_runs,
                 fit_task):

    if fitting_space == 'fsaverage' or fitting_space == 'fsnative':
        
        tc_dict = dd(lambda:dd(dict))
        for hemi in ['L', 'R']:
            for task_name in prf_stim.task_names:
                tc_task = []
                tc_paths = sorted(Path(opj(data_path,'fmriprep',subj)).glob(opj('**',subj+'_ses-*_task-'+task_name+'_run-*_space-'+fitting_space+'_hemi-'+hemi+'.func.gii')))

                print("For task "+task_name+", hemisphere "+hemi+" of subject "+subj+", a total of "+str(len(tc_paths))+" runs were found.")
                
                if fit_runs is not None and fit_runs>=len(tc_paths):
                    print(f"{fit_runs} fit_runs requested but only {len(tc_paths)} runs were found.")
                    raise ValueError

                if not crossvalidate or fit_task is not None:
                    #if CV over tasks, or if no CV, use all runs
                    fit_runs = len(tc_paths)
                
                    
                for tc_path in tc_paths[:fit_runs]:
                    tc_run = nb.load(str(tc_path))
                    #no need to pass further args, only filtering 1 condition
                    tc_task.append(sgfilter_predictions(np.array([arr.data for arr in tc_run.darrays]).T[...,discard_volumes:],
                                                     window_length=window_length,
                                                     polyorder=polyorder,
                                                     highpass=highpass,
                                                     add_mean=add_mean))
    
                    #when scanning sub-001 i mistakenly set the length of the 4F scan to 147, while it should have been 145
                    #therefore, there are two extra images at the end to discard in that time series.
                    #from sub-002 onwards, this was corrected.
                    if subj == 'sub-001' and task_name=='4F':
                        tc_task[-1] = tc_task[-1][...,:-2]

                
                tc_dict[hemi][task_name]['timecourse'] = np.median(tc_task, axis=0)
                tc_dict[hemi][task_name]['baseline'] = np.median(tc_dict[hemi][task_name]['timecourse'][...,prf_stim.late_iso_dict[task_name]],
                                                   axis=-1)
   
            if crossvalidate:
                for task_name in test_prf_stim.task_names:                          
                    tc_task = []
                    tc_paths = sorted(Path(opj(data_path,'fmriprep',subj)).glob(opj('**',subj+'_ses-*_task-'+task_name+'_run-*_space-'+fitting_space+'_hemi-'+hemi+'.func.gii')))
    
                    print("For task "+task_name+", hemisphere "+hemi+" of subject "+subj+", a total of "+str(len(tc_paths))+" runs were found.")
    
                    if fit_task is not None:
                        #if CV is over tasks, can use all runs for test data
                        fit_runs = 0
    
                    for tc_path in tc_paths[fit_runs:]:
                        tc_run = nb.load(str(tc_path))
                        #no need to pass further args, only filtering 1 condition
                        tc_task.append(sgfilter_predictions(np.array([arr.data for arr in tc_run.darrays]).T[...,discard_volumes:],
                                                         window_length=window_length,
                                                         polyorder=polyorder,
                                                         highpass=highpass,
                                                         add_mean=add_mean))
        
                        #when scanning sub-001 i mistakenly set the length of the 4F scan to 147, while it should have been 145
                        #therefore, there are two extra images at the end to discard in that time series.
                        #from sub-002 onwards, this was corrected.
                        if subj == 'sub-001' and task_name=='4F':
                            tc_task[-1] = tc_task[-1][...,:-2]
    
                    
                    tc_dict[hemi][task_name]['timecourse_test'] = np.median(tc_task, axis=0)
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
        nonlow_var = (np.abs(tc_full_iso - tc_mean[...,np.newaxis]).max(-1) > (tc_mean*min_percent_var/100)) \
             * (np.abs(tc_full_iso_test - tc_mean_test[...,np.newaxis]).max(-1) > (tc_mean_test*min_percent_var/100)) \
             * (tc_mean>0) * (tc_mean_test>0)

        if roi_idx is not None:
            mask = roi_mask(roi_idx, nonlow_var)
        else:
            mask = nonlow_var
            
        tc_dict_combined = dict()
        tc_dict_combined['mask'] = mask

        #conversion to +- of % of mean
        if data_scaling in ["psc", "percent_signal_change"]:
            tc_full_iso_nonzerovar = 100*(tc_full_iso[mask] / iso_full[mask,np.newaxis])
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

    else:

        #############preparing the data (VOLUME FITTING)
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


def create_retinotopy_colormaps():
    hue, alpha = np.meshgrid(np.linspace(
        0, 1, 256, endpoint=False), 1-np.linspace(0, 1, 256))

    hsv = np.zeros(list(hue.shape)+[3])


    # convert angles to colors, using correlations as weights
    hsv[..., 0] = hue  # angs_discrete  # angs_n
    # np.sqrt(rsq) #np.ones_like(rsq)  # np.sqrt(rsq)
    hsv[..., 1] = np.ones_like(alpha)
    # np.nan_to_num(rsq ** -3) # np.ones_like(rsq)#n
    hsv[..., 2] = np.ones_like(alpha)
    hsv[-1,:,2] = 0

    rgb = colors.hsv_to_rgb(hsv)
    rgba = np.vstack((rgb.T, alpha[..., np.newaxis].T)).T
    pl.imshow(rgba)
    hsv_fn = os.path.join(os.path.split(cortex.database.default_filestore)[
                          0], 'colormaps', 'Retinotopy_HSV_alpha.png')
    sp.misc.imsave(hsv_fn, rgba)

    hue, alpha = np.meshgrid(
        np.fmod(np.linspace(0, 2, 256), 1.0), 1-np.linspace(0, 1, 256))
    hsv = np.zeros(list(hue.shape)+[3])
    # convert angles to colors, using correlations as weights
    hsv[..., 0] = hue  # angs_discrete  # angs_n
    # np.sqrt(rsq) #np.ones_like(rsq)  # np.sqrt(rsq)
    hsv[..., 1] = np.ones_like(alpha)
    # np.nan_to_num(rsq ** -3) # np.ones_like(rsq)#n
    hsv[..., 2] = np.ones_like(alpha)
    hsv[-1,:,2] = 0

    rgb = colors.hsv_to_rgb(hsv)
    rgba = np.vstack((rgb.T, alpha[..., np.newaxis].T)).T
    pl.imshow(rgba)
    hsv_fn = os.path.join(os.path.split(cortex.database.default_filestore)[
                          0], 'colormaps', 'Retinotopy_HSV_2x_alpha.png')
    sp.misc.imsave(hsv_fn, rgba)
    #####
    jet = mpimg.imread('/Users/marcoaqil/pycortex/filestore/colormaps/jet_r.png')
    jet = colors.rgb_to_hsv(jet[...,:3])

    hue, alpha = np.meshgrid(jet[...,0], 1-np.linspace(0, 1, 256))

    hsv = np.zeros(list(hue.shape)+[3])

    # convert angles to colors, using correlations as weights
    hsv[..., 0] = hue  # angs_discrete  # angs_n
    # np.sqrt(rsq) #np.ones_like(rsq)  # np.sqrt(rsq)
    hsv[..., 1] = np.ones_like(alpha)
    # np.nan_to_num(rsq ** -3) # np.ones_like(rsq)#n
    hsv[..., 2] = np.ones_like(alpha)#rdbu[...,2]
    hsv[-1,:,2] = 0

    rgb = colors.hsv_to_rgb(hsv)
    rgba = np.vstack((rgb.T, alpha[..., np.newaxis].T)).T
    pl.imshow(rgba)
    hsv_fn = os.path.join(os.path.split(cortex.database.default_filestore)[
                          0], 'colormaps', 'Jet_r_2D_alpha.png')
    sp.misc.imsave(hsv_fn, rgba)


def roi_mask(roi, array):
    array_2 = np.zeros_like(array)
    array_2[roi] = 1
    masked_array = array * array_2
    return masked_array



def fwhmax_fwatmin(model, params, normalize_RFs=False, return_profiles=False):
    model = model.lower()
    x=np.linspace(-50,50,1000).astype('float32')

    prf = params[...,3] * np.exp(-0.5*x[...,np.newaxis]**2 / params[...,2]**2)
    vol_prf =  2*np.pi*params[...,2]**2

    if model in ['dog', 'norm']:
        srf = params[...,5] * np.exp(-0.5*x[...,np.newaxis]**2 / params[...,6]**2)
        vol_srf = 2*np.pi*params[...,6]**2

    if normalize_RFs==True:

        if model == 'gauss':
            profile =  prf / vol_prf
        elif model == 'css':
            #amplitude is outside exponent in CSS
            profile = (prf / vol_prf)**params[...,5] * params[...,3]**(1 - params[...,5])
        elif model =='dog':
            profile = prf / vol_prf - \
                       srf / vol_srf
        elif model == 'norm':
            profile = (prf / vol_prf + params[...,7]) /\
                      (srf / vol_srf + params[...,8]) - params[...,7]/params[...,8]
    else:
        if model == 'gauss':
            profile = prf
        elif model == 'css':
            #amplitude is outside exponent in CSS
            profile = prf**params[...,5] * params[...,3]**(1 - params[...,5])
        elif model =='dog':
            profile = prf - srf
        elif model == 'norm':
            profile = (prf + params[...,7])/(srf + params[...,8]) - params[...,7]/params[...,8]


    half_max = np.max(profile, axis=0)/2
    fwhmax = np.abs(2*x[np.argmin(np.abs(half_max-profile), axis=0)])


    if model in ['dog', 'norm']:

        min_profile = np.min(profile, axis=0)
        fwatmin = np.abs(2*x[np.argmin(np.abs(min_profile-profile), axis=0)])

        result = fwhmax, fwatmin
    else:
        result = fwhmax

    if return_profiles:
        if model == 'norm':
            #accounting for the previous subtraction of baseline
            profile += params[...,7]/params[...,8]

        return result, profile.T
    else:
        return result


def combine_results(subj, fitting_space, results_folder, suffix_list, 
                    raw_timecourse_path=None, normalize_RFs=False, ref_img_path=None):
    mask = np.load(opj(results_folder, subj+'_mask_space-'+fitting_space+'.npy'))
    for i, suffix in enumerate(suffix_list):
        if i == 0:
            try:
                gauss = np.load(opj(results_folder,subj+'_iterparams-gauss_space-'+fitting_space+suffix+'.npy'))
            except Exception as e:
                gauss = 0
                print(e)
                pass
            try:
                css = np.load(opj(results_folder,subj+'_iterparams-css_space-'+fitting_space+suffix+'.npy'))
            except Exception as e:
                css = 0
                print(e)
                pass
            try:
                dog = np.load(opj(results_folder,subj+'_iterparams-dog_space-'+fitting_space+suffix+'.npy'))
            except Exception as e:
                dog = 0
                print(e)
                pass
            try:
                norm = np.load(opj(results_folder,subj+'_iterparams-norm_space-'+fitting_space+suffix+'.npy'))
            except Exception as e:
                norm = 0
                print(e)
                pass
            try:
                gauss_grid = np.load(opj(results_folder,subj+'_gridparams-gauss_space-'+fitting_space+suffix+'.npy'))
            except Exception as e:
                gauss_grid = 0
                print(e)
                pass
            try:
                norm_grid = np.load(opj(results_folder,subj+'_gridparams-norm_space-'+fitting_space+suffix+'.npy'))
            except Exception as e:
                norm_grid = 0
                print(e)
                pass
        else:
            try:
                gauss_it = np.load(opj(results_folder,subj+'_iterparams-gauss_space-'+fitting_space+suffix+'.npy'))
            except Exception as e:
                print(e)
                pass
            try:
                css_it = np.load(opj(results_folder,subj+'_iterparams-css_space-'+fitting_space+suffix+'.npy'))
            except Exception as e:
                print(e)
                pass
            try:
                dog_it = np.load(opj(results_folder,subj+'_iterparams-dog_space-'+fitting_space+suffix+'.npy'))
            except Exception as e:
                print(e)
                pass
            try:
                norm_it = np.load(opj(results_folder,subj+'_iterparams-norm_space-'+fitting_space+suffix+'.npy'))
            except Exception as e:
                print(e)
                pass
            try:
                gauss[(gauss[:,-1]<gauss_it[:,-1])] = np.copy(gauss_it[(gauss[:,-1]<gauss_it[:,-1])])
            except Exception as e:
                print(e)
                pass
            try:
                css[(css[:,-1]<css_it[:,-1])] = np.copy(css_it[(css[:,-1]<css_it[:,-1])])
            except Exception as e:
                print(e)
                pass
            try:
                dog[(dog[:,-1]<dog_it[:,-1])] = np.copy(dog_it[(dog[:,-1]<dog_it[:,-1])])
            except Exception as e:
                print(e)
                pass
            try:
                norm[(norm[:,-1]<norm_it[:,-1])] = np.copy(norm_it[(norm[:,-1]<norm_it[:,-1])])
            except Exception as e:
                print(e)
                pass
    raw_tc_stats = dict()
    if raw_timecourse_path is not None:
        
        tc_raw = np.load(raw_timecourse_path)
        
        tc_mean = tc_raw.mean(-1)
        tc_mean_full = np.zeros(mask.shape)
        tc_mean_full[mask] = tc_mean
        raw_tc_stats['Mean'] = tc_mean_full
        
        tc_var = tc_raw.var(-1)
        tc_var_full = np.zeros(mask.shape)
        tc_var_full[mask] = tc_var
        raw_tc_stats['Variance'] = tc_var_full
        
        tc_tsnr_full = np.zeros(mask.shape)
        tc_tsnr_full[mask] = tc_mean/np.sqrt(tc_var)
        raw_tc_stats['TSNR'] = tc_tsnr_full
    
    return {'Gauss grid':gauss_grid, 'Norm grid':norm_grid, 'Gauss':gauss,
            'CSS':css, 'DoG':dog, 'Norm':norm,
            'mask':mask, 'normalize_RFs':normalize_RFs, 
            'Timecourse Stats':raw_tc_stats, 'ref_img_path':ref_img_path}


def process_results(results_dict, return_norm_profiles):
    for k, v in results_dict.items():
        if 'sub-' not in k:
            process_results(v, return_norm_profiles)
        elif 'Results' in v and 'Processed Results' not in v:
            normalize_RFs = v['Results']['normalize_RFs']
            mask = v['Results']['mask']

            #store processed results in nested default dictionary
            processed_results = dd(lambda:dd(lambda:np.zeros(mask.shape)))

            #loop over contents of single-subject analysis results (models and mask)
            for k2, v2 in v['Results'].items():
                if k2 != 'mask' and isinstance(v2, np.ndarray) and 'grid' not in k2 and 'Stats' not in k2:

                    processed_results['RSq'][k2][mask] = v2[:,-1]
                    processed_results['Eccentricity'][k2][mask] = np.sqrt(v2[:,0]**2+v2[:,1]**2)
                    processed_results['Polar Angle'][k2][mask] = np.arctan2(v2[:,1], v2[:,0])
                    processed_results['Amplitude'][k2][mask] = v2[:,3]

                    if k2 == 'CSS':
                        processed_results['CSS Exponent'][k2][mask] = v2[:,5]

                    if k2 == 'DoG':
                        (processed_results['Size (fwhmax)'][k2][mask],
                        processed_results['Surround Size (fwatmin)'][k2][mask]) = fwhmax_fwatmin(k2, v2, normalize_RFs)

                    elif k2 == 'Norm':
                        processed_results['Norm Param. B'][k2][mask] = v2[:,7]
                        processed_results['Norm Param. D'][k2][mask] = v2[:,8]
                        processed_results['Ratio (B/D)'][k2][mask] = v2[:,7]/v2[:,8]

                        if return_norm_profiles and len(mask.shape)<2:
                            processed_results['pRF Profiles'][k2] = np.zeros((mask.shape[0],1000))
                            ((processed_results['Size (fwhmax)'][k2][mask],
                            processed_results['Surround Size (fwatmin)'][k2][mask]),
                            processed_results['pRF Profiles'][k2][mask]) = fwhmax_fwatmin(k2, v2, normalize_RFs, True)
                        else:
                            (processed_results['Size (fwhmax)'][k2][mask],
                            processed_results['Surround Size (fwatmin)'][k2][mask]) = fwhmax_fwatmin(k2, v2, normalize_RFs, False)

                    else:
                        processed_results['Size (fwhmax)'][k2][mask] = fwhmax_fwatmin(k2, v2, normalize_RFs)
                elif 'Stats' in k2:
                    v[k2] = v['Results'][k2]
                    

            v['Processed Results'] = {ke : dict(va) for ke, va in processed_results.items()}


def get_subjects(main_dict,subject_list = []):
    for k, v in main_dict.items():
        if 'sub-' not in k:# and isinstance(v, (dict,dd)):
            get_subjects(v, subject_list)
        else:
            if k not in subject_list:
                subject_list.append(k)
    return subject_list


class visualize_results(object):
    def __init__(self):
        self.main_dict = dd(lambda:dd(lambda:dd(dict)))
        
    def transfer_parse_labels(self, fs_dir):
        self.idx_rois = dd(dict)
        for subj in self.subjects:
            if self.transfer_rois:
                src_subject='fsaverage'
            else:
                src_subject=subj
                    
            self.fs_dir = fs_dir
        
            wang_rois = ["V1v", "V1d", "V2v", "V2d", "V3v", "V3d", "hV4", "VO1", "VO2", "PHC1", "PHC2",
                "TO2", "TO1", "LO2", "LO1", "V3B", "V3A", "IPS0", "IPS1", "IPS2", "IPS3", "IPS4", 
                "IPS5", "SPL1", "FEF"]
            for roi in wang_rois:
                try:
                    self.idx_rois[subj][roi], _ = cortex.freesurfer.get_label(subject=subj,
                                                          label='wang2015atlas.'+roi,
                                                          fs_dir=self.fs_dir,
                                                          src_subject=src_subject)
                except Exception as e:
                    print(e)
        
            self.idx_rois[subj]['visual_system'] = np.concatenate(tuple([self.idx_rois[subj][roi] for roi in self.idx_rois[subj]]), axis=0)
            self.idx_rois[subj]['V1']=np.concatenate((self.idx_rois[subj]['V1v'],self.idx_rois[subj]['V1d']))
            self.idx_rois[subj]['V2']=np.concatenate((self.idx_rois[subj]['V2v'],self.idx_rois[subj]['V2d']))
            self.idx_rois[subj]['V3']=np.concatenate((self.idx_rois[subj]['V3v'],self.idx_rois[subj]['V3d']))
        
            #parse custom ROIs if they have been created
            for roi in ['custom.V1','custom.V2','custom.V3']:
                try:
                    self.idx_rois[subj][roi], _ = cortex.freesurfer.get_label(subject=subj,
                                                          label=roi,
                                                          fs_dir=self.fs_dir,
                                                          src_subject=subj)
                except Exception as e:
                    print(e)
                    pass
                
            #For ROI-based fitting
            if self.output_custom_V1V2V3:
                V1V2V3 = np.concatenate((self.idx_rois[subj]['custom.V1'],self.idx_rois[subj]['custom.V2'],self.idx_rois[subj]['custom.V3']))
                np.save('/Users/marcoaqil/PRFMapping/PRFMapping-Deriv-hires/prfpy/'+subj+'_roi-V1V2V3.npy', V1V2V3)
        
    def pycortex_plots(self):        
          
        for space, space_res in self.main_dict.items():
            if 'fs' in space:
                plotted_rois = dd(lambda:False)
                plotted_stats = dd(lambda:False)
                for analysis, analysis_res in space_res.items():       
                    for subj, subj_res in analysis_res.items():
                        print(space+" "+analysis+" "+subj)
                        
                        if subj not in cortex.db.subjects:
                            cortex.freesurfer.import_subj(subj, freesurfer_subject_dir=self.fs_dir, 
                                  whitematter_surf='smoothwm')
                        
                        p_r = subj_res['Processed Results']
                        models = p_r['RSq'].keys()
                        
                        tc_stats = subj_res['Timecourse Stats']
                        mask = subj_res['Results']['mask']
                        
                        
                        #######Raw bold timecourse vein threshold
                        if subj == 'sub-006':
                            tc_min = 45000
                        elif subj == 'sub-007':
                            tc_min = 35000
                        elif subj == 'sub-001':
                            tc_min = 35000
                            
                        ######limits for eccentricity
                        ecc_min=0.3
                        ecc_max=3.0
                        ######max prf size
                        w_max = 90
                        
                        #housekeeping
                        rsq = np.vstack(tuple([elem for _,elem in p_r['RSq'].items()])).T
                        ecc = np.vstack(tuple([elem for _,elem in p_r['Eccentricity'].items()])).T
                        polar = np.vstack(tuple([elem for _,elem in p_r['Polar Angle'].items()])).T
                        amp = np.vstack(tuple([elem for _,elem in p_r['Amplitude'].items()])).T
                        fw_hmax = np.vstack(tuple([elem for _,elem in p_r['Size (fwhmax)'].items()])).T
            
                        #alpha dictionary
                        p_r['Alpha'] = {}          
                        p_r['Alpha']['all'] = rsq.max(-1) * (tc_stats['Mean']>tc_min) * (ecc.min(-1)<ecc_max) * (ecc.max(-1)>ecc_min)
                        
                        for model in models:
                            p_r['Alpha'][model] = p_r['RSq'][model] * (p_r['Eccentricity'][model]>ecc_min) * (p_r['Eccentricity'][model]<ecc_max) *(p_r['Amplitude'][model]>0) * (tc_stats['Mean']>tc_min) * (p_r['Size (fwhmax)'][model]<w_max)
                       
                        if self.only_roi is not None:
                            for key in p_r['Alpha']:
                                p_r['Alpha'][key] = roi_mask(self.idx_rois[subj][self.only_roi], p_r['Alpha'][key])
                                         
                        ##START PYCORTEX VISUALIZATIONS
                        #data quality/stats cortex visualization 
                        if space == 'fsnative' and self.plot_stats_cortex and not plotted_stats[subj] :
                            mean_ts_vert = cortex.Vertex2D(tc_stats['Mean'], mask*(tc_stats['Mean']>tc_min), subject=subj, cmap='Jet_2D_alpha')
                            var_ts_vert = cortex.Vertex2D(tc_stats['Variance'], mask*(tc_stats['Mean']>tc_min), subject=subj, cmap='Jet_2D_alpha')
                            tsnr_vert = cortex.Vertex2D(tc_stats['TSNR'], mask*(tc_stats['Mean']>tc_min), subject=subj, cmap='Jet_2D_alpha')
            
                            data_stats ={'mean':mean_ts_vert.raw, 'var':var_ts_vert.raw, 'tsnr':tsnr_vert.raw}
            
                            self.js_handle_stats = cortex.webgl.show(data_stats, with_curvature=False, with_labels=True, with_rois=True, with_borders=True, with_colorbar=True)
            
                            plotted_stats[subj] = True
                        
                        if self.plot_rois_cortex and not plotted_rois[subj]:
                            
                            ds_rois = {}
                            data = np.zeros_like(mask).astype('int')
            
                            for i, roi in enumerate(self.idx_rois[subj]):
            
                                roi_data = np.zeros_like(mask)
                                roi_data[self.idx_rois[subj][roi]] = 1
                                if 'custom' not in roi and 'visual' not in roi:
                                    data[self.idx_rois[subj][roi]] = i+1

                                ds_rois[roi] = cortex.Vertex2D(roi_data, roi_data.astype('bool'), subj, cmap='RdBu_r_alpha').raw
            
                                #need a correctly flattened brain to do this
                                #cortex.add_roi(ds_rois[roi], name=roi, open_inkscape=False, add_path=True)
            
                            ds_rois['Wang2015Atlas'] = cortex.Vertex2D(data, data.astype('bool'), subj, cmap='Retinotopy_HSV_2x_alpha').raw
                            self.js_handle_rois = cortex.webgl.show(ds_rois, with_curvature=False, with_labels=True, with_rois=True, with_borders=True, with_colorbar=True)
            
                            plotted_rois[subj] = True
                                                    
                        #output freesurefer-format polar angle maps to draw custom ROIs in freeview    
                        if self.output_freesurfer_polar_maps:
                                          
                            lh_c = read_morph_data(opj(self.fs_dir, subj+'/surf/lh.curv'))
            
                            polar_freeview = np.median(polar, axis=-1)
            
                            alpha_freeview = rsq.max(-1) * (amp.min(-1)>0) * (tc_stats['Mean']>tc_min) #* (ecc.max(-1)<ecc_max) * (ecc.min(-1)>ecc_min)
            
                            polar_freeview[alpha_freeview<0.2] = -10
            
                            write_morph_data(opj(self.fs_dir, subj+'/surf/lh.polar')
                                                                   ,polar_freeview[:lh_c.shape[0]])
                            write_morph_data(opj(self.fs_dir, subj+'/surf/rh.polar')
                                                                   ,polar_freeview[lh_c.shape[0]:])
                            
            
                        if self.plot_rsq_cortex:              
                            ds_rsq = {}
                            if 'CSS' in models and 'Gauss' in models:
                                ds_rsq['CSS - Gauss'] = cortex.Vertex2D(p_r['RSq']['CSS']-p_r['RSq']['Gauss'], p_r['Alpha']['all'], subject=subj,
                                                                          vmin=0, vmax=0.1, vmin2=0.3, vmax2=0.6, cmap='Jet_2D_alpha').raw                
                            if 'DoG' in models and 'Gauss' in models:
                                ds_rsq['DoG - Gauss'] = cortex.Vertex2D(p_r['RSq']['DoG']-p_r['RSq']['Gauss'], p_r['Alpha']['all'], subject=subj,
                                                                      vmin=-0.02, vmax=0.02, vmin2=0.3, vmax2=0.6, cmap='Jet_2D_alpha').raw
                            if 'Norm' in models and 'Gauss' in models:
                                ds_rsq['Norm - Gauss'] = cortex.Vertex2D(p_r['RSq']['Norm']-p_r['RSq']['Gauss'], p_r['Alpha']['all'], subject=subj,
                                                                      vmin=0, vmax=0.1, vmin2=0.3, vmax2=0.6, cmap='Jet_2D_alpha').raw
                            if 'Norm' in models and 'DoG' in models:
                                ds_rsq['Norm - DoG'] = cortex.Vertex2D(p_r['RSq']['Norm']-p_r['RSq']['DoG'], p_r['Alpha']['all'], subject=subj,
                                                                      vmin=0, vmax=0.1, vmin2=0.3, vmax2=0.6, cmap='Jet_2D_alpha').raw
                            if 'Norm' in models and 'CSS' in models:
                                ds_rsq['Norm - CSS'] = cortex.Vertex2D(p_r['RSq']['Norm']-p_r['RSq']['CSS'], p_r['Alpha']['all'], subject=subj, 
                                                                      vmin=-0.02, vmax=0.02, vmin2=0.3, vmax2=0.6, cmap='Jet_2D_alpha').raw
                                
                            if 'Processed Results' in self.main_dict['T1w'][analysis][subj] and self.compare_volume_surface:
                                ds_rsq_comp={}
                                volume_rsq = self.main_dict['T1w'][analysis][subj]['Processed Results']['RSq']['Norm']
                                ref_img = nb.load(self.main_dict['T1w'][analysis][subj]['Results']['ref_img_path'])
                                
                                #rsq_img = nb.Nifti1Image(volume_rsq, ref_img.affine, ref_img.header)

                                xfm_trans = cortex.xfm.Transform(np.identity(4), ref_img)
                                xfm_trans.save(subj, 'func_space_transform')
                                
                                ds_rsq_comp['Norm CV rsq (volume fit)'] = cortex.Volume2D(volume_rsq.T, volume_rsq.T, subj, 'func_space_transform',
                                                                          vmin=0.2, vmax=0.6, vmin2=0.05, vmax2=0.2, cmap='Jet_2D_alpha')
                                ds_rsq_comp['Norm CV rsq (surface fit)'] = cortex.Vertex2D(p_r['RSq']['Norm'], p_r['RSq']['Norm'], subject=subj,
                                                                          vmin=0.2, vmax=0.6, vmin2=0.05, vmax2=0.2, cmap='Jet_2D_alpha').raw
                                self.js_handle_rsq_comp = cortex.webgl.show(ds_rsq_comp, with_curvature=False, with_labels=True, with_rois=True, with_borders=True, with_colorbar=True)

                            self.js_handle_rsq = cortex.webgl.show(ds_rsq, with_curvature=False, with_labels=True, with_rois=True, with_borders=True, with_colorbar=True) 
                            
                        if self.plot_ecc_cortex:
                            ds_ecc = {}
                            for model in models:
                                ds_ecc[model] = cortex.Vertex2D(p_r['Eccentricity'][model], p_r['Alpha'][model], subject=subj, 
                                                                vmin=ecc_min, vmax=ecc_max, vmin2=0.2, vmax2=0.6, cmap='Jet_r_2D_alpha').raw
            
                            self.js_handle_ecc = cortex.webgl.show(ds_ecc, with_curvature=False, with_labels=True, with_rois=True, with_borders=True, with_colorbar=True)
            
                        if self.plot_polar_cortex:
                            ds_polar = {}
                            for model in models:
                                ds_polar[model] = cortex.Vertex2D(p_r['Polar Angle'][model], p_r['Alpha'][model], subject=subj, 
                                                                  vmin2=0.2, vmax2=0.6, cmap='Retinotopy_HSV_2x_alpha').raw
                            
                            if 'Processed Results' in self.main_dict['T1w'][analysis][subj] and self.compare_volume_surface:
                                ds_polar_comp={}
                                volume_rsq = self.main_dict['T1w'][analysis][subj]['Processed Results']['RSq']['Norm']
                                volume_polar = self.main_dict['T1w'][analysis][subj]['Processed Results']['Polar Angle']['Norm']
                                ref_img = nb.load(self.main_dict['T1w'][analysis][subj]['Results']['ref_img_path'])                                

                                xfm_trans = cortex.xfm.Transform(np.identity(4), ref_img)
                                xfm_trans.save(subj, 'func_space_transform')
                                
                                ds_polar_comp['Norm CV polar (volume fit)'] = cortex.Volume2D(volume_polar.T, volume_rsq.T, subj, 'func_space_transform',
                                                                          vmin2=0.05, vmax2=0.2, cmap='Retinotopy_HSV_2x_alpha')
                                ds_polar_comp['Norm CV polar (surface fit)'] = cortex.Vertex2D(p_r['Polar Angle']['Norm'], p_r['RSq']['Norm'], subject=subj,
                                                                          vmin2=0.05, vmax2=0.2, cmap='Retinotopy_HSV_2x_alpha').raw
                                self.js_handle_polar_comp = cortex.webgl.show(ds_polar_comp, with_curvature=False, with_labels=True, with_rois=True, with_borders=True, with_colorbar=True)

                            
                            self.js_handle_polar = cortex.webgl.show(ds_polar, with_curvature=False, with_labels=True, with_rois=True, with_borders=True, with_colorbar=True)
            
                        if self.plot_size_cortex:
                            ds_size = {}
                            for model in models:
                                ds_size[model] = cortex.Vertex2D(p_r['Size (fwhmax)'][model], p_r['Alpha'][model], subject=subj, 
                                                                 vmin=0, vmax=6, vmin2=0.2, vmax2=0.6, cmap='Jet_2D_alpha').raw
                  
                            self.js_handle_size = cortex.webgl.show(ds_size, with_curvature=False, with_labels=True, with_rois=True, with_borders=True, with_colorbar=True)
            
            
                        if self.plot_amp_cortex:
                            ds_amp = {}
                            for model in models:
                                ds_amp[model] = cortex.Vertex2D(p_r['Amplitude'][model], p_r['Alpha'][model], subject=subj, 
                                                                vmin=-1, vmax=1, vmin2=0.2, vmax2=0.6, cmap='Jet_2D_alpha').raw
            
                            self.js_handle_amp = cortex.webgl.show(ds_amp, with_curvature=False, with_labels=True, with_rois=True, with_borders=True, with_colorbar=True)
                            
                        if self.plot_css_exp_cortex and 'CSS' in models:
                            ds_css_exp = {}
                            ds_css_exp['CSS Exponent'] = cortex.Vertex2D(p_r['CSS Exponent']['CSS'], p_r['Alpha']['CSS'], subject=subj, 
                                                                         vmin=0, vmax=0.75, vmin2=0.2, vmax2=0.6, cmap='Jet_2D_alpha').raw
            
                            self.js_handle_css_exp = cortex.webgl.show(ds_css_exp, with_curvature=False, with_labels=True, with_rois=True, with_borders=True, with_colorbar=True)
                            
                        if self.plot_surround_size_cortex:
                            ds_surround_size = {}
                            if 'DoG' in models:
                                ds_surround_size['DoG'] = cortex.Vertex2D(p_r['Surround Size (fwatmin)']['DoG'], p_r['Alpha']['DoG'], subject=subj, 
                                                                         vmin=0, vmax=50, vmin2=0.2, vmax2=0.6, cmap='Jet_2D_alpha').raw
                            if 'Norm' in models:
                                ds_surround_size['Norm'] = cortex.Vertex2D(p_r['Surround Size (fwatmin)']['Norm'], p_r['Alpha']['Norm'], subject=subj, 
                                                                         vmin=0, vmax=50, vmin2=0.2, vmax2=0.6, cmap='Jet_2D_alpha').raw                    
            
                            self.js_handle_surround_size = cortex.webgl.show(ds_surround_size, with_curvature=False, with_labels=True, with_rois=True, with_borders=True, with_colorbar=True)    
                            
                        if self.plot_norm_baselines_cortex and 'Norm' in models:
                            ds_norm_baselines = {}
                            ds_norm_baselines['Norm Param. B'] = cortex.Vertex2D(p_r['Norm Param. B']['Norm'], p_r['Alpha']['Norm'], subject=subj, 
                                                                         vmin=0, vmax=50, vmin2=0.2, vmax2=0.6, cmap='Jet_2D_alpha').raw                    
                            ds_norm_baselines['Norm Param. D'] = cortex.Vertex2D(p_r['Norm Param. D']['Norm'], p_r['Alpha']['Norm'], subject=subj, 
                                                                         vmin=0, vmax=50, vmin2=0.2, vmax2=0.6, cmap='Jet_2D_alpha').raw
                            ds_norm_baselines['Ratio (B/D)'] = cortex.Vertex2D(p_r['Ratio (B/D)']['Norm'], p_r['Alpha']['Norm'], subject=subj, 
                                                                         vmin=0, vmax=50, vmin2=0.2, vmax2=0.6, cmap='Jet_2D_alpha').raw
                            
                            self.js_handle_norm_baselines = cortex.webgl.show(ds_norm_baselines, with_curvature=False, with_labels=True, with_rois=True, with_borders=True, with_colorbar=True)    
                              
    
        
    def save_pycortex_views(self, js_handle, base_str):
        views = dict(dorsal=dict(radius=191, altitude=73, azimuth=178, pivot=0),
                     medial=dict(radius=10, altitude=101, azimuth=359, pivot=167),
                     lateral=dict(radius=277, altitude=90, azimuth=177, pivot=123),
                     ventral=dict(radius=221, altitude=131, azimuth=175, pivot=0)
                    )
        
        surfaces = dict(inflated=dict(unfold=1))#,
                       # fiducial=dict(unfold=0.0))
        
        # select path for the generated images on disk
        image_path = '/Users/marcoaqil/PRFMapping/Figures/'
        
        # pattern of the saved images names
        file_pattern = "{base}_{view}_{surface}.png"
        
        # utility functions to set the different views
        prefix = dict(altitude='camera.', azimuth='camera.',
                      pivot='surface.{subject}.', radius='camera.', target='camera.',
                      unfold='surface.{subject}.')
        _tolists = lambda p: {prefix[k]+k:[v] for k,v in p.items()}
        _combine = lambda a,b: ( lambda c: [c, c.update(b)][0] )(dict(a))
        
        
        # Save images by iterating over the different views and surfaces
        for view,vparams in views.items():
            for surf,sparams in surfaces.items():
                # Combine basic, view, and surface parameters
                params = _combine(vparams, sparams)
                # Set the view
                print(params)
                time.sleep(5)
                js_handle._set_view(**_tolists(params))
                time.sleep(5)
                # Save image
                filename = file_pattern.format(base=base_str, view=view, surface=surf)
                output_path = os.path.join(image_path, filename)
                js_handle.getImage(output_path, size =(3000, 2000))
        
                # the block below trims the edges of the image:
                # wait for image to be written
                while not os.path.exists(output_path):
                    pass
                time.sleep(0.5)
                try:
                    import subprocess
                    subprocess.call(["convert", "-trim", output_path, output_path])
                except:
                    pass
        
    def ecc_size_roi_plots(self, rois, rsq_thresh, save_figures):
        
        pl.rcParams.update({'font.size': 16})
        for space, space_res in self.main_dict.items():
            if 'fs' in space:
                for analysis, analysis_res in space_res.items():       
                    for subj, subj_res in analysis_res.items():
                        print(space+" "+analysis+" "+subj)
            
                        # binned eccentricity vs other parameters relationships       
            
                        model_colors = {'Gauss':'blue','CSS':'orange','DoG':'green','Norm':'red'}
                                                
                        #model_symbols = {'Gauss':'^','CSS':'o','DoG':'v','Norm':'D'}
                        roi_colors = dd(lambda:'blue')
                        roi_colors['custom.V1']= 'black'
                        roi_colors['custom.V2']= 'red'
                        roi_colors['custom.V3']= 'pink'
            
                        fw_hmax_stats = dd(lambda:dd(list))
                        ecc_stats = dd(lambda:dd(list))
            
                        for roi in rois:
            
                            pl.figure(roi+' fw_hmax', figsize=(8, 6), frameon=False)
           
                            for model in subj_res['Processed Results']['Size (fwhmax)'].keys():                                
            
                                #model-specific alpha? or all models same alpha?
                                alpha_roi = roi_mask(self.idx_rois[subj][roi], subj_res['Processed Results']['Alpha'][model])>rsq_thresh
                                
                                ecc_model_roi = subj_res['Processed Results']['Eccentricity'][model][alpha_roi]
                                fwhmax_model_roi = subj_res['Processed Results']['Size (fwhmax)'][model][alpha_roi]
                                rsq_model_roi = subj_res['Processed Results']['RSq'][model][alpha_roi]
                                
                                ecc_sorted = np.argsort(ecc_model_roi)
                                split_ecc_bins = np.array_split(ecc_sorted, 10)
                               
                                for ecc_quantile in split_ecc_bins:
                                    fw_hmax_stats[roi][model].append(weightstats.DescrStatsW(fwhmax_model_roi[ecc_quantile],
                                                                                          weights=rsq_model_roi[ecc_quantile]))
            
                                    ecc_stats[roi][model].append(weightstats.DescrStatsW(ecc_model_roi[ecc_quantile],
                                                                                          weights=rsq_model_roi[ecc_quantile]))
            
                       
                                WLS = LinearRegression()
                                WLS.fit(ecc_model_roi.reshape(-1, 1), fwhmax_model_roi, sample_weight=rsq_model_roi)
                                pl.plot([ss.mean for ss in ecc_stats[roi][model]],
                                        WLS.predict(np.array([ss.mean for ss in ecc_stats[roi][model]]).reshape(-1, 1)),
                                        color=model_colors[model])
                                            
                                print(roi+" "+model+" "+str(WLS.score(ecc_model_roi.reshape(-1, 1), fwhmax_model_roi, sample_weight=rsq_model_roi)))
            
                                pl.errorbar([ss.mean for ss in ecc_stats[roi][model]],
                                   [ss.mean for ss in fw_hmax_stats[roi][model]],
                                   yerr=np.array([np.abs(ss.zconfint_mean(alpha=0.05)-ss.mean) for ss in fw_hmax_stats[roi][model]]).T,
                                   xerr=np.array([np.abs(ss.zconfint_mean(alpha=0.05)-ss.mean) for ss in ecc_stats[roi][model]]).T,
                                   fmt='s', mfc=model_colors[model], mec='black', label=model, ecolor=model_colors[model])
            
                            pl.xlabel('Eccentricity (degrees)')
                            pl.ylabel(roi.replace('custom.','')+' pRF size (degrees)')
                            pl.legend(loc=0)
                            if save_figures:
                                pl.savefig('/Users/marcoaqil/PRFMapping/Figures/'+subj+'_'+
                                           roi.replace('custom.','')+'_fw-hmax.png', dpi=200, bbox_inches='tight')
                                
                        for model in subj_res['Processed Results']['Size (fwhmax)'].keys():
                            pl.figure(model+' fw_hmax', figsize=(8, 6), frameon=False)
                            for roi in rois:
                                #model-specific alpha? or all models same alpha?
                                alpha_roi = roi_mask(self.idx_rois[subj][roi], subj_res['Processed Results']['Alpha'][model])>rsq_thresh
                                
                                ecc_model_roi = subj_res['Processed Results']['Eccentricity'][model][alpha_roi]
                                fwhmax_model_roi = subj_res['Processed Results']['Size (fwhmax)'][model][alpha_roi]
                                rsq_model_roi = subj_res['Processed Results']['RSq'][model][alpha_roi]
                                
                                ecc_sorted = np.argsort(ecc_model_roi)
                                split_ecc_bins = np.array_split(ecc_sorted, 10)
                               
                                for ecc_quantile in split_ecc_bins:
                                    fw_hmax_stats[roi][model].append(weightstats.DescrStatsW(fwhmax_model_roi[ecc_quantile],
                                                                                          weights=rsq_model_roi[ecc_quantile]))
            
                                    ecc_stats[roi][model].append(weightstats.DescrStatsW(ecc_model_roi[ecc_quantile],
                                                                                          weights=rsq_model_roi[ecc_quantile]))
            
                       
                                WLS = LinearRegression()
                                WLS.fit(ecc_model_roi.reshape(-1, 1), fwhmax_model_roi, sample_weight=rsq_model_roi)
                                pl.plot([ss.mean for ss in ecc_stats[roi][model]],
                                        WLS.predict(np.array([ss.mean for ss in ecc_stats[roi][model]]).reshape(-1, 1)),
                                        color=roi_colors[roi])
                                            
                                print(roi+" "+model+" "+str(WLS.score(ecc_model_roi.reshape(-1, 1), fwhmax_model_roi, sample_weight=rsq_model_roi)))
            
                                pl.errorbar([ss.mean for ss in ecc_stats[roi][model]],
                                   [ss.mean for ss in fw_hmax_stats[roi][model]],
                                   yerr=np.array([np.abs(ss.zconfint_mean(alpha=0.05)-ss.mean) for ss in fw_hmax_stats[roi][model]]).T,
                                   xerr=np.array([np.abs(ss.zconfint_mean(alpha=0.05)-ss.mean) for ss in ecc_stats[roi][model]]).T,
                                   fmt='s', mfc=roi_colors[roi], mec='black', label=roi.replace('custom.',''), ecolor=roi_colors[roi])
            
                            pl.xlabel('Eccentricity (degrees)')
                            pl.ylabel(model+' pRF size (degrees)')
                            pl.legend(loc=0)
                            if save_figures:
                                pl.savefig('/Users/marcoaqil/PRFMapping/Figures/'+subj+'_'+
                                           model+'_fw-hmax.png', dpi=200, bbox_inches='tight')

    def ecc_surround_roi_plots(self, rois, rsq_thresh, save_figures):
        
        pl.rcParams.update({'font.size': 16})
        for space, space_res in self.main_dict.items():
            if 'fs' in space:
                for analysis, analysis_res in space_res.items():       
                    for subj, subj_res in analysis_res.items():
                        print(space+" "+analysis+" "+subj)
            
                        # binned eccentricity vs other parameters relationships       
            
                        model_colors = {'Gauss':'blue','CSS':'orange','DoG':'green','Norm':'red'}
                                                
                        #model_symbols = {'Gauss':'^','CSS':'o','DoG':'v','Norm':'D'}
                        roi_colors = dd(lambda:'blue')
                        roi_colors['custom.V1']= 'black'
                        roi_colors['custom.V2']= 'red'
                        roi_colors['custom.V3']= 'pink'
            
                        fw_atmin_stats = dd(lambda:dd(list))
                        ecc_stats = dd(lambda:dd(list))
                        
                        #exclude surrounds sizes larger than this (no surround)
                        w_max=90
            
                        for roi in rois:
            
                            pl.figure(roi+' fw_atmin', figsize=(8, 6), frameon=False)
           
                            for model in subj_res['Processed Results']['Surround Size (fwatmin)'].keys():                                
            
                                #model-specific alpha? or all models same alpha?
                                alpha_roi = (roi_mask(self.idx_rois[subj][roi], subj_res['Processed Results']['Alpha'][model])>rsq_thresh) * (subj_res['Processed Results']['Surround Size (fwatmin)'][model]<w_max)
                                
                                ecc_model_roi = subj_res['Processed Results']['Eccentricity'][model][alpha_roi]
                                fwatmin_model_roi = subj_res['Processed Results']['Surround Size (fwatmin)'][model][alpha_roi]
                                rsq_model_roi = subj_res['Processed Results']['RSq'][model][alpha_roi]
                                
                                ecc_sorted = np.argsort(ecc_model_roi)
                                split_ecc_bins = np.array_split(ecc_sorted, 10)
                               
                                for ecc_quantile in split_ecc_bins:
                                    fw_atmin_stats[roi][model].append(weightstats.DescrStatsW(fwatmin_model_roi[ecc_quantile],
                                                                                          weights=rsq_model_roi[ecc_quantile]))
            
                                    ecc_stats[roi][model].append(weightstats.DescrStatsW(ecc_model_roi[ecc_quantile],
                                                                                          weights=rsq_model_roi[ecc_quantile]))
            
                       
                                WLS = LinearRegression()
                                WLS.fit(ecc_model_roi.reshape(-1, 1), fwatmin_model_roi, sample_weight=rsq_model_roi)
                                pl.plot([ss.mean for ss in ecc_stats[roi][model]],
                                        WLS.predict(np.array([ss.mean for ss in ecc_stats[roi][model]]).reshape(-1, 1)),
                                        color=model_colors[model])
                                            
                                print(roi+" "+model+" "+str(WLS.score(ecc_model_roi.reshape(-1, 1), fwatmin_model_roi, sample_weight=rsq_model_roi)))
            
                                pl.errorbar([ss.mean for ss in ecc_stats[roi][model]],
                                   [ss.mean for ss in fw_atmin_stats[roi][model]],
                                   yerr=np.array([np.abs(ss.zconfint_mean(alpha=0.05)-ss.mean) for ss in fw_atmin_stats[roi][model]]).T,
                                   xerr=np.array([np.abs(ss.zconfint_mean(alpha=0.05)-ss.mean) for ss in ecc_stats[roi][model]]).T,
                                   fmt='s', mfc=model_colors[model], mec='black', label=model, ecolor=model_colors[model])
            
                            pl.xlabel('Eccentricity (degrees)')
                            pl.ylabel(roi.replace('custom.','')+' pRF Surround Size (degrees)')
                            pl.legend(loc=0)
                            if save_figures:
                                pl.savefig('/Users/marcoaqil/PRFMapping/Figures/'+subj+'_'+
                                           roi.replace('custom.','')+'_fw-atmin.png', dpi=200, bbox_inches='tight')
                                
                        for model in subj_res['Processed Results']['Surround Size (fwatmin)'].keys():
                            pl.figure(model+' fw_atmin', figsize=(8, 6), frameon=False)
                            for roi in rois:
                                #model-specific alpha? or all models same alpha?
                                alpha_roi = (roi_mask(self.idx_rois[subj][roi], subj_res['Processed Results']['Alpha'][model])>rsq_thresh) * (subj_res['Processed Results']['Surround Size (fwatmin)'][model]<w_max)
                                
                                ecc_model_roi = subj_res['Processed Results']['Eccentricity'][model][alpha_roi]
                                fwatmin_model_roi = subj_res['Processed Results']['Surround Size (fwatmin)'][model][alpha_roi]
                                rsq_model_roi = subj_res['Processed Results']['RSq'][model][alpha_roi]
                                
                                ecc_sorted = np.argsort(ecc_model_roi)
                                split_ecc_bins = np.array_split(ecc_sorted, 10)
                               
                                for ecc_quantile in split_ecc_bins:
                                    fw_atmin_stats[roi][model].append(weightstats.DescrStatsW(fwatmin_model_roi[ecc_quantile],
                                                                                          weights=rsq_model_roi[ecc_quantile]))
            
                                    ecc_stats[roi][model].append(weightstats.DescrStatsW(ecc_model_roi[ecc_quantile],
                                                                                          weights=rsq_model_roi[ecc_quantile]))
            
                       
                                WLS = LinearRegression()
                                WLS.fit(ecc_model_roi.reshape(-1, 1), fwatmin_model_roi, sample_weight=rsq_model_roi)
                                pl.plot([ss.mean for ss in ecc_stats[roi][model]],
                                        WLS.predict(np.array([ss.mean for ss in ecc_stats[roi][model]]).reshape(-1, 1)),
                                        color=roi_colors[roi])
                                            
                                print(roi+" "+model+" "+str(WLS.score(ecc_model_roi.reshape(-1, 1), fwatmin_model_roi, sample_weight=rsq_model_roi)))
            
                                pl.errorbar([ss.mean for ss in ecc_stats[roi][model]],
                                   [ss.mean for ss in fw_atmin_stats[roi][model]],
                                   yerr=np.array([np.abs(ss.zconfint_mean(alpha=0.05)-ss.mean) for ss in fw_atmin_stats[roi][model]]).T,
                                   xerr=np.array([np.abs(ss.zconfint_mean(alpha=0.05)-ss.mean) for ss in ecc_stats[roi][model]]).T,
                                   fmt='s', mfc=roi_colors[roi], mec='black', label=roi.replace('custom.',''), ecolor=roi_colors[roi])
            
                            pl.xlabel('Eccentricity (degrees)')
                            pl.ylabel(model+' pRF Surround Size (degrees)')
                            pl.legend(loc=0)
                            if save_figures:
                                pl.savefig('/Users/marcoaqil/PRFMapping/Figures/'+subj+'_'+
                                           model+'_fw-atmin.png', dpi=200, bbox_inches='tight')            
            
            
    def ecc_css_exp_roi_plots(self, rois, rsq_thresh, save_figures):
        
        pl.rcParams.update({'font.size': 16})
        for space, space_res in self.main_dict.items():
            if 'fs' in space:
                for analysis, analysis_res in space_res.items():       
                    for subj, subj_res in analysis_res.items():
                        print(space+" "+analysis+" "+subj)
            
                        # binned eccentricity vs other parameters relationships       
            
                        roi_colors = dd(lambda:'blue')
                        roi_colors['custom.V1']= 'black'
                        roi_colors['custom.V2']= 'red'
                        roi_colors['custom.V3']= 'pink'
            
                        css_exp_stats = dd(lambda:dd(list))
                        ecc_stats = dd(lambda:dd(list))
                        
                        pl.figure('css_exp', figsize=(8, 6), frameon=False)
                        for roi in rois:

                            if 'CSS' in subj_res['Processed Results']['RSq'].keys():                                
                                model = 'CSS'
                                #model-specific alpha? or all models same alpha?
                                alpha_roi = (roi_mask(self.idx_rois[subj][roi], subj_res['Processed Results']['Alpha'][model])>rsq_thresh)
                                
                                ecc_model_roi = subj_res['Processed Results']['Eccentricity'][model][alpha_roi]
                                css_exp_roi = subj_res['Processed Results']['CSS Exponent'][model][alpha_roi]
                                rsq_model_roi = subj_res['Processed Results']['RSq'][model][alpha_roi]
                                
                                ecc_sorted = np.argsort(ecc_model_roi)
                                split_ecc_bins = np.array_split(ecc_sorted, 10)
                               
                                for ecc_quantile in split_ecc_bins:
                                    css_exp_stats[roi][model].append(weightstats.DescrStatsW(css_exp_roi[ecc_quantile],
                                                                                          weights=rsq_model_roi[ecc_quantile]))
            
                                    ecc_stats[roi][model].append(weightstats.DescrStatsW(ecc_model_roi[ecc_quantile],
                                                                                          weights=rsq_model_roi[ecc_quantile]))
            
                       
                                # WLS = LinearRegression()
                                # WLS.fit(ecc_model_roi.reshape(-1, 1), css_exp_roi, sample_weight=rsq_model_roi)
                                # pl.plot([ss.mean for ss in ecc_stats[roi][model]],
                                #         WLS.predict(np.array([ss.mean for ss in ecc_stats[roi][model]]).reshape(-1, 1)),
                                #         color=roi_colors[roi])
                                            
                                # print(roi+" "+model+" "+str(WLS.score(ecc_model_roi.reshape(-1, 1), css_exp_roi, sample_weight=rsq_model_roi)))
            
                                pl.errorbar([ss.mean for ss in ecc_stats[roi][model]],
                                   [ss.mean for ss in css_exp_stats[roi][model]],
                                   yerr=np.array([np.abs(ss.zconfint_mean(alpha=0.05)-ss.mean) for ss in css_exp_stats[roi][model]]).T,
                                   xerr=np.array([np.abs(ss.zconfint_mean(alpha=0.05)-ss.mean) for ss in ecc_stats[roi][model]]).T,
                                   fmt='s', mfc=roi_colors[roi], mec='black', label=roi.replace('custom.',''), ecolor=roi_colors[roi])
            
                            pl.xlabel('Eccentricity (degrees)')
                            pl.ylabel('CSS Exponent')
                            pl.legend(loc=0)
                            if save_figures:
                                pl.savefig('/Users/marcoaqil/PRFMapping/Figures/'+subj+'_'+
                                           roi.replace('custom.','')+'_css-exp.png', dpi=200, bbox_inches='tight')            
            
    def ecc_norm_baselines_roi_plots(self, rois, rsq_thresh, save_figures):
        
        pl.rcParams.update({'font.size': 16})
        for space, space_res in self.main_dict.items():
            if 'fs' in space:
                for analysis, analysis_res in space_res.items():       
                    for subj, subj_res in analysis_res.items():
                        print(space+" "+analysis+" "+subj)
            
                        # binned eccentricity vs other parameters relationships       
            
                        roi_colors = dd(lambda:'blue')
                        roi_colors['custom.V1']= 'black'
                        roi_colors['custom.V2']= 'red'
                        roi_colors['custom.V3']= 'pink'
                        
                        params = {}
                        params['Norm Param. B'] = 'o'
                        params['Norm Param. D'] = 'o'
                        params['Ratio (B/D)'] = 'o'     
                        
                        
                        symbol={}
                        symbol['ABCD_100'] = 'o'
                        symbol['ACD_100'] = 's'
                        symbol['ABC_100'] = 'D'
            
                        norm_baselines_stats = dd(lambda:dd(list))
                        ecc_stats = dd(lambda:dd(list))                      
                        
                        for param in params:
                            pl.figure(analysis+param, figsize=(8, 6), frameon=False)
                            if 'Norm' in subj_res['Processed Results']['RSq'].keys():
                                model = 'Norm'
                                for roi in rois:
                                    
                                    #model-specific alpha? or all models same alpha?
                                    alpha_roi = (roi_mask(self.idx_rois[subj][roi], subj_res['Processed Results']['Alpha'][model])>rsq_thresh)
                                    
                                    ecc_model_roi = subj_res['Processed Results']['Eccentricity'][model][alpha_roi]
                                    rsq_model_roi = subj_res['Processed Results']['RSq'][model][alpha_roi]
                                    
                                    ecc_sorted = np.argsort(ecc_model_roi)
                                    split_ecc_bins = np.array_split(ecc_sorted, 8)
                                    
                                    norm_baselines_roi = subj_res['Processed Results'][param][model][alpha_roi]
                                   
                                    for ecc_quantile in split_ecc_bins:
                                        norm_baselines_stats[roi][param].append(weightstats.DescrStatsW(norm_baselines_roi[ecc_quantile],
                                                                                              weights=rsq_model_roi[ecc_quantile]))
                
                                        ecc_stats[roi][param].append(weightstats.DescrStatsW(ecc_model_roi[ecc_quantile],
                                                                                              weights=rsq_model_roi[ecc_quantile]))
                
                           
                                    # WLS = LinearRegression()
                                    # WLS.fit(ecc_model_roi.reshape(-1, 1), norm_baselines_roi, sample_weight=rsq_model_roi)
                                    # pl.plot([ss.mean for ss in ecc_stats[roi][param]],
                                    #          WLS.predict(np.array([ss.mean for ss in ecc_stats[roi][param]]).reshape(-1, 1)),
                                    #          color=roi_colors[roi])
                                                
                                    # print(roi+" "+model+" "+str(WLS.score(ecc_model_roi.reshape(-1, 1), norm_baselines_roi, sample_weight=rsq_model_roi)))
                
                                    pl.errorbar([ss.mean for ss in ecc_stats[roi][param]],
                                       [ss.mean for ss in norm_baselines_stats[roi][param]],
                                       yerr=np.array([np.abs(ss.zconfint_mean(alpha=0.05)-ss.mean) for ss in norm_baselines_stats[roi][param]]).T,
                                       xerr=np.array([np.abs(ss.zconfint_mean(alpha=0.05)-ss.mean) for ss in ecc_stats[roi][param]]).T,
                                       fmt=symbol[analysis], mfc=roi_colors[roi], mec='black', label=analysis.replace('_100','')+' '+roi.replace('custom.',''), ecolor=roi_colors[roi])
            
                                pl.xlabel('Eccentricity (degrees)')
                                pl.ylabel(param)
                                pl.legend(loc=0)
                                if save_figures:
                                    pl.savefig('/Users/marcoaqil/PRFMapping/Figures/'+subj+'_'+
                                               param.replace("/","").replace('.','').replace(' ','_')+'.png', dpi=200, bbox_inches='tight')
                                    
                                    
    def rsq_roi_plots(self, rois, rsq_thresh, save_figures):
        bar_position = 0
        last_bar_position = dd(lambda:0)
        x_ticks=[]
        x_labels=[]
        pl.rcParams.update({'font.size': 16})
        for space, space_res in self.main_dict.items():
            if 'fs' in space:
                for analysis, analysis_res in space_res.items():       
                    for subj, subj_res in analysis_res.items():
                        print(space+" "+analysis+" "+subj)
            
                        # binned eccentricity vs other parameters relationships       
            
                        model_colors = {'Gauss':'blue','CSS':'orange','DoG':'green','Norm':'red'}
                        

                        for roi in rois:
                            bar_position=last_bar_position[roi]
                            pl.figure(roi+'RSq', figsize=(8, 6), frameon=False)
                            pl.ylabel(roi.replace('custom.','')+' Mean RSq')
                            alpha_roi = roi_mask(self.idx_rois[subj][roi], subj_res['Processed Results']['Alpha']['all'])>rsq_thresh
                            
                            for model in [k for k in subj_res['Processed Results']['RSq'].keys()]:                                
            
                                bar_height = np.mean(subj_res['Processed Results']['RSq'][model][alpha_roi]-subj_res['Processed Results']['RSq']['Gauss'][alpha_roi])
                                bar_err = sem(subj_res['Processed Results']['RSq'][model][alpha_roi])
                                pl.bar(bar_position, bar_height, width=0.1, yerr=bar_err, color=model_colors[model],edgecolor='black')
                                x_ticks.append(bar_position)
                                if 'ABCD' in analysis:
                                    x_labels.append(model)
                                else:
                                    x_labels.append(analysis.replace('_100','')+'\n'+model)
                                bar_position += 0.1
                            last_bar_position[roi] = bar_position
                            pl.xticks(x_ticks,x_labels)


                               

            
                                
        
