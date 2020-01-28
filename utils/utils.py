import os
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
from matplotlib import colors
import cortex
import nibabel as nb
from collections import defaultdict as dd
from pathlib import Path
import matplotlib.image as mpimg

opj = os.path.join

from prfpy.timecourse import sgfilter_predictions
from prfpy.stimulus import PRFStimulus2D


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

    # # late-empty DM periods (for calculation of BOLD baseline)
    # shifted_dm = np.zeros_like(dm_full)
    
    # # number of TRs in which activity may linger (hrf)
    # shifted_dm[..., 7:] = dm_full[..., :-7]
    
    # late_iso_dict = {}
    # late_iso_dict['periods'] = np.where((np.sum(dm_full, axis=(0, 1)) == 0) & (
    #     np.sum(shifted_dm, axis=(0, 1)) == 0))[0]
    
    # start=0
    # for i, task_name in enumerate(task_names):
    #     stop=start+task_lengths[i]
    #     if task_name not in screenshot_paths[i]:
    #         print("WARNING: check that screenshot paths and task names are in the same order")
    #     late_iso_dict[task_name] = late_iso_dict['periods'][np.where((late_iso_dict['periods']>=start) & (late_iso_dict['periods']<stop))]
            
    #     start+=task_lengths[i]

    late_iso_dict = {}
    for i, task_name in enumerate(task_names):
        #to estimate baseline across conditions
        late_iso_dict[task_name] = np.concatenate((np.arange(baseline_volumes_begin_end[0]),np.arange(task_lengths[i]-baseline_volumes_begin_end[1], task_lengths[i])))

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
                 
                 crossvalidate,
                 fit_runs,
                 fit_task):

    if fitting_space == 'fsaverage' or fitting_space == 'fsnative':
        tc_dict = dd(lambda:dd(lambda:dd))
        tc_full_iso_dict = {}
        tc_full_iso_nonzerovar_dict = {}
        if crossvalidate:
            tc_full_iso_dict_test = {}
        
        for hemi in ['L', 'R']:
            for task_name in prf_stim.task_names:
                tc_task = []
                tc_paths = sorted(Path(opj(data_path,'fmriprep',subj)).glob(opj('**',subj+'_ses-*_task-'+task_name+'_run-*_space-'+fitting_space+'_hemi-'+hemi+'.func.gii')))

                print("For task "+task_name+", hemisphere "+hemi+" of subject "+subj+", a total of "+str(len(tc_paths))+" runs were found.")

                if not crossvalidate or fit_task is not None:
                    fit_runs = len(tc_paths)

                for tc_path in tc_paths:
                    tc_run = nb.load(str(tc_path))
    
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

                
                tc_dict[hemi][task_name]['timecourse'] = np.median([run for run in tc_task[:fit_runs]], axis=0)
                tc_dict[hemi][task_name]['baseline'] = np.median(tc_dict[hemi][task_name]['timecourse'][...,prf_stim.late_iso_dict[task_name]],
                                                   axis=-1)
            
            if crossvalidate:
                for task_name in test_prf_stim.task_names:                          
                    tc_task = []
                    tc_paths = sorted(Path(opj(data_path,'fmriprep',subj)).glob(opj('**',subj+'_ses-*_task-'+task_name+'_run-*_space-'+fitting_space+'_hemi-'+hemi+'.func.gii')))
    
                    print("For task "+task_name+", hemisphere "+hemi+" of subject "+subj+", a total of "+str(len(tc_paths))+" runs were found.")
    
                    if fit_task is not None:
                        fit_runs = 0
    
                    for tc_path in tc_paths:
                        tc_run = nb.load(str(tc_path))
        
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
    
                    
                    tc_dict[hemi][task_name]['timecourse_test'] = np.median([run for run in tc_task[fit_runs:]], axis=0)
                    tc_dict[hemi][task_name]['baseline_test'] = np.median(tc_dict[hemi][task_name]['timecourse_test'][...,test_prf_stim.late_iso_dict[task_name]],
                                                       axis=-1)


            #shift timeseries so they have the same average value in proper baseline periods across conditions
            iso_full = np.median([tc_dict[hemi][task_name]['baseline'] for task_name in prf_stim.task_names], axis=0)
    
            for task_name in prf_stim.task_names:
                iso_diff = iso_full - tc_dict[hemi][task_name]['baseline']
                tc_dict[hemi][task_name]['timecourse'] += iso_diff[...,np.newaxis]
               
            tc_full_iso_dict[hemi]=np.concatenate(tuple([tc_dict[hemi][task_name]['timecourse'] for task_name in prf_stim.task_names]), axis=-1)

            if crossvalidate:
                iso_full_test = np.median([tc_dict[hemi][task_name]['baseline_test'] for task_name in test_prf_stim.task_names], axis=0)
        
                for task_name in test_prf_stim.task_names:
                    iso_diff_test = iso_full_test - tc_dict[hemi][task_name]['baseline_test']
                    tc_dict[hemi][task_name]['timecourse_test'] += iso_diff_test[...,np.newaxis]
                   
                tc_full_iso_dict_test[hemi]=np.concatenate(tuple([tc_dict[hemi][task_name]['timecourse_test'] for task_name in test_prf_stim.task_names]), axis=-1)


        tc_full_iso = np.concatenate((tc_full_iso_dict['L'], tc_full_iso_dict['R']), axis=0)
        tc_mean = tc_full_iso.mean(-1)

        if crossvalidate:
            tc_full_iso_test = np.concatenate((tc_full_iso_dict_test['L'], tc_full_iso_dict_test['R']), axis=0)
            tc_mean_test = tc_full_iso_test.mean(-1)

        #masking flat or nearly flat timecourses
        nonlow_var = (np.abs(tc_full_iso - tc_mean[...,np.newaxis]).max(-1) > tc_mean*min_percent_var/100) * tc_mean>0

        if roi_idx is not None:
            mask = roi_mask(roi_idx, nonlow_var)
        else:
            mask = nonlow_var

        tc_full_iso_nonzerovar_dict['mask'] = mask

        #conversion to +- of % of mean
        if data_scaling in ["psc", "percent_signal_change"]:
            tc_full_iso_nonzerovar = 100*(tc_full_iso[mask] / tc_mean[mask,np.newaxis])
            if crossvalidate:
                tc_full_iso_nonzerovar_test = 100*(tc_full_iso_test[mask] / tc_mean_test[mask,np.newaxis])
        elif data_scaling == None:
            tc_full_iso_nonzerovar = tc_full_iso[mask]
            if crossvalidate:
                tc_full_iso_nonzerovar_test = tc_full_iso_test[mask]
        else:
            print("Warning: data scaling option not recognized. Using raw data.")
            tc_full_iso_nonzerovar = tc_full_iso[mask]
            if crossvalidate:
                tc_full_iso_nonzerovar_test = tc_full_iso_test[mask]

        order = np.random.permutation(tc_full_iso_nonzerovar.shape[0])

        tc_full_iso_nonzerovar_dict['order'] = order

        tc_full_iso_nonzerovar_dict['tc'] = tc_full_iso_nonzerovar[order]
        if crossvalidate:
            tc_full_iso_nonzerovar_dict['tc_test'] = tc_full_iso_nonzerovar_test[order]

        return tc_full_iso_nonzerovar_dict

    else:

        #############preparing the data (VOLUME FITTING)
        tc_dict=dd(lambda:dd)
        tc_full_iso_nonzerovar_dict = {}

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

            if not crossvalidate or fit_task is not None:
                fit_runs = len(tc_paths)

            for tc_path in tc_paths:
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
    
    
            tc_dict[task_name]['timecourse'] = np.median([run for run in tc_task[:fit_runs]], axis=0)
            tc_dict[task_name]['baseline'] = np.median(tc_dict[task_name]['timecourse'][...,prf_stim.late_iso_dict[task_name]],
                                                   axis=-1)

        if crossvalidate:
            for task_name in test_prf_stim.task_names:                
                tc_task = []
                tc_paths = sorted(Path(opj(data_path,'fmriprep',subj)).glob(opj('**',subj+'_ses-*_task-'+task_name+'_run-*_space-'+fitting_space+'_desc-preproc_bold.nii.gz')))
    
                print("For task "+task_name+", of subject "+subj+", a total of "+str(len(tc_paths))+" runs were found.")

                if fit_task is not None:
                    fit_runs = 0
                    
                for tc_path in tc_paths:
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
                    
                    
                tc_dict[task_name]['timecourse_test'] = np.median([run for run in tc_task[fit_runs:]], axis=0)
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
        nonlow_var = np.reshape(combined_brain_mask, tc_full_iso.shape[0]) * (np.abs(tc_full_iso - tc_mean[...,np.newaxis]).max(-1) > tc_mean*min_percent_var/100) * tc_mean>0

        if roi_idx is not None:
            mask = roi_mask(roi_idx, nonlow_var)
        else:
            mask = nonlow_var

        tc_full_iso_nonzerovar_dict['mask'] = np.reshape(mask, combined_brain_mask.shape)

        #conversion to +- of % of mean
        if data_scaling in ["psc", "percent_signal_change"]:
            tc_full_iso_nonzerovar = 100*(tc_full_iso[mask]/ tc_mean[mask,np.newaxis])
            if crossvalidate:
                tc_full_iso_nonzerovar_test = 100*(tc_full_iso_test[mask] / tc_mean_test[mask,np.newaxis])
        elif data_scaling == None:
            tc_full_iso_nonzerovar = tc_full_iso[mask]
            if crossvalidate:
                tc_full_iso_nonzerovar_test = tc_full_iso_test[mask]
        else:
            print("Warning: data scaling option not recognized. Using raw data.")
            tc_full_iso_nonzerovar = tc_full_iso[mask]
            if crossvalidate:
                tc_full_iso_nonzerovar_test = tc_full_iso_test[mask]

        order = np.random.permutation(tc_full_iso_nonzerovar.shape[0])

        tc_full_iso_nonzerovar_dict['order'] = order

        tc_full_iso_nonzerovar_dict['tc'] = tc_full_iso_nonzerovar[order]
        if crossvalidate:
            tc_full_iso_nonzerovar_dict['tc_test'] = tc_full_iso_nonzerovar_test[order]

        return tc_full_iso_nonzerovar_dict


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
    plt.imshow(rgba)
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
    plt.imshow(rgba)
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
    plt.imshow(rgba)
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


def combine_results(subj, fitting_space, results_folder, suffix_list, normalize_RFs=False):
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

    tc_stats = dict()
    tc_raw = np.load(opj(results_folder, subj+'_timecourse_space-'+fitting_space+'.npy'))

    tc_mean_full = np.zeros(mask.shape)
    tc_mean_full[mask] = tc_raw.mean(-1)
    tc_stats['Mean'] = tc_mean_full

    tc_var_full = np.zeros(mask.shape)
    tc_var_full[mask] = tc_raw.var(-1)
    tc_stats['Variance'] = tc_var_full

    tc_stats['TSNR'] = tc_mean_full/np.sqrt(tc_var_full)

    return {'Gauss grid':gauss_grid, 'Norm grid':norm_grid, 'Gauss':gauss,
            'CSS':css, 'DoG':dog, 'Norm':norm,
            'mask':mask, 'normalize_RFs':normalize_RFs, 'Timecourse Stats':tc_stats}

def process_results(results_dict, return_norm_profiles):
    for k, v in results_dict.items():
        if 'sub-' not in k:
            process_results(v, return_norm_profiles)
        else:
            normalize_RFs = v['Results']['normalize_RFs']
            mask = v['Results']['mask']

            #store processed results in nested default dictionary
            processed_results = dd(lambda:dd(lambda:np.zeros(mask.shape)))

            #loop over contents of single-subject analysis results (models and mask)
            for k2, v2 in v['Results'].items():
                if k2 != 'mask' and isinstance(v2, np.ndarray) and 'grid' not in k2 and 'Stats' not in k2:

                    processed_results['RSq'][k2][mask] = v2[:,-1]
                    processed_results['Eccentricity'][k2][mask] = np.sqrt(v2[:,0]**2+v2[:,1]**2)
                    #note that this takes into account the swapped axes in prfpy (which was fixed on )
                    processed_results['Polar Angle'][k2][mask] = np.arctan2(-v2[:,0], v2[:,1])
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

                        if return_norm_profiles:
                            processed_results['pRF Profiles'][k2] = np.zeros((mask.shape[0],1000))
                            (processed_results['Size (fwhmax)'][k2][mask],
                            processed_results['Surround Size (fwatmin)'][k2][mask]),
                            processed_results['pRF Profiles'][k2][mask] = fwhmax_fwatmin(k2, v2, normalize_RFs, True)
                        else:
                            (processed_results['Size (fwhmax)'][k2][mask],
                            processed_results['Surround Size (fwatmin)'][k2][mask]) = fwhmax_fwatmin(k2, v2, normalize_RFs, False)

                    else:
                        processed_results['Size (fwhmax)'][k2][mask] = fwhmax_fwatmin(k2, v2, normalize_RFs)

            v['Processed results'] = {ke : dict(va) for ke, va in processed_results.items()}


def get_subjects(main_dict,subject_list = []):
    for k, v in main_dict.items():
        if 'sub-' not in k:# and isinstance(v, (dict,dd)):
            get_subjects(v, subject_list)
        else:
            if k not in subject_list:
                subject_list.append(k)
    return subject_list
