#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 21 15:06:47 2020

@author: marcoaqil
"""


import os
opj = os.path.join
import yaml


#subj_list = [sj for sj in os.listdir('/scratch-shared/marcoaq/PRFMapping/PRFMapping-HCP') \
#             if len(sj)==6 and sj.isdecimal() and sj not in ['999999','111312', '951457']\
#                 and len([pr for pr in os.listdir('/home/marcoaq/PRFMapping/PRFMapping-HCP/prfpy') if sj in pr])<6]

#subj_list = [sj for sj in os.listdir('/home/marcoaq/PRFMapping/PRFMapping-HCP') if len(sj)==6 and sj.isdecimal()]
#subj_list.sort()
#subj_list = subj_list[10:30]

subj_list = ['sub-001', 'sub-002']

analysis_settings = '/home/marcoaq/savetimecourse_analysis_settings.yml'
chunk_nr = 0

with open(analysis_settings) as f:
    analysis_info = yaml.safe_load(f)

if "mkl_num_threads" in analysis_info:
    import mkl
    standard_max_threads = mkl.get_max_threads()
    print(standard_max_threads)
    mkl.set_num_threads(analysis_info["mkl_num_threads"])

import numpy as np

from utils.preproc_utils import create_full_stim, prepare_data


# note that screenshot paths and task names should be in the same order
n_pix = analysis_info["n_pix"]
discard_volumes = analysis_info["discard_volumes"]
screenshot_paths = analysis_info["screenshot_paths"]
screen_size_cm = analysis_info["screen_size_cm"]
screen_distance_cm = analysis_info["screen_distance_cm"]
TR = analysis_info["TR"]
normalize_integral_dx = analysis_info["normalize_integral_dx"]

task_names = analysis_info["task_names"]
data_path = analysis_info["data_path"]
fitting_space = analysis_info["fitting_space"]
save_raw_timecourse = analysis_info["save_raw_timecourse"]

#this is specific only for the save_timecourse script
if 'save_fit_timecourse' in analysis_info:
    save_fit_timecourse = analysis_info["save_fit_timecourse"]
else:
    save_fit_timecourse = False

filter_predictions = analysis_info["filter_predictions"]
filter_type = analysis_info["filter_type"]

first_modes_to_remove = analysis_info["first_modes_to_remove"]
last_modes_to_remove_percent = analysis_info["last_modes_to_remove_percent"]

window_length = analysis_info["window_length"]
polyorder = analysis_info["polyorder"]
highpass = analysis_info["highpass"]
add_mean = analysis_info["add_mean"]

filter_params = {x:analysis_info[x] for x in ["first_modes_to_remove",
                                              "last_modes_to_remove_percent",
                                              "window_length",
                                              "polyorder",
                                              "highpass",
                                              "add_mean"]}

n_jobs = analysis_info["n_jobs"]
hrf = analysis_info["hrf"]
verbose = analysis_info["verbose"]
rsq_threshold = analysis_info["rsq_threshold"]
models_to_fit = analysis_info["models_to_fit"]
n_batches = analysis_info["n_batches"]
fit_hrf = analysis_info["fit_hrf"]
fix_bold_baseline = analysis_info["fix_bold_baseline"]
if fix_bold_baseline:
    norm_bold_baseline = analysis_info["norm_bold_baseline"]


crossvalidate = analysis_info["crossvalidate"]
save_noise_ceiling = False #just to set default

if crossvalidate and "fit_task" in analysis_info and "fit_runs" in analysis_info:
    print("Can only specify one between fit_task and fit_runs for crossvalidation.")
    raise IOError
elif crossvalidate and "fit_task" in analysis_info:
    print("Performing crossvalidation over tasks.") 
    fit_task = analysis_info["fit_task"]
    fit_runs = None
elif crossvalidate and "fit_runs" in analysis_info:
    print("Performing crossvalidation over runs.")    
    fit_task = None
    fit_runs = analysis_info["fit_runs"]
    save_noise_ceiling = analysis_info["save_noise_ceiling"]
else:
    print("Not performing crossvalidation.")
    fit_task = None
    fit_runs = None        

single_hrf = analysis_info["single_hrf"]
return_noise_ceiling_fraction = analysis_info["return_noise_ceiling_fraction"]
    
xtol = analysis_info["xtol"]
ftol = analysis_info["ftol"]

dm_edges_clipping = analysis_info["dm_edges_clipping"]
min_percent_var = analysis_info["min_percent_var"]

param_bounds = analysis_info["param_bounds"]
pos_prfs_only = analysis_info["pos_prfs_only"]
normalize_RFs = analysis_info["normalize_RFs"]

param_constraints = analysis_info["param_constraints"]
surround_sigma_larger_than_centre = analysis_info["surround_sigma_larger_than_centre"]
positive_centre_only = analysis_info["positive_centre_only"]

n_chunks = analysis_info["n_chunks"]
refit_mode = analysis_info["refit_mode"].lower()
save_runs = analysis_info["save_runs"]

if "norm" in models_to_fit and "norm_model_variant" in analysis_info:
    norm_model_variant = analysis_info["norm_model_variant"]
else:
    norm_model_variant = "abcd"


if crossvalidate and fit_task is not None:
    
    #creating stimulus from screenshots
    prf_stim = create_full_stim(screenshot_paths,
                n_pix,
                discard_volumes,
                dm_edges_clipping,
                screen_size_cm,
                screen_distance_cm,
                TR,
                [fit_task],
                normalize_integral_dx)
    
    test_prf_stim = create_full_stim(screenshot_paths,
                n_pix,
                discard_volumes,
                dm_edges_clipping,
                screen_size_cm,
                screen_distance_cm,
                TR,
                [task for task in task_names if task is not fit_task],
                normalize_integral_dx)
else:
    prf_stim = create_full_stim(screenshot_paths,
                n_pix,
                discard_volumes,
                dm_edges_clipping,
                screen_size_cm,
                screen_distance_cm,
                TR,
                task_names,
                normalize_integral_dx)
    #for all other cases, a separate test-set stimulus it not needed
    test_prf_stim = prf_stim

prf_stims = dict()
for i,task in enumerate(task_names):
    prf_stims[task] = create_full_stim([screenshot_paths[i]],
                n_pix,
                discard_volumes,
                dm_edges_clipping,
                screen_size_cm,
                screen_distance_cm,
                TR,
                [task],
                normalize_integral_dx)


if "data_scaling" in analysis_info:
    data_scaling = analysis_info["data_scaling"]
else:
    data_scaling = None

if not param_bounds and norm_model_variant != "abcd":
    print("Norm model variant "+norm_model_variant+" was selected, \
          but param_bounds=False. param_bounds will be set to True.")
    param_bounds = True

#analysis_time = datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
   
data_path = opj(data_path,'prfpy')

for subj in subj_list:
    print(subj)    
    if "roi_idx_path" in analysis_info:
        roi_idx_path = analysis_info["roi_idx_path"].replace("$subj$", subj)
        if os.path.exists(roi_idx_path):
            roi_idx = np.load(roi_idx_path)
            print(f"Using ROI mask from: {roi_idx_path}")
        else:
            print("ROI mask does not exist. Fitting all.")
            roi_idx = None
    else:
        roi_idx = None
    
   
    
        
    if chunk_nr == 0 and len(save_runs)>0:
        for i, task in enumerate(task_names):
            prf_stim_single_task = prf_stims[task]
            
            test_prf_stim_single_task = prf_stim_single_task
            
            for run in save_runs:
                run_tc_dict = prepare_data(subj,
                                    prf_stim_single_task,
                                    test_prf_stim_single_task,
                                    
                                    discard_volumes,
                                    min_percent_var,
                                    fix_bold_baseline,
                                    
                                    filter_type,                               
                                    filter_params,
                                   
                                    data_path[:-5],
                                    fitting_space,
                                    data_scaling,
                                    roi_idx,
                                    save_raw_timecourse=False,
                                    
                                    crossvalidate=False,
                                    fit_runs=[run],
                                    fit_task=None,
                                    save_noise_ceiling=False)
    
                order = run_tc_dict['order']
                mask = run_tc_dict['mask']
                    
                tc_ordered = np.zeros_like(run_tc_dict['tc'])
                tc_ordered[order] = np.copy(run_tc_dict['tc'])
                
                np.save(opj(data_path.replace('scratch-shared','home'),f"{subj}_timecourse_space-{fitting_space}_task-{task}_run-{run}"), tc_ordered)
                np.save(opj(data_path.replace('scratch-shared','home'),f"{subj}_mask_space-{fitting_space}_task-{task}_run-{run}"), mask)
                
                        
    
    if save_raw_timecourse:
        print("Saving raw data")
        tc_full_iso_nonzerovar_dict = prepare_data(subj,
                                                   prf_stim,
                                                   test_prf_stim,
                                                   
                                                   discard_volumes,
                                                   min_percent_var,
                                                   fix_bold_baseline,
                                                   
                                                   filter_type,
                                                   
                                                   filter_params,
                                                   
                                                   data_path[:-5],
                                                   fitting_space,
                                                   data_scaling,
                                                   roi_idx,
                                                   save_raw_timecourse,
                                                   
                                                   crossvalidate,
                                                   fit_runs,
                                                   fit_task,
                                                   save_noise_ceiling)


    if save_fit_timecourse:
        print("Saving full fit-timecourse not just single runs/conditions")
        tc_full_iso_nonzerovar_dict = prepare_data(subj,
                                                   prf_stim,
                                                   test_prf_stim,
                                                   
                                                   discard_volumes,
                                                   min_percent_var,
                                                   fix_bold_baseline,
                                                   
                                                   filter_type,
                                                   
                                                   filter_params,
                                                   
                                                   data_path[:-5],
                                                   fitting_space,
                                                   data_scaling,
                                                   roi_idx,
                                                   False,
                                                   
                                                   crossvalidate,
                                                   fit_runs,
                                                   fit_task,
                                                   save_noise_ceiling)


        order = tc_full_iso_nonzerovar_dict['order']

        if crossvalidate:
            save_path = opj(data_path, subj+"_timecourse-test_space-"+fitting_space)
            np.save(save_path, tc_full_iso_nonzerovar_dict['tc_test'][order])

        save_path = opj(data_path, subj+"_mask_space-"+fitting_space)
        np.save(save_path, tc_full_iso_nonzerovar_dict['mask'][order])

        save_path = opj(data_path, subj+"_timecourse_space-"+fitting_space)
        np.save(save_path, tc_full_iso_nonzerovar_dict['tc'][order])


