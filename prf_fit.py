#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug  6 10:51:41 2019

@author: marcoaqil
"""

import os
opj = os.path.join
import yaml
import glob
import sys
from datetime import datetime
import time
from pathlib import Path

subj = sys.argv[1]

session = sys.argv[2]

analysis_settings = sys.argv[3]
chunk_nr = int(sys.argv[4])

with open(analysis_settings) as f:
    analysis_info = yaml.safe_load(f)

if "mkl_num_threads" in analysis_info:
    import mkl
    standard_max_threads = mkl.get_max_threads()
    print(standard_max_threads)
    mkl.set_num_threads(analysis_info["mkl_num_threads"])

import numpy as np
from scipy.optimize import LinearConstraint, NonlinearConstraint

from prfpytools.preproc_utils import create_full_stim, prepare_data, compute_clipping

from prfpy.model import Iso2DGaussianModel, Norm_Iso2DGaussianModel, DoG_Iso2DGaussianModel, CSS_Iso2DGaussianModel
from prfpy.fit import Iso2DGaussianFitter, Norm_Iso2DGaussianFitter, DoG_Iso2DGaussianFitter, CSS_Iso2DGaussianFitter

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


analysis_info["subj"] = subj
analysis_info["session"] = session

pybest = analysis_info["pybest"]

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


if isinstance(analysis_info["hrf"],list):
    hrf = analysis_info["hrf"]
elif isinstance(analysis_info["hrf"], str):
    hrf = np.load(analysis_info["hrf"].replace('$subj',subj).replace('$ses',session))

verbose = analysis_info["verbose"]
rsq_threshold = analysis_info["rsq_threshold"]
models_to_fit = analysis_info["models_to_fit"]
n_batches = analysis_info["n_batches"]

norm_full_grid = analysis_info["norm_full_grid"]
grid_fit_hrf_norm = analysis_info["grid_fit_hrf_norm"]
grid_fit_hrf_gauss = analysis_info["grid_fit_hrf_gauss"]
iter_fit_hrf = analysis_info["iter_fit_hrf"]

use_previous_gaussian_fitter_hrf = analysis_info["use_previous_gaussian_fitter_hrf"]

fix_bold_baseline = analysis_info["fix_bold_baseline"]
    
dog_grid = analysis_info["dog_grid"]
css_grid = analysis_info["css_grid"]

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

if "batch_submission_system" in analysis_info:
    batch_submission_system = analysis_info["batch_submission_system"]
else:
    batch_submission_system = None

if "norm" in models_to_fit and "norm_model_variant" in analysis_info:
    norm_model_variant = analysis_info["norm_model_variant"]
else:
    norm_model_variant = ["abcd"]

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

if "data_scaling" in analysis_info:
    data_scaling = analysis_info["data_scaling"]
else:
    data_scaling = None

if not param_bounds and norm_model_variant != "abcd":
    print("Norm model variant "+norm_model_variant+" was selected, \
          but param_bounds=False. param_bounds will be set to True.")
    param_bounds = True



#DM masking based on screen delim if possible
dm_edges_clipping = compute_clipping(analysis_info)

analysis_info["dm_edges_clipping"] = dm_edges_clipping


if batch_submission_system is not None:
    if batch_submission_system.lower() == 'slurm':
        job_id = int(os.getenv('SLURM_ARRAY_JOB_ID'))
        analysis_time = os.popen(f'sacct -j {job_id} -o submit -X --noheader | uniq').read().replace('T','-').replace(' \n','').replace(':','-')
else:
    analysis_time = datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
    job_id = np.random.randint(1e5, 1e6)

analysis_info["analysis_time"] = analysis_time
analysis_info["job_id"] = job_id

data_path = opj(data_path,'prfpy')
if not os.path.isdir(data_path):
    os.mkdir(data_path)

save_path = opj(data_path, f"{subj}_{session}_analysis_settings")


if os.path.exists(save_path+".yml"):
    previous_analysis_info = None
    delay = 0
    while previous_analysis_info == None:
        with open(save_path+".yml") as f:
            previous_analysis_info = yaml.safe_load(f)
        time.sleep(5)
        delay+=5
        #max 10 minutes wait, if more something went wrong
        if delay > 600:
            break
    
    if previous_analysis_info['job_id'] != job_id:
                
        previous_analysis_time = previous_analysis_info["analysis_time"]        
        previous_analysis_refit_mode = previous_analysis_info["refit_mode"]

        analysis_info["previous_analysis_time"] = previous_analysis_time
        analysis_info["previous_analysis_refit_mode"] = previous_analysis_refit_mode

        with open(save_path+previous_analysis_time+".yml", 'w+') as outfile:
            yaml.dump(previous_analysis_info, outfile)        
  
        with open(save_path+".yml", 'w+') as outfile:
            yaml.dump(analysis_info, outfile)
                          
    else:
        previous_analysis_time = previous_analysis_info["previous_analysis_time"]
        previous_analysis_refit_mode = previous_analysis_info["previous_analysis_refit_mode"]        
        
else:
    time.sleep(2)
    analysis_info["previous_analysis_time"] = ""
    analysis_info["previous_analysis_refit_mode"] = ""
    with open(save_path+".yml", 'w+') as outfile:
        yaml.dump(analysis_info, outfile)



if verbose == True:
    print("Creating PRF stimulus from screenshots...")    
    
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
    
if chunk_nr == 0 and len(save_runs)>0:
    for i, task in enumerate(task_names):
        prf_stim_single_task = create_full_stim([screenshot_paths[i]],
            n_pix,
            discard_volumes,
            dm_edges_clipping,
            screen_size_cm,
            screen_distance_cm,
            TR,
            [task],
            normalize_integral_dx)
        
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
                                save_noise_ceiling=False,
                                session=session,
                                pybest=pybest)

            order = run_tc_dict['order']
            mask = run_tc_dict['mask']
                
            tc_ordered = np.zeros_like(run_tc_dict['tc'])
            tc_ordered[order] = np.copy(run_tc_dict['tc'])
            
            np.save(opj(data_path.replace('scratch-shared','home'),f"{subj}_{session}_timecourse_space-{fitting_space}_task-{task}_run-{run}"), tc_ordered)
            np.save(opj(data_path.replace('scratch-shared','home'),f"{subj}_{session}_mask_space-{fitting_space}_task-{task}_run-{run}"), mask)
            
            
            


if "timecourse_data_path" in analysis_info:
    print("Using time series from: "+analysis_info["timecourse_data_path"])
    tc_full_iso_nonzerovar_dict = {}
    tc_full_iso_nonzerovar_dict['tc'] = np.load(analysis_info["timecourse_data_path"])
    tc_full_iso_nonzerovar_dict['order'] = np.arange(tc_full_iso_nonzerovar_dict['tc'].shape[0])
    tc_full_iso_nonzerovar_dict['mask'] = np.ones(tc_full_iso_nonzerovar_dict['tc'].shape[0]).astype('bool')
    
    if crossvalidate:
        if "timecourse_test_data_path" in analysis_info:
            tc_full_iso_nonzerovar_dict['tc_test'] = np.load(analysis_info["timecourse_test_data_path"])
        else:
            print("Please also provide 'timecourse_test_data_path' path for crossvalidation (filename must contain 'timecourse-test').")
            raise IOError


        save_path = opj(data_path, f"{subj}_{session}_timecourse-test_space-{fitting_space}" )
        np.save(save_path, tc_full_iso_nonzerovar_dict['tc_test'])

    save_path = opj(data_path,f"{subj}_{session}_order_space-{fitting_space}")
    np.save(save_path, tc_full_iso_nonzerovar_dict['order'])

    save_path = opj(data_path, f"{subj}_{session}_mask_space-{fitting_space}")
    np.save(save_path, tc_full_iso_nonzerovar_dict['mask'])

    save_path = opj(data_path, f"{subj}_{session}_timecourse_space-{fitting_space}")
    np.save(save_path, tc_full_iso_nonzerovar_dict['tc'])


elif os.path.exists(opj(data_path, f"{subj}_{session}_timecourse_space-{fitting_space}.npy")):
    print("Using time series from: "+opj(data_path, f"{subj}_{session}_timecourse_space-{fitting_space}.npy"))
    tc_full_iso_nonzerovar_dict = {}
    tc_full_iso_nonzerovar_dict['tc'] = np.load(opj(data_path, f"{subj}_{session}_timecourse_space-{fitting_space}.npy"))
    tc_full_iso_nonzerovar_dict['mask'] = np.load(opj(data_path, f"{subj}_{session}_mask_space-{fitting_space}.npy"))
    tc_full_iso_nonzerovar_dict['order'] = np.load(opj(data_path, f"{subj}_{session}_order_space-{fitting_space}.npy"))

    if crossvalidate:
        tc_full_iso_nonzerovar_dict['tc_test'] = np.load(opj(data_path, f"{subj}_{session}_timecourse-test_space-{fitting_space}.npy"))

else:
    if chunk_nr == 0:
        print("Preparing data for fitting (see utils.prepare_data)...")
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
                                                   save_noise_ceiling,
                                                   session,
                                                   pybest)

        if crossvalidate:
            save_path = opj(data_path, f"{subj}_{session}_timecourse-test_space-{fitting_space}")
            np.save(save_path, tc_full_iso_nonzerovar_dict['tc_test'])

        save_path = opj(data_path, f"{subj}_{session}_order_space-{fitting_space}")
        np.save(save_path, tc_full_iso_nonzerovar_dict['order'])

        save_path = opj(data_path, f"{subj}_{session}_mask_space-{fitting_space}")
        np.save(save_path, tc_full_iso_nonzerovar_dict['mask'])

        save_path = opj(data_path, f"{subj}_{session}_timecourse_space-{fitting_space}")
        np.save(save_path, tc_full_iso_nonzerovar_dict['tc'])

    else:

        while not os.path.exists(opj(data_path, f"{subj}_{session}_timecourse_space-{fitting_space}.npy")):
            time.sleep(10)
        else:
            time.sleep(10)
            print("Using time series from: "+opj(data_path, f"{subj}_{session}_timecourse_space-{fitting_space}.npy"))
            tc_full_iso_nonzerovar_dict = {}
            tc_full_iso_nonzerovar_dict['tc'] = np.load(opj(data_path, f"{subj}_{session}_timecourse_space-{fitting_space}.npy"))
            tc_full_iso_nonzerovar_dict['mask'] = np.load(opj(data_path, f"{subj}_{session}_mask_space-{fitting_space}.npy"))
            tc_full_iso_nonzerovar_dict['order'] = np.load(opj(data_path, f"{subj}_{session}_order_space-{fitting_space}.npy"))

            if crossvalidate:
                print("Using test-time series from: "+opj(data_path, f"{subj}_{session}_timecourse-test_space-{fitting_space}.npy"))
                tc_full_iso_nonzerovar_dict['tc_test'] = np.load(opj(data_path, f"{subj}_{session}_timecourse-test_space-{fitting_space}.npy"))


tc_full_iso_nonzerovar_dict['tc'] = np.array_split(tc_full_iso_nonzerovar_dict['tc'], n_chunks)[chunk_nr]
if crossvalidate:
    tc_full_iso_nonzerovar_dict['tc_test'] = np.array_split(tc_full_iso_nonzerovar_dict['tc_test'], n_chunks)[chunk_nr]

if "mkl_num_threads" in analysis_info:
    mkl.set_num_threads(1)


if verbose == True:
    print("Finished preparing data for fitting. Now creating and fitting models...")

# gauss grid params
grid_nr = 20
max_ecc_size = prf_stim.screen_size_degrees/2.0
sizes, eccs, polars = max_ecc_size * np.linspace(0.25, 1, grid_nr)**2, \
    max_ecc_size * np.linspace(0.1, 1, grid_nr)**2, \
    np.linspace(0, 2*np.pi, grid_nr)

# to set up parameter bounds in iterfit
inf = np.inf
eps = 1e-1
ss = prf_stim.screen_size_degrees

if grid_fit_hrf_gauss:
    hrf_1_grid_gauss, hrf_2_grid_gauss = np.linspace(0,10,grid_nr), np.linspace(0,0,1)
else:
    hrf_1_grid_gauss, hrf_2_grid_gauss = None, None

if grid_fit_hrf_norm:
    hrf_1_grid_norm, hrf_2_grid_norm = np.linspace(0,10,grid_nr), np.linspace(0,0,1)
else:
    hrf_1_grid_norm, hrf_2_grid_norm = None, None

if norm_full_grid:
    sizes_norm, eccs_norm, polars_norm = sizes[::2], eccs[::4], polars[::2]
    hrf_1_grid_norm, hrf_2_grid_norm = hrf_1_grid_norm[::4], hrf_2_grid_norm
else:
    sizes_norm, eccs_norm, polars_norm = None, None, None


# model parameter bounds
gauss_bounds, css_bounds, dog_bounds, norm_bounds = None, None, None, None

gauss_grid_bounds = [(0,1000)] #only prf amplitudes between 0 and 1000
css_grid_bounds = [(0,1000)] #only prf amplitudes between 0 and 1000
dog_grid_bounds = [(0,1000),(0,1000)] #only prf amplitudes between 0 and 1000, only surround amplitudes between 0 and 1000
norm_grid_bounds = [(0,1000),(0,1000)] #only prf amplitudes between 0 and 1000, only neural baseline values between 0 and 1000

if param_bounds:
    gauss_bounds = [(-1.5*max_ecc_size, 1.5*max_ecc_size),  # x
                    (-1.5*max_ecc_size, 1.5*max_ecc_size),  # y
                    (eps, 1.5*ss),  # prf size
                    (0, 1000),  # prf amplitude
                    (0, 1000)]  # bold baseline
        
    css_bounds = [(-1.5*max_ecc_size, 1.5*max_ecc_size),  # x
                    (-1.5*max_ecc_size, 1.5*max_ecc_size),  # y
                    (eps, 1.5*ss),  # prf size
                    (0, 1000),  # prf amplitude
                    (0, 1000),  # bold baseline
                    (0.01, 3)]  # CSS exponent
    
    dog_bounds = [(-1.5*max_ecc_size, 1.5*max_ecc_size),  # x
                    (-1.5*max_ecc_size, 1.5*max_ecc_size),  # y
                    (eps, 1.5*ss),  # prf size
                    (0, 1000),  # prf amplitude
                    (0, 1000),  # bold baseline
                    (0, 1000),  # surround amplitude
                    (eps, 3*ss)]  # surround size

    norm_bounds = [(-1.5*max_ecc_size, 1.5*max_ecc_size),  # x
                    (-1.5*max_ecc_size, 1.5*max_ecc_size),  # y
                    (eps, 1.5*ss),  # prf size
                    (0, 1000),  # prf amplitude
                    (0, 1000),  # bold baseline
                    (0, 1000),  # surround amplitude
                    (eps, 3*ss),  # surround size
                    (0, 1000),  # neural baseline
                    (1e-6, 1000)]  # surround baseline

    if not pos_prfs_only:
        gauss_grid_bounds[0] = (-1000,1000)
        gauss_bounds[3] = (-1000,1000)
        css_grid_bounds[0] = (-1000,1000)
        css_bounds[3] = (-1000,1000)


if param_bounds and fix_bold_baseline:
    fixed_grid_baseline = 0
    norm_bounds[4] = (0,0)
    gauss_bounds[4] = (0,0)
    css_bounds[4] = (0,0)
    dog_bounds[4] = (0,0)

#second bound set to zero to avoid potential negative hrf-response given by the disp. derivative
if param_bounds:
    if iter_fit_hrf:
        gauss_bounds += [(0,10),(0,0)]
        css_bounds += [(0,10),(0,0)]
        dog_bounds += [(0,10),(0,0)]
        norm_bounds += [(0,10),(0,0)]
    else:
        gauss_bounds += [(hrf[1],hrf[1]),(hrf[2],hrf[2])]
        css_bounds += [(hrf[1],hrf[1]),(hrf[2],hrf[2])]
        dog_bounds += [(hrf[1],hrf[1]),(hrf[2],hrf[2])]
        norm_bounds += [(hrf[1],hrf[1]),(hrf[2],hrf[2])]   





if dog_grid:
    dog_surround_amplitude_grid=np.array([0.05,0.1,0.25,0.5,0.75,1,2], dtype='float32')
    dog_surround_size_grid=np.array([3,5,8,11,14,17,20,23,26], dtype='float32')
    
if css_grid:
    css_exponent_grid=np.array([0.05,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1,1.1], dtype='float32')
    
# norm grid params
#note: prfpy surround size can be intepreted as proportion or absolute. check
dict_norm_model_variants = dict()

if "abcd" in norm_model_variant:
    dict_norm_model_variants['norm_bounds_abcd'] = np.copy(norm_bounds)
    # surround_amplitude_grid=np.array([0.05,0.2,0.4,0.7,1,3], dtype='float32')
    # surround_size_grid=np.array([3,5,8,12,18], dtype='float32')
    # neural_baseline_grid=np.array([0,1,10,100], dtype='float32')
    # surround_baseline_grid=np.array([0.1,1.0,10.0,100.0], dtype='float32')

    # surround_amplitude_grid=np.array([0.05,0.2,0.4,0.7,1,3], dtype='float32')
    # surround_size_grid=np.array([0.5,1,2.5,4,7], dtype='float32')
    # neural_baseline_grid=np.array([0,1,10,100], dtype='float32')
    # surround_baseline_grid=np.array([0.1,1.0,10.0,100.0], dtype='float32')

    dict_norm_model_variants['surround_amplitude_grid_abcd']=np.array([0.05,0.2,0.4,0.7,1,3], dtype='float32')
    dict_norm_model_variants['surround_size_grid_abcd']=np.array([3,5,8,12,18], dtype='float32')
    dict_norm_model_variants['neural_baseline_grid_abcd']=np.array([0,1,10,100], dtype='float32')
    dict_norm_model_variants['surround_baseline_grid_abcd']=np.array([0.1,1.0,10.0,100.0], dtype='float32')

    #finer grid
    # dict_norm_model_variants['surround_amplitude_grid_abcd']=np.array([0.01,0.02,0.05,0.1,0.2,0.5,0.75,1,2,3], dtype='float32')
    # dict_norm_model_variants['surround_size_grid_abcd']=np.array([3,5,8,12,18], dtype='float32')
    # dict_norm_model_variants['neural_baseline_grid_abcd']=np.array([0,1,10,100], dtype='float32')
    # dict_norm_model_variants['surround_baseline_grid_abcd']=np.array([0.1,0.5,1.0,2,5,10.0,20,50,75,100.0], dtype='float32')


if "abc" in norm_model_variant:
    dict_norm_model_variants['norm_bounds_abc'] = np.copy(norm_bounds)

    dict_norm_model_variants['surround_amplitude_grid_abc']=np.array([0.05,0.2,0.4,0.7,1,3], dtype='float32')
    dict_norm_model_variants['surround_size_grid_abc']=np.array([3,5,8,12,18], dtype='float32')
    dict_norm_model_variants['neural_baseline_grid_abc']=np.array([0,1,10,100], dtype='float32')
    dict_norm_model_variants['surround_baseline_grid_abc']=np.array([1], dtype='float32')

    dict_norm_model_variants['norm_bounds_abc'][8] = (1, 1)  # fix surround baseline
        


#this ensures that all models use the same optimizer, even if only some
#have constraints
if not param_constraints:
    constraints_gauss, constraints_css, constraints_dog, constraints_norm = None,None,None,None
else:
    constraints_gauss, constraints_css, constraints_dog, constraints_norm = [],[],[],[]

    #specific parameter constraints   
    if surround_sigma_larger_than_centre:
    
        #enforcing surround size larger than prf size in DoG model
        A_ssc_dog = np.array([[0,0,-1,0,0,0,1,0,0]])
    
        constraints_dog.append(LinearConstraint(A_ssc_dog,
                                                    lb=0,
                                                    ub=+inf))
        
        #enforcing surround size larger than prf size in norm
        A_ssc_norm = np.array([[0,0,-1,0,0,0,1,0,0,0,0]])
    
        constraints_norm.append(LinearConstraint(A_ssc_norm,
                                                    lb=0,
                                                    ub=+inf))
    
    
    if positive_centre_only:
        #enforcing positive central amplitude in DoG
        def positive_centre_prf_dog(x):
            if normalize_RFs:
                return x[3]/(2*np.pi*x[2]**2)-x[5]/(2*np.pi*x[6]**2)
            else:
                return x[3] - x[5]
    
        constraints_dog.append(NonlinearConstraint(positive_centre_prf_dog,
                                                    lb=0,
                                                    ub=+inf))
    
        #enforcing positive central amplitude in norm
        def positive_centre_prf_norm(x):
            if normalize_RFs:
                return (x[3]/(2*np.pi*x[2]**2)+x[7])/(x[5]/(2*np.pi*x[6]**2)+x[8]) - x[7]/x[8]
            else:
                return (x[3]+x[7])/(x[5]+x[8]) - x[7]/x[8]
        
        constraints_norm.append(NonlinearConstraint(positive_centre_prf_norm,
                                                    lb=0,
                                                    ub=+inf))


# MODEL COMPARISON
print("Started modeling at: "+datetime.now().strftime('%Y/%m/%d %H:%M:%S'))

# Gaussian model
gg = Iso2DGaussianModel(stimulus=prf_stim,
                          hrf=hrf,
                          filter_predictions=filter_predictions,
                          filter_type=filter_type,
                          filter_params=filter_params,
                          normalize_RFs=normalize_RFs)


gf = Iso2DGaussianFitter(
    data=tc_full_iso_nonzerovar_dict['tc'], model=gg, n_jobs=n_jobs)

if 'gauss' in models_to_fit:
    # gaussian grid fit
    if "gauss_gridparams_path" not in analysis_info and "gauss_iterparams_path" not in analysis_info:
        save_path = opj(data_path, f"{subj}_{session}_gridparams-gauss_space-{fitting_space}{chunk_nr}")

        if not os.path.exists(f"{save_path}.npy") or refit_mode == "overwrite":

            print("Starting Gaussian grid fit at "+datetime.now().strftime('%Y/%m/%d %H:%M:%S'))
            gf.grid_fit(ecc_grid=eccs,
                    polar_grid=polars,
                    size_grid=sizes,
                    verbose=verbose,
                    n_batches=n_batches,
                    fixed_grid_baseline=fixed_grid_baseline,
                    grid_bounds=gauss_grid_bounds,
                    hrf_1_grid=hrf_1_grid_gauss,
                    hrf_2_grid=hrf_2_grid_gauss)
            print("Gaussian gridfit completed at "+datetime.now().strftime('%Y/%m/%d %H:%M:%S')+
            ". voxels/vertices above "+str(rsq_threshold)+": "+str(np.sum(gf.gridsearch_params[:, -1]>rsq_threshold))+" out of "+
            str(gf.data.shape[0]))
            print("Mean rsq>"+str(rsq_threshold)+": "+str(np.mean(gf.gridsearch_params[gf.gridsearch_params[:, -1]>rsq_threshold, -1])))

            np.save(save_path, gf.gridsearch_params)

        elif os.path.exists(f"{save_path}.npy") and refit_mode in ["iterate", "skip"]:
            gf.gridsearch_params = np.load(f"{save_path}.npy")

    elif "gauss_gridparams_path" in analysis_info:

        gauss_gridparams_path = glob.glob(analysis_info[f"gauss_gridparams_path"].replace('$subj',subj).replace('$ses',session))[0]

        if len(glob.glob(analysis_info[f"gauss_gridparams_path"].replace('$subj',subj).replace('$ses',session)))>1:
            print("warning: ambiguous starting params (more than one path found). check your files.")

        prev_gauss_grid_fit_params = np.load(gauss_gridparams_path)
        prev_gauss_grid_fit_mask = np.load(gauss_gridparams_path.replace(f"gridparams-gauss","mask"))

        prev_gauss_grid_fit_params_unmasked = np.zeros((prev_gauss_grid_fit_mask.shape[0],prev_gauss_grid_fit_params.shape[1]))
        prev_gauss_grid_fit_params_unmasked[prev_gauss_grid_fit_mask] = prev_gauss_grid_fit_params

        prev_gauss_grid_fit_params_thismask = prev_gauss_grid_fit_params_unmasked[tc_full_iso_nonzerovar_dict['mask']][tc_full_iso_nonzerovar_dict['order']]

        gf.gridsearch_params = np.array_split(prev_gauss_grid_fit_params_thismask, n_chunks)[chunk_nr]



    # gaussian iterative fit
    save_path = opj(data_path, f"{subj}_{session}_iterparams-gauss_space-{fitting_space}{chunk_nr}")
    if "gauss_iterparams_path" in analysis_info:

        gauss_iterparams_path = glob.glob(analysis_info[f"gauss_iterparams_path"].replace('$subj',subj).replace('$ses',session))[0]

        if len(glob.glob(analysis_info[f"gauss_iterparams_path"].replace('$subj',subj).replace('$ses',session)))>1:
            print("warning: ambiguous starting params (more than one path found). check your files.")

        prev_gauss_iter_fit_params = np.load(gauss_iterparams_path)
        prev_gauss_iter_fit_mask = np.load(gauss_iterparams_path.replace(f"iterparams-gauss","mask"))

        prev_gauss_iter_fit_params_unmasked = np.zeros((prev_gauss_iter_fit_mask.shape[0],prev_gauss_iter_fit_params.shape[1]))
        prev_gauss_iter_fit_params_unmasked[prev_gauss_iter_fit_mask] = prev_gauss_iter_fit_params

        prev_gauss_iter_fit_params_thismask = prev_gauss_iter_fit_params_unmasked[tc_full_iso_nonzerovar_dict['mask']][tc_full_iso_nonzerovar_dict['order']]

        gf.iterative_search_params = np.array_split(prev_gauss_iter_fit_params_thismask, n_chunks)[chunk_nr]

        if refit_mode in ["overwrite", "iterate"]:

            print("Starting Gaussian iterfit at "+datetime.now().strftime('%Y/%m/%d %H:%M:%S'))

            if not iter_fit_hrf:
                gauss_bounds = np.array(gauss_bounds)
                gauss_bounds = np.repeat(gauss_bounds[np.newaxis,...], gf.iterative_search_params.shape[0], axis=0)

                gauss_bounds[:,-2,:] = np.tile(gf.iterative_search_params[:,-3],(2,1)).T
                gauss_bounds[:,-1,:] = np.tile(gf.iterative_search_params[:,-2],(2,1)).T

            gf.iterative_fit(rsq_threshold=rsq_threshold, verbose=verbose,
                            starting_params=gf.iterative_search_params,
                            bounds=gauss_bounds,
                            constraints=constraints_gauss,
                                xtol=xtol,
                                ftol=ftol)

            print("Gaussian iterfit completed at "+datetime.now().strftime('%Y/%m/%d %H:%M:%S')+". Mean rsq>"+str(rsq_threshold)+": "+str(np.mean(gf.iterative_search_params[gf.rsq_mask, -1])))
        
            if crossvalidate:
                gf.crossvalidate_fit(tc_full_iso_nonzerovar_dict['tc_test'],
                                    test_stimulus=test_prf_stim,
                                    single_hrf=single_hrf)
                print("Gaussian Crossvalidation completed at "+datetime.now().strftime('%Y/%m/%d %H:%M:%S')+". Mean rsq>"+str(rsq_threshold)+": "+str(np.mean(gf.iterative_search_params[gf.rsq_mask, -1])))
                
                if hasattr(gf, 'noise_ceiling'):
                    print("Mean noise-ceiling >"+str(rsq_threshold)+": "+str(np.mean(gf.noise_ceiling[gf.rsq_mask])))                
                    noise_ceiling_fraction = gf.iterative_search_params[gf.rsq_mask, -1]/gf.noise_ceiling[gf.rsq_mask]
                    print("Mean noise-ceiling-fraction rsq>"+str(rsq_threshold)+": "+str(np.mean(noise_ceiling_fraction)))
                    
                    if return_noise_ceiling_fraction:
                        gf.iterative_search_params[gf.rsq_mask, -1] = noise_ceiling_fraction
            
            np.save(save_path, gf.iterative_search_params)


    else:

        if not os.path.exists(f"{save_path}.npy") or refit_mode == "overwrite":

            print("Starting Gaussian iterfit at "+datetime.now().strftime('%Y/%m/%d %H:%M:%S'))

            gf.iterative_fit(rsq_threshold=rsq_threshold, verbose=verbose,
                            bounds=gauss_bounds,
                            constraints=constraints_gauss,
                                xtol=xtol,
                                ftol=ftol)
            
            print("Gaussian iterfit completed at "+datetime.now().strftime('%Y/%m/%d %H:%M:%S')+". Mean rsq>"+str(rsq_threshold)+": "+str(np.mean(gf.iterative_search_params[gf.rsq_mask, -1])))
        
            if crossvalidate:
                gf.crossvalidate_fit(tc_full_iso_nonzerovar_dict['tc_test'],
                                    test_stimulus=test_prf_stim,
                                    single_hrf=single_hrf)
                print("Gaussian Crossvalidation completed at "+datetime.now().strftime('%Y/%m/%d %H:%M:%S')+". Mean rsq>"+str(rsq_threshold)+": "+str(np.mean(gf.iterative_search_params[gf.rsq_mask, -1])))

                if hasattr(gf, 'noise_ceiling'):
                    print("Mean noise-ceiling >"+str(rsq_threshold)+": "+str(np.mean(gf.noise_ceiling[gf.rsq_mask])))   
                    noise_ceiling_fraction = gf.iterative_search_params[gf.rsq_mask, -1]/gf.noise_ceiling[gf.rsq_mask]
                    print("Mean noise-ceiling-fraction rsq>"+str(rsq_threshold)+": "+str(np.mean(noise_ceiling_fraction)))
                    
                    if return_noise_ceiling_fraction:
                        gf.iterative_search_params[gf.rsq_mask, -1] = noise_ceiling_fraction            

                
            np.save(save_path, gf.iterative_search_params)
        elif os.path.exists(f"{save_path}.npy") and refit_mode == "iterate":

            if previous_analysis_refit_mode != "iterate" or (datetime.fromtimestamp(os.stat(f"{save_path}.npy").st_mtime)) < datetime(\
                                                            int(previous_analysis_time.split('-')[0]),
                                                            int(previous_analysis_time.split('-')[1]),
                                                            int(previous_analysis_time.split('-')[2]),
                                                            int(previous_analysis_time.split('-')[3]),
                                                            int(previous_analysis_time.split('-')[4]),
                                                            int(previous_analysis_time.split('-')[5]), 0):

                print("Starting Gaussian iterfit at "+datetime.now().strftime('%Y/%m/%d %H:%M:%S'))
        
                gf.iterative_fit(rsq_threshold=rsq_threshold, verbose=verbose,
                                starting_params=np.load(f"{save_path}.npy"),
                                bounds=gauss_bounds,
                                constraints=constraints_gauss,
                                xtol=xtol,
                                ftol=ftol)
                
                print("Gaussian iterfit completed at "+datetime.now().strftime('%Y/%m/%d %H:%M:%S')+". Mean rsq>"+str(rsq_threshold)+": "+str(np.mean(gf.iterative_search_params[gf.rsq_mask, -1])))
        
                if crossvalidate:
                    gf.crossvalidate_fit(tc_full_iso_nonzerovar_dict['tc_test'],
                                    test_stimulus=test_prf_stim,
                                    single_hrf=single_hrf)

                    print("Gaussian Crossvalidation completed at "+datetime.now().strftime('%Y/%m/%d %H:%M:%S')+". Mean rsq>"+str(rsq_threshold)+": "+str(np.mean(gf.iterative_search_params[gf.rsq_mask, -1])))

                    if hasattr(gf, 'noise_ceiling'):
                        print("Mean noise-ceiling >"+str(rsq_threshold)+": "+str(np.mean(gf.noise_ceiling[gf.rsq_mask])))   
                        noise_ceiling_fraction = gf.iterative_search_params[gf.rsq_mask, -1]/gf.noise_ceiling[gf.rsq_mask]
                        print("Mean noise-ceiling-fraction rsq>"+str(rsq_threshold)+": "+str(np.mean(noise_ceiling_fraction)))
                        
                        if return_noise_ceiling_fraction:
                            gf.iterative_search_params[gf.rsq_mask, -1] = noise_ceiling_fraction
        
                np.save(save_path, gf.iterative_search_params)
            else:
                gf.iterative_search_params = np.load(f"{save_path}.npy")

        elif os.path.exists(f"{save_path}.npy") and refit_mode == "skip":
            gf.iterative_search_params = np.load(f"{save_path}.npy")




# CSS iterative fit
if "CSS" in models_to_fit:
    gg_css = CSS_Iso2DGaussianModel(stimulus=prf_stim,
                                      hrf=hrf,
                                      filter_predictions=filter_predictions,
                                      filter_type=filter_type,
                                      filter_params=filter_params,                                     
                                      normalize_RFs=normalize_RFs)
    gf_css = CSS_Iso2DGaussianFitter(
        data=tc_full_iso_nonzerovar_dict['tc'], model=gg_css, n_jobs=n_jobs,
        previous_gaussian_fitter=gf, use_previous_gaussian_fitter_hrf=use_previous_gaussian_fitter_hrf)


    # CSS grid fit
    if "css_gridparams_path" not in analysis_info and "css_iterparams_path" not in analysis_info and css_grid:
        save_path = opj(data_path, f"{subj}_{session}_gridparams-css_space-{fitting_space}{chunk_nr}")
    
        if not os.path.exists(f"{save_path}.npy") or refit_mode == "overwrite":
    
            print("Starting CSS grid fit at "+datetime.now().strftime('%Y/%m/%d %H:%M:%S'))
            gf_css.grid_fit(exponent_grid=css_exponent_grid,
                    verbose=verbose,
                    n_batches=n_batches,
                    fixed_grid_baseline=fixed_grid_baseline,
                    grid_bounds=css_grid_bounds,
                    hrf_1_grid=hrf_1_grid_gauss,
                    hrf_2_grid=hrf_2_grid_gauss)
            print("CSS gridfit completed at "+datetime.now().strftime('%Y/%m/%d %H:%M:%S')+
              ". voxels/vertices above "+str(rsq_threshold)+": "+str(np.sum(gf_css.gridsearch_params[:, -1]>rsq_threshold))+" out of "+
              str(gf_css.data.shape[0]))
            print("Mean rsq>"+str(rsq_threshold)+": "+str(np.mean(gf_css.gridsearch_params[gf_css.gridsearch_params[:, -1]>rsq_threshold, -1])))
    
            np.save(save_path, gf_css.gridsearch_params)
    
        elif os.path.exists(f"{save_path}.npy") and refit_mode in ["iterate", "skip"]:
            gf_css.gridsearch_params = np.load(f"{save_path}.npy")
    
    elif "css_gridparams_path" in analysis_info:

        css_gridparams_path = glob.glob(analysis_info[f"css_gridparams_path"].replace('$subj',subj).replace('$ses',session))[0]

        if len(glob.glob(analysis_info[f"css_gridparams_path"].replace('$subj',subj).replace('$ses',session)))>1:
            print("warning: ambiguous starting params (more than one path found). check your files.")

        prev_css_grid_fit_params = np.load(css_gridparams_path)
        prev_css_grid_fit_mask = np.load(css_gridparams_path.replace(f"gridparams-css","mask"))

        prev_css_grid_fit_params_unmasked = np.zeros((prev_css_grid_fit_mask.shape[0],prev_css_grid_fit_params.shape[1]))
        prev_css_grid_fit_params_unmasked[prev_css_grid_fit_mask] = prev_css_grid_fit_params

        prev_css_grid_fit_params_thismask = prev_css_grid_fit_params_unmasked[tc_full_iso_nonzerovar_dict['mask']][tc_full_iso_nonzerovar_dict['order']]

        gf_css.gridsearch_params = np.array_split(prev_css_grid_fit_params_thismask, n_chunks)[chunk_nr]
    
    
    save_path = opj(data_path, f"{subj}_{session}_iterparams-css_space-{fitting_space}{chunk_nr}")

    if "css_iterparams_path" in analysis_info:

        css_iterparams_path = analysis_info[f"css_iterparams_path"].replace('$subj',subj).replace('$ses',session)

        prev_css_iter_fit_params = np.load(css_iterparams_path)
        prev_css_iter_fit_mask = np.load(css_iterparams_path.replace(f"iterparams-css","mask"))

        prev_css_iter_fit_params_unmasked = np.zeros((prev_css_iter_fit_mask.shape[0],prev_css_iter_fit_params.shape[1]))
        prev_css_iter_fit_params_unmasked[prev_css_iter_fit_mask] = prev_css_iter_fit_params

        prev_css_iter_fit_params_thismask = prev_css_iter_fit_params_unmasked[tc_full_iso_nonzerovar_dict['mask']][tc_full_iso_nonzerovar_dict['order']]

        gf_css.iterative_search_params = np.array_split(prev_css_iter_fit_params_thismask, n_chunks)[chunk_nr]


        if refit_mode in ["overwrite", "iterate"]:
    
            print("Starting CSS iterfit at "+datetime.now().strftime('%Y/%m/%d %H:%M:%S'))

            if not iter_fit_hrf:
                css_bounds = np.array(css_bounds)
                css_bounds = np.repeat(css_bounds[np.newaxis,...], gf_css.iterative_search_params.shape[0], axis=0)

                css_bounds[:,-2,:] = np.tile(gf_css.iterative_search_params[:,-3],(2,1)).T
                css_bounds[:,-1,:] = np.tile(gf_css.iterative_search_params[:,-2],(2,1)).T
    
            gf_css.iterative_fit(rsq_threshold=rsq_threshold, verbose=verbose,
                             starting_params=gf_css.iterative_search_params,
                             bounds=css_bounds,
                             constraints=constraints_css,
                             xtol=xtol,
                             ftol=ftol)
            print("CSS iterfit completed at "+datetime.now().strftime('%Y/%m/%d %H:%M:%S')+". Mean rsq>"+str(rsq_threshold)+": "+str(np.mean(gf_css.iterative_search_params[gf_css.rsq_mask, -1])))
            
            if crossvalidate:
                gf_css.crossvalidate_fit(tc_full_iso_nonzerovar_dict['tc_test'],
                                 test_stimulus=test_prf_stim,
                                 single_hrf=single_hrf)
                print("CSS Crossvalidation completed at "+datetime.now().strftime('%Y/%m/%d %H:%M:%S')+". Mean rsq>"+str(rsq_threshold)+": "+str(np.mean(gf_css.iterative_search_params[gf_css.rsq_mask, -1])))

                if hasattr(gf_css, 'noise_ceiling'):
                    print("Mean noise-ceiling >"+str(rsq_threshold)+": "+str(np.mean(gf_css.noise_ceiling[gf_css.rsq_mask])))   
                    noise_ceiling_fraction = gf_css.iterative_search_params[gf_css.rsq_mask, -1]/gf_css.noise_ceiling[gf_css.rsq_mask]
                    print("Mean noise-ceiling-fraction rsq>"+str(rsq_threshold)+": "+str(np.mean(noise_ceiling_fraction)))
                    
                    if return_noise_ceiling_fraction:
                        gf_css.iterative_search_params[gf_css.rsq_mask, -1] = noise_ceiling_fraction

            np.save(save_path, gf_css.iterative_search_params)
    


    else:


        if not os.path.exists(f"{save_path}.npy") or refit_mode == "overwrite":

            print("Starting CSS iterfit at "+datetime.now().strftime('%Y/%m/%d %H:%M:%S'))
    
            gf_css.iterative_fit(rsq_threshold=rsq_threshold, verbose=verbose,
                                 bounds=css_bounds,
                                 constraints=constraints_css,
                             xtol=xtol,
                             ftol=ftol)
            print("CSS iterfit completed at "+datetime.now().strftime('%Y/%m/%d %H:%M:%S')+". Mean rsq>"+str(rsq_threshold)+": "+str(np.mean(gf_css.iterative_search_params[gf_css.rsq_mask, -1])))
            
            if crossvalidate:
                gf_css.crossvalidate_fit(tc_full_iso_nonzerovar_dict['tc_test'],
                                 test_stimulus=test_prf_stim,
                                 single_hrf=single_hrf)
                print("CSS Crossvalidation completed at "+datetime.now().strftime('%Y/%m/%d %H:%M:%S')+". Mean rsq>"+str(rsq_threshold)+": "+str(np.mean(gf_css.iterative_search_params[gf_css.rsq_mask, -1])))

                if hasattr(gf_css, 'noise_ceiling'):
                    print("Mean noise-ceiling >"+str(rsq_threshold)+": "+str(np.mean(gf_css.noise_ceiling[gf_css.rsq_mask])))  
                    noise_ceiling_fraction = gf_css.iterative_search_params[gf_css.rsq_mask, -1]/gf_css.noise_ceiling[gf_css.rsq_mask]
                    print("Mean noise-ceiling-fraction rsq>"+str(rsq_threshold)+": "+str(np.mean(noise_ceiling_fraction)))
                    
                    if return_noise_ceiling_fraction:
                        gf_css.iterative_search_params[gf_css.rsq_mask, -1] = noise_ceiling_fraction
    
            np.save(save_path, gf_css.iterative_search_params)
    
        elif os.path.exists(f"{save_path}.npy") and refit_mode == "iterate":
    
            if previous_analysis_refit_mode != "iterate" or (datetime.fromtimestamp(os.stat(f"{save_path}.npy").st_mtime)) < datetime(\
                                                        int(previous_analysis_time.split('-')[0]),
                                                        int(previous_analysis_time.split('-')[1]),
                                                        int(previous_analysis_time.split('-')[2]),
                                                        int(previous_analysis_time.split('-')[3]),
                                                        int(previous_analysis_time.split('-')[4]),
                                                        int(previous_analysis_time.split('-')[5]), 0):
                print("Starting CSS iterfit at "+datetime.now().strftime('%Y/%m/%d %H:%M:%S'))
        
                gf_css.iterative_fit(rsq_threshold=rsq_threshold, verbose=verbose,
                                     starting_params=np.load(f"{save_path}.npy"),
                                     bounds=css_bounds,
                                     constraints=constraints_css,
                             xtol=xtol,
                             ftol=ftol)
                print("CSS iterfit completed at "+datetime.now().strftime('%Y/%m/%d %H:%M:%S')+". Mean rsq>"+str(rsq_threshold)+": "+str(np.mean(gf_css.iterative_search_params[gf_css.rsq_mask, -1])))
                
                if crossvalidate:
                    gf_css.crossvalidate_fit(tc_full_iso_nonzerovar_dict['tc_test'],
                                 test_stimulus=test_prf_stim,
                                 single_hrf=single_hrf)
                    print("CSS Crossvalidation completed at "+datetime.now().strftime('%Y/%m/%d %H:%M:%S')+". Mean rsq>"+str(rsq_threshold)+": "+str(np.mean(gf_css.iterative_search_params[gf_css.rsq_mask, -1])))

                    if hasattr(gf_css, 'noise_ceiling'):
                        print("Mean noise-ceiling >"+str(rsq_threshold)+": "+str(np.mean(gf_css.noise_ceiling[gf_css.rsq_mask])))  
                        noise_ceiling_fraction = gf_css.iterative_search_params[gf_css.rsq_mask, -1]/gf_css.noise_ceiling[gf_css.rsq_mask]
                        print("Mean noise-ceiling-fraction rsq>"+str(rsq_threshold)+": "+str(np.mean(noise_ceiling_fraction)))
                        
                        if return_noise_ceiling_fraction:
                            gf_css.iterative_search_params[gf_css.rsq_mask, -1] = noise_ceiling_fraction


                np.save(save_path, gf_css.iterative_search_params)
    

    
        elif os.path.exists(f"{save_path}.npy") and refit_mode == "skip":
            gf_css.iterative_search_params = np.load(f"{save_path}.npy")



if "DoG" in models_to_fit:    
    # difference of gaussians iterative fit
    gg_dog = DoG_Iso2DGaussianModel(stimulus=prf_stim,
                                      hrf=hrf,
                                      filter_predictions=filter_predictions,
                                      filter_type=filter_type,
                                      filter_params=filter_params,                                     
                                      normalize_RFs=normalize_RFs)

    gf_dog = DoG_Iso2DGaussianFitter(data=tc_full_iso_nonzerovar_dict['tc'],
                                     model=gg_dog,
                                     n_jobs=n_jobs,
                                     previous_gaussian_fitter=gf,
                                     use_previous_gaussian_fitter_hrf=use_previous_gaussian_fitter_hrf)

    # DoG grid fit
    if "dog_gridparams_path" not in analysis_info and "dog_iterparams_path" not in analysis_info and dog_grid:
        save_path = opj(data_path, f"{subj}_{session}_gridparams-dog_space-{fitting_space}{chunk_nr}")
    
        if not os.path.exists(f"{save_path}.npy") or refit_mode == "overwrite":
    
            print("Starting DoG grid fit at "+datetime.now().strftime('%Y/%m/%d %H:%M:%S'))
            gf_dog.grid_fit(surround_amplitude_grid=dog_surround_amplitude_grid,
                            surround_size_grid=dog_surround_size_grid,
                            verbose=verbose,
                            n_batches=n_batches,
                            fixed_grid_baseline=fixed_grid_baseline,
                            grid_bounds=dog_grid_bounds,
                            hrf_1_grid=hrf_1_grid_gauss,
                            hrf_2_grid=hrf_2_grid_gauss)
            print("DoG gridfit completed at "+datetime.now().strftime('%Y/%m/%d %H:%M:%S')+
              ". voxels/vertices above "+str(rsq_threshold)+": "+str(np.sum(gf_dog.gridsearch_params[:, -1]>rsq_threshold))+" out of "+
              str(gf_dog.data.shape[0]))
            print("Mean rsq>"+str(rsq_threshold)+": "+str(np.mean(gf_dog.gridsearch_params[gf_dog.gridsearch_params[:, -1]>rsq_threshold, -1])))
    
            np.save(save_path, gf_dog.gridsearch_params)
    
        elif os.path.exists(f"{save_path}.npy") and refit_mode in ["iterate", "skip"]:
            gf_dog.gridsearch_params = np.load(f"{save_path}.npy")
    
    elif "dog_gridparams_path" in analysis_info:

        dog_gridparams_path = glob.glob(analysis_info[f"dog_gridparams_path"].replace('$subj',subj).replace('$ses',session))[0]

        if len(glob.glob(analysis_info[f"dog_gridparams_path"].replace('$subj',subj).replace('$ses',session)))>1:
            print("warning: ambiguous starting params (more than one path found). check your files.")

        prev_dog_grid_fit_params = np.load(dog_gridparams_path)
        prev_dog_grid_fit_mask = np.load(dog_gridparams_path.replace(f"gridparams-dog","mask"))

        prev_dog_grid_fit_params_unmasked = np.zeros((prev_dog_grid_fit_mask.shape[0],prev_dog_grid_fit_params.shape[1]))
        prev_dog_grid_fit_params_unmasked[prev_dog_grid_fit_mask] = prev_dog_grid_fit_params

        prev_dog_grid_fit_params_thismask = prev_dog_grid_fit_params_unmasked[tc_full_iso_nonzerovar_dict['mask']][tc_full_iso_nonzerovar_dict['order']]

        gf_dog.gridsearch_params = np.array_split(prev_dog_grid_fit_params_thismask, n_chunks)[chunk_nr]

   
    save_path = opj(data_path, f"{subj}_{session}_iterparams-dog_space-{fitting_space}{chunk_nr}")

    if "dog_iterparams_path" in analysis_info:

        dog_iterparams_path = glob.glob(analysis_info[f"dog_iterparams_path"].replace('$subj',subj).replace('$ses',session))[0]

        if len(glob.glob(analysis_info[f"dog_iterparams_path"].replace('$subj',subj).replace('$ses',session)))>1:
            print("warning: ambiguous starting params (more than one path found). check your files.")

        prev_dog_iter_fit_params = np.load(dog_iterparams_path)
        prev_dog_iter_fit_mask = np.load(dog_iterparams_path.replace(f"iterparams-dog","mask"))

        prev_dog_iter_fit_params_unmasked = np.zeros((prev_dog_iter_fit_mask.shape[0],prev_dog_iter_fit_params.shape[1]))
        prev_dog_iter_fit_params_unmasked[prev_dog_iter_fit_mask] = prev_dog_iter_fit_params

        prev_dog_iter_fit_params_thismask = prev_dog_iter_fit_params_unmasked[tc_full_iso_nonzerovar_dict['mask']][tc_full_iso_nonzerovar_dict['order']]

        gf_dog.iterative_search_params = np.array_split(prev_dog_iter_fit_params_thismask, n_chunks)[chunk_nr]

        if refit_mode in ["overwrite", "iterate"]:
    
            print("Starting DoG iterfit at "+datetime.now().strftime('%Y/%m/%d %H:%M:%S'))

            if not iter_fit_hrf:
                dog_bounds = np.array(dog_bounds)
                dog_bounds = np.repeat(dog_bounds[np.newaxis,...], gf_dog.iterative_search_params.shape[0], axis=0)

                dog_bounds[:,-2,:] = np.tile(gf_dog.iterative_search_params[:,-3],(2,1)).T
                dog_bounds[:,-1,:] = np.tile(gf_dog.iterative_search_params[:,-2],(2,1)).T
    
            gf_dog.iterative_fit(rsq_threshold=rsq_threshold, verbose=verbose,
                             starting_params=gf_dog.iterative_search_params,
                             bounds=dog_bounds,
                             constraints=constraints_dog,
                             xtol=xtol,
                             ftol=ftol)
            print("DoG iterfit completed at "+datetime.now().strftime('%Y/%m/%d %H:%M:%S')+". Mean rsq>"+str(rsq_threshold)+": "+str(np.mean(gf_dog.iterative_search_params[gf_dog.rsq_mask, -1])))
            
            if crossvalidate:
                gf_dog.crossvalidate_fit(tc_full_iso_nonzerovar_dict['tc_test'],
                                 test_stimulus=test_prf_stim,
                                 single_hrf=single_hrf)
                print("DoG Crossvalidation completed at "+datetime.now().strftime('%Y/%m/%d %H:%M:%S')+". Mean rsq>"+str(rsq_threshold)+": "+str(np.mean(gf_dog.iterative_search_params[gf_dog.rsq_mask, -1])))

                if hasattr(gf_dog, 'noise_ceiling'):
                    print("Mean noise-ceiling >"+str(rsq_threshold)+": "+str(np.mean(gf_dog.noise_ceiling[gf_dog.rsq_mask])))  
                    noise_ceiling_fraction = gf_dog.iterative_search_params[gf_dog.rsq_mask, -1]/gf_dog.noise_ceiling[gf_dog.rsq_mask]
                    print("Mean noise-ceiling-fraction rsq>"+str(rsq_threshold)+": "+str(np.mean(noise_ceiling_fraction)))
                    
                    if return_noise_ceiling_fraction:
                        gf_dog.iterative_search_params[gf_dog.rsq_mask, -1] = noise_ceiling_fraction

            np.save(save_path, gf_dog.iterative_search_params)



    else:

        if not os.path.exists(f"{save_path}.npy") or refit_mode == "overwrite":
            print("Starting DoG iterfit at "+datetime.now().strftime('%Y/%m/%d %H:%M:%S'))
    
            gf_dog.iterative_fit(rsq_threshold=rsq_threshold, verbose=verbose,
                                         bounds=dog_bounds,
                                         constraints=constraints_dog,
                                         xtol=xtol,
                                         ftol=ftol)
            print("DoG iterfit completed at "+datetime.now().strftime('%Y/%m/%d %H:%M:%S')+". Mean rsq>"+str(rsq_threshold)+": "+str(np.mean(gf_dog.iterative_search_params[gf_dog.rsq_mask, -1])))
            
            if crossvalidate:
                gf_dog.crossvalidate_fit(tc_full_iso_nonzerovar_dict['tc_test'],
                                 test_stimulus=test_prf_stim,
                                 single_hrf=single_hrf)
                print("DoG Crossvalidation completed at "+datetime.now().strftime('%Y/%m/%d %H:%M:%S')+". Mean rsq>"+str(rsq_threshold)+": "+str(np.mean(gf_dog.iterative_search_params[gf_dog.rsq_mask, -1])))
                
                if hasattr(gf_dog, 'noise_ceiling'):
                    print("Mean noise-ceiling >"+str(rsq_threshold)+": "+str(np.mean(gf_dog.noise_ceiling[gf_dog.rsq_mask])))  
                    noise_ceiling_fraction = gf_dog.iterative_search_params[gf_dog.rsq_mask, -1]/gf_dog.noise_ceiling[gf_dog.rsq_mask]
                    print("Mean noise-ceiling-fraction rsq>"+str(rsq_threshold)+": "+str(np.mean(noise_ceiling_fraction)))
                    
                    if return_noise_ceiling_fraction:
                        gf_dog.iterative_search_params[gf_dog.rsq_mask, -1] = noise_ceiling_fraction
    
            np.save(save_path, gf_dog.iterative_search_params)
    
    
    
    
        elif os.path.exists(f"{save_path}.npy") and refit_mode == "iterate":
    
            if previous_analysis_refit_mode != "iterate" or (datetime.fromtimestamp(os.stat(f"{save_path}.npy").st_mtime)) < datetime(\
                                                        int(previous_analysis_time.split('-')[0]),
                                                        int(previous_analysis_time.split('-')[1]),
                                                        int(previous_analysis_time.split('-')[2]),
                                                        int(previous_analysis_time.split('-')[3]),
                                                        int(previous_analysis_time.split('-')[4]),
                                                        int(previous_analysis_time.split('-')[5]), 0):
                print("Starting DoG iterfit at "+datetime.now().strftime('%Y/%m/%d %H:%M:%S'))
        
                gf_dog.iterative_fit(rsq_threshold=rsq_threshold, verbose=verbose,
                                     starting_params=np.load(f"{save_path}.npy"),
                                     bounds=dog_bounds,
                                     constraints=constraints_dog,
                                     xtol=xtol,
                                     ftol=ftol)
                print("DoG iterfit completed at "+datetime.now().strftime('%Y/%m/%d %H:%M:%S')+". Mean rsq>"+str(rsq_threshold)+": "+str(np.mean(gf_dog.iterative_search_params[gf_dog.rsq_mask, -1])))
                
                if crossvalidate:
                    gf_dog.crossvalidate_fit(tc_full_iso_nonzerovar_dict['tc_test'],
                                 test_stimulus=test_prf_stim,
                                 single_hrf=single_hrf)
                    print("DoG Crossvalidation completed at "+datetime.now().strftime('%Y/%m/%d %H:%M:%S')+". Mean rsq>"+str(rsq_threshold)+": "+str(np.mean(gf_dog.iterative_search_params[gf_dog.rsq_mask, -1])))

                    if hasattr(gf_dog, 'noise_ceiling'):
                        print("Mean noise-ceiling >"+str(rsq_threshold)+": "+str(np.mean(gf_dog.noise_ceiling[gf_dog.rsq_mask])))  
                        noise_ceiling_fraction = gf_dog.iterative_search_params[gf_dog.rsq_mask, -1]/gf_dog.noise_ceiling[gf_dog.rsq_mask]
                        print("Mean noise-ceiling-fraction rsq>"+str(rsq_threshold)+": "+str(np.mean(noise_ceiling_fraction)))
                        
                        if return_noise_ceiling_fraction:
                            gf_dog.iterative_search_params[gf_dog.rsq_mask, -1] = noise_ceiling_fraction
                    
                np.save(save_path, gf_dog.iterative_search_params)
    
    

    
        elif os.path.exists(f"{save_path}.npy") and refit_mode == "skip":
            gf_dog.iterative_search_params = np.load(f"{save_path}.npy")





if "norm" in models_to_fit:
    for variant in norm_model_variant:
        # normalization iterative fit
        gg_norm = Norm_Iso2DGaussianModel(stimulus=prf_stim,
                                            hrf=hrf,
                                            filter_predictions=filter_predictions,
                                            filter_type=filter_type,
                                            filter_params=filter_params,                                       
                                            normalize_RFs=normalize_RFs)

        gf_norm = Norm_Iso2DGaussianFitter(data=tc_full_iso_nonzerovar_dict['tc'],
                                        model=gg_norm,
                                        n_jobs=n_jobs,
                                        previous_gaussian_fitter=gf,
                                        use_previous_gaussian_fitter_hrf=use_previous_gaussian_fitter_hrf)
        
        #normalization grid stage
        if f"norm{variant}_gridparams_path" not in analysis_info and f"norm{variant}_iterparams_path" not in analysis_info:

            save_path = opj(data_path, f"{subj}_{session}_gridparams-norm{variant}_space-{fitting_space}{chunk_nr}")

            if not os.path.exists(f"{save_path}.npy") or refit_mode == "overwrite":


                print("Starting norm grid fit at "+datetime.now().strftime('%Y/%m/%d %H:%M:%S'))
                gf_norm.grid_fit(dict_norm_model_variants[f'surround_amplitude_grid_{variant}'], 
                            dict_norm_model_variants[f'surround_size_grid_{variant}'],
                            dict_norm_model_variants[f'neural_baseline_grid_{variant}'],
                            dict_norm_model_variants[f'surround_baseline_grid_{variant}'],
                            verbose=verbose,
                            n_batches=n_batches,
                            rsq_threshold=rsq_threshold,
                            fixed_grid_baseline=fixed_grid_baseline,
                            grid_bounds=norm_grid_bounds,
                            hrf_1_grid=hrf_1_grid_norm,
                            hrf_2_grid=hrf_2_grid_norm,
                            ecc_grid=eccs_norm, size_grid=sizes_norm, polar_grid=polars_norm)
            
                print(f"Norm {variant} gridfit completed at "+datetime.now().strftime('%Y/%m/%d %H:%M:%S')+". Mean rsq>"+str(rsq_threshold)+": "+str(np.mean(gf_norm.gridsearch_params[gf_norm.gridsearch_rsq_mask, -1])))
            
                np.save(save_path, gf_norm.gridsearch_params)
            elif os.path.exists(f"{save_path}.npy") and refit_mode in ["iterate","skip"]:
                gf_norm.gridsearch_params = np.load(f"{save_path}.npy")

        elif f"norm{variant}_gridparams_path" in analysis_info:

            norm_gridparams_path = glob.glob(analysis_info[f"norm{variant}_gridparams_path"].replace('$subj',subj).replace('$ses',session))[0]

            if len(glob.glob(analysis_info[f"norm{variant}_gridparams_path"].replace('$subj',subj).replace('$ses',session)))>1:
                print("warning: ambiguous starting params (more than one path found). check your files.")

            prev_norm_grid_fit_params = np.load(norm_gridparams_path)
            prev_norm_grid_fit_mask = np.load(norm_gridparams_path.replace(f"gridparams-norm{variant}","mask"))

            prev_norm_grid_fit_params_unmasked = np.zeros((prev_norm_grid_fit_mask.shape[0],prev_norm_grid_fit_params.shape[1]))
            prev_norm_grid_fit_params_unmasked[prev_norm_grid_fit_mask] = prev_norm_grid_fit_params

            prev_norm_grid_fit_params_thismask = prev_norm_grid_fit_params_unmasked[tc_full_iso_nonzerovar_dict['mask']][tc_full_iso_nonzerovar_dict['order']]

            gf_norm.gridsearch_params = np.array_split(prev_norm_grid_fit_params_thismask, n_chunks)[chunk_nr]


        save_path = opj(data_path, f"{subj}_{session}_iterparams-norm{variant}_space-{fitting_space}{chunk_nr}")

        if f"norm{variant}_iterparams_path" in analysis_info:

            norm_iterparams_path = glob.glob(analysis_info[f"norm{variant}_iterparams_path"].replace('$subj',subj).replace('$ses',session))[0]

            if len(glob.glob(analysis_info[f"norm{variant}_iterparams_path"].replace('$subj',subj).replace('$ses',session)))>1:
                print("warning: ambiguous starting params (more than one path found). check your files.")

            prev_norm_iter_fit_params = np.load(norm_iterparams_path)
            prev_norm_iter_fit_mask = np.load(norm_iterparams_path.replace(f"iterparams-norm{variant}","mask"))

            prev_norm_iter_fit_params_unmasked = np.zeros((prev_norm_iter_fit_mask.shape[0],prev_norm_iter_fit_params.shape[1]))
            prev_norm_iter_fit_params_unmasked[prev_norm_iter_fit_mask] = prev_norm_iter_fit_params

            prev_norm_iter_fit_params_thismask = prev_norm_iter_fit_params_unmasked[tc_full_iso_nonzerovar_dict['mask']][tc_full_iso_nonzerovar_dict['order']]

            gf_norm.iterative_search_params = np.array_split(prev_norm_iter_fit_params_thismask, n_chunks)[chunk_nr]

            if refit_mode in ["overwrite", "iterate"]:
        
                print(f"Starting Norm {variant} iterfit at "+datetime.now().strftime('%Y/%m/%d %H:%M:%S'))

                if not iter_fit_hrf:
                    dict_norm_model_variants[f'norm_bounds_{variant}'] = np.array(dict_norm_model_variants[f'norm_bounds_{variant}'])
                    dict_norm_model_variants[f'norm_bounds_{variant}'] = np.repeat(dict_norm_model_variants[f'norm_bounds_{variant}'][np.newaxis,...], gf_norm.iterative_search_params.shape[0], axis=0)

                    dict_norm_model_variants[f'norm_bounds_{variant}'][:,-2,:] = np.tile(gf_norm.iterative_search_params[:,-3],(2,1)).T
                    dict_norm_model_variants[f'norm_bounds_{variant}'][:,-1,:] = np.tile(gf_norm.iterative_search_params[:,-2],(2,1)).T

        
                gf_norm.iterative_fit(rsq_threshold=rsq_threshold, verbose=verbose,
                                starting_params=gf_norm.iterative_search_params,
                                bounds=dict_norm_model_variants[f'norm_bounds_{variant}'],
                                constraints=constraints_norm,
                                xtol=xtol,
                                ftol=ftol)
                print(f"Norm {variant} iterfit completed at "+datetime.now().strftime('%Y/%m/%d %H:%M:%S')+". Mean rsq>"+str(rsq_threshold)+": "+str(np.mean(gf_norm.iterative_search_params[gf_norm.rsq_mask, -1])))
                
                if crossvalidate:
                    gf_norm.crossvalidate_fit(tc_full_iso_nonzerovar_dict['tc_test'],
                                    test_stimulus=test_prf_stim,
                                    single_hrf=single_hrf)
                    print(f"Norm {variant} Crossvalidation completed at "+datetime.now().strftime('%Y/%m/%d %H:%M:%S')+". Mean rsq>"+str(rsq_threshold)+": "+str(np.mean(gf_norm.iterative_search_params[gf_norm.rsq_mask, -1])))

                    if hasattr(gf_norm, 'noise_ceiling'):
                        print("Mean noise-ceiling >"+str(rsq_threshold)+": "+str(np.mean(gf_norm.noise_ceiling[gf_norm.rsq_mask])))  
                        noise_ceiling_fraction = gf_norm.iterative_search_params[gf_norm.rsq_mask, -1]/gf_norm.noise_ceiling[gf_norm.rsq_mask]
                        print("Mean noise-ceiling-fraction rsq>"+str(rsq_threshold)+": "+str(np.mean(noise_ceiling_fraction)))
                        
                        if return_noise_ceiling_fraction:
                            gf_norm.iterative_search_params[gf_norm.rsq_mask, -1] = noise_ceiling_fraction
                
                np.save(save_path, gf_norm.iterative_search_params)


        else:
            if not os.path.exists(f"{save_path}.npy") or refit_mode == "overwrite":
        
                print(f"Starting norm {variant} iterfit at "+datetime.now().strftime('%Y/%m/%d %H:%M:%S'))
        
                gf_norm.iterative_fit(rsq_threshold=rsq_threshold, verbose=verbose,
                                            bounds=dict_norm_model_variants[f'norm_bounds_{variant}'],
                                            constraints=constraints_norm,
                                            xtol=xtol,
                                            ftol=ftol)
                print(f"Norm {variant} iterfit completed at "+datetime.now().strftime('%Y/%m/%d %H:%M:%S')+". Mean rsq>"+str(rsq_threshold)+": "+str(np.mean(gf_norm.iterative_search_params[gf_norm.rsq_mask, -1])))
                
                if crossvalidate:
                    gf_norm.crossvalidate_fit(tc_full_iso_nonzerovar_dict['tc_test'],
                                    test_stimulus=test_prf_stim,
                                    single_hrf=single_hrf)
                    print(f"Norm {variant} Crossvalidation completed at "+datetime.now().strftime('%Y/%m/%d %H:%M:%S')+". Mean rsq>"+str(rsq_threshold)+": "+str(np.mean(gf_norm.iterative_search_params[gf_norm.rsq_mask, -1])))

                    if hasattr(gf_norm, 'noise_ceiling'):
                        print("Mean noise-ceiling >"+str(rsq_threshold)+": "+str(np.mean(gf_norm.noise_ceiling[gf_norm.rsq_mask])))  
                        noise_ceiling_fraction = gf_norm.iterative_search_params[gf_norm.rsq_mask, -1]/gf_norm.noise_ceiling[gf_norm.rsq_mask]
                        print("Mean noise-ceiling-fraction rsq>"+str(rsq_threshold)+": "+str(np.mean(noise_ceiling_fraction)))
                        
                        if return_noise_ceiling_fraction:
                            gf_norm.iterative_search_params[gf_norm.rsq_mask, -1] = noise_ceiling_fraction
                
                np.save(save_path, gf_norm.iterative_search_params)
        
        
            elif os.path.exists(f"{save_path}.npy") and refit_mode == "iterate":
        
                if previous_analysis_refit_mode != "iterate" or (datetime.fromtimestamp(os.stat(f"{save_path}.npy").st_mtime)) < datetime(\
                                                            int(previous_analysis_time.split('-')[0]),
                                                            int(previous_analysis_time.split('-')[1]),
                                                            int(previous_analysis_time.split('-')[2]),
                                                            int(previous_analysis_time.split('-')[3]),
                                                            int(previous_analysis_time.split('-')[4]),
                                                            int(previous_analysis_time.split('-')[5]), 0):
        
                    print(f"Starting norm {variant} iterfit at "+datetime.now().strftime('%Y/%m/%d %H:%M:%S'))
            
                    gf_norm.iterative_fit(rsq_threshold=rsq_threshold, verbose=verbose,
                                        starting_params=np.load(f"{save_path}.npy"),
                                                bounds=dict_norm_model_variants[f'norm_bounds_{variant}'],
                                                constraints=constraints_norm,
                                                xtol=xtol,
                                                ftol=ftol)
                    print("Norm iterfit completed at "+datetime.now().strftime('%Y/%m/%d %H:%M:%S')+". Mean rsq>"+str(rsq_threshold)+": "+str(np.mean(gf_norm.iterative_search_params[gf_norm.rsq_mask, -1])))
                
                    if crossvalidate:
                        gf_norm.crossvalidate_fit(tc_full_iso_nonzerovar_dict['tc_test'],
                                    test_stimulus=test_prf_stim,
                                    single_hrf=single_hrf)
                        print(f"Norm {variant} Crossvalidation completed at "+datetime.now().strftime('%Y/%m/%d %H:%M:%S')+". Mean rsq>"+str(rsq_threshold)+": "+str(np.mean(gf_norm.iterative_search_params[gf_norm.rsq_mask, -1])))
                
                        if hasattr(gf_norm, 'noise_ceiling'):
                            print("Mean noise-ceiling >"+str(rsq_threshold)+": "+str(np.mean(gf_norm.noise_ceiling[gf_norm.rsq_mask])))  
                            noise_ceiling_fraction = gf_norm.iterative_search_params[gf_norm.rsq_mask, -1]/gf_norm.noise_ceiling[gf_norm.rsq_mask]
                            print("Mean noise-ceiling-fraction rsq>"+str(rsq_threshold)+": "+str(np.mean(noise_ceiling_fraction)))
                            
                            if return_noise_ceiling_fraction:
                                gf_norm.iterative_search_params[gf_norm.rsq_mask, -1] = noise_ceiling_fraction    
                                
                                
                    np.save(save_path, gf_norm.iterative_search_params)
        

        
            elif os.path.exists(f"{save_path}.npy") and refit_mode == "skip":
                gf_norm.iterative_search_params = np.load(f"{save_path}.npy")
