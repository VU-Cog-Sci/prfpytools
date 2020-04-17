#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug  6 10:51:41 2019

@author: marcoaqil
"""

import os
opj = os.path.join
import yaml
import sys
from datetime import datetime
import time

subj = sys.argv[1]
analysis_settings = sys.argv[2]
chunk_nr = int(sys.argv[3])

with open(analysis_settings) as f:
    analysis_info = yaml.safe_load(f)

if "mkl_num_threads" in analysis_info:
    import mkl
    standard_max_threads = mkl.get_max_threads()
    print(standard_max_threads)
    mkl.set_num_threads(analysis_info["mkl_num_threads"])

import numpy as np
from scipy.optimize import LinearConstraint, NonlinearConstraint

from utils.utils import create_full_stim, prepare_data

from prfpy.model import Iso2DGaussianModel, Norm_Iso2DGaussianModel, DoG_Iso2DGaussianModel, CSS_Iso2DGaussianModel
from prfpy.fit import Iso2DGaussianFitter, Norm_Iso2DGaussianFitter, DoG_Iso2DGaussianFitter, CSS_Iso2DGaussianFitter

# note that screenshot paths and task names should be in the same order
n_pix = analysis_info["n_pix"]
discard_volumes = analysis_info["discard_volumes"]
screenshot_paths = analysis_info["screenshot_paths"]
screen_size_cm = analysis_info["screen_size_cm"]
screen_distance_cm = analysis_info["screen_distance_cm"]
TR = analysis_info["TR"]

task_names = analysis_info["task_names"]
data_path = analysis_info["data_path"]
fitting_space = analysis_info["fitting_space"]
save_raw_timecourse = analysis_info["save_raw_timecourse"]

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
baseline_volumes_begin_end = analysis_info["baseline_volumes_begin_end"]
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

analysis_time = datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
analysis_info["analysis_time"] = analysis_time

data_path = opj(data_path,'prfpy')
save_path = opj(data_path, subj+"_analysis_settings")


if chunk_nr == 0:
    if os.path.exists(save_path+".yml"):
        with open(save_path+".yml") as f:
            previous_analysis_info = yaml.safe_load(f)

        previous_analysis_time = previous_analysis_info["analysis_time"]
        previous_analysis_refit_mode = previous_analysis_info["refit_mode"].lower()

        analysis_info["previous_analysis_time"] = previous_analysis_time
        analysis_info["previous_analysis_refit_mode"] = previous_analysis_refit_mode

        os.rename(save_path+".yml",save_path+previous_analysis_time+".yml")
    else:
        analysis_info["previous_analysis_time"] = ""
        analysis_info["previous_analysis_refit_mode"] = ""

    with open(save_path+".yml", 'w+') as outfile:
        yaml.dump(analysis_info, outfile)
else:
    while True:
        time.sleep(1)
        try:
            with open(save_path+".yml") as f:
                current_an = yaml.safe_load(f)
            previous_analysis_time = current_an["previous_analysis_time"]
            previous_analysis_refit_mode = current_an["previous_analysis_refit_mode"]
        except:
            continue
        break



if verbose == True:
    print("Creating PRF stimulus from screenshots...")
    
    
if crossvalidate and fit_task is not None:
    
    #creating stimulus from screenshots
    prf_stim = create_full_stim(screenshot_paths,
                n_pix,
                discard_volumes,
                baseline_volumes_begin_end,
                dm_edges_clipping,
                screen_size_cm,
                screen_distance_cm,
                TR,
                [fit_task])
    
    test_prf_stim = create_full_stim(screenshot_paths,
                n_pix,
                discard_volumes,
                baseline_volumes_begin_end,
                dm_edges_clipping,
                screen_size_cm,
                screen_distance_cm,
                TR,
                [task for task in task_names if task is not fit_task])
else:
    prf_stim = create_full_stim(screenshot_paths,
                n_pix,
                discard_volumes,
                baseline_volumes_begin_end,
                dm_edges_clipping,
                screen_size_cm,
                screen_distance_cm,
                TR,
                task_names)
    #for all other cases, a separate test-set stimulus it not needed
    test_prf_stim = prf_stim
    
if chunk_nr == 0 and len(save_runs)>0:
    for i, task in enumerate(task_names):
        prf_stim_single_task = create_full_stim(screenshot_paths[i],
            n_pix,
            discard_volumes,
            baseline_volumes_begin_end,
            dm_edges_clipping,
            screen_size_cm,
            screen_distance_cm,
            TR,
            [task])
        
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
            
            np.save(opj(data_path.replace('scratch_shared','home'),f"{subj}_timecourse_task-{task}_run-{run}"), tc_ordered)
            np.save(opj(data_path.replace('scratch_shared','home'),f"{subj}_mask_task-{task}_run-{run}"), mask)
            
            
            


if "timecourse_data_path" in analysis_info:
    print("Using time series from: "+analysis_info["timecourse_data_path"])
    tc_full_iso_nonzerovar_dict = {}
    tc_full_iso_nonzerovar_dict['tc'] = np.load(analysis_info["timecourse_data_path"])
    if crossvalidate:
        if "timecourse_test_data_path" in analysis_info:
            tc_full_iso_nonzerovar_dict['tc_test'] = np.load(analysis_info["timecourse_test_data_path"])
        else:
            print("Please also provide 'timecourse_test_data_path' path for crossvalidation (filename must contain 'timecourse-test').")
            raise IOError
elif os.path.exists(opj(data_path, subj+"_timecourse_space-"+fitting_space+".npy")):
    print("Using time series from: "+opj(data_path, subj+"_timecourse_space-"+fitting_space+".npy"))
    tc_full_iso_nonzerovar_dict = {}
    tc_full_iso_nonzerovar_dict['tc'] = np.load(opj(data_path, subj+"_timecourse_space-"+fitting_space+".npy"))
    if crossvalidate:
        tc_full_iso_nonzerovar_dict['tc_test'] = np.load(opj(data_path, subj+"_timecourse-test_space-"+fitting_space+".npy"))

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
                                                   save_noise_ceiling)

        if crossvalidate:
            save_path = opj(data_path, subj+"_timecourse-test_space-"+fitting_space)
            np.save(save_path, tc_full_iso_nonzerovar_dict['tc_test'])

        save_path = opj(data_path, subj+"_order_space-"+fitting_space)
        np.save(save_path, tc_full_iso_nonzerovar_dict['order'])

        save_path = opj(data_path, subj+"_mask_space-"+fitting_space)
        np.save(save_path, tc_full_iso_nonzerovar_dict['mask'])

        save_path = opj(data_path, subj+"_timecourse_space-"+fitting_space)
        np.save(save_path, tc_full_iso_nonzerovar_dict['tc'])

    else:

        while not os.path.exists(opj(data_path, subj+"_timecourse_space-"+fitting_space+".npy")):
            time.sleep(30)
        else:
            print("Using time series from: "+opj(data_path, subj+"_timecourse_space-"+fitting_space+".npy"))
            tc_full_iso_nonzerovar_dict = {}
            tc_full_iso_nonzerovar_dict['tc'] = np.load(opj(data_path, subj+"_timecourse_space-"+fitting_space+".npy"))
            if crossvalidate:
                print("Using test-time series from: "+opj(data_path, subj+"_timecourse-test_space-"+fitting_space+".npy"))
                tc_full_iso_nonzerovar_dict['tc_test'] = np.load(opj(data_path, subj+"_timecourse-test_space-"+fitting_space+".npy"))


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

# model parameter bounds
gauss_bounds, css_bounds, dog_bounds, norm_bounds = None, None, None, None

if param_bounds:
    gauss_bounds = [(-max_ecc_size, max_ecc_size),  # x
                    (-max_ecc_size, max_ecc_size),  # y
                    (eps, ss),  # prf size
                    (0, +inf),  # prf amplitude
                    (0, +inf)]  # bold baseline
    
    css_bounds = [(-max_ecc_size, max_ecc_size),  # x
                  (-max_ecc_size, max_ecc_size),  # y
                  (eps, ss),  # prf size
                  (0, +inf),  # prf amplitude
                  (0, +inf),  # bold baseline
                  (0.01, 3)]  # CSS exponent
    
    dog_bounds = [(-max_ecc_size, max_ecc_size),  # x
                  (-max_ecc_size, max_ecc_size),  # y
                  (eps, ss),  # prf size
                  (0, +inf),  # prf amplitude
                  (0, +inf),  # bold baseline
                  (0, +inf),  # surround amplitude
                  (eps, 2*ss)]  # surround size

    norm_bounds = [(-max_ecc_size, max_ecc_size),  # x
               (-max_ecc_size, max_ecc_size),  # y
               (eps, ss),  # prf size
               (0, +inf),  # prf amplitude
               (0, +inf),  # bold baseline
               (0, +inf),  # surround amplitude
               (eps, 2*ss),  # surround size
               (0, +inf),  # neural baseline
               (1e-6, +inf)]  # surround baseline

# norm grid params
if norm_model_variant == "abcd":
    surround_amplitude_grid=np.array([0,0.05,0.2,1], dtype='float32')
    surround_size_grid=np.array([3,6,10,20], dtype='float32')
    neural_baseline_grid=np.array([0,1,10,100], dtype='float32')
    surround_baseline_grid=np.array([0.1,1.0,10.0,100.0], dtype='float32')
    

elif norm_model_variant == "abc":
    surround_amplitude_grid=np.array([0,0.05,0.2,1,2,5,10], dtype='float32')
    surround_size_grid=np.array([2,3,4,6,10,15], dtype='float32')
    neural_baseline_grid=np.array([0,0.1,0.5,1,2,4,8,10], dtype='float32')
    surround_baseline_grid=np.array([1], dtype='float32')

    norm_bounds[8] = (1, 1)  # fix surround baseline
        
elif norm_model_variant == "acd":
    surround_amplitude_grid=np.array([0,0.05,0.2,1,2,5,10], dtype='float32')
    surround_size_grid=np.array([2,3,4,6,10,15], dtype='float32')
    neural_baseline_grid=np.array([1], dtype='float32')
    surround_baseline_grid=np.array([0.1,1.0,10.0,100.0], dtype='float32')

    norm_bounds[7] = (1, 1)  # fix neural baseline

elif norm_model_variant == "abd":
    surround_amplitude_grid=np.array([1], dtype='float32')
    surround_size_grid=np.array([2,3,4,5,6,8,10,15], dtype='float32')
    neural_baseline_grid=np.array([0,0.1,0.5,1,2,4,8,10], dtype='float32')
    surround_baseline_grid=np.array([0.1,1.0,10.0,100.0], dtype='float32')

    norm_bounds[5] = (1, 1)  # fix surround amplitude 


elif norm_model_variant == "ab":
    surround_amplitude_grid=np.array([1], dtype='float32')
    surround_size_grid=np.array([1,2,3,4,5,6,8,10,15], dtype='float32')
    neural_baseline_grid=np.array([0,0.1,0.5,1,2,4,6,8,10,100], dtype='float32')
    surround_baseline_grid=np.array([1], dtype='float32')
    
    norm_bounds[5] = (1, 1)  # fix surround amplitude 
    norm_bounds[8] = (1, 1)  # fix surround baseline


if param_bounds and fix_bold_baseline:
    gauss_bounds[4] = (100,100)
    css_bounds[4] = (100,100)
    dog_bounds[4] = (100,100)
    norm_bounds[4] = (norm_bold_baseline,norm_bold_baseline)


#second bound set to zero to avoid potential negative hrf-response given by the disp. derivative
if param_bounds and fit_hrf:
    gauss_bounds += [(0,10),(0,0)]
    css_bounds += [(0,10),(0,0)]
    dog_bounds += [(0,10),(0,0)]
    norm_bounds += [(0,10),(0,0)]
    
if param_bounds and not fit_hrf and single_hrf and refit_mode == 'iterate':
    #ugly AF. using max() just because it should all be zeros aside from the median values
    gauss_hrf = np.load(opj(data_path, subj+"_iterparams-gauss_space-"+fitting_space+str(chunk_nr)+".npy"))[:,-3].max()
    gauss_bounds += [(gauss_hrf,gauss_hrf),(0,0)]

    if "CSS" in models_to_fit:
        css_hrf = np.load(opj(data_path, subj+"_iterparams-css_space-"+fitting_space+str(chunk_nr)+".npy"))[:,-3].max()
        css_bounds += [(css_hrf,css_hrf),(0,0)]
    if "DoG" in models_to_fit:
        dog_hrf = np.load(opj(data_path, subj+"_iterparams-dog_space-"+fitting_space+str(chunk_nr)+".npy"))[:,-3].max()
        dog_bounds += [(dog_hrf,dog_hrf),(0,0)]
    if "norm" in models_to_fit:
        norm_hrf = np.load(opj(data_path, subj+"_iterparams-norm_space-"+fitting_space+str(chunk_nr)+".npy"))[:,-3].max()  
        norm_bounds += [(norm_hrf,norm_hrf),(0,0)]    
    
#this ensures that all models use the same optimizer, even if only some
#have constraints
if not param_constraints:
    constraints_gauss, constraints_css, constraints_dog, constraints_norm = None,None,None,None
else:
    constraints_gauss, constraints_css, constraints_dog, constraints_norm = [],[],[],[]

    #specific parameter constraints   
    if surround_sigma_larger_than_centre:
    
        #enforcing surround size larger than prf size in DoG model
        if fit_hrf:
            A_ssc_dog = np.array([[0,0,-1,0,0,0,1,0,0]])
        else:
            A_ssc_dog = np.array([[0,0,-1,0,0,0,1]])
    
        constraints_dog.append(LinearConstraint(A_ssc_dog,
                                                    lb=0,
                                                    ub=+inf))
        
        #enforcing surround size larger than prf size in norm
        if fit_hrf:
            A_ssc_norm = np.array([[0,0,-1,0,0,0,1,0,0,0,0]])
        else:
            A_ssc_norm = np.array([[0,0,-1,0,0,0,1,0,0]])
    
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
    data=tc_full_iso_nonzerovar_dict['tc'], model=gg, n_jobs=n_jobs, fit_hrf=fit_hrf)

# gaussian fit
if "grid_data_path" not in analysis_info and "gauss_iterparams_path" not in analysis_info:
    save_path = opj(data_path, subj+"_gridparams-gauss_space-"+fitting_space+str(chunk_nr))

    if not os.path.exists(save_path+".npy") or refit_mode == "overwrite":

        print("Starting Gaussian grid fit at "+datetime.now().strftime('%Y/%m/%d %H:%M:%S'))
        gf.grid_fit(ecc_grid=eccs,
                polar_grid=polars,
                size_grid=sizes,
                verbose=verbose,
                n_batches=n_batches,
                pos_prfs_only=pos_prfs_only)
        print("Gaussian gridfit completed at "+datetime.now().strftime('%Y/%m/%d %H:%M:%S')+
          ". voxels/vertices above "+str(rsq_threshold)+": "+str(np.sum(gf.gridsearch_params[:, -1]>rsq_threshold))+" out of "+
          str(gf.data.shape[0]))
        print("Median rsq>"+str(rsq_threshold)+": "+str(np.median(gf.gridsearch_params[gf.gridsearch_params[:, -1]>rsq_threshold, -1])))

        np.save(save_path, gf.gridsearch_params)

    elif os.path.exists(save_path+".npy") and refit_mode in ["iterate", "skip"]:
        gf.gridsearch_params = np.load(save_path+".npy")

elif "grid_data_path" in analysis_info:
    gf.gridsearch_params = np.array_split(np.load(analysis_info["grid_data_path"]), n_chunks)[chunk_nr]



# gaussian iterative fit
save_path = opj(data_path, subj+"_iterparams-gauss_space-"+fitting_space+str(chunk_nr))
if "gauss_iterparams_path" in analysis_info:
    gf.iterative_search_params = np.array_split(np.load(analysis_info["gauss_iterparams_path"]), n_chunks)[chunk_nr]

    if refit_mode in ["overwrite", "iterate"]:

        print("Starting Gaussian iterfit at "+datetime.now().strftime('%Y/%m/%d %H:%M:%S'))

        gf.iterative_fit(rsq_threshold=rsq_threshold, verbose=verbose,
                         starting_params=gf.iterative_search_params,
                         bounds=gauss_bounds,
                         constraints=constraints_gauss,
                             xtol=xtol,
                             ftol=ftol)

        print("Gaussian iterfit completed at "+datetime.now().strftime('%Y/%m/%d %H:%M:%S')+". Median rsq>"+str(rsq_threshold)+": "+str(np.median(gf.iterative_search_params[gf.rsq_mask, -1])))
      
        if crossvalidate:
            gf.crossvalidate_fit(tc_full_iso_nonzerovar_dict['tc_test'],
                                 test_stimulus=test_prf_stim,
                                 single_hrf=single_hrf)
            print("Gaussian Crossvalidation completed at "+datetime.now().strftime('%Y/%m/%d %H:%M:%S')+". Median rsq>"+str(rsq_threshold)+": "+str(np.median(gf.iterative_search_params[gf.rsq_mask, -1])))
            
            if hasattr(gf, 'noise_ceiling'):
                print("Median noise-ceiling >"+str(rsq_threshold)+": "+str(np.median(gf.noise_ceiling[gf.rsq_mask])))                
                noise_ceiling_fraction = gf.iterative_search_params[gf.rsq_mask, -1]/gf.noise_ceiling[gf.rsq_mask]
                print("Median noise-ceiling-fraction rsq>"+str(rsq_threshold)+": "+str(np.median(noise_ceiling_fraction)))
                
                if return_noise_ceiling_fraction:
                    gf.iterative_search_params[gf.rsq_mask, -1] = noise_ceiling_fraction
        
        np.save(save_path, gf.iterative_search_params)


else:

    if not os.path.exists(save_path+".npy") or refit_mode == "overwrite":

        print("Starting Gaussian iterfit at "+datetime.now().strftime('%Y/%m/%d %H:%M:%S'))

        gf.iterative_fit(rsq_threshold=rsq_threshold, verbose=verbose,
                         bounds=gauss_bounds,
                         constraints=constraints_gauss,
                             xtol=xtol,
                             ftol=ftol)
        
        print("Gaussian iterfit completed at "+datetime.now().strftime('%Y/%m/%d %H:%M:%S')+". Median rsq>"+str(rsq_threshold)+": "+str(np.median(gf.iterative_search_params[gf.rsq_mask, -1])))
      
        if crossvalidate:
            gf.crossvalidate_fit(tc_full_iso_nonzerovar_dict['tc_test'],
                                 test_stimulus=test_prf_stim,
                                 single_hrf=single_hrf)
            print("Gaussian Crossvalidation completed at "+datetime.now().strftime('%Y/%m/%d %H:%M:%S')+". Median rsq>"+str(rsq_threshold)+": "+str(np.median(gf.iterative_search_params[gf.rsq_mask, -1])))

            if hasattr(gf, 'noise_ceiling'):
                print("Median noise-ceiling >"+str(rsq_threshold)+": "+str(np.median(gf.noise_ceiling[gf.rsq_mask])))   
                noise_ceiling_fraction = gf.iterative_search_params[gf.rsq_mask, -1]/gf.noise_ceiling[gf.rsq_mask]
                print("Median noise-ceiling-fraction rsq>"+str(rsq_threshold)+": "+str(np.median(noise_ceiling_fraction)))
                
                if return_noise_ceiling_fraction:
                    gf.iterative_search_params[gf.rsq_mask, -1] = noise_ceiling_fraction            

            
        np.save(save_path, gf.iterative_search_params)
    elif os.path.exists(save_path+".npy") and refit_mode == "iterate":

        if previous_analysis_refit_mode != "iterate" or (datetime.fromtimestamp(os.stat(save_path+".npy").st_mtime)) < datetime(\
                                                        int(previous_analysis_time.split('-')[0]),
                                                        int(previous_analysis_time.split('-')[1]),
                                                        int(previous_analysis_time.split('-')[2]),
                                                        int(previous_analysis_time.split('-')[3]),
                                                        int(previous_analysis_time.split('-')[4]),
                                                        int(previous_analysis_time.split('-')[5]), 0):

            print("Starting Gaussian iterfit at "+datetime.now().strftime('%Y/%m/%d %H:%M:%S'))
    
            gf.iterative_fit(rsq_threshold=rsq_threshold, verbose=verbose,
                             starting_params=np.load(save_path+".npy"),
                             bounds=gauss_bounds,
                             constraints=constraints_gauss,
                             xtol=xtol,
                             ftol=ftol)
            
            print("Gaussian iterfit completed at "+datetime.now().strftime('%Y/%m/%d %H:%M:%S')+". Median rsq>"+str(rsq_threshold)+": "+str(np.median(gf.iterative_search_params[gf.rsq_mask, -1])))
      
            if crossvalidate:
                gf.crossvalidate_fit(tc_full_iso_nonzerovar_dict['tc_test'],
                                 test_stimulus=test_prf_stim,
                                 single_hrf=single_hrf)

                print("Gaussian Crossvalidation completed at "+datetime.now().strftime('%Y/%m/%d %H:%M:%S')+". Median rsq>"+str(rsq_threshold)+": "+str(np.median(gf.iterative_search_params[gf.rsq_mask, -1])))

                if hasattr(gf, 'noise_ceiling'):
                    print("Median noise-ceiling >"+str(rsq_threshold)+": "+str(np.median(gf.noise_ceiling[gf.rsq_mask])))   
                    noise_ceiling_fraction = gf.iterative_search_params[gf.rsq_mask, -1]/gf.noise_ceiling[gf.rsq_mask]
                    print("Median noise-ceiling-fraction rsq>"+str(rsq_threshold)+": "+str(np.median(noise_ceiling_fraction)))
                    
                    if return_noise_ceiling_fraction:
                        gf.iterative_search_params[gf.rsq_mask, -1] = noise_ceiling_fraction
    
            np.save(save_path, gf.iterative_search_params)
        else:
            gf.iterative_search_params = np.load(save_path+".npy")

    elif os.path.exists(save_path+".npy") and refit_mode == "skip":
        gf.iterative_search_params = np.load(save_path+".npy")




# CSS iterative fit
if "CSS" in models_to_fit:
    gg_css = CSS_Iso2DGaussianModel(stimulus=prf_stim,
                                      hrf=hrf,
                                      filter_predictions=filter_predictions,
                                      filter_type=filter_type,
                                      filter_params=filter_params,                                     
                                      normalize_RFs=normalize_RFs)
    gf_css = CSS_Iso2DGaussianFitter(
        data=tc_full_iso_nonzerovar_dict['tc'], model=gg_css, n_jobs=n_jobs, fit_hrf=fit_hrf,
        previous_gaussian_fitter=gf)

    save_path = opj(data_path, subj+"_iterparams-css_space-"+fitting_space+str(chunk_nr))

    if "css_iterparams_path" in analysis_info:
        gf_css.iterative_search_params = np.array_split(np.load(analysis_info["css_iterparams_path"]), n_chunks)[chunk_nr]
        if refit_mode in ["overwrite", "iterate"]:
    
            print("Starting CSS iterfit at "+datetime.now().strftime('%Y/%m/%d %H:%M:%S'))
    
            gf_css.iterative_fit(rsq_threshold=rsq_threshold, verbose=verbose,
                             starting_params=gf_css.iterative_search_params,
                             bounds=css_bounds,
                             constraints=constraints_css,
                             xtol=xtol,
                             ftol=ftol)
            print("CSS iterfit completed at "+datetime.now().strftime('%Y/%m/%d %H:%M:%S')+". Median rsq>"+str(rsq_threshold)+": "+str(np.median(gf_css.iterative_search_params[gf_css.rsq_mask, -1])))
            
            if crossvalidate:
                gf_css.crossvalidate_fit(tc_full_iso_nonzerovar_dict['tc_test'],
                                 test_stimulus=test_prf_stim,
                                 single_hrf=single_hrf)
                print("CSS Crossvalidation completed at "+datetime.now().strftime('%Y/%m/%d %H:%M:%S')+". Median rsq>"+str(rsq_threshold)+": "+str(np.median(gf_css.iterative_search_params[gf_css.rsq_mask, -1])))

                if hasattr(gf_css, 'noise_ceiling'):
                    print("Median noise-ceiling >"+str(rsq_threshold)+": "+str(np.median(gf_css.noise_ceiling[gf_css.rsq_mask])))   
                    noise_ceiling_fraction = gf_css.iterative_search_params[gf_css.rsq_mask, -1]/gf_css.noise_ceiling[gf_css.rsq_mask]
                    print("Median noise-ceiling-fraction rsq>"+str(rsq_threshold)+": "+str(np.median(noise_ceiling_fraction)))
                    
                    if return_noise_ceiling_fraction:
                        gf_css.iterative_search_params[gf_css.rsq_mask, -1] = noise_ceiling_fraction

            np.save(save_path, gf_css.iterative_search_params)
    


    else:


        if not os.path.exists(save_path+".npy") or refit_mode == "overwrite":
            print("Starting CSS iterfit at "+datetime.now().strftime('%Y/%m/%d %H:%M:%S'))
    
            gf_css.iterative_fit(rsq_threshold=rsq_threshold, verbose=verbose,
                                 bounds=css_bounds,
                                 constraints=constraints_css,
                             xtol=xtol,
                             ftol=ftol)
            print("CSS iterfit completed at "+datetime.now().strftime('%Y/%m/%d %H:%M:%S')+". Median rsq>"+str(rsq_threshold)+": "+str(np.median(gf_css.iterative_search_params[gf_css.rsq_mask, -1])))
            
            if crossvalidate:
                gf_css.crossvalidate_fit(tc_full_iso_nonzerovar_dict['tc_test'],
                                 test_stimulus=test_prf_stim,
                                 single_hrf=single_hrf)
                print("CSS Crossvalidation completed at "+datetime.now().strftime('%Y/%m/%d %H:%M:%S')+". Median rsq>"+str(rsq_threshold)+": "+str(np.median(gf_css.iterative_search_params[gf_css.rsq_mask, -1])))

                if hasattr(gf_css, 'noise_ceiling'):
                    print("Median noise-ceiling >"+str(rsq_threshold)+": "+str(np.median(gf_css.noise_ceiling[gf_css.rsq_mask])))  
                    noise_ceiling_fraction = gf_css.iterative_search_params[gf_css.rsq_mask, -1]/gf_css.noise_ceiling[gf_css.rsq_mask]
                    print("Median noise-ceiling-fraction rsq>"+str(rsq_threshold)+": "+str(np.median(noise_ceiling_fraction)))
                    
                    if return_noise_ceiling_fraction:
                        gf_css.iterative_search_params[gf_css.rsq_mask, -1] = noise_ceiling_fraction
    
            np.save(save_path, gf_css.iterative_search_params)
    
        elif os.path.exists(save_path+".npy") and refit_mode == "iterate":
    
            if previous_analysis_refit_mode != "iterate" or (datetime.fromtimestamp(os.stat(save_path+".npy").st_mtime)) < datetime(\
                                                        int(previous_analysis_time.split('-')[0]),
                                                        int(previous_analysis_time.split('-')[1]),
                                                        int(previous_analysis_time.split('-')[2]),
                                                        int(previous_analysis_time.split('-')[3]),
                                                        int(previous_analysis_time.split('-')[4]),
                                                        int(previous_analysis_time.split('-')[5]), 0):
                print("Starting CSS iterfit at "+datetime.now().strftime('%Y/%m/%d %H:%M:%S'))
        
                gf_css.iterative_fit(rsq_threshold=rsq_threshold, verbose=verbose,
                                     starting_params=np.load(save_path+".npy"),
                                     bounds=css_bounds,
                                     constraints=constraints_css,
                             xtol=xtol,
                             ftol=ftol)
                print("CSS iterfit completed at "+datetime.now().strftime('%Y/%m/%d %H:%M:%S')+". Median rsq>"+str(rsq_threshold)+": "+str(np.median(gf_css.iterative_search_params[gf_css.rsq_mask, -1])))
                
                if crossvalidate:
                    gf_css.crossvalidate_fit(tc_full_iso_nonzerovar_dict['tc_test'],
                                 test_stimulus=test_prf_stim,
                                 single_hrf=single_hrf)
                    print("CSS Crossvalidation completed at "+datetime.now().strftime('%Y/%m/%d %H:%M:%S')+". Median rsq>"+str(rsq_threshold)+": "+str(np.median(gf_css.iterative_search_params[gf_css.rsq_mask, -1])))

                    if hasattr(gf_css, 'noise_ceiling'):
                        print("Median noise-ceiling >"+str(rsq_threshold)+": "+str(np.median(gf_css.noise_ceiling[gf_css.rsq_mask])))  
                        noise_ceiling_fraction = gf_css.iterative_search_params[gf_css.rsq_mask, -1]/gf_css.noise_ceiling[gf_css.rsq_mask]
                        print("Median noise-ceiling-fraction rsq>"+str(rsq_threshold)+": "+str(np.median(noise_ceiling_fraction)))
                        
                        if return_noise_ceiling_fraction:
                            gf_css.iterative_search_params[gf_css.rsq_mask, -1] = noise_ceiling_fraction


                np.save(save_path, gf_css.iterative_search_params)
    

    
        elif os.path.exists(save_path+".npy") and refit_mode == "skip":
            gf_css.iterative_search_params = np.load(save_path+".npy")



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
                                     fit_hrf=fit_hrf,
                                     previous_gaussian_fitter=gf)

    save_path = opj(data_path, subj+"_iterparams-dog_space-"+fitting_space+str(chunk_nr))

    if "dog_iterparams_path" in analysis_info:
        gf_dog.iterative_search_params = np.array_split(np.load(analysis_info["dog_iterparams_path"]), n_chunks)[chunk_nr]
        if refit_mode in ["overwrite", "iterate"]:
    
            print("Starting DoG iterfit at "+datetime.now().strftime('%Y/%m/%d %H:%M:%S'))
    
            gf_dog.iterative_fit(rsq_threshold=rsq_threshold, verbose=verbose,
                             starting_params=gf_dog.iterative_search_params,
                             bounds=dog_bounds,
                             constraints=constraints_dog,
                             xtol=xtol,
                             ftol=ftol)
            print("DoG iterfit completed at "+datetime.now().strftime('%Y/%m/%d %H:%M:%S')+". Median rsq>"+str(rsq_threshold)+": "+str(np.median(gf_dog.iterative_search_params[gf_dog.rsq_mask, -1])))
            
            if crossvalidate:
                gf_dog.crossvalidate_fit(tc_full_iso_nonzerovar_dict['tc_test'],
                                 test_stimulus=test_prf_stim,
                                 single_hrf=single_hrf)
                print("DoG Crossvalidation completed at "+datetime.now().strftime('%Y/%m/%d %H:%M:%S')+". Median rsq>"+str(rsq_threshold)+": "+str(np.median(gf_dog.iterative_search_params[gf_dog.rsq_mask, -1])))

                if hasattr(gf_dog, 'noise_ceiling'):
                    print("Median noise-ceiling >"+str(rsq_threshold)+": "+str(np.median(gf_dog.noise_ceiling[gf_dog.rsq_mask])))  
                    noise_ceiling_fraction = gf_dog.iterative_search_params[gf_dog.rsq_mask, -1]/gf_dog.noise_ceiling[gf_dog.rsq_mask]
                    print("Median noise-ceiling-fraction rsq>"+str(rsq_threshold)+": "+str(np.median(noise_ceiling_fraction)))
                    
                    if return_noise_ceiling_fraction:
                        gf_dog.iterative_search_params[gf_dog.rsq_mask, -1] = noise_ceiling_fraction

            np.save(save_path, gf_dog.iterative_search_params)



    else:

        if not os.path.exists(save_path+".npy") or refit_mode == "overwrite":
            print("Starting DoG iterfit at "+datetime.now().strftime('%Y/%m/%d %H:%M:%S'))
    
            gf_dog.iterative_fit(rsq_threshold=rsq_threshold, verbose=verbose,
                                         bounds=dog_bounds,
                                         constraints=constraints_dog,
                                         xtol=xtol,
                                         ftol=ftol)
            print("DoG iterfit completed at "+datetime.now().strftime('%Y/%m/%d %H:%M:%S')+". Median rsq>"+str(rsq_threshold)+": "+str(np.median(gf_dog.iterative_search_params[gf_dog.rsq_mask, -1])))
            
            if crossvalidate:
                gf_dog.crossvalidate_fit(tc_full_iso_nonzerovar_dict['tc_test'],
                                 test_stimulus=test_prf_stim,
                                 single_hrf=single_hrf)
                print("DoG Crossvalidation completed at "+datetime.now().strftime('%Y/%m/%d %H:%M:%S')+". Median rsq>"+str(rsq_threshold)+": "+str(np.median(gf_dog.iterative_search_params[gf_dog.rsq_mask, -1])))
                
                if hasattr(gf_dog, 'noise_ceiling'):
                    print("Median noise-ceiling >"+str(rsq_threshold)+": "+str(np.median(gf_dog.noise_ceiling[gf_dog.rsq_mask])))  
                    noise_ceiling_fraction = gf_dog.iterative_search_params[gf_dog.rsq_mask, -1]/gf_dog.noise_ceiling[gf_dog.rsq_mask]
                    print("Median noise-ceiling-fraction rsq>"+str(rsq_threshold)+": "+str(np.median(noise_ceiling_fraction)))
                    
                    if return_noise_ceiling_fraction:
                        gf_dog.iterative_search_params[gf_dog.rsq_mask, -1] = noise_ceiling_fraction
    
            np.save(save_path, gf_dog.iterative_search_params)
    
    
    
    
        elif os.path.exists(save_path+".npy") and refit_mode == "iterate":
    
            if previous_analysis_refit_mode != "iterate" or (datetime.fromtimestamp(os.stat(save_path+".npy").st_mtime)) < datetime(\
                                                        int(previous_analysis_time.split('-')[0]),
                                                        int(previous_analysis_time.split('-')[1]),
                                                        int(previous_analysis_time.split('-')[2]),
                                                        int(previous_analysis_time.split('-')[3]),
                                                        int(previous_analysis_time.split('-')[4]),
                                                        int(previous_analysis_time.split('-')[5]), 0):
                print("Starting DoG iterfit at "+datetime.now().strftime('%Y/%m/%d %H:%M:%S'))
        
                gf_dog.iterative_fit(rsq_threshold=rsq_threshold, verbose=verbose,
                                     starting_params=np.load(save_path+".npy"),
                                     bounds=dog_bounds,
                                     constraints=constraints_dog,
                                     xtol=xtol,
                                     ftol=ftol)
                print("DoG iterfit completed at "+datetime.now().strftime('%Y/%m/%d %H:%M:%S')+". Median rsq>"+str(rsq_threshold)+": "+str(np.median(gf_dog.iterative_search_params[gf_dog.rsq_mask, -1])))
                
                if crossvalidate:
                    gf_dog.crossvalidate_fit(tc_full_iso_nonzerovar_dict['tc_test'],
                                 test_stimulus=test_prf_stim,
                                 single_hrf=single_hrf)
                    print("DoG Crossvalidation completed at "+datetime.now().strftime('%Y/%m/%d %H:%M:%S')+". Median rsq>"+str(rsq_threshold)+": "+str(np.median(gf_dog.iterative_search_params[gf_dog.rsq_mask, -1])))

                    if hasattr(gf_dog, 'noise_ceiling'):
                        print("Median noise-ceiling >"+str(rsq_threshold)+": "+str(np.median(gf_dog.noise_ceiling[gf_dog.rsq_mask])))  
                        noise_ceiling_fraction = gf_dog.iterative_search_params[gf_dog.rsq_mask, -1]/gf_dog.noise_ceiling[gf_dog.rsq_mask]
                        print("Median noise-ceiling-fraction rsq>"+str(rsq_threshold)+": "+str(np.median(noise_ceiling_fraction)))
                        
                        if return_noise_ceiling_fraction:
                            gf_dog.iterative_search_params[gf_dog.rsq_mask, -1] = noise_ceiling_fraction
                    
                np.save(save_path, gf_dog.iterative_search_params)
    
    

    
        elif os.path.exists(save_path+".npy") and refit_mode == "skip":
            gf_dog.iterative_search_params = np.load(save_path+".npy")





if "norm" in models_to_fit:
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
                                       fit_hrf=fit_hrf,
                                       previous_gaussian_fitter=gf)
    
    #normalization grid stage
    if "norm_gridparams_path" not in analysis_info and "norm_iterparams_path" not in analysis_info:

        save_path = opj(data_path, subj+"_gridparams-norm_space-"+fitting_space+str(chunk_nr))

        if not os.path.exists(save_path+".npy") or refit_mode == "overwrite":


            print("Starting norm grid fit at "+datetime.now().strftime('%Y/%m/%d %H:%M:%S'))
            gf_norm.grid_fit(surround_amplitude_grid,
                         surround_size_grid,
                         neural_baseline_grid,
                         surround_baseline_grid,
                         verbose=verbose,
                         n_batches=n_batches,
                         rsq_threshold=rsq_threshold,
                         pos_prfs_only=pos_prfs_only)
        
            print("Norm gridfit completed at "+datetime.now().strftime('%Y/%m/%d %H:%M:%S')+". Median rsq>"+str(rsq_threshold)+": "+str(np.median(gf_norm.gridsearch_params[gf_norm.gridsearch_rsq_mask, -1])))
        
            np.save(save_path, gf_norm.gridsearch_params)
        elif os.path.exists(save_path+".npy") and refit_mode in ["iterate","skip"]:
            gf_norm.gridsearch_params = np.load(save_path+".npy")

    else:
        gf_norm.gridsearch_params = np.array_split(np.load(analysis_info["norm_gridparams_path"]), n_chunks)[chunk_nr]


    save_path = opj(data_path, subj+"_iterparams-norm_space-"+fitting_space+str(chunk_nr))

    if "norm_iterparams_path" in analysis_info:
        gf_norm.iterative_search_params = np.array_split(np.load(analysis_info["norm_iterparams_path"]), n_chunks)[chunk_nr]
        if refit_mode in ["overwrite", "iterate"]:
    
            print("Starting Norm iterfit at "+datetime.now().strftime('%Y/%m/%d %H:%M:%S'))
    
            gf_norm.iterative_fit(rsq_threshold=rsq_threshold, verbose=verbose,
                             starting_params=gf_norm.iterative_search_params,
                             bounds=norm_bounds,
                             constraints=constraints_norm,
                             xtol=xtol,
                             ftol=ftol)
            print("Norm iterfit completed at "+datetime.now().strftime('%Y/%m/%d %H:%M:%S')+". Median rsq>"+str(rsq_threshold)+": "+str(np.median(gf_norm.iterative_search_params[gf_norm.rsq_mask, -1])))
            
            if crossvalidate:
                gf_norm.crossvalidate_fit(tc_full_iso_nonzerovar_dict['tc_test'],
                                 test_stimulus=test_prf_stim,
                                 single_hrf=single_hrf)
                print("Norm Crossvalidation completed at "+datetime.now().strftime('%Y/%m/%d %H:%M:%S')+". Median rsq>"+str(rsq_threshold)+": "+str(np.median(gf_norm.iterative_search_params[gf_norm.rsq_mask, -1])))

                if hasattr(gf_norm, 'noise_ceiling'):
                    print("Median noise-ceiling >"+str(rsq_threshold)+": "+str(np.median(gf_norm.noise_ceiling[gf_norm.rsq_mask])))  
                    noise_ceiling_fraction = gf_norm.iterative_search_params[gf_norm.rsq_mask, -1]/gf_norm.noise_ceiling[gf_norm.rsq_mask]
                    print("Median noise-ceiling-fraction rsq>"+str(rsq_threshold)+": "+str(np.median(noise_ceiling_fraction)))
                    
                    if return_noise_ceiling_fraction:
                        gf_norm.iterative_search_params[gf_norm.rsq_mask, -1] = noise_ceiling_fraction
            
            np.save(save_path, gf_norm.iterative_search_params)


    else:
        if not os.path.exists(save_path+".npy") or refit_mode == "overwrite":
    
            print("Starting norm iterfit at "+datetime.now().strftime('%Y/%m/%d %H:%M:%S'))
    
            gf_norm.iterative_fit(rsq_threshold=rsq_threshold, verbose=verbose,
                                           bounds=norm_bounds,
                                           constraints=constraints_norm,
                                           xtol=xtol,
                                           ftol=ftol)
            print("Norm iterfit completed at "+datetime.now().strftime('%Y/%m/%d %H:%M:%S')+". Median rsq>"+str(rsq_threshold)+": "+str(np.median(gf_norm.iterative_search_params[gf_norm.rsq_mask, -1])))
            
            if crossvalidate:
                gf_norm.crossvalidate_fit(tc_full_iso_nonzerovar_dict['tc_test'],
                                 test_stimulus=test_prf_stim,
                                 single_hrf=single_hrf)
                print("Norm Crossvalidation completed at "+datetime.now().strftime('%Y/%m/%d %H:%M:%S')+". Median rsq>"+str(rsq_threshold)+": "+str(np.median(gf_norm.iterative_search_params[gf_norm.rsq_mask, -1])))

                if hasattr(gf_norm, 'noise_ceiling'):
                    print("Median noise-ceiling >"+str(rsq_threshold)+": "+str(np.median(gf_norm.noise_ceiling[gf_norm.rsq_mask])))  
                    noise_ceiling_fraction = gf_norm.iterative_search_params[gf_norm.rsq_mask, -1]/gf_norm.noise_ceiling[gf_norm.rsq_mask]
                    print("Median noise-ceiling-fraction rsq>"+str(rsq_threshold)+": "+str(np.median(noise_ceiling_fraction)))
                    
                    if return_noise_ceiling_fraction:
                        gf_norm.iterative_search_params[gf_norm.rsq_mask, -1] = noise_ceiling_fraction
            
            np.save(save_path, gf_norm.iterative_search_params)
    
    
        elif os.path.exists(save_path+".npy") and refit_mode == "iterate":
    
            if previous_analysis_refit_mode != "iterate" or (datetime.fromtimestamp(os.stat(save_path+".npy").st_mtime)) < datetime(\
                                                        int(previous_analysis_time.split('-')[0]),
                                                        int(previous_analysis_time.split('-')[1]),
                                                        int(previous_analysis_time.split('-')[2]),
                                                        int(previous_analysis_time.split('-')[3]),
                                                        int(previous_analysis_time.split('-')[4]),
                                                        int(previous_analysis_time.split('-')[5]), 0):
    
                print("Starting norm iterfit at "+datetime.now().strftime('%Y/%m/%d %H:%M:%S'))
        
                gf_norm.iterative_fit(rsq_threshold=rsq_threshold, verbose=verbose,
                                      starting_params=np.load(save_path+".npy"),
                                              bounds=norm_bounds,
                                              constraints=constraints_norm,
                                              xtol=xtol,
                                              ftol=ftol)
                print("Norm iterfit completed at "+datetime.now().strftime('%Y/%m/%d %H:%M:%S')+". Median rsq>"+str(rsq_threshold)+": "+str(np.median(gf_norm.iterative_search_params[gf_norm.rsq_mask, -1])))
            
                if crossvalidate:
                    gf_norm.crossvalidate_fit(tc_full_iso_nonzerovar_dict['tc_test'],
                                 test_stimulus=test_prf_stim,
                                 single_hrf=single_hrf)
                    print("Norm Crossvalidation completed at "+datetime.now().strftime('%Y/%m/%d %H:%M:%S')+". Median rsq>"+str(rsq_threshold)+": "+str(np.median(gf_norm.iterative_search_params[gf_norm.rsq_mask, -1])))
            
                    if hasattr(gf_norm, 'noise_ceiling'):
                        print("Median noise-ceiling >"+str(rsq_threshold)+": "+str(np.median(gf_norm.noise_ceiling[gf_norm.rsq_mask])))  
                        noise_ceiling_fraction = gf_norm.iterative_search_params[gf_norm.rsq_mask, -1]/gf_norm.noise_ceiling[gf_norm.rsq_mask]
                        print("Median noise-ceiling-fraction rsq>"+str(rsq_threshold)+": "+str(np.median(noise_ceiling_fraction)))
                        
                        if return_noise_ceiling_fraction:
                            gf_norm.iterative_search_params[gf_norm.rsq_mask, -1] = noise_ceiling_fraction    
                            
                            
                np.save(save_path, gf_norm.iterative_search_params)
    

    
        elif os.path.exists(save_path+".npy") and refit_mode == "skip":
            gf_norm.iterative_search_params = np.load(save_path+".npy")
