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

subj = sys.argv[1]
analysis_settings = sys.argv[2]

with open(analysis_settings) as f:
    analysis_info = yaml.safe_load(f)

if "mkl_num_threads" in analysis_info:
    import mkl
    standard_max_threads = mkl.get_max_threads()
    print(standard_max_threads)
    mkl.set_num_threads(analysis_info["mkl_num_threads"])

import numpy as np

from utils.utils import create_dm_from_screenshots, prepare_surface_data, prepare_volume_data
from prfpy.stimulus import PRFStimulus2D
from prfpy.grid import Iso2DGaussianGridder, Norm_Iso2DGaussianGridder, DoG_Iso2DGaussianGridder
from prfpy.fit import Iso2DGaussianFitter, Norm_Iso2DGaussianFitter, DoG_Iso2DGaussianFitter


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
window_length = analysis_info["window_length"]
n_jobs = analysis_info["n_jobs"]
hrf = analysis_info["hrf"]
gradient_method = analysis_info["gradient_method"]
verbose = analysis_info["verbose"]
rsq_threshold = analysis_info["rsq_threshold"]
models_to_fit = analysis_info["models_to_fit"]



if verbose == True:
    print("Creating PRF stimulus from screenshots...")

dm_list = []

for screenshot_path in screenshot_paths:
    # create stimulus
    dm_list.append(create_dm_from_screenshots(screenshot_path,
                                              n_pix)[..., discard_volumes:])


task_lengths = [dm.shape[-1] for dm in dm_list]

dm_full = np.concatenate(tuple(dm_list), axis=-1)

prf_stim = PRFStimulus2D(screen_size_cm=screen_size_cm,
                         screen_distance_cm=screen_distance_cm,
                         design_matrix=dm_full,
                         TR=TR)


# late-empty DM periods (for calculation of BOLD baseline)
shifted_dm = np.zeros_like(dm_full)

# number of TRs in which activity may linger (hrf)
shifted_dm[..., 7:] = dm_full[..., :-7]

late_iso_dict = {}
late_iso_dict['periods'] = np.where((np.sum(dm_full, axis=(0, 1)) == 0) & (
    np.sum(shifted_dm, axis=(0, 1)) == 0))[0]

for i, task_name in enumerate(task_names):
    if task_name not in screenshot_paths[i]:
        print("WARNING: check that screenshot paths and task names are in the same order")
    late_iso_dict[task_name] = np.split(
        late_iso_dict['periods'], len(task_names))[i]

if verbose == True:
    print("Finished stimulus setup. Now preparing data for fitting...")

if "timecourse_data_path" not in analysis_info:
    if fitting_space == "fsaverage" or fitting_space == "fsnative":
    
        tc_full_iso_nonzerovar_dict = prepare_surface_data(subj,
                                                           task_names,
                                                           discard_volumes,
                                                           window_length,
                                                           late_iso_dict,
                                                           data_path,
                                                           fitting_space)
    
    else:
    
        tc_full_iso_nonzerovar_dict = prepare_volume_data(subj,
                                                          task_names,
                                                          discard_volumes,
                                                          window_length,
                                                          late_iso_dict,
                                                          data_path,
                                                          fitting_space)
    
    save_path = opj(data_path, subj+"_timecourse_space-"+fitting_space)
    
    if os.path.exists(save_path+".npy"):
        save_path+=datetime.now().strftime('%Y%m%d%H%M%S')
    
    np.save(save_path, tc_full_iso_nonzerovar_dict['tc'])
else:
    #mainly for testing purposes
    tc_full_iso_nonzerovar_dict = {}
    tc_full_iso_nonzerovar_dict['tc'] = np.load(analysis_info["timecourse_data_path"])
    


if verbose == True:
    print("Finished preparing data for fitting. Now creating and fitting models...")

# grid params
grid_nr = 20
max_ecc_size = prf_stim.max_ecc
sizes, eccs, polars = max_ecc_size * np.linspace(0.25, 1, grid_nr)**2, \
    max_ecc_size * np.linspace(0.1, 1, grid_nr)**2, \
    np.linspace(0, 2*np.pi, grid_nr)

# to avoid dividing by zero
inf = np.inf
eps = 1e-1
ss = prf_stim.screen_size_degrees

# MODEL COMPARISON
print("Started modeling at: "+datetime.now().strftime('%Y/%m/%d %H:%M:%S'))

# Gaussian model
gg = Iso2DGaussianGridder(stimulus=prf_stim,
                          hrf=hrf,
                          filter_predictions=True,
                          window_length=window_length,
                          task_lengths=task_lengths)


gf = Iso2DGaussianFitter(
    data=tc_full_iso_nonzerovar_dict['tc'], gridder=gg, n_jobs=n_jobs,
    bounds=[(-10*ss, 10*ss),  # x
            (-10*ss, 10*ss),  # y
            (eps, 10*ss),  # prf size
            (-inf, +inf),  # prf amplitude
            (0, +inf)],  # bold baseline
    gradient_method=gradient_method)

# gaussian fit
if "grid_data_path" not in analysis_info and "gauss_iterparams_path" not in analysis_info:
    gf.grid_fit(ecc_grid=eccs,
                polar_grid=polars,
                size_grid=sizes)
    print("Gaussian gridfit completed at "+datetime.now().strftime('%Y/%m/%d %H:%M:%S')+". rsq: "+str(gf.gridsearch_params[:, -1].mean()))

    save_path = opj(data_path, subj+"_gauss-gridparams_space-"+fitting_space)
    if os.path.exists(save_path+".npy"):
        save_path+=datetime.now().strftime('%Y%m%d%H%M%S')
    
    np.save(save_path, gf.gridsearch_params)

elif "grid_data_path" in analysis_info:
    gf.gridsearch_params = np.load(analysis_info["grid_data_path"])

if "mkl_num_threads" in analysis_info:
    mkl.set_num_threads(standard_max_threads)

# gaussian iterative fit
if "gauss_iterparams_path" in analysis_info and "gauss" not in models_to_fit:
    gf.iterative_search_params = np.load(analysis_info["gauss_iterparams_path"])
else:
    gf.iterative_fit(rsq_threshold=rsq_threshold, verbose=verbose)

    print("Gaussian iterfit completed at "+datetime.now().strftime('%Y/%m/%d %H:%M:%S')+". rsq: "+str(gf.iterative_search_params[gf.rsq_mask, -1].mean()))

    save_path = opj(data_path, subj+"_gauss-iterparams_space-"+fitting_space)
    if os.path.exists(save_path+".npy"):
        save_path+=datetime.now().strftime('%Y%m%d%H%M%S')
    np.save(save_path, gf.iterative_search_params)



starting_params = np.insert(gf.iterative_search_params, -1, 1.0, axis=-1)


# CSS iterative fit
if "CSS" in models_to_fit:
    gf_css = Iso2DGaussianFitter(
        data=tc_full_iso_nonzerovar_dict['tc'], gridder=gg, n_jobs=n_jobs, fit_css=True,
        bounds=[(-10*ss, 10*ss),  # x
                (-10*ss, 10*ss),  # y
                (eps, 10*ss),  # prf size
                (-inf, +inf),  # prf amplitude
                (0, +inf),  # bold baseline
                (0.001, 3)],  # CSS exponent
        gradient_method=gradient_method)

    gf_css.iterative_fit(rsq_threshold=0.4,
                         gridsearch_params=starting_params, verbose=verbose)

    save_path = opj(data_path, subj+"_CSS-iterparams_space-"+fitting_space)
    if os.path.exists(save_path+".npy"):
        save_path+=datetime.now().strftime('%Y%m%d%H%M%S')
    np.save(save_path, gf_css.iterative_search_params)

    print("CSS iterfit completed at "+datetime.now().strftime('%Y/%m/%d %H:%M:%S')+". rsq: "+str(gf_css.iterative_search_params[gf_css.rsq_mask, -1].mean()))

if "DoG" in models_to_fit:    
    # difference of gaussians iterative fit
    gg_dog = DoG_Iso2DGaussianGridder(stimulus=prf_stim,
                                      hrf=hrf,
                                      filter_predictions=True,
                                      window_length=window_length,
                                      task_lengths=task_lengths)

    gf_dog = DoG_Iso2DGaussianFitter(data=tc_full_iso_nonzerovar_dict['tc'],
                                     gridder=gg_dog,
                                     n_jobs=n_jobs,
                                     bounds=[(-10*ss, 10*ss),  # x
                                             (-10*ss, 10*ss),  # y
                                             (eps, 10*ss),  # prf size
                                             (-inf, +inf),  # prf amplitude
                                             (0, +inf),  # bold baseline
                                             (-inf, +inf),  # surround amplitude
                                             (eps, 20*ss)],  # surround size
                                     gradient_method=gradient_method)

    gf_dog.iterative_fit(rsq_threshold=0.4,
                         gridsearch_params=starting_params, verbose=verbose)

    save_path = opj(data_path, subj+"_DoG-iterparams_space-"+fitting_space)
    if os.path.exists(save_path+".npy"):
        save_path+=datetime.now().strftime('%Y%m%d%H%M%S')

    np.save(save_path, gf_dog.iterative_search_params)


    print("DoG iterfit completed at "+datetime.now().strftime('%Y/%m/%d %H:%M:%S')+". rsq: "+str(gf_dog.iterative_search_params[gf_dog.rsq_mask, -1].mean()))

if "norm" in models_to_fit:
    # normalization iterative fit
    gg_norm = Norm_Iso2DGaussianGridder(stimulus=prf_stim,
                                        hrf=hrf,
                                        filter_predictions=True,
                                        window_length=window_length,
                                        task_lengths=task_lengths)

    gf_norm = Norm_Iso2DGaussianFitter(data=tc_full_iso_nonzerovar_dict['tc'],
                                       gridder=gg_norm,
                                       n_jobs=n_jobs,
                                       bounds=[(-10*ss, 10*ss),  # x
                                               (-10*ss, 10*ss),  # y
                                               (eps, 10*ss),  # prf size
                                               (-inf, +inf),  # prf amplitude
                                               (0, +inf),  # bold baseline
                                               (-inf, +inf),  # neural baseline
                                               (-inf, +inf),  # surround amplitude
                                               (eps, 20*ss),  # surround size
                                               (-inf, +inf)],  # surround baseline
                                       gradient_method=gradient_method)

    gf_norm.iterative_fit(rsq_threshold=0.4,
                          gridsearch_params=starting_params, verbose=verbose)

    save_path = opj(data_path, subj+"_norm-iterparams_space-"+fitting_space)
    if os.path.exists(save_path+".npy"):
        save_path+=datetime.now().strftime('%Y%m%d%H%M%S')

    np.save(save_path, gf_norm.iterative_search_params)

    print("Norm iterfit completed at "+datetime.now().strftime('%Y/%m/%d %H:%M:%S')+". rsq: "+str(gf_norm.iterative_search_params[gf_norm.rsq_mask, -1].mean()))


