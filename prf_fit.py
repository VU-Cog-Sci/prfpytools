#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug  6 10:51:41 2019

@author: marcoaqil
"""

from prfpy.fit import Iso2DGaussianFitter, Norm_Iso2DGaussianFitter, DoG_Iso2DGaussianFitter
from prfpy.grid import Iso2DGaussianGridder, Norm_Iso2DGaussianGridder, DoG_Iso2DGaussianGridder
from prfpy.stimulus import PRFStimulus2D
from utils.utils import create_dm_from_screenshots, prepare_surface_data, prepare_volume_data
import sys
import yaml
import os
opj = os.path.join


import numpy as np




subj = sys.argv[1]
analysis_settings = sys.argv[2]

with open(analysis_settings) as f:
    analysis_info = yaml.safe_load(f)

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
iso_periods = np.where(np.sum(dm_full, axis=(0, 1)) == 0)[0]
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

if verbose==True:
    print("Finished stimulus setup. Now preparing data for fitting...")

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


if verbose==True:
    print("Finished preparing data for fitting. Now creating and fitting models...")
    
# grid params
grid_nr = 20
max_ecc_size = 16
sizes, eccs, polars = max_ecc_size * np.linspace(0.25, 1, grid_nr)**2, \
    max_ecc_size * np.linspace(0.1, 1, grid_nr)**2, \
    np.linspace(0, 2*np.pi, grid_nr)

# to avoid dividing by zero
inf = np.inf
eps = 1e-6

####MODEL COMPARISON

#Gaussian model
gg = Iso2DGaussianGridder(stimulus=prf_stim,
                          hrf=hrf,
                          filter_predictions=True,
                          window_length=window_length,
                          task_lengths=task_lengths)


gf = Iso2DGaussianFitter(
    data=tc_full_iso_nonzerovar_dict['tc'], gridder=gg, n_jobs=n_jobs,
    bounds=[(-10*n_pix, 10*n_pix),  # x
            (-10*n_pix, 10*n_pix),  # y
            (eps, 10*n_pix),  # prf size
            (-inf, +inf),  # prf amplitude
            (0, +inf)],  # bold baseline
    gradient_method=gradient_method)

#gaussian grid fit
gf.grid_fit(ecc_grid=eccs,
            polar_grid=polars,
            size_grid=sizes)

#gaussian iterative fit
gf.iterative_fit(rsq_threshold=rsq_threshold, verbose=verbose) 

#CSS model iterative fit
gf_css = Iso2DGaussianFitter(
    data=tc_full_iso_nonzerovar_dict['tc'], gridder=gg, n_jobs=n_jobs, fit_css=True,
    bounds=[(-10*n_pix, 10*n_pix),  # x
            (-10*n_pix, 10*n_pix),  # y
            (eps, 10*n_pix),  # prf size
            (-inf, +inf),  # prf amplitude
            (0, +inf),  # bold baseline
            (eps, 3)],  # CSS exponent
    gradient_method=gradient_method)

gf_css.iterative_fit(rsq_threshold=rsq_threshold, verbose=verbose) 

# normalization model iterative fit
gg_norm = Norm_Iso2DGaussianGridder(stimulus=prf_stim,
                                    hrf=hrf,
                                    filter_predictions=True,
                                    window_length=window_length,
                                    task_lengths=task_lengths)

gf_norm = Norm_Iso2DGaussianFitter(data=tc_full_iso_nonzerovar_dict['tc'],
                                   gridder=gg_norm,
                                   n_jobs=n_jobs,
                                   bounds=[(-10*n_pix, 10*n_pix),  # x
                                           (-10*n_pix, 10*n_pix),  # y
                                           (eps, 10*n_pix),  # prf size
                                           (-inf, +inf),  # prf amplitude
                                           (0, +inf),  # bold baseline
                                           (0, +inf),  # neural baseline
                                           (0, +inf),  # surround amplitude
                                           (eps, 10*n_pix),  # surround size
                                           (eps, +inf)],  # surround baseline
                                   gradient_method=gradient_method)

gf_norm.iterative_fit(rsq_threshold=rsq_threshold,
                      gridsearch_params=gf.gridsearch_params, verbose=verbose)

# difference of gaussians iterative fit
gg_dog = DoG_Iso2DGaussianGridder(stimulus=prf_stim,
                                  hrf=hrf,
                                  filter_predictions=True,
                                  window_length=window_length,
                                  task_lengths=task_lengths)

gf_dog = DoG_Iso2DGaussianFitter(data=tc_full_iso_nonzerovar_dict['tc'],
                                 gridder=gg_dog,
                                 n_jobs=n_jobs,
                                 bounds=[(-10*n_pix, 10*n_pix),  # x
                                         (-10*n_pix, 10*n_pix),  # y
                                         (eps, 10*n_pix),  # prf size
                                         (0, +inf),  # prf amplitude
                                         (0, +inf),  # bold baseline
                                         (0, +inf),  # surround amplitude
                                         (eps, 10*n_pix)],  # surround size
                                 gradient_method=gradient_method)

gf_dog.iterative_fit(rsq_threshold=rsq_threshold,
                     gridsearch_params=gf.gridsearch_params, verbose=verbose)

print(gf.iterative_search_params[gf.rsq_mask,-1].mean())
print(gf.iterative_search_params[gf.rsq_mask,-1].mean())
print(gf_dog.iterative_search_params[gf.rsq_mask,-1].mean())
print(gf_norm.iterative_search_params[gf.rsq_mask,-1].mean())
