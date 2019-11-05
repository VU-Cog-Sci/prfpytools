#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov  2 11:51:11 2019

@author: marcoaqil
"""


import os
import numpy as np
opj = os.path.join
import yaml
import sys

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
models_to_fit = analysis_info["models_to_fit"]
n_batches = analysis_info["n_batches"]
fit_hrf = analysis_info["fit_hrf"]

dm_edges_clipping = analysis_info["dm_edges_clipping"]
baseline_volumes_begin_end = analysis_info["baseline_volumes_begin_end"]
min_percent_var = analysis_info["min_percent_var"]

n_chunks = analysis_info["n_chunks"]
refit_mode = analysis_info["refit_mode"]

data_path = opj(data_path,'prfpy')

for model in models_to_fit:
    model = model.lower()
    if model in ["gauss", "norm"]:
        grid_path = opj(data_path, subj+"_gridparams-"+model+"_space-"+fitting_space)
        grid_result = np.concatenate(tuple([np.load(grid_path+str(chunk_nr)+".npy") for chunk_nr in range(n_chunks)]), axis=0)
        np.save(grid_path, grid_result)

    iter_path = opj(data_path, subj+"_iterparams-"+model+"_space-"+fitting_space)
    model_result = np.concatenate(tuple([np.load(iter_path+str(chunk_nr)+".npy") for chunk_nr in range(n_chunks)]), axis=0)
    np.save(iter_path, model_result)



