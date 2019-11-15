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
from datetime import datetime

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
refit_mode = analysis_info["refit_mode"].lower()

data_path = opj(data_path,'prfpy')

analysis_time = analysis_info["analysis_time"]

#first check if iteration was completed
if refit_mode == "iterate":
    for model in models_to_fit:

        if np.any(np.array([(datetime.fromtimestamp(os.stat(opj(data_path,
            subj+"_iterparams-"+model+"_space-"+fitting_space+str(chunk_nr)+".npy")).st_mtime)) < datetime(\
                                                        int(analysis_time.split('-')[0]),
                                                        int(analysis_time.split('-')[1]),
                                                        int(analysis_time.split('-')[2]),
                                                        int(analysis_time.split('-')[3]),
                                                        int(analysis_time.split('-')[4]),
                                                        int(analysis_time.split('-')[5]), 0) for chunk_nr in range(n_chunks)])):

            print("iterate mode refitting was not completed")
            raise IOError



for model in models_to_fit:
    model = model.lower()
    if model in ["gauss", "norm"] and refit_mode != "iterate":
        grid_path = opj(data_path, subj+"_gridparams-"+model+"_space-"+fitting_space)
        grid_result = np.concatenate(tuple([np.load(grid_path+str(chunk_nr)+".npy") for chunk_nr in range(n_chunks)]), axis=0)

        grid_path+=analysis_time

        np.save(grid_path.replace('scratch-shared', 'home'), grid_result)

    iter_path = opj(data_path, subj+"_iterparams-"+model+"_space-"+fitting_space)


    model_result = np.concatenate(tuple([np.load(iter_path+str(chunk_nr)+".npy") for chunk_nr in range(n_chunks)]), axis=0)

    iter_path+=analysis_time

    np.save(iter_path.replace('scratch-shared', 'home'), model_result)

print("harvest completed")

