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
from shutil import copyfile as cp

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
save_raw_timecourse = analysis_info["save_raw_timecourse"]

window_length = analysis_info["window_length"]
n_jobs = analysis_info["n_jobs"]
hrf = analysis_info["hrf"]
verbose = analysis_info["verbose"]
rsq_threshold = analysis_info["rsq_threshold"]
models_to_fit = analysis_info["models_to_fit"]
n_batches = analysis_info["n_batches"]
fit_hrf = analysis_info["fit_hrf"]
crossvalidate = analysis_info["crossvalidate"]
    
save_noise_ceiling = analysis_info["save_noise_ceiling"]

dm_edges_clipping = analysis_info["dm_edges_clipping"]
baseline_volumes_begin_end = analysis_info["baseline_volumes_begin_end"]
min_percent_var = analysis_info["min_percent_var"]

n_chunks = analysis_info["n_chunks"]
refit_mode = analysis_info["refit_mode"].lower()

data_path = opj(data_path,'prfpy')

analysis_time = analysis_info["analysis_time"]

previous_analysis_time = analysis_info["previous_analysis_time"]
previous_analysis_refit_mode = analysis_info["previous_analysis_refit_mode"]

if refit_mode == previous_analysis_refit_mode and refit_mode!="overwrite":
    analysis_time = previous_analysis_time

#first check if iteration was completed
unfinished_chunks=[]
if refit_mode in ["skip", "overwrite"]:
    exists = np.array([os.path.exists(opj(data_path, subj+"_iterparams-norm_space-"\
                        +fitting_space+str(chunk_nr)+".npy")) for chunk_nr in range(n_chunks)])
    finished_chunks = np.where(exists)[0]
    if np.any(exists==False):
        for value in np.where(exists==False)[0]:
            unfinished_chunks.append(value)


else:
    finished_chunks = np.arange(n_chunks)

if refit_mode in ["iterate", "overwrite"]:
    for model in models_to_fit:
        model=model.lower()
        last_edited = np.array([(datetime.fromtimestamp(os.stat(opj(data_path,
                                                        subj+"_iterparams-"+model+"_space-"+fitting_space+str(chunk_nr)+".npy")).st_mtime)) < datetime(\
                                                        int(analysis_time.split('-')[0]),
                                                        int(analysis_time.split('-')[1]),
                                                        int(analysis_time.split('-')[2]),
                                                        int(analysis_time.split('-')[3]),
                                                        int(analysis_time.split('-')[4]),
                                                        int(analysis_time.split('-')[5]), 0) for chunk_nr in finished_chunks])


        if np.any(last_edited):

            print("Fitting of "+model+" model has not been completed")
            for value in np.where(last_edited)[0]:
                unfinished_chunks.append(value)

                filepath = opj(data_path,subj+"_iterparams-"+model+"_space-"+fitting_space+str(value)+".npy")
                if refit_mode == "overwrite":
                    print("Renaming unfinished files to _old, so can run next in skip mode.")
                    os.rename(filepath,filepath[:-4]+"_old.npy")



if len(unfinished_chunks)>0:
    print("Make sure to be in skip or iterate mode first! Then run:")
    str_resub='"( '
    for value in np.unique(unfinished_chunks):
        str_resub+=(str(value)+' ')
    str_resub+=')"'
    print("python array_submit_prf_fit_only.py "+subj+" analysis_settings_cartesius.yml "+str_resub)
    

    sys.exit("harvest not completed. resubmit chunks.")

order = np.load(opj(data_path, subj+"_order_space-"+fitting_space+".npy"))

for model in models_to_fit:
    model = model.lower()
    if model in ["gauss", "norm"] and refit_mode != "iterate":
        grid_path = opj(data_path, subj+"_gridparams-"+model+"_space-"+fitting_space)
        grid_result = np.concatenate(tuple([np.load(grid_path+str(chunk_nr)+".npy") for chunk_nr in range(n_chunks)]), axis=0)

        grid_path+=analysis_time

        grid_result[order] = np.copy(grid_result)

        np.save(grid_path.replace('scratch-shared', 'home'), grid_result)

    iter_path = opj(data_path, subj+"_iterparams-"+model+"_space-"+fitting_space)

    model_result = np.concatenate(tuple([np.load(iter_path+str(chunk_nr)+".npy") for chunk_nr in range(n_chunks)]), axis=0)

    iter_path+=analysis_time

    model_result[order] = np.copy(model_result)

    np.save(iter_path.replace('scratch-shared', 'home'), model_result)

#mask is saved in original order so only need to copypaste
mask_path=opj(data_path,  f"{subj}_mask_space-{fitting_space}.npy")
cp(mask_path, mask_path.replace('scratch-shared', 'home').replace('.npy',analysis_time+'.npy'))

#save noise ceiling
if save_noise_ceiling:
    nc_path = opj(data_path,f"{subj}_noise-ceiling_space-{fitting_space}.npy")
    cp(nc_path, nc_path.replace('scratch-shared', 'home').replace('.npy',analysis_time+'.npy'))


if len(sys.argv)>3:
    print("Grabbing timecourse for this analysis...")

    #these are saved in randomized order, so must take into account. like for models above
    tc_path=opj(data_path,  subj+"_timecourse_space-"+fitting_space+".npy")
    tc = np.load(tc_path)
    tc_ordered = np.zeros_like(tc)
    tc_ordered[order] = np.copy(tc)
    np.save(tc_path.replace('scratch-shared', 'home').replace('.npy',analysis_time+'.npy'), tc_ordered)
    
    if save_raw_timecourse:
        tc_raw_path=opj(data_path,  subj+"_timecourse-raw_space-"+fitting_space+".npy")
        cp(tc_raw_path, tc_raw_path.replace('scratch-shared', 'home').replace('.npy',analysis_time+'.npy'))
    
        
    if crossvalidate:
        tc_test_path=opj(data_path,  subj+"_timecourse-test_space-"+fitting_space+".npy")
        tc_test = np.load(tc_test_path)
        tc_test_ordered = np.zeros_like(tc_test)
        tc_test_ordered[order] = np.copy(tc_test)
        np.save(tc_test_path.replace('scratch-shared', 'home').replace('.npy',analysis_time+'.npy'), tc_test_ordered)
        if save_raw_timecourse:
            tc_test_raw_path=opj(data_path,  subj+"_timecourse-test-raw_space-"+fitting_space+".npy")
            cp(tc_test_raw_path, tc_test_raw_path.replace('scratch-shared', 'home').replace('.npy',analysis_time+'.npy'))

        
cp(analysis_settings, analysis_settings.replace('scratch-shared', 'home').replace('.yml',analysis_time+'.yml'))
print("harvest completed")

