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
chunk_nr = int(sys.argv[3])

with open(analysis_settings) as f:
    analysis_info = yaml.safe_load(f)

if "mkl_num_threads" in analysis_info:
    import mkl
    standard_max_threads = mkl.get_max_threads()
    print(standard_max_threads)
    mkl.set_num_threads(analysis_info["mkl_num_threads"])

import numpy as np

from utils.utils import create_full_stim, prepare_data

from prfpy.grid import Iso2DGaussianGridder, Norm_Iso2DGaussianGridder, DoG_Iso2DGaussianGridder, CSS_Iso2DGaussianGridder
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
prepare_data = analysis_info["prepare_data"]

analysis_time = datetime.now().strftime('%Y%m%d%H%M%S')

data_path = opj(data_path,'prfpy')

save_path = opj(data_path, subj+"_analysis_settings")

#if os.path.exists(save_path+".yml"):
#    save_path+=analysis_time

with open(save_path+".yml", 'w+') as outfile:
    yaml.dump(analysis_info, outfile)

if verbose == True:
    print("Creating PRF stimulus from screenshots...")

#creating stimulus from screenshots
task_lengths, prf_stim, late_iso_dict = create_full_stim(screenshot_paths,
                n_pix,
                discard_volumes,
                baseline_volumes_begin_end,
                dm_edges_clipping,
                screen_size_cm,
                screen_distance_cm,
                TR,
                task_names)


if "prepare_data" == True:
    tc_full_iso_nonzerovar_dict = prepare_data(subj,
                                               task_names,
                                               discard_volumes,
                                               min_percent_var,
                                               window_length,
                                               late_iso_dict,
                                               data_path[:-5],
                                               fitting_space)

    save_path = opj(data_path, subj+"_timecourse_space-"+fitting_space)

    np.save(save_path, tc_full_iso_nonzerovar_dict['tc'])

    save_path = opj(data_path, subj+"_nonlow_var_mask_space-"+fitting_space)

    np.save(save_path, tc_full_iso_nonzerovar_dict['nonlow_var_mask'])

elif "timecourse_data_path" in analysis_info:

    tc_full_iso_nonzerovar_dict = {}
    tc_full_iso_nonzerovar_dict['tc'] = np.load(analysis_info["timecourse_data_path"])

elif os.path.exists(opj(data_path, subj+"_timecourse_space-"+fitting_space+".npy")):
    print("Using time series from: "+opj(data_path, subj+"_timecourse_space-"+fitting_space+".npy"))
    tc_full_iso_nonzerovar_dict = {}
    tc_full_iso_nonzerovar_dict['tc'] = np.load(opj(data_path, subj+"_timecourse_space-"+fitting_space+".npy"))

else:
    print("prepare_data is False but a timecourse file was not provided or found. Exiting.")
    raise IOError


tc_full_iso_nonzerovar_dict['tc'] = np.array_split(tc_full_iso_nonzerovar_dict['tc'], n_chunks)[chunk_nr]

if "mkl_num_threads" in analysis_info:
    mkl.set_num_threads(1)


if verbose == True:
    print("Finished preparing data for fitting. Now creating and fitting models...")

# grid params
grid_nr = 20
max_ecc_size = prf_stim.max_ecc
sizes, eccs, polars = max_ecc_size * np.linspace(0.25, 1, grid_nr)**2, \
    max_ecc_size * np.linspace(0.1, 1, grid_nr)**2, \
    np.linspace(0, 2*np.pi, grid_nr)

# to set up parameter bounds in iterfit
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
    data=tc_full_iso_nonzerovar_dict['tc'], gridder=gg, n_jobs=n_jobs)
gf.fit_hrf = fit_hrf

# gaussian fit
if "grid_data_path" not in analysis_info and "gauss_iterparams_path" not in analysis_info:
    save_path = opj(data_path, subj+"_gridparams-gauss_space-"+fitting_space+str(chunk_nr))

    if not os.path.exists(save_path+".npy") or refit_mode == "overwrite":

        print("Starting Gaussian grid fit at "+datetime.now().strftime('%Y/%m/%d %H:%M:%S'))
        gf.grid_fit(ecc_grid=eccs,
                polar_grid=polars,
                size_grid=sizes,
                verbose=verbose,
                n_batches=n_batches)
        print("Gaussian gridfit completed at "+datetime.now().strftime('%Y/%m/%d %H:%M:%S')+\
          ". voxels/vertices above "+str(rsq_threshold)+": "+str(np.sum(gf.gridsearch_params[:, -1]>rsq_threshold)))
        print("Mean rsq>"+str(rsq_threshold)+": "+str(gf.gridsearch_params[gf.gridsearch_params[:, -1]>rsq_threshold, -1].mean()))


       
        np.save(save_path, gf.gridsearch_params)

elif "grid_data_path" in analysis_info:
    gf.gridsearch_params = np.array_split(np.load(analysis_info["grid_data_path"]), n_chunks)[chunk_nr]



# gaussian iterative fit
if "gauss_iterparams_path" in analysis_info:
    gf.iterative_search_params = np.array_split(np.load(analysis_info["gauss_iterparams_path"]), n_chunks)[chunk_nr]
else:

    save_path = opj(data_path, subj+"_iterparams-gauss_space-"+fitting_space+str(chunk_nr))

    if not os.path.exists(save_path+".npy") or refit_mode == "overwrite":

        print("Starting Gaussian iter fit at "+datetime.now().strftime('%Y/%m/%d %H:%M:%S'))

        gf.iterative_fit(rsq_threshold=rsq_threshold, verbose=verbose,
                         bounds=[(-10*ss, 10*ss),  # x
                                 (-10*ss, 10*ss),  # y
                                 (eps, 10*ss),  # prf size
                                 (-inf, +inf),  # prf amplitude
                                 (0, +inf)],  # bold baseline
                         gradient_method=gradient_method,
                         fit_hrf=fit_hrf)

        print("Gaussian iterfit completed at "+datetime.now().strftime('%Y/%m/%d %H:%M:%S')+". Mean rsq>"+str(rsq_threshold)+": "+str(gf.iterative_search_params[gf.rsq_mask, -1].mean()))

            
        np.save(save_path, gf.iterative_search_params)
    elif os.path.exists(save_path+".npy") and refit_mode == "iterate":
        print("Starting Gaussian iter fit at "+datetime.now().strftime('%Y/%m/%d %H:%M:%S'))

        gf.iterative_fit(rsq_threshold=rsq_threshold, verbose=verbose,
                         starting_params=np.load(save_path+".npy"),
                         bounds=[(-10*ss, 10*ss),  # x
                                 (-10*ss, 10*ss),  # y
                                 (eps, 10*ss),  # prf size
                                 (-inf, +inf),  # prf amplitude
                                 (0, +inf)],  # bold baseline
                         gradient_method=gradient_method,
                         fit_hrf=fit_hrf)

        print("Gaussian iterfit completed at "+datetime.now().strftime('%Y/%m/%d %H:%M:%S')+". Mean rsq>"+str(rsq_threshold)\
+": "+str(gf.iterative_search_params[gf.rsq_mask, -1].mean()))


        np.save(save_path, gf.iterative_search_params)
    elif os.path.exists(save_path+".npy") and refit_mode == "skip":
        gf.iterative_search_params = np.load(save_path+".npy")



#iter gaussian result as starting params for all subsequent modeling
#starting_params = np.insert(gf.iterative_search_params, -1, 1.0, axis=-1)


# CSS iterative fit
if "CSS" in models_to_fit:
    save_path = opj(data_path, subj+"_iterparams-css_space-"+fitting_space+str(chunk_nr))

    gg_css = CSS_Iso2DGaussianGridder(stimulus=prf_stim,
                                      hrf=hrf,
                                      filter_predictions=True,
                                      window_length=window_length,
                                      task_lengths=task_lengths)
    gf_css = CSS_Iso2DGaussianFitter(
        data=tc_full_iso_nonzerovar_dict['tc'], gridder=gg_css, n_jobs=n_jobs,
        previous_gaussian_fitter=gf)
    
    if not os.path.exists(save_path+".npy") or refit_mode == "overwrite":
        print("Starting CSS iter fit at "+datetime.now().strftime('%Y/%m/%d %H:%M:%S'))

        gf_css.iterative_fit(rsq_threshold=rsq_threshold, verbose=verbose,
                             bounds=[(-10*ss, 10*ss),  # x
                                     (-10*ss, 10*ss),  # y
                                     (eps, 10*ss),  # prf size
                                     (-inf, +inf),  # prf amplitude
                                     (0, +inf),  # bold baseline
                                     (0.001, 3)],  # CSS exponent
                             gradient_method=gradient_method,
                             fit_hrf=fit_hrf)

        np.save(save_path, gf_css.iterative_search_params)

        print("CSS iterfit completed at "+datetime.now().strftime('%Y/%m/%d %H:%M:%S')+". Mean rsq>"+str(rsq_threshold)+": "+str(gf_css.iterative_search_params[gf_css.rsq_mask, -1].mean()))
    elif os.path.exists(save_path+".npy") and refit_mode == "iterate":
        print("Starting CSS iter fit at "+datetime.now().strftime('%Y/%m/%d %H:%M:%S'))

        gf_css.iterative_fit(rsq_threshold=rsq_threshold, verbose=verbose,
                             starting_params=np.load(save_path+".npy"),
                             bounds=[(-10*ss, 10*ss),  # x
                                     (-10*ss, 10*ss),  # y
                                     (eps, 10*ss),  # prf size
                                     (-inf, +inf),  # prf amplitude
                                     (0, +inf),  # bold baseline
                                     (0.001, 3)],  # CSS exponent
                             gradient_method=gradient_method,
                             fit_hrf=fit_hrf)

        np.save(save_path, gf_css.iterative_search_params)

        print("CSS iterfit completed at "+datetime.now().strftime('%Y/%m/%d %H:%M:%S')+". Mean rsq>"+str(rsq_threshold)+": "\
+str(gf_css.iterative_search_params[gf_css.rsq_mask, -1].mean()))

    elif os.path.exists(save_path+".npy") and refit_mode == "skip":
        gf_css.iterative_search_params = np.load(save_path+".npy")



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
                                     previous_gaussian_fitter=gf)
    
    save_path = opj(data_path, subj+"_iterparams-dog_space-"+fitting_space+str(chunk_nr))

    if not os.path.exists(save_path+".npy") or refit_mode == "overwrite":
        print("Starting DoG iter fit at "+datetime.now().strftime('%Y/%m/%d %H:%M:%S'))

        gf_dog.iterative_fit(rsq_threshold=rsq_threshold, verbose=verbose,
                                     bounds=[(-10*ss, 10*ss),  # x
                                             (-10*ss, 10*ss),  # y
                                             (eps, 10*ss),  # prf size
                                             (-inf, +inf),  # prf amplitude
                                             (0, +inf),  # bold baseline
                                             (-inf, +inf),  # surround amplitude
                                             (eps, 20*ss)],  # surround size
                                     gradient_method=gradient_method,
                                     fit_hrf=fit_hrf)

        np.save(save_path, gf_dog.iterative_search_params)


        print("DoG iterfit completed at "+datetime.now().strftime('%Y/%m/%d %H:%M:%S')+". Mean rsq>"+str(rsq_threshold)+": "+str(gf_dog.iterative_search_params[gf_dog.rsq_mask, -1].mean()))


    elif os.path.exists(save_path+".npy") and refit_mode == "iterate":
        print("Starting DoG iter fit at "+datetime.now().strftime('%Y/%m/%d %H:%M:%S'))

        gf_dog.iterative_fit(rsq_threshold=rsq_threshold, verbose=verbose,
                             starting_params=np.load(save_path+".npy"),
                             bounds=[(-10*ss, 10*ss),  # x
                                             (-10*ss, 10*ss),  # y
                                             (eps, 10*ss),  # prf size
                                             (-inf, +inf),  # prf amplitude
                                             (0, +inf),  # bold baseline
                                             (-inf, +inf),  # surround amplitude
                                             (eps, 20*ss)],  # surround size
                                     gradient_method=gradient_method,
                                     fit_hrf=fit_hrf)

        np.save(save_path, gf_dog.iterative_search_params)


        print("DoG iterfit completed at "+datetime.now().strftime('%Y/%m/%d %H:%M:%S')+". Mean rsq>"+str(rsq_threshold)+": "+str\
(gf_dog.iterative_search_params[gf_dog.rsq_mask, -1].mean()))
    elif os.path.exists(save_path+".npy") and refit_mode == "skip":
        gf_dog.iterative_search_params = np.load(save_path+".npy")






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
                                       previous_gaussian_fitter=gf)
    
    #normalization grid stage
    if "norm_gridparams_path" not in analysis_info:

        save_path = opj(data_path, subj+"_gridparams-norm_space-"+fitting_space+str(chunk_nr))

        if not os.path.exists(save_path+".npy") or refit_mode == "overwrite":
            surround_amplitude_grid=np.array([0,0.05,0.2,1], dtype='float32')
            surround_size_grid=np.array([3,6,15,30], dtype='float32')
            neural_baseline_grid=np.array([0,1,10,100], dtype='float32')
            surround_baseline_grid=np.array([1.0,10.0,100.0,1000.0], dtype='float32')


            print("Starting norm grid fit at "+datetime.now().strftime('%Y/%m/%d %H:%M:%S'))
            gf_norm.grid_fit(surround_amplitude_grid,
                         surround_size_grid,
                         neural_baseline_grid,
                         surround_baseline_grid,
                         verbose=verbose,
                         n_batches=n_batches,
                         rsq_threshold=rsq_threshold)
        
            print("Norm gridfit completed at "+datetime.now().strftime('%Y/%m/%d %H:%M:%S')+". Mean rsq>"+str(rsq_threshold)+": "+str(gf_norm.gridsearch_params[gf_norm.gridsearch_rsq_mask, -1].mean()))
        
            np.save(save_path, gf_norm.gridsearch_params)
        elif os.path.exists(save_path+".npy") and (refit_mode == "iterate" or refit_mode == "skip"):
            gf_norm.gridsearch_params = np.load(save_path+".npy")

    else:
        gf_norm.gridsearch_params = np.array_split(np.load(analysis_info["norm_gridparams_path"]), n_chunks)[chunk_nr]


    save_path = opj(data_path, subj+"_iterparams-norm_space-"+fitting_space+str(chunk_nr))

    if not os.path.exists(save_path+".npy") or refit_mode == "overwrite":

        print("Starting norm iter fit at "+datetime.now().strftime('%Y/%m/%d %H:%M:%S'))

        gf_norm.iterative_fit(rsq_threshold=rsq_threshold, verbose=verbose,
                                       bounds=[(-10*ss, 10*ss),  # x
                                               (-10*ss, 10*ss),  # y
                                               (eps, 10*ss),  # prf size
                                               (-inf, +inf),  # prf amplitude
                                               (0, +inf),  # bold baseline
                                               (0, +inf),  # surround amplitude
                                               (eps, 20*ss),  # surround size
                                               (-inf, +inf),  # neural baseline
                                               (1e-6, +inf)],  # surround baseline
                                       gradient_method=gradient_method,
                                       fit_hrf=fit_hrf)

        np.save(save_path, gf_norm.iterative_search_params)

        print("Norm iterfit completed at "+datetime.now().strftime('%Y/%m/%d %H:%M:%S')+". Mean rsq>"+str(rsq_threshold)+": "+str(gf_norm.iterative_search_params[gf_norm.rsq_mask, -1].mean()))

    elif os.path.exists(save_path+".npy") and refit_mode == "iterate":

        print("Starting norm iter fit at "+datetime.now().strftime('%Y/%m/%d %H:%M:%S'))

        gf_norm.iterative_fit(rsq_threshold=rsq_threshold, verbose=verbose,
                              starting_params=np.load(save_path+".npy"),
                                      bounds=[(-10*ss, 10*ss),  # x
                                               (-10*ss, 10*ss),  # y
                                               (eps, 10*ss),  # prf size
                                               (-inf, +inf),  # prf amplitude
                                               (0, +inf),  # bold baseline
                                               (0, +inf),  # surround amplitude
                                               (eps, 20*ss),  # surround size
                                               (-inf, +inf),  # neural baseline
                                               (1e-6, +inf)],  # surround baseline
                                       gradient_method=gradient_method,
                                       fit_hrf=fit_hrf)

        np.save(save_path, gf_norm.iterative_search_params)

        print("Norm iterfit completed at "+datetime.now().strftime('%Y/%m/%d %H:%M:%S')+". Mean rsq>"+str(rsq_threshold)+": \
"+str(gf_norm.iterative_search_params[gf_norm.rsq_mask, -1].mean()))

    elif os.path.exists(save_path+".npy") and refit_mode == "skip":
        gf_norm.iterative_search_params = np.load(save_path+".npy")
