#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 17 10:37:08 2020

@author: marcoaqil
"""
import os
import numpy as np
from tqdm import tqdm
import yaml
import matplotlib.pyplot as pl
import cortex
import cifti
import nibabel as nb
from matplotlib import cm, colors
from mpl_toolkits.mplot3d import Axes3D
from collections import defaultdict as dd
from copy import deepcopy
import itertools
from pathlib import Path
from utils.postproc_utils import model_wrapper, create_model_rf_wrapper, colorbar, norm_1d_sr_function, norm_2d_sr_function
from utils.preproc_utils import create_full_stim
#import seaborn as sns

import time
from scipy.stats import sem, ks_2samp, ttest_1samp, wilcoxon, pearsonr, ttest_ind, ttest_rel, zscore, spearmanr
from scipy.optimize import minimize

opj = os.path.join

from statsmodels.stats import weightstats
from sklearn.linear_model import LinearRegression
from sklearn.decomposition import PCA, KernelPCA
from sklearn.cross_decomposition import CCA, PLSCanonical, PLSRegression

from sklearn.metrics import mean_squared_error
from nibabel.freesurfer.io import read_morph_data, write_morph_data
from utils.preproc_utils import roi_mask
#from builtins import input

class visualize_results(object):
    def __init__(self, results):
        self.main_dict = deepcopy(results.main_dict) 
        self.js_handle_dict = dd(lambda:dd(lambda:dd(dict)))
        
        self.get_spaces()
        self.get_subjects(self.main_dict)
        
        self.prf_stims = dict()
        
    def fix_hcp_ordering(self, curr_dict, hcp_atlas_mask_path):
        
        f=nb.load(hcp_atlas_mask_path[0])
        data_1 = np.array([arr.data for arr in f.darrays])[0].astype('bool')

        f=nb.load(hcp_atlas_mask_path[1])
        data_2 = np.array([arr.data for arr in f.darrays])[0].astype('bool')
        for k, v in curr_dict.items():
            if isinstance(v, np.ndarray):
                if v.ndim == 1:
                    #processed results
                    lh_data = np.zeros(data_1.shape, dtype=v.dtype)
                    rh_data = np.zeros(data_2.shape, dtype=v.dtype)
                else:
                    #results
                    lh_data = np.zeros((data_1.shape[0], v.shape[1]), dtype=v.dtype)
                    rh_data = np.zeros((data_2.shape[0], v.shape[1]), dtype=v.dtype)
                    
                lh_data[data_1] = v[:np.sum(data_1)]
                rh_data[data_2] = v[np.sum(data_1):(np.sum(data_1) + np.sum(data_2))]
        
                resample = np.concatenate((lh_data,rh_data))

                curr_dict[k] = np.copy(resample)
           
                
            elif isinstance(v, dict):      
                self.fix_hcp_ordering(v, hcp_atlas_mask_path)
            else:
                pass
        
        
    def get_subjects(self, curr_dict, subject_list = []):
        for k, v in curr_dict.items():
            if 'fsaverage' not in k:
                if 'sub-' not in k and not k.isdecimal():# and isinstance(v, (dict,dd)):
                    self.get_subjects(v, subject_list)
                else:
                    if k not in subject_list:
                        subject_list.append(k)
            else:
                
                if 'fsaverage' == k and 'fsaverage' not in subject_list:
                    subject_list.append('fsaverage')
                    self.get_subjects(v, subject_list)
                else:
                    if k not in subject_list:
                        subject_list.append(k)
        
        self.subjects = subject_list
        return
    
    def get_spaces(self):
        self.spaces = self.main_dict.keys()
        return
        
    def define_rois_and_flatmaps(self, fs_dir, output_rois_path, import_flatmaps, output_rois, hcp_atlas_path = None):
        self.idx_rois = dd(dict)
        self.fs_dir = fs_dir
        self.get_spaces()
        self.get_subjects(self.main_dict)
        for subj in self.subjects:
            if os.path.exists(opj(self.fs_dir, subj)):
            
                if subj not in cortex.db.subjects:
                    print("importing subject from freesurfer")
                    print(f"note: this command often files when executed on mac OS via jupyter notebook.\
                          Rather, source freesurfer and execute it in ipython: \
                              cortex.freesurfer.import_subj({subj}, freesurfer_subject_dir={self.fs_dir}, \
                          whitematter_surf='smoothwm')")
                    cortex.freesurfer.import_subj(subj, freesurfer_subject_dir=self.fs_dir, 
                          whitematter_surf='smoothwm')
                if import_flatmaps:
                    try:
                        print('importing flatmaps from freesurfer')
                        cortex.freesurfer.import_flat(subject=subj, patch='full', hemis=['lh', 'rh'], 
                                                  freesurfer_subject_dir=self.fs_dir, clean=True)
                    except Exception as e:
                        print(e)
                        pass
                            
                if os.path.exists(opj(self.fs_dir, subj, 'label', 'lh.wang2015atlas.V1d.label')):
                    src_subject=subj
                else:
                    src_subject='fsaverage'
                        
            
                wang_rois = ["V1v", "V1d", "V2v", "V2d", "V3v", "V3d", "hV4", "VO1", "VO2", "PHC1", "PHC2",
                    "TO2", "TO1", "LO2", "LO1", "V3B", "V3A", "IPS0", "IPS1", "IPS2", "IPS3", "IPS4", 
                    "IPS5", "SPL1", "FEF"]
                for roi in wang_rois:
                    try:
                        self.idx_rois[subj][roi], _ = cortex.freesurfer.get_label(subj,
                                                              label='wang2015atlas.'+roi,
                                                              fs_dir=self.fs_dir,
                                                              src_subject=src_subject,
                                                              verbose=True)
                    except Exception as e:
                        print(e)
            
                self.idx_rois[subj]['visual_system'] = np.concatenate(tuple([self.idx_rois[subj][roi] for roi in self.idx_rois[subj]]), axis=0)
                self.idx_rois[subj]['V1']=np.concatenate((self.idx_rois[subj]['V1v'],self.idx_rois[subj]['V1d']))
                self.idx_rois[subj]['V2']=np.concatenate((self.idx_rois[subj]['V2v'],self.idx_rois[subj]['V2d']))
                self.idx_rois[subj]['V3']=np.concatenate((self.idx_rois[subj]['V3v'],self.idx_rois[subj]['V3d']))
                self.idx_rois[subj]['VO'] = np.concatenate((self.idx_rois[subj]['VO1'],self.idx_rois[subj]['VO2']))
                self.idx_rois[subj]['TO'] = np.concatenate((self.idx_rois[subj]['TO1'],self.idx_rois[subj]['TO2']))
                self.idx_rois[subj]['LO'] = np.concatenate((self.idx_rois[subj]['LO1'],self.idx_rois[subj]['LO2']))
                self.idx_rois[subj]['V3AB'] = np.concatenate((self.idx_rois[subj]['V3A'],self.idx_rois[subj]['V3B']))
                self.idx_rois[subj]['IPS'] = np.concatenate(tuple([self.idx_rois[subj][rroi] for rroi in self.idx_rois[subj].keys() if 'IPS' in rroi]))            
                #parse custom ROIs if they have been created
                for roi in [el for el in os.listdir(opj(self.fs_dir, subj, 'label')) if 'custom' in el]:
                    roi = roi.replace('lh.','').replace('rh.','').replace('.label','')
                    try:
                        self.idx_rois[subj][roi], _ = cortex.freesurfer.get_label(subj,
                                                              label=roi,
                                                              fs_dir=self.fs_dir,
                                                              src_subject=subj,
                                                              hemisphere=['lh', 'rh'])
                    except Exception as e:
                        print(e)
                        pass
            elif 'HCP' in self.spaces and subj.isdecimal() and hcp_atlas_path != None:
                #HCP data
                print('Using HCP atlas to define ROIs')
                atlas = np.concatenate((nb.load(hcp_atlas_path+'.L.gii').darrays[0].data,
                                        nb.load(hcp_atlas_path+'.R.gii').darrays[0].data))
                with open(hcp_atlas_path+'.json') as f:
                    atlas_labels = yaml.safe_load(f)
                for ke,va in atlas_labels.items():
                    self.idx_rois[subj]['HCPQ1Q6.'+ke] = np.where(atlas == va)
                    self.idx_rois[subj]['HCPQ1Q6.'+ke[:-1]] = np.where((atlas == atlas_labels[ke[:-1]+'R']) | (atlas == atlas_labels[ke[:-1]+'L']))[0]
                
                self.idx_rois[subj]['HCPQ1Q6.V3AB'] = np.concatenate((self.idx_rois[subj]['HCPQ1Q6.V3A'],self.idx_rois[subj]['HCPQ1Q6.V3B']))
                self.idx_rois[subj]['HCPQ1Q6.LO'] = np.concatenate((self.idx_rois[subj]['HCPQ1Q6.LO1'],self.idx_rois[subj]['HCPQ1Q6.LO2'],self.idx_rois[subj]['HCPQ1Q6.LO3']))
                self.idx_rois[subj]['HCPQ1Q6.TO'] = np.concatenate((self.idx_rois[subj]['HCPQ1Q6.MST'],self.idx_rois[subj]['HCPQ1Q6.MT']))
                

                
            if import_flatmaps:
                for roi_name, roi_idx in self.idx_rois[subj].items():
                    if 'custom' in roi_name:
                        #need a correctly flattened brain to do this
                        try:
                            roi_data = np.zeros(cortex.db.get_surfinfo(subj).data.shape).astype('bool')
                            roi_data[roi_idx] = 1
                            roi_vertices=cortex.Vertex(roi_data, subj)
                            cortex.add_roi(roi_vertices, name=roi_name, open_inkscape=False, add_path=True)
                        except Exception as e:
                            print(e)
                            pass        
                        
            #For ROI-based fitting
            if len(output_rois)>0:
                try:
                    rois = np.concatenate(tuple([self.idx_rois[subj][roi] for roi in self.output_rois]), axis=0)
                    np.save(opj(output_rois_path, f"prfpy/{subj}_combined-rois.npy"), rois)
                except Exception as e:
                    print(e)
                    pass    
        
        if 'fsaverage' in self.spaces and 'HCP' in self.spaces:
            for sj in [s for s in self.subjects if 'fsaverage' in s and 'fsaverage' != s]:
                self.idx_rois[sj] = deepcopy(self.idx_rois['fsaverage'])

        return

    def set_alpha(self, space_names='all', only_models=None, ecc_min=0, ecc_max=5):
        
        self.only_models=only_models
        self.tc_min = dict()
        if space_names == 'all':
            spaces = [item for item in self.main_dict.items()]
        else:
            spaces = [item for item in self.main_dict.items() if item[0] in space_names] 

        for space, space_res in spaces:
            if 'fs' in space or 'HCP' in space:
                for analysis, analysis_res in space_res.items():       
                    for subj, subj_res in analysis_res.items():
                        print(space+" "+analysis+" "+subj)
                            
                        p_r = subj_res['Processed Results']
                        models = p_r['RSq'].keys()
                                                
                        if space != 'fsaverage':
                            tc_stats = subj_res['Timecourse Stats']
                        else:
                            tc_stats=dict()
                            tc_stats['Mean'] = np.ones_like(p_r['RSq'][only_models[0]])
                       
                        #######Raw bold timecourse vein threshold
                        if subj == 'sub-006':
                            self.tc_min[subj] = 45000
                        elif subj == 'sub-007':
                            self.tc_min[subj] = 40000
                        elif subj == 'sub-001':
                            self.tc_min[subj] = 35000
                        else:
                            self.tc_min[subj] = 0
                            
                        ######limits for eccentricity
                        self.ecc_min=ecc_min
                        self.ecc_max=ecc_max
                     
              
                        #housekeeping
                        if only_models is None:
                            rsq = np.vstack(tuple([elem for _,elem in p_r['RSq'].items()])).T
                            ecc = np.vstack(tuple([elem for _,elem in p_r['Eccentricity'].items()])).T

                        else:
                            rsq = np.vstack(tuple([elem for k,elem in p_r['RSq'].items() if k in only_models])).T
                            ecc = np.vstack(tuple([elem for k,elem in p_r['Eccentricity'].items() if k in only_models])).T                            
            
                        #alpha dictionary
                        p_r['Alpha'] = {}          
                        p_r['Alpha']['all'] = rsq.max(-1) * (tc_stats['Mean']>self.tc_min[subj]) * (ecc.min(-1)<self.ecc_max) * (ecc.max(-1)>self.ecc_min) * (rsq.min(-1)>0) #* (p_r['Noise Ceiling']['Noise Ceiling (RSq)']>0)
                        
                        for model in models:
                            p_r['Alpha'][model] = p_r['RSq'][model] * (p_r['Eccentricity'][model]>self.ecc_min) * (p_r['Eccentricity'][model]<self.ecc_max)\
                                * (tc_stats['Mean']>self.tc_min[subj]) 
                       
        return


    def pycortex_plots(self, rois, rsq_thresh,
                       space_names = 'fsnative', analysis_names = 'all', subject_ids='all',
                       timecourse_folder = None, screenshot_paths = []):        
        pl.rcParams.update({'font.size': 16})
        pl.rcParams.update({'pdf.fonttype':42})        
        self.click=0
        #######PYCORTEX PICKERFUN
        #function to plot prf and timecourses when clicking surface vertex in webgl        
        def clicker_function(voxel,hemi,vertex):
            if space == 'fsnative' or space == 'HCP':
                print('recovering vertex index...')
                #translate javascript indeing to python
                lctm, rctm = cortex.utils.get_ctmmap(pycortex_subj, method='mg2', level=9)
                if hemi == 'left':
                    index = lctm[int(vertex)]
                    #print(f"{model} rsq {p_r['RSq'][model][index]}")
                else:
                    index = len(lctm)+rctm[int(vertex)]
                    #print(f"{model} rsq {p_r['RSq'][model][index]}") 
                
                    
                #sorting = np.argsort(subj_res['Processed Results']['RSq']['CSS'])[::-1]#-subj_res['Processed Results']['RSq']['Norm_abcd'])
                #alpha_sort = subj_res['Processed Results']['Alpha']['all'][sorting]
                #sorting = sorting[(alpha_sort>rsq_thresh)]#*(subj_res['Processed Results']['RSq']['Norm_abcd'][sorting]-subj_res['Processed Results']['RSq']['DoG'][sorting]>0.1)*(subj_res['Processed Results']['RSq']['Norm_abcd'][sorting]>0.6)]
                #index = sorting[self.click]
                ###
                #surr_vertices = [2624, 285120, 5642, 144995, 5273, 2927, 1266]
                #both_vertices = [9372, 9383, 17766, 9384]#[161130, 9372, 9383, 302707, 17766, 9395, 9384]
                #index = both_vertices[self.click]
                
                #subj 148133HCP
                #index = 87097 #(here dividind b and d for same value doesnt change timecure at all)
                #index = 45089 #(here dividing b and d by same value changes timecourses greatly)
                
                #index = 136959
                
                print('recovering data and model timecourses...')
                
                vertex_info = f""
    
                #recover needed information
                an_info = subj_res['analysis_info']       
                #this was added later
                if 'normalize_integral_dx' not in an_info:
                    an_info['normalize_integral_dx'] = False                
    
                if not hasattr(self, 'prf_stim') or self.prf_stim.task_names != an_info['task_names']:
        

                    self.prf_stim = create_full_stim(screenshot_paths=screenshot_paths,
                                n_pix=an_info['n_pix'],
                                discard_volumes=an_info['discard_volumes'],
                                baseline_volumes_begin_end=an_info['baseline_volumes_begin_end'],
                                dm_edges_clipping=an_info['dm_edges_clipping'],
                                screen_size_cm=an_info['screen_size_cm'],
                                screen_distance_cm=an_info['screen_distance_cm'],
                                TR=an_info['TR'],
                                task_names=an_info['task_names'],
                                normalize_integral_dx=an_info['normalize_integral_dx'])
    
                    
                tc_paths = [str(path) for path in sorted(Path(timecourse_folder).glob(f"{subj}_timecourse_space-{an_info['fitting_space']}_task-*_run-*.npy"))]    
                mask_paths = [tc_path.replace('timecourse_','mask_') for tc_path in tc_paths]
                #all_task_names = np.unique(np.array([elem.replace('task-','') for path in tc_paths for elem in path.split('_')  if 'task' in elem]))
                all_runs = np.unique(np.array([int(elem.replace('run-','').replace('.npy','')) for path in tc_paths for elem in path.split('_')  if 'run' in elem]))
    
                tc = dict()
                tc_err = dict()
                tc_test = dict()
                tc_fit = dict()
                
                for task in an_info['task_names']:
                    if task not in self.prf_stims:
                        self.prf_stims[task] = create_full_stim(screenshot_paths=[s_p for s_p in screenshot_paths if task in s_p],
                                n_pix=an_info['n_pix'],
                                discard_volumes=an_info['discard_volumes'],
                                baseline_volumes_begin_end=an_info['baseline_volumes_begin_end'],
                                dm_edges_clipping=an_info['dm_edges_clipping'],
                                screen_size_cm=an_info['screen_size_cm'],
                                screen_distance_cm=an_info['screen_distance_cm'],
                                TR=an_info['TR'],
                                task_names=[task],
                                normalize_integral_dx=an_info['normalize_integral_dx'])                    
                        
                    tc_runs=[]
                    
                    for run in all_runs:
                        mask_run = [np.load(mask_path) for mask_path in mask_paths if task in mask_path and f"run-{run}" in mask_path][0]
                        
                        if space == 'HCP':
                            tc_run_idx = np.sum(subj_res['mask'][:index])
                        else:
                            tc_run_idx = np.sum(mask_run[:index])
                        
                        tc_runs.append([np.load(tc_path)[tc_run_idx] for tc_path in tc_paths if task in tc_path and f"run-{run}" in tc_path][0])
                        
                        #tc_runs[-1] *=(100/tc_runs[-1].mean(-1))[...,np.newaxis]
                        #tc_runs[-1] += (tc_runs[-1].mean(-1)-np.median(tc_runs[-1][...,self.prf_stims[task].late_iso_dict[task]], axis=-1))[...,np.newaxis]
                      
                    tc[task] = np.mean(tc_runs, axis=0)
                    tc_err[task] = sem(tc_runs, axis=0)
                    
                    #tc[task] *= (100/tc[task].mean(-1))[...,np.newaxis]
                    #tc[task] += (tc[task].mean(-1)-np.median(tc[task][...,self.prf_stims[task].late_iso_dict[task]], axis=-1))[...,np.newaxis]
                    
                    #fit and test timecourses separately
                    if an_info['crossvalidate'] and 'fit_runs' in an_info:
                        tc_test[task] = np.mean([tc_runs[i] for i in all_runs if i not in an_info['fit_runs']], axis=0)
                        tc_fit[task] = np.mean([tc_runs[i] for i in all_runs if i in an_info['fit_runs']], axis=0)
                        
                        #tc_test[task] *= (100/tc_test[task].mean(-1))[...,np.newaxis]
                        #tc_test[task] += (tc_test[task].mean(-1)-np.median(tc_test[task][...,self.prf_stims[task].late_iso_dict[task]], axis=-1))[...,np.newaxis]
                        #tc_fit[task] *= (100/tc_fit[task].mean(-1))[...,np.newaxis]
                        #tc_fit[task] += (tc_fit[task].mean(-1)-np.median(tc_fit[task][...,self.prf_stims[task].late_iso_dict[task]], axis=-1))[...,np.newaxis]     
                        
    
                    
                tc_full = np.concatenate(tuple([tc[task] for task in tc]), axis=0) - an_info['norm_bold_baseline']
                tc_err = np.concatenate(tuple([tc_err[task] for task in tc_err]), axis=0)
                
                if an_info['crossvalidate'] and 'fit_runs' in an_info:
                    tc_full_test = np.concatenate(tuple([tc_test[task] for task in tc_test]), axis=0) - an_info['norm_bold_baseline']
                    tc_full_fit = np.concatenate(tuple([tc_fit[task] for task in tc_fit]), axis=0) - an_info['norm_bold_baseline']    
                    #timecourse reliability stats
                    vertex_info+=f"CV timecourse reliability stats\n"
                    vertex_info+=f"fit-test timecourses pearson R {pearsonr(tc_full_test,tc_full_fit)[0]:.4f}\n"
                    vertex_info+=f"fit-test timecourses R-squared {1-np.sum((tc_full_fit-tc_full_test)**2)/(tc_full_test.var(-1)*tc_full_test.shape[-1]):.4f}\n\n"
            
                preds = dict()
                prfs = dict()
                
                self.prf_models = dict()
                
                for model in self.only_models:
                    self.prf_models[model] = model_wrapper(model,
                                       stimulus=self.prf_stim,
                                       hrf=an_info['hrf'],
                                       filter_predictions=an_info['filter_predictions'],
                                       filter_type=an_info['filter_type'],
                                       filter_params={x:an_info[x] for x in ["first_modes_to_remove",
                                                                             "last_modes_to_remove_percent",
                                                                             "window_length",
                                                                             "polyorder",
                                                                             "highpass",
                                                                             "add_mean"]},
                                       normalize_RFs=an_info['normalize_RFs'])
                    if space == 'HCP':
                        internal_idx = index
                    else:
                        internal_idx = np.sum(subj_res['mask'][:index])
                    
                    params = np.copy(subj_res['Results'][model][internal_idx,:-1])

                    preds[model] = self.prf_models[model].return_prediction(*list(params))[0] - an_info['norm_bold_baseline']
                    
                    np.set_printoptions(suppress=True, precision=4)
                    vertex_info+=f"{model} params: {params} \n"
                    vertex_info+=f"Rsq {model} full tc: {1-np.sum((preds[model]-tc_full)**2)/(tc_full.var(-1)*tc_full.shape[-1]):.4f}\n"
                    
                    if an_info['crossvalidate'] and 'fit_runs' in an_info:
                        vertex_info+=f"Rsq {model} fit tc: {1-np.sum((preds[model]-tc_full_fit)**2)/(tc_full_fit.var(-1)*tc_full_fit.shape[-1]):.4f}\n"
                        vertex_info+=f"Rsq {model} test tc: {1-np.sum((preds[model]-tc_full_test)**2)/(tc_full_test.var(-1)*tc_full_test.shape[-1]):.4f}\n"
    
    
                        
                        
                        if model != 'Gauss':
                            tc_full_test_gauss_resid = tc_full_test[tc_full_test<0]#(tc_full_test - preds['Gauss'][0])
                            model_gauss_resid = preds[model][tc_full_test<0]#(preds[model] - preds['Gauss'])[0]
    
                            vertex_info+=f"Rsq {model} negative portions test tc: {1-np.sum((model_gauss_resid-tc_full_test_gauss_resid)**2)/(tc_full_test_gauss_resid.var(-1)*tc_full_test_gauss_resid.shape[-1]):.4f}\n"
                            vertex_info+=f"Rsq {model} negative portions test pearson R {pearsonr(tc_full_test_gauss_resid,model_gauss_resid)[0]:.4f}\n"
                            
                            
                            #vertex_info+=f"Rsq {model} gauss resid test tc: {1-np.sum((model_gauss_resid-tc_full_test_gauss_resid)**2)/(tc_full_test_gauss_resid.var(-1)*tc_full_test_gauss_resid.shape[-1]):.4f}\n"
                            #vertex_info+=f"Rsq {model} weighted resid tc: {1-np.sum(((preds[model]-preds['Gauss'])*(preds[model]-tc_full_fit))**2)/(tc_full_fit.var(-1)*tc_full_fit.shape[-1]):.4f}\n"
                        
                    prfs[model] = create_model_rf_wrapper(model,self.prf_stim,params,an_info['normalize_RFs'])
                    
                    for key in subj_res['Processed Results']:
                        if model in subj_res['Processed Results'][key]:
                            vertex_info+=f"{key} {model} {subj_res['Processed Results'][key][model][index]:.8f}\n"
                    
                    vertex_info+="\n"
                    
                pl.ion()
    
                if self.click==0:
                    self.f, self.axes = pl.subplots(2,2,figsize=(10, 30),frameon=True, gridspec_kw={'width_ratios': [8, 2], 'height_ratios': [1,1]})
                    self.f.set_tight_layout(True)
                else:
                    self.cbar.remove()
                    for ax1 in self.axes:
                        for ax2 in ax1: 
                            ax2.clear()
                
                tseconds = an_info['TR']*np.arange(len(tc_full))
                cmap = cm.get_cmap('tab10')
                
                self.axes[0,0].plot(tseconds,np.zeros(len(tc_full)),linestyle='--',linewidth=0.1, color='black', zorder=0)
        
                
    
                if an_info['crossvalidate'] and 'fit_runs' in an_info:
                    
                    print(np.std([tc_full_test,tc_full_fit],axis=0))
                    
                    self.axes[0,0].errorbar(tseconds, tc_full, yerr=tc_err, label='Data',  fmt = 's:k', markersize=1,  linewidth=0.3, zorder=1) 
                    
                    #self.axes[0,0].plot(tseconds, tc_full_test, label='Data (test)', linestyle = ':', marker='^', markersize=1, color=cmap(4), linewidth=0.75, alpha=0.5) 
                    #self.axes[0,0].plot(tseconds, tc_full_fit, label='Data (fit)', linestyle = ':', marker='v', markersize=1, color=cmap(5), linewidth=0.75, alpha=0.5) 
                else:
                    self.axes[0,0].errorbar(tseconds, tc_full, yerr=tc_err, label='Data',  fmt = 's:k', markersize=1,  linewidth=0.3, zorder=1) 
                    
    
                
                for model in self.only_models: 
                    if 'Norm' in model:
                        self.axes[0,0].plot(tseconds, preds[model], linewidth=0.75, color=cmap(3), label=f"Norm ({subj_res['Processed Results']['RSq'][model][index]:.2f})", zorder=2)
                    elif model == 'DoG':
                        self.axes[0,0].plot(tseconds, preds[model], linewidth=0.75, color=cmap(2), label=f"DoG ({subj_res['Processed Results']['RSq'][model][index]:.2f})", zorder=3)
                    elif model == 'CSS':
                        self.axes[0,0].plot(tseconds, preds[model], linewidth=0.75, color=cmap(1), label=f"CSS ({subj_res['Processed Results']['RSq'][model][index]:.2f})", zorder=4)
                    elif model == 'Gauss':
                        self.axes[0,0].plot(tseconds, preds[model], linewidth=0.75, color=cmap(0), label=f"Gauss ({subj_res['Processed Results']['RSq'][model][index]:.2f})", zorder=5)
                        
                
                
       
                
                if np.any(['Norm' in model for model in self.only_models]) or 'DoG' in self.only_models:
                    try:
                        if ((subj_res['Processed Results']['RSq']['DoG'][index]-subj_res['Processed Results']['RSq']['Gauss'][index]) > 0.05) or ((subj_res['Processed Results']['RSq']['Norm_abcd'][index]-subj_res['Processed Results']['RSq']['CSS'][index]) > 0.05):
                            surround_effect = np.min([preds[model] for model in preds if 'Norm' in model or 'DoG' in model],axis=0)
                            surround_effect[surround_effect>0] = preds['Gauss'][surround_effect>0]
                            self.axes[0,0].fill_between(tseconds,
                                                         surround_effect, 
                                             preds['Gauss'], label='Surround suppression', alpha=0.2, color='gray')
                    except:
                        pass
    
              
                self.axes[0,0].legend(ncol=3, fontsize=8, loc=9)
                #self.axes[0,0].set_ylim(-3.5,6.5)
                self.axes[0,0].set_xlim(0,tseconds.max())
                #self.axes[0,0].set_ylabel('% signal change')
                self.axes[0,0].set_title(f"Vertex {index} timecourse")
                
                if prfs['Norm_abcd'][0].min() < 0:
                    im = self.axes[0,1].imshow(prfs['Norm_abcd'][0], cmap='RdBu_r', vmin=prfs['Norm_abcd'][0].min(), vmax=-prfs['Norm_abcd'][0].min())
                else:
                    im = self.axes[0,1].imshow(prfs['Norm_abcd'][0], cmap='RdBu_r', vmax=prfs['Norm_abcd'][0].max(), vmin=-prfs['Norm_abcd'][0].max())
                    
                self.cbar = colorbar(im)
                self.axes[0,1].set_title(f"pRF")
                
                #self.cbar = self.f.colorbar(im, ax=self.axes[0,1], use_gridspec=True)
                
                self.axes[0,1].axis('on')            
                self.axes[1,0].axis('off')
                self.axes[1,1].axis('off')
                self.axes[1,0].text(0,1,vertex_info, fontsize=10)
                
                
                self.click+=1
          
            return


        if space_names == 'all':
            spaces = [item for item in self.main_dict.items()]
        else:
            spaces = [item for item in self.main_dict.items() if item[0] in space_names] 

        for space, space_res in spaces:
            plotted_rois = dd(lambda:False)
            plotted_stats = dd(lambda:False)
            if analysis_names == 'all':
                analyses = space_res.items()
            else:
                analyses = [item for item in space_res.items() if item[0] in analysis_names] 
            for analysis, analysis_res in analyses:  
                if subject_ids == 'all':
                    subjects = analysis_res.items()
                else:
                    subjects = [item for item in analysis_res.items() if item[0] in subject_ids] 
                for subj, subj_res in subjects:
                    
                    pycortex_subj = subj
                    
                    if subj not in cortex.db.subjects and os.path.exists(opj(self.fs_dir,subj)):
                        print("subject not present in pycortex database. attempting to import...")
                        cortex.freesurfer.import_subj(subj, freesurfer_subject_dir=self.fs_dir, 
                              whitematter_surf='smoothwm')
                    elif subj.isdecimal() and space == 'HCP':
                        pycortex_subj = '999999'
                    elif space == 'fsaverage':
                        pycortex_subj = 'fsaverage'

                    p_r = subj_res['Processed Results']
                    models = p_r['RSq'].keys()
                    
                    if space != 'fsaverage':
                        tc_stats = subj_res['Timecourse Stats']
                        mask = subj_res['mask']
                    else:
                        mask = np.ones_like(p_r['RSq'][list(models)[0]])

         
                    if rois != 'all':
                        for key in p_r['Alpha']:
                            p_r['Alpha'][key] = roi_mask(np.concatenate(tuple([self.idx_rois[subj][r] for r in rois])), p_r['Alpha'][key])
                                     

                    #output freesurefer-format polar angle maps to draw custom ROIs in freeview    
                    if self.output_freesurfer_maps:
                                      
                        lh_c = read_morph_data(opj(self.fs_dir, subj+'/surf/lh.curv'))
        
                        polar_freeview = np.copy(p_r['Polar Angle']['Norm_abcd'])#np.mean(polar, axis=-1)
                        ecc_freeview = np.copy(p_r['Eccentricity']['Norm_abcd'])#np.mean(ecc, axis=-1)
                        
                        if subj != 'fsaverage':
                            alpha_freeview = p_r['RSq']['Norm_abcd']* (tc_stats['Mean']>self.tc_min[subj])# rsq.max(-1) * (tc_stats['Mean']>self.tc_min[subj]) * (rsq.min(-1)>0)
                        else:
                            alpha_freeview = p_r['RSq']['Norm_abcd']
                            
                        polar_freeview[alpha_freeview<rsq_thresh] = -10
                        ecc_freeview[alpha_freeview<rsq_thresh] = -10
        
                        write_morph_data(opj(self.fs_dir, subj+'/surf/lh.polar_norm')
                                                               ,polar_freeview[:lh_c.shape[0]])
                        write_morph_data(opj(self.fs_dir, subj+'/surf/rh.polar_norm')
                                                               ,polar_freeview[lh_c.shape[0]:])
                        write_morph_data(opj(self.fs_dir, subj+'/surf/lh.ecc_norm')
                                                               ,ecc_freeview[:lh_c.shape[0]])
                        write_morph_data(opj(self.fs_dir, subj+'/surf/rh.ecc_norm')
                                                               ,ecc_freeview[lh_c.shape[0]:])
                        
                        
                        polar_freeview_masked = np.copy(polar_freeview)
                        ecc_freeview_masked = np.copy(ecc_freeview)
                        alpha_freeview_masked = alpha_freeview * (ecc_freeview<self.ecc_max) * (ecc_freeview>self.ecc_min)
        
                        polar_freeview_masked[alpha_freeview_masked<rsq_thresh] = -10
                        ecc_freeview_masked[alpha_freeview_masked<rsq_thresh] = -10
        
                        write_morph_data(opj(self.fs_dir, subj+'/surf/lh.polar_masked_norm')
                                                               ,polar_freeview_masked[:lh_c.shape[0]])
                        write_morph_data(opj(self.fs_dir, subj+'/surf/rh.polar_masked_norm')
                                                               ,polar_freeview_masked[lh_c.shape[0]:])
                        write_morph_data(opj(self.fs_dir, subj+'/surf/lh.ecc_masked_norm')
                                                               ,ecc_freeview_masked[:lh_c.shape[0]])
                        write_morph_data(opj(self.fs_dir, subj+'/surf/rh.ecc_masked_norm')
                                                               ,ecc_freeview_masked[lh_c.shape[0]:])
                        
                    ##START PYCORTEX VISUALIZATIONS                            
                    #data quality/stats cortex visualization 
                    if self.plot_stats_cortex and not plotted_stats[subj] :
                        
                        stats_mask = mask*(tc_stats['Mean']>self.tc_min[subj])
                        if rois != 'all':
                            stats_mask = roi_mask(np.concatenate(tuple([self.idx_rois[subj][r] for r in rois])), stats_mask)
                        
                        
                        
                        mean_ts_vert = cortex.Vertex2D(tc_stats['Mean'], stats_mask, subject=pycortex_subj, cmap='Jet_2D_alpha')
                        var_ts_vert = cortex.Vertex2D(tc_stats['Variance'], stats_mask, subject=pycortex_subj, cmap='Jet_2D_alpha')
                        tsnr_vert = cortex.Vertex2D(tc_stats['TSNR'], stats_mask, subject=pycortex_subj, cmap='Jet_2D_alpha')
        
                        data_stats ={'mean':mean_ts_vert.raw, 'var':var_ts_vert.raw, 'tsnr':tsnr_vert.raw}
        
                        self.js_handle_dict[space][analysis][subj]['js_handle_stats'] = cortex.webgl.show(data_stats, pickerfun=clicker_function, with_curvature=False, with_labels=True, with_rois=True, with_borders=True, with_colorbar=True)
        
                        plotted_stats[subj] = True
                    
                    if self.plot_rois_cortex and not plotted_rois[subj]:
                        
                        
                        ds_rois = {}
                        data = np.zeros_like(mask).astype('int')
                        custom_rois_data = np.zeros_like(mask).astype('int')
                        hcp_rois_data = np.zeros_like(mask).astype('int')
        
                        for i, roi in enumerate(self.idx_rois[subj]):
        
                            roi_data = np.zeros_like(mask)
                            roi_data[self.idx_rois[subj][roi]] = 1
                            if 'custom' not in roi and 'visual' not in roi and 'HCP' not in roi:
                                data[self.idx_rois[subj][roi]] = i+1
                            if 'custom' in roi and 'Pole' not in roi:
                                custom_rois_data[self.idx_rois[subj][roi]] = i+1
                            if 'HCPQ1Q6.' in roi:
                                hcp_rois_data[self.idx_rois[subj][roi]] = i+1

                            ds_rois[roi] = cortex.Vertex2D(roi_data, roi_data.astype('bool'), subject=pycortex_subj, cmap='RdBu_r_alpha').raw
        

                        if data.sum()>0:
                            ds_rois['Wang2015Atlas'] = cortex.Vertex2D(data, data.astype('bool'), subject=pycortex_subj, cmap='Retinotopy_HSV_2x_alpha').raw
                        if custom_rois_data.sum()>0:
                            ds_rois['Custom ROIs'] = cortex.Vertex2D(custom_rois_data, custom_rois_data.astype('bool'), subject=pycortex_subj, cmap='Retinotopy_HSV_2x_alpha').raw 
                        if hcp_rois_data.sum()>0:
                            ds_rois['HCP ROIs'] = cortex.Vertex2D(hcp_rois_data, hcp_rois_data.astype('bool'), subject=pycortex_subj, cmap='Retinotopy_HSV_2x_alpha').raw 
                        
                        self.js_handle_dict[space][analysis][subj]['js_handle_rois'] = cortex.webgl.show(ds_rois, pickerfun=clicker_function, with_curvature=False, with_labels=True, with_rois=True, with_borders=True, with_colorbar=True)
        
                        plotted_rois[subj] = True
                                                

                        
        
                    if self.plot_rsq_cortex:              
                        ds_rsq = dict()
                        
                        
                        best_model = np.argmax([p_r['RSq'][model] for model in self.only_models],axis=0)

                        ds_rsq['Best model'] = cortex.Vertex2D(best_model, p_r['Alpha']['all'], subject=pycortex_subj,
                                                                      vmin2=rsq_thresh, vmax2=0.6, cmap='BROYG_2D').raw 


                        for model in self.only_models:
                            ds_rsq[model] = cortex.Vertex2D(p_r['RSq'][model], p_r['Alpha'][model], subject=pycortex_subj, 
                                                            vmin=rsq_thresh, vmax=0.8, vmin2=rsq_thresh, vmax2=0.6, cmap='Jet_2D_alpha').raw
                            
                        if 'CSS' in models and p_r['RSq']['CSS'].sum()>0:
                            ds_rsq['CSS - Gauss'] = cortex.Vertex2D(p_r['RSq']['CSS']-p_r['RSq']['Gauss'], p_r['Alpha']['all'], subject=pycortex_subj,
                                                                      vmin=-0.1, vmax=0.1, vmin2=rsq_thresh, vmax2=0.6, cmap='Jet_2D_alpha').raw   
                            
                        if 'DoG' in models and p_r['RSq']['DoG'].sum()>0:
                            ds_rsq['DoG - Gauss'] = cortex.Vertex2D(p_r['RSq']['DoG']-p_r['RSq']['Gauss'], p_r['Alpha']['all'], subject=pycortex_subj,
                                                                  vmin=-0.1, vmax=0.1, vmin2=rsq_thresh, vmax2=0.6, cmap='Jet_2D_alpha').raw
                        
                        if 'Norm_abcd' in self.only_models and 'Gauss' in self.only_models:

                            ds_rsq[f'Norm_abcd - Gauss'] = cortex.Vertex2D(p_r['RSq']['Norm_abcd']-p_r['RSq']['Gauss'], p_r['Alpha']['all'], subject=pycortex_subj,
                                                                      vmin=-0.1, vmax=0.1, vmin2=rsq_thresh, vmax2=0.6, cmap='Jet_2D_alpha').raw

                            if 'CSS' in self.only_models and 'DoG' in self.only_models:
                            
                                ds_rsq[f'Norm_abcd - DoG'] = cortex.Vertex2D(p_r['RSq']['Norm_abcd']-p_r['RSq']['DoG'], p_r['Alpha']['all'], subject=pycortex_subj,
                                                                      vmin=-0.1, vmax=0.1, vmin2=rsq_thresh, vmax2=0.6, cmap='Jet_2D_alpha').raw

                                ds_rsq[f'Norm_abcd - CSS'] = cortex.Vertex2D(p_r['RSq']['Norm_abcd']-p_r['RSq']['CSS'], p_r['Alpha']['all'], subject=pycortex_subj, 
                                                                      vmin=-0.1, vmax=0.1, vmin2=rsq_thresh, vmax2=0.6, cmap='Jet_2D_alpha').raw
 

                        for model in [model for model in self.only_models if 'Norm' in model and 'Norm_abcd' != model]:

                            ds_rsq[f'{model} - Norm_abcd'] = cortex.Vertex2D(p_r['RSq'][model]-p_r['RSq']['Norm_abcd'], p_r['Alpha']['all'], subject=pycortex_subj,
                                                                      vmin=-0.1, vmax=0.1, vmin2=rsq_thresh, vmax2=0.6, cmap='Jet_2D_alpha').raw
                            
                            
                        if 'Processed Results' in self.main_dict['T1w'][analysis][subj] and self.compare_volume_surface:
                            
                            ds_rsq_comp = dict()
                            volume_rsq = self.main_dict['T1w'][analysis][subj]['Processed Results']['RSq']['Norm']
                            ref_img = nb.load(self.main_dict['T1w'][analysis][subj]['Results']['ref_img_path'])
                            
                            #rsq_img = nb.Nifti1Image(volume_rsq, ref_img.affine, ref_img.header)

                            xfm_trans = cortex.xfm.Transform(np.identity(4), ref_img)
                            xfm_trans.save(subj, 'func_space_transform')
                            
                            ds_rsq_comp['Norm_abcd CV rsq (volume fit)'] = cortex.Volume2D(volume_rsq.T, volume_rsq.T, subj, 'func_space_transform',
                                                                      vmin=rsq_thresh, vmax=0.6, vmin2=0.05, vmax2=rsq_thresh, cmap='Jet_2D_alpha')
                            ds_rsq_comp['Norm_abcd CV rsq (surface fit)'] = cortex.Vertex2D(p_r['RSq']['Norm_abcd'], p_r['RSq']['Norm_abcd'], subject=pycortex_subj,
                                                                      vmin=rsq_thresh, vmax=0.6, vmin2=0.05, vmax2=rsq_thresh, cmap='Jet_2D_alpha').raw
                            self.js_handle_dict[space][analysis][subj]['js_handle_rsq_comp'] = cortex.webgl.show(ds_rsq_comp, pickerfun=clicker_function, with_curvature=False, with_labels=True, with_rois=True, with_borders=True, with_colorbar=True)
                        
                        
                        self.js_handle_dict[space][analysis][subj]['js_handle_rsq'] = cortex.webgl.show(ds_rsq, pickerfun=clicker_function, with_curvature=False, with_labels=True, with_rois=True, with_borders=True, with_colorbar=True) 
                        
                    if self.plot_ecc_cortex:
                        ds_ecc = dict()
                        
                        for model in self.only_models:
                            ds_ecc[model] = cortex.Vertex2D(p_r['Eccentricity'][model], p_r['Alpha'][model], subject=pycortex_subj, 
                                                            vmin=self.ecc_min, vmax=self.ecc_max-1, vmin2=rsq_thresh, vmax2=0.6, cmap='Jet_r_2D_alpha').raw
        
                        self.js_handle_dict[space][analysis][subj]['js_handle_ecc'] = cortex.webgl.show(ds_ecc, pickerfun=clicker_function, with_curvature=False, with_labels=True, with_rois=True, with_borders=True, with_colorbar=True)
        
                    if self.plot_polar_cortex:
                        ds_polar = dict()
                        
                        for model in self.only_models:
                            ds_polar[model] = cortex.Vertex2D(p_r['Polar Angle'][model], p_r['Alpha'][model], subject=pycortex_subj, 
                                                              vmin2=rsq_thresh, vmax2=0.6, cmap='Retinotopy_HSV_2x_alpha').raw
                        
                        if 'Processed Results' in self.main_dict['T1w'][analysis][subj] and self.compare_volume_surface:
                            ds_polar_comp = dict()
                            
                            volume_rsq = self.main_dict['T1w'][analysis][subj]['Processed Results']['RSq']['Norm_abcd']
                            volume_polar = self.main_dict['T1w'][analysis][subj]['Processed Results']['Polar Angle']['Norm_abcd']
                            ref_img = nb.load(self.main_dict['T1w'][analysis][subj]['Results']['ref_img_path'])                                

                            xfm_trans = cortex.xfm.Transform(np.identity(4), ref_img)
                            xfm_trans.save(subj, 'func_space_transform')
                            
                            ds_polar_comp['Norm_abcd CV polar (volume fit)'] = cortex.Volume2D(volume_polar.T, volume_rsq.T, subj, 'func_space_transform',
                                                                      vmin2=0.05, vmax2=rsq_thresh, cmap='Retinotopy_HSV_2x_alpha')
                            ds_polar_comp['Norm_abcd CV polar (surface fit)'] = cortex.Vertex2D(p_r['Polar Angle']['Norm_abcd'], p_r['RSq']['Norm_abcd'], subject=pycortex_subj,
                                                                      vmin2=0.05, vmax2=rsq_thresh, cmap='Retinotopy_HSV_2x_alpha').raw
                            self.js_handle_dict[space][analysis][subj]['js_handle_polar_comp'] = cortex.webgl.show(ds_polar_comp, pickerfun=clicker_function, with_curvature=False, with_labels=True, with_rois=True, with_borders=True, with_colorbar=True)

                        
                        self.js_handle_dict[space][analysis][subj]['js_handle_polar'] = cortex.webgl.show(ds_polar, pickerfun=clicker_function, with_curvature=False, with_labels=True, with_rois=True, with_borders=True, with_colorbar=True)
        
                    if self.plot_size_cortex:
                        ds_size = dict()
                        
                        for model in self.only_models:
                            ds_size[f"{model} fwhmax"] = cortex.Vertex2D(p_r['Size (fwhmax)'][model], p_r['Alpha'][model], subject=pycortex_subj, 
                                                             vmin=0, vmax=6, vmin2=rsq_thresh, vmax2=0.6, cmap='Jet_2D_alpha').raw
                            ds_size[f"{model} sigma_1"] = cortex.Vertex2D(p_r['Size (sigma_1)'][model], p_r['Alpha'][model], subject=pycortex_subj, 
                                                             vmin=0, vmax=6, vmin2=rsq_thresh, vmax2=0.6, cmap='Jet_2D_alpha').raw
                            
                        self.js_handle_dict[space][analysis][subj]['js_handle_size'] = cortex.webgl.show(ds_size, pickerfun=clicker_function, with_curvature=False, with_labels=True, with_rois=True, with_borders=True, with_colorbar=True)
        
        
                    if self.plot_amp_cortex:
                        ds_amp = dict()
                        
                        for model in self.only_models:
                            ds_amp[model] = cortex.Vertex2D(p_r['Amplitude'][model], p_r['Alpha'][model], subject=pycortex_subj, 
                                                            vmin=np.nanquantile(p_r['Amplitude'][model][p_r['Alpha'][model]>rsq_thresh],0.1), 
                                                            vmax=np.nanquantile(p_r['Amplitude'][model][p_r['Alpha'][model]>rsq_thresh],0.9), vmin2=rsq_thresh, vmax2=0.6, cmap='Jet_2D_alpha').raw
        
                        self.js_handle_dict[space][analysis][subj]['js_handle_amp'] = cortex.webgl.show(ds_amp, pickerfun=clicker_function, with_curvature=False, with_labels=True, with_rois=True, with_borders=True, with_colorbar=True)
                        
                    if self.plot_css_exp_cortex and 'CSS' in self.only_models:
                        ds_css_exp = dict()
                        
                        ds_css_exp['CSS Exponent'] = cortex.Vertex2D(p_r['CSS Exponent']['CSS'], p_r['Alpha']['CSS'], subject=pycortex_subj, 
                                                                     vmin=0, vmax=1, vmin2=rsq_thresh, vmax2=0.6, cmap='Jet_2D_alpha').raw
        
                        self.js_handle_dict[space][analysis][subj]['js_handle_css_exp'] = cortex.webgl.show(ds_css_exp, pickerfun=clicker_function, with_curvature=False, with_labels=True, with_rois=True, with_borders=True, with_colorbar=True)
                        
                    if self.plot_surround_size_cortex:
                        ds_surround_size = dict()
                        
                        
                        for model in self.only_models:
                            if model == 'DoG':
                                ds_surround_size[f'DoG fwatmin'] = cortex.Vertex2D(p_r['Surround Size (fwatmin)']['DoG'], p_r['Alpha']['DoG'], subject=pycortex_subj, 
                                                                     vmin=0, vmax=50, vmin2=rsq_thresh, vmax2=0.6, cmap='Jet_2D_alpha').raw
                                ds_surround_size[f'DoG sigma_2'] = cortex.Vertex2D(p_r['Size (sigma_2)']['DoG'], p_r['Alpha']['DoG'], subject=pycortex_subj, 
                                                                     vmin=0, vmax=50, vmin2=rsq_thresh, vmax2=0.6, cmap='Jet_2D_alpha').raw
                            elif 'Norm' in model:
                                ds_surround_size[f"{model} fwatmin"] = cortex.Vertex2D(p_r['Surround Size (fwatmin)'][model], p_r['Alpha'][model], subject=pycortex_subj, 
                                                                     vmin=0, vmax=50, vmin2=rsq_thresh, vmax2=0.6, cmap='Jet_2D_alpha').raw                    
                                ds_surround_size[f"{model} sigma_2"] = cortex.Vertex2D(p_r['Size (sigma_2)'][model], p_r['Alpha'][model], subject=pycortex_subj, 
                                                                     vmin=0, vmax=10, vmin2=rsq_thresh, vmax2=0.6, cmap='Jet_2D_alpha').raw    
                                
                        self.js_handle_dict[space][analysis][subj]['js_handle_surround_size'] = cortex.webgl.show(ds_surround_size, pickerfun=clicker_function, with_curvature=False, with_labels=True, with_rois=True, with_borders=True, with_colorbar=True)    

                    if self.plot_suppression_index_cortex:
                        ds_suppression_index = dict()
                        
                        
                        for model in self.only_models:
                            if model == 'DoG':
                                ds_suppression_index[f'{model} SI (full)'] = cortex.Vertex2D(p_r['Suppression Index (full)'][model], p_r['Alpha'][model], subject=pycortex_subj, 
                                                                     vmin=0, vmax=10, vmin2=rsq_thresh, vmax2=0.6, cmap='Jet_2D_alpha').raw
                                ds_suppression_index[f'{model} SI (aperture)'] = cortex.Vertex2D(p_r['Suppression Index'][model], p_r['Alpha'][model], subject=pycortex_subj, 
                                                                     vmin=0, vmax=1.5, vmin2=rsq_thresh, vmax2=0.6, cmap='Jet_2D_alpha').raw                                    
                            elif 'Norm' in model:
                                ds_suppression_index[f'{model} NI (full)'] = cortex.Vertex2D(p_r['Suppression Index (full)'][model], p_r['Alpha'][model], subject=pycortex_subj, 
                                                                     vmin=1, vmax=20, vmin2=rsq_thresh, vmax2=0.6, cmap='Jet_2D_alpha').raw
                                ds_suppression_index[f'{model} NI (aperture)'] = cortex.Vertex2D(p_r['Suppression Index'][model], p_r['Alpha'][model], subject=pycortex_subj, 
                                                                     vmin=0, vmax=1.5, vmin2=rsq_thresh, vmax2=0.6, cmap='Jet_2D_alpha').raw                      
        
                        self.js_handle_dict[space][analysis][subj]['js_handle_suppression_index'] = cortex.webgl.show(ds_suppression_index, pickerfun=clicker_function, with_curvature=False, with_labels=True, with_rois=True, with_borders=True, with_colorbar=True)    

                    if self.plot_size_ratio_cortex:
                        ds_size_ratio = dict()           
                        for model in self.only_models:
                            if model in p_r['Size ratio (sigma_2/sigma_1)']:
                                ds_size_ratio[f'{model} size ratio'] = cortex.Vertex2D(p_r['Size ratio (sigma_2/sigma_1)'][model], p_r['Alpha'][model], subject=pycortex_subj, 
                                                                     vmin=4, vmax=16, vmin2=rsq_thresh, vmax2=0.6, cmap='Jet_2D_alpha').raw                                                      
        
                        self.js_handle_dict[space][analysis][subj]['js_handle_size_ratio'] = cortex.webgl.show(ds_size_ratio, pickerfun=clicker_function, with_curvature=False, with_labels=True, with_rois=True, with_borders=True, with_colorbar=True)    
                    
                    if self.plot_receptors_cortex:
                        if 'Receptor Maps' in p_r:
                            ds_receptors = dict()           
                            for receptor in p_r['Receptor Maps']:
                                
                                ds_receptors[receptor] = cortex.Vertex2D(p_r['Receptor Maps'][receptor], p_r['Alpha']['all'], subject=pycortex_subj, 
                                                                         vmin=np.nanquantile(p_r['Receptor Maps'][receptor][p_r['Alpha']['all']>rsq_thresh],0.1), vmax=np.nanquantile(p_r['Receptor Maps'][receptor][p_r['Alpha']['all']>rsq_thresh],0.9), 
                                                                         vmin2=rsq_thresh, vmax2=0.6, cmap='Jet_2D_alpha').raw                                                      
            
                            self.js_handle_dict[space][analysis][subj]['js_handle_receptors'] = cortex.webgl.show(ds_receptors, pickerfun=clicker_function, with_curvature=False, with_labels=True, with_rois=True, with_borders=True, with_colorbar=True)    

                         
                    if self.plot_norm_baselines_cortex:
                        ds_norm_baselines = dict()
                        
                        
                        for model in [model for model in self.only_models if 'Norm' in model]:

                            ds_norm_baselines[f'{model} Param. B'] = cortex.Vertex2D(p_r['Norm Param. B'][model], p_r['Alpha'][model], subject=pycortex_subj, 
                                                                         vmin=np.nanquantile(p_r['Norm Param. B'][model][p_r['Alpha'][model]>rsq_thresh],0.1), 
                                                                         vmax=np.nanquantile(p_r['Norm Param. B'][model][p_r['Alpha'][model]>rsq_thresh],0.9), vmin2=rsq_thresh, vmax2=0.6, cmap='Jet_2D_alpha').raw                    
                            ds_norm_baselines[f'{model} Param. D'] = cortex.Vertex2D(p_r['Norm Param. D'][model], p_r['Alpha'][model], subject=pycortex_subj, 
                                                                         vmin=np.nanquantile(p_r['Norm Param. D'][model][p_r['Alpha'][model]>rsq_thresh],0.1), 
                                                                         vmax=np.nanquantile(p_r['Norm Param. D'][model][p_r['Alpha'][model]>rsq_thresh],0.9), vmin2=rsq_thresh, vmax2=0.6, cmap='Jet_2D_alpha').raw
                            
                            # if 'Ratio (B/D)' in p_r:
                            #     ds_norm_baselines[f'{model} Ratio (B/D)'] = cortex.Vertex2D(p_r['Ratio (B/D)'][model], p_r['Alpha'][model], subject=pycortex_subj, 
                            #                                              vmin=np.nanquantile(p_r['Ratio (B/D)'][model][p_r['Alpha'][model]>rsq_thresh],0.1), 
                            #                                              vmax=np.nanquantile(p_r['Ratio (B/D)'][model][p_r['Alpha'][model]>rsq_thresh],0.9), vmin2=rsq_thresh, vmax2=0.6, cmap='Jet_2D_alpha').raw
                        
                        self.js_handle_dict[space][analysis][subj]['js_handle_norm_baselines'] = cortex.webgl.show(ds_norm_baselines, pickerfun=clicker_function, with_curvature=False, with_labels=True, with_rois=True, with_borders=True, with_colorbar=True)    
        print('-----')                              
        return    
        
    def save_pycortex_views(self, js_handle, base_str, views):

        
        surfaces = dict(inflated=dict(unfold=1))#,
                       # fiducial=dict(unfold=0.0))
        
        # select path for the generated images on disk
        image_path = '/Users/marcoaqil/PRFMapping/Figures/'
        
        # pattern of the saved images names
        file_pattern = "{base}_{view}_{surface}.png"
        
        # utility functions to set the different views
        prefix = dict(altitude='camera.', azimuth='camera.',
                      pivot='surface.{subject}.', radius='camera.', target='camera.',
                      unfold='surface.{subject}.')
        _tolists = lambda p: {prefix[k]+k:[v] for k,v in p.items()}
        _combine = lambda a,b: ( lambda c: [c, c.update(b)][0] )(dict(a))
        
        
        # Save images by iterating over the different views and surfaces
        for view,vparams in views.items():
            for surf,sparams in surfaces.items():
                # Combine basic, view, and surface parameters
                params = _combine(vparams, sparams)
                # Set the view
                print(params)
                time.sleep(5)
                js_handle._set_view(**_tolists(params))
                time.sleep(5)
                # Save image
                filename = file_pattern.format(base=base_str, view=view, surface=surf)
                output_path = os.path.join(image_path, filename)
                js_handle.getImage(output_path, size =(3200, 2800))
        
                # the block below trims the edges of the image:
                # wait for image to be written
                while not os.path.exists(output_path):
                    pass
                time.sleep(0.5)
                try:
                    import subprocess
                    subprocess.call(["convert", "-trim", output_path, output_path])
                except:
                    pass
                
    def project_to_fsaverage(self, models, parameters, space_names = 'all', analysis_names = 'all', subject_ids='all',
                             hcp_atlas_mask_path = None,
                             hcp_cii_file_path = None, hcp_old_sphere = None, hcp_new_sphere = None, hcp_old_area = None, hcp_new_area = None,
                             hcp_temp_folder = None):
        if 'fsaverage' not in self.main_dict:
            self.main_dict['fsaverage'] = dd(lambda:dd(lambda:dd(lambda:dd(dict))))
        
        if parameters[0] != 'RSq':
            parameters.insert(0,'RSq')
            
        if space_names == 'all':
            spaces = [item for item in self.main_dict.items()]
        else:
            spaces = [item for item in self.main_dict.items() if item[0] in space_names] 
            
        for space, space_res in spaces:
            
            if analysis_names == 'all':
                analyses = space_res.items()
            else:
                analyses = [item for item in space_res.items() if item[0] in analysis_names] 
            for analysis, analysis_res in analyses:       
                if subject_ids == 'all':
                    subjects = [item for item in analysis_res.items()]
                else:
                    subjects = [item for item in analysis_res.items() if item[0] in subject_ids]
                    
                
                for model in models:
                    #fsaverage_rsq = dict()
                    for parameter in parameters:
                        
                        fsaverage_param = dict()
                        
                        for subj, subj_res in subjects:
                            print(space+" "+analysis+" "+subj)
                            p_r = subj_res['Processed Results']
                            
                            if space == 'fsnative':
                            
                                lh_c = read_morph_data(opj(self.fs_dir, f"{subj}/surf/lh.curv"))
                
                                param = np.copy(p_r[parameter][model])
                                
                                lh_file_path = opj(self.fs_dir, f"{subj}/surf/lh.{''.join(filter(str.isalnum, parameter))}_{model}")
                                rh_file_path = opj(self.fs_dir, f"{subj}/surf/rh.{''.join(filter(str.isalnum, parameter))}_{model}")
    
                                write_morph_data(lh_file_path, param[:lh_c.shape[0]])
                                write_morph_data(rh_file_path, param[lh_c.shape[0]:])
                                
                                rh_fsaverage_path = f"{rh_file_path.replace(subj,'fsaverage')}_{subj}"
                                lh_fsaverage_path = f"{lh_file_path.replace(subj,'fsaverage')}_{subj}"
                                
                                os.system("export FREESURFER_HOME=/Applications/freesurfer/7.1.0/")
                                os.system("source $FREESURFER_HOME/SetUpFreeSurfer.sh")
                                os.system(f"export SUBJECTS_DIR={self.fs_dir}")
                                os.system(f"mri_surf2surf --srcsubject {subj} --srcsurfval {lh_file_path} --trgsubject fsaverage --trgsurfval {lh_fsaverage_path} --hemi lh --trg_type curv")
                                os.system(f"mri_surf2surf --srcsubject {subj} --srcsurfval {rh_file_path} --trgsubject fsaverage --trgsurfval {rh_fsaverage_path} --hemi rh --trg_type curv")
    
                                lh_fsaverage_param = read_morph_data(lh_fsaverage_path)
                                rh_fsaverage_param = read_morph_data(rh_fsaverage_path)
                                
                                fsaverage_param[subj] = np.concatenate((lh_fsaverage_param,rh_fsaverage_param))
                                self.main_dict['fsaverage'][analysis][subj]['Processed Results'][parameter][model] = np.copy(fsaverage_param[subj])
                                
                                #if parameter == 'RSq':
                                #    fsaverage_rsq[subj] = np.nan_to_num(np.concatenate((lh_fsaverage_param,rh_fsaverage_param)))
                                #    fsaverage_rsq[subj][fsaverage_rsq[subj]<0] = 0
                            elif space == 'HCP':
                                f=nb.load(hcp_atlas_mask_path[0])
                                data_1 = np.array([arr.data for arr in f.darrays])[0].astype('bool')
                        
                                f=nb.load(hcp_atlas_mask_path[1])
                                data_2 = np.array([arr.data for arr in f.darrays])[0].astype('bool')
                                
                                
                                cifti_brain_model = cifti.read(hcp_cii_file_path)[1][1]
                                
                                output = np.zeros(len(cifti_brain_model))
                                output[:np.sum(data_1)] = p_r[parameter][model][:len(data_1)][data_1]
                                output[np.sum(data_1):(np.sum(data_1) + np.sum(data_2))] = p_r[parameter][model][len(data_1):][data_2]
                                
                                temp_filenames = ['temp_cii.nii', 'temp_cii_subvol.nii.gz', 'temp_gii_L.func.gii',
                                                  'temp_gii_R.func.gii', 'fsaverage_gii_L.func.gii', 'fsaverage_gii_R.func.gii']
                                temp_paths = [opj(hcp_temp_folder, el) for el in temp_filenames]
                                
                                cifti.write(temp_paths[0], output.reshape(1,-1), 
                                            (cifti.Scalar.from_names([parameter]), cifti_brain_model))
                                
                                os.system(f"wb_command -cifti-separate '{temp_paths[0]}' COLUMN -volume-all '{temp_paths[1]}' \
                                          -metric CORTEX_LEFT '{temp_paths[2]}' -metric CORTEX_RIGHT '{temp_paths[3]}'")
                                          
                                os.system(f"wb_command -metric-resample '{temp_paths[2]}' '{hcp_old_sphere.replace('?','L')}' \
                                          '{hcp_new_sphere.replace('?','L')}' ADAP_BARY_AREA '{temp_paths[4]}' \
                                          -area-metrics '{hcp_old_area.replace('?','L')}' '{hcp_new_area.replace('?','L')}'")

                                os.system(f"wb_command -metric-resample '{temp_paths[3]}' '{hcp_old_sphere.replace('?','R')}' \
                                          '{hcp_new_sphere.replace('?','R')}' ADAP_BARY_AREA '{temp_paths[5]}' \
                                          -area-metrics '{hcp_old_area.replace('?','R')}' '{hcp_new_area.replace('?','R')}'")
                                
                                a = nb.load(temp_paths[4])
                                b = nb.load(temp_paths[5])
                                
                                fsaverage_param[subj] = np.concatenate((np.array([arr.data for arr in a.darrays])[0],np.array([arr.data for arr in b.darrays])[0]))
                                self.main_dict['fsaverage'][analysis][subj]['Processed Results'][parameter][model] = np.copy(fsaverage_param[subj])
                                          
                
                                

                        fsaverage_group_average = np.nanmean([fsaverage_param[sid] for sid in fsaverage_param], axis=0)
                        
                        # for i in range(len(fsaverage_group_average)):
                        #     fsaverage_group_average[i] = weightstats.DescrStatsW(np.array([fsaverage_param[sid][i] for sid in fsaverage_param]),
                        #                                                          weights=np.array([fsaverage_rsq[sid][i] for sid in fsaverage_rsq])).mean
                        
                        self.main_dict['fsaverage'][analysis]['fsaverage']['Processed Results'][parameter][model] = np.copy(fsaverage_group_average)
 

                        
                for model in models:
                    if 'Norm' in model and 'Norm Param. B' in parameters and 'Norm Param. D' in parameters:
                        self.main_dict['fsaverage'][analysis]['fsaverage']['Processed Results']['Ratio (B/D)'][model] = self.main_dict['fsaverage'][analysis]['fsaverage']['Processed Results']['Norm Param. B'][model]/self.main_dict['fsaverage'][analysis]['fsaverage']['Processed Results']['Norm Param. D'][model]

            for model in models:
                for parameter in parameters:
                    self.main_dict['fsaverage']['Mean analysis']['fsaverage']['Processed Results'][parameter][model] = np.nanmean([self.main_dict['fsaverage'][an[0]]['fsaverage']['Processed Results'][parameter][model] for an in analyses], axis=0)
                    
                        
        return
    
    
    def quant_plots(self, x_parameter, y_parameter, rois, rsq_thresh, save_figures, figure_path,
                    space_names = 'fsnative', analysis_names = 'all', subject_ids='all', y_parameter_toplevel=None, 
                    ylim={}, xlim={}, log_yaxis=False, log_xaxis = False, nr_bins = 8, rsq_weights=True,
                    x_param_model=None, violin=False, scatter=False, diff_gauss=False, diff_gauss_x=False,
                    means_only=False, stats_on_plot=False, bin_by='size', zscore_ydata=False, zscore_xdata=False,
                    zconfint_err_alpha = None, exp_fit = False, show_legend=False, each_subj_on_group=False,
                    bold_voxel_volume = None, quantile_exclusion=0.999):
        """
        

        Parameters
        ----------
        x_parameter : TYPE
            DESCRIPTION.
        y_parameter : TYPE
            DESCRIPTION.
        rois : TYPE
            DESCRIPTION.
        rsq_thresh : TYPE
            DESCRIPTION.
        save_figures : TYPE
            DESCRIPTION.
        analysis_names : TYPE, optional
            DESCRIPTION. The default is 'all'.
        subject_ids : TYPE, optional
            DESCRIPTION. The default is 'all'.
        ylim : TYPE, optional
            DESCRIPTION. The default is None.
        x_param_model : TYPE, optional
            DESCRIPTION. The default is None.
        violin : TYPE, optional
            DESCRIPTION. The default is False.
        scatter : TYPE, optional
            DESCRIPTION. The default is False.
        diff_gauss : TYPE, optional
            DESCRIPTION. The default is False.
        means_only : bool, optional
            whether to only plot the means/distribs of y parameter, without y as function of x. The default is False.
        stats_on_plot : TYPE, optional
            DESCRIPTION. The default is False.

        Returns
        -------
        mid : TYPE
            DESCRIPTION.

        """
       
        pl.rcParams.update({'font.size': 22})
        pl.rcParams.update({'pdf.fonttype':42})
        pl.rcParams.update({'figure.max_open_warning': 0})
        pl.rcParams['axes.spines.right'] = False
        pl.rcParams['axes.spines.top'] = False
        
        base_fig_path = figure_path
        
        cmap_values = list(np.linspace(0.9, 0.0, len([r for r in rois if 'all' not in r and 'combined' not in r and 'Brain' not in r])))
        
        cmap_values += [0 for r in rois if 'all' in r or 'combined' in r or 'Brain' in r]
        
        cmap_rois = cm.get_cmap('nipy_spectral')(cmap_values)#
        
        #making black into dark gray for visualization reasons
        cmap_rois[(cmap_rois == [0,0,0,1]).sum(1)==4] = [0.33,0.33,0.33,1]

        if space_names == 'all':
            spaces = [item for item in self.main_dict.items()]
        else:
            spaces = [item for item in self.main_dict.items() if item[0] in space_names] 

        for space, space_res in spaces:
            if 'fs' in space or 'HCP' in space:
                
                if quantile_exclusion is None:
                    #surrounds larger than this are excluded from surround size calculations
                    w_max=60
                    #remove absurd suppression index values
                    supp_max=1000
                    #css max
                    css_max=1
                    #max b and d
                    bd_max=1000
                #bar or violin width
                bar_or_violin_width = 0.3

                
                        
                if analysis_names == 'all':
                    analyses = [item for item in space_res.items()]
                else:
                    analyses = [item for item in space_res.items() if item[0] in analysis_names] 
                    
                if len(analyses)>1:
                    analyses.append(('Analyses mean', {sub:{} for sub in analyses[0][1]}))
                    

                alpha = dd(lambda:dd(lambda:dd(lambda:dd(dict))))
                x_par = dd(lambda:dd(lambda:dd(lambda:dd(dict))))
                y_par = dd(lambda:dd(lambda:dd(lambda:dd(dict))))
                rsq = dd(lambda:dd(lambda:dd(lambda:dd(dict))))
    
                x_par_stats = dd(lambda:dd(lambda:dd(lambda:dd(list))))
                y_par_stats = dd(lambda:dd(lambda:dd(lambda:dd(list))))
                rsq_stats = dd(lambda:dd(lambda:dd(lambda:dd(list))))
                bootstrap_fits = dd(lambda:dd(lambda:dd(lambda:dd(list))))
                
                
                for analysis, analysis_res in analyses:       
                    if subject_ids == 'all':
                        subjects = [item for item in analysis_res.items()]
                    else:
                        subjects = [item for item in analysis_res.items() if item[0] in subject_ids]
                    
                    if len(subjects)>1:
                        subjects.append(('Group', {}))
                        
                    figure_path = base_fig_path

                    if not os.path.exists(figure_path):
                        os.makedirs(figure_path)
                    figure_path = opj(figure_path, space)
                    if not os.path.exists(figure_path):
                        os.makedirs(figure_path)
                    figure_path = opj(figure_path, analysis)
                    if not os.path.exists(figure_path):
                        os.makedirs(figure_path)                            
                    
                    upsampling_corr_factors = []
                    for subj, subj_res in subjects:
                        print(f"{space} {analysis} {subj}")
                         
                        #upsampling correction: fsnative has approximately 3 times as many datapoints as original
                        if subj != 'Group':
                            if bold_voxel_volume is not None:
                                
                                print("Make sure bold_voxel_volume is specified in mm^3")
                                
                                try:
                                    if subj.isdecimal() and space == 'HCP':
                                        pycortex_subj = '999999'
                                    else:
                                        pycortex_subj = subj
                                        
                                    aseg = nb.load(opj(cortex.database.default_filestore,pycortex_subj,'anatomicals','aseg.nii.gz'))
                                    anat_vox_vol = aseg.header.get_zooms()[0]*aseg.header.get_zooms()[1]*aseg.header.get_zooms()[2]
                                    cortex_volume = ((aseg.get_fdata()==42).sum()+(aseg.get_fdata()==3).sum())*anat_vox_vol
                                    nr_bold_voxels = cortex_volume/bold_voxel_volume
                                    nr_surf_vertices = cortex.db.get_surfinfo(pycortex_subj).data.shape
            
                                    upsampling_corr_factor = nr_surf_vertices / nr_bold_voxels
                                    
              
                                except Exception as e:
                                    print(e)
                                    print("Unable to perform upsampling correction.")
                                    upsampling_corr_factor = 1
                                    pass
                                    
                            else:
                                print("BOLD voxel volume not specified. Not performing upsampling correction.")
                                upsampling_corr_factor = 1
                            upsampling_corr_factors.append(upsampling_corr_factor)
                        else:
                            upsampling_corr_factor = np.mean(upsampling_corr_factors)
                            
                        print(f"Upsampling correction factor: {upsampling_corr_factor}")
                        
                        x_ticks=[]
                        x_labels=[]    
                        bar_position = 0
                        
                        # binned eccentricity vs other parameters relationships       
            
                        model_colors = {'Gauss':'blue','CSS':'orange','DoG':'green','Norm_abcd':'red'}
                                                

    
                        for i, roi in enumerate(rois):                              
                            for model in self.only_models:                                

                                
                                if 'mean' not in analysis or 'fsaverage' in subj:
                                    if 'sub' in subj or 'fsaverage' in subj or subj.isdecimal():
                                        if 'rsq' in y_parameter.lower():
                                            #comparing same vertices for model performance
                                            curr_alpha = subj_res['Processed Results']['Alpha']['all']>rsq_thresh
                                        else:
                                            #otherwise model-specific alpha
                                            curr_alpha = (subj_res['Processed Results']['Alpha'][model]>rsq_thresh)
                                            
                                        if roi in self.idx_rois[subj]:
    
                                            alpha[analysis][subj][model][roi] = (roi_mask(self.idx_rois[subj][roi], curr_alpha)) 
       
                                        else:
                                            #if ROI is not defined
                                            #if Brain use all available vertices
                                            if roi == 'Brain':
                                                alpha[analysis][subj][model][roi] = curr_alpha
                                            elif roi == 'combined':
                                                alpha[analysis][subj][model][roi] = (roi_mask(np.concatenate(tuple([self.idx_rois[subj][r] for r in rois if ('combined' not in r and 'Brain' not in r and r in self.idx_rois[subj])])), curr_alpha))    
                                            elif roi == 'all_custom':
                                                alpha[analysis][subj][model][roi] = (roi_mask(np.concatenate(tuple([self.idx_rois[subj][r] for r in self.idx_rois[subj] if 'custom' in r])), curr_alpha))    
                                            elif space == 'fsaverage' and roi in self.idx_rois['fsaverage']:
                                                alpha[analysis][subj][model][roi] = (roi_mask(self.idx_rois['fsaverage'][roi], curr_alpha))
                                            else:
                                                #, otherwise none
                                                print(f"{roi}: undefined ROI")
                                                alpha[analysis][subj][model][roi] = np.zeros_like(curr_alpha).astype('bool')
                                        
                                        
                                        #manual exclusion of outliers
                                        if quantile_exclusion is None:
                                            print("Using manual exclusion, see quant_plots function. Set quantile_exclusion=1 for no exclusion.")
                                            if y_parameter == 'Surround Size (fwatmin)' or x_parameter == 'Surround Size (fwatmin)':# and model == 'DoG':
                                                #exclude too large surround (no surround)
                                                alpha[analysis][subj][model][roi] *= (subj_res['Processed Results']['Surround Size (fwatmin)'][model]<w_max)
                                                if x_param_model!=None:
                                                    alpha[analysis][subj][model][roi] *= (subj_res['Processed Results']['Surround Size (fwatmin)'][x_param_model]<w_max)
    
                                            
                                            if y_parameter == 'Surround/Centre Amplitude'  or x_parameter == 'Surround/Centre Amplitude' :# and model == 'DoG':
                                                #exclude too large surround (no surround)
                                                if x_param_model is not None:
                                                    alpha[analysis][subj][model][roi] *= (subj_res['Processed Results']['Surround/Centre Amplitude'][x_param_model]<w_max)
                                                else:
                                                    alpha[analysis][subj][model][roi] *= (subj_res['Processed Results']['Surround/Centre Amplitude'][model]<w_max)
    
                                                    
                                            if 'Suppression' in y_parameter:
                                                #exclude nonsensical suppression index values
                                                alpha[analysis][subj][model][roi] *= (subj_res['Processed Results'][y_parameter][model]<supp_max)                                            
                                            if 'Suppression' in x_parameter:
                                                #exclude nonsensical suppression index values
                                                alpha[analysis][subj][model][roi] *= (subj_res['Processed Results'][x_parameter][model]<supp_max)
                                                if x_param_model is not None:
                                                    alpha[analysis][subj][model][roi] *= (subj_res['Processed Results'][x_parameter][x_param_model]<supp_max)
                                                    
                                            if 'CSS Exponent' in x_parameter:
                                                alpha[analysis][subj][model][roi] *= (subj_res['Processed Results'][x_parameter][x_param_model]<css_max)
                                            
                                            if 'Norm Param.' in y_parameter:
                                                alpha[analysis][subj][model][roi] *= (subj_res['Processed Results'][y_parameter][model]<bd_max)                                            
                                            if 'Norm Param.' in x_parameter:
                                                alpha[analysis][subj][model][roi] *= (subj_res['Processed Results'][x_parameter][model]<bd_max)                                            
                                                if x_param_model is not None:
                                                    alpha[analysis][subj][model][roi] *= (subj_res['Processed Results'][x_parameter][x_param_model]<bd_max)
                                        
                                        else:
                                            print(np.sum(alpha[analysis][subj][model][roi]))
                                            print(np.nanquantile(subj_res['Processed Results'][y_parameter][model],1-quantile_exclusion))
                                            print(np.nanquantile(subj_res['Processed Results'][y_parameter][model],quantile_exclusion))
                                                
                                            alpha[analysis][subj][model][roi] *= (subj_res['Processed Results'][y_parameter][model]<np.nanquantile(subj_res['Processed Results'][y_parameter][model],quantile_exclusion))*(subj_res['Processed Results'][y_parameter][model]>np.nanquantile(subj_res['Processed Results'][y_parameter][model],1-quantile_exclusion))  
                                            print(np.sum(alpha[analysis][subj][model][roi]))
                                            if x_param_model is not None:
                                                print(subj_res['Processed Results'][x_parameter][x_param_model])
                                                alpha[analysis][subj][model][roi] *= (subj_res['Processed Results'][x_parameter][x_param_model]<np.nanquantile(subj_res['Processed Results'][x_parameter][x_param_model],quantile_exclusion))*(subj_res['Processed Results'][x_parameter][x_param_model]>np.nanquantile(subj_res['Processed Results'][x_parameter][x_param_model],1-quantile_exclusion))  
                                                print(np.sum(alpha[analysis][subj][model][roi]))
                                            else:
                                                alpha[analysis][subj][model][roi] *= (subj_res['Processed Results'][x_parameter][model]<np.nanquantile(subj_res['Processed Results'][x_parameter][model],quantile_exclusion))*(subj_res['Processed Results'][x_parameter][model]>np.nanquantile(subj_res['Processed Results'][x_parameter][model],1-quantile_exclusion))  

                                            
                                            
         
                                        #if 'ccrsq' in y_parameter.lower():
                                        #    alpha[analysis][subj][model][roi] *= (subj_res['Processed Results'][y_parameter][model]>0)
                                        if y_parameter_toplevel is None:
                                            alpha[analysis][subj][model][roi] *= np.isfinite(subj_res['Processed Results'][y_parameter][model])
                                        else:
                                            alpha[analysis][subj][model][roi] *= np.isfinite(subj_res['Processed Results'][y_parameter_toplevel][y_parameter])
                                       
                                        if x_param_model is not None:
                                            
                                            alpha[analysis][subj][model][roi] *= np.isfinite(subj_res['Processed Results'][x_parameter][x_param_model])
                                            
                                            x_par[analysis][subj][model][roi] = subj_res['Processed Results'][x_parameter][x_param_model][alpha[analysis][subj][model][roi]]                                          
                                        
                                        
                                        else:
                                            #remove nans and infinities
                                            alpha[analysis][subj][model][roi] *= np.isfinite(subj_res['Processed Results'][x_parameter][model])                                    

                                            x_par[analysis][subj][model][roi] = subj_res['Processed Results'][x_parameter][model][alpha[analysis][subj][model][roi]]
                                            
                                        #handling special case of plotting receptors as y-parameter, since they are not part of any model
                                        if y_parameter_toplevel == None:
                                            y_par[analysis][subj][model][roi] = (subj_res['Processed Results'][y_parameter][model][alpha[analysis][subj][model][roi]])
                                        else:
                                            y_par[analysis][subj][model][roi] = (subj_res['Processed Results'][y_parameter_toplevel][y_parameter][alpha[analysis][subj][model][roi]])
                                            
                                        
                                        
                                        if diff_gauss:
                                            y_par[analysis][subj][model][roi] -= subj_res['Processed Results'][y_parameter]['Gauss'][alpha[analysis][subj][model][roi]]
                                        
                                        if diff_gauss_x:
                                            x_par[analysis][subj][model][roi] -= subj_res['Processed Results'][x_parameter]['Gauss'][alpha[analysis][subj][model][roi]]
                                        
                                            

                                            #set negative ccrsq and rsq to zero
                                            #if 'ccrsq' in y_parameter.lower():
                                            #    y_par[analysis][subj][model][roi][y_par[analysis][subj][model][roi]<0] = 0
                 
    
                                        #r - squared weighting
                                        
                                        #no need for rsq-weighting if plotting rsq
                                        if 'rsq' in y_parameter.lower():
                                            rsq[analysis][subj][model][roi] = np.ones_like(y_par[analysis][subj][model][roi])
                                            
                                            #rsq[analysis][subj][model][roi] = np.copy(subj_res['Processed Results']['Noise Ceiling']['Noise Ceiling (RSq)'][alpha[analysis][subj][model][roi]])
                                            #rsq[analysis][subj][model][roi][~np.isfinite(subj_res['Processed Results']['Noise Ceiling']['Noise Ceiling (RSq)'][alpha[analysis][subj][model][roi]])] = 0
                                            #rsq[analysis][subj][model][roi][subj_res['Processed Results']['Noise Ceiling']['Noise Ceiling (RSq)'][alpha[analysis][subj][model][roi]]<0] = 0
                                            
                                            #rsq[analysis][subj][model][roi] = np.mean([subj_res['Processed Results']['RSq'][mod][alpha[analysis][subj][model][roi]] for mod in self.only_models],axis=0)
    
                                            
                                            
                                        else:
                                            rsq[analysis][subj][model][roi] = np.copy(subj_res['Processed Results']['RSq'][model][alpha[analysis][subj][model][roi]])
                                            
                                            #if plotting different model parameters use mean-rsquared as weighting
                                            if x_param_model != None and x_param_model in subj_res['Processed Results']['RSq']:
                                                rsq_x_param = np.copy(subj_res['Processed Results']['RSq'][x_param_model][alpha[analysis][subj][model][roi]])
                                                rsq_x_param[subj_res['Processed Results']['RSq'][x_param_model][alpha[analysis][subj][model][roi]]<0] = 0
                                                
                                                rsq[analysis][subj][model][roi] += rsq_x_param
                                                rsq[analysis][subj][model][roi] /= 2
                                        
                                        #x_par[analysis][subj][model][roi] =  zscore(x_par[analysis][subj][model][roi])
                                        #y_par[analysis][subj][model][roi] = zscore(y_par[analysis][subj][model][roi])
                                        #rsq[analysis][subj][model][roi]
                                                
                                        
                                        
        
                                    elif len(subjects)>1 and 'fsaverage' not in subjects:
                                        #group stats
                                        x_par_group = np.concatenate(tuple([x_par[analysis][sid][model][roi] for sid in x_par[analysis] if 'Group' not in sid]))
                                        
                                        y_par_group = np.concatenate(tuple([y_par[analysis][sid][model][roi] for sid in y_par[analysis] if 'Group' not in sid]))
                                        rsq_group = np.concatenate(tuple([rsq[analysis][sid][model][roi] for sid in rsq[analysis] if 'Group' not in sid]))
                                        
                                        x_par[analysis][subj][model][roi] = np.copy(x_par_group)
                                        y_par[analysis][subj][model][roi] = np.copy(y_par_group)
                                        rsq[analysis][subj][model][roi] = np.copy(rsq_group)
                                    
                                elif 'fsaverage' not in subjects:
                                    #mean analysis
                                    ans = [an[0] for an in analyses if 'mean' not in an[0]]
                                    alpha[analysis][subj][model][roi] = np.hstack(tuple([alpha[an][subj][model][roi] for an in ans]))
                                    x_par[analysis][subj][model][roi] = np.hstack(tuple([x_par[an][subj][model][roi] for an in ans]))
                                    y_par[analysis][subj][model][roi] = np.hstack(tuple([y_par[an][subj][model][roi] for an in ans]))
                                    rsq[analysis][subj][model][roi] = np.hstack(tuple([rsq[an][subj][model][roi] for an in ans]))
                        
                                    
                                
                                
                                
                                if zscore_xdata:
                                    x_par[analysis][subj][model][roi] = zscore(x_par[analysis][subj][model][roi])
                                if zscore_ydata:
                                    y_par[analysis][subj][model][roi] = zscore(y_par[analysis][subj][model][roi])
                                
                                    
                                
                                if not rsq_weights:
                                    rsq[analysis][subj][model][roi] = np.ones_like(rsq[analysis][subj][model][roi])
                            

                        for i, roi in enumerate([r for r in rois if 'all' not in r and 'combined' not in r and 'Brain' not in r]):
                            
                            samples_in_roi = len(y_par[analysis][subj][model][roi])
                            print(f"Samples in ROI: {samples_in_roi}")
                            
                            if i>0:
                                bar_position += (2*bar_or_violin_width)
                                
                            label_position = bar_position+(0.5*bar_or_violin_width)*(len(self.only_models)-1)                     
                            if 'hV4' in roi and len(self.only_models)>1:
                                label_position+=bar_or_violin_width
                                
                            x_ticks.append(label_position)
                            x_labels.append(roi.replace('custom.','').replace('HCPQ1Q6.','')+'\n')   
                            
                            for model in self.only_models:    
                                                                
                                pl.figure(f"{analysis} {subj} Mean {y_parameter}", figsize=(8, 8), frameon=True)
                                
                                if log_yaxis:
                                    pl.gca().set_yscale('log')
                                
                                if 'Eccentricity' in y_parameter or 'Size' in y_parameter:
                                    pl.ylabel(f"{subj} Mean {y_parameter} (°)")
                                elif 'B/D' in y_parameter:
                                    pl.ylabel(f"{subj} Mean {y_parameter} (% signal change)")
                                else:
                                    pl.ylabel(f"{subj} Mean {y_parameter}")
                                    
                                if 'Mean' in xlim:
                                    pl.xlim(xlim['Mean'])
                                    
                                if 'Mean' in ylim:
                                    pl.ylim(ylim['Mean'])
                                    
                                full_roi_stats = weightstats.DescrStatsW(y_par[analysis][subj][model][roi],
                                                            weights=rsq[analysis][subj][model][roi])
                                bar_height = full_roi_stats.mean
                                
                                if zconfint_err_alpha is not None:                                    
                                    bar_err = (np.abs(full_roi_stats.zconfint_mean(alpha=zconfint_err_alpha) - bar_height)).reshape(2,1)*upsampling_corr_factor**0.5                                    
                                else:
                                    bar_err = full_roi_stats.std_mean*upsampling_corr_factor**0.5
                                    


                                if len(self.only_models)>1:
                                    if violin:
                                        viol_plot = pl.violinplot(y_par[analysis][subj][model][roi], [bar_position],
                                                      widths=[bar_or_violin_width], showextrema=False, showmeans=True, showmedians=True)
                                        
                                        for viol in viol_plot['bodies']:
                                            viol.set_facecolor(model_colors[model])
                                            viol.set_edgecolor('black')
                                            viol.set_alpha(1.0)
                                            
                                        viol_plot['cmeans'].set_color('black')
                                        viol_plot['cmedians'].set_color('white')
                                        
                                        #boxplots
                                        
                                        # bp = pl.boxplot(y_par[analysis][subj][model][roi], positions=[bar_position], showfliers=False, showmeans=True,
                                        #                 widths=bar_or_violin_width, meanline=True, patch_artist=True)
                                        # for box in bp['boxes']:
                                        #     # change outline color
                                        #     box.set(color='black')
                                        #     # change fill color
                                        #     box.set(facecolor = model_colors[model])
                                        
                                        # ## change color and linewidth of the whiskers
                                        # for whisker in bp['whiskers']:
                                        #     whisker.set(color=model_colors[model])
                                        
                                        # ## change color and linewidth of the caps
                                        # for cap in bp['caps']:
                                        #     cap.set(color='black')
                                        
                                        # ## change color and linewidth of the medians
                                        # for median in bp['medians']:
                                        #     median.set(color='white')
                                            
                                        # for mean in bp['means']:
                                        #     mean.set(color='black')
                                            
                                            
                                        bar_height = y_par[analysis][subj][model][roi].max()
                                    else:
                                        if subj == 'Group' and each_subj_on_group:
                                            bar_err = None
                                        
                                        pl.bar(bar_position, bar_height, width=bar_or_violin_width, yerr=bar_err, 
                                               edgecolor=model_colors[model], label=model, color=model_colors[model])
                                        
                                        if subj == 'Group' and each_subj_on_group:
                                            ssj_datapoints_x = np.linspace(bar_position-0.66*bar_or_violin_width, bar_position+0.66*bar_or_violin_width, len(subjects)-1)
                                            for ssj_nr, ssj in enumerate(subjects[:-1]):
                                                
                                                ssj_stats = weightstats.DescrStatsW(y_par[analysis][ssj[0]][model][roi],
                                                            weights=rsq[analysis][ssj[0]][model][roi])
                                                
                                                if zconfint_err_alpha is not None:
                                                    yerr_sj = (np.abs(ssj_stats.zconfint_mean(alpha=zconfint_err_alpha) - ssj_stats.mean)).reshape(2,1)*upsampling_corr_factor**0.5
                                                else:
                                                    yerr_sj = ssj_stats.std_mean*upsampling_corr_factor**0.5
                                                    
                                                pl.errorbar(ssj_datapoints_x[ssj_nr], ssj_stats.mean,
                                                yerr=yerr_sj,  
                                                fmt='s',  mec='k', color=model_colors[model], ecolor='k')
                                    

                                    
                                    
                                    
                                    if stats_on_plot:
                                        if diff_gauss:
                                            base_model = 'Gauss'
                                        else:
                                            base_model = [m for m in self.only_models if 'Norm' in m][0]

                                        
                                        #do model comparison stats only once, at the last model
                                        if self.only_models.index(model) == (len(self.only_models)-1):

                                            if violin:
                                                text_height = np.max([y_par[analysis][subj][m][roi].max() for m in self.only_models])
                                            else:
                                                text_height = np.max([y_par[analysis][subj][m][roi].mean()+sem(y_par[analysis][subj][m][roi])*upsampling_corr_factor for m in self.only_models])
                                            
                                            y1, y2 = pl.gca().get_window_extent().get_points()[:, 1]
                                            window_size_points = y2-y1
                                            
                                            if 'Mean' in ylim:
                                                axis_height = ylim['Mean'][1]-ylim['Mean'][0]
                                                #16 is font size
                                                text_distance = 1.5*(16*axis_height)/window_size_points
                                            else:
                                                axis_height = np.max([[y_par[analysis][subj][m][r].mean()+sem(y_par[analysis][subj][m][r])*upsampling_corr_factor for m in self.only_models] for r in rois])-\
                                                                np.min([[y_par[analysis][subj][m][r].mean()-sem(y_par[analysis][subj][m][r])*upsampling_corr_factor for m in self.only_models] for r in rois])
                                                #a bit more distance since axis_height is an approximation
                                                text_distance = 2*(16*axis_height)/window_size_points
     
                                            


                                            for mod in [m for m in self.only_models if base_model != m]:
                                                
                                                
                                                diff = y_par[analysis][subj][base_model][roi] - y_par[analysis][subj][mod][roi]
                                                
                                                pvals = []
                                                
                                                for ccc in range(10000):                                         
                                                                                                               
                                                    #if ccc<50:
                                                    #    pl.figure(f"null distribs {analysis} {subj} {mod} {roi}")
                                                    #    pl.hist(null_distrib, bins=50, color=cmap_rois[i])
                                                    
                                                    
                                                    #correct for upsampling
                                                    samp_idx = np.random.randint(0, len(diff), int(len(diff)/upsampling_corr_factor))

                                                    observ = diff[samp_idx]
                                                    
                                                    null_distrib = np.sign(np.random.rand(len(diff[samp_idx]))-0.5)*diff[samp_idx]
                                                    
                                                    if diff_gauss:
                                                        #test whether other models improve on gauss
                                                        pvals.append(null_distrib.mean() <= observ.mean())
                                                    else:
                                                        #test whether norm improves over other models
                                                        pvals.append(null_distrib.mean() >= observ.mean())
                                                        
                                                    #pvals.append(wilcoxon(observ, null_distrib, alternative='greater')[1])
                                                    #pvals.append(ks_2samp(observ, null_distrib, alternative='less')[1])
    
                                                #pl.figure(f"{analysis} {subj} Mean {y_parameter}", figsize=(8, 8), frameon=True) 
                                                        
                                                    
                                                pval = np.mean(pvals) 
                                                print(f"{mod} pval: {pval}")
                                                
                                                pval_str = ""
                                                
                                                #compute p values
                                                if pval<0.01:
                                                    if diff_gauss:
                                                        text_color = model_colors[mod]
                                                    else:
                                                        text_color = model_colors[base_model]
                                                    pval_str+="*"
                                                    if pval<1e-4:
                                                        pval_str+="*"
                                                        if pval<1e-6:
                                                            pval_str+="*"
                                                elif pval>0.99:
                                                    if diff_gauss:
                                                        text_color = model_colors[base_model]        
                                                    else:
                                                        text_color = model_colors[mod]                                                    
                                                    pval_str+="*"
                                                    if pval>(1-1e-4):
                                                        pval_str+="*"
                                                        if pval>(1-1e-6):
                                                            pval_str+="*"                                                    


                                                def plot_comparison_bracket():
                                                    #dh = text_distance
                                                    
                                                    barh = text_distance/2
                                                    y = max(ly, ry) + barh
                                                    #barx = [lx, lx, rx, rx]
                                                    #bary = [y, y+barh, y+barh, y]
                                                    mid = ((lx+rx)/2, y+0.5*barh)
                                                    
                                                    #pl.plot(barx, bary, c='black')
                                                    pl.plot([lx,lx], [y,y+barh], c=c_left)
                                                    pl.plot([rx,rx], [y,y+barh], c=c_right)
                                                    pl.plot([lx,rx],[y+barh, y+barh], c='black')
                                                    return mid

                                                if mod == 'CSS':
                                                    if diff_gauss:
                                                        css_text_pos = text_height+1.5*text_distance
                                                        c_left = 'blue'
                                                        c_right = 'orange'
                                                        lx, ly = bar_position-3*bar_or_violin_width, css_text_pos
                                                        rx, ry = bar_position-bar_or_violin_width, css_text_pos
                                                    else:
                                                        css_text_pos = text_height
                                                        c_left = 'orange'
                                                        c_right = 'red'                                                    
                                                        lx, ly = bar_position-bar_or_violin_width, css_text_pos
                                                        rx, ry = bar_position, css_text_pos
                                                        
                                                    
                                                    
                                                elif mod == 'DoG':
                                                    if diff_gauss:
                                                        dog_text_pos = text_height
                                                        c_left = 'blue'
                                                        c_right = 'green'
                                                        lx, ly = bar_position-3*bar_or_violin_width, dog_text_pos
                                                        rx, ry = bar_position-2*bar_or_violin_width, dog_text_pos
                                                    else:
                                                        dog_text_pos = text_height+1.5*text_distance     
                                                        c_left = 'green'
                                                        c_right = 'red'
                                                        lx, ly = bar_position-2*bar_or_violin_width, dog_text_pos
                                                        rx, ry = bar_position, dog_text_pos                                                        
                                                                                                     
                                                    
                                                    
                                                elif mod == 'Gauss' or 'Norm' in mod:
                                                    c_left = 'blue'
                                                    c_right = 'red'
                                                    lx, ly = bar_position-3*bar_or_violin_width, text_height+3*text_distance  
                                                    rx, ry = bar_position, text_height+3*text_distance                                                    
                                                    
                                                mid = plot_comparison_bracket()
                                                    
                                                pl.text(*mid, pval_str, fontsize=16, color=text_color, weight = 'bold', ha='center', va='bottom')
                               
                                    
                                else:
                                    if violin:
                                        try:
                                            viol_plot = pl.violinplot(y_par[analysis][subj][model][roi], [bar_position],
                                                          widths=[bar_or_violin_width], showextrema=False, showmeans=True, showmedians=True)
                                            
                                            for viol in viol_plot['bodies']:
                                                viol.set_facecolor(cmap_rois[i])
                                                viol.set_edgecolor('black')
                                                viol.set_alpha(1.0)
                                                
                                            viol_plot['cmeans'].set_color('black')
                                            viol_plot['cmedians'].set_color('white')
                                            
                                            
                                            #boxplot
                                            # bp = pl.boxplot(y_par[analysis][subj][model][roi], positions=[bar_position], showfliers=False, showmeans=True,
                                            #                 widths=bar_or_violin_width, meanline=True, patch_artist=True)
                                            # for box in bp['boxes']:
                                            #     # change outline color
                                            #     box.set(color='black')
                                            #     # change fill color
                                            #     box.set(facecolor = cmap_rois[i])
                                            
                                            # ## change color and linewidth of the whiskers
                                            # for whisker in bp['whiskers']:
                                            #     whisker.set(color=model_colors[model])
                                            
                                            # ## change color and linewidth of the caps
                                            # for cap in bp['caps']:
                                            #     cap.set(color='black')
                                            
                                            # ## change color and linewidth of the medians
                                            # for median in bp['medians']:
                                            #     median.set(color='white')
                                                
                                            # for mean in bp['means']:
                                            #     mean.set(color='black')                                            
                                            
                                            
                                        except:
                                            pass
                                                    
                                    else:
                                        if subj == 'Group' and each_subj_on_group:
                                            bar_err = None
                                            
                                        pl.bar(bar_position, bar_height, width=bar_or_violin_width, yerr=bar_err, 
                                               edgecolor=cmap_rois[i], color=cmap_rois[i])
                                        
                                        if subj == 'Group' and each_subj_on_group:
                                            ssj_datapoints_x = np.linspace(bar_position-0.66*bar_or_violin_width, bar_position+0.66*bar_or_violin_width, len(subjects)-1)
                                            for ssj_nr, ssj in enumerate(subjects[:-1]):
                                                
                                                ssj_stats = weightstats.DescrStatsW(y_par[analysis][ssj[0]][model][roi],
                                                            weights=rsq[analysis][ssj[0]][model][roi])
                                                
                                                if zconfint_err_alpha is not None:
                                                    yerr_sj = (np.abs(ssj_stats.zconfint_mean(alpha=zconfint_err_alpha) - ssj_stats.mean)).reshape(2,1)*upsampling_corr_factor**0.5
                                                else:
                                                    yerr_sj = ssj_stats.std_mean*upsampling_corr_factor**0.5
                                                    
                                                pl.errorbar(ssj_datapoints_x[ssj_nr], ssj_stats.mean,
                                                yerr=yerr_sj,   
                                                fmt='s',  mec='k', color=cmap_rois[i], ecolor='k')                                        
                                
                                bar_position += bar_or_violin_width

                                pl.xticks(x_ticks,x_labels)
                                
                                # handles, labels = pl.gca().get_legend_handles_labels()
                                # by_label = dict(zip(labels, handles))
                                # if len(self.only_models) == 1:
                                #     pl.legend(by_label.values(), by_label.keys())
                                    
                                if save_figures:

                                    pl.savefig(opj(figure_path, f"{subj} {model} Mean {y_parameter.replace('/','')}.pdf"), dpi=600, bbox_inches='tight')
                                                          

                        
                        if not means_only:
                            for i, roi in enumerate(rois):
                                ###################
                                #x vs y param by ROI
                                pl.figure(f"{analysis} {subj} {roi.replace('custom.','').replace('HCPQ1Q6.','')} {y_parameter} VS {x_parameter}", figsize=(8, 8), frameon=True)
                                if log_yaxis:
                                    pl.gca().set_yscale('log')
                                if log_xaxis:
                                    pl.gca().set_xscale('log')
                                
                                if roi in ylim:
                                    pl.ylim(ylim[roi])
                                if roi in xlim:
                                    pl.xlim(xlim[roi])
                                    
                                for model in self.only_models:
                                    #bin stats 
                                    x_par_sorted = np.argsort(x_par[analysis][subj][model][roi])
                                    
                                    
                                    try:
                                        
                                        if bin_by == 'space':
                                        #equally spaced bins
                                            x_par_range = np.linspace(np.nanquantile(x_par[analysis][subj][model][roi], 0.05), np.nanquantile(x_par[analysis][subj][model][roi], 0.95), nr_bins)
                                            split_x_par_bins = np.array_split(x_par_sorted, [np.nanargmin(np.abs(el-np.sort(x_par[analysis][subj][model][roi]))) for el in x_par_range])
                                        elif bin_by == 'size':
                                        #equally sized bins
                                            split_x_par_bins = np.array_split(x_par_sorted, nr_bins)
                                    
                                    
                                        for x_par_quantile in split_x_par_bins:
                                            #ddof_correction_quantile = ddof_corr*np.sum(rsq[analysis][subj][model][roi][x_par_quantile])
                                            
                                            y_par_stats[analysis][subj][model][roi].append(weightstats.DescrStatsW(y_par[analysis][subj][model][roi][x_par_quantile],
                                                                                                  weights=rsq[analysis][subj][model][roi][x_par_quantile]))
                    
                                            x_par_stats[analysis][subj][model][roi].append(weightstats.DescrStatsW(x_par[analysis][subj][model][roi][x_par_quantile],
                                                                                                  weights=rsq[analysis][subj][model][roi][x_par_quantile]))
                                            
                                            rsq_stats[analysis][subj][model][roi].append(weightstats.DescrStatsW(rsq[analysis][subj][model][roi][x_par_quantile],
                                                                                                  weights=np.ones_like(rsq[analysis][subj][model][roi][x_par_quantile])))
                                    
                                        curr_x_bins = np.array([ss.mean for ss in x_par_stats[analysis][subj][model][roi]])
                                        curr_y_bins = np.array([ss.mean for ss in y_par_stats[analysis][subj][model][roi]])
                                    except:
                                        pass    
    
                                    if not scatter:                
                                        try:
                                            WLS = LinearRegression()
                                            WLS.fit(x_par[analysis][subj][model][roi].reshape(-1, 1), y_par[analysis][subj][model][roi], sample_weight=rsq[analysis][subj][model][roi])

                                            wls_score = WLS.score(x_par[analysis][subj][model][roi].reshape(-1, 1),
                                                                  y_par[analysis][subj][model][roi].reshape(-1, 1),
                                                                sample_weight=rsq[analysis][subj][model][roi].reshape(-1, 1))
                                            
                                            _, pval_lin = spearmanr(x_par[analysis][subj][model][roi],y_par[analysis][subj][model][roi])
                                            
                                            print(f"{roi} {model} WLS score {wls_score}")
                                            
                                            
                                            if exp_fit:
                                                #start points for sigmoid fits
                                                x0s = [[1,1,0.5,1],[-1,1,0.5,1],[-10,10,1,10],[10,10,1,10],[1,100,0.5,1],[-1,100,0.5,1],[-5,20,0.28,56]]
                                                curr_min_res = np.inf
                                                
                                                for x0 in x0s:
                                                                            
                                                    res = minimize(lambda x,a,y:np.sum(rsq[analysis][subj][model][roi]*(y-x[0]/(x[1]*np.exp(-a*x[2])+1)-x[3])**2), x0=x0,
                                                                   args=(x_par[analysis][subj][model][roi],
                                                                         y_par[analysis][subj][model][roi]),
                                                                   method='Powell', options={'ftol':1e-8, 'xtol':1e-8})
                                                    if res['fun']<curr_min_res:
                                                        exp_res = res
                                                        curr_min_res = res['fun']
                                                
                                                
                                                
                                                exp_pred = exp_res['x'][3]+exp_res['x'][0]/(exp_res['x'][1]*np.exp(-exp_res['x'][2]*np.linspace(curr_x_bins.min(),curr_x_bins.max(),100))+1)
                                                full_exp_pred = exp_res['x'][3]+exp_res['x'][0]/(exp_res['x'][1]*np.exp(-exp_res['x'][2]*x_par[analysis][subj][model][roi])+1)
                                                
                                                rsq_pred=1-(np.sum((y_par[analysis][subj][model][roi]-full_exp_pred)**2))/(len(y_par[analysis][subj][model][roi])*np.var(y_par[analysis][subj][model][roi]))
                                                
                                                        
                                            for c in range(200):
                                                
                                                sample = np.random.randint(0, len(x_par[analysis][subj][model][roi]), int(len(x_par[analysis][subj][model][roi])/upsampling_corr_factor))
                                                
                                                WLS_bootstrap = LinearRegression()
                                                WLS_bootstrap.fit(x_par[analysis][subj][model][roi][sample].reshape(-1, 1), y_par[analysis][subj][model][roi][sample], sample_weight=rsq[analysis][subj][model][roi][sample])
                                                
                                                bootstrap_fits[analysis][subj][model][roi].append(WLS_bootstrap.predict(curr_x_bins.reshape(-1, 1)))
                                        

                                            if len(self.only_models)>1:
                                                current_color = model_colors[model]
                                                current_label = model
                                            else:
                                                current_color = cmap_rois[i]
                                                current_label = roi.replace('custom.','').replace('HCPQ1Q6.','')
                                                
                                                
                                                
                                            pl.plot(curr_x_bins,
                                                WLS.predict(curr_x_bins.reshape(-1, 1)),
                                                color=current_color, label=f"Lin. R2={wls_score:.2f}")# p={pval_lin:.2e}")
                                            
                                            if exp_fit:
                                                pl.plot(np.linspace(curr_x_bins.min(),curr_x_bins.max(),100), exp_pred, 
                                                        color=current_color, label=f"Sigm. R2={rsq_pred:.2f}", ls='--')
                                      
                                            #conf interval shading s
                                            pl.fill_between(curr_x_bins,
                                                        np.min(bootstrap_fits[analysis][subj][model][roi], axis=0),
                                                        np.max(bootstrap_fits[analysis][subj][model][roi], axis=0),
                                                        alpha=0.2, color=current_color, label=f"Lin. R2={wls_score:.2f}")# p={pval_lin:.2e}")
                                            
                                            

                                                    
                                        except Exception as e:
                                            print(e)
                                            pass
                                        
                                        try:
                                            if zconfint_err_alpha is not None:
                                                curr_yerr = np.array([np.abs(ss.zconfint_mean(alpha=zconfint_err_alpha)-ss.mean) for ss in y_par_stats[analysis][subj][model][roi]]).T*upsampling_corr_factor**0.5
                                                curr_xerr = np.array([np.abs(ss.zconfint_mean(alpha=zconfint_err_alpha)-ss.mean) for ss in x_par_stats[analysis][subj][model][roi]]).T*upsampling_corr_factor**0.5
                                            else:
                                                curr_yerr = np.array([ss.std_mean for ss in y_par_stats[analysis][subj][model][roi]])*upsampling_corr_factor**0.5
                                                curr_xerr = np.array([ss.std_mean for ss in x_par_stats[analysis][subj][model][roi]])*upsampling_corr_factor**0.5
                                                
                                            
                                            pl.errorbar(curr_x_bins,
                                                curr_y_bins,
                                                yerr=curr_yerr,
                                                xerr=curr_xerr,
                                                fmt='s',  mec='black', label=current_label, color=current_color)#, mfc=model_colors[model], ecolor=model_colors[model])

                                           
                                                
                                        except Exception as e:
                                            print(e)
                                            pass
                                    else:
                                        try:
                                            if len(self.only_models)>1:
                                                current_color = model_colors[model]
                                                current_label = model
                                            else:
                                                current_color = cmap_rois[i]
                                                current_label = roi.replace('custom.','').replace('HCPQ1Q6.','')
                                                
                                            scatter_cmap = colors.LinearSegmentedColormap.from_list(
                                                'alpha_rsq', [(0, (*colors.to_rgb(current_color),0)), (1, current_color)])
                                                
                                            pl.scatter(x_par[analysis][subj][model][roi], y_par[analysis][subj][model][roi], marker='o', s=0.01,
                                                label=current_label, zorder=len(rois)-i, c=rsq[analysis][subj][model][roi], cmap=scatter_cmap)

                                        except Exception as e:
                                            print(e)
                                            pass                            
                            

                                if 'Eccentricity' in x_parameter or 'Size' in x_parameter:
                                    pl.xlabel(f"{x_parameter} (°)")
                                elif 'B/D' in x_parameter:
                                    pl.xlabel(f"{x_parameter} (% signal change)")                                
                                else:
                                    pl.xlabel(f"{x_parameter}")  
        
                                if x_param_model != None:
                                    pl.xlabel(f"{x_param_model} {pl.gca().get_xlabel()}")
                                
                                if 'Eccentricity' in y_parameter or 'Size' in y_parameter:
                                    pl.ylabel(f"{subj} {roi.replace('custom.','').replace('HCPQ1Q6.','')} {y_parameter} (°)")
                                elif 'B/D' in y_parameter:
                                    pl.ylabel(f"{subj} {roi.replace('custom.','').replace('HCPQ1Q6.','')} {y_parameter} (% signal change)")
                                else:
                                    pl.ylabel(f"{subj} {roi.replace('custom.','').replace('HCPQ1Q6.','')} {y_parameter}")
                                    
                                
                                    
                                if show_legend:
                                
                                    handles, labels = pl.gca().get_legend_handles_labels()
        
                                    legend_dict = dd(list)
                                    for cc, label in enumerate(labels):
                                        legend_dict[label].append(handles[cc])
                                        
                                    for label in legend_dict:
                                        legend_dict[label] = tuple(legend_dict[label])
        
                                    pl.legend([legend_dict[label] for label in legend_dict], legend_dict.keys())  
                                    
                                if save_figures:
                                    
                                    if x_param_model != None:
                                        pl.savefig(opj(figure_path, f"{subj} {roi.replace('custom.','').replace('HCPQ1Q6.','')} {y_parameter.replace('/','')} VS {x_param_model} {x_parameter.replace('/','')}.pdf"), dpi=600, bbox_inches='tight')
                                    
                                    else:      
                                        pl.savefig(opj(figure_path, f"{subj} {roi.replace('custom.','').replace('HCPQ1Q6.','')} {y_parameter.replace('/','')} VS {x_parameter.replace('/','')}.pdf"), dpi=600, bbox_inches='tight')


                        ########params by model (all rois)
                        if not means_only:        
                            for model in self.only_models:
                                
                                pl.figure(f"{analysis} {subj} {model} {y_parameter} VS {x_parameter}", figsize=(8, 8), frameon=True)
                                if log_yaxis:
                                    pl.gca().set_yscale('log')
                                if log_xaxis:
                                    pl.gca().set_xscale('log')
                                
                                if model in ylim:
                                    pl.ylim(ylim[model])
                                if model in xlim:
                                    pl.xlim(xlim[model])
                                    
                                for i, roi in enumerate([r for r in rois if 'all' not in r and 'combined' not in r and 'Brain' not in r]):
                                    if not scatter:
                                        curr_x_bins = np.array([ss.mean for ss in x_par_stats[analysis][subj][model][roi]])
                                        curr_y_bins = np.array([ss.mean for ss in y_par_stats[analysis][subj][model][roi]])
                                        try:
                                            WLS = LinearRegression()
                                            WLS.fit(x_par[analysis][subj][model][roi].reshape(-1, 1), y_par[analysis][subj][model][roi], sample_weight=rsq[analysis][subj][model][roi])
                                            
                                                                    
                                            current_color = cmap_rois[i]
                                            current_label = roi.replace('custom.','').replace('HCPQ1Q6.','')
                                                    
                                            pl.plot(curr_x_bins,
                                                WLS.predict(curr_x_bins.reshape(-1, 1)),
                                                color=current_color, label=current_label)
                                                #color=roi_colors[roi]) 

                                                                              
                                            wls_score = WLS.score(curr_x_bins.reshape(-1, 1),
                                                                  curr_y_bins.reshape(-1, 1),
                                                                sample_weight=np.array([ss.mean for ss in rsq_stats[analysis][subj][model][roi]]).reshape(-1, 1))
                                            print(f"{roi} {model} WLS score {wls_score}")
                                        except Exception as e:
                                            print(e)
                                            pass
                                        
                                        try:
                                            #conf interval shading
                                            pl.fill_between(curr_x_bins,
                                                            np.min(bootstrap_fits[analysis][subj][model][roi], axis=0),
                                                            np.max(bootstrap_fits[analysis][subj][model][roi], axis=0),
                                                            alpha=0.2, color=current_color, label=current_label)
                                            
                                            #data points with errors
                                            if zconfint_err_alpha is not None:
                                                curr_yerr = np.array([np.abs(ss.zconfint_mean(alpha=zconfint_err_alpha)-ss.mean) for ss in y_par_stats[analysis][subj][model][roi]]).T*upsampling_corr_factor**0.5
                                                curr_xerr = np.array([np.abs(ss.zconfint_mean(alpha=zconfint_err_alpha)-ss.mean) for ss in x_par_stats[analysis][subj][model][roi]]).T*upsampling_corr_factor**0.5
                                            else:
                                                curr_yerr = np.array([ss.std_mean for ss in y_par_stats[analysis][subj][model][roi]])*upsampling_corr_factor**0.5
                                                curr_xerr = np.array([ss.std_mean for ss in x_par_stats[analysis][subj][model][roi]])*upsampling_corr_factor**0.5
                                           
                                            
                                            pl.errorbar(curr_x_bins,
                                                curr_y_bins,
                                                yerr=curr_yerr,
                                                xerr=curr_xerr,
                                            fmt='s', mec='black', label=current_label, color=current_color)#, mfc=roi_colors[roi], ecolor=roi_colors[roi])
                                        except Exception as e:
                                            print(e)
                                            pass
                                    else:
                                        
                                        try:
    
                                            current_color = cmap_rois[i]
                                            current_label = roi.replace('custom.','').replace('HCPQ1Q6.','')
                                            
                                            scatter_cmap = colors.LinearSegmentedColormap.from_list(
                                                'alpha_rsq', [(0, (*colors.to_rgb(current_color),0)), (1, current_color)])
                                                
                                            pl.scatter(x_par[analysis][subj][model][roi], y_par[analysis][subj][model][roi], marker='o', s=0.01,
                                                label=current_label, zorder=len(rois)-i, c=rsq[analysis][subj][model][roi], cmap=scatter_cmap)
    
                                        except Exception as e:
                                            print(e)
                                            pass       
                                    
                                
                                
                                if 'Eccentricity' in x_parameter or 'Size' in x_parameter:
                                    pl.xlabel(f"{x_parameter} (°)")
                                elif 'B/D' in x_parameter:
                                    pl.xlabel(f"{x_parameter} (% signal change)")   
                                else:
                                    pl.xlabel(f"{x_parameter}")
                                    
                                if x_param_model != None:
                                    pl.xlabel(f"{x_param_model} {pl.gca().get_xlabel()}")                                    
                                
                                if 'Eccentricity' in y_parameter or 'Size' in y_parameter:
                                    pl.ylabel(f"{subj} {model} {y_parameter} (°)")
                                elif 'B/D' in y_parameter:
                                    pl.ylabel(f"{subj} {model} {y_parameter} (% signal change)")   
                                else:
                                    pl.ylabel(f"{subj} {model} {y_parameter}")
                                    
                                # handles, labels = pl.gca().get_legend_handles_labels()
    
                                # legend_dict = dd(list)
                                # for cc, label in enumerate(labels):
                                #     legend_dict[label].append(handles[cc])
                                    
                                # for label in legend_dict:
                                #     legend_dict[label] = tuple(legend_dict[label])
    
                                # pl.legend([legend_dict[label] for label in legend_dict], legend_dict.keys())                      
    
                                if save_figures:

                                    if x_param_model is not None:
                                        pl.savefig(opj(figure_path, f"{subj} {model} {y_parameter.replace('/','')} VS {x_param_model} {x_parameter.replace('/','')}.pdf"), dpi=600, bbox_inches='tight')
                                    else:
                                        pl.savefig(opj(figure_path,  f"{subj} {model} {y_parameter.replace('/','')} VS {x_parameter.replace('/','')}.pdf"), dpi=600, bbox_inches='tight')

        return
    
    def multidim_analysis(self, parameters, rois, rsq_thresh, save_figures, figure_path, space_names = 'fsnative',
                    analysis_names = 'all', subject_ids='all', y_parameter_toplevel=None,
                    x_dims_idx=None, y_dims_idx=None, zscore_data=False, size_response_curves = False,
                    plot_corr_matrix = False, regress_params=False, multidim_y = False, cv_regression = False,
                    zconfint_err_alpha = None, bold_voxel_volume = None, quantile_exclusion=0.999):
        
        np.set_printoptions(precision=4,suppress=True)

        if not os.path.exists(figure_path):
            os.makedirs(figure_path)
       
        pl.rcParams.update({'font.size': 28})
        pl.rcParams.update({'pdf.fonttype':42})
        pl.rcParams.update({'figure.max_open_warning': 0})
        pl.rcParams['axes.spines.right'] = False
        pl.rcParams['axes.spines.top'] = False
        
        cmap_values = list(np.linspace(0.9, 0.0, len([r for r in rois if 'custom.' in r])))
        
        cmap_values += [0 for r in rois if 'custom.' not in r]
        
        cmap_rois = cm.get_cmap('nipy_spectral')(cmap_values)#
        
        #making black into dark gray for visualization reasons
        cmap_rois[(cmap_rois == [0,0,0,1]).sum(1)==4] = [0.33,0.33,0.33,1]

        if space_names == 'all':
            spaces = [item for item in self.main_dict.items()]
        else:
            spaces = [item for item in self.main_dict.items() if item[0] in space_names] 

        for space, space_res in spaces:
        
            
            if quantile_exclusion is None:
                #surrounds larger than this are excluded from surround size calculations
                w_max=60
                #remove absurd suppression index values
                supp_max=1000
                #css max
                css_max=1
                #max b and d
                bd_max=1000
            
            dimensions = []
                    
            if analysis_names == 'all':
                analyses = [item for item in space_res.items()]
            else:
                analyses = [item for item in space_res.items() if item[0] in analysis_names] 
                
            if len(analyses)>1:
                analyses.append(('Analyses mean', {sub:{} for sub in analyses[0][1]}))
                

            alpha = dd(lambda:dd(lambda:dd(dict)))
            multidim_param_array = dd(lambda:dd(lambda:dd(list)))
            dimensions = dd(lambda:dd(lambda:dd(list)))
            
            
            response_functions = dd(list)
               
            
            for analysis, analysis_res in analyses:       
                if subject_ids == 'all':
                    subjects = [item for item in analysis_res.items()]
                else:
                    subjects = [item for item in analysis_res.items() if item[0] in subject_ids]
                
                if len(subjects)>1 and space != 'fsaverage':
                    subjects.append(('Group', {}))
                
                upsampling_corr_factors = []
                for subj, subj_res in subjects:
                    print(space+" "+analysis+" "+subj)

                    #upsampling correction: fsnative has approximately 3 times as many datapoints as original
                    if subj != 'Group':
                        if bold_voxel_volume is not None:
                            
                            print("Make sure bold_voxel_volume is specified in mm^3")
                            
                            try:
                                if subj.isdecimal() and space == 'HCP':
                                    pycortex_subj = '999999'
                                else:
                                    pycortex_subj = subj
                                    
                                aseg = nb.load(opj(cortex.database.default_filestore,pycortex_subj,'anatomicals','aseg.nii.gz'))
                                anat_vox_vol = aseg.header.get_zooms()[0]*aseg.header.get_zooms()[1]*aseg.header.get_zooms()[2]
                                cortex_volume = ((aseg.get_fdata()==42).sum()+(aseg.get_fdata()==3).sum())*anat_vox_vol
                                nr_bold_voxels = cortex_volume/bold_voxel_volume
                                nr_surf_vertices = cortex.db.get_surfinfo(pycortex_subj).data.shape
        
                                upsampling_corr_factor = nr_surf_vertices / nr_bold_voxels
                                
          
                            except Exception as e:
                                print(e)
                                print("Unable to perform upsampling correction.")
                                upsampling_corr_factor = 1
                                pass
                                
                        else:
                            print("BOLD voxel volume not specified. Not performing upsampling correction.")
                            upsampling_corr_factor = 1
                        upsampling_corr_factors.append(upsampling_corr_factor)
                    else:
                        upsampling_corr_factor = np.mean(upsampling_corr_factors)
                        
                    print(f"Upsampling correction factor: {upsampling_corr_factor}")
                  
                    
                    # binned eccentricity vs other parameters relationships       
        
                    model_colors = {'Gauss':'blue','CSS':'orange','DoG':'green','Norm_abcd':'red'}                                                

                    for i, roi in enumerate(rois):                              
                        if 'mean' not in analysis or 'fsaverage' in subj:
                            if 'sub' in subj or 'fsaverage' in subj:
                                curr_alpha = subj_res['Processed Results']['Alpha']['all']>rsq_thresh
                                for param in parameters:                                

                                    if roi in self.idx_rois[subj]:

                                        #comparing same vertices
                                        alpha[analysis][subj][roi] = roi_mask(self.idx_rois[subj][roi], curr_alpha)
                                        
                                    
                                    else:
                                        #if ROI is not defined
                                        #if Brain use all available vertices
                                        if roi == 'Brain':
                                            alpha[analysis][subj][roi] = curr_alpha
                                        #if all, combine all other rois
                                        elif roi == 'combined':
                                            alpha[analysis][subj][roi] = roi_mask(np.concatenate(tuple([self.idx_rois[subj][r] for r in rois if ('combined' not in r and 'Brain' not in r and r in self.idx_rois[subj])])), curr_alpha)
                                        
                                        elif roi == 'all_custom':
                                            alpha[analysis][subj][roi] = roi_mask(np.concatenate(tuple([self.idx_rois[subj][r] for r in self.idx_rois[subj] if 'custom' in r])), curr_alpha)
                                        
                                        elif space == 'fsaverage' and roi in self.idx_rois['fsaverage']:
                                            alpha[analysis][subj][roi] = (roi_mask(self.idx_rois['fsaverage'][roi], curr_alpha))
                                                                                          
                                        else:
                                            #, otherwise none
                                            print(f"{roi}: undefined ROI")
                                            alpha[analysis][subj][roi] = np.zeros_like(curr_alpha).astype('bool')
                                            
                                            
                                    
                                    for model in self.only_models:
                                        if model in subj_res['Processed Results'][param] and alpha[analysis][subj][roi].sum()>0:
                                            alpha[analysis][subj][roi] *= np.isfinite(subj_res['Processed Results'][param][model])
                                            
                                            #manual exclusion of outliers
                                            if quantile_exclusion is None:    
                                                print("Using manual exclusion, see multidim_analysis function. Set quantile_exclusion=1 for no exclusion.")

                                                if param == 'Surround Size (fwatmin)':
                                                    #exclude too large surround (no surround)
                                                    alpha[analysis][subj][roi] *= (subj_res['Processed Results'][param][model]<w_max)
                                                if param == 'CSS Exponent':
                                                    alpha[analysis][subj][roi] *= (subj_res['Processed Results'][param][model]<css_max)
                                                if 'Suppression' in param:
                                                    alpha[analysis][subj][roi] *= (subj_res['Processed Results'][param][model]<supp_max)
                                                if 'Norm Param.' in param:    
                                                    alpha[analysis][subj][roi] *= (subj_res['Processed Results'][param][model]<bd_max)
                                                
                                                # if 'Amplitude' in param:
                                                #     print(param)
                                                #     print(subj_res['Processed Results'][param][model][alpha[analysis][subj][roi]].max())
                                                #     alpha[analysis][subj][roi] *= (subj_res['Processed Results'][param][model]<bd_max)   
                                                    
                                                #     print(subj_res['Processed Results'][param][model][alpha[analysis][subj][roi]].max())
                                             
    
                                                # param_max = subj_res['Processed Results'][param][model][alpha[analysis][subj][roi]].mean()+3*np.std(subj_res['Processed Results'][param][model][alpha[analysis][subj][roi]])
                                                # param_min = subj_res['Processed Results'][param][model][alpha[analysis][subj][roi]].mean()-3*np.std(subj_res['Processed Results'][param][model][alpha[analysis][subj][roi]])
                                                
                                                
                                                # alpha[analysis][subj][roi] *= (subj_res['Processed Results'][param][model]<param_max)
                                                # alpha[analysis][subj][roi] *= (subj_res['Processed Results'][param][model]>param_min)
                                                # print(f"max min {param} {param_max} {param_min} {np.var(subj_res['Processed Results'][param][model][alpha[analysis][subj][roi]])}")
                                            else:
                                                if model in subj_res['Processed Results'][param] and np.sum(subj_res['Processed Results'][param][model])>0:
                                                    alpha[analysis][subj][roi] *= (subj_res['Processed Results'][param][model]<np.nanquantile(subj_res['Processed Results'][param][model],quantile_exclusion))*(subj_res['Processed Results'][param][model]>np.nanquantile(subj_res['Processed Results'][param][model],1-quantile_exclusion))  

                                                
                    for i, roi in enumerate(rois):                              
                        if 'mean' not in analysis or 'fsaverage' in subj:
                            if 'sub' in subj or 'fsaverage' in subj:
                                for param in parameters:                                           
                                    
                                    for model in self.only_models:
                                        if model in subj_res['Processed Results'][param]:                       
                                            dimensions[analysis][subj][roi].append(f"{param} {model}")
                                            #print(param)
                                            #print(subj_res['Processed Results'][param][model][alpha[analysis][subj][roi]].max())
                                            
                                            if zscore_data:
                                                multidim_param_array[analysis][subj][roi].append(zscore(subj_res['Processed Results'][param][model][alpha[analysis][subj][roi]]))
                                            else:
                                                multidim_param_array[analysis][subj][roi].append(subj_res['Processed Results'][param][model][alpha[analysis][subj][roi]])
                                                
                                    
                                    if y_parameter_toplevel != None:
                                        if param in subj_res['Processed Results'][y_parameter_toplevel]:
                                            dimensions[analysis][subj][roi].append(f"{param}")
                                            if zscore_data:
                                                multidim_param_array[analysis][subj][roi].append(zscore(subj_res['Processed Results'][y_parameter_toplevel][param][alpha[analysis][subj][roi]]))
                                            else:
                                                multidim_param_array[analysis][subj][roi].append(subj_res['Processed Results'][y_parameter_toplevel][param][alpha[analysis][subj][roi]])
                                                
                                
                                multidim_param_array[analysis][subj][roi] = np.array([x for _,x in sorted(zip(dimensions[analysis][subj][roi],multidim_param_array[analysis][subj][roi]))])

                                
                            elif len(subjects)>1 and space != 'fsaverage':
                                #group data
                                dimensions[analysis][subj][roi] = dimensions[analysis][subjects[0][0]][roi]
                                alpha[analysis][subj][roi] = np.hstack(tuple([alpha[analysis][sid][roi] for sid in alpha[analysis] if 'sub' in sid]))
                                multidim_param_array[analysis][subj][roi] = np.hstack(tuple([multidim_param_array[analysis][sid][roi] for sid in multidim_param_array[analysis] if 'sub' in sid]))
 
                                
                        elif space != 'fsaverage':
                            #all analyses data
                            ans = [an[0] for an in analyses if 'mean' not in an[0]]
                            dimensions[analysis][subj][roi] = dimensions[analyses[0][0]][subj][roi]
                            alpha[analysis][subj][roi] = np.hstack(tuple([alpha[an][subj][roi] for an in ans]))
                            multidim_param_array[analysis][subj][roi] = np.hstack(tuple([multidim_param_array[an][subj][roi] for an in ans]))

                            
                    
                        ordered_dimensions = sorted(dimensions[analysis][subj][roi])
                
                for subj, subj_res in subjects:
                    
                    
                    for i, roi in enumerate(rois):  
                        
                        if ('mean' in analysis) or 'fsaverage' in subj:
                            
                            
                            #print(multidim_param_array[analysis][subj][roi].shape)
                            
                            if plot_corr_matrix:
                                print(f"{analysis} {subj} {roi}")
                                pl.figure(figsize=(12,12))
                                correlation_matrix = np.corrcoef(multidim_param_array[analysis][subj][roi])
                                im = pl.imshow(correlation_matrix, vmin=-0.5, vmax=0.5, cmap='RdBu_r')
                                #pl.xlim((0,correlation_matrix.shape[0]))
                                #pl.ylim((0,correlation_matrix.shape[0]))
                                
                                for i in range(correlation_matrix.shape[0]):
                                    for j in range(correlation_matrix.shape[1]):
                                        pl.text(i, j, f"{correlation_matrix[i,j]:.2f}", fontsize=14, color='black', weight = 'bold', ha='center', va='center')
                                    
                                
                                ticks = []
                                tick_colors = []
                                rec_dims = []
                                for dim, dim_name in enumerate(ordered_dimensions):
                                    for model in self.only_models:
                                        if model in dim_name:
                                            tick_colors.append(model_colors[model])
                                            ticks.append(dim_name.replace(model,''))
                                    if y_parameter_toplevel != None:
                                        if dim_name in subj_res['Processed Results'][y_parameter_toplevel]:
                                            tick_colors.append('black')
                                            ticks.append(dim_name)
                                            rec_dims.append(dim)
                                
                                pl.xticks(np.arange(len(ordered_dimensions)), ticks, rotation='vertical')
                                pl.yticks(np.arange(len(ordered_dimensions)), ticks)
                                colorbar(im)
                                
                                for tick_x, tick_y, tick_color in zip(pl.gca().get_xticklabels(), pl.gca().get_yticklabels(), tick_colors):
                                    tick_x.set_color(tick_color)
                                    tick_y.set_color(tick_color)

                #regressions
                pls_result_dict = dd(lambda:dd(list))
                for subj, subj_res in subjects:
                    
                    
                    for i, roi in enumerate(rois):  
                        
                        if ('mean' in analysis) or 'fsaverage' in subj:
                                                            
                            
                            if regress_params:
                                print(f"{analysis} {subj} {roi}")
                                if x_dims_idx is None or y_dims_idx is None:    
                                    if y_parameter_toplevel != None:
                                        x_dims = [dim for dim, _ in enumerate(ordered_dimensions) if dim not in rec_dims]
                                        y_dims = rec_dims
                                    else:
                                        x_dims = list(set([dim for dim, dim_name in enumerate(ordered_dimensions) if 'Norm_abcd' in dim_name]))
                                        y_dims = list(set([dim for dim, dim_name in enumerate(ordered_dimensions) if 'Norm_abcd' not in dim_name]))
                                else:
                                    
                                    x_dims = list(set([dim for dim, dim_name in enumerate(ordered_dimensions) for x in x_dims_idx if dim_name.startswith(parameters[x])]))
                                    y_dims = list(set([dim for dim, dim_name in enumerate(ordered_dimensions) for y in y_dims_idx if dim_name.startswith(parameters[y])]))
                                    
                                print(f"X-dims: {[ordered_dimensions[x_dim] for x_dim in x_dims]}")
                                print(f"Y-dims: {[ordered_dimensions[y_dim] for y_dim in y_dims]}")
                                
                                X = multidim_param_array[analysis][subj][roi][x_dims].T
                                print(f"Sample size: {X.shape[0]}")
                                
                                
                                try:
                                    if not multidim_y:
                                        print("1D y")
                                        for y_dim in y_dims:
                                            
                                            y = multidim_param_array[analysis][subj][roi][y_dim].T
                                            
                                            
                                            ls1 = LinearRegression()
                                            ls1.fit(X,y)
                                            print(f"Estimated betas {ordered_dimensions[y_dim]} {ls1.coef_}") 
                                            rsq_prediction = ls1.score(X,y)
                                            print(f"RSq {rsq_prediction}")
                                            corr_prediction = np.corrcoef(ls1.predict(X), y)[0,1]
                                            print(f"corr prediction {corr_prediction}")
                                            
                                            if cv_regression:
                                                for cv_subj in [s for s,_ in subjects if s!='fsaverage' and s!='Group' and s!=subj]:
                                                    print(f"CV regression (fit on {subj}, test on {cv_subj})")
                                                    X_cv = multidim_param_array[analysis][cv_subj][roi][x_dims].T
                                                    y_cv = multidim_param_array[analysis][cv_subj][roi][y_dim].T
                                                    rsq_prediction = ls1.score(X_cv,y_cv)
                                                    print(f"CV RSq {rsq_prediction}")
                                                    corr_prediction = np.corrcoef(ls1.predict(X), y)[0,1]
                                                    print(f"CV corr prediction {corr_prediction}")                                                    


                                    else:
                                        
                                        print("Multidim Y")
                                        Y = multidim_param_array[analysis][subj][roi][y_dims].T
                                        

                                        
                                        for n_components in range(1, 1+X.shape[1]):
                                            pls = PLSRegression(n_components)
                                            pls.fit(X, Y)
                                            print(f"Performing PLS with {n_components} components")

                                            print(f"Estimated betas: {[ordered_dimensions[y_dim] for y_dim in y_dims]}")
                                            for i,x_dim in enumerate(x_dims):
                                                print(f"{ordered_dimensions[x_dim]}: {np.array(pls.coef_)[i]}, total beta: {np.abs(np.array(pls.coef_)[i]).sum():.3f}")
                                                

                                            pred = pls.predict(X)
                                            corr_prediction = []
                                            rsq_prediction = []
                                            
                                            for p in range(pred.shape[1]):
                                                corr_prediction.append(np.corrcoef(pred[:,p], Y[:,p])[0,1])
                                                rsq_prediction.append(1-np.sum((pred[:,p]-Y[:,p])**2)/(pred.shape[0]*Y[:,p].var()))
                                                
                                            print(f"RSq total {pls.score(X,Y)}")
                                            print(f"RSq predictions {np.array(rsq_prediction)}")
                                            print(f"corr predictions {np.array(corr_prediction)}\n")
                                            
                                            if cv_regression == True:
                                                cv_corr_pred = []
                                                cv_rsq_pred = []
                                                cv_rsq_total = []
                                                for cv_subj in [s for s,_ in subjects if s!='fsaverage' and s!='Group' and s!=subj]:
                                                    print(f"CV regression (fit on {subj}, test on {cv_subj})")
                                                    X_cv = multidim_param_array[analysis][cv_subj][roi][x_dims].T
                                                    Y_cv = multidim_param_array[analysis][cv_subj][roi][y_dims].T
                                                    
                                                    pred = pls.predict(X_cv)
                                                    corr_prediction = []
                                                    rsq_prediction = []
                                                    
                                                    for p in range(pred.shape[1]):
                                                        corr_prediction.append(np.corrcoef(pred[:,p], Y_cv[:,p])[0,1])
                                                        rsq_prediction.append(1-np.sum((pred[:,p]-Y_cv[:,p])**2)/(pred.shape[0]*Y_cv[:,p].var()))
                                                        
                                                    cv_rsq_total.append(pls.score(X_cv,Y_cv))
                                                    cv_corr_pred.append(corr_prediction)
                                                    cv_rsq_pred.append(rsq_prediction)
                                                        
                                                print(f"CVRSq total {np.mean(cv_rsq_total,axis=0)}")
                                                print(f"CVRSq predictions {np.array(cv_rsq_pred).mean(0)}")
                                                print(f"CV corr predictions {np.array(cv_corr_pred).mean(0)}")
                                                
                                                pls_result_dict[roi][f"{n_components} components"].append(np.mean(cv_rsq_total,axis=0))
                                        
                                except Exception as e:
                                    print(e)
                                    pass
                
                for key in pls_result_dict:
                    for key2 in pls_result_dict[key]:
                        print(f"{key} {key2} CVRSq total {np.mean(pls_result_dict[key][key2]):.3f}")
                #size response curves
                for subj, subj_res in subjects:
                    if 'analysis_info' in subj_res:
                        an_info = subj_res['analysis_info']
                        ss_deg = 2.0 * np.degrees(np.arctan(an_info['screen_size_cm'] /(2.0*an_info['screen_distance_cm'])))
                        n_pix = an_info['n_pix']                    
                    
                    for i, roi in enumerate(rois):  
                        
                        if ('mean' in analysis) or 'fsaverage' in subj:
                                                      
                            if size_response_curves:
                                print(f"{analysis} {subj} {roi}")  

                                
                                curve_par_dict = dict()
                                
                                dim_stim = 2
                                bar_stim = False
                                center_prfs = True
                                plot_data = False
                                plot_curves = True                               
                                resp_measure = 'Max (model-based)'
                                normalize_response = True
                                confint = True
                                
                                x = np.linspace(-ss_deg/2,ss_deg/2,500)
                                #correcting for the real size of design matrix VS smoother space used for better size-response curves
                                dx = n_pix/len(x)
                                
                                #1d stims
                                if dim_stim == 1:
                                    stims = [np.zeros_like(x) for n in range(int(x.shape[-1]/4))]
                                else:
                                    factr=4
                                    stims = [np.zeros((x.shape[0],x.shape[0])) for n in range(int(x.shape[-1]/(2*factr)))]
                                    stim_sizes=[]
                                    
                                print(len(stims))
                                for pp, stim in enumerate(stims):
                                    
                                    if dim_stim == 1 or bar_stim == True:
                                        #2d rectangle or 1d
                                        stim[int(stim.shape[0]/2)-pp:int(stim.shape[0]/2)+pp] = 1
                                    
                                    else:
                                        #2d circle
                                        xx,yy = np.meshgrid(x,x)
                                        stim[((xx**2+yy**2)**0.5)<(x.max()*pp/(len(stims)*factr))] = 1
                                        stim_sizes.append(2*(x.max()*pp/(len(stims)*factr)))
                                
                                if subj == 'Group' and roi == 'custom.V1':
                                    pl.figure()
                                    if dim_stim == 1:
                                        pl.imshow(stims)
                                    else:
                                        pl.imshow(stims[-1])
                                    
                                
                                if dim_stim == 1:
                                    #1d stim sizes
                                    stim_sizes = (np.max(x)-np.min(x))*np.sum(stims,axis=-1)/x.shape[0]
                                elif bar_stim == True:
                                    #2d stim sizes (rectangle)
                                    stim_sizes = (np.max(x)-np.min(x))*np.sum(stims,axis=(-1,-2))/x.shape[0]**2                                    
     
                                # for c in range(100):
                                #     sample = np.random.randint(0, len(multidim_param_array[analysis][subj][roi]), int(len(multidim_param_array[analysis][subj][roi])/upsampling_corr_factor))
                                        
                                #     for par in ['Amplitude', 'Norm Param. B', 'Surround Amplitude', 'Norm Param. D',
                                #             'Size (sigma_1)', 'Size (sigma_2)']:
                                                
                                
                                #         curve_par_dict[par] = weightstats.DescrStatsW(multidim_param_array[analysis][subj][roi][ordered_dimensions.index(f'{par} Norm_abcd')][sample],
                                #                         weights=multidim_param_array[analysis][subj][roi][ordered_dimensions.index(f'RSq Norm_abcd')][sample]).mean
                                        
                                #     response_functions.append(norm_1d_sr_function(curve_par_dict['Amplitude'],curve_par_dict['Norm Param. B'],curve_par_dict['Surround Amplitude'],
                                #                                             curve_par_dict['Norm Param. D'],curve_par_dict['Size (sigma_1)'],curve_par_dict['Size (sigma_2)'],x,stims))

                                
                                for par in ['Amplitude', 'Norm Param. B', 'Surround Amplitude', 'Norm Param. D',
                                             'Size (sigma_1)', 'Size (sigma_2)', 'Eccentricity', 'Polar Angle', 'RSq']:
                                    #curve_par_dict[par] = np.median(multidim_param_array[analysis][subj][roi][ordered_dimensions.index(f'{par} Norm_abcd')])
                                    
                                    curve_par_dict[par] = weightstats.DescrStatsW(multidim_param_array[analysis][subj][roi][ordered_dimensions.index(f'{par} Norm_abcd')],
                                                        weights=multidim_param_array[analysis][subj][roi][ordered_dimensions.index('RSq Norm_abcd')])
                                    
                                    #curve_par_dict[par] = np.copy(multidim_param_array[analysis][subj][roi][ordered_dimensions.index(f'{par} Norm_abcd')])
                                    #print(f"{roi} {par} {curve_par_dict[par]}")
                                
                                # for vertex in tqdm(range(curve_par_dict['Amplitude'].shape[0])):
                                #     response_functions.append(norm_2d_sr_function(curve_par_dict['Amplitude'][vertex],curve_par_dict['Norm Param. B'][vertex],curve_par_dict['Surround Amplitude'][vertex],
                                #                                               curve_par_dict['Norm Param. D'][vertex],curve_par_dict['Size (sigma_1)'][vertex],curve_par_dict['Size (sigma_2)'][vertex],x,x,stims,
                                #                                               mu_x=0,#curve_par_dict['Eccentricity'][vertex]*np.cos(curve_par_dict['Polar Angle'][vertex]),
                                #                                               mu_y=0))#curve_par_dict['Eccentricity'][vertex]*np.sin(curve_par_dict['Polar Angle'][vertex])))
                                    
                                    
                                #     response_functions.append(norm_1d_sr_function(curve_par_dict['Amplitude'][vertex],curve_par_dict['Norm Param. B'][vertex],curve_par_dict['Surround Amplitude'][vertex],
                                #                                               curve_par_dict['Norm Param. D'][vertex],curve_par_dict['Size (sigma_1)'][vertex],curve_par_dict['Size (sigma_2)'][vertex],x,stims,
                                #                                               mu_x=0)#curve_par_dict['Eccentricity'][vertex]))#*np.sign(np.cos(curve_par_dict['Polar Angle'][vertex]))))
                                
                                if center_prfs:
                                    ecc = 0
                                    #ecc_max_min = (0,0)
                                else:
                                    ecc = curve_par_dict['Eccentricity'].mean
                                    #ecc_max_min = curve_par_dict['Eccentricity'].zconfint_mean(alpha=0.05/upsampling_corr_factor)
                                    
                                if dim_stim == 1:    
                                    mean_srf = norm_1d_sr_function(curve_par_dict['Amplitude'].mean,curve_par_dict['Norm Param. B'].mean,curve_par_dict['Surround Amplitude'].mean,
                                                                                  curve_par_dict['Norm Param. D'].mean,curve_par_dict['Size (sigma_1)'].mean,curve_par_dict['Size (sigma_2)'].mean,x,stims,
                                                                                  mu_x=ecc*np.sign(np.cos(curve_par_dict['Polar Angle'].mean)))
        
                                else:
                                    mean_srf = norm_2d_sr_function(dx**2*curve_par_dict['Amplitude'].mean,curve_par_dict['Norm Param. B'].mean,
                                                               dx**2*curve_par_dict['Surround Amplitude'].mean,
                                                                                curve_par_dict['Norm Param. D'].mean,curve_par_dict['Size (sigma_1)'].mean,curve_par_dict['Size (sigma_2)'].mean,x,x,stims,
                                                                                mu_x=ecc*np.cos(curve_par_dict['Polar Angle'].mean),
                                                                                mu_y=ecc*np.sin(curve_par_dict['Polar Angle'].mean))
                                    
                                
                                    
                                if confint and subj != 'Group':
                                    
                                    # combinations = [list(i) for i in itertools.product([0, 1], repeat=8)]
                                    
                                    # for cc in combinations[::5]:
                                    #     if dim_stim == 1:  
                                    #         response_functions.append(norm_1d_sr_function(curve_par_dict['Amplitude'].zconfint_mean(alpha=0.05/upsampling_corr_factor)[cc[0]],curve_par_dict['Norm Param. B'].zconfint_mean(alpha=0.05/upsampling_corr_factor)[cc[1]],curve_par_dict['Surround Amplitude'].zconfint_mean(alpha=0.05/upsampling_corr_factor)[cc[2]],
                                    #                                                   curve_par_dict['Norm Param. D'].zconfint_mean(alpha=0.05/upsampling_corr_factor)[cc[3]],curve_par_dict['Size (sigma_1)'].zconfint_mean(alpha=0.05/upsampling_corr_factor)[cc[4]],curve_par_dict['Size (sigma_2)'].zconfint_mean(alpha=0.05/upsampling_corr_factor)[cc[5]],x,stims,
                                    #                                                   mu_x=ecc_max_min[cc[6]]*np.sign(np.cos(curve_par_dict['Polar Angle'].zconfint_mean(alpha=0.05/upsampling_corr_factor)[cc[7]]))))
            
                                    #     else:
                                    #         response_functions.append(norm_2d_sr_function(dx**2*curve_par_dict['Amplitude'].zconfint_mean(alpha=0.05/upsampling_corr_factor)[cc[0]],curve_par_dict['Norm Param. B'].zconfint_mean(alpha=0.05/upsampling_corr_factor)[cc[1]],
                                    #                                dx**2*curve_par_dict['Surround Amplitude'].zconfint_mean(alpha=0.05/upsampling_corr_factor)[cc[2]],
                                    #                                                 curve_par_dict['Norm Param. D'].zconfint_mean(alpha=0.05/upsampling_corr_factor)[cc[3]],curve_par_dict['Size (sigma_1)'].zconfint_mean(alpha=0.05/upsampling_corr_factor)[cc[4]],curve_par_dict['Size (sigma_2)'].zconfint_mean(alpha=0.05/upsampling_corr_factor)[cc[5]],x,x,stims,
                                    #                                                 mu_x=ecc_max_min[cc[6]]*np.cos(curve_par_dict['Polar Angle'].zconfint_mean(alpha=0.05/upsampling_corr_factor)[cc[7]]),
                                    #                                                 mu_y=ecc_max_min[cc[6]]*np.sin(curve_par_dict['Polar Angle'].zconfint_mean(alpha=0.05/upsampling_corr_factor)[cc[7]])))
                                    response_functions[roi].append(mean_srf)  
                                    #response_functions = np.array(response_functions)
                                
                                #srf_stats = weightstats.DescrStatsW(response_functions,weights=curve_par_dict['RSq'])
                                #mean_srf = srf_stats.mean#np.mean(response_functions, axis=0)
                                #max_srf = srf_stats.zconfint_mean(alpha=0.01/upsampling_corr_factor)[1]# mean_srf+sem(response_functions, axis=0)
                                #min_srf = srf_stats.zconfint_mean(alpha=0.01/upsampling_corr_factor)[0]#mean_srf-sem(response_functions, axis=0)
                                

                                if normalize_response: 
                                    
                                    mean_srf /= mean_srf.max()
                                    
                                    

                                if plot_data:
                                    actual_response_1R = weightstats.DescrStatsW(multidim_param_array['fit-task-1R_fit-runs-all'][subj][roi][ordered_dimensions.index(resp_measure)],
                                                                              weights=multidim_param_array['fit-task-1R_fit-runs-all'][subj][roi][ordered_dimensions.index('RSq Norm_abcd')])
                                    actual_response_2R = weightstats.DescrStatsW(multidim_param_array['fit-task-2R_fit-runs-all'][subj][roi][ordered_dimensions.index(resp_measure)],
                                                                              weights=multidim_param_array['fit-task-2R_fit-runs-all'][subj][roi][ordered_dimensions.index('RSq Norm_abcd')])
                                    actual_response_4R = weightstats.DescrStatsW(multidim_param_array['fit-task-4R_fit-runs-all'][subj][roi][ordered_dimensions.index(resp_measure)],
                                                                              weights=multidim_param_array['fit-task-4R_fit-runs-all'][subj][roi][ordered_dimensions.index('RSq Norm_abcd')])
                                    actual_response_1S = weightstats.DescrStatsW(multidim_param_array['fit-task-1S_fit-runs-all'][subj][roi][ordered_dimensions.index(resp_measure)],
                                                                              weights=multidim_param_array['fit-task-1S_fit-runs-all'][subj][roi][ordered_dimensions.index('RSq Norm_abcd')])
                                    actual_response_4F = weightstats.DescrStatsW(multidim_param_array['fit-task-4F_fit-runs-all'][subj][roi][ordered_dimensions.index(resp_measure)],
                                                                              weights=multidim_param_array['fit-task-4F_fit-runs-all'][subj][roi][ordered_dimensions.index('RSq Norm_abcd')])
                                   
                                    actual_response_1 = weightstats.DescrStatsW(np.concatenate((multidim_param_array['fit-task-1R_fit-runs-all'][subj][roi][ordered_dimensions.index(resp_measure)],
                                                                                              multidim_param_array['fit-task-1S_fit-runs-all'][subj][roi][ordered_dimensions.index(resp_measure)])),
                                                                              weights=np.concatenate((multidim_param_array['fit-task-1R_fit-runs-all'][subj][roi][ordered_dimensions.index('RSq Norm_abcd')],
                                                                                                      multidim_param_array['fit-task-1S_fit-runs-all'][subj][roi][ordered_dimensions.index('RSq Norm_abcd')])))
                                    actual_response_4 = weightstats.DescrStatsW(np.concatenate((multidim_param_array['fit-task-4R_fit-runs-all'][subj][roi][ordered_dimensions.index(resp_measure)],
                                                                                              multidim_param_array['fit-task-4F_fit-runs-all'][subj][roi][ordered_dimensions.index(resp_measure)])),
                                                                            weights=np.concatenate((multidim_param_array['fit-task-4R_fit-runs-all'][subj][roi][ordered_dimensions.index('RSq Norm_abcd')],
                                                                                                      multidim_param_array['fit-task-4F_fit-runs-all'][subj][roi][ordered_dimensions.index('RSq Norm_abcd')])))
                                   
                                    
                                    data_sr = [actual_response_1.mean, actual_response_1.mean,actual_response_2R.mean,actual_response_4.mean,actual_response_4.mean]/np.max([actual_response_1.mean, actual_response_1.mean,actual_response_2R.mean,actual_response_4.mean,actual_response_4.mean])
                                    if zconfint_err_alpha is not None:
                                        yerr_data_sr = np.array([np.abs(ss.zconfint_mean(alpha=zconfint_err_alpha)-ss.mean) for ss in [actual_response_1,actual_response_1,actual_response_2R,actual_response_4,actual_response_4]]).T*upsampling_corr_factor**0.5
                                    else:
                                        yerr_data_sr = np.array([ss.std_mean for ss in [actual_response_1,actual_response_1,actual_response_2R,actual_response_4,actual_response_4]])*upsampling_corr_factor**0.5
                                    
                                    #yerr_data_sr /= np.max([actual_response_1R.mean,actual_response_1S.mean,actual_response_2R.mean,actual_response_4R.mean,actual_response_4F.mean])
                                    

                                if subj == 'Group':
                                    pl.figure(f"Size response {roi.replace('custom.','').replace('HCPQ1Q6.','')}", figsize = (8,8))
                                    
                                    if plot_curves:
                                        pl.plot(stim_sizes, mean_srf, c=cmap_rois[i], label=roi.replace('custom.','').replace('HCPQ1Q6.',''), linewidth=3)
                                    
                                    if plot_data:
                                        pl.errorbar([0.5, 1.25, 2.5], data_sr[np.r_[0,2,3]], yerr_data_sr[:,np.r_[0,2,3]], marker='s', mec='k', linestyle='-', c=cmap_rois[i], label=roi.replace('custom.','').replace('HCPQ1Q6.',''))# label='Same speed')
                                        #pl.errorbar([0.5, 1.25, 2.5], data_sr[np.r_[1,2,4]], yerr_data_sr[:,np.r_[1,2,4]], marker='v',  mec='k',linestyle='-', c=cmap_rois[i], label='Same STCE')
                                    
                                    #pl.plot(stim_sizes,np.zeros(len(stim_sizes)),linestyle='--',linewidth=0.8, alpha=0.8, color='black',zorder=0)
                                    pl.ylabel("Response")
                                    pl.xlabel("Stimulus size (°)")
                                    pl.legend(fontsize=28,loc=8)   
                                    if save_figures:
                                        pl.savefig(opj(figure_path,f"sr_functions_{roi.replace('custom.','').replace('HCPQ1Q6.','')}.pdf"),dpi=600, bbox_inches='tight')
                                    
                                    
                                    pl.figure('Size response all rois', figsize = (8,8))

                                    
                                    if roi == 'all_custom':
                                        if plot_data:
                                            pl.errorbar([0.5, 1.25, 2.5], data_sr[np.r_[0,2,3]], yerr_data_sr[:,np.r_[0,2,3]], marker='s', mec='k', linestyle='-', c='grey', label=roi.replace('custom.','').replace('HCPQ1Q6.',''))# label='Same speed')
                                            #pl.errorbar([0.5, 1.25, 2.5], data_sr[np.r_[1,2,4]], yerr_data_sr[:,np.r_[1,2,4]], marker='v',  mec='k',linestyle='', c='grey', label='Same STCE (all rois)')
                                                                                
                                    else:
                                        if plot_data:
                                            pl.errorbar([0.5, 1.25, 2.5], data_sr[np.r_[0,2,3]], 0, marker='s', markersize=6, zorder=len(rois)-i, mec='k', linestyle='-', c=cmap_rois[i], label=roi.replace('custom.','').replace('HCPQ1Q6.',''))# label='Same speed')
                                        if plot_curves:
                                            pl.plot(stim_sizes, mean_srf, c=cmap_rois[i], label=roi.replace('custom.','').replace('HCPQ1Q6.',''), linewidth=2, zorder=len(rois))
                                    
                                    if confint:
                                        
                                        if normalize_response:
                                            response_functions[roi] /= mean_srf.max()
                                        pl.fill_between(stim_sizes, mean_srf+sem(response_functions[roi], axis=0),
                                                        mean_srf-sem(response_functions[roi], axis=0), color=cmap_rois[i], label=roi.replace('custom.','').replace('HCPQ1Q6.',''), alpha=0.2, zorder=len(rois))
                                                                                
                                        # pl.fill_between(stim_sizes, np.sort(response_functions[roi], axis=0)[1],
                                        #                 np.sort(response_functions[roi], axis=0)[-2], color=cmap_rois[i], alpha=0.2, zorder=len(rois)-i)
                                    handles, labels = pl.gca().get_legend_handles_labels()
        
                                    legend_dict = dd(list)
                                    for cc, label in enumerate(labels):
                                        legend_dict[label].append(handles[cc])
                                        
                                    for label in legend_dict:
                                        legend_dict[label] = tuple(legend_dict[label])
        
                                    pl.legend([legend_dict[label] for label in legend_dict], legend_dict.keys(), fontsize=20,loc=8)  
                                    #if i == 0:
                                    #    pl.plot(stim_sizes,np.zeros(len(stim_sizes)),linestyle='--',linewidth=0.8, alpha=0.8, color='black', zorder=0)
                                    
                                    #pl.ylabel("Response")
                                    pl.xlabel("Stimulus size (°)")
                                    pl.tick_params(labelleft=False)

                                    if save_figures:
                                        pl.savefig(opj(figure_path,'sr_functions_allrois.pdf'), dpi=600, bbox_inches='tight')
                                    


                                else:
                                    #subject curves
                                    pl.figure(f"Size response {roi.replace('custom.','').replace('HCPQ1Q6.','')}", figsize = (8,8))

                                    if np.sum(alpha[analysis][subj][roi]) > 0:
                                        if plot_curves:
                                            pl.plot(stim_sizes, mean_srf, c=cmap_rois[i], alpha=0.5, linewidth=1)
                                        if plot_data:
                                            pl.errorbar([0.5, 1.25, 2.5], data_sr[np.r_[0,2,3]], 0, alpha=0.25,  linestyle='-', c=cmap_rois[i])#markersize=4, marker='s',
                                            #pl.errorbar([0.5, 1.25, 2.5], data_sr[np.r_[1,2,4]], yerr_data_sr[:,np.r_[1,2,4]], alpha=0.4, markersize=4, marker='v',linestyle='', c=cmap_rois[i])
                                                 
                                


                                     #css_datapoints = np.geomspace(0.8, 24, num=13)
                                    #pl.scatter(css_datapoints,np.zeros_like(css_datapoints))                               
                            

            #print(f"mean rsq pred {np.mean(rsq_predictions)}")
            #print(f"mean corr pred {np.mean(corr_predictions)}")


                            # #from sklearn.decomposition import FastICA
                            # #from sklearn.manifold import MDS
                            # pca = PCA()#n_components=2, svd_solver='full')
                            # #mds = MDS()
                            # #ica = FastICA()
                            
                            # transformed_data = pca.fit_transform(multidim_param_array[analysis][subj][roi].T)
                            # #transformed_data = ica.fit_transform(multidim_param_array[analysis][subj][roi].T)
                            # #transformed_data = mds.fit_transform(multidim_param_array[analysis][subj][roi].T)
                            # #transformed_data = multidim_param_array[analysis][subj][roi].T
                            
                            
                            # print(transformed_data.shape)
                            
                            # start = 0
                            # stop = 0
                            # means=[]
                            # # for dims in list(itertools.combinations(np.arange(transformed_data.shape[1]),2)):
                            # #     fig = pl.figure(f'scatter {dims}', figsize=(6,6))
                                
                            # #     ax = fig.add_subplot(111)
                            # #     ax.grid(False)

                            # #     #sample = np.random.randint(0, (stop-start), 200)
                                
                            # #     #means.append(np.mean(transformed_data[start:stop,:], axis=0))

                            # #     #means_dist = means[np.newaxis,...]-transformed_data[start:stop,:]
                                
                            # #     #ax.scatter(transformed_data[start:stop,0].mean(), transformed_data[start:stop,1].mean(), transformed_data[start:stop,2].mean(), s=40, c=cmap_rois[g],  alpha=1) #zorder=len(rois)-i,
                            # #     #ax.scatter(transformed_data[start:stop,dims[0]], transformed_data[start:stop,dims[1]], s=1, c=cmap_rois[g],  alpha=0.5, zorder=len(rois)-1-g)
                            # #     #ax.scatter(transformed_data[:,dims[0]], transformed_data[:,dims[1]], s=1, c=cmap_rois[i],  alpha=0.5, zorder=len(rois)-i)

                            # for g,rrr in enumerate(rois): 
                            #     if 'custom' in rrr:
                            #         stop += np.sum(alpha[analysis][subj][rrr])
                            #         for dims in list(itertools.combinations(np.arange(transformed_data.shape[1]),2)):
                            #             fig = pl.figure(f'scatter {dims}', figsize=(6,6))
                                        
                            #             ax = fig.add_subplot(111)
                            #             ax.grid(False)

                            #             #sample = np.random.randint(0, (stop-start), 200)
                                        
                            #             means.append(np.mean(transformed_data[start:stop,:], axis=0))

                            #             #means_dist = means[np.newaxis,...]-transformed_data[start:stop,:]
                                        
                            #             #ax.scatter(transformed_data[start:stop,0].mean(), transformed_data[start:stop,1].mean(), transformed_data[start:stop,2].mean(), s=40, c=cmap_rois[g],  alpha=1) #zorder=len(rois)-i,
                            #             ax.scatter(transformed_data[start:stop,dims[0]], transformed_data[start:stop,dims[1]], s=1, c=cmap_rois[g],  alpha=0.5, zorder=len(rois)-g)
                            #             #ax.scatter(transformed_data[:,dims[0]], transformed_data[:,dims[1]], s=1, c=cmap_rois[g],  alpha=0.5, zorder=len(rois)-g)

                            #             #ax.scatter(transformed_data[start:stop,0][sample], transformed_data[start:stop,1][sample],transformed_data[start:stop,2][sample], s=1, c=cmap_rois[g],  alpha=0.5) #zorder=len(rois)-i,

                            #             #ax.scatter(np.median(transformed_data[start:stop,0]), np.median(transformed_data[start:stop,1]), s=10, c=cmap_rois[g],  alpha=1)
                            #         start += np.sum(alpha[analysis][subj][rrr])
                            
                            # print(np.var(means, axis=0))
                            # np.set_printoptions(suppress=True)
                            # print(ordered_dimensions)
                            # print(pca.components_)
                            # print(pca.explained_variance_ratio_)



        return
      
   

                                
