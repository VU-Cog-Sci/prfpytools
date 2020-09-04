#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 17 10:37:08 2020

@author: marcoaqil
"""
import os
import numpy as np
import matplotlib.pyplot as pl
import cortex
import nibabel as nb
from matplotlib import cm
from collections import defaultdict as dd
from copy import deepcopy
from pathlib import Path
from utils.postproc_utils import model_wrapper, create_model_rf_wrapper
from utils.preproc_utils import create_full_stim
#import seaborn as sns

import time
from scipy.stats import sem, ks_2samp, ttest_1samp, wilcoxon, pearsonr, ttest_ind, ttest_rel

opj = os.path.join

from statsmodels.stats import weightstats
from sklearn.linear_model import LinearRegression
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
        
    def get_subjects(self, curr_dict, subject_list = []):
        for k, v in curr_dict.items():
            if 'sub-' not in k:# and isinstance(v, (dict,dd)):
                self.get_subjects(v, subject_list)
            else:
                if k not in subject_list:
                    subject_list.append(k)
        
        self.subjects = subject_list
        return
    
    def get_spaces(self):
        self.spaces = self.main_dict.keys()
        return
        
    def import_rois_and_flatmaps(self, fs_dir):
        self.idx_rois = dd(dict)
        self.fs_dir = fs_dir
        self.get_spaces()
        self.get_subjects(self.main_dict)
        for subj in self.subjects:
            
            if subj not in cortex.db.subjects:
                print("importing subject from freesurfer")
                print(f"note: this command often files when executed on mac OS via jupyter notebook.\
                      Rather, source freesurfer and execute it in ipython: \
                          cortex.freesurfer.import_subj({subj}, freesurfer_subject_dir={self.fs_dir}, \
                      whitematter_surf='smoothwm')")
                cortex.freesurfer.import_subj(subj, freesurfer_subject_dir=self.fs_dir, 
                      whitematter_surf='smoothwm')
            if self.import_flatmaps:
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
        
            #parse custom ROIs if they have been created
            for roi in [el for el in os.listdir(opj(self.fs_dir, subj, 'label')) if 'custom' in el]:
                roi = roi.replace('lh.','').replace('rh.','').replace('.label','')
                try:
                    self.idx_rois[subj][roi], _ = cortex.freesurfer.get_label(subj,
                                                          label=roi,
                                                          fs_dir=self.fs_dir,
                                                          src_subject=subj)
                except Exception as e:
                    print(e)
                    pass
                
            if self.import_rois:
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
            if len(self.output_rois)>0:
                try:
                    rois = np.concatenate(tuple([self.idx_rois[subj][roi] for roi in self.output_rois]), axis=0)
                    np.save('/Users/marcoaqil/PRFMapping/PRFMapping-Deriv-hires/prfpy/'+subj+'_combined-rois.npy', rois)
                except Exception as e:
                    print(e)
                    pass    
        return

    def set_alpha(self, only_models=None, ecc_min=0, ecc_max=5):
        self.only_models=only_models
        self.tc_min = dict()
        for space, space_res in self.main_dict.items():
            if 'fs' in space:
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
                        ######max prf size (implemented in surround and size plotting functions)
                        #w_max = 90                        
              
                        #housekeeping
                        if only_models is None:
                            rsq = np.vstack(tuple([elem for _,elem in p_r['RSq'].items()])).T
                            ecc = np.vstack(tuple([elem for _,elem in p_r['Eccentricity'].items()])).T
                            #fw_hmax = np.vstack(tuple([elem for _,elem in p_r['Size (fwhmax)'].items()])).T
                        else:
                            rsq = np.vstack(tuple([elem for k,elem in p_r['RSq'].items() if k in only_models])).T
                            ecc = np.vstack(tuple([elem for k,elem in p_r['Eccentricity'].items() if k in only_models])).T                            
            
                        #alpha dictionary
                        p_r['Alpha'] = {}          
                        p_r['Alpha']['all'] = rsq.max(-1) * (tc_stats['Mean']>self.tc_min[subj]) * (ecc.min(-1)<self.ecc_max) * (ecc.max(-1)>self.ecc_min) * (rsq.min(-1)>0) #* (p_r['Noise Ceiling']['Noise Ceiling (RSq)']>0)
                        
                        for model in models:
                            p_r['Alpha'][model] = p_r['RSq'][model] * (p_r['Eccentricity'][model]>self.ecc_min) * (p_r['Eccentricity'][model]<self.ecc_max)\
                                * (tc_stats['Mean']>self.tc_min[subj]) #*(p_r['Size (fwhmax)'][model]<w_max)
                       
        return


    def pycortex_plots(self, rois, rsq_thresh, analysis_names = 'all', subject_ids='all'):        
        pl.rcParams.update({'font.size': 16})
        pl.rcParams.update({'pdf.fonttype':42})        
        self.click=0
        #######PYCORTEX PICKERFUN
        #function to plot prf and timecourses when clicking surface vertex in webgl        
        def clicker_function(voxel,hemi,vertex):

            print('recovering vertex index...')
            #translate javascript indeing to python
            lctm, rctm = cortex.utils.get_ctmmap(subj, method='mg2', level=9)
            if hemi == 'left':
                index = lctm[int(vertex)]
                #print(f"{model} rsq {p_r['RSq'][model][index]}")
            else:
                index = len(lctm)+rctm[int(vertex)]
                #print(f"{model} rsq {p_r['RSq'][model][index]}") 
            
                
            #sorting = np.argsort(subj_res['Processed Results']['RSq']['CSS']-subj_res['Processed Results']['RSq']['Norm_abcd'])
            #alpha_sort = subj_res['Processed Results']['Alpha']['all'][sorting]
            #sorting = sorting[(alpha_sort>rsq_thresh)*(subj_res['Processed Results']['RSq']['Norm_abcd'][sorting]-subj_res['Processed Results']['RSq']['DoG'][sorting]>0.1)*(subj_res['Processed Results']['RSq']['Norm_abcd'][sorting]>0.6)]
            #index = sorting[self.click]
            #surr_vertices = [2624, 285120, 5642, 144995, 5273, 2927, 1266]
            #both_vertices = [9372, 9383, 17766, 9384]#[161130, 9372, 9383, 302707, 17766, 9395, 9384]
            #index = both_vertices[self.click]
            
            
            print('recovering data and model timecourses...')
            
            vertex_info = f""

            #recover needed information
            an_info = subj_res['analysis_info']
            if 'timecourse_folder' not in an_info:
                an_info['timecourse_folder'] = '/Users/marcoaqil/PRFMapping/PRFMapping-Deriv-hires/prfpy/FS7_results/timecourses/'

            if not hasattr(self, 'prf_stim') or self.prf_stim.task_names != an_info['task_names']:
                self.prf_stim = create_full_stim(screenshot_paths=[opj(an_info['timecourse_folder'],f'task-{task}_screenshots') for task in an_info['task_names']],
                            n_pix=an_info['n_pix'],
                            discard_volumes=an_info['discard_volumes'],
                            baseline_volumes_begin_end=an_info['baseline_volumes_begin_end'],
                            dm_edges_clipping=an_info['dm_edges_clipping'],
                            screen_size_cm=an_info['screen_size_cm'],
                            screen_distance_cm=an_info['screen_distance_cm'],
                            TR=an_info['TR'],
                            task_names=an_info['task_names'])

                
            tc_paths = [str(path) for path in sorted(Path(an_info['timecourse_folder']).glob(f"{subj}_timecourse_space-{an_info['fitting_space']}_task-*_run-*.npy"))]    
            mask_paths = [tc_path.replace('timecourse_','mask_') for tc_path in tc_paths]
            #all_task_names = np.unique(np.array([elem.replace('task-','') for path in tc_paths for elem in path.split('_')  if 'task' in elem]))
            all_runs = np.unique(np.array([int(elem.replace('run-','').replace('.npy','')) for path in tc_paths for elem in path.split('_')  if 'run' in elem]))

            tc = dict()
            tc_err = dict()
            tc_test = dict()
            tc_fit = dict()
            
            for task in an_info['task_names']:
                if task not in self.prf_stims:
                    self.prf_stims[task] = create_full_stim(screenshot_paths=[opj(an_info['timecourse_folder'],f'task-{task}_screenshots')],
                            n_pix=an_info['n_pix'],
                            discard_volumes=an_info['discard_volumes'],
                            baseline_volumes_begin_end=an_info['baseline_volumes_begin_end'],
                            dm_edges_clipping=an_info['dm_edges_clipping'],
                            screen_size_cm=an_info['screen_size_cm'],
                            screen_distance_cm=an_info['screen_distance_cm'],
                            TR=an_info['TR'],
                            task_names=[task])                    
                    
                tc_runs=[]
                
                for run in all_runs:
                    mask_run = [np.load(mask_path) for mask_path in mask_paths if task in mask_path and f"run-{run}" in mask_path][0]
                    masked_idx = np.sum(mask_run[:index])
                    
                    tc_runs.append([np.load(tc_path)[masked_idx] for tc_path in tc_paths if task in tc_path and f"run-{run}" in tc_path][0])
                    
                    tc_runs[-1] *=(100/tc_runs[-1].mean(-1))[...,np.newaxis]
                    tc_runs[-1] += (tc_runs[-1].mean(-1)-np.median(tc_runs[-1][...,self.prf_stims[task].late_iso_dict[task]], axis=-1))[...,np.newaxis]
                  
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
                    

                
            tc_full = np.concatenate(tuple([tc[task] for task in tc]), axis=0) - 100
            tc_err = np.concatenate(tuple([tc_err[task] for task in tc_err]), axis=0)
            
            if an_info['crossvalidate'] and 'fit_runs' in an_info:
                tc_full_test = np.concatenate(tuple([tc_test[task] for task in tc_test]), axis=0) - 100
                tc_full_fit = np.concatenate(tuple([tc_fit[task] for task in tc_fit]), axis=0) - 100    
                #timecourse reliability stats
                vertex_info+=f"CV timecourse reliability stats\n"
                vertex_info+=f"fit-test timecourses pearson R {pearsonr(tc_full_test,tc_full_fit)[0]:.4f}\n"
                vertex_info+=f"fit-test timecourses R-squared {1-np.sum((tc_full_fit-tc_full_test)**2)/(tc_full_test.var(-1)*tc_full_test.shape[-1]):.4f}\n\n"
        
            preds = dict()
            prfs = dict()
            for model in self.only_models:
                gg = model_wrapper(model,
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
                
                params = subj_res['Results'][model][np.sum(subj_res['mask'][:index]),:-1]
                preds[model] = gg.return_prediction(*list(params))[0] - 100
                
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
                vertex_info+="\n"
                
            pl.ion()

            if self.click==0:
                self.f, self.axes = pl.subplots(2,2,figsize=(10, 10),frameon=False, gridspec_kw={'width_ratios': [8, 2], 'height_ratios': [1,1]})
                self.f.set_tight_layout(True)
            else:
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
                if ((subj_res['Processed Results']['RSq']['DoG'][index]-subj_res['Processed Results']['RSq']['Gauss'][index]) > 0.05) or ((subj_res['Processed Results']['RSq']['Norm_abcd'][index]-subj_res['Processed Results']['RSq']['CSS'][index]) > 0.05):
                    surround_effect = np.min([preds[model] for model in preds if 'Norm' in model or 'DoG' in model],axis=0)
                    surround_effect[surround_effect>0] = preds['Gauss'][surround_effect>0]
                    self.axes[0,0].fill_between(tseconds,
                                                 surround_effect, 
                                     preds['Gauss'], label='Surround suppression', alpha=0.2, color='gray')

          
            self.axes[0,0].legend(ncol=3, fontsize=8, loc=9)
            #self.axes[0,0].set_ylim(-3.5,6.5)
            self.axes[0,0].set_xlim(0,330)
            #self.axes[0,0].set_ylabel('% signal change')
            self.axes[0,0].set_title(f"Vertex {index} timecourse")
            
            if prfs['Norm_abcd'][0].min() < 0:
                self.axes[0,1].imshow(prfs['Norm_abcd'][0], cmap='RdBu_r', vmin=prfs['Norm_abcd'][0].min(), vmax=-prfs['Norm_abcd'][0].min())
            else:
                self.axes[0,1].imshow(prfs['Norm_abcd'][0], cmap='RdBu_r', vmax=prfs['Norm_abcd'][0].max(), vmin=-prfs['Norm_abcd'][0].max())
                
            
            self.axes[0,1].set_title(f"Vertex {index} pRF")
            
            self.axes[1,0].axis('off')
            self.axes[1,1].axis('off')
            self.axes[1,0].text(0,1,vertex_info, fontsize=10)
            
            self.click+=1
          
            return

          
        for space, space_res in self.main_dict.items():
            if 'fs' in space:
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
                        
                        if subj not in cortex.db.subjects:
                            print("subject not present in pycortex database. attempting to import...")
                            cortex.freesurfer.import_subj(subj, freesurfer_subject_dir=self.fs_dir, 
                                  whitematter_surf='smoothwm')
                        
                        p_r = subj_res['Processed Results']
                        models = p_r['RSq'].keys()
                        
                        if space != 'fsaverage':
                            tc_stats = subj_res['Timecourse Stats']
                            mask = subj_res['mask']


                  
                        if rois != 'all':
                            for key in p_r['Alpha']:
                                p_r['Alpha'][key] = roi_mask(self.idx_rois[subj][rois], p_r['Alpha'][key])
                                         

                        #output freesurefer-format polar angle maps to draw custom ROIs in freeview    
                        if self.output_freesurfer_maps:
                                          
                            lh_c = read_morph_data(opj(self.fs_dir, subj+'/surf/lh.curv'))
            
                            polar_freeview = np.copy(p_r['Polar Angle']['Norm_abcd'])#np.mean(polar, axis=-1)
                            ecc_freeview = np.copy(p_r['Eccentricity']['Norm_abcd'])#np.mean(ecc, axis=-1)
                                      
                            alpha_freeview = p_r['RSq']['Norm_abcd']* (tc_stats['Mean']>self.tc_min[subj])# rsq.max(-1) * (tc_stats['Mean']>self.tc_min[subj]) * (rsq.min(-1)>0)
            
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
                        if space == 'fsnative' and self.plot_stats_cortex and not plotted_stats[subj] :
                            
                            mean_ts_vert = cortex.Vertex2D(tc_stats['Mean'], mask*(tc_stats['Mean']>self.tc_min[subj]), subject=subj, cmap='Jet_2D_alpha')
                            var_ts_vert = cortex.Vertex2D(tc_stats['Variance'], mask*(tc_stats['Mean']>self.tc_min[subj]), subject=subj, cmap='Jet_2D_alpha')
                            tsnr_vert = cortex.Vertex2D(tc_stats['TSNR'], mask*(tc_stats['Mean']>self.tc_min[subj]), subject=subj, cmap='Jet_2D_alpha')
            
                            data_stats ={'mean':mean_ts_vert.raw, 'var':var_ts_vert.raw, 'tsnr':tsnr_vert.raw}
            
                            self.js_handle_dict[space][analysis][subj]['js_handle_stats'] = cortex.webgl.show(data_stats, pickerfun=clicker_function, with_curvature=False, with_labels=True, with_rois=True, with_borders=True, with_colorbar=True)
            
                            plotted_stats[subj] = True
                        
                        if self.plot_rois_cortex and not plotted_rois[subj]:
                            
                            
                            ds_rois = {}
                            data = np.zeros_like(mask).astype('int')
                            custom_rois_data = np.zeros_like(mask).astype('int')
            
                            for i, roi in enumerate(self.idx_rois[subj]):
            
                                roi_data = np.zeros_like(mask)
                                roi_data[self.idx_rois[subj][roi]] = 1
                                if 'custom' not in roi and 'visual' not in roi:
                                    data[self.idx_rois[subj][roi]] = i+1
                                if 'custom' in roi and 'Pole' not in roi:
                                    custom_rois_data[self.idx_rois[subj][roi]] = i+1

                                ds_rois[roi] = cortex.Vertex2D(roi_data, roi_data.astype('bool'), subj, cmap='RdBu_r_alpha').raw
            

            
                            ds_rois['Wang2015Atlas'] = cortex.Vertex2D(data, data.astype('bool'), subj, cmap='Retinotopy_HSV_2x_alpha').raw
                            ds_rois['Custom ROIs'] = cortex.Vertex2D(custom_rois_data, custom_rois_data.astype('bool'), subj, cmap='Retinotopy_HSV_2x_alpha').raw 
                            
                            self.js_handle_dict[space][analysis][subj]['js_handle_rois'] = cortex.webgl.show(ds_rois, pickerfun=clicker_function, with_curvature=False, with_labels=True, with_rois=True, with_borders=True, with_colorbar=True)
            
                            plotted_rois[subj] = True
                                                    

                            
            
                        if self.plot_rsq_cortex:              
                            ds_rsq = dict()
                            
                            
                            best_model = np.argmax([p_r['RSq'][model] for model in ['Gauss','Norm_abcd','CSS','DoG']],axis=0)

                            ds_rsq['Best model'] = cortex.Vertex2D(best_model, p_r['Alpha']['all'], subject=subj,
                                                                          vmin2=rsq_thresh, vmax2=0.6, cmap='BROYG_2D').raw 


                            for model in self.only_models:
                                ds_rsq[model] = cortex.Vertex2D(p_r['RSq'][model], p_r['Alpha'][model], subject=subj, 
                                                                vmin=rsq_thresh, vmax=0.8, vmin2=rsq_thresh, vmax2=0.6, cmap='Jet_2D_alpha').raw
                                
                            if 'CSS' in models and 'Gauss' in self.only_models:
                                ds_rsq['CSS - Gauss'] = cortex.Vertex2D(p_r['RSq']['CSS']-p_r['RSq']['Gauss'], p_r['Alpha']['all'], subject=subj,
                                                                          vmin=-0.1, vmax=0.1, vmin2=rsq_thresh, vmax2=0.6, cmap='Jet_2D_alpha').raw   
                                
                            if 'DoG' in models and 'Gauss' in self.only_models:
                                ds_rsq['DoG - Gauss'] = cortex.Vertex2D(p_r['RSq']['DoG']-p_r['RSq']['Gauss'], p_r['Alpha']['all'], subject=subj,
                                                                      vmin=-0.1, vmax=0.1, vmin2=rsq_thresh, vmax2=0.6, cmap='Jet_2D_alpha').raw
                            
                            if 'Norm_abcd' in self.only_models and 'CSS' in self.only_models and 'DoG' in self.only_models and 'Gauss' in self.only_models:

                                ds_rsq[f'Norm_abcd - Gauss'] = cortex.Vertex2D(p_r['RSq']['Norm_abcd']-p_r['RSq']['Gauss'], p_r['Alpha']['all'], subject=subj,
                                                                          vmin=-0.1, vmax=0.1, vmin2=rsq_thresh, vmax2=0.6, cmap='Jet_2D_alpha').raw

                                ds_rsq[f'Norm_abcd - DoG'] = cortex.Vertex2D(p_r['RSq']['Norm_abcd']-p_r['RSq']['DoG'], p_r['Alpha']['all'], subject=subj,
                                                                          vmin=-0.1, vmax=0.1, vmin2=rsq_thresh, vmax2=0.6, cmap='Jet_2D_alpha').raw

                                ds_rsq[f'Norm_abcd - CSS'] = cortex.Vertex2D(p_r['RSq']['Norm_abcd']-p_r['RSq']['CSS'], p_r['Alpha']['all'], subject=subj, 
                                                                          vmin=-0.1, vmax=0.1, vmin2=rsq_thresh, vmax2=0.6, cmap='Jet_2D_alpha').raw
 

                            for model in [model for model in self.only_models if 'Norm' in model and 'Norm_abcd' != model]:

                                ds_rsq[f'{model} - Norm_abcd'] = cortex.Vertex2D(p_r['RSq'][model]-p_r['RSq']['Norm_abcd'], p_r['Alpha']['all'], subject=subj,
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
                                ds_rsq_comp['Norm_abcd CV rsq (surface fit)'] = cortex.Vertex2D(p_r['RSq']['Norm_abcd'], p_r['RSq']['Norm_abcd'], subject=subj,
                                                                          vmin=rsq_thresh, vmax=0.6, vmin2=0.05, vmax2=rsq_thresh, cmap='Jet_2D_alpha').raw
                                self.js_handle_dict[space][analysis][subj]['js_handle_rsq_comp'] = cortex.webgl.show(ds_rsq_comp, pickerfun=clicker_function, with_curvature=False, with_labels=True, with_rois=True, with_borders=True, with_colorbar=True)
                            

                            self.js_handle_dict[space][analysis][subj]['js_handle_rsq'] = cortex.webgl.show(ds_rsq, pickerfun=clicker_function, with_curvature=False, with_labels=True, with_rois=True, with_borders=True, with_colorbar=True) 
                            
                        if self.plot_ecc_cortex:
                            ds_ecc = dict()
                            
                            for model in self.only_models:
                                ds_ecc[model] = cortex.Vertex2D(p_r['Eccentricity'][model], p_r['Alpha'][model], subject=subj, 
                                                                vmin=self.ecc_min, vmax=self.ecc_max-1, vmin2=rsq_thresh, vmax2=0.6, cmap='Jet_r_2D_alpha').raw
            
                            self.js_handle_dict[space][analysis][subj]['js_handle_ecc'] = cortex.webgl.show(ds_ecc, pickerfun=clicker_function, with_curvature=False, with_labels=True, with_rois=True, with_borders=True, with_colorbar=True)
            
                        if self.plot_polar_cortex:
                            ds_polar = dict()
                            
                            for model in self.only_models:
                                ds_polar[model] = cortex.Vertex2D(p_r['Polar Angle'][model], p_r['Alpha'][model], subject=subj, 
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
                                ds_polar_comp['Norm_abcd CV polar (surface fit)'] = cortex.Vertex2D(p_r['Polar Angle']['Norm_abcd'], p_r['RSq']['Norm_abcd'], subject=subj,
                                                                          vmin2=0.05, vmax2=rsq_thresh, cmap='Retinotopy_HSV_2x_alpha').raw
                                self.js_handle_dict[space][analysis][subj]['js_handle_polar_comp'] = cortex.webgl.show(ds_polar_comp, pickerfun=clicker_function, with_curvature=False, with_labels=True, with_rois=True, with_borders=True, with_colorbar=True)

                            
                            self.js_handle_dict[space][analysis][subj]['js_handle_polar'] = cortex.webgl.show(ds_polar, pickerfun=clicker_function, with_curvature=False, with_labels=True, with_rois=True, with_borders=True, with_colorbar=True)
            
                        if self.plot_size_cortex:
                            ds_size = dict()
                            
                            for model in self.only_models:
                                ds_size[model] = cortex.Vertex2D(p_r['Size (fwhmax)'][model], p_r['Alpha'][model], subject=subj, 
                                                                 vmin=0, vmax=6, vmin2=rsq_thresh, vmax2=0.6, cmap='Jet_2D_alpha').raw
                  
                            self.js_handle_dict[space][analysis][subj]['js_handle_size'] = cortex.webgl.show(ds_size, pickerfun=clicker_function, with_curvature=False, with_labels=True, with_rois=True, with_borders=True, with_colorbar=True)
            
            
                        if self.plot_amp_cortex:
                            ds_amp = dict()
                            
                            for model in self.only_models:
                                ds_amp[model] = cortex.Vertex2D(p_r['Amplitude'][model], p_r['Alpha'][model], subject=subj, 
                                                                vmin=-1, vmax=1, vmin2=rsq_thresh, vmax2=0.6, cmap='Jet_2D_alpha').raw
            
                            self.js_handle_dict[space][analysis][subj]['js_handle_amp'] = cortex.webgl.show(ds_amp, pickerfun=clicker_function, with_curvature=False, with_labels=True, with_rois=True, with_borders=True, with_colorbar=True)
                            
                        if self.plot_css_exp_cortex and 'CSS' in self.only_models:
                            ds_css_exp = dict()
                            
                            ds_css_exp['CSS Exponent'] = cortex.Vertex2D(p_r['CSS Exponent']['CSS'], p_r['Alpha']['CSS'], subject=subj, 
                                                                         vmin=0, vmax=1, vmin2=rsq_thresh, vmax2=0.6, cmap='Jet_2D_alpha').raw
            
                            self.js_handle_dict[space][analysis][subj]['js_handle_css_exp'] = cortex.webgl.show(ds_css_exp, pickerfun=clicker_function, with_curvature=False, with_labels=True, with_rois=True, with_borders=True, with_colorbar=True)
                            
                        if self.plot_surround_size_cortex:
                            ds_surround_size = dict()
                            
                            
                            for model in self.only_models:
                                if model == 'DoG':
                                    ds_surround_size['DoG'] = cortex.Vertex2D(p_r['Surround Size (fwatmin)']['DoG'], p_r['Alpha']['DoG'], subject=subj, 
                                                                         vmin=0, vmax=50, vmin2=rsq_thresh, vmax2=0.6, cmap='Jet_2D_alpha').raw
                                elif 'Norm' in model:
                                    ds_surround_size[model] = cortex.Vertex2D(p_r['Surround Size (fwatmin)'][model], p_r['Alpha'][model], subject=subj, 
                                                                         vmin=0, vmax=50, vmin2=rsq_thresh, vmax2=0.6, cmap='Jet_2D_alpha').raw                    
            
                            self.js_handle_dict[space][analysis][subj]['js_handle_surround_size'] = cortex.webgl.show(ds_surround_size, pickerfun=clicker_function, with_curvature=False, with_labels=True, with_rois=True, with_borders=True, with_colorbar=True)    

                        if self.plot_suppression_index_cortex:
                            ds_suppression_index = dict()
                            
                            
                            for model in self.only_models:
                                if model == 'DoG':
                                    ds_suppression_index[f'{model} (full)'] = cortex.Vertex2D(p_r['Suppression Index (full)'][model], p_r['Alpha'][model], subject=subj, 
                                                                         vmin=0, vmax=10, vmin2=rsq_thresh, vmax2=0.6, cmap='Jet_2D_alpha').raw
                                    ds_suppression_index[f'{model} (aperture)'] = cortex.Vertex2D(p_r['Suppression Index'][model], p_r['Alpha'][model], subject=subj, 
                                                                         vmin=0, vmax=1.5, vmin2=rsq_thresh, vmax2=0.6, cmap='Jet_2D_alpha').raw                                    
                                elif 'Norm' in model:
                                    ds_suppression_index[f'{model} (full)'] = cortex.Vertex2D(p_r['Suppression Index (full)'][model], p_r['Alpha'][model], subject=subj, 
                                                                         vmin=0, vmax=10, vmin2=rsq_thresh, vmax2=0.6, cmap='Jet_2D_alpha').raw
                                    ds_suppression_index[f'{model} (aperture)'] = cortex.Vertex2D(p_r['Suppression Index'][model], p_r['Alpha'][model], subject=subj, 
                                                                         vmin=0, vmax=1.5, vmin2=rsq_thresh, vmax2=0.6, cmap='Jet_2D_alpha').raw                      
            
                            self.js_handle_dict[space][analysis][subj]['js_handle_suppression_index'] = cortex.webgl.show(ds_suppression_index, pickerfun=clicker_function, with_curvature=False, with_labels=True, with_rois=True, with_borders=True, with_colorbar=True)    
                             
                        if self.plot_norm_baselines_cortex:
                            ds_norm_baselines = dict()
                            
                            
                            for model in [model for model in self.only_models if 'Norm' in model]:

                                ds_norm_baselines[f'{model} Param. B'] = cortex.Vertex2D(p_r['Norm Param. B'][model], p_r['Alpha'][model], subject=subj, 
                                                                             vmin=5, vmax=50, vmin2=rsq_thresh, vmax2=0.6, cmap='Jet_2D_alpha').raw                    
                                ds_norm_baselines[f'{model} Param. D'] = cortex.Vertex2D(p_r['Norm Param. D'][model], p_r['Alpha'][model], subject=subj, 
                                                                             vmin=15, vmax=90, vmin2=rsq_thresh, vmax2=0.6, cmap='Jet_2D_alpha').raw
                                ds_norm_baselines[f'{model} Ratio (B/D)'] = cortex.Vertex2D(p_r['Ratio (B/D)'][model], p_r['Alpha'][model], subject=subj, 
                                                                             vmin=0, vmax=1.5, vmin2=rsq_thresh, vmax2=0.6, cmap='Jet_2D_alpha').raw
                            
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
                
    def project_to_fsaverage(self, models, parameters, analysis_names = 'all', subject_ids='all'):
        if 'fsaverage' not in self.main_dict:
            self.main_dict['fsaverage'] = dd(lambda:dd(lambda:dd(lambda:dd(dict))))
        
        if parameters[0] != 'RSq':
            parameters.insert(0,'RSq')
            
        for space, space_res in self.main_dict.items():
            if 'fsnative' in space:
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
                        fsaverage_rsq = dict()
                        for parameter in parameters:
                            
                            fsaverage_param = dict()
                            
                            for subj, subj_res in subjects:
                                print(space+" "+analysis+" "+subj)
                                p_r = subj_res['Processed Results']
                                
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
                                
                                if parameter == 'RSq':
                                    fsaverage_rsq[subj] = np.nan_to_num(np.concatenate((lh_fsaverage_param,rh_fsaverage_param)))
                                    fsaverage_rsq[subj][fsaverage_rsq[subj]<0] = 0
                                    
                                    
                            #fsaverage_group_average = np.ma.average(np.array([fsaverage_param[sid] for sid in fsaverage_param]),
                            #                                      weights=np.array([fsaverage_rsq[sid] for sid in fsaverage_rsq]),
                            #                                      axis=0)
                            fsaverage_group_average = np.mean([fsaverage_param[sid] for sid in fsaverage_param], axis=0)
                            
                            for i in range(len(fsaverage_group_average)):
                                fsaverage_group_average[i] = weightstats.DescrStatsW(np.array([fsaverage_param[sid][i] for sid in fsaverage_param]),
                                                                                     weights=np.array([fsaverage_rsq[sid][i] for sid in fsaverage_rsq])).mean
                            
                            self.main_dict['fsaverage'][analysis]['fsaverage']['Processed Results'][parameter][model] = np.nan_to_num(fsaverage_group_average)
                            
                    for model in models:
                        if 'Norm' in model:
                            self.main_dict['fsaverage'][analysis]['fsaverage']['Processed Results']['Ratio (B/D)'][model] = self.main_dict['fsaverage'][analysis]['fsaverage']['Processed Results']['Norm Param. B'][model]/self.main_dict['fsaverage'][analysis]['fsaverage']['Processed Results']['Norm Param. D'][model]

                        
        return
    
    
    def quant_plots(self, x_parameter, y_parameter, rois, rsq_thresh, save_figures, 
                    analysis_names = 'all', subject_ids='all', ylim=None,
                    x_param_model=None, violin=False, scatter=False, diff_gauss=False,
                    means_only=False, stats_on_plot=False):
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

        for space, space_res in self.main_dict.items():
            if 'fsnative' in space:
                
                #upsampling correction: fsnative has approximately 3 times as many datapoints as original
                upsampling_corr_factor = 3
                #surrounds larger than this are excluded from surround size calculations
                w_max=40
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
                
                
                for analysis, analysis_res in analyses:       
                    if subject_ids == 'all':
                        subjects = [item for item in analysis_res.items()]
                    else:
                        subjects = [item for item in analysis_res.items() if item[0] in subject_ids]
                    
                    if len(subjects)>1:
                        subjects.append(('Group', {}))



                    for subj, subj_res in subjects:
                        print(space+" "+analysis+" "+subj)
                        x_ticks=[]
                        x_labels=[]    
                        bar_position = 0
                        
                        # binned eccentricity vs other parameters relationships       
            
                        model_colors = {'Gauss':'blue','CSS':'orange','DoG':'green','Norm_abcd':'red'}
                                                

    
                        for i, roi in enumerate(rois):                              
                            for model in self.only_models:                                

                                
                                if 'mean' not in analysis:
                                    if 'sub' in subj:
                                        if roi in self.idx_rois[subj]:
                                            if 'rsq' in y_parameter.lower():
                                                #comparing same vertices for model performance
                                                alpha[analysis][subj][model][roi] = (roi_mask(self.idx_rois[subj][roi], subj_res['Processed Results']['Alpha']['all']>rsq_thresh)) #* (subj_res['Processed Results']['Size (fwhmax)'][model]<w_max)
            
                                            else:
                                                #otherwise model specific alpha
                                                alpha[analysis][subj][model][roi] = (roi_mask(self.idx_rois[subj][roi], subj_res['Processed Results']['Alpha'][model]>rsq_thresh)) #* (subj_res['Processed Results']['Size (fwhmax)'][model]<w_max)
                                                
                                                
                                            if y_parameter == 'Surround Size (fwatmin)':
                                                #exclude too large surround (no surround)
                                                alpha[analysis][subj][model][roi] *= (subj_res['Processed Results']['Surround Size (fwatmin)'][model]<w_max)
                                            
                                        
                                        else:
                                            #if ROI is not defined
                                            #if Brain use all available vertices
                                            if roi == 'Brain':
                                                if 'rsq' in y_parameter.lower():
                                                    alpha[analysis][subj][model][roi] = (subj_res['Processed Results']['Alpha']['all'])>rsq_thresh
                                                else:
                                                    alpha[analysis][subj][model][roi] = (subj_res['Processed Results']['Alpha'][model])>rsq_thresh#np.zeros_like(subj_res['Processed Results']['Alpha'][model]).astype('bool')
                                            #if all, combine all other rois
                                            elif roi == 'all':
                                                if 'rsq' in y_parameter.lower():
                                                    #comparing same vertices for model performance
                                                    alpha[analysis][subj][model][roi] = (roi_mask(np.concatenate(tuple([self.idx_rois[subj][r] for r in rois if 'custom' in r])), subj_res['Processed Results']['Alpha']['all']>rsq_thresh)) #* (subj_res['Processed Results']['Size (fwhmax)'][model]<w_max)
                
                                                else:
                                                    #otherwise model specific alpha
                                                    alpha[analysis][subj][model][roi] = (roi_mask(np.concatenate(tuple([self.idx_rois[subj][roi] for roi in rois])), subj_res['Processed Results']['Alpha'][model]>rsq_thresh)) #* (subj_res['Processed Results']['Size (fwhmax)'][model]<w_max)
                 
                                            else:
                                                #, otherwise none
                                                print("undefined ROI")
                                                alpha[analysis][subj][model][roi] = np.zeros_like(subj_res['Processed Results']['Alpha'][model]).astype('bool')
                                                
         
                                        #if 'ccrsq' in y_parameter.lower():
                                        #    alpha[analysis][subj][model][roi] *= (subj_res['Processed Results'][y_parameter][model]>0)
                                       
                                        if x_param_model != None:
                                            
                                            alpha[analysis][subj][model][roi] *= np.isfinite(subj_res['Processed Results'][x_parameter][x_param_model])
                                            alpha[analysis][subj][model][roi] *= np.isfinite(subj_res['Processed Results'][y_parameter][model]) 
                                            
                                            y_par[analysis][subj][model][roi] = subj_res['Processed Results'][y_parameter][model][alpha[analysis][subj][model][roi]]
                                            x_par[analysis][subj][model][roi] = subj_res['Processed Results'][x_parameter][x_param_model][alpha[analysis][subj][model][roi]]                                          
                                        
                                        
                                        else:
                                            
                                            #remove nans and infinities
                                            alpha[analysis][subj][model][roi] *= np.isfinite(subj_res['Processed Results'][x_parameter][model])                                    
                                            alpha[analysis][subj][model][roi] *= np.isfinite(subj_res['Processed Results'][y_parameter][model])
                                            
                                            if diff_gauss:
                                                y_par[analysis][subj][model][roi] = (subj_res['Processed Results'][y_parameter][model][alpha[analysis][subj][model][roi]])-subj_res['Processed Results'][y_parameter]['Gauss'][alpha[analysis][subj][model][roi]]
                                            else:
                                                y_par[analysis][subj][model][roi] = (subj_res['Processed Results'][y_parameter][model][alpha[analysis][subj][model][roi]])
                                            
                                            
                                            x_par[analysis][subj][model][roi] = subj_res['Processed Results'][x_parameter][model][alpha[analysis][subj][model][roi]]
                                            
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
                                            if x_param_model != None:
                                                rsq_x_param = np.copy(subj_res['Processed Results']['RSq'][x_param_model][alpha[analysis][subj][model][roi]])
                                                rsq_x_param[subj_res['Processed Results']['RSq'][x_param_model][alpha[analysis][subj][model][roi]]<0] = 0
                                                
                                                rsq[analysis][subj][model][roi] += rsq_x_param
                                                rsq[analysis][subj][model][roi] /= 2
        
                                    else:
                                        #group stats
                                        x_par_group = np.concatenate(tuple([x_par[analysis][sid][model][roi] for sid in x_par[analysis] if 'sub' in sid]))
                                        
                                        y_par_group = np.concatenate(tuple([y_par[analysis][sid][model][roi] for sid in y_par[analysis] if 'sub' in sid]))
                                        rsq_group = np.concatenate(tuple([rsq[analysis][sid][model][roi] for sid in rsq[analysis] if 'sub' in sid]))
                                        
                                        x_par[analysis][subj][model][roi] = np.copy(x_par_group)
                                        y_par[analysis][subj][model][roi] = np.copy(y_par_group)
                                        rsq[analysis][subj][model][roi] = np.copy(rsq_group)
                                    
                                else:
                                    #mean analysis
                                    ans = [an[0] for an in analyses if 'mean' not in an[0]]
                                    alpha[analysis][subj][model][roi] = np.hstack(tuple([alpha[an][subj][model][roi] for an in ans]))
                                    x_par[analysis][subj][model][roi] = np.hstack(tuple([x_par[an][subj][model][roi] for an in ans]))
                                    y_par[analysis][subj][model][roi] = np.hstack(tuple([y_par[an][subj][model][roi] for an in ans]))
                                    rsq[analysis][subj][model][roi] = np.hstack(tuple([rsq[an][subj][model][roi] for an in ans]))
                                    

                        for i, roi in enumerate(rois):
                            if i>0:
                                bar_position += (2*bar_or_violin_width)
                                
                            label_position = bar_position+(0.5*bar_or_violin_width)*(len(self.only_models)-1)                     
                            if 'hV4' in roi and len(self.only_models)>1:
                                label_position+=bar_or_violin_width
                                
                            x_ticks.append(label_position)
                            x_labels.append(roi.replace('custom.','')+'\n')   
                            
                            for model in self.only_models:    
                                
                                    
                                pl.figure(f"{analysis} {subj} Mean {y_parameter}", figsize=(8, 8), frameon=False)
 
                                pl.ylabel(f"{subj} Mean {y_parameter}")
                                
                                if ylim != None:
                                    pl.ylim(ylim[0],ylim[1])
                                                                           
                                full_roi_stats = weightstats.DescrStatsW(y_par[analysis][subj][model][roi],
                                                        weights=rsq[analysis][subj][model][roi])
                                
                                bar_height = full_roi_stats.mean
                                bar_err = np.abs(full_roi_stats.zconfint_mean(alpha=0.05/upsampling_corr_factor) - bar_height).reshape(2,1)
                                

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
                                        
                                        pl.bar(bar_position, bar_height, width=bar_or_violin_width, yerr=bar_err, 
                                               edgecolor='black', label=model, color=model_colors[model])
                                    

                                    
                                    
                                    
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
                                            
                                            if ylim!=None:
                                                axis_height = ylim[1]-ylim[0]
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
                                                
                                                for ccc in range(100000):                                         
                                                                                                               
                                                    #if ccc<50:
                                                    #    pl.figure(f"null distribs {analysis} {subj} {mod} {roi}")
                                                    #    pl.hist(null_distrib, bins=50, color=f"C{i+4}")
                                                    
                                                    
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
    
                                                #pl.figure(f"{analysis} {subj} Mean {y_parameter}", figsize=(8, 8), frameon=False) 
                                                        
                                                    
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
                                    pl.bar(bar_position, bar_height, width=bar_or_violin_width, yerr=bar_err, 
                                       edgecolor='black', color=f"C{i+4}")                                        
                                
                                bar_position += bar_or_violin_width

                                pl.xticks(x_ticks,x_labels)
                                handles, labels = pl.gca().get_legend_handles_labels()
                                by_label = dict(zip(labels, handles))
                                if len(self.only_models) == 1:
                                    pl.legend(by_label.values(), by_label.keys())
                                    
                                if save_figures:
                                    if len(analyses)>1:
                                        os.makedirs(f"/Users/marcoaqil/PRFMapping/Figures/{analysis}", exist_ok=True)
                                        pl.savefig(f"/Users/marcoaqil/PRFMapping/Figures/{analysis}/{subj} {model} Mean {y_parameter.replace('/','')}.pdf", dpi=300, bbox_inches='tight')
                                                          
                                    else:                      
                                        pl.savefig(f"/Users/marcoaqil/PRFMapping/Figures/{subj} {model} Mean {y_parameter.replace('/','')}.pdf", dpi=300, bbox_inches='tight')

                        
                        if not means_only:
                            for i, roi in enumerate(rois):
                                ###################
                                #x vs y param by ROI
                                pl.figure(f"{analysis} {subj} {roi.replace('custom.','')} {y_parameter} VS {x_parameter}", figsize=(8, 8), frameon=False)
                                #pl.gca().set_yscale('log')
                                #pl.gca().set_xscale('log')
                                
                                if ylim != None:
                                    pl.ylim(ylim[0],ylim[1])
                                    
                                for model in self.only_models:
                                    #bin stats
                                    x_par_sorted = np.argsort(x_par[analysis][subj][model][roi])
                                    split_x_par_bins = np.array_split(x_par_sorted, 8) #8
                                   
                                    for x_par_quantile in split_x_par_bins:
                                        #ddof_correction_quantile = ddof_corr*np.sum(rsq[analysis][subj][model][roi][x_par_quantile])
                                        
                                        y_par_stats[analysis][subj][model][roi].append(weightstats.DescrStatsW(y_par[analysis][subj][model][roi][x_par_quantile],
                                                                                              weights=rsq[analysis][subj][model][roi][x_par_quantile]))
                
                                        x_par_stats[analysis][subj][model][roi].append(weightstats.DescrStatsW(x_par[analysis][subj][model][roi][x_par_quantile],
                                                                                              weights=rsq[analysis][subj][model][roi][x_par_quantile]))
                                        
    
                                    if not scatter:                
                                        try:
                                            WLS = LinearRegression()
                                            WLS.fit(x_par[analysis][subj][model][roi].reshape(-1, 1), y_par[analysis][subj][model][roi], sample_weight=rsq[analysis][subj][model][roi])
                                            if len(self.only_models)>1:
                                                p=pl.plot([ss.mean for ss in x_par_stats[analysis][subj][model][roi]],
                                                WLS.predict(np.array([ss.mean for ss in x_par_stats[analysis][subj][model][roi]]).reshape(-1, 1)),
                                                color=model_colors[model])
                                            else:
                                                p=pl.plot([ss.mean for ss in x_par_stats[analysis][subj][model][roi]],
                                                WLS.predict(np.array([ss.mean for ss in x_par_stats[analysis][subj][model][roi]]).reshape(-1, 1)),
                                                color=f"C{i+4}")                                        
                                                    
                                            print(roi+" "+model+" "+str(WLS.score(x_par[analysis][subj][model][roi].reshape(-1, 1), y_par[analysis][subj][model][roi], sample_weight=rsq[analysis][subj][model][roi])))
                                        except Exception as e:
                                            print(e)
                                            pass
                                        
                                        try:
                                            if len(self.only_models)>1:
                                                pl.errorbar([ss.mean for ss in x_par_stats[analysis][subj][model][roi]],
                                                [ss.mean for ss in y_par_stats[analysis][subj][model][roi]],
                                                yerr=np.array([np.abs(ss.zconfint_mean(alpha=0.05/upsampling_corr_factor)-ss.mean) for ss in y_par_stats[analysis][subj][model][roi]]).T,
                                                xerr=np.array([np.abs(ss.zconfint_mean(alpha=0.05/upsampling_corr_factor)-ss.mean) for ss in x_par_stats[analysis][subj][model][roi]]).T,
                                                fmt='s',  mec='black', label=model, color=p[-1].get_color())#, mfc=model_colors[model], ecolor=model_colors[model])
                                            else:
                                                pl.errorbar([ss.mean for ss in x_par_stats[analysis][subj][model][roi]],
                                                [ss.mean for ss in y_par_stats[analysis][subj][model][roi]],
                                                yerr=np.array([np.abs(ss.zconfint_mean(alpha=0.05/upsampling_corr_factor)-ss.mean) for ss in y_par_stats[analysis][subj][model][roi]]).T,
                                                xerr=np.array([np.abs(ss.zconfint_mean(alpha=0.05/upsampling_corr_factor)-ss.mean) for ss in x_par_stats[analysis][subj][model][roi]]).T,
                                                fmt='s',  mec='black', label=roi.replace('custom.',''), color=p[-1].get_color())#, mfc=model_colors[model], ecolor=model_colors[model])
                                           
                                                
                                        except Exception as e:
                                            print(e)
                                            pass
                                    else:
                                        try:
                                            if len(self.only_models)>1:
                                                p=pl.scatter(x_par[analysis][subj][model][roi], y_par[analysis][subj][model][roi], marker='o', s=1,
                                                color=model_colors[model], label=model)
                                            else:
                                                p=pl.scatter(x_par[analysis][subj][model][roi], y_par[analysis][subj][model][roi], marker='o', s=1,
                                                color=f"C{i+4}", label=roi.replace('custom.',''))  
                                        except Exception as e:
                                            print(e)
                                            pass                            
                            
                                pl.xlabel(f"{x_parameter}")
                                pl.ylabel(f"{subj} {roi.replace('custom.','')} {y_parameter}")
                                pl.legend(loc=0)
                                if save_figures:
                                    if len(analyses)>1:
                                        os.makedirs(f"/Users/marcoaqil/PRFMapping/Figures/{analysis}", exist_ok=True)
                                        pl.savefig(f"/Users/marcoaqil/PRFMapping/Figures/{analysis}/{subj} {roi.replace('custom.','')} {y_parameter.replace('/','')} VS {x_parameter.replace('/','')}.pdf", dpi=300, bbox_inches='tight')
                                                          
                                    else:                                 
                                        pl.savefig(f"/Users/marcoaqil/PRFMapping/Figures/{subj} {roi.replace('custom.','')} {y_parameter.replace('/','')} VS {x_parameter.replace('/','')}.pdf", dpi=300, bbox_inches='tight')


                        ########params by model
                        if not means_only:        
                            for model in self.only_models:
                                pl.figure(f"{analysis} {subj} {model} {y_parameter} VS {x_parameter}", figsize=(8, 8), frameon=False)
                                #pl.gca().set_yscale('log')
                                #pl.gca().set_xscale('log')
                                
                                if ylim != None:
                                    pl.ylim(ylim[0],ylim[1])
                                #pl.xlim(0.2,4)
                                for i, roi in enumerate(rois):
                                    try:
                                        WLS = LinearRegression()
                                        WLS.fit(x_par[analysis][subj][model][roi].reshape(-1, 1), y_par[analysis][subj][model][roi], sample_weight=rsq[analysis][subj][model][roi])
                                        
                                        bootstrap_fits = []
                                        for c in range(100):
                                            
                                            sample = np.random.randint(0, len(x_par[analysis][subj][model][roi]), int(0.9*len(x_par[analysis][subj][model][roi])))
                                            
                                            WLS_bootstrap = LinearRegression()
                                            WLS_bootstrap.fit(x_par[analysis][subj][model][roi][sample].reshape(-1, 1), y_par[analysis][subj][model][roi][sample], sample_weight=rsq[analysis][subj][model][roi][sample])
                                            
                                            bootstrap_fits.append(WLS_bootstrap.predict(np.array([ss.mean for ss in x_par_stats[analysis][subj][model][roi]]).reshape(-1, 1)))
                                    
                                        p=pl.plot([ss.mean for ss in x_par_stats[analysis][subj][model][roi]],
                                            WLS.predict(np.array([ss.mean for ss in x_par_stats[analysis][subj][model][roi]]).reshape(-1, 1)),
                                            color=f"C{i+4}", label=roi.replace('custom.',''))
                                            #color=roi_colors[roi]) 
                                                                          
                                                   
                                        print(f"{roi} {model} WLS score {WLS.score(x_par[analysis][subj][model][roi].reshape(-1, 1), y_par[analysis][subj][model][roi], sample_weight=rsq[analysis][subj][model][roi])}")
                                    except Exception as e:
                                        print(e)
                                        pass
                                    
                                    try:
                                        #conf interval shading
                                        pl.fill_between([ss.mean for ss in x_par_stats[analysis][subj][model][roi]],
                                                        np.min(bootstrap_fits, axis=0),
                                                        np.max(bootstrap_fits, axis=0),
                                                        alpha=0.2, color=p[-1].get_color(), label=roi.replace('custom.',''))
                                        
                                        #data points with errors
                                        # pl.errorbar([ss.mean for ss in x_par_stats[analysis][subj][model][roi]],
                                        # [ss.mean for ss in y_par_stats[analysis][subj][model][roi]],
                                        # yerr=np.array([np.abs(ss.zconfint_mean(alpha=0.05)-ss.mean) for ss in y_par_stats[analysis][subj][model][roi]]).T,
                                        # xerr=np.array([np.abs(ss.zconfint_mean(alpha=0.05)-ss.mean) for ss in x_par_stats[analysis][subj][model][roi]]).T,
                                        # fmt='s', mec='black', label=roi.replace('custom.',''), color=p[-1].get_color())#, mfc=roi_colors[roi], ecolor=roi_colors[roi])
                                    except Exception as e:
                                        print(e)
                                        pass
                                pl.xlabel(f"{x_parameter}")
                                pl.ylabel(f"{subj} {model} {y_parameter}")
                                handles, labels = pl.gca().get_legend_handles_labels()
                                #this is to merge multiple labels
                                #by_label = dict(zip(labels, handles))
    
                                legend_dict = dd(list)
                                for cc, label in enumerate(labels):
                                    legend_dict[label].append(handles[cc])
                                    
                                for label in legend_dict:
                                    legend_dict[label] = tuple(legend_dict[label])
    
                                pl.legend([legend_dict[label] for label in legend_dict], legend_dict.keys())                      
    
                                if save_figures:
                                    if len(analyses)>1:
                                        os.makedirs(f"/Users/marcoaqil/PRFMapping/Figures/{analysis}", exist_ok=True)
                                        pl.savefig(f"/Users/marcoaqil/PRFMapping/Figures/{analysis}/{subj} {model} {y_parameter.replace('/','')} VS {x_parameter.replace('/','')}.pdf", dpi=300, bbox_inches='tight')
                                                          
                                    else:                                 
                                        pl.savefig(f"/Users/marcoaqil/PRFMapping/Figures/{subj} {model} {y_parameter.replace('/','')} VS {x_parameter.replace('/','')}.pdf", dpi=300, bbox_inches='tight')


        return
    
    
    def ecc_surround_roi_plots(self, rois, rsq_thresh, save_figures, analysis_names = 'all'):
        
        pl.rcParams.update({'font.size': 16})
        pl.rcParams.update({'pdf.fonttype':42})
        for space, space_res in self.main_dict.items():
            if 'fs' in space:
                if analysis_names == 'all':
                    analyses = space_res.items()
                else:
                    analyses = [item for item in space_res.items() if item[0] in analysis_names] 
                for analysis, analysis_res in analyses:       
                    for subj, subj_res in analysis_res.items():
                        print(space+" "+analysis+" "+subj)
            
                        # binned eccentricity vs other parameters relationships       
            
                        # model_colors = {'Gauss':'blue','CSS':'orange','DoG':'green','Norm':'red'}
                                                
                        # #model_symbols = {'Gauss':'^','CSS':'o','DoG':'v','Norm':'D'}
                        # roi_colors = dd(lambda:'blue')
                        # roi_colors['custom.V1']= 'black'
                        # roi_colors['custom.V2']= 'red'
                        # roi_colors['custom.V3']= 'pink'

                        # #model_symbols = {'Gauss':'^','CSS':'o','DoG':'v','Norm':'D'}
                        # roi_colors = dd(lambda:'blue')
                        # roi_colors['custom.V1']= 'black'
                        # roi_colors['custom.V2']= 'red'
                        # roi_colors['custom.V3']= 'pink'
                        # roi_colors['custom.hV4']='blue'
                        # roi_colors['custom.V3AB']='orange'
                        # #roi_colors['custom.']
                        # symbol={}
                        # symbol['custom.V1'] = 's'
                        # symbol['custom.V2'] = '^'
                        # symbol['custom.V3'] = 'v'
                        # symbol['custom.LO'] = 'D'
                        # symbol['custom.TO'] = 'o'
                        
                        fw_atmin_stats = dd(lambda:dd(list))
                        ecc_stats = dd(lambda:dd(list))
                        
                        #exclude surrounds sizes larger than this (no surround)
                        w_max=60
            
                        for roi in rois:
            
                            pl.figure(subj+roi+' fw_atmin', figsize=(8, 8), frameon=False)
           
                            for model in [k for k in subj_res['Processed Results']['Surround Size (fwatmin)'].keys() if k in self.only_models]:
            
                                #model-specific alpha? or all models same alpha?
                                alpha_roi = roi_mask(self.idx_rois[subj][roi], subj_res['Processed Results']['Alpha'][model]>rsq_thresh) * (subj_res['Processed Results']['Surround Size (fwatmin)'][model]<w_max)
                                
                                ecc_model_roi = subj_res['Processed Results']['Eccentricity'][model][alpha_roi]
                                fwatmin_model_roi = subj_res['Processed Results']['Surround Size (fwatmin)'][model][alpha_roi]
                                rsq_model_roi = subj_res['Processed Results']['RSq'][model][alpha_roi]
                                
                                ecc_sorted = np.argsort(ecc_model_roi)
                                split_ecc_bins = np.array_split(ecc_sorted, 10)
                               
                                for ecc_quantile in split_ecc_bins:
                                    fw_atmin_stats[roi][model].append(weightstats.DescrStatsW(fwatmin_model_roi[ecc_quantile],
                                                                                          weights=rsq_model_roi[ecc_quantile]))
            
                                    ecc_stats[roi][model].append(weightstats.DescrStatsW(ecc_model_roi[ecc_quantile],
                                                                                          weights=rsq_model_roi[ecc_quantile]))
            
                       
                                WLS = LinearRegression()
                                WLS.fit(ecc_model_roi.reshape(-1, 1), fwatmin_model_roi, sample_weight=rsq_model_roi)
                                p=pl.plot([ss.mean for ss in ecc_stats[roi][model]],
                                        WLS.predict(np.array([ss.mean for ss in ecc_stats[roi][model]]).reshape(-1, 1)))
                                        #,color=model_colors[model])
                                            
                                print(roi+" "+model+" "+str(WLS.score(ecc_model_roi.reshape(-1, 1), fwatmin_model_roi, sample_weight=rsq_model_roi)))
            
                                pl.errorbar([ss.mean for ss in ecc_stats[roi][model]],
                                   [ss.mean for ss in fw_atmin_stats[roi][model]],
                                   yerr=np.array([np.abs(ss.zconfint_mean(alpha=0.05)-ss.mean) for ss in fw_atmin_stats[roi][model]]).T,
                                   xerr=np.array([np.abs(ss.zconfint_mean(alpha=0.05)-ss.mean) for ss in ecc_stats[roi][model]]).T,
                                   fmt='s',  mec='black', label=model,color=p[-1].get_color())#, mfc=model_colors[model],ecolor=model_colors[model])
            
                            pl.xlabel('Eccentricity (degrees)')
                            pl.ylabel(f"{subj} {roi.replace('custom.','')} pRF Surround Size (degrees)")
                            pl.legend(loc=0)
                            if save_figures:
                                pl.savefig('/Users/marcoaqil/PRFMapping/Figures/'+subj+'_'+
                                           roi.replace('custom.','')+'_fw-atmin.pdf', dpi=300, bbox_inches='tight')


                        fw_atmin_stats = dd(lambda:dd(list))
                        ecc_stats = dd(lambda:dd(list))
                                
                        for model in [k for k in subj_res['Processed Results']['Surround Size (fwatmin)'].keys() if k in self.only_models]:
                            pl.figure(subj+model+' fw_atmin', figsize=(8, 8), frameon=False)
                            for i, roi in enumerate(rois):
                                #model-specific alpha? or all models same alpha?
                                alpha_roi = roi_mask(self.idx_rois[subj][roi], subj_res['Processed Results']['Alpha'][model]>rsq_thresh) * (subj_res['Processed Results']['Surround Size (fwatmin)'][model]<w_max)
                                
                                ecc_model_roi = subj_res['Processed Results']['Eccentricity'][model][alpha_roi]
                                fwatmin_model_roi = subj_res['Processed Results']['Surround Size (fwatmin)'][model][alpha_roi]
                                rsq_model_roi = subj_res['Processed Results']['RSq'][model][alpha_roi]
                                
                                ecc_sorted = np.argsort(ecc_model_roi)
                                split_ecc_bins = np.array_split(ecc_sorted, 10)
                               
                                for ecc_quantile in split_ecc_bins:
                                    fw_atmin_stats[roi][model].append(weightstats.DescrStatsW(fwatmin_model_roi[ecc_quantile],
                                                                                          weights=rsq_model_roi[ecc_quantile]))
            
                                    ecc_stats[roi][model].append(weightstats.DescrStatsW(ecc_model_roi[ecc_quantile],
                                                                                          weights=rsq_model_roi[ecc_quantile]))
            
                       
                                WLS = LinearRegression()
                                WLS.fit(ecc_model_roi.reshape(-1, 1), fwatmin_model_roi, sample_weight=rsq_model_roi)
                                p=pl.plot([ss.mean for ss in ecc_stats[roi][model]],
                                        WLS.predict(np.array([ss.mean for ss in ecc_stats[roi][model]]).reshape(-1, 1)),
                                        color=f"C{i+4}")#,
                                        #color=roi_colors[roi])
                                            
                                print(roi+" "+model+" "+str(WLS.score(ecc_model_roi.reshape(-1, 1), fwatmin_model_roi, sample_weight=rsq_model_roi)))
            
                                pl.errorbar([ss.mean for ss in ecc_stats[roi][model]],
                                   [ss.mean for ss in fw_atmin_stats[roi][model]],
                                   yerr=np.array([np.abs(ss.zconfint_mean(alpha=0.05)-ss.mean) for ss in fw_atmin_stats[roi][model]]).T,
                                   xerr=np.array([np.abs(ss.zconfint_mean(alpha=0.05)-ss.mean) for ss in ecc_stats[roi][model]]).T,
                                   fmt='s', mec='black', label=roi.replace('custom.',''),color=p[-1].get_color())#, mfc=roi_colors[roi], ecolor=roi_colors[roi])
            
                            pl.xlabel('Eccentricity (degrees)')
                            pl.ylabel(f'{subj} {model} pRF Surround Size (degrees)')
                            pl.legend(loc=0)
                            if save_figures:
                                pl.savefig('/Users/marcoaqil/PRFMapping/Figures/'+subj+'_'+
                                           model+'_fw-atmin.pdf', dpi=300, bbox_inches='tight')            
            
        return     

       
    def ecc_css_exp_roi_plots(self, rois, rsq_thresh, save_figures, analysis_names = 'all'):
        
        pl.rcParams.update({'font.size': 16})
        pl.rcParams.update({'pdf.fonttype':42})
        for space, space_res in self.main_dict.items():
            if 'fs' in space:
                if analysis_names == 'all':
                    analyses = space_res.items()
                else:
                    analyses = [item for item in space_res.items() if item[0] in analysis_names] 
                for analysis, analysis_res in analyses:       
                    for subj, subj_res in analysis_res.items():
                        print(space+" "+analysis+" "+subj)
            
                        # binned eccentricity vs other parameters relationships       
            
                        # roi_colors = dd(lambda:'blue')
                        # roi_colors['custom.V1']= 'black'
                        # roi_colors['custom.V2']= 'red'
                        # roi_colors['custom.V3']= 'pink'
            
                        css_exp_stats = dd(lambda:dd(list))
                        ecc_stats = dd(lambda:dd(list))
                        
                        pl.figure('css_exp', figsize=(8, 8), frameon=False)
                        for i, roi in enumerate(rois):

                            if 'CSS' in subj_res['Processed Results']['RSq'].keys():                                
                                model = 'CSS'
                                #model-specific alpha? or all models same alpha?
                                alpha_roi = roi_mask(self.idx_rois[subj][roi], subj_res['Processed Results']['Alpha'][model]>rsq_thresh)
                                
                                ecc_model_roi = subj_res['Processed Results']['Eccentricity'][model][alpha_roi]
                                css_exp_roi = subj_res['Processed Results']['CSS Exponent'][model][alpha_roi]
                                rsq_model_roi = subj_res['Processed Results']['RSq'][model][alpha_roi]
                                
                                ecc_sorted = np.argsort(ecc_model_roi)
                                split_ecc_bins = np.array_split(ecc_sorted, 10)
                               
                                for ecc_quantile in split_ecc_bins:
                                    css_exp_stats[roi][model].append(weightstats.DescrStatsW(css_exp_roi[ecc_quantile],
                                                                                          weights=rsq_model_roi[ecc_quantile]))
            
                                    ecc_stats[roi][model].append(weightstats.DescrStatsW(ecc_model_roi[ecc_quantile],
                                                                                          weights=rsq_model_roi[ecc_quantile]))
            
                       
                                # WLS = LinearRegression()
                                # WLS.fit(ecc_model_roi.reshape(-1, 1), css_exp_roi, sample_weight=rsq_model_roi)
                                # pl.plot([ss.mean for ss in ecc_stats[roi][model]],
                                #         WLS.predict(np.array([ss.mean for ss in ecc_stats[roi][model]]).reshape(-1, 1)),
                                #         color=roi_colors[roi])
                                            
                                # print(roi+" "+model+" "+str(WLS.score(ecc_model_roi.reshape(-1, 1), css_exp_roi, sample_weight=rsq_model_roi)))
            
                                pl.errorbar([ss.mean for ss in ecc_stats[roi][model]],
                                   [ss.mean for ss in css_exp_stats[roi][model]],
                                   yerr=np.array([np.abs(ss.zconfint_mean(alpha=0.05)-ss.mean) for ss in css_exp_stats[roi][model]]).T,
                                   xerr=np.array([np.abs(ss.zconfint_mean(alpha=0.05)-ss.mean) for ss in ecc_stats[roi][model]]).T,
                                   fmt='s', mec='black', label=roi.replace('custom.',''),color=f"C{i+4}")#), mfc=roi_colors[roi], ecolor=roi_colors[roi])
            
                            pl.xlabel('Eccentricity (degrees)')
                            pl.ylabel(subj+' CSS Exponent')
                            pl.legend(loc=0)
                            if save_figures:
                                pl.savefig('/Users/marcoaqil/PRFMapping/Figures/'+subj+'_'+
                                           roi.replace('custom.','')+'_css-exp.pdf', dpi=300, bbox_inches='tight')            
        return     

             
    def ecc_norm_baselines_roi_plots(self, rois, rsq_thresh, save_figures, analysis_names='all'):
        
        pl.rcParams.update({'font.size': 16})
        pl.rcParams.update({'pdf.fonttype':42})
        for space, space_res in self.main_dict.items():
            if 'fs' in space:
                if analysis_names == 'all':
                    analyses = space_res.items()
                else:
                    analyses = [item for item in space_res.items() if item[0] in analysis_names] 
                for analysis, analysis_res in analyses:         
                    for subj, subj_res in analysis_res.items():
                        print(space+" "+analysis+" "+subj)
            
                        # binned eccentricity vs other parameters relationships       
            
                        # roi_colors = dd(lambda:'blue')
                        # roi_colors['custom.V1']= 'black'
                        # roi_colors['custom.V2']= 'red'
                        # roi_colors['custom.V3']= 'pink'
                        
                        params = {}
                        params['Norm Param. B'] = 'o'
                        params['Norm Param. D'] = 'o'
                        params['Ratio (B/D)'] = 'o'    
                        #params['AD-BC (nonlinearity?)'] = 'o'
                        
                        
                        symbol={}
                        symbol['custom.V1'] = 's'
                        symbol['custom.V2'] = '^'
                        symbol['custom.V3'] = 'v'
                        symbol['custom.LO'] = 'D'
                        symbol['custom.TO'] = 'o'
            

                        for param in params:
                            for model in [k for k in self.only_models if 'Norm' in k]:
                                subj_res['Processed Results']['AD-BC (nonlinearity?)'][model]=np.zeros(subj_res['mask'].shape)
                                subj_res['Processed Results']['AD-BC (nonlinearity?)'][model][subj_res['mask']] = subj_res['Results'][model][:,3]*subj_res['Results'][model][:,8]- subj_res['Results'][model][:,5]*subj_res['Results'][model][:,7]
                                
                                
                                ecc_stats = dd(lambda:dd(list)) 
                                norm_baselines_stats = dd(lambda:dd(list))
                                
                                bar_position = 0
                                x_ticks=[]
                                x_labels=[] 
                                
                                for i,roi in enumerate(rois):
                                    pl.figure(analysis+param+subj+model, figsize=(8, 8), frameon=False)
                                    if param == 'Norm Param. B':
                                        pl.ylim(-5,100)
                                    elif param == 'Norm Param. D':
                                        pl.ylim(-5,100)
                                    elif param == 'Ratio (B/D)':
                                        pl.ylim(-1,8)
                                        

                                    alpha_roi = roi_mask(self.idx_rois[subj][roi], subj_res['Processed Results']['Alpha'][model]>rsq_thresh)#* (subj_res['Processed Results']['RSq']['DoG']<subj_res['Processed Results']['RSq']['Gauss'])
                                    
                                    ecc_model_roi = subj_res['Processed Results']['Eccentricity'][model][alpha_roi]
                                    rsq_model_roi = subj_res['Processed Results']['RSq'][model][alpha_roi]
                                    
                                    ecc_sorted = np.argsort(ecc_model_roi)
                                    split_ecc_bins = np.array_split(ecc_sorted, 8)
                                    
                                    norm_baselines_roi = subj_res['Processed Results'][param][model][alpha_roi]
                                   
                                    for ecc_quantile in split_ecc_bins:
                                        norm_baselines_stats[roi][param].append(weightstats.DescrStatsW(norm_baselines_roi[ecc_quantile],
                                                                                              weights=rsq_model_roi[ecc_quantile]))
                
                                        ecc_stats[roi][param].append(weightstats.DescrStatsW(ecc_model_roi[ecc_quantile],
                                                                                              weights=rsq_model_roi[ecc_quantile]))
                
                           
                                    WLS = LinearRegression()
                                    WLS.fit(ecc_model_roi.reshape(-1, 1), norm_baselines_roi, sample_weight=rsq_model_roi)
                                    p=pl.plot([ss.mean for ss in ecc_stats[roi][param]],
                                              WLS.predict(np.array([ss.mean for ss in ecc_stats[roi][param]]).reshape(-1, 1)),
                                              color=f"C{i+4}")#,
                                              #color=roi_colors[roi])
                                                
                                    #print(roi+" "+model+" "+str(WLS.score(ecc_model_roi.reshape(-1, 1), norm_baselines_roi, sample_weight=rsq_model_roi)))
                
                                    pl.errorbar([ss.mean for ss in ecc_stats[roi][param]],
                                       [ss.mean for ss in norm_baselines_stats[roi][param]],
                                       yerr=np.array([np.abs(ss.zconfint_mean(alpha=0.05)-ss.mean) for ss in norm_baselines_stats[roi][param]]).T,
                                       xerr=np.array([np.abs(ss.zconfint_mean(alpha=0.05)-ss.mean) for ss in ecc_stats[roi][param]]).T,
                                       fmt=symbol[roi],  mec='black', label=roi.replace('custom.',''), color=p[-1].get_color())#, mfc=roi_colors[roi],ecolor=roi_colors[roi])
                                    
                                    pl.figure(analysis+param+subj+model+'mean', figsize=(8, 8), frameon=False)
                                    
                                    if param == 'Norm Param. B':
                                        pl.ylim(0,31)
                                    elif param == 'Norm Param. D':
                                        pl.ylim(0,31)
                                    elif param == 'Ratio (B/D)':
                                        pl.ylim(0,3.1)
                                        
                                    full_roi_stats = weightstats.DescrStatsW(norm_baselines_roi, weights=rsq_model_roi)
                                    
                                    bar_height = full_roi_stats.mean
                                    bar_err = np.array(np.abs(full_roi_stats.zconfint_mean(alpha=0.05)-bar_height)).reshape((-1,1))

                                    pl.bar(bar_position, bar_height, width=0.1, yerr=bar_err,edgecolor='black',color=p[-1].get_color())#
                                    x_ticks.append(bar_position)
                                    x_labels.append(roi.replace('custom.',''))
                                    bar_position+=0.15
                                    pl.ylabel(f"{subj} {param}")
                                    
                                    pl.xticks(x_ticks,x_labels)
                                    
                                if save_figures:
                                    pl.savefig(f"/Users/marcoaqil/PRFMapping/Figures/{subj}_mean-{param.replace('/','').replace('.','').replace(' ','_')}.pdf", dpi=300, bbox_inches='tight')
                                                                        
                                pl.figure(analysis+param+subj+model, figsize=(8, 8), frameon=False)    
                                pl.xlabel('Eccentricity (degrees)')
                                pl.ylabel(f"{subj} {param}")
                                pl.legend(loc=0)
                                if save_figures:
                                    pl.savefig(f"/Users/marcoaqil/PRFMapping/Figures/{subj}_ecc-{param.replace('/','').replace('.','').replace(' ','_')}.pdf", dpi=300, bbox_inches='tight')
                                    
        return     

                                   
    def rsq_roi_plots(self, rois, rsq_thresh, save_figures, analysis_names='all', plot_hist=False, print_stats=False):

        pl.rcParams.update({'font.size': 16})
        pl.rcParams.update({'pdf.fonttype':42})
        pl.rc('figure', facecolor='w')
        for space, space_res in self.main_dict.items():
            if 'fs' in space:
                if analysis_names == 'all':
                    analyses = space_res.items()
                else:
                    analyses = [item for item in space_res.items() if item[0] in analysis_names] 
                for analysis, analysis_res in analyses:
                    model_rsq=dd(lambda:dd(dict))
                    
                    for i, (subj, subj_res) in enumerate(analysis_res.items()):
                        print(space+" "+analysis+" "+subj)
                        x_ticks=[]
                        x_labels=[]    
                        bar_position = 0
                        last_bar_position = 0#dd(lambda:0)
            
                        # binned eccentricity vs other parameters relationships       
            
                        #model_colors = {'Gauss':'blue','CSS':'orange','DoG':'green','Norm':'red'}

                        cmap = cm.get_cmap('tab10')

                        for roi in rois:
                            bar_position=last_bar_position+0.1
                            pl.figure(analysis+subj+' RSq', figsize=(8, 8), frameon=False)
                            pl.ylabel(subj+' R-squared')  
                            pl.ylim((0.0,0.7))
                            
                            alpha_roi = roi_mask(self.idx_rois[subj][roi], subj_res['Processed Results']['Alpha']['all']>rsq_thresh) #* (subj_res['Processed Results']['RSq']['CSS']>subj_res['Processed Results']['RSq']['Gauss']) * (subj_res['Processed Results']['RSq']['DoG']>subj_res['Processed Results']['RSq']['Gauss'])
                            print(self.idx_rois[subj][roi].shape)
                            print(alpha_roi.sum())
                            
                            
                            model_list = [k for k in subj_res['Processed Results']['RSq'].keys() if k in self.only_models]
                            model_list.sort()
                            model_list[0], model_list[2] = model_list[2], model_list[0]
                            model_colors = {model:cmap(val) for model,val in zip(model_list,[0,2,1,3])}
                            
                            #for model in model_list:
                            #   alpha_roi *= (subj_res['Processed Results']['CCrsq_task-1R'][model]>0.0)
                            #alpha_roi *= np.nanmax(np.array([subj_res['Processed Results']['CCrsq_task-1S'][model] for model in model_list]), axis=0)>0.0
                            
                            #print(alpha_roi.sum())     
                            x_ticks.append(bar_position+0.15)
                            x_labels.append(roi.replace('custom.','')+'\n')
                            for model in model_list:
                                                            
                                model_rsq[roi][model][subj] = (subj_res['Processed Results']['RSq'][model][alpha_roi])#-subj_res['Processed Results']['RSq']['Gauss'][alpha_roi])#*100 / subj_res['Processed Results']['RSq']['Gauss'][alpha_roi]
                                
                                bar_height = np.mean(model_rsq[roi][model][subj])

                                bar_err = sem(model_rsq[roi][model][subj])

                                pl.bar(bar_position, bar_height, width=0.1, yerr=bar_err,edgecolor='black', label=model, color=model_colors[model])
                                
                                if i+1 == len(analysis_res.keys()):
                                    group_rsq = np.concatenate(tuple([model_rsq[roi][model][k] for k in model_rsq[roi][model]]))#.flatten()   
                                    group = ''.join([k for k in model_rsq[roi][model]])

                                    pl.figure(analysis+'group RSq', figsize=(8, 8), frameon=False)
                                    pl.ylabel(group+' R-squared')  
                                    pl.ylim((0,0.7))
                                    bar_height = np.mean(group_rsq)
                                    bar_err = sem(group_rsq)
                                    pl.bar(bar_position, bar_height, width=0.1, yerr=bar_err,edgecolor='black', label=model, color=model_colors[model])
                                
                                    pl.xticks(x_ticks,x_labels)
                                    handles, labels = pl.gca().get_legend_handles_labels()
                                    by_label = dict(zip(labels, handles))
                                    pl.legend(by_label.values(), by_label.keys())
                                    pl.figure(analysis+subj+' RSq', figsize=(8, 8), frameon=False)
                                
                                bar_position += 0.1
                              

                            last_bar_position = bar_position
                            pl.xticks(x_ticks,x_labels)
                            handles, labels = pl.gca().get_legend_handles_labels()
                            by_label = dict(zip(labels, handles))
                            pl.legend(by_label.values(), by_label.keys())
                            
                            if print_stats:
                             
                                if 'CSS' in model_list and 'DoG' in model_list:
                                    for model in [model for model in model_list if 'Norm' in model]:
                                        surround_voxels = subj_res['Processed Results']['RSq']['DoG'][alpha_roi]>subj_res['Processed Results']['RSq']['Gauss'][alpha_roi]
                                        nonlinear_voxels = subj_res['Processed Results']['RSq']['CSS'][alpha_roi]>subj_res['Processed Results']['RSq']['Gauss'][alpha_roi]
                                        
                                        print(analysis+' '+roi)
                                        print(f"{roi} voxels above {rsq_thresh} threshold within stimulus eccentricity: {np.sum(alpha_roi)} out of {len(self.idx_rois[subj][roi])}")
                                        
                                        print(f"{model}-CSS in {roi} surround voxels: {ks_2samp(subj_res['Processed Results']['RSq'][model][alpha_roi][surround_voxels],subj_res['Processed Results']['RSq']['CSS'][alpha_roi][surround_voxels])}")
                                        print(f"{model}-DoG in {roi} nonlinear voxels: {ks_2samp(subj_res['Processed Results']['RSq'][model][alpha_roi][nonlinear_voxels],subj_res['Processed Results']['RSq']['DoG'][alpha_roi][nonlinear_voxels])}")
                                        
                                        norm_css_surrvox = subj_res['Processed Results']['RSq'][model][alpha_roi][surround_voxels]-subj_res['Processed Results']['RSq']['CSS'][alpha_roi][surround_voxels]
                                        norm_dog_nonlvox = subj_res['Processed Results']['RSq'][model][alpha_roi][nonlinear_voxels]-subj_res['Processed Results']['RSq']['DoG'][alpha_roi][nonlinear_voxels]
                                        norm_css_nonlvox = subj_res['Processed Results']['RSq'][model][alpha_roi][nonlinear_voxels]-subj_res['Processed Results']['RSq']['CSS'][alpha_roi][nonlinear_voxels]
                                        norm_dog_surrvox = subj_res['Processed Results']['RSq'][model][alpha_roi][surround_voxels]-subj_res['Processed Results']['RSq']['DoG'][alpha_roi][surround_voxels]
                                        
                                        # if noise_ceiling is not None:
                                        #     norm_css_surrvox /= noise_ceiling[alpha_roi][surround_voxels]
                                        #     norm_dog_nonlvox /= noise_ceiling[alpha_roi][nonlinear_voxels]
                                        #     norm_css_nonlvox /= noise_ceiling[alpha_roi][nonlinear_voxels]
                                        #     norm_dog_surrvox /= noise_ceiling[alpha_roi][surround_voxels]
                                            
                                        print(f"{model}-CSS in {roi} surround voxels: {ttest_1samp(norm_css_surrvox,0)}")
                                        print(f"{model}-DoG in {roi} nonlinear voxels: {ttest_1samp(norm_dog_nonlvox,0)}")
                                        
                                        print(f"{model}-CSS in {roi} surround voxels: {wilcoxon(norm_css_surrvox)}")
                                        print(f"{model}-DoG in {roi} nonlinear voxels: {wilcoxon(norm_dog_nonlvox)}")
                                                                                      
                                        print(f"{model}-CSS in {roi} surround voxels: {np.mean(norm_css_surrvox)}")
                                        print(f"{model}-DoG in {roi} nonlinear voxels: {np.mean(norm_dog_nonlvox)}")
                                    
                                if plot_hist:
                                    fig, axs = pl.subplots(2, 2, sharey=True, sharex=True)
                                    fig.suptitle(roi.replace('custom.',''))
                                    axs[0,0].set_xlabel('Norm-CSS')
                                    axs[0,0].set_ylabel('Number of vertices')
                                    axs[0,1].set_xlabel('Norm-CSS') 
                                    axs[0,0].set_title('Surround vertices')
                                    axs[0,1].set_title('Nonlinear vertices')
                                    
                                    h1 = axs[0,0].hist(norm_css_surrvox,bins=100)
                                    h2 = axs[0,1].hist(norm_css_nonlvox,bins=100)
                                    

                                    axs[1,0].set_xlabel('Norm-DoG')
                                    axs[1,0].set_ylabel('Number of vertices')
                                    axs[1,1].set_xlabel('Norm-DoG') 
                                    
                                    h3 = axs[1,0].hist(norm_dog_surrvox,bins=100)
                                    h4 = axs[1,1].hist(norm_dog_nonlvox,bins=100)
                                    
                                    height = 1+int(np.max([h1[0].max(),h2[0].max(), h3[0].max(), h4[0].max()]))
                                    
                                    axs[0,0].plot(np.zeros(height), np.arange(height), c='black', linestyle='--')   
                                    axs[0,1].plot(np.zeros(height), np.arange(height), c='black', linestyle='--')                               
                                    axs[1,0].plot(np.zeros(height), np.arange(height), c='black', linestyle='--')   
                                    axs[1,1].plot(np.zeros(height), np.arange(height), c='black', linestyle='--')                                

                            print('---------------')
                        if save_figures:
                            pl.savefig(f"/Users/marcoaqil/PRFMapping/Figures/{subj}_rsq.pdf", dpi=300, bbox_inches='tight')
                        print('\n')
                            
        return     

                                
