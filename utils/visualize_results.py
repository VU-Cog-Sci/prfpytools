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
from collections import defaultdict as dd
from copy import deepcopy

import time
from scipy.stats import sem, ks_2samp, ttest_1samp, wilcoxon

opj = os.path.join

from statsmodels.stats import weightstats
from sklearn.linear_model import LinearRegression
from nibabel.freesurfer.io import read_morph_data, write_morph_data
from utils.preproc_utils import roi_mask

class visualize_results(object):
    def __init__(self, results):
        self.main_dict = deepcopy(results.main_dict) 
        self.get_spaces()
        self.get_subjects(self.main_dict)
        
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
                                                          src_subject=subj,
                                                          verbose=True)
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

    def set_alpha(self, only_models=None):
        self.tc_min = dict()
        for space, space_res in self.main_dict.items():
            if 'fs' in space:
                for analysis, analysis_res in space_res.items():       
                    for subj, subj_res in analysis_res.items():
                        print(space+" "+analysis+" "+subj)
                            
                        p_r = subj_res['Processed Results']
                        models = p_r['RSq'].keys()
                                                

                        tc_stats = subj_res['Timecourse Stats']
                       
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
                        self.ecc_min=0.125
                        self.ecc_max=5
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
                        p_r['Alpha']['all'] = rsq.max(-1) * (tc_stats['Mean']>self.tc_min[subj]) * (ecc.min(-1)<self.ecc_max) * (ecc.max(-1)>self.ecc_min) * (rsq.min(-1)>0)
                        
                        for model in models:
                            p_r['Alpha'][model] = p_r['RSq'][model] * (p_r['Eccentricity'][model]>self.ecc_min) * (p_r['Eccentricity'][model]<self.ecc_max)\
                                * (tc_stats['Mean']>self.tc_min[subj]) #*(p_r['Size (fwhmax)'][model]<w_max)
                       
        return

    def pycortex_plots(self, rois, rsq_thresh, analysis_names = 'all'):        
          
        for space, space_res in self.main_dict.items():
            if 'fs' in space:
                plotted_rois = dd(lambda:False)
                plotted_stats = dd(lambda:False)
                if analysis_names == 'all':
                    analyses = space_res.items()
                else:
                    analyses = [item for item in space_res.items() if item[0] in analysis_names] 
                for analysis, analysis_res in analyses:    
                    for subj, subj_res in analysis_res.items():
                        
                        if subj not in cortex.db.subjects:
                            cortex.freesurfer.import_subj(subj, freesurfer_subject_dir=self.fs_dir, 
                                  whitematter_surf='smoothwm')
                        
                        p_r = subj_res['Processed Results']
                        models = p_r['RSq'].keys()
                        
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
                                      
                            alpha_freeview = p_r['RSq']['Norm']* (tc_stats['Mean']>self.tc_min[subj])# rsq.max(-1) * (tc_stats['Mean']>self.tc_min[subj]) * (rsq.min(-1)>0)
            
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
            
                            self.js_handle_stats = cortex.webgl.show(data_stats, with_curvature=False, with_labels=True, with_rois=True, with_borders=True, with_colorbar=True)
            
                            plotted_stats[subj] = True
                        
                        if self.plot_rois_cortex and not plotted_rois[subj]:
                            
                            ds_rois = {}
                            data = np.zeros_like(mask).astype('int')
            
                            for i, roi in enumerate(self.idx_rois[subj]):
            
                                roi_data = np.zeros_like(mask)
                                roi_data[self.idx_rois[subj][roi]] = 1
                                if 'custom' not in roi and 'visual' not in roi:
                                    data[self.idx_rois[subj][roi]] = i+1

                                ds_rois[roi] = cortex.Vertex2D(roi_data, roi_data.astype('bool'), subj, cmap='RdBu_r_alpha').raw
            

            
                            ds_rois['Wang2015Atlas'] = cortex.Vertex2D(data, data.astype('bool'), subj, cmap='Retinotopy_HSV_2x_alpha').raw
                            self.js_handle_rois = cortex.webgl.show(ds_rois, with_curvature=False, with_labels=True, with_rois=True, with_borders=True, with_colorbar=True)
            
                            plotted_rois[subj] = True
                                                    

                            
            
                        if self.plot_rsq_cortex:              
                            ds_rsq = {}
                            only_models = ['Gauss', 'Norm_abcd', 'CSS', 'DoG']
                            best_model = np.argmax([p_r['RSq'][model] for model in only_models],axis=0)
                            ds_rsq['Best model'] = cortex.Vertex2D(best_model, p_r['Alpha']['all'], subject=subj,
                                                                          vmin2=rsq_thresh, vmax2=0.6, cmap='BROYG_2D').raw 
                            
                            if 'CSS' in models and 'Gauss' in models:
                                ds_rsq['CSS - Gauss'] = cortex.Vertex2D(p_r['RSq']['CSS']-p_r['RSq']['Gauss'], p_r['Alpha']['all'], subject=subj,
                                                                          vmin=-0.05, vmax=0.05, vmin2=rsq_thresh, vmax2=0.6, cmap='Jet_2D_alpha').raw   
                                
                            if 'DoG' in models and 'Gauss' in models:
                                ds_rsq['DoG - Gauss'] = cortex.Vertex2D(p_r['RSq']['DoG']-p_r['RSq']['Gauss'], p_r['Alpha']['all'], subject=subj,
                                                                      vmin=-0.05, vmax=0.05, vmin2=rsq_thresh, vmax2=0.6, cmap='Jet_2D_alpha').raw
                            
                            for model in [model for model in models if 'Norm' in model]:

                                ds_rsq[f'{model} - Gauss'] = cortex.Vertex2D(p_r['RSq'][model]-p_r['RSq']['Gauss'], p_r['Alpha']['all'], subject=subj,
                                                                          vmin=-0.05, vmax=0.05, vmin2=rsq_thresh, vmax2=0.6, cmap='Jet_2D_alpha').raw

                                ds_rsq[f'{model} - DoG'] = cortex.Vertex2D(p_r['RSq'][model]-p_r['RSq']['DoG'], p_r['Alpha']['all'], subject=subj,
                                                                          vmin=-0.05, vmax=0.05, vmin2=rsq_thresh, vmax2=0.6, cmap='Jet_2D_alpha').raw

                                ds_rsq[f'{model} - CSS'] = cortex.Vertex2D(p_r['RSq'][model]-p_r['RSq']['CSS'], p_r['Alpha']['all'], subject=subj, 
                                                                          vmin=-0.05, vmax=0.05, vmin2=rsq_thresh, vmax2=0.6, cmap='Jet_2D_alpha').raw
                                
                            if 'Processed Results' in self.main_dict['T1w'][analysis][subj] and self.compare_volume_surface:
                                ds_rsq_comp={}
                                volume_rsq = self.main_dict['T1w'][analysis][subj]['Processed Results']['RSq']['Norm']
                                ref_img = nb.load(self.main_dict['T1w'][analysis][subj]['Results']['ref_img_path'])
                                
                                #rsq_img = nb.Nifti1Image(volume_rsq, ref_img.affine, ref_img.header)

                                xfm_trans = cortex.xfm.Transform(np.identity(4), ref_img)
                                xfm_trans.save(subj, 'func_space_transform')
                                
                                ds_rsq_comp['Norm_abcd CV rsq (volume fit)'] = cortex.Volume2D(volume_rsq.T, volume_rsq.T, subj, 'func_space_transform',
                                                                          vmin=rsq_thresh, vmax=0.6, vmin2=0.05, vmax2=rsq_thresh, cmap='Jet_2D_alpha')
                                ds_rsq_comp['Norm_abcd CV rsq (surface fit)'] = cortex.Vertex2D(p_r['RSq']['Norm_abcd'], p_r['RSq']['Norm_abcd'], subject=subj,
                                                                          vmin=rsq_thresh, vmax=0.6, vmin2=0.05, vmax2=rsq_thresh, cmap='Jet_2D_alpha').raw
                                self.js_handle_rsq_comp = cortex.webgl.show(ds_rsq_comp, with_curvature=False, with_labels=True, with_rois=True, with_borders=True, with_colorbar=True)

                            self.js_handle_rsq = cortex.webgl.show(ds_rsq, with_curvature=False, with_labels=True, with_rois=True, with_borders=True, with_colorbar=True) 
                            
                        if self.plot_ecc_cortex:
                            ds_ecc = {}
                            for model in models:
                                ds_ecc[model] = cortex.Vertex2D(p_r['Eccentricity'][model], p_r['Alpha'][model], subject=subj, 
                                                                vmin=self.ecc_min, vmax=self.ecc_max, vmin2=rsq_thresh, vmax2=0.6, cmap='Jet_r_2D_alpha').raw
            
                            self.js_handle_ecc = cortex.webgl.show(ds_ecc, with_curvature=False, with_labels=True, with_rois=True, with_borders=True, with_colorbar=True)
            
                        if self.plot_polar_cortex:
                            ds_polar = {}
                            for model in models:
                                ds_polar[model] = cortex.Vertex2D(p_r['Polar Angle'][model], p_r['Alpha'][model], subject=subj, 
                                                                  vmin2=rsq_thresh, vmax2=0.6, cmap='Retinotopy_HSV_2x_alpha').raw
                            
                            if 'Processed Results' in self.main_dict['T1w'][analysis][subj] and self.compare_volume_surface:
                                ds_polar_comp={}
                                volume_rsq = self.main_dict['T1w'][analysis][subj]['Processed Results']['RSq']['Norm_abcd']
                                volume_polar = self.main_dict['T1w'][analysis][subj]['Processed Results']['Polar Angle']['Norm_abcd']
                                ref_img = nb.load(self.main_dict['T1w'][analysis][subj]['Results']['ref_img_path'])                                

                                xfm_trans = cortex.xfm.Transform(np.identity(4), ref_img)
                                xfm_trans.save(subj, 'func_space_transform')
                                
                                ds_polar_comp['Norm_abcd CV polar (volume fit)'] = cortex.Volume2D(volume_polar.T, volume_rsq.T, subj, 'func_space_transform',
                                                                          vmin2=0.05, vmax2=rsq_thresh, cmap='Retinotopy_HSV_2x_alpha')
                                ds_polar_comp['Norm_abcd CV polar (surface fit)'] = cortex.Vertex2D(p_r['Polar Angle']['Norm_abcd'], p_r['RSq']['Norm_abcd'], subject=subj,
                                                                          vmin2=0.05, vmax2=rsq_thresh, cmap='Retinotopy_HSV_2x_alpha').raw
                                self.js_handle_polar_comp = cortex.webgl.show(ds_polar_comp, with_curvature=False, with_labels=True, with_rois=True, with_borders=True, with_colorbar=True)

                            
                            self.js_handle_polar = cortex.webgl.show(ds_polar, with_curvature=False, with_labels=True, with_rois=True, with_borders=True, with_colorbar=True)
            
                        if self.plot_size_cortex:
                            ds_size = {}
                            for model in models:
                                ds_size[model] = cortex.Vertex2D(p_r['Size (fwhmax)'][model], p_r['Alpha'][model], subject=subj, 
                                                                 vmin=0, vmax=6, vmin2=rsq_thresh, vmax2=0.6, cmap='Jet_2D_alpha').raw
                  
                            self.js_handle_size = cortex.webgl.show(ds_size, with_curvature=False, with_labels=True, with_rois=True, with_borders=True, with_colorbar=True)
            
            
                        if self.plot_amp_cortex:
                            ds_amp = {}
                            for model in models:
                                ds_amp[model] = cortex.Vertex2D(p_r['Amplitude'][model], p_r['Alpha'][model], subject=subj, 
                                                                vmin=-1, vmax=1, vmin2=rsq_thresh, vmax2=0.6, cmap='Jet_2D_alpha').raw
            
                            self.js_handle_amp = cortex.webgl.show(ds_amp, with_curvature=False, with_labels=True, with_rois=True, with_borders=True, with_colorbar=True)
                            
                        if self.plot_css_exp_cortex and 'CSS' in models:
                            ds_css_exp = {}
                            ds_css_exp['CSS Exponent'] = cortex.Vertex2D(p_r['CSS Exponent']['CSS'], p_r['Alpha']['CSS'], subject=subj, 
                                                                         vmin=0, vmax=1, vmin2=rsq_thresh, vmax2=0.6, cmap='Jet_2D_alpha').raw
            
                            self.js_handle_css_exp = cortex.webgl.show(ds_css_exp, with_curvature=False, with_labels=True, with_rois=True, with_borders=True, with_colorbar=True)
                            
                        if self.plot_surround_size_cortex:
                            ds_surround_size = {}
                            for model in models:
                                if model == 'DoG':
                                    ds_surround_size['DoG'] = cortex.Vertex2D(p_r['Surround Size (fwatmin)']['DoG'], p_r['Alpha']['DoG'], subject=subj, 
                                                                         vmin=0, vmax=50, vmin2=rsq_thresh, vmax2=0.6, cmap='Jet_2D_alpha').raw
                                elif 'Norm' in model:
                                    ds_surround_size[model] = cortex.Vertex2D(p_r['Surround Size (fwatmin)'][model], p_r['Alpha'][model], subject=subj, 
                                                                         vmin=0, vmax=50, vmin2=rsq_thresh, vmax2=0.6, cmap='Jet_2D_alpha').raw                    
            
                            self.js_handle_surround_size = cortex.webgl.show(ds_surround_size, with_curvature=False, with_labels=True, with_rois=True, with_borders=True, with_colorbar=True)    
                            
                        if self.plot_norm_baselines_cortex:
                            ds_norm_baselines = {}
                            for model in [model for model in models if 'Norm_abcd' in model]:

                                ds_norm_baselines[f'{model} Param. B'] = cortex.Vertex2D(p_r['Norm Param. B'][model], p_r['Alpha'][model], subject=subj, 
                                                                             vmin=0, vmax=1, vmin2=rsq_thresh, vmax2=0.6, cmap='Jet_2D_alpha').raw                    
                                ds_norm_baselines[f'{model} Param. D'] = cortex.Vertex2D(p_r['Norm Param. D'][model], p_r['Alpha'][model], subject=subj, 
                                                                             vmin=0, vmax=1, vmin2=rsq_thresh, vmax2=0.6, cmap='Jet_2D_alpha').raw
                                ds_norm_baselines[f'{model} Ratio (B/D)'] = cortex.Vertex2D(p_r['Ratio (B/D)'][model], p_r['Alpha'][model], subject=subj, 
                                                                             vmin=0, vmax=0.75, vmin2=rsq_thresh, vmax2=0.6, cmap='Jet_2D_alpha').raw
                            
                            self.js_handle_norm_baselines = cortex.webgl.show(ds_norm_baselines, with_curvature=False, with_labels=True, with_rois=True, with_borders=True, with_colorbar=True)    
        print('-----')                              
        return    
        
    def save_pycortex_views(self, js_handle, base_str):
        views = dict(dorsal=dict(radius=191, altitude=73, azimuth=178, pivot=0),
                     medial=dict(radius=10, altitude=101, azimuth=359, pivot=167),
                     lateral=dict(radius=277, altitude=90, azimuth=177, pivot=123),
                     ventral=dict(radius=221, altitude=131, azimuth=175, pivot=0)
                    )
        
        surfaces = dict(inflated=dict(unfold=1))#,
                       # fiducial=dict(unfold=0.0))
        
        # select path for the generated images on disk
        image_path = '/Users/marcoaqil/PRFMapping/Figures/'
        
        # pattern of the saved images names
        file_pattern = "{base}_{view}_{surface}.pdf"
        
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
                js_handle.getImage(output_path, size =(3000, 2000))
        
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
        
    def ecc_size_roi_plots(self, rois, rsq_thresh, save_figures, analysis_names = 'all'):
        
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
                        # roi_colors['custom.hV4']='blue'
                        # roi_colors['custom.V3AB']='orange'
                        # #roi_colors['custom.']
                        
                        w_max=90
            
                        fw_hmax_stats = dd(lambda:dd(list))
                        ecc_stats = dd(lambda:dd(list))
            
                        for roi in rois:
            
                            pl.figure(f'{subj} {roi} fw_hmax', figsize=(8, 6), frameon=False)
           
                            for model in [k for k in subj_res['Processed Results']['Size (fwhmax)'].keys() if 'Norm_abcd' in k or 'Norm' not in k]:                                
            
                                #model-specific alpha? or all models same alpha?
                                alpha_roi = (roi_mask(self.idx_rois[subj][roi], subj_res['Processed Results']['Alpha'][model]>rsq_thresh)) * (subj_res['Processed Results']['Size (fwhmax)'][model]<w_max)
                                
                                ecc_model_roi = subj_res['Processed Results']['Eccentricity'][model][alpha_roi]
                                fwhmax_model_roi = subj_res['Processed Results']['Size (fwhmax)'][model][alpha_roi]
                                rsq_model_roi = subj_res['Processed Results']['RSq'][model][alpha_roi]
                                
                                ecc_sorted = np.argsort(ecc_model_roi)
                                split_ecc_bins = np.array_split(ecc_sorted, 10)
                               
                                for ecc_quantile in split_ecc_bins:
                                    fw_hmax_stats[roi][model].append(weightstats.DescrStatsW(fwhmax_model_roi[ecc_quantile],
                                                                                          weights=rsq_model_roi[ecc_quantile]))
            
                                    ecc_stats[roi][model].append(weightstats.DescrStatsW(ecc_model_roi[ecc_quantile],
                                                                                          weights=rsq_model_roi[ecc_quantile]))
            
                       
                                WLS = LinearRegression()
                                WLS.fit(ecc_model_roi.reshape(-1, 1), fwhmax_model_roi, sample_weight=rsq_model_roi)
                                p=pl.plot([ss.mean for ss in ecc_stats[roi][model]],
                                        WLS.predict(np.array([ss.mean for ss in ecc_stats[roi][model]]).reshape(-1, 1)))
                                        #color=model_colors[model])
                                            
                                print(roi+" "+model+" "+str(WLS.score(ecc_model_roi.reshape(-1, 1), fwhmax_model_roi, sample_weight=rsq_model_roi)))
            
                                pl.errorbar([ss.mean for ss in ecc_stats[roi][model]],
                                   [ss.mean for ss in fw_hmax_stats[roi][model]],
                                   yerr=np.array([np.abs(ss.zconfint_mean(alpha=0.05)-ss.mean) for ss in fw_hmax_stats[roi][model]]).T,
                                   xerr=np.array([np.abs(ss.zconfint_mean(alpha=0.05)-ss.mean) for ss in ecc_stats[roi][model]]).T,
                                   fmt='s',  mec='black', label=model, color=p[-1].get_color())#, mfc=model_colors[model], ecolor=model_colors[model])
            
                            pl.xlabel('Eccentricity (degrees)')
                            pl.ylabel(roi.replace('custom.','')+' pRF size (degrees)')
                            pl.legend(loc=0)
                            if save_figures:
                                pl.savefig('/Users/marcoaqil/PRFMapping/Figures/'+subj+'_'+
                                           roi.replace('custom.','')+'_fw-hmax.pdf', dpi=200, bbox_inches='tight')
                                
                        for model in [k for k in subj_res['Processed Results']['Size (fwhmax)'].keys() if 'Norm_abcd' in k or 'Norm' not in k]:
                            pl.figure(f'{subj} {model} fw_hmax', figsize=(8, 6), frameon=False)
                            for roi in rois:
                                #model-specific alpha? or all models same alpha?
                                alpha_roi = roi_mask(self.idx_rois[subj][roi], subj_res['Processed Results']['Alpha'][model]>rsq_thresh)
                                
                                ecc_model_roi = subj_res['Processed Results']['Eccentricity'][model][alpha_roi]
                                fwhmax_model_roi = subj_res['Processed Results']['Size (fwhmax)'][model][alpha_roi]
                                rsq_model_roi = subj_res['Processed Results']['RSq'][model][alpha_roi]
                                
                                ecc_sorted = np.argsort(ecc_model_roi)
                                split_ecc_bins = np.array_split(ecc_sorted, 10)
                               
                                for ecc_quantile in split_ecc_bins:
                                    fw_hmax_stats[roi][model].append(weightstats.DescrStatsW(fwhmax_model_roi[ecc_quantile],
                                                                                          weights=rsq_model_roi[ecc_quantile]))
            
                                    ecc_stats[roi][model].append(weightstats.DescrStatsW(ecc_model_roi[ecc_quantile],
                                                                                          weights=rsq_model_roi[ecc_quantile]))
            
                       
                                WLS = LinearRegression()
                                WLS.fit(ecc_model_roi.reshape(-1, 1), fwhmax_model_roi, sample_weight=rsq_model_roi)
                                p=pl.plot([ss.mean for ss in ecc_stats[roi][model]],
                                        WLS.predict(np.array([ss.mean for ss in ecc_stats[roi][model]]).reshape(-1, 1)))
                                        #color=roi_colors[roi])
                                            
                                print(roi+" "+model+" "+str(WLS.score(ecc_model_roi.reshape(-1, 1), fwhmax_model_roi, sample_weight=rsq_model_roi)))
            
                                pl.errorbar([ss.mean for ss in ecc_stats[roi][model]],
                                   [ss.mean for ss in fw_hmax_stats[roi][model]],
                                   yerr=np.array([np.abs(ss.zconfint_mean(alpha=0.05)-ss.mean) for ss in fw_hmax_stats[roi][model]]).T,
                                   xerr=np.array([np.abs(ss.zconfint_mean(alpha=0.05)-ss.mean) for ss in ecc_stats[roi][model]]).T,
                                   fmt='s', mec='black', label=roi.replace('custom.',''), color=p[-1].get_color())#, mfc=roi_colors[roi], ecolor=roi_colors[roi])
            
                            pl.xlabel('Eccentricity (degrees)')
                            pl.ylabel(model+' pRF size (degrees)')
                            pl.legend(loc=0)
                            if save_figures:
                                pl.savefig('/Users/marcoaqil/PRFMapping/Figures/'+subj+'_'+
                                           model+'_fw-hmax.pdf', dpi=200, bbox_inches='tight')

        return
    
    
    def ecc_surround_roi_plots(self, rois, rsq_thresh, save_figures):
        
        pl.rcParams.update({'font.size': 16})
        pl.rcParams.update({'pdf.fonttype':42})
        for space, space_res in self.main_dict.items():
            if 'fs' in space:
                for analysis, analysis_res in space_res.items():       
                    for subj, subj_res in analysis_res.items():
                        print(space+" "+analysis+" "+subj)
            
                        # binned eccentricity vs other parameters relationships       
            
                        # model_colors = {'Gauss':'blue','CSS':'orange','DoG':'green','Norm':'red'}
                                                
                        # #model_symbols = {'Gauss':'^','CSS':'o','DoG':'v','Norm':'D'}
                        # roi_colors = dd(lambda:'blue')
                        # roi_colors['custom.V1']= 'black'
                        # roi_colors['custom.V2']= 'red'
                        # roi_colors['custom.V3']= 'pink'
            
                        fw_atmin_stats = dd(lambda:dd(list))
                        ecc_stats = dd(lambda:dd(list))
                        
                        #exclude surrounds sizes larger than this (no surround)
                        w_max=90
            
                        for roi in rois:
            
                            pl.figure(roi+' fw_atmin', figsize=(8, 6), frameon=False)
           
                            for model in subj_res['Processed Results']['Surround Size (fwatmin)'].keys():                                
            
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
                                pl.plot([ss.mean for ss in ecc_stats[roi][model]],
                                        WLS.predict(np.array([ss.mean for ss in ecc_stats[roi][model]]).reshape(-1, 1)))
                                        #,color=model_colors[model])
                                            
                                print(roi+" "+model+" "+str(WLS.score(ecc_model_roi.reshape(-1, 1), fwatmin_model_roi, sample_weight=rsq_model_roi)))
            
                                pl.errorbar([ss.mean for ss in ecc_stats[roi][model]],
                                   [ss.mean for ss in fw_atmin_stats[roi][model]],
                                   yerr=np.array([np.abs(ss.zconfint_mean(alpha=0.05)-ss.mean) for ss in fw_atmin_stats[roi][model]]).T,
                                   xerr=np.array([np.abs(ss.zconfint_mean(alpha=0.05)-ss.mean) for ss in ecc_stats[roi][model]]).T,
                                   fmt='s',  mec='black', label=model)#, mfc=model_colors[model],ecolor=model_colors[model])
            
                            pl.xlabel('Eccentricity (degrees)')
                            pl.ylabel(roi.replace('custom.','')+' pRF Surround Size (degrees)')
                            pl.legend(loc=0)
                            if save_figures:
                                pl.savefig('/Users/marcoaqil/PRFMapping/Figures/'+subj+'_'+
                                           roi.replace('custom.','')+'_fw-atmin.pdf', dpi=200, bbox_inches='tight')
                                
                        for model in subj_res['Processed Results']['Surround Size (fwatmin)'].keys():
                            pl.figure(model+' fw_atmin', figsize=(8, 6), frameon=False)
                            for roi in rois:
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
                                pl.plot([ss.mean for ss in ecc_stats[roi][model]],
                                        WLS.predict(np.array([ss.mean for ss in ecc_stats[roi][model]]).reshape(-1, 1)))#,
                                        #color=roi_colors[roi])
                                            
                                print(roi+" "+model+" "+str(WLS.score(ecc_model_roi.reshape(-1, 1), fwatmin_model_roi, sample_weight=rsq_model_roi)))
            
                                pl.errorbar([ss.mean for ss in ecc_stats[roi][model]],
                                   [ss.mean for ss in fw_atmin_stats[roi][model]],
                                   yerr=np.array([np.abs(ss.zconfint_mean(alpha=0.05)-ss.mean) for ss in fw_atmin_stats[roi][model]]).T,
                                   xerr=np.array([np.abs(ss.zconfint_mean(alpha=0.05)-ss.mean) for ss in ecc_stats[roi][model]]).T,
                                   fmt='s', mec='black', label=roi.replace('custom.',''))#, mfc=roi_colors[roi], ecolor=roi_colors[roi])
            
                            pl.xlabel('Eccentricity (degrees)')
                            pl.ylabel(model+' pRF Surround Size (degrees)')
                            pl.legend(loc=0)
                            if save_figures:
                                pl.savefig('/Users/marcoaqil/PRFMapping/Figures/'+subj+'_'+
                                           model+'_fw-atmin.pdf', dpi=200, bbox_inches='tight')            
            
        return     

       
    def ecc_css_exp_roi_plots(self, rois, rsq_thresh, save_figures):
        
        pl.rcParams.update({'font.size': 16})
        pl.rcParams.update({'pdf.fonttype':42})
        for space, space_res in self.main_dict.items():
            if 'fs' in space:
                for analysis, analysis_res in space_res.items():       
                    for subj, subj_res in analysis_res.items():
                        print(space+" "+analysis+" "+subj)
            
                        # binned eccentricity vs other parameters relationships       
            
                        # roi_colors = dd(lambda:'blue')
                        # roi_colors['custom.V1']= 'black'
                        # roi_colors['custom.V2']= 'red'
                        # roi_colors['custom.V3']= 'pink'
            
                        css_exp_stats = dd(lambda:dd(list))
                        ecc_stats = dd(lambda:dd(list))
                        
                        pl.figure('css_exp', figsize=(8, 6), frameon=False)
                        for roi in rois:

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
                                   fmt='s', mec='black', label=roi.replace('custom.',''))#), mfc=roi_colors[roi], ecolor=roi_colors[roi])
            
                            pl.xlabel('Eccentricity (degrees)')
                            pl.ylabel('CSS Exponent')
                            pl.legend(loc=0)
                            if save_figures:
                                pl.savefig('/Users/marcoaqil/PRFMapping/Figures/'+subj+'_'+
                                           roi.replace('custom.','')+'_css-exp.pdf', dpi=200, bbox_inches='tight')            
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
                        params['AD-BC (nonlinearity?)'] = 'o'
                        
                        
                        # symbol={}
                        # symbol['ABCD_100'] = 'o'
                        # symbol['ACD_100'] = 's'
                        # symbol['ABC_100'] = 'D'
            

                        for param in params:
                            for model in [model for model in subj_res['Processed Results']['RSq'].keys() if 'Norm_abcd' == model]:
                                subj_res['Processed Results']['AD-BC (nonlinearity?)'][model]=np.zeros(subj_res['mask'].shape)
                                subj_res['Processed Results']['AD-BC (nonlinearity?)'][model][subj_res['mask']] = subj_res['Results'][model][:,3]*subj_res['Results'][model][:,8]- subj_res['Results'][model][:,5]*subj_res['Results'][model][:,7]
                                
                                
                                ecc_stats = dd(lambda:dd(list)) 
                                norm_baselines_stats = dd(lambda:dd(list))
                                
                                bar_position = 0
                                x_ticks=[]
                                x_labels=[] 
                                
                                for roi in rois:
                                    pl.figure(analysis+param+subj+model, figsize=(8, 6), frameon=False)
                                    #model-specific alpha? or all models same alpha?
                                    alpha_roi = roi_mask(self.idx_rois[subj][roi], subj_res['Processed Results']['Alpha'][model]>rsq_thresh)#* (subj_res['Processed Results']['RSq']['CSS']>subj_res['Processed Results']['RSq']['Gauss'])
                                    
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
                                              WLS.predict(np.array([ss.mean for ss in ecc_stats[roi][param]]).reshape(-1, 1)))#,
                                              #color=roi_colors[roi])
                                                
                                    #print(roi+" "+model+" "+str(WLS.score(ecc_model_roi.reshape(-1, 1), norm_baselines_roi, sample_weight=rsq_model_roi)))
                
                                    pl.errorbar([ss.mean for ss in ecc_stats[roi][param]],
                                       [ss.mean for ss in norm_baselines_stats[roi][param]],
                                       yerr=np.array([np.abs(ss.zconfint_mean(alpha=0.05)-ss.mean) for ss in norm_baselines_stats[roi][param]]).T,
                                       xerr=np.array([np.abs(ss.zconfint_mean(alpha=0.05)-ss.mean) for ss in ecc_stats[roi][param]]).T,
                                       fmt='s',  mec='black', label=roi.replace('custom.',''), color=p[-1].get_color())#, mfc=roi_colors[roi],ecolor=roi_colors[roi])
                                    
                                    pl.figure(analysis+param+subj+model+'mean', figsize=(8, 6), frameon=False)
                                    full_roi_stats = weightstats.DescrStatsW(norm_baselines_roi, weights=rsq_model_roi)
                                    
                                    bar_height = full_roi_stats.mean
                                    bar_err = np.array(np.abs(full_roi_stats.zconfint_mean(alpha=0.05)-bar_height)).reshape((-1,1))

                                    pl.bar(bar_position, bar_height, width=0.1, yerr=bar_err,edgecolor='black',color=p[-1].get_color())#
                                    x_ticks.append(bar_position)
                                    x_labels.append(roi.replace('custom.',''))
                                    bar_position+=0.15
                                    
                                    pl.xticks(x_ticks,x_labels)
                                    
                                    
                                pl.figure(analysis+param+subj+model, figsize=(8, 6), frameon=False)    
                                pl.xlabel('Eccentricity (degrees)')
                                pl.ylabel(param)
                                pl.legend(loc=0)
                                if save_figures:
                                    pl.savefig('/Users/marcoaqil/PRFMapping/Figures/'+subj+'_'+
                                               param.replace("/","").replace('.','').replace(' ','_')+'.pdf', dpi=200, bbox_inches='tight')
                                    
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
                        import matplotlib

                        cmap = matplotlib.cm.get_cmap('tab10')

                        for roi in rois:
                            bar_position=last_bar_position+0.1
                            pl.figure(analysis+subj+' RSq', figsize=(8, 6), frameon=False)
                            pl.ylabel(subj+' % of Gauss model R-squared')  
                            pl.ylim((0.24,0.71))
                            
                            alpha_roi = roi_mask(self.idx_rois[subj][roi], subj_res['Processed Results']['Alpha']['all']>rsq_thresh) #* (subj_res['Processed Results']['RSq']['CSS']>subj_res['Processed Results']['RSq']['Gauss']) * (subj_res['Processed Results']['RSq']['DoG']>subj_res['Processed Results']['RSq']['Gauss'])
                            print(self.idx_rois[subj][roi].shape)
                            print(alpha_roi.sum())
                            
                            
                            model_list = [k for k in subj_res['Processed Results']['RSq'].keys() if 'Norm_abcd' == k or 'Norm' not in k]
                            model_list.sort()
                            model_list[0], model_list[2] = model_list[2], model_list[0]
                            model_colors = {model:cmap(val) for model,val in zip(model_list,[0,2,1,3])}
                            
                            # for model in model_list:
                            #     alpha_roi *= (subj_res['Processed Results']['CCrsq_task-4R'][model]>0.0)
                                 
                            #print(alpha_roi.sum())     
                            x_ticks.append(bar_position+0.15)
                            x_labels.append(roi.replace('custom.','')+'\n')
                            for model in model_list:
                                                            
                                model_rsq[roi][model][subj] = (subj_res['Processed Results']['RSq'][model][alpha_roi])#-subj_res['Processed Results']['RSq']['Gauss'][alpha_roi])#*100 / subj_res['Processed Results']['RSq']['Gauss'][alpha_roi]
                                
                                group_rsq = np.concatenate(tuple([model_rsq[roi][model][k] for k in model_rsq[roi][model]]))#.flatten()   
                                group = ''.join([k for k in model_rsq[roi][model]])
                                #print(group_rsq)
                                bar_height = np.mean(model_rsq[roi][model][subj])
                                #print(bar_height)
                                bar_err = sem(model_rsq[roi][model][subj])
                                #print(bar_err)
                                pl.bar(bar_position, bar_height, width=0.1, yerr=bar_err,edgecolor='black', label=model, color=model_colors[model])
                                
                                if i+1 == len(analysis_res.keys()):
                                    pl.figure(analysis+'group RSq', figsize=(8, 6), frameon=False)
                                    pl.ylabel(group+' % of Gauss model R-squared')  
                                    pl.ylim((0.24,0.71))
                                    bar_height = np.mean(group_rsq)
                                    bar_err = sem(group_rsq)
                                    pl.bar(bar_position, bar_height, width=0.1, yerr=bar_err,edgecolor='black', label=model, color=model_colors[model])
                                
                                    pl.xticks(x_ticks,x_labels)
                                    handles, labels = pl.gca().get_legend_handles_labels()
                                    by_label = dict(zip(labels, handles))
                                    pl.legend(by_label.values(), by_label.keys())
                                    pl.figure(analysis+subj+' RSq', figsize=(8, 6), frameon=False)
                                
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
                            pl.savefig(f"/Users/marcoaqil/PRFMapping/Figures/{subj}_rsq.pdf", dpi=200, bbox_inches='tight')
                        print('\n')
                            
        return     

                                
