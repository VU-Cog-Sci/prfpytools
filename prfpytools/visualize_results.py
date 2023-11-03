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
from joblib import Parallel, delayed
from matplotlib import cm, colors
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.patheffects as peff
from collections import defaultdict as dd
from copy import deepcopy
import itertools
from pathlib import Path
from prfpytools.postproc_utils import model_wrapper, create_model_rf_wrapper, colorbar, norm_1d_sr_function, norm_2d_sr_function, Vertex2D_fix, simple_colorbar
from prfpytools.postproc_utils import reduced_graph_ft, graph_randomization
import cmasher as cmr
from prfpytools.preproc_utils import roi_mask, inverse_roi_mask, create_full_stim
#import seaborn as sns
from skimage import filters

import time
import scipy as sp
from scipy.stats import sem, ks_2samp, ttest_1samp, wilcoxon, pearsonr, ttest_ind, ttest_rel, zscore, spearmanr, gaussian_kde
from scipy.optimize import minimize

opj = os.path.join

from statsmodels.stats import weightstats
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
#from sklearn.decomposition import PCA, KernelPCA
from sklearn.cross_decomposition import CCA, PLSCanonical, PLSRegression
from wpca import PCA, WPCA
from sklearn.metrics import mean_squared_error
from nibabel.freesurfer.io import read_morph_data, write_morph_data


class visualize_results(object):
    def __init__(self, results):
        """
        Parameters
        ----------
        results : TYPE
            DESCRIPTION.

        Returns
        -------
        None.

        """
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
                if 'sub-' not in k and not k.isdecimal() and '999999' not in k and 'rsq' not in k:# and isinstance(v, (dict,dd)):
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
    
    def define_groups(self, groups_dict):
        self.groups_dict = groups_dict
        self.groups = groups_dict.keys()

        # if 'placebo' in self.groups:
        #     for subj_pla in self.groups_dict['placebo']:

        #         gen_subj = subj_pla.split('_')[0]

        #         all_ses_sj = [s for s in self.subjects if gen_subj in s]

        #         for space in self.spaces:
        #             if 'fs' in space or 'HCP' in space:
        #                 space_res = self.main_dict[space]
        #                 for analysis, analysis_res in space_res.items():
        #                     if subj_pla in analysis_res:
        #                         p_r_pla = analysis_res[subj_pla]['Processed Results']      
        #                         for subj, subj_res in analysis_res.items():
        #                             if subj in all_ses_sj:
        #                                 p_r = subj_res['Processed Results']
        #                                 p_r['Noise Ceiling']['Noise Ceiling (CC) (placebo)'] = np.copy(p_r_pla['Noise Ceiling']['Noise Ceiling (CC)'])

        return
                

    def find_group(self, sj):
        found=False
        
        for group in self.groups:
            if sj in self.groups_dict[group]:
                found = True
                this_sj_group = group
        if found:
            return this_sj_group
        else:
            return None
            
    
    def compute_diff(self, space_names = 'fsnative', analysis_names = 'all', subject_ids='all',parameter_names='all',base_group='placebo'):

        if space_names == 'all':
            spaces = [item for item in self.main_dict.items()]
        else:
            spaces = [item for item in self.main_dict.items() if item[0] in space_names] 

        for space, space_res in spaces:                
                                       
            if analysis_names == 'all':
                analyses = [item for item in space_res.items()]
            else:
                analyses = [item for item in space_res.items() if item[0] in analysis_names] 
                         
            for analysis, analysis_res in analyses:       
                if subject_ids == 'all':
                    subjects = [item for item in analysis_res.items()]
                else:
                    subjects = [item for item in analysis_res.items() if item[0] in subject_ids]

                subjects_pla = [s[0] for s in subjects if s[0] in self.groups_dict[base_group]]
                
                for subj, subj_res in subjects:

                    current_sj_group = self.find_group(subj)
                    #print(subj)
                    #print(current_sj_group)
                    if current_sj_group != None and current_sj_group != base_group:

                        gen_subj = subj.split('_')[0]
                        
                        subj_pla = [s for s in subjects_pla if gen_subj in s][0]
                        #print(subj_pla)

                        p_r = analysis_res[subj]['Processed Results']
                        p_r_pla = analysis_res[subj_pla]['Processed Results'] 

                        if parameter_names == 'all':
                            parameters = [item for item in p_r.items() if 'Alpha' not in item[0]]
                        else:
                            parameters = [item for item in p_r.items() if item[0] in parameter_names]

                        for param, param_res in parameters:
                            if f'{current_sj_group}' not in param and f'{base_group}' not in param and param in p_r_pla:
                                models = [item for item in param_res.items()]
                                for model, model_res in models:
                                    if model in p_r['Alpha']:
                                        this_alpha = (p_r['Alpha'][model] + p_r_pla['Alpha'][model])/2

                                        this_alpha[p_r['Alpha'][model]<=0] = 0
                                        this_alpha[p_r_pla['Alpha'][model]<=0] = 0

                                        p_r[f'Mean Masked RSq {current_sj_group}-{base_group}'][model] = this_alpha
                                    else:
                                        this_alpha = np.ones_like(p_r[param][model])

                                    if model in p_r_pla[param]:

                                        p_r[f'{param} {current_sj_group}-{base_group}'][model] = p_r[param][model] - p_r_pla[param][model]

                                        p_r[f'{param} {current_sj_group}-{base_group}'][model][this_alpha<=0] = 0


    
        return


    def compute_ratio(self, space_names = 'fsnative', analysis_names = 'all', subject_ids='all',parameter_names='all',base_group='placebo'):

        if space_names == 'all':
            spaces = [item for item in self.main_dict.items()]
        else:
            spaces = [item for item in self.main_dict.items() if item[0] in space_names] 

        for space, space_res in spaces:                
                                       
            if analysis_names == 'all':
                analyses = [item for item in space_res.items()]
            else:
                analyses = [item for item in space_res.items() if item[0] in analysis_names] 
                         
            for analysis, analysis_res in analyses:       
                if subject_ids == 'all':
                    subjects = [item for item in analysis_res.items()]
                else:
                    subjects = [item for item in analysis_res.items() if item[0] in subject_ids]

                subjects_pla = [s[0] for s in subjects if s[0] in self.groups_dict[base_group]]
                
                for subj, subj_res in subjects:

                    current_sj_group = self.find_group(subj)
                    #print(subj)
                    #print(current_sj_group)
                    if current_sj_group != None and current_sj_group != base_group:

                        gen_subj = subj.split('_')[0]
                        
                        subj_pla = [s for s in subjects_pla if gen_subj in s][0]
                        #print(subj_pla)

                        p_r = analysis_res[subj]['Processed Results']
                        p_r_pla = analysis_res[subj_pla]['Processed Results'] 

                        if parameter_names == 'all':
                            parameters = [item for item in p_r.items() if 'Alpha' not in item[0]]
                        else:
                            parameters = [item for item in p_r.items() if item[0] in parameter_names]

                        for param, param_res in parameters:
                            if f'{current_sj_group}' not in param and f'{base_group}' not in param:
                                models = [item for item in param_res.items()]
                                for model, model_res in models:
                                    if model in p_r['Alpha']:
                                        this_alpha = (p_r['Alpha'][model] + p_r_pla['Alpha'][model])/2

                                        this_alpha[p_r['Alpha'][model]<=0] = 0
                                        this_alpha[p_r_pla['Alpha'][model]<=0] = 0

                                        p_r[f'Mean Masked RSq {current_sj_group}-{base_group}'][model] = this_alpha
                                    else:
                                        this_alpha = np.ones_like(p_r[param][model])

                                    p_r[f'{param} {current_sj_group}/{base_group}'][model] = (p_r[param][model] - p_r_pla[param][model])/p_r_pla[param][model]

                                    p_r[f'{param} {current_sj_group}/{base_group}'][model][this_alpha<=0] = 0


    
        return



        
    def define_rois_and_flatmaps(self, fs_dir, output_rois_path, import_flatmaps, output_rois, hcp_atlas_path = None):
        self.idx_rois = dd(dict)
        self.idx_rois_borders = dd(dict)
        self.fs_dir = fs_dir
        self.get_spaces()
        self.get_subjects(self.main_dict)
        for subj in self.subjects:
            pycortex_subj = subj.split('_')[0]
            if os.path.exists(opj(self.fs_dir, pycortex_subj)):
            
                if pycortex_subj not in cortex.db.subjects:
                    print("importing subject from freesurfer")
                    print(f"note: this command often files when executed on mac OS via jupyter notebook.\
                          Rather, source freesurfer and execute it in ipython: \
                              cortex.freesurfer.import_subj({pycortex_subj}, freesurfer_subject_dir={self.fs_dir}, \
                          whitematter_surf='smoothwm')")
                    cortex.freesurfer.import_subj(pycortex_subj, freesurfer_subject_dir=self.fs_dir, 
                          whitematter_surf='smoothwm')
                    #force udate db
                    cortex.db._subjects = None
                    cortex.db.subjects
                    
                if import_flatmaps:
                    try:
                        print('importing flatmaps from freesurfer')
                        cortex.freesurfer.import_flat(subject=pycortex_subj, patch='full', hemis=['lh', 'rh'], 
                                                  freesurfer_subject_dir=self.fs_dir, clean=True)
                    except Exception as e:
                        print(e)
                        pass
                            
                if os.path.exists(opj(self.fs_dir, pycortex_subj, 'label', 'lh.wang2015atlas.V1d.label')):
                    src_subject=pycortex_subj
                else:
                    src_subject='fsaverage'
                        
            
                wang_rois = ["V1v", "V1d", "V2v", "V2d", "V3v", "V3d", "hV4", "VO1", "VO2", "PHC1", "PHC2",
                    "TO2", "TO1", "LO2", "LO1", "V3B", "V3A", "IPS0", "IPS1", "IPS2", "IPS3", "IPS4", 
                    "IPS5", "SPL1", "FEF"]
                for roi in wang_rois:
                    try:
                        self.idx_rois[subj][roi], _ = cortex.freesurfer.get_label(pycortex_subj,
                                                              label='wang2015atlas.'+roi,
                                                              fs_dir=self.fs_dir,
                                                              src_subject=src_subject,
                                                              verbose=True)
                    except Exception as e:
                        print(e)
            
                self.idx_rois[subj]['visual_system'] = np.concatenate(tuple([self.idx_rois[subj][roi] for roi in self.idx_rois[subj]]), axis=0)
                self.idx_rois[subj]['V1'] = np.concatenate((self.idx_rois[subj]['V1v'],self.idx_rois[subj]['V1d']))
                self.idx_rois[subj]['V2'] = np.concatenate((self.idx_rois[subj]['V2v'],self.idx_rois[subj]['V2d']))
                self.idx_rois[subj]['V3'] = np.concatenate((self.idx_rois[subj]['V3v'],self.idx_rois[subj]['V3d']))
                self.idx_rois[subj]['VO'] = np.concatenate((self.idx_rois[subj]['VO1'],self.idx_rois[subj]['VO2']))
                self.idx_rois[subj]['TO'] = np.concatenate((self.idx_rois[subj]['TO1'],self.idx_rois[subj]['TO2']))
                self.idx_rois[subj]['LO'] = np.concatenate((self.idx_rois[subj]['LO1'],self.idx_rois[subj]['LO2']))
                self.idx_rois[subj]['V3AB'] = np.concatenate((self.idx_rois[subj]['V3A'],self.idx_rois[subj]['V3B']))
                self.idx_rois[subj]['IPS'] = np.concatenate(tuple([self.idx_rois[subj][rroi] for rroi in self.idx_rois[subj].keys() if 'IPS' in rroi]))            
                #parse custom ROIs if they have been created
                for roi in [el for el in os.listdir(opj(self.fs_dir, pycortex_subj, 'label')) if 'custom' in el]:
                    roi = roi.replace('lh.','').replace('rh.','').replace('.label','')
                    try:
                        self.idx_rois[subj][roi], _ = cortex.freesurfer.get_label(pycortex_subj,
                                                              label=roi,
                                                              fs_dir=self.fs_dir,
                                                              src_subject=subj,
                                                              hemisphere=['lh', 'rh'])
                    except Exception as e:
                        print(e)
                        pass

                #parse glasser ROIs if they have been created (keep hemispheres separate)
                # for roi in [el for el in os.listdir(opj(self.fs_dir, subj, 'label')) if 'glasser' in el]:
                #     roi = roi.replace('.label','')#.replace('lh.','').replace('rh.','')
                #     try:
                #         if 'lh.' in roi:
                #             self.idx_rois[subj][roi], _ = cortex.freesurfer.get_label(subj,
                #                                               label=roi.replace('lh.',''),
                #                                               fs_dir=self.fs_dir,
                #                                               src_subject=subj,
                #                                               hemisphere=['lh'])
                #         elif 'rh.' in roi:
                #             self.idx_rois[subj][roi], _ = cortex.freesurfer.get_label(subj,
                #                                               label=roi.replace('rh.',''),
                #                                               fs_dir=self.fs_dir,
                #                                               src_subject=subj,
                #                                               hemisphere=['rh'])                        
                #     except Exception as e:
                #         print(e)
                #         pass

                #parse glasser ROIs if they have been created
                for roi in [el for el in os.listdir(opj(self.fs_dir, pycortex_subj, 'label')) if 'glasser' in el]:
                    roi = roi.replace('.label','').replace('lh.','').replace('rh.','')
                    try:
                        
                        self.idx_rois[subj][roi], _ = cortex.freesurfer.get_label(pycortex_subj,
                                                              label=roi,
                                                              fs_dir=self.fs_dir,
                                                              src_subject=subj,
                                                              hemisphere=['lh', 'rh'])                   
                    except Exception as e:
                        print(e)
                        pass


                #parse exclusion ROIs if they have been created
                for roi in [el for el in os.listdir(opj(self.fs_dir, pycortex_subj, 'label')) if 'excl' in el]:
                    roi = roi.replace('lh.','').replace('rh.','').replace('.label','')
                    try:
                        self.idx_rois[subj][roi], _ = cortex.freesurfer.get_label(pycortex_subj,
                                                              label=roi,
                                                              fs_dir=self.fs_dir,
                                                              src_subject=subj,
                                                              hemisphere=['lh', 'rh'])
                    except Exception as e:
                        print(e)
                        pass
                    
            elif 'HCP' in self.spaces and (subj.isdecimal() or '999999' in subj) and hcp_atlas_path != None:
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
                            roi_data = np.zeros(cortex.db.get_surfinfo(pycortex_subj).data.shape).astype('bool')
                            roi_data[roi_idx] = 1
                            roi_vertices=cortex.Vertex(roi_data, pycortex_subj)
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
        
        if 'fsaverage' in self.spaces:
            for sj in [s for s in self.subjects if 'fsaverage' in s and 'fsaverage' != s]:
                self.idx_rois[sj] = deepcopy(self.idx_rois['fsaverage'])


        return
    
                    
                
    def compute_roi_borders(self, subject_ids = ['fsaverage'], rois_prefix='glasser', only_rois = [], previous_borders_path=None):
               
        
        def comp_border(face):

            return np.any(np.isin([np.isin(face,roi).sum() for roi in rois],[1,2]))

        
        for subj in subject_ids:
            assert subj in self.idx_rois, "subject not found"

            pycortex_subj = subj.split('_')[0]
            
            if previous_borders_path == None:
            
                left, right = getattr(cortex.db,pycortex_subj).surfaces.inflated.get()
    
                faces = np.concatenate((left[1],len(left[0])+right[1]))
                
            
                zz = np.zeros(len(left[0])+len(right[0]))
                
                rois_names = [r for r in self.idx_rois[subj] if rois_prefix in r]

                if len(only_rois)>0:
                    rois_names = [r for r in rois_names if r in only_rois]

                rois = [self.idx_rois[subj][r] for r in rois_names]


           
                
                res = np.array(Parallel(n_jobs=8, verbose=True)(delayed(comp_border)(face) for face in faces))
                    
                zz[np.ravel(faces[res])] = 1
                            
                if 'ses-' in subj:
                    for sj in [s for s in self.idx_rois if subj.split('_')[0] in s]:
                        self.idx_rois_borders[sj][rois_prefix] = np.copy(zz)
                else:
                    self.idx_rois_borders[subj][rois_prefix] = np.copy(zz)
            else:
                
                if 'ses-' in subj:
                    for sj in [s for s in self.idx_rois if subj.split('_')[0] in s]:
                        self.idx_rois_borders[sj][rois_prefix] = np.load(previous_borders_path)
                else:
                    self.idx_rois_borders[subj][rois_prefix] = np.load(previous_borders_path)
            
            return
                        
            

    def set_alpha(self, space_names='all', only_models=None, ecc_min=0, ecc_max=5, alpha_weight='RSq', threshold_li=True, excluded_rois=[], tc_min=dict()):
        
        self.only_models=only_models
        self.tc_min = tc_min

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

                        if alpha_weight not in p_r:
                            print("alpha weight not found, using rsq")
                            alpha_weight = 'RSq'
                        else:
                            if len([model for model in only_models if model in p_r[alpha_weight]])==0:
                                print("alpha weight not found, using rsq")
                                alpha_weight = 'RSq'

                        models = p_r[alpha_weight].keys()
                                                
                        if 'Timecourse Stats' in p_r.keys(): #space != 'fsaverage' and 'rsq' not in subj:
                            tc_stats = p_r['Timecourse Stats']
                        else:
                            tc_stats=dict()
                            tc_stats['Mean'] = np.ones_like(subj_res['mask']) #p_r[alpha_weight][[mod for mod in self.only_models if mod in p_r[alpha_weight]][0]])
                       
                        #######Raw bold timecourse vein threshold
                        if subj not in self.tc_min:
                            self.tc_min[subj] = -np.inf
                            
                        ######limits for eccentricity
                        self.ecc_min=ecc_min
                        self.ecc_max=ecc_max
                     
              
                        #housekeeping
                        if only_models == None:
                            rsq = np.vstack(tuple([elem for _,elem in p_r[alpha_weight].items()])).T
                            ecc = np.vstack(tuple([elem for _,elem in p_r['Eccentricity'].items()])).T

                        else:
                            
                            rsq = np.vstack(tuple([elem for k,elem in p_r[alpha_weight].items() if k in only_models])).T
                            ecc = np.vstack(tuple([elem for k,elem in p_r['Eccentricity'].items() if k in only_models])).T                            
            
                        #alpha dictionary
                        p_r['Alpha'] = {}          
                        p_r['Alpha']['all'] = rsq.max(-1) * (ecc.min(-1)<self.ecc_max) * (ecc.max(-1)>self.ecc_min) * (rsq.min(-1)>0) #* (p_r['Noise Ceiling']['Noise Ceiling (RSq)']>0)
                        

                        if 'Mean' in tc_stats:
                            #p_r['Alpha']['all'] *= (tc_stats['Mean']>self.tc_min[subj])
                            p_r['Alpha']['all'][tc_stats['Mean']<self.tc_min[subj]] = -1

                        
                        for model in models:
                            p_r['Alpha'][model] = p_r[alpha_weight][model] * (p_r['Eccentricity'][model]>self.ecc_min) * (p_r['Eccentricity'][model]<self.ecc_max)\
                                 #* (p_r['Noise Ceiling']['Noise Ceiling (RSq)']!=0)
                            if 'Mean' in tc_stats:
                                #p_r['Alpha'][model] *= (tc_stats['Mean']>self.tc_min[subj])
                                p_r['Alpha'][model][tc_stats['Mean']<self.tc_min[subj]] = -1
                                
                            if threshold_li and '#Subjects with CVRSq>0' in p_r:
                                
                                if model in p_r['#Subjects with CVRSq>0']:
                                    
                                    min_number_sjs = filters.threshold_li(p_r['#Subjects with CVRSq>0'][model])
                                    print(f"thresholding at {min_number_sjs} subjects")
                                    
                                    p_r['Alpha'][model] *= (p_r['#Subjects with CVRSq>0'][model]>min_number_sjs)
                                    p_r['Alpha']['all'] *= (p_r['#Subjects with CVRSq>0'][model]>min_number_sjs)
                            
                            if len(excluded_rois)>0:
                                if 'fsaverage' in space or 'fsaverage' in subj:
                                    roi_subj = 'fsaverage'
                                else:
                                    roi_subj = subj
                                if all(r in self.idx_rois[roi_subj] for r in excluded_rois):

                                    p_r['Alpha'][model] = (inverse_roi_mask(np.concatenate(tuple([self.idx_rois[roi_subj][r] for r in excluded_rois])), p_r['Alpha'][model]))
                                    p_r['Alpha']['all'] = (inverse_roi_mask(np.concatenate(tuple([self.idx_rois[roi_subj][r] for r in excluded_rois])), p_r['Alpha']['all']))   
                                else:
                                    print("WARNING: excluded_rois contains undefined rois. roi exclusion not performed.")
                        
                       
        return


    def pycortex_plots(self, rois, rsq_thresh,
                       space_names = 'fsnative', analysis_names = 'all', subject_ids='all', param_diffs=[],
                       timecourse_folder = None, screenshot_paths = [], save_colorbars=False, pycortex_cmap = 'nipy_spectral',
                       rsq_max_opacity = 0.5, pycortex_image_path = None, roi_borders_name = None,
                       clickerfun='standard'):        
        pl.rcParams.update({'font.size': 16})
        pl.rcParams.update({'pdf.fonttype':42})        
        self.click=0
        if pycortex_image_path != None:
            if not os.path.exists(pycortex_image_path):
                os.makedirs(pycortex_image_path)
        self.pycortex_image_path = pycortex_image_path

        #######PYCORTEX PICKERFUNS
        #function to plot prf and timecourses when clicking surface vertex in webgl        

        def groups_clicker_function(voxel,hemi,vertex):
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
                
                    
                print('recovering data and model timecourses...')

                color_dict = {'placebo':'blue','5mg':'orange','10mg':'green'}                
    
                #recover needed information
                an_info = subj_res['analysis_info']       
                #this was added later
                if 'normalize_integral_dx' not in an_info:
                    an_info['normalize_integral_dx'] = False    


                gen_subj = subj.split('_')[0]

                this_subj_groups = dict()
                
                for group in self.groups:
                    this_subj_groups[group] = [s for s in self.subjects if gen_subj in s and s in self.groups_dict[group]][0]
          

    
                if not hasattr(self, 'prf_stim') or self.prf_stim.task_names != an_info['task_names']:
        

                    self.prf_stim = create_full_stim(screenshot_paths=screenshot_paths,
                                n_pix=an_info['n_pix'],
                                discard_volumes=an_info['discard_volumes'],
                                dm_edges_clipping=an_info['dm_edges_clipping'],
                                screen_size_cm=an_info['screen_size_cm'],
                                screen_distance_cm=an_info['screen_distance_cm'],
                                TR=an_info['TR'],
                                task_names=an_info['task_names'],
                                normalize_integral_dx=an_info['normalize_integral_dx'])
                    

                pl.ion()
    
                if self.click==0:
                    gs_kw = dict(width_ratios=[1, 1, 1], height_ratios=[1, 1, 1, 20/8])
                    self.cbar=dict()

                    self.f, self.axes = pl.subplot_mosaic([[f'timecourse' for group in this_subj_groups.keys()],
                                                [f'timecourse diffs' for group in this_subj_groups.keys()],
                                                [f'{group} prf' for group in this_subj_groups.keys()],
                                                [f'{group} vertex info' for group in this_subj_groups.keys()]],
                                                gridspec_kw=gs_kw,
                                                figsize=(18, 44))#, tight_layout=True)

                else:
                    for k in self.cbar:
                        self.cbar[k].remove()
                    for k, ax in self.axes.items():
                        ax.clear()


                preds = dd(dict)
                prfs = dd(dict) 
                tc = dd(dict)
                tc_err = dd(dict)
                tc_test = dd(dict)
                tc_fit = dd(dict)

                tc_full = dict()
                tc_full_err = dict()

                tc_full_test = dict()
                tc_full_fit = dict()

                base_group = 'placebo'
    
                for n_group, group in enumerate(this_subj_groups):

                    this_subj = this_subj_groups[group]
                    this_subj_res = analysis_res[this_subj]

                    vertex_info = ""



                    tc_paths = [str(path) for path in sorted(Path(timecourse_folder).glob(f"{this_subj}_timecourse_space-{an_info['fitting_space']}_task-*_run-*.npy"))]   

                    mask_paths = [tc_path.replace('timecourse_','mask_') for tc_path in tc_paths]
                    #all_task_names = np.unique(np.array([elem.replace('task-','') for path in tc_paths for elem in path.split('_')  if 'task' in elem]))
                    all_runs = np.unique(np.array([int(elem.replace('run-','').replace('.npy','')) for path in tc_paths for elem in path.split('_')  if 'run-' in elem]))


                    
                    for i,task in enumerate(an_info['task_names']):
                        if task not in self.prf_stims:
                            self.prf_stims[task] = create_full_stim(screenshot_paths=[screenshot_paths[i]],
                                    n_pix=an_info['n_pix'],
                                    discard_volumes=an_info['discard_volumes'],
                                    dm_edges_clipping=an_info['dm_edges_clipping'],
                                    screen_size_cm=an_info['screen_size_cm'],
                                    screen_distance_cm=an_info['screen_distance_cm'],
                                    TR=an_info['TR'],
                                    task_names=[task],
                                    normalize_integral_dx=an_info['normalize_integral_dx'])                    
                            
                        tc_runs=[]

                        #red_per = self.prf_stims[task].late_iso_dict['periods'][(self.prf_stims[task].late_iso_dict['periods']<20) | (self.prf_stims[task].late_iso_dict['periods']>234)]

                        
                        for run in all_runs:
                            mask_run = [np.load(mask_path) for mask_path in mask_paths if f"task-{task}_" in mask_path and f"run-{run}." in mask_path][0]
                            
                            if space == 'HCP':
                                tc_run_idx = np.sum(this_subj_res['mask'][:index])
                            else:
                                tc_run_idx = np.sum(mask_run[:index])
                            
                            tc_runs.append([np.load(tc_path)[tc_run_idx] for tc_path in tc_paths if f"task-{task}_" in tc_path and f"run-{run}." in tc_path][0])


                            tc_runs[-1] -= np.median(tc_runs[-1][...,self.prf_stims[task].late_iso_dict[task]], axis=-1)[...,np.newaxis]

                            
                        
                        tc[group][task] = np.mean(tc_runs, axis=0)
                        
                        tc_err[group][task] = sem(tc_runs, axis=0)
                        
                        tc[group][task] -= np.median(tc[group][task][...,self.prf_stims[task].late_iso_dict[task]], axis=-1)[...,np.newaxis]
                        #tc[task] -= np.median(tc[task][...,red_per], axis=-1)[...,np.newaxis]


                        #if 'CVmean' in analysis or 'CVmedian' in analysis:
                        #    vertex_info+=f"WARNING: predictions based on mean/median CV parameters are not usually meaningful\n"

                        vertex_info+=f"{task} late iso dict median: {np.median(tc[group][task][self.prf_stims[task].late_iso_dict[task]]):.4f}\n"

                    
                        #fit and test timecourses separately
                        if an_info['crossvalidate']:
                            if 'fit_runs' in an_info:
                                tc_test[group][task] = np.mean([tc_runs[i] for i in all_runs if i not in an_info['fit_runs']], axis=0)
                                tc_fit[group][task] = np.mean([tc_runs[i] for i in all_runs if i in an_info['fit_runs']], axis=0)
                            elif 'fit_task' not in an_info:
                                print("warning: fit runs not specified. guessing based on space. check code.")
                                if space == 'HCP':
                                    fit_runs = [0]
                                elif space == 'fsnative':
                                    fit_runs = [0,2,4]
                                tc_test[group][task] = np.mean([tc_runs[i] for i in all_runs if i not in fit_runs], axis=0)
                                tc_fit[group][task] = np.mean([tc_runs[i] for i in all_runs if i in fit_runs], axis=0)
                                
                            #tc_test[task] *= (100/tc_test[task].mean(-1))[...,np.newaxis]
                            tc_test[group][task] -= np.median(tc_test[group][task][...,self.prf_stims[task].late_iso_dict[task]], axis=-1)[...,np.newaxis]
                            #tc_fit[task] *= (100/tc_fit[task].mean(-1))[...,np.newaxis]
                            tc_fit[group][task] -= np.median(tc_fit[group][task][...,self.prf_stims[task].late_iso_dict[task]], axis=-1)[...,np.newaxis]     
                        
    
                    
                    tc_full[group] = np.concatenate(tuple([tc[group][task] for task in tc[group]]), axis=0)
                    tc_err[group] = np.concatenate(tuple([tc_err[group][task] for task in tc_err[group]]), axis=0)
                    
                    if an_info['crossvalidate'] and 'fit_task' not in an_info:
                        tc_full_test[group] = np.concatenate(tuple([tc_test[group][task] for task in tc_test[group]]), axis=0)
                        tc_full_fit[group] = np.concatenate(tuple([tc_fit[group][task] for task in tc_fit[group]]), axis=0)   
                        #timecourse reliability stats
                        vertex_info+="CV timecourse reliability stats\n"
                        vertex_info+=f"fit-test timecourses corrcoeff {np.corrcoef(tc_full_test[group],tc_full_fit[group])[0,1]:.4f}\n"
                        vertex_info+=f"fit-test timecourses R-squared {1-np.sum((tc_full_fit[group]-tc_full_test[group])**2)/(tc_full_test[group].var(-1)*tc_full_test[group].shape[-1]):.4f}\n\n"
                

                    
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
                            internal_idx = np.sum(this_subj_res['mask'][:index])

                        
                        if 'CVmean' in analysis or 'CVmedian' in analysis:
                            #need to combine params/predictions from folds?
                            pass

                        
                        params = np.copy(this_subj_res['Results'][model][internal_idx,:-1])
                        

                        preds[group][model] = self.prf_models[model].return_prediction(*list(params))[0]
                        
                        
                        #mdff = (preds[model]-tc_full).mean()
                        #tc_full += mdff
                        
                        np.set_printoptions(suppress=True, precision=4)
                        vertex_info+=f"{this_subj} {model} params: {params} \n"
                        vertex_info+=f"Rsq {this_subj} {model} full tc: {1-np.sum((preds[group][model]-tc_full[group])**2)/(tc_full[group].var(-1)*tc_full[group].shape[-1]):.4f}\n"
                        
                        if an_info['crossvalidate'] and 'fit_task' not in an_info:
                            vertex_info+=f"Rsq {this_subj} {model} fit tc: {1-np.sum((preds[group][model]-tc_full_fit[group])**2)/(tc_full_fit[group].var(-1)*tc_full_fit[group].shape[-1]):.4f}\n"
                            vertex_info+=f"Rsq {this_subj} {model} test tc: {1-np.sum((preds[group][model]-tc_full_test[group])**2)/(tc_full_test[group].var(-1)*tc_full_test[group].shape[-1]):.4f}\n"
        
        
                            
                            
                            # if model != 'Gauss':
                            #     tc_full_test_gauss_resid = tc_full_test[tc_full_test<0]#(tc_full_test - preds['Gauss'][0])
                            #     model_gauss_resid = preds[model][tc_full_test<0]#(preds[model] - preds['Gauss'])[0]
        
                            #     vertex_info+=f"Rsq {this_subj} {model} negative portions test tc: {1-np.sum((model_gauss_resid-tc_full_test_gauss_resid)**2)/(tc_full_test_gauss_resid.var(-1)*tc_full_test_gauss_resid.shape[-1]):.4f}\n"
                            #     vertex_info+=f"Rsq {this_subj} {model} negative portions test pearson R {pearsonr(tc_full_test_gauss_resid,model_gauss_resid)[0]:.4f}\n"
                                
                                
                                #vertex_info+=f"Rsq {model} gauss resid test tc: {1-np.sum((model_gauss_resid-tc_full_test_gauss_resid)**2)/(tc_full_test_gauss_resid.var(-1)*tc_full_test_gauss_resid.shape[-1]):.4f}\n"
                                #vertex_info+=f"Rsq {model} weighted resid tc: {1-np.sum(((preds[model]-preds['Gauss'])*(preds[model]-tc_full_fit))**2)/(tc_full_fit.var(-1)*tc_full_fit.shape[-1]):.4f}\n"
                            
                        prfs[group][model] = create_model_rf_wrapper(model,self.prf_stim,params,an_info['normalize_RFs'])
                        
                        for key in this_subj_res['Processed Results']:
                            #if model in subj_res['Processed Results'][key]:
                            for mm in this_subj_res['Processed Results'][key]:
                                vertex_info+=f"{this_subj} {key} {mm} {this_subj_res['Processed Results'][key][mm][index]:.4f}\n"
                        
                        vertex_info+="\n"
                        

                    
                    tseconds = an_info['TR']*np.arange(len(tc_full[group]))
                    
                    self.axes['timecourse'].plot(tseconds,np.zeros(len(tc_full[group])),linestyle='--',linewidth=0.1, color='black', zorder=0)
                    self.axes['timecourse diffs'].plot(tseconds,np.zeros(len(tc_full[group])),linestyle='--',linewidth=0.1, color='black', zorder=0)
                          
                    
                    self.axes['timecourse'].errorbar(tseconds, tc_full[group], yerr=0, label='Data',  marker = 's', mfc=color_dict[group], mec='k', markersize=6,  linewidth=1, zorder=1) 
                    self.axes['timecourse diffs'].errorbar(tseconds, tc_full[group] - tc_full[base_group], yerr=0, label='Data',  marker = 's', mfc=color_dict[group], mec='k', markersize=6,  linewidth=1, zorder=1) 
                     
        
                    
                    for model in self.only_models: 
                        if 'Norm' in model:
                            self.axes['timecourse'].plot(tseconds, preds[group][model], linewidth=3, color=color_dict[group], label=f"Norm ({this_subj_res['Processed Results']['RSq'][model][index]:.2f})", zorder=2)
                        elif model == 'DoG':
                            self.axes['timecourse'].plot(tseconds, preds[group][model], linewidth=3, color=color_dict[group], label=f"DoG ({this_subj_res['Processed Results']['RSq'][model][index]:.2f})", zorder=3)
                        elif model == 'CSS':
                            self.axes['timecourse'].plot(tseconds, preds[group][model], linewidth=3, color=color_dict[group], label=f"CSS ({this_subj_res['Processed Results']['RSq'][model][index]:.2f})", zorder=4)
                        elif model == 'Gauss':
                            self.axes['timecourse'].plot(tseconds, preds[group][model], linewidth=3, color=color_dict[group], label=f"Gauss ({this_subj_res['Processed Results']['RSq'][model][index]:.2f})", zorder=5)


                        if 'Norm' in model:
                            self.axes['timecourse diffs'].plot(tseconds, preds[group][model]-preds[base_group][model], linewidth=3, color=color_dict[group], label=f"Norm ({this_subj_res['Processed Results']['RSq'][model][index]:.2f})", zorder=2)
                        elif model == 'DoG':
                            self.axes['timecourse diffs'].plot(tseconds, preds[group][model]-preds[base_group][model], linewidth=3, color=color_dict[group], label=f"DoG ({this_subj_res['Processed Results']['RSq'][model][index]:.2f})", zorder=3)
                        elif model == 'CSS':
                            self.axes['timecourse diffs'].plot(tseconds, preds[group][model]-preds[base_group][model], linewidth=3, color=color_dict[group], label=f"CSS ({this_subj_res['Processed Results']['RSq'][model][index]:.2f})", zorder=4)
                        elif model == 'Gauss':
                            self.axes['timecourse diffs'].plot(tseconds, preds[group][model]-preds[base_group][model], linewidth=3, color=color_dict[group], label=f"Gauss ({this_subj_res['Processed Results']['RSq'][model][index]:.2f})", zorder=5)
                            
                    
                    fillin_surround = False
                    fillin_late_iso_dict = False
                    fillin_emptyscreen = True

                    if fillin_late_iso_dict:
                        for tc_ax in ['timecourse', 'timecourse diffs']:
                            if n_group == len(this_subj_groups.keys())-1:
                                min_lid = np.zeros_like(tseconds)
                                max_lid = np.zeros_like(tseconds)
                                min_lid[self.prf_stim.late_iso_dict['periods']] = self.axes[tc_ax].get_ylim()[0]
                                max_lid[self.prf_stim.late_iso_dict['periods']] = self.axes[tc_ax].get_ylim()[1]

                                self.axes[tc_ax].fill_between(tseconds,min_lid,max_lid,label='Late iso dict', alpha=0.2, color='gray')

                    if fillin_emptyscreen:
                        for tc_ax in ['timecourse', 'timecourse diffs']:
                            if n_group == len(this_subj_groups.keys())-1:
                                min_lid = np.zeros_like(tseconds)
                                max_lid = np.zeros_like(tseconds)
                                min_lid[np.sum(self.prf_stim.design_matrix, axis=(0, 1)) == 0] = self.axes[tc_ax].get_ylim()[0]
                                max_lid[np.sum(self.prf_stim.design_matrix, axis=(0, 1)) == 0] = self.axes[tc_ax].get_ylim()[1]

                                self.axes[tc_ax].fill_between(tseconds,min_lid,max_lid,label='Empty Screen', alpha=0.2, color='gray')
                    
                    if fillin_surround:
                    
                        if np.any(['Norm' in model for model in self.only_models]) or 'DoG' in self.only_models:
                            try:
                                if ((this_subj_res['Processed Results']['RSq']['DoG'][index]-this_subj_res['Processed Results']['RSq']['Gauss'][index]) > 0.05) or ((this_subj_res['Processed Results']['RSq']['Norm_abcd'][index]-this_subj_res['Processed Results']['RSq']['CSS'][index]) > 0.05):
                                    surround_effect = np.min([preds[group][model] for model in preds[group] if 'Norm' in model or 'DoG' in model],axis=0)
                                    surround_effect[surround_effect>0] = preds[group]['Gauss'][surround_effect>0]
                                    self.axes['timecourse'].fill_between(tseconds,
                                                                surround_effect, 
                                                    preds[group]['Gauss'], label='Surround suppression', alpha=0.2, color='gray')
                            except:
                                pass
        

                    for tc_ax in ['timecourse', 'timecourse diffs']:        
                        self.axes[tc_ax].legend(ncol=3, fontsize=8, loc=9)

                        self.axes[tc_ax].set_xlim(0,tseconds.max())
                    
                    if prfs[group]['Norm_abcd'][0].min() < 0:
                        im = self.axes[f'{group} prf'].imshow(prfs[group]['Norm_abcd'][0], cmap='RdBu_r', vmin=prfs[group]['Norm_abcd'][0].min(), vmax=-prfs[group]['Norm_abcd'][0].min())
                    else:
                        im = self.axes[f'{group} prf'].imshow(prfs[group]['Norm_abcd'][0], cmap='RdBu_r', vmax=prfs[group]['Norm_abcd'][0].max(), vmin=-prfs[group]['Norm_abcd'][0].max())
                        
                    self.cbar[f'{group} prf'] = colorbar(im)
                    self.axes[f'{group} prf'].set_title(f'{group} prf')
                    
                    #self.cbar = self.f.colorbar(im, ax=self.axes[0,1], use_gridspec=True)
                    
                    self.axes[f'{group} prf'].axis('on')            
                    self.axes[f'{group} vertex info'].axis('off')
                    self.axes[f'{group} vertex info'].text(0,1,vertex_info, fontsize=10, va='top')
                
                
                self.click+=1
          
            return

      


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
                
                #index = 164215
                
                print('recovering data and model timecourses...')
                
                vertex_info = ""
    
                #recover needed information
                an_info = subj_res['analysis_info']       
                #this was added later
                if 'normalize_integral_dx' not in an_info:
                    an_info['normalize_integral_dx'] = False                
    
                if not hasattr(self, 'prf_stim') or self.prf_stim.task_names != an_info['task_names']:
        

                    self.prf_stim = create_full_stim(screenshot_paths=screenshot_paths,
                                n_pix=an_info['n_pix'],
                                discard_volumes=an_info['discard_volumes'],
                                dm_edges_clipping=an_info['dm_edges_clipping'],
                                screen_size_cm=an_info['screen_size_cm'],
                                screen_distance_cm=an_info['screen_distance_cm'],
                                TR=an_info['TR'],
                                task_names=an_info['task_names'],
                                normalize_integral_dx=an_info['normalize_integral_dx'])
    
                if 'ses-all' in subj:

                    tc_paths = [str(path) for path in sorted(Path(timecourse_folder).glob(f"{subj.split('_')[0]}_ses-*_timecourse_space-{an_info['fitting_space']}_task-*_run-*.npy"))]   

                    mask_paths = [tc_path.replace('timecourse_','mask_') for tc_path in tc_paths]
                    #all_task_names = np.unique(np.array([elem.replace('task-','') for path in tc_paths for elem in path.split('_')  if 'task' in elem]))
                    all_runs = np.unique(np.array([int(elem.replace('run-','').replace('.npy','')) for path in tc_paths for elem in path.split('_')  if 'run-' in elem]))
                    all_ses = np.unique(np.array([int(elem.replace('ses-','').replace('.npy','')) for path in tc_paths for elem in path.split('_')  if 'ses-' in elem]))

    
                else:    
                    tc_paths = [str(path) for path in sorted(Path(timecourse_folder).glob(f"{subj}_timecourse_space-{an_info['fitting_space']}_task-*_run-*.npy"))]   

                    mask_paths = [tc_path.replace('timecourse_','mask_') for tc_path in tc_paths]
                    #all_task_names = np.unique(np.array([elem.replace('task-','') for path in tc_paths for elem in path.split('_')  if 'task' in elem]))
                    all_runs = np.unique(np.array([int(elem.replace('run-','').replace('.npy','')) for path in tc_paths for elem in path.split('_')  if 'run-' in elem]))
                    all_ses = []
    
                tc = dict()
                tc_err = dict()
                tc_test = dict()
                tc_fit = dict()
                
                for i,task in enumerate(an_info['task_names']):
                    if task not in self.prf_stims:
                        self.prf_stims[task] = create_full_stim(screenshot_paths=[screenshot_paths[i]],
                                n_pix=an_info['n_pix'],
                                discard_volumes=an_info['discard_volumes'],
                                dm_edges_clipping=an_info['dm_edges_clipping'],
                                screen_size_cm=an_info['screen_size_cm'],
                                screen_distance_cm=an_info['screen_distance_cm'],
                                TR=an_info['TR'],
                                task_names=[task],
                                normalize_integral_dx=an_info['normalize_integral_dx'])                    
                        
                    tc_runs=[]

                    #red_per = self.prf_stims[task].late_iso_dict['periods'][(self.prf_stims[task].late_iso_dict['periods']<20) | (self.prf_stims[task].late_iso_dict['periods']>234)]

                    
                    for run in all_runs:
                        if len(all_ses)>0:
                            for ses in all_ses:
                                mask_run = [np.load(mask_path) for mask_path in mask_paths if f"task-{task}_" in mask_path and f"run-{run}." in mask_path and f"ses-{ses}" in mask_path][0]
                                
                                if space == 'HCP':
                                    tc_run_idx = np.sum(subj_res['mask'][:index])
                                else:
                                    tc_run_idx = np.sum(mask_run[:index])
                                
                                tc_runs.append([np.load(tc_path)[tc_run_idx] for tc_path in tc_paths if f"task-{task}_" in tc_path and f"run-{run}." in tc_path and f"ses-{ses}" in tc_path][0])


                                tc_runs[-1] -= np.median(tc_runs[-1][...,self.prf_stims[task].late_iso_dict[task]], axis=-1)[...,np.newaxis]
                        else:
                            mask_run = [np.load(mask_path) for mask_path in mask_paths if f"task-{task}_" in mask_path and f"run-{run}." in mask_path][0]
                            
                            if space == 'HCP':
                                tc_run_idx = np.sum(subj_res['mask'][:index])
                            else:
                                tc_run_idx = np.sum(mask_run[:index])
                            
                            tc_runs.append([np.load(tc_path)[tc_run_idx] for tc_path in tc_paths if f"task-{task}_" in tc_path and f"run-{run}." in tc_path][0])


                            tc_runs[-1] -= np.median(tc_runs[-1][...,self.prf_stims[task].late_iso_dict[task]], axis=-1)[...,np.newaxis]

                        
                      
                    tc[task] = np.mean(tc_runs, axis=0)
                    
                    tc_err[task] = sem(tc_runs, axis=0)
                    
                    tc[task] -= np.median(tc[task][...,self.prf_stims[task].late_iso_dict[task]], axis=-1)[...,np.newaxis]
                    #tc[task] -= np.median(tc[task][...,red_per], axis=-1)[...,np.newaxis]


                    #if 'CVmean' in analysis or 'CVmedian' in analysis:
                    #    vertex_info+=f"WARNING: predictions based on mean/median CV parameters are not usually meaningful\n"

                    vertex_info+=f"{task} late iso dict median: {np.median(tc[task][self.prf_stims[task].late_iso_dict[task]]):.3f}\n"

                    
                    #fit and test timecourses separately
                    if an_info['crossvalidate']:
                        if 'fit_runs' in an_info:
                            tc_test[task] = np.mean([tc_runs[i] for i in all_runs if i not in an_info['fit_runs']], axis=0)
                            tc_fit[task] = np.mean([tc_runs[i] for i in all_runs if i in an_info['fit_runs']], axis=0)
                        elif 'fit_task' not in an_info:
                            print("warning: fit runs not specified. guessing based on space. check code.")
                            if space == 'HCP':
                                fit_runs = [0]
                            elif space == 'fsnative':
                                fit_runs = [0,2,4]
                            tc_test[task] = np.mean([tc_runs[i] for i in all_runs if i not in fit_runs], axis=0)
                            tc_fit[task] = np.mean([tc_runs[i] for i in all_runs if i in fit_runs], axis=0)
                            
                        #tc_test[task] *= (100/tc_test[task].mean(-1))[...,np.newaxis]
                        tc_test[task] -= np.median(tc_test[task][...,self.prf_stims[task].late_iso_dict[task]], axis=-1)[...,np.newaxis]
                        #tc_fit[task] *= (100/tc_fit[task].mean(-1))[...,np.newaxis]
                        tc_fit[task] -= np.median(tc_fit[task][...,self.prf_stims[task].late_iso_dict[task]], axis=-1)[...,np.newaxis]     
                        
    
                    
                tc_full = np.concatenate(tuple([tc[task] for task in tc]), axis=0)
                tc_err = np.concatenate(tuple([tc_err[task] for task in tc_err]), axis=0)
                
                if an_info['crossvalidate'] and 'fit_task' not in an_info:
                    tc_full_test = np.concatenate(tuple([tc_test[task] for task in tc_test]), axis=0)
                    tc_full_fit = np.concatenate(tuple([tc_fit[task] for task in tc_fit]), axis=0)   
                    #timecourse reliability stats
                    vertex_info+="CV timecourse reliability stats\n"
                    vertex_info+=f"fit-test timecourses corrcoeff {np.corrcoef(tc_full_test,tc_full_fit)[0,1]:.4f}\n"
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

                    
                    if 'CVmean' in analysis or 'CVmedian' in analysis:
                        #need to combine params/predictions from folds?
                        pass

                    
                    params = np.copy(subj_res['Results'][model][internal_idx,:-1])
                    

                    preds[model] = self.prf_models[model].return_prediction(*list(params))[0]
                    
                    
                    #mdff = (preds[model]-tc_full).mean()
                    #tc_full += mdff
                    
                    np.set_printoptions(suppress=True, precision=4)
                    vertex_info+=f"{model} params: {params} \n"
                    vertex_info+=f"Rsq {model} full tc: {1-np.sum((preds[model]-tc_full)**2)/(tc_full.var(-1)*tc_full.shape[-1]):.4f}\n"
                    
                    if an_info['crossvalidate'] and 'fit_task' not in an_info:
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
                        #if model in subj_res['Processed Results'][key]:
                        for mm in subj_res['Processed Results'][key]:
                            vertex_info+=f"{key} {mm} {subj_res['Processed Results'][key][mm][index]:.8f}\n"
                    
                    vertex_info+="\n"
                    
                pl.ion()
    
                if self.click==0:
                    self.f, self.axes = pl.subplots(2,2,figsize=(18, 25),frameon=True, gridspec_kw={'width_ratios': [8, 2], 'height_ratios': [1,1]})
                    self.f.set_tight_layout(True)
                else:
                    self.cbar.remove()
                    for ax1 in self.axes:
                        for ax2 in ax1: 
                            ax2.clear()
                
                tseconds = an_info['TR']*np.arange(len(tc_full))
                cmap = cm.get_cmap('tab10')
                
                self.axes[0,0].plot(tseconds,np.zeros(len(tc_full)),linestyle='--',linewidth=0.1, color='black', zorder=0)
        
                
    
                if an_info['crossvalidate'] and 'fit_task' not in an_info and 'fit_runs' in an_info:
                    
                    #print(np.std([tc_full_test,tc_full_fit],axis=0))
                    for i,tc_run in enumerate(tc_runs):
                        if i in an_info['fit_runs']:
                            self.axes[0,0].plot(tseconds, tc_run, label=f'Run {i} (fit)', linestyle = ':', marker='^', markersize=3, color=cmap(5), linewidth=0.5, alpha=0.5)
                        else:
                            self.axes[0,0].plot(tseconds, tc_run, label=f'Run {i} (test)', linestyle = ':', marker='v', markersize=3, color=cmap(5), linewidth=0.5, alpha=0.5)
                    
                    #self.axes[0,0].plot(tseconds, tc_full_test, label='Data (test)', linestyle = ':', marker='^', markersize=3, color=cmap(4), linewidth=0.5, alpha=0.5) 
                    #self.axes[0,0].plot(tseconds, tc_full_fit, label='Data (fit)', linestyle = ':', marker='v', markersize=3, color=cmap(5), linewidth=0.5, alpha=0.5) 
                else:
                    for i,tc_run in enumerate(tc_runs):
                        
                        self.axes[0,0].plot(tseconds, tc_run, label=f'Run {i} (fit)', linestyle = ':', marker='^', markersize=3, color=cmap(5), linewidth=0.5, alpha=0.5)                    
                
                self.axes[0,0].errorbar(tseconds, tc_full, yerr=0, label='Data',  fmt = 'sk', markersize=5,  linewidth=0, zorder=1) 
                    
    
                
                for model in self.only_models: 
                    if 'Norm' in model:
                        self.axes[0,0].plot(tseconds, preds[model], linewidth=5, color=cmap(3), label=f"Norm ({subj_res['Processed Results']['RSq'][model][index]:.2f})", zorder=2)
                    elif model == 'DoG':
                        self.axes[0,0].plot(tseconds, preds[model], linewidth=4, color=cmap(2), label=f"DoG ({subj_res['Processed Results']['RSq'][model][index]:.2f})", zorder=3)
                    elif model == 'CSS':
                        self.axes[0,0].plot(tseconds, preds[model], linewidth=3, color=cmap(1), label=f"CSS ({subj_res['Processed Results']['RSq'][model][index]:.2f})", zorder=4)
                    elif model == 'Gauss':
                        self.axes[0,0].plot(tseconds, preds[model], linewidth=2, color=cmap(0), label=f"Gauss ({subj_res['Processed Results']['RSq'][model][index]:.2f})", zorder=5)
                        
                
                fillin_surround = False
                fillin_late_iso_dict = True

                if fillin_late_iso_dict:
                    min_lid = np.zeros_like(tseconds)
                    max_lid = np.zeros_like(tseconds)
                    min_lid[self.prf_stim.late_iso_dict['periods']] = self.axes[0,0].get_ylim()[0]
                    max_lid[self.prf_stim.late_iso_dict['periods']] = self.axes[0,0].get_ylim()[1]

                    self.axes[0,0].fill_between(tseconds,min_lid,max_lid,label='Late iso dict', alpha=0.2, color='gray')
                
                if fillin_surround:
                
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
                self.axes[0,0].set_title(f"Sj: {subj}, Vx: {index} timecourse")
                
                if prfs['Norm_abcd'][0].min() < 0:
                    im = self.axes[0,1].imshow(prfs['Norm_abcd'][0], cmap='RdBu_r', vmin=prfs['Norm_abcd'][0].min(), vmax=-prfs['Norm_abcd'][0].min())
                else:
                    im = self.axes[0,1].imshow(prfs['Norm_abcd'][0], cmap='RdBu_r', vmax=prfs['Norm_abcd'][0].max(), vmin=-prfs['Norm_abcd'][0].max())
                    
                self.cbar = colorbar(im)
                self.axes[0,1].set_title("pRF")
                
                #self.cbar = self.f.colorbar(im, ax=self.axes[0,1], use_gridspec=True)
                
                self.axes[0,1].axis('on')            
                self.axes[1,0].axis('off')
                self.axes[1,1].axis('off')
                self.axes[1,0].text(0,1,vertex_info, fontsize=10)
                
                
                self.click+=1
          
            return
        

        if clickerfun == 'groups':
            pickerfun = groups_clicker_function
        else:
            pickerfun = clicker_function


        
        if space_names == 'all':
            spaces = [item for item in self.main_dict.items()]
        else:
            spaces = [item for item in self.main_dict.items() if item[0] in space_names] 

        for space, space_res in spaces:
            plotted_rois = dd(lambda:False)
            plotted_stats = dd(lambda:False)
            alpha = dd(lambda:dd(lambda:dd(dict)))
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
                    
                    pycortex_subj = subj.split('_')[0]
                    
                    if pycortex_subj not in cortex.db.subjects and os.path.exists(opj(self.fs_dir,pycortex_subj)):
                        print("subject not present in pycortex database. attempting to import...")
                        cortex.freesurfer.import_subj(pycortex_subj, freesurfer_subject_dir=self.fs_dir, 
                              whitematter_surf='smoothwm')
                    elif (subj.isdecimal() or '999999' in subj) and space == 'HCP':
                        pycortex_subj = '999999'
                    elif space == 'fsaverage':
                        pycortex_subj = 'fsaverage'

                    p_r = subj_res['Processed Results']
                    models = p_r['RSq'].keys()

                    tc_stats = p_r['Timecourse Stats']
                    if space != 'fsaverage':
                        
                        mask = subj_res['mask']
                    else:
                        
                        mask = np.ones_like(p_r['RSq'][list(models)[0]])

         
                    if roi_borders_name != None:
                        roi_borders = self.idx_rois_borders[subj][roi_borders_name]
                    else:
                        roi_borders = None
                    
                    
                    for model in self.only_models+['all']:
                        curr_alpha = np.copy(p_r['Alpha'][model])
                    
                        if all(roi in self.idx_rois[subj] for roi in rois):
    
                            alpha[analysis][subj][model] = (roi_mask(np.concatenate(tuple([self.idx_rois[subj][r] for r in rois])), curr_alpha)) 
       
                        else:
                            #if ROI != defined
                            #if Brain use all available vertices
                            if rois == 'Brain':
                                alpha[analysis][subj][model] = curr_alpha
                            elif rois == 'combined':
                                alpha[analysis][subj][model] = (roi_mask(np.concatenate(tuple([self.idx_rois[subj][r] for r in rois if ('combined' not in r and 'Brain' not in r and r in self.idx_rois[subj])])), curr_alpha))    
                            elif rois == 'all_custom':
                                alpha[analysis][subj][model] = (roi_mask(np.concatenate(tuple([self.idx_rois[subj][r] for r in self.idx_rois[subj] if 'custom' in r])), curr_alpha))    
                            elif space == 'fsaverage' and all(roi in self.idx_rois['fsaverage'] for roi in rois):
                                alpha[analysis][subj][model] = (roi_mask(np.concatenate(tuple([self.idx_rois['fsaverage'][r] for r in rois])), curr_alpha))
                            else:
                                #, otherwise none
                                print("undefined ROI")
                                alpha[analysis][subj][model] = np.zeros_like(curr_alpha).astype('bool')    
                    

                    #output freesurefer-format polar angle maps to draw custom ROIs in freeview    
                    if self.output_freesurfer_maps:
                        
                        fs_sj = pycortex_subj
                        
                        lh_c = read_morph_data(opj(self.fs_dir, fs_sj+'/surf/lh.curv'))
        
                        polar_freeview = np.copy(p_r['Polar Angle']['Norm_abcd'])#np.mean(polar, axis=-1)
                        ecc_freeview = np.copy(p_r['Eccentricity']['Norm_abcd'])#np.mean(ecc, axis=-1)
                        alpha_freeview = np.copy(alpha[analysis][subj]['Norm_abcd'])
                        
                        # if space!='fsaverage' and hasattr(self, 'tc_min'):
                        #     if subj in self.tc_min:
                        #         alpha_freeview = p_r['RSq']['Norm_abcd']* (tc_stats['Mean']>self.tc_min[subj])# rsq.max(-1) * (tc_stats['Mean']>self.tc_min[subj]) * (rsq.min(-1)>0)
                        # else:
                        #     alpha_freeview = p_r['RSq']['Norm_abcd']
                            
                        #print(rsq_thresh)
                        #print(alpha_freeview)
                        #print(np.sum(alpha_freeview<=rsq_thresh))
                            
                        polar_freeview[alpha_freeview<=rsq_thresh] = -10
                        ecc_freeview[alpha_freeview<=rsq_thresh] = -10
                        
                        #print(np.sum(polar_freeview == -10))
        
                        write_morph_data(opj(self.fs_dir, f"{fs_sj}/surf/lh.polar_norm_maxecc{self.ecc_max:.1f}_minrsq{rsq_thresh:.2f}")
                                                               ,polar_freeview[:lh_c.shape[0]])
                        write_morph_data(opj(self.fs_dir, f"{fs_sj}/surf/rh.polar_norm_maxecc{self.ecc_max:.1f}_minrsq{rsq_thresh:.2f}")
                                                               ,polar_freeview[lh_c.shape[0]:])
                        write_morph_data(opj(self.fs_dir, f"{fs_sj}/surf/lh.ecc_norm_maxecc{self.ecc_max:.1f}_minrsq{rsq_thresh:.2f}")
                                                               ,ecc_freeview[:lh_c.shape[0]])
                        write_morph_data(opj(self.fs_dir, f"{fs_sj}/surf/rh.ecc_norm_maxecc{self.ecc_max:.1f}_minrsq{rsq_thresh:.2f}")
                                                               ,ecc_freeview[lh_c.shape[0]:])
                        
                        

                        
                    ##START PYCORTEX VISUALIZATIONS                            
                    #data quality/stats cortex visualization 
                    if self.plot_stats_cortex: # and not plotted_stats[subj] :
                        ds_stats = dict()

                        for stat in tc_stats.keys():

                            if 'Mean' in stat:
                                vmin_stat = np.nanquantile(tc_stats['Mean'],0.1)
                                vmax_stat = np.nanquantile(tc_stats['Mean'],0.9)
                            elif 'TSNR' in stat:
                                vmin_stat = np.nanquantile(tc_stats['TSNR'][alpha[analysis][subj]['all']>rsq_thresh],0.1)
                                vmax_stat = np.nanquantile(tc_stats['TSNR'][alpha[analysis][subj]['all']>rsq_thresh],0.9)
                            elif 'Variance' in stat and 'psc' not in stat:
                                vmin_stat = np.nanquantile(tc_stats['Variance'][alpha[analysis][subj]['all']>rsq_thresh],0.1)
                                vmax_stat = np.nanquantile(tc_stats['Variance'][alpha[analysis][subj]['all']>rsq_thresh],0.9)
                            else:
                                vmin_stat = np.nanquantile(tc_stats[stat][alpha[analysis][subj]['all']>rsq_thresh],0.1)
                                vmax_stat = np.nanquantile(tc_stats[stat][alpha[analysis][subj]['all']>rsq_thresh],0.9)

                            ds_stats[f"{subj} {stat}"] = Vertex2D_fix(tc_stats[stat], alpha[analysis][subj]['all'], #np.ones_like(alpha[analysis][subj]['all']),
                                                            vmin=vmin_stat, 
                                                            vmax=vmax_stat,
                                                            vmin2=rsq_thresh, vmax2=rsq_max_opacity, subject=pycortex_subj, cmap=pycortex_cmap, roi_borders=roi_borders)
                            
                            fig = simple_colorbar(vmin=vmin_stat, 
                                                    vmax=vmax_stat,
                                                cmap_name=pycortex_cmap, ori='horizontal', param_name=f'{subj} {stat}')
                                
                            if self.pycortex_image_path != None and save_colorbars:
                                fig.savefig(f"{self.pycortex_image_path}/{subj}_{stat}_cbar.pdf", dpi=600, bbox_inches='tight', transparent=True)    

        
                        self.js_handle_dict[space][analysis][subj]['Timecourse Stats'] = cortex.webgl.show(ds_stats, pickerfun=pickerfun,  overlays_visible=[], labels_visible=[]) 
        
                        ##plotted_stats[subj] = True
                    
                    if self.plot_rois_cortex and not plotted_rois[subj]:
                        
                        plot_single_rois = False
                        use_alpha_rois = False
                        plot_my_vx = False
                        
                        ds_rois = {}
                        data = np.zeros_like(mask).astype('int')
                        custom_rois_data = np.zeros_like(mask).astype('int')
                        hcp_rois_data = np.zeros_like(mask).astype('int')
                        glasser_rois_data = np.zeros_like(mask)
                        

                        if plot_my_vx:
                            myvx = np.zeros_like(mask).astype('int')
                            myvx[87020] = 1
                            myvx[45156] = 1
                            ds_rois['my vx'] = Vertex2D_fix(myvx, myvx.astype('bool'), subject=pycortex_subj, cmap=pycortex_cmap, vmin=0, vmax=data.max(), vmin2=0, vmax2=1, roi_borders=roi_borders)
            
                        for i, roi in enumerate([r for r in self.idx_rois[subj] if 'custom' in r and 'Pole' not in r]):        
                            roi_data = np.zeros_like(mask)
                            roi_data[self.idx_rois[subj][roi]] = 1
                            custom_rois_data[self.idx_rois[subj][roi]] = i+1
                            if plot_single_rois:
                                ds_rois[roi] = Vertex2D_fix(roi_data, roi_data.astype('bool'), subject=pycortex_subj, cmap=pycortex_cmap, vmin=0, vmax=roi_data.max(), vmin2=0, vmax2=1, roi_borders=roi_borders)
                            
                        for i, roi in enumerate([r for r in self.idx_rois[subj] if 'custom' not in r and 'visual' not in r and 'HCP' not in r and 'glasser' not in r]):        
                            roi_data = np.zeros_like(mask)
                            roi_data[self.idx_rois[subj][roi]] = 1 
                            data[self.idx_rois[subj][roi]] = i+1
                            if plot_single_rois:                                  
                                ds_rois[roi] = Vertex2D_fix(roi_data, roi_data.astype('bool'), subject=pycortex_subj, cmap=pycortex_cmap, vmin=0, vmax=roi_data.max(), vmin2=0, vmax2=1, roi_borders=roi_borders)
                            
                        for i, roi in enumerate([r for r in self.idx_rois[subj] if 'HCPQ1Q6.' in r]):        
                            roi_data = np.zeros_like(mask)
                            roi_data[self.idx_rois[subj][roi]] = 1
                            hcp_rois_data[self.idx_rois[subj][roi]] = i+1                              
                            ds_rois[roi] = Vertex2D_fix(roi_data, roi_data.astype('bool'), subject=pycortex_subj, cmap=pycortex_cmap, vmin=0, vmax=roi_data.max(), vmin2=0, vmax2=1, roi_borders=roi_borders)

                        cmap_values = np.linspace(0.9, 0.0, len([r for r in self.idx_rois[subj] if 'glasser' in r]))
                        #print(cmap_values)

                        for i, roi in enumerate([r for r in self.idx_rois[subj] if 'glasser' in r]):        
                            roi_data = np.zeros_like(mask)
                            roi_data[self.idx_rois[subj][roi]] = 1
                            glasser_rois_data[self.idx_rois[subj][roi]] = cmap_values[i]
                            #print(glasser_rois_data.sum())
                            if plot_single_rois:                                                             
                                ds_rois[roi] = Vertex2D_fix(roi_data, roi_data.astype('bool'), subject=pycortex_subj, cmap=pycortex_cmap, vmin=0, vmax=roi_data.max(), vmin2=0, vmax2=1, roi_borders=roi_borders)
        
                        

                        if use_alpha_rois:
                            alpha_rois = alpha[analysis][subj][self.only_models[0]]
                            vmin2_rois = rsq_thresh
                            vmax2_rois = rsq_max_opacity
                        else:
                            vmin2_rois = 0
                            vmax2_rois = 1

                        if data.sum()>0:
                            if not use_alpha_rois:
                                alpha_rois = data.astype('bool')
                            ds_rois['Wang2015Atlas'] = Vertex2D_fix(data, alpha_rois, subject=pycortex_subj, cmap=pycortex_cmap, vmin=0, vmax=data.max(), vmin2=vmin2_rois, vmax2=vmax2_rois, roi_borders=roi_borders)
                        if custom_rois_data.sum()>0:
                            if not use_alpha_rois:
                                alpha_rois = custom_rois_data.astype('bool')
                            ds_rois['Custom ROIs'] = Vertex2D_fix(custom_rois_data, alpha_rois, subject=pycortex_subj, cmap=pycortex_cmap, vmin=0, vmax=custom_rois_data.max(), vmin2=vmin2_rois, vmax2=vmax2_rois, roi_borders=roi_borders)
                        if hcp_rois_data.sum()>0:
                            if not use_alpha_rois:
                                alpha_rois = hcp_rois_data.astype('bool')
                            ds_rois['HCP ROIs'] = Vertex2D_fix(hcp_rois_data, alpha_rois, subject=pycortex_subj, cmap=pycortex_cmap, vmin=0, vmax=hcp_rois_data.max(), vmin2=vmin2_rois, vmax2=vmax2_rois, roi_borders=roi_borders)
                        if glasser_rois_data.sum()>0:
                            if not use_alpha_rois:
                                alpha_rois = glasser_rois_data.astype('bool')
                            ds_rois['Glasser ROIs'] = Vertex2D_fix(glasser_rois_data, alpha_rois, subject=pycortex_subj, cmap=pycortex_cmap, vmin=0, vmax=glasser_rois_data.max(), vmin2=vmin2_rois, vmax2=vmax2_rois, roi_borders=roi_borders) 
                                                
                        self.js_handle_dict[space][analysis][subj]['rois'] = cortex.webgl.show(ds_rois, pickerfun=pickerfun,  overlays_visible=[], labels_visible=[]) 
        
                        plotted_rois[subj] = True 
                                                


                    if self.plot_diffs:
                        ds_diffs = dict()
                        diff_params = [item for item in p_r.items() if np.any([s in item[0] for s in param_diffs]) and ('-' in item[0] or '/' in item[0])]

                        for param, param_res in diff_params:
                            if 'Timecourse Stats' not in param:
                                models = [item for item in param_res.items() if item[0] in self.only_models]
                            else:
                                models = [item for item in param_res.items()]

                            groupcomp = param.split(' ')[-1].replace('/','-')
                            for model, model_res in models:

                                if model in p_r[f'Mean Masked RSq {groupcomp}']:
                                    this_alpha = p_r[f'Mean Masked RSq {groupcomp}'][model]

                                # elif 'Noise' in model:
                                #     if 'ses' in subj and space == 'fsnative':
                                #         gen_subj = subj.split('_')[0]
                            
                                #         subj_pla = [s for s in self.groups_dict['placebo'] if gen_subj in s][0]
                                #         this_alpha = analysis_res[subj_pla]['Processed Results']['Noise Ceiling']['Noise Ceiling (CC)']
                                #         this_alpha[this_alpha<0] = 0
                                #     else:
                                #         this_alpha = np.ones_like(p_r[param][model])
                                    
                                else:
                                    this_alpha = np.ones_like(p_r[param][model])
                                

                                #if '-' in param:
                                vmaxdiff = np.max((np.abs(np.nanquantile(p_r[param][model][this_alpha>rsq_thresh],0.1)),np.abs(np.nanquantile(p_r[param][model][this_alpha>rsq_thresh],0.9))))
                                vmindiff = -vmaxdiff
                                # else:
                                #     vmindiff = np.nanquantile(p_r[param][model][this_alpha>rsq_thresh],0.1)
                                #     if vmindiff<1:
                                #         vmaxdiff = 1 + (1-vmindiff)
                                #     else:
                                #         vmaxdiff = np.nanquantile(p_r[param][model][this_alpha>rsq_thresh],0.9)
                                #     print(f'{param} {vmindiff} {vmaxdiff}')

                                ds_diffs[f'{subj} {param} {model}'] = Vertex2D_fix(p_r[param][model], 
                                                                      this_alpha, 
                                                                      subject=pycortex_subj, 
                                                            vmin=vmindiff, 
                                                            vmax=vmaxdiff,
                                                                      vmin2=rsq_thresh, vmax2=rsq_max_opacity, cmap=pycortex_cmap, 
                                                                      roi_borders=roi_borders)

                                fig = simple_colorbar(vmin=vmindiff, 
                                                vmax=vmaxdiff,
                                                cmap_name=pycortex_cmap, ori='horizontal', param_name=f"{model}_{param}")
                                
                                if self.pycortex_image_path != None and save_colorbars:
                                    fig.savefig(f"{self.pycortex_image_path}/{model}_{param}_cbar.pdf", dpi=600, bbox_inches='tight', transparent=True)

                            
                        self.js_handle_dict[space][analysis][subj]['Diffs'] = cortex.webgl.show(ds_diffs, pickerfun=pickerfun,  overlays_visible=[], labels_visible=[])  



               
        
                    if self.plot_rsq_cortex:              
                        ds_rsq = dict()
                    
                        
                        if len(self.only_models)>1:
                            best_model = np.argmax([p_r['RSq'][model] for model in self.only_models],axis=0)

                            ds_rsq[f'{subj} Best model'] = Vertex2D_fix(best_model, alpha[analysis][subj]['all'], subject=pycortex_subj, vmin=0, vmax=len(self.only_models),
                                                                      vmin2=rsq_thresh, vmax2=rsq_max_opacity, cmap='BROYG', roi_borders=roi_borders)


                        for model in self.only_models:

                            ds_rsq[f"{subj} {model} rsq"] = Vertex2D_fix(p_r['RSq'][model], 
                                                            np.ones_like(p_r['RSq'][model]), 
                                                            subject=pycortex_subj, 
                                                            vmin=rsq_thresh,#0,#np.nanquantile(p_r['RSq'][model][alpha[analysis][subj][model]>rsq_thresh],0.1), 
                                                            vmax=rsq_max_opacity,#0.15,#np.nanquantile(p_r['RSq'][model][alpha[analysis][subj][model]>rsq_thresh],0.9),
                                                            vmin2=rsq_thresh, vmax2=rsq_max_opacity, cmap=pycortex_cmap, roi_borders=roi_borders)
                            
                            fig = simple_colorbar(vmin=rsq_thresh,#0,#np.nanquantile(p_r['RSq'][model][alpha[analysis][subj][model]>rsq_thresh],0.1), 
                                            vmax=rsq_max_opacity,#0.15,#np.nanquantile(p_r['RSq'][model][alpha[analysis][subj][model]>rsq_thresh],0.9),
                                            cmap_name=pycortex_cmap, ori='horizontal', param_name='$R^2$')
                            
                            if self.pycortex_image_path != None and save_colorbars:
                                fig.savefig(f"{self.pycortex_image_path}/{model}_rsq_cbar.pdf", dpi=600, bbox_inches='tight', transparent=True)
                            
                        # if 'Noise Ceiling' in p_r:  ## sorted([p for p in paths if os.path.isdir(p) and np.any(['Noise Ceiling' in p for s in groups_dict['5mg']])])

                        #     for nc_type in p_r['Noise Ceiling']:

                        #         ds_rsq[f'{subj} {nc_type}'] = Vertex2D_fix(p_r['Noise Ceiling'][nc_type],
                        #                                         np.ones_like(p_r['Noise Ceiling'][nc_type]), subject=pycortex_subj, 
                        #                                         vmin=0,#np.nanquantile(p_r['Noise Ceiling'][nc_type][alpha[analysis][subj][model]>rsq_thresh],0.1), 
                        #                                         vmax=0.5,#np.nanquantile(p_r['Noise Ceiling'][nc_type][alpha[analysis][subj][model]>rsq_thresh],0.9),
                        #                                         vmin2=rsq_thresh, vmax2=rsq_max_opacity, cmap=pycortex_cmap, roi_borders=roi_borders)   
                                    
                        #         fig = simple_colorbar(vmin=0,#np.nanquantile(p_r['Noise Ceiling'][nc_type][alpha[analysis][subj][model]>rsq_thresh],0.1), 
                        #                                 vmax=0.5,#np.nanquantile(p_r['Noise Ceiling'][nc_type][alpha[analysis][subj][model]>rsq_thresh],0.9),
                        #                             cmap_name=pycortex_cmap, ori='horizontal', param_name=f'{subj} {nc_type}')
                                    
                        #         if self.pycortex_image_path != None and save_colorbars:
                        #             fig.savefig(f"{self.pycortex_image_path}/nc_cbar.pdf", dpi=600, bbox_inches='tight', transparent=True)                                
                              
    
                            
                        if '#Subjects with CVRSq>0' in p_r:
                            for model in self.only_models:
                                ds_rsq[f"#subj cvrsq>0 {subj} {model}"] = Vertex2D_fix(p_r['#Subjects with CVRSq>0'][model],
                                                               np.ones_like(p_r['#Subjects with CVRSq>0'][model]), subject=pycortex_subj, 
                                                            vmin=0,#np.nanquantile(p_r['Noise Ceiling'][nc_type][alpha[analysis][subj][model]>rsq_thresh],0.1), 
                                                            vmax=100,#np.nanquantile(p_r['Noise Ceiling'][nc_type][alpha[analysis][subj][model]>rsq_thresh],0.9),
                                                            vmin2=rsq_thresh, vmax2=rsq_max_opacity, cmap=pycortex_cmap, roi_borders=roi_borders)   
                                
                                fig = simple_colorbar(vmin=0,#np.nanquantile(p_r['RSq'][model][alpha[analysis][subj][model]>rsq_thresh],0.1), 
                                                vmax=100,#np.nanquantile(p_r['RSq'][model][alpha[analysis][subj][model]>rsq_thresh],0.9),
                                                cmap_name=pycortex_cmap, ori='horizontal', param_name='#Subjects with cv$R^2$>0')
                                
                                if self.pycortex_image_path != None and save_colorbars:
                                    fig.savefig(f"{self.pycortex_image_path}/{model}_nsubj_cbar.pdf", dpi=600, bbox_inches='tight', transparent=True)     

                        mdiff=False           

                        if 'CSS' in models and p_r['RSq']['CSS'].sum()>0:
                            mdiff=True
                            ds_rsq[f'{subj} CSS - Gauss'] = Vertex2D_fix(p_r['RSq']['CSS']-p_r['RSq']['Gauss'], alpha[analysis][subj]['all'], subject=pycortex_subj,
                                                                      vmin=-0.1, vmax=0.1, vmin2=rsq_thresh, vmax2=rsq_max_opacity, cmap=pycortex_cmap, roi_borders=roi_borders)  
                            
                        if 'DoG' in models and p_r['RSq']['DoG'].sum()>0:
                            mdiff=True
                            ds_rsq[f'{subj} DoG - Gauss'] = Vertex2D_fix(p_r['RSq']['DoG']-p_r['RSq']['Gauss'], alpha[analysis][subj]['all'], subject=pycortex_subj,
                                                                  vmin=-0.1, vmax=0.1, vmin2=rsq_thresh, vmax2=rsq_max_opacity, cmap=pycortex_cmap, roi_borders=roi_borders)
                        
                        if 'Norm_abcd' in self.only_models and 'Gauss' in self.only_models:
                            mdiff=True
                            ds_rsq[f'{subj} Norm_abcd - Gauss'] = Vertex2D_fix(p_r['RSq']['Norm_abcd']-p_r['RSq']['Gauss'], alpha[analysis][subj]['all'], subject=pycortex_subj,
                                                                      vmin=-0.1, vmax=0.1, vmin2=rsq_thresh, vmax2=rsq_max_opacity, cmap=pycortex_cmap, roi_borders=roi_borders)

                            if 'CSS' in self.only_models and 'DoG' in self.only_models:
                                mdiff=True
                                ds_rsq[f'{subj} Norm_abcd - DoG'] = Vertex2D_fix(p_r['RSq']['Norm_abcd']-p_r['RSq']['DoG'], alpha[analysis][subj]['all'], subject=pycortex_subj,
                                                                      vmin=-0.1, vmax=0.1, vmin2=rsq_thresh, vmax2=rsq_max_opacity, cmap=pycortex_cmap, roi_borders=roi_borders)

                                ds_rsq[f'{subj} Norm_abcd - CSS'] = Vertex2D_fix(p_r['RSq']['Norm_abcd']-p_r['RSq']['CSS'], alpha[analysis][subj]['all'], subject=pycortex_subj, 
                                                                      vmin=-0.1, vmax=0.1, vmin2=rsq_thresh, vmax2=rsq_max_opacity, cmap=pycortex_cmap, roi_borders=roi_borders)
 

                        for model in [model for model in self.only_models if 'Norm' in model and 'Norm_abcd' != model]:
                            mdiff=True
                            ds_rsq[f'{subj} Norm_abcd - {model}'] = Vertex2D_fix(p_r['RSq'][model]-p_r['RSq']['Norm_abcd'], alpha[analysis][subj]['all'], subject=pycortex_subj,
                                                                      vmin=-0.1, vmax=0.1, vmin2=rsq_thresh, vmax2=rsq_max_opacity, cmap=pycortex_cmap, roi_borders=roi_borders)

                        if mdiff:    
                            fig = simple_colorbar(vmin=-0.1, 
                                            vmax=0.1,
                                            cmap_name=pycortex_cmap, ori='horizontal', param_name='$R^2$ difference')
                            
                            if self.pycortex_image_path != None and save_colorbars:
                                fig.savefig(f"{self.pycortex_image_path}/rsqdiff_cbar.pdf", dpi=600, bbox_inches='tight', transparent=True)   

                        if 'Processed Results' in self.main_dict['T1w'][analysis][subj] and self.compare_volume_surface:
                            
                            ds_rsq_comp = dict()
                            volume_rsq = self.main_dict['T1w'][analysis][subj]['Processed Results']['RSq']['Norm']
                            ref_img = nb.load(self.main_dict['T1w'][analysis][subj]['Results']['ref_img_path'])
                            
                            #rsq_img = nb.Nifti1Image(volume_rsq, ref_img.affine, ref_img.header)

                            xfm_trans = cortex.xfm.Transform(np.identity(4), ref_img)
                            xfm_trans.save(subj, 'func_space_transform')
                            
                            ds_rsq_comp[f'{subj} Norm_abcd CV rsq (volume fit)'] = cortex.Volume2D(volume_rsq.T, volume_rsq.T, subj, 'func_space_transform',
                                                                      vmin=rsq_thresh, vmax=0.6, vmin2=0.05, vmax2=rsq_thresh, cmap=pycortex_cmap)
                            ds_rsq_comp[f'{subj} Norm_abcd CV rsq (surface fit)'] = Vertex2D_fix(p_r['RSq']['Norm_abcd'], p_r['RSq']['Norm_abcd'], subject=pycortex_subj,
                                                                      vmin=rsq_thresh, vmax=0.6, vmin2=0.05, vmax2=rsq_thresh, cmap=pycortex_cmap, roi_borders=roi_borders)
                            self.js_handle_dict[space][analysis][subj]['rsq_comp'] = cortex.webgl.show(ds_rsq_comp, pickerfun=pickerfun,  overlays_visible=[], labels_visible=[]) 
                        
                        
                        self.js_handle_dict[space][analysis][subj]['rsq'] = cortex.webgl.show(ds_rsq, pickerfun=pickerfun,  overlays_visible=[], labels_visible=[])  
                        
                    if self.plot_ecc_cortex:
                        ds_ecc = dict()
                        
                        for model in self.only_models:
                            #print("Note: eccentricity plot has vmax set at 0.95 quantile")
                            ds_ecc[f"{subj} {model} Eccentricity"] = Vertex2D_fix(p_r['Eccentricity'][model], alpha[analysis][subj][model], subject=pycortex_subj, 
                                                            vmin=self.ecc_min,#np.nanquantile(p_r['Eccentricity'][model][alpha[analysis][subj][model]>rsq_thresh],0.1), 
                                                            vmax=0.75*self.ecc_max,#np.nanquantile(p_r['Eccentricity'][model][alpha[analysis][subj][model]>rsq_thresh],0.9), 
                                                            vmin2=rsq_thresh, vmax2=rsq_max_opacity, cmap=pycortex_cmap, roi_borders=roi_borders) #np.nanquantile(alpha[analysis][subj][model][alpha[analysis][subj][model]>rsq_thresh],0.9
        
                            fig = simple_colorbar(vmin=self.ecc_min,#np.nanquantile(p_r['Eccentricity'][model][alpha[analysis][subj][model]>rsq_thresh],0.1), 
                                            vmax=0.75*self.ecc_max,#np.nanquantile(p_r['Eccentricity'][model][alpha[analysis][subj][model]>rsq_thresh],0.9),
                                            cmap_name=pycortex_cmap, ori='polar', param_name='Eccentricity (°)')
                            
                            if self.pycortex_image_path != None and save_colorbars:
                                fig.savefig(f"{self.pycortex_image_path}/{model}_ecc_cbar.pdf", dpi=600, bbox_inches='tight', transparent=True)
                        
                        self.js_handle_dict[space][analysis][subj]['ecc'] = cortex.webgl.show(ds_ecc, pickerfun=pickerfun,  overlays_visible=[], labels_visible=[]) 
        
                    if self.plot_polar_cortex:
                        ds_polar = dict()
                        
                        for model in self.only_models:
                            #print(p_r['Polar Angle'][model])
                            ds_polar[f"{subj} {model} polar angle HSV2"] = Vertex2D_fix(p_r['Polar Angle'][model], alpha[analysis][subj][model], subject=pycortex_subj, 
                                                              vmin=-3.1415, vmax=3.1415,
                                                              vmin2=rsq_thresh, vmax2=rsq_max_opacity, cmap='hsvx2', roi_borders=roi_borders)
                            
                            fig = simple_colorbar(vmin=-3.1415, vmax=3.1415,
                                            cmap_name='hsvx2', ori='polar', param_name='Polar Angle (°)')                                     
                            
                            if self.pycortex_image_path != None and save_colorbars:
                                fig.savefig(f"{self.pycortex_image_path}/{model}_polar_hsvx2cbar.pdf", dpi=600, bbox_inches='tight', transparent=True)

                            ds_polar[f"{subj} {model} polar angle HSV1"] = Vertex2D_fix(p_r['Polar Angle'][model], alpha[analysis][subj][model], subject=pycortex_subj, 
                                                                        vmin=-3.1415, vmax=3.1415,
                                                               vmin2=rsq_thresh, vmax2=rsq_max_opacity, cmap='hsv', roi_borders=roi_borders)
                            
                            fig = simple_colorbar(vmin=-3.1415, vmax=3.1415,
                                            cmap_name='hsv', ori='polar', param_name='Polar Angle (°)')                                     
                            
                            if self.pycortex_image_path != None and save_colorbars:
                                fig.savefig(f"{self.pycortex_image_path}/{model}_polar_hsvcbar.pdf", dpi=600, bbox_inches='tight', transparent=True)                            
                        
                        if 'Processed Results' in self.main_dict['T1w'][analysis][subj] and self.compare_volume_surface:
                            ds_polar_comp = dict()
                            
                            volume_rsq = self.main_dict['T1w'][analysis][subj]['Processed Results']['RSq']['Norm_abcd']
                            volume_polar = self.main_dict['T1w'][analysis][subj]['Processed Results']['Polar Angle']['Norm_abcd']
                            ref_img = nb.load(self.main_dict['T1w'][analysis][subj]['Results']['ref_img_path'])                                

                            xfm_trans = cortex.xfm.Transform(np.identity(4), ref_img)
                            xfm_trans.save(subj, 'func_space_transform')
                            
                            ds_polar_comp[f'{subj} Norm_abcd CV polar (volume fit)'] = cortex.Volume2D(volume_polar.T, volume_rsq.T, subj, 'func_space_transform',
                                                                      vmin2=0.05, vmax2=rsq_thresh, cmap='hsvx2')
                            ds_polar_comp[f'{subj} Norm_abcd CV polar (surface fit)'] = Vertex2D_fix(p_r['Polar Angle']['Norm_abcd'], p_r['RSq']['Norm_abcd'], subject=pycortex_subj,
                                                                      vmin2=0.05, vmax2=rsq_thresh, cmap='hsvx2', roi_borders=roi_borders)
                            self.js_handle_dict[space][analysis][subj]['polar_comp'] = cortex.webgl.show(ds_polar_comp, pickerfun=pickerfun,  overlays_visible=[], labels_visible=[]) 

                        
                        self.js_handle_dict[space][analysis][subj]['polar'] = cortex.webgl.show(ds_polar, pickerfun=pickerfun,  overlays_visible=[], labels_visible=[]) 
        
                    if self.plot_size_cortex:
                        ds_size = dict()
                        
                        for model in self.only_models:

                            ds_size[f"{subj} {model} fwhmax"] = Vertex2D_fix(p_r['Size (fwhmax)'][model], alpha[analysis][subj][model], subject=pycortex_subj, 
                                                             vmin=np.nanquantile(p_r['Size (fwhmax)'][model][alpha[analysis][subj][model]>rsq_thresh],0.01), 
                                                             vmax=np.nanquantile(p_r['Size (fwhmax)'][model][alpha[analysis][subj][model]>rsq_thresh],0.99), 
                                                             vmin2=rsq_thresh, vmax2=rsq_max_opacity, cmap=pycortex_cmap, roi_borders=roi_borders)
                            
                                
                            ds_size[f"{subj} {model} sigma_1"] = Vertex2D_fix(p_r['Size (sigma_1)'][model], alpha[analysis][subj][model], subject=pycortex_subj, 
                                                             vmin=1.1,#np.nanquantile(p_r['Size (sigma_1)'][model][alpha[analysis][subj][model]>rsq_thresh],0.01), 
                                                             vmax=4.4,#np.nanquantile(p_r['Size (sigma_1)'][model][alpha[analysis][subj][model]>rsq_thresh],0.99), 
                                                             vmin2=rsq_thresh, vmax2=rsq_max_opacity, cmap=pycortex_cmap, roi_borders=roi_borders)
                            
                            fig = simple_colorbar(vmin=1.1,#np.nanquantile(p_r['Size (sigma_1)'][model][alpha[analysis][subj][model]>rsq_thresh],0.025), 
                                            vmax=4.4,#np.nanquantile(p_r['Size (sigma_1)'][model][alpha[analysis][subj][model]>rsq_thresh],0.9),
                                            cmap_name=pycortex_cmap, ori='horizontal', param_name='$\sigma_1$ (°)')
                            
                            if self.pycortex_image_path != None and save_colorbars:
                                fig.savefig(f"{self.pycortex_image_path}/{model}_sigma1_cbar.pdf", dpi=600, bbox_inches='tight', transparent=True)                            
                            
                        self.js_handle_dict[space][analysis][subj]['size'] = cortex.webgl.show(ds_size, pickerfun=pickerfun,  overlays_visible=[], labels_visible=[]) 
        
        
                    if self.plot_amp_cortex:
                        ds_amp = dict()
                        
                        for model in self.only_models:
                            ds_amp[f"{subj} {model} Amplitude"] = Vertex2D_fix(p_r['Amplitude'][model], alpha[analysis][subj][model], subject=pycortex_subj, 
                                                            vmin=np.nanquantile(p_r['Amplitude'][model][alpha[analysis][subj][model]>rsq_thresh],0.1), 
                                                            vmax=np.nanquantile(p_r['Amplitude'][model][alpha[analysis][subj][model]>rsq_thresh],0.9), vmin2=rsq_thresh, vmax2=rsq_max_opacity, cmap=pycortex_cmap, roi_borders=roi_borders)

                            fig = simple_colorbar(vmin=np.nanquantile(p_r['Amplitude'][model][alpha[analysis][subj][model]>rsq_thresh],0.1), 
                                            vmax=np.nanquantile(p_r['Amplitude'][model][alpha[analysis][subj][model]>rsq_thresh],0.9),
                                            cmap_name=pycortex_cmap, ori='horizontal', param_name='Amplitude')
                            
                            if self.pycortex_image_path != None and save_colorbars:
                                fig.savefig(f"{self.pycortex_image_path}/{model}_amplitude_cbar.pdf", dpi=600, bbox_inches='tight', transparent=True)  
                            
                            if model == 'DoG' or 'Norm' in model:
                                ds_amp[f"{subj} {model} Surround Amplitude"] = Vertex2D_fix(p_r['Surround Amplitude'][model], alpha[analysis][subj][model], subject=pycortex_subj, 
                                                            vmin=np.nanquantile(p_r['Surround Amplitude'][model][alpha[analysis][subj][model]>rsq_thresh],0.1), 
                                                            vmax=np.nanquantile(p_r['Surround Amplitude'][model][alpha[analysis][subj][model]>rsq_thresh],0.9), vmin2=rsq_thresh, vmax2=rsq_max_opacity, cmap=pycortex_cmap, roi_borders=roi_borders)       
        
                                fig = simple_colorbar(vmin=np.nanquantile(p_r['Surround Amplitude'][model][alpha[analysis][subj][model]>rsq_thresh],0.1), 
                                                vmax=np.nanquantile(p_r['Surround Amplitude'][model][alpha[analysis][subj][model]>rsq_thresh],0.9),
                                                cmap_name=pycortex_cmap, ori='horizontal', param_name='Surround Amplitude')
                                
                                if self.pycortex_image_path != None and save_colorbars:
                                    fig.savefig(f"{self.pycortex_image_path}/{model}_surround_amplitude_cbar.pdf", dpi=600, bbox_inches='tight', transparent=True)  

                        
                        self.js_handle_dict[space][analysis][subj]['amp'] = cortex.webgl.show(ds_amp, pickerfun=pickerfun,  overlays_visible=[], labels_visible=[]) 

                        
                    if self.plot_css_exp_cortex and 'CSS' in self.only_models:
                        ds_css_exp = dict()
                        
                        ds_css_exp[f'{subj} CSS Exponent'] = Vertex2D_fix(p_r['CSS Exponent']['CSS'], alpha[analysis][subj]['CSS'], subject=pycortex_subj, 
                                                                     vmin=0, vmax=1, vmin2=rsq_thresh, vmax2=rsq_max_opacity, cmap=pycortex_cmap, roi_borders=roi_borders)
        
                        self.js_handle_dict[space][analysis][subj]['css_exp'] = cortex.webgl.show(ds_css_exp, pickerfun=pickerfun,  overlays_visible=[], labels_visible=[]) 
                        
                    if self.plot_surround_size_cortex:
                        ds_surround_size = dict()
                        
                        
                        for model in self.only_models:
                            if model == 'DoG' or 'Norm' in model:

                                ds_surround_size[f"{subj} {model} fwatmin"] = Vertex2D_fix(p_r['Surround Size (fwatmin)'][model], alpha[analysis][subj][model], subject=pycortex_subj, 
                                                            vmin=np.nanquantile(p_r['Surround Size (fwatmin)'][model][alpha[analysis][subj][model]>rsq_thresh],0.1), 
                                                            vmax=np.nanquantile(p_r['Surround Size (fwatmin)'][model][alpha[analysis][subj][model]>rsq_thresh],0.9),
                                                            vmin2=rsq_thresh, vmax2=rsq_max_opacity, cmap=pycortex_cmap, roi_borders=roi_borders)                    
                                ds_surround_size[f"{subj} {model} sigma_2"] = Vertex2D_fix(p_r['Size (sigma_2)'][model], alpha[analysis][subj][model], subject=pycortex_subj, 
                                                            vmin=np.nanquantile(p_r['Size (sigma_2)'][model][alpha[analysis][subj][model]>rsq_thresh],0.1), 
                                                            vmax=np.nanquantile(p_r['Size (sigma_2)'][model][alpha[analysis][subj][model]>rsq_thresh],0.9),
                                                            vmin2=rsq_thresh, vmax2=rsq_max_opacity, cmap=pycortex_cmap, roi_borders=roi_borders)   

                                fig = simple_colorbar(vmin=np.nanquantile(p_r['Size (sigma_2)'][model][alpha[analysis][subj][model]>rsq_thresh],0.1), 
                                                vmax=np.nanquantile(p_r['Size (sigma_2)'][model][alpha[analysis][subj][model]>rsq_thresh],0.9),
                                                cmap_name=pycortex_cmap, ori='horizontal', param_name='$\sigma_2$ (°)')
                                
                                if self.pycortex_image_path != None and save_colorbars:
                                    fig.savefig(f"{self.pycortex_image_path}/{model}_sigma2_cbar.pdf", dpi=600, bbox_inches='tight', transparent=True)   
                                
                                
                        self.js_handle_dict[space][analysis][subj]['surround_size'] = cortex.webgl.show(ds_surround_size, pickerfun=pickerfun,  overlays_visible=[], labels_visible=[])  

                    if self.plot_suppression_index_cortex:
                        ds_suppression_index = dict()
                        
                        
                        for model in self.only_models:
                            if model == 'DoG':
                                ds_suppression_index[f'{subj} {model} SI (full)'] = Vertex2D_fix(p_r['Suppression Index (full)'][model], alpha[analysis][subj][model], subject=pycortex_subj, 
                                                                     vmin=0, vmax=10, vmin2=rsq_thresh, vmax2=rsq_max_opacity, cmap=pycortex_cmap, roi_borders=roi_borders)
                                ds_suppression_index[f'{subj} {model} SI (aperture)'] = Vertex2D_fix(p_r['Suppression Index'][model], alpha[analysis][subj][model], subject=pycortex_subj, 
                                                                     vmin=0, vmax=1.5, vmin2=rsq_thresh, vmax2=rsq_max_opacity, cmap=pycortex_cmap, roi_borders=roi_borders)                                    
                            elif 'Norm' in model:
                                ds_suppression_index[f'{subj} {model} NI (full)'] = Vertex2D_fix(p_r['Suppression Index (full)'][model], alpha[analysis][subj][model], subject=pycortex_subj, 
                                                                     vmin=1, vmax=20, vmin2=rsq_thresh, vmax2=rsq_max_opacity, cmap=pycortex_cmap, roi_borders=roi_borders)
                                ds_suppression_index[f'{subj} {model} NI (aperture)'] = Vertex2D_fix(p_r['Suppression Index'][model], alpha[analysis][subj][model], subject=pycortex_subj, 
                                                                     vmin=0, vmax=1.5, vmin2=rsq_thresh, vmax2=rsq_max_opacity, cmap=pycortex_cmap, roi_borders=roi_borders)                     
        
                        self.js_handle_dict[space][analysis][subj]['suppression_index'] = cortex.webgl.show(ds_suppression_index, pickerfun=pickerfun,  overlays_visible=[], labels_visible=[])    

                    if self.plot_size_ratio_cortex:
                        ds_size_ratio = dict()           
                        for model in self.only_models:
                            if model in p_r['Size ratio (sigma_2/sigma_1)']:
                                ds_size_ratio[f'{subj} {model} size ratio'] = Vertex2D_fix(p_r['Size ratio (sigma_2/sigma_1)'][model], alpha[analysis][subj][model], subject=pycortex_subj, 
                                                            vmin=np.nanquantile(p_r['Size ratio (sigma_2/sigma_1)'][model][alpha[analysis][subj][model]>rsq_thresh],0.1), 
                                                            vmax=np.nanquantile(p_r['Size ratio (sigma_2/sigma_1)'][model][alpha[analysis][subj][model]>rsq_thresh],0.9), vmin2=rsq_thresh, vmax2=rsq_max_opacity, cmap=pycortex_cmap, roi_borders=roi_borders)                                                      

                                fig = simple_colorbar(vmin=np.nanquantile(p_r['Size ratio (sigma_2/sigma_1)'][model][alpha[analysis][subj][model]>rsq_thresh],0.1), 
                                                vmax=np.nanquantile(p_r['Size ratio (sigma_2/sigma_1)'][model][alpha[analysis][subj][model]>rsq_thresh],0.9),
                                                cmap_name=pycortex_cmap, ori='horizontal', param_name='Size ratio ($\sigma_2/\sigma_1$)')
                                
                                if self.pycortex_image_path != None and save_colorbars:
                                    fig.savefig(f"{self.pycortex_image_path}/{model}_sizeratio_cbar.pdf", dpi=600, bbox_inches='tight', transparent=True)  

                        self.js_handle_dict[space][analysis][subj]['size_ratio'] = cortex.webgl.show(ds_size_ratio, pickerfun=pickerfun,  overlays_visible=[], labels_visible=[])     

                    if self.plot_hrf_cortex:
                        ds_hrf = dict()           
                        for model in self.only_models:
                            for key in [k for k in p_r if 'hrf' in k]:
                                if model in p_r[key]:
                                    ds_hrf[f'{subj} {model} {key}'] = Vertex2D_fix(p_r[key][model], alpha[analysis][subj][model], subject=pycortex_subj, 
                                                                vmin=np.nanquantile(p_r[key][model][alpha[analysis][subj][model]>rsq_thresh],0.1), 
                                                                vmax=np.nanquantile(p_r[key][model][alpha[analysis][subj][model]>rsq_thresh],0.9), vmin2=rsq_thresh, vmax2=rsq_max_opacity, cmap=pycortex_cmap, roi_borders=roi_borders)                                                      

                                    fig = simple_colorbar(vmin=np.nanquantile(p_r[key][model][alpha[analysis][subj][model]>rsq_thresh],0.1), 
                                                    vmax=np.nanquantile(p_r[key][model][alpha[analysis][subj][model]>rsq_thresh],0.9),
                                                    cmap_name=pycortex_cmap, ori='horizontal', param_name=key)
                                    
                                    if self.pycortex_image_path != None and save_colorbars:
                                        fig.savefig(f"{self.pycortex_image_path}/{model}_sizeratio_cbar.pdf", dpi=600, bbox_inches='tight', transparent=True)  

                        self.js_handle_dict[space][analysis][subj]['hrf_params'] = cortex.webgl.show(ds_hrf, pickerfun=pickerfun,  overlays_visible=[], labels_visible=[])     
                    

                    if self.plot_receptors_cortex:
                        if 'Receptor Maps' in p_r:
                            ds_receptors = dict()           
                            for receptor in p_r['Receptor Maps']:
                                if len(self.only_models)>1:
                                    rec_mod_alpha = 'all'
                                else:
                                    rec_mod_alpha = self.only_models[0]

                                ds_receptors[f'{subj} {receptor}'] = Vertex2D_fix(p_r['Receptor Maps'][receptor], alpha[analysis][subj][rec_mod_alpha], subject=pycortex_subj, 
                                                                         vmin=np.nanquantile(p_r['Receptor Maps'][receptor][alpha[analysis][subj][rec_mod_alpha]>rsq_thresh],0.1),
                                                                         vmax=np.nanquantile(p_r['Receptor Maps'][receptor][alpha[analysis][subj][rec_mod_alpha]>rsq_thresh],0.9), 
                                                                         vmin2=rsq_thresh, vmax2=rsq_max_opacity, cmap=pycortex_cmap, roi_borders=roi_borders)   

                                fig = simple_colorbar(vmin=np.nanquantile(p_r['Receptor Maps'][receptor][alpha[analysis][subj][rec_mod_alpha]>rsq_thresh],0.1),
                                                vmax=np.nanquantile(p_r['Receptor Maps'][receptor][alpha[analysis][subj][rec_mod_alpha]>rsq_thresh],0.9), 
                                                cmap_name=pycortex_cmap, ori='horizontal', param_name=f'{receptor} (pmol/ml)')
                                
                                if self.pycortex_image_path != None and save_colorbars:
                                    fig.savefig(f"{self.pycortex_image_path}/{receptor}_cbar.pdf", dpi=600, bbox_inches='tight', transparent=True)                                                    
            
                            self.js_handle_dict[space][analysis][subj]['receptors'] = cortex.webgl.show(ds_receptors, pickerfun=pickerfun,  overlays_visible=[], labels_visible=[])     

                         
                    if self.plot_norm_baselines_cortex:
                        ds_norm_baselines = dict()
                        
                        
                        for model in [model for model in self.only_models if 'Norm' in model]:
                            

                            ds_norm_baselines[f'{subj} {model} Param. B'] = Vertex2D_fix(p_r['Norm Param. B'][model], alpha[analysis][subj][model], subject=pycortex_subj, 
                                                                         vmin=np.nanquantile(p_r['Norm Param. B'][model][alpha[analysis][subj][model]>rsq_thresh],0.1), 
                                                                         vmax=np.nanquantile(p_r['Norm Param. B'][model][alpha[analysis][subj][model]>rsq_thresh],0.9), 
                                                                         vmin2=rsq_thresh, vmax2=rsq_max_opacity, cmap=pycortex_cmap, roi_borders=roi_borders)     
                            
                            fig = simple_colorbar(vmin=np.nanquantile(p_r['Norm Param. B'][model][alpha[analysis][subj][model]>rsq_thresh],0.1), 
                                            vmax=np.nanquantile(p_r['Norm Param. B'][model][alpha[analysis][subj][model]>rsq_thresh],0.9),
                                            cmap_name=pycortex_cmap, ori='horizontal', param_name='Norm Param. B')
                            
                            if self.pycortex_image_path != None and save_colorbars:
                                fig.savefig(f"{self.pycortex_image_path}/{model}_paramB_cbar.pdf", dpi=600, bbox_inches='tight', transparent=True)  
                            
                                
                            ds_norm_baselines[f'{subj} {model} Param. D'] = Vertex2D_fix(p_r['Norm Param. D'][model], alpha[analysis][subj][model], subject=pycortex_subj, 
                                                                         vmin=np.nanquantile(p_r['Norm Param. D'][model][alpha[analysis][subj][model]>rsq_thresh],0.1), 
                                                                         vmax=np.nanquantile(p_r['Norm Param. D'][model][alpha[analysis][subj][model]>rsq_thresh],0.9),
                                                                         vmin2=rsq_thresh, vmax2=rsq_max_opacity, cmap=pycortex_cmap, roi_borders=roi_borders)
                            
                            fig = simple_colorbar(vmin=np.nanquantile(p_r['Norm Param. D'][model][alpha[analysis][subj][model]>rsq_thresh],0.1), 
                                            vmax=np.nanquantile(p_r['Norm Param. D'][model][alpha[analysis][subj][model]>rsq_thresh],0.9),
                                            cmap_name=pycortex_cmap, ori='horizontal', param_name='Norm Param. D')
                            
                            if self.pycortex_image_path != None and save_colorbars:
                                fig.savefig(f"{self.pycortex_image_path}/{model}_paramD_cbar.pdf", dpi=600, bbox_inches='tight', transparent=True)                              
                            
                            if 'Ratio (B/D)' in p_r:
                                ds_norm_baselines[f'{subj} {model} Ratio (B/D)'] = Vertex2D_fix(p_r['Ratio (B/D)'][model], alpha[analysis][subj][model], subject=pycortex_subj, 
                                                                          vmin=np.nanquantile(p_r['Ratio (B/D)'][model][alpha[analysis][subj][model]>rsq_thresh],0.1), 
                                                                          vmax=np.nanquantile(p_r['Ratio (B/D)'][model][alpha[analysis][subj][model]>rsq_thresh],0.9), vmin2=rsq_thresh, vmax2=rsq_max_opacity, cmap=pycortex_cmap, roi_borders=roi_borders)

                                fig = simple_colorbar(vmin=np.nanquantile(p_r['Ratio (B/D)'][model][alpha[analysis][subj][model]>rsq_thresh],0.1), 
                                                vmax=np.nanquantile(p_r['Ratio (B/D)'][model][alpha[analysis][subj][model]>rsq_thresh],0.9),
                                                cmap_name=pycortex_cmap, ori='horizontal', param_name='Ratio (B/D)')
                                
                                if self.pycortex_image_path != None and save_colorbars:
                                    fig.savefig(f"{self.pycortex_image_path}/{model}_BDratio_cbar.pdf", dpi=600, bbox_inches='tight', transparent=True)   
                            
                        self.js_handle_dict[space][analysis][subj]['norm_baselines'] = cortex.webgl.show(ds_norm_baselines, pickerfun=pickerfun,  overlays_visible=[], labels_visible=[]) 

                    if self.plot_correlations_per_roi != None:
                        ds_correlations = dict()

                        for x_param_topl in self.plot_correlations_per_roi['x_params_topl']:
                            for y_param_topl in self.plot_correlations_per_roi['y_params_topl']:
                                for x_param_lowl in self.plot_correlations_per_roi['x_params_lowl']:
                                    for y_param_lowl in self.plot_correlations_per_roi['y_params_lowl']:
                                        
                                        if x_param_lowl in self.only_models and y_param_lowl in self.only_models:
                                            all_rois_alpha = (alpha[analysis][subj][x_param_lowl]+alpha[analysis][subj][y_param_lowl])/2       
                                        elif x_param_lowl in self.only_models:
                                            all_rois_alpha = alpha[analysis][subj][x_param_lowl]                                           
                                        elif y_param_lowl in self.only_models:
                                            all_rois_alpha = alpha[analysis][subj][y_param_lowl]
                                            
                                        corr_per_roi_data = np.zeros_like(mask).astype('float')
                                        alpha_per_roi = np.zeros_like(mask).astype('float')

                                        for roi in [r for r in self.idx_rois[subj] if self.plot_correlations_per_roi['atlas'] in r]:
                                            rsq_weights = roi_mask(self.idx_rois[subj][roi], all_rois_alpha)
                                            
                                            if np.sum(rsq_weights>rsq_thresh)>10:
                                            
                                                covariance = np.cov(p_r[x_param_topl][x_param_lowl][rsq_weights>rsq_thresh], p_r[y_param_topl][y_param_lowl][rsq_weights>rsq_thresh],
                                                       aweights=rsq_weights[rsq_weights>rsq_thresh])
                                                
                                                corr = np.dot(np.diag(np.power(np.diag(covariance),-0.5)),np.dot(covariance,np.diag(np.power(np.diag(covariance),-0.5))))[0,1]
                                                
                                                corr_per_roi_data[rsq_weights>rsq_thresh] = corr
                                                alpha_per_roi[rsq_weights>rsq_thresh] = np.mean(rsq_weights[rsq_weights>rsq_thresh])
                                                
                                            

                                        ds_correlations[f'{subj} {x_param_topl} {x_param_lowl} VS {y_param_topl} {y_param_lowl}'] = Vertex2D_fix(corr_per_roi_data, alpha_per_roi, subject=pycortex_subj, 
                                                                          vmin=np.nanquantile(corr_per_roi_data[all_rois_alpha>rsq_thresh],0.1), vmin2=rsq_thresh, vmax2=rsq_max_opacity,
                                                                          vmax=np.nanquantile(corr_per_roi_data[all_rois_alpha>rsq_thresh],0.9), #alpha=(np.clip(alpha_per_roi, rsq_thresh, 0.6)-rsq_thresh)/(0.6-rsq_thresh),
                                                                          cmap=pycortex_cmap, roi_borders=roi_borders)                   


                        self.js_handle_dict[space][analysis][subj]['correlations_per_roi'] = cortex.webgl.show(ds_correlations, pickerfun=pickerfun,  overlays_visible=[], labels_visible=[])     
                    
                    if self.plot_means_per_roi != None:
                        ds_means = dict()
                        
                        for y_param_topl in self.plot_means_per_roi['y_params_topl']:
                                
                            for y_param_lowl in self.plot_means_per_roi['y_params_lowl']:
                                                                               
                                if y_param_lowl in self.only_models:
                                    all_rois_alpha = alpha[analysis][subj][y_param_lowl]
                                else:
                                    all_rois_alpha = np.ones_like(alpha[analysis][subj][y_param_lowl])
                                    
                                mean_per_roi_data = np.zeros_like(mask).astype('float')
                                alpha_per_roi = np.zeros_like(mask).astype('float')

                                for roi in [r for r in self.idx_rois[subj] if self.plot_means_per_roi['atlas'] in r]:
                                    rsq_weights = roi_mask(self.idx_rois[subj][roi], all_rois_alpha)
                                    
                                    if np.sum(rsq_weights>rsq_thresh)>10:
                                    
                                        mean_per_roi_data[rsq_weights>rsq_thresh] = (p_r[y_param_topl][y_param_lowl][rsq_weights>rsq_thresh]*rsq_weights[rsq_weights>rsq_thresh]).sum(0)/rsq_weights[rsq_weights>rsq_thresh].sum(0)
                                        alpha_per_roi[rsq_weights>rsq_thresh] = np.mean(rsq_weights[rsq_weights>rsq_thresh])
                                        
                                    

                                ds_means[f'{subj} Mean {y_param_topl} {y_param_lowl} per roi'] = Vertex2D_fix(mean_per_roi_data, alpha_per_roi, subject=pycortex_subj, 
                                                                  vmin=0,#np.nanquantile(mean_per_roi_data[all_rois_alpha>rsq_thresh],0.1), 
                                                                  vmin2=rsq_thresh, vmax2=rsq_max_opacity,
                                                                  vmax=100,#np.nanquantile(mean_per_roi_data[all_rois_alpha>rsq_thresh],0.9), #alpha=(np.clip(alpha_per_roi, rsq_thresh, 0.6)-rsq_thresh)/(0.6-rsq_thresh),
                                                                  cmap=pycortex_cmap, roi_borders=roi_borders)                  


                        self.js_handle_dict[space][analysis][subj]['means_per_roi'] = cortex.webgl.show(ds_means, pickerfun=pickerfun,  overlays_visible=[], labels_visible=[]) 
                                
        
        
        
        print('-----')                              
        return    
        
    def save_pycortex_views(self, space, analysis, subj, js_handle, param_to_save, views_dict, set_views=True, image_path = '/Crucial X8/marcoaqil/PRFMapping/Figures/'):

        time.sleep(0.3)
        self.js_handle_dict[space][analysis][subj][js_handle].setData([param_to_save])
        time.sleep(0.3)
        if subj in param_to_save:
            base_str = f"{param_to_save}".replace('/','')
        else:
            base_str = f"{subj}_{param_to_save}".replace('/','')
 
        # Save images by iterating over the different views and surfaces
        for view, view_params in views_dict.items():
            filename = f"{base_str}_{view}.png"
            output_path = os.path.join(image_path, filename)
            #print(view)
            if set_views:
                for param_name, param_value in view_params.items():
                    #print(param_name)
                    #print(param_value)
                    time.sleep(0.3)
                    self.js_handle_dict[space][analysis][subj][js_handle].ui.set(param_name, param_value)
                
                
            # Save image    
            time.sleep(0.3)       
            self.js_handle_dict[space][analysis][subj][js_handle].getImage(output_path, size =(3200, 2400))
        
            # the block below trims the edges of the image:
            # wait for image to be written
            while not os.path.exists(output_path):
                pass
            time.sleep(0.3)
            try:
                import subprocess
                subprocess.call(["convert", "-trim", output_path, output_path])
            except:
                pass
                
    def project_to_fsaverage(self, models, parameters, space_names = 'all', analysis_names = 'all', subject_ids='all',
                             weight = 'RSq', groupname = '',
                             hcp_atlas_mask_path = None, 
                             hcp_cii_file_path = None, hcp_old_sphere = None, hcp_new_sphere = None, hcp_old_area = None, hcp_new_area = None,
                             hcp_temp_folder = None):
        if 'fsaverage' not in self.main_dict:
            self.main_dict['fsaverage'] = dd(lambda:dd(lambda:dd(lambda:dd(dict))))
        
        if parameters[0] != weight:
            if 'Timecourse Stats' not in parameters[0]:
                parameters.insert(0,weight)
            
        if 'Polar Angle' in parameters or 'Eccentricity' in parameters:
            print("Are you sure you want to resample polar angle and eccentricity? this can cause interpolation issues. better to use x_pos and y_pos instead.")
            
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
                    fsaverage_rsq = dict()
                    for parameter in tqdm(parameters):
                        
                        fsaverage_param = dict()
                        
                        for subj, subj_res in tqdm(subjects):
                            print(f"{space} {analysis} {subj} {parameter} {model}")
                            p_r = subj_res['Processed Results']

                    
                            if space == 'fsnative':
                                if 'ses' in subj:
                                    fs_subj = subj.split('_')[0]
                                else:
                                    fs_subj = subj
                            
                                lh_c = read_morph_data(opj(self.fs_dir, f"{fs_subj}/surf/lh.curv"))
                                
                                if parameter in p_r:
                                    if model in p_r[parameter]:
                                        param = np.copy(p_r[parameter][model])
                                else:
                                    if parameter == 'x_pos':
                                        param = p_r['Eccentricity'][model]*np.cos(p_r['Polar Angle'][model])
                                    elif parameter == 'y_pos':
                                        param = p_r['Eccentricity'][model]*np.sin(p_r['Polar Angle'][model])
                                    else:
                                        print('parameter not found')
                                        continue
                                        
                                            
                                
                                lh_file_path = opj(self.fs_dir, f"{fs_subj}/surf/lh.{subj}_{''.join(filter(str.isalnum, parameter))}_{''.join(filter(str.isalnum, model))}")
                                rh_file_path = opj(self.fs_dir, f"{fs_subj}/surf/rh.{subj}_{''.join(filter(str.isalnum, parameter))}_{''.join(filter(str.isalnum, model))}")
    
                                write_morph_data(lh_file_path, param[:lh_c.shape[0]])
                                write_morph_data(rh_file_path, param[lh_c.shape[0]:])

                                lh_fsaverage_path =  opj(self.fs_dir, f"fsaverage/surf/lh.{subj}_{''.join(filter(str.isalnum, parameter))}_{''.join(filter(str.isalnum, model))}")
                                rh_fsaverage_path = opj(self.fs_dir, f"fsaverage/surf/rh.{subj}_{''.join(filter(str.isalnum, parameter))}_{''.join(filter(str.isalnum, model))}")
                                
                                os.system("export FREESURFER_HOME=/Applications/freesurfer/7.2.0/")
                                os.system("source $FREESURFER_HOME/SetUpFreeSurfer.sh")
                                os.system(f"export SUBJECTS_DIR='{self.fs_dir}'")
                                os.system(f"mri_surf2surf --srcsubject {fs_subj} --srcsurfval '{lh_file_path}' --trgsubject fsaverage --trgsurfval '{lh_fsaverage_path}' --hemi lh --trg_type curv")
                                os.system(f"mri_surf2surf --srcsubject {fs_subj} --srcsurfval '{rh_file_path}' --trgsubject fsaverage --trgsurfval '{rh_fsaverage_path}' --hemi rh --trg_type curv")
    
                                lh_fsaverage_param = read_morph_data(lh_fsaverage_path)
                                rh_fsaverage_param = read_morph_data(rh_fsaverage_path)
                                
                                
                                if parameter == weight:
                                    fsaverage_rsq[subj] = np.nan_to_num(np.concatenate((lh_fsaverage_param,rh_fsaverage_param)))
                                    fsaverage_rsq[subj][fsaverage_rsq[subj]<0] = 0
                                    print(np.all(np.isfinite(fsaverage_rsq[subj])))
                                    self.main_dict['fsaverage'][analysis][subj]['Processed Results'][parameter][model] = np.copy(fsaverage_rsq[subj])   
                                else:
                                    fsaverage_param[subj] = np.concatenate((lh_fsaverage_param,rh_fsaverage_param))
                                    self.main_dict['fsaverage'][analysis][subj]['Processed Results'][parameter][model] = np.copy(fsaverage_param[subj])    
                                    
                            elif space == 'HCP':
                                f=nb.load(hcp_atlas_mask_path[0])
                                data_1 = np.array([arr.data for arr in f.darrays])[0].astype('bool')
                        
                                f=nb.load(hcp_atlas_mask_path[1])
                                data_2 = np.array([arr.data for arr in f.darrays])[0].astype('bool')
                                
                                
                                cifti_brain_model = cifti.read(hcp_cii_file_path)[1][1]
                                
                                output = np.zeros(len(cifti_brain_model))
                                
                                if parameter in p_r:                              
                                    output[:np.sum(data_1)] = p_r[parameter][model][:len(data_1)][data_1]
                                    output[np.sum(data_1):(np.sum(data_1) + np.sum(data_2))] = p_r[parameter][model][len(data_1):][data_2]
                                else:
                                    if parameter == 'x_pos':
                                        output[:np.sum(data_1)] = p_r['Eccentricity'][model][:len(data_1)][data_1]*np.cos(p_r['Polar Angle'][model][:len(data_1)][data_1])
                                        output[np.sum(data_1):(np.sum(data_1) + np.sum(data_2))] = p_r['Eccentricity'][model][len(data_1):][data_2]*np.cos(p_r['Polar Angle'][model][len(data_1):][data_2])                          
                                    elif parameter == 'y_pos':
                                        output[:np.sum(data_1)] = p_r['Eccentricity'][model][:len(data_1)][data_1]*np.sin(p_r['Polar Angle'][model][:len(data_1)][data_1])
                                        output[np.sum(data_1):(np.sum(data_1) + np.sum(data_2))] = p_r['Eccentricity'][model][len(data_1):][data_2]*np.sin(p_r['Polar Angle'][model][len(data_1):][data_2])   
                                    else:
                                        print(f"WARNING: unidentified parameter {parameter}")              
                                        raise IOError
                                    
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

                                if parameter == weight:
                                    fsaverage_rsq[subj] = np.nan_to_num(np.concatenate((np.array([arr.data for arr in a.darrays])[0],np.array([arr.data for arr in b.darrays])[0])))
                                    fsaverage_rsq[subj][fsaverage_rsq[subj]<0] = 0
                                    print(np.any(np.isnan(fsaverage_rsq[subj])))
                                    print(np.any(~np.isfinite(fsaverage_rsq[subj])))                                    
                                    self.main_dict['fsaverage'][analysis][subj]['Processed Results'][parameter][model] = np.copy(fsaverage_rsq[subj])   
                                else:           
                                    fsaverage_param[subj] = np.concatenate((np.array([arr.data for arr in a.darrays])[0],np.array([arr.data for arr in b.darrays])[0]))
                                    print(np.any(np.isnan(fsaverage_param[subj])))
                                    print(np.any(~np.isfinite(fsaverage_param[subj])))
                                    self.main_dict['fsaverage'][analysis][subj]['Processed Results'][parameter][model] = np.copy(fsaverage_param[subj])
                                    

                        
                        if parameter == weight:
                            fsaverage_group_average = np.nanmean([fsaverage_rsq[sid] for sid in fsaverage_rsq], axis=0)
                        elif 'Timecourse Stats' in parameter:
                            fsaverage_group_average = np.nanmean([fsaverage_param[sid] for sid in fsaverage_param], axis=0)
                        # elif '-' in parameter:
                        #     fsaverage_group_average = np.nanmean([fsaverage_param[sid] for sid in fsaverage_param], axis=0)
                        else:

                            data = np.array([fsaverage_param[sid] for sid in fsaverage_param])
                            weights = np.array([fsaverage_rsq[sid] for sid in fsaverage_rsq])
                            
                            fsaverage_group_average = (data*weights).sum(0)/weights.sum(0)
                        
                        self.main_dict['fsaverage'][analysis][f'fsaverage{groupname}']['Processed Results'][parameter][model] = np.copy(np.array(fsaverage_group_average))
 

            if len(analyses)>1:            
                for model in models:
                    for parameter in parameters:
                        self.main_dict['fsaverage']['Mean analysis'][f'fsaverage{groupname}']['Processed Results'][parameter][model] = np.nanmean([self.main_dict['fsaverage'][an[0]][f'fsaverage{groupname}']['Processed Results'][parameter][model] for an in analyses], axis=0)
                        
                        
        for analysis, analysis_res in self.main_dict['fsaverage'].items():
            for subj, subj_res in analysis_res.items():
                for model in models:
                    
                    if 'x_pos' in parameters and 'y_pos' in parameters:
                        if model in subj_res['Processed Results']['y_pos'] and model in subj_res['Processed Results']['x_pos']:
                            subj_res['Processed Results']['Polar Angle'][model] = np.arctan2(subj_res['Processed Results']['y_pos'][model],subj_res['Processed Results']['x_pos'][model])
                            subj_res['Processed Results']['Eccentricity'][model] = np.sqrt(subj_res['Processed Results']['y_pos'][model]**2+subj_res['Processed Results']['x_pos'][model]**2)
                    if 'Size (sigma_1)' in parameters and 'Size (sigma_2)' in parameters:
                        if model in subj_res['Processed Results']['Size (sigma_1)'] and model in subj_res['Processed Results']['Size (sigma_2)']:
                            subj_res['Processed Results']['Size ratio (sigma_2/sigma_1)'][model] = subj_res['Processed Results']['Size (sigma_2)'][model]/subj_res['Processed Results']['Size (sigma_1)'][model]

                    if 'Norm Param. B' in parameters and 'Norm Param. D' in parameters:
                        if model in subj_res['Processed Results']['Norm Param. B'] and model in subj_res['Processed Results']['Norm Param. D']:
                            subj_res['Processed Results']['Ratio (B/D)'][model] = subj_res['Processed Results']['Norm Param. B'][model]/subj_res['Processed Results']['Norm Param. D'][model]
       
        return
    

    def group_quant_plots(self, x_parameter, y_parameter, rois, rsq_thresh, save_figures, figure_path,
                    space_names = 'fsnative', analysis_names = 'all', subject_ids='all', groups=[],
                    x_parameter_toplevel='', y_parameter_toplevel='', 
                    ylim={}, xlim={}, log_yaxis=False, log_xaxis = False, nr_bins = 8, weights='RSq',
                    x_param_model='', violin=False, scatter=False, diff_norm=False, diff_gauss=False, diff_gauss_x=False,
                    rois_on_plot = False, rsq_alpha_plot = False,
                    means_only=False, stats_on_plot=False, only_stats=False, bin_by='size', zscore_ydata=False, zscore_xdata=False,
                    zconfint_err_alpha = None, fit = True,exp_fit = False, show_legend=False, each_subj_on_group=False,
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
            DESCRIPTION. The default == None.
        x_param_model : TYPE, optional
            DESCRIPTION. The default == None.
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
        
        cmap_values = list(np.linspace(0.9, 0.0, len([r for r in rois if r not in ['Brain', 'all_custom', 'combined']])))
        
        cmap_values += [0 for r in rois  if r in ['Brain', 'all_custom', 'combined']]
        
        cmap_rois = cm.get_cmap('nipy_spectral')(cmap_values)#

        self.curr_rois_names = []
        
        #making black into dark gray for visualization reasons
        cmap_rois[(cmap_rois == [0,0,0,1]).sum(1)==4] = [0.33,0.33,0.33,1]

        if space_names == 'all':
            spaces = [item for item in self.main_dict.items()]
        else:
            spaces = [item for item in self.main_dict.items() if item[0] in space_names] 

        for space, space_res in spaces:
            if 'fs' in space or 'HCP' in space:
                
                if quantile_exclusion == None:
                    #surrounds larger than this are excluded from surround size calculations
                    w_max=60
                    #remove absurd suppression index values
                    supp_max=1000
                    #css max
                    css_max=1
                    #max b and d
                    bd_max=1000
                #bar or violin width
                if len(self.only_models)>1:
                    bar_or_violin_width = 0.3
                else:
                    bar_or_violin_width = 1.5

                
                        
                if analysis_names == 'all':
                    analyses = [item for item in space_res.items()]
                else:
                    analyses = [item for item in space_res.items() if item[0] in analysis_names] 
                    
                if len(analyses)>1:
                    analyses.append(('Analyses mean', {sub:{} for sub in analyses[0][1]}))
                    

                alpha = dd(lambda:dd(lambda:dd(lambda:dd(dict))))
                x_par = dd(lambda:dd(lambda:dd(lambda:dd(dict))))
                y_par = dd(lambda:dd(lambda:dd(lambda:dd(dict))))
                rsq_y = dd(lambda:dd(lambda:dd(lambda:dd(dict))))
                rsq_x = dd(lambda:dd(lambda:dd(lambda:dd(dict))))
                rsq_regr_weight = dd(lambda:dd(lambda:dd(lambda:dd(list))))
    
                x_par_stats = dd(lambda:dd(lambda:dd(lambda:dd(list))))
                y_par_stats = dd(lambda:dd(lambda:dd(lambda:dd(list))))
                rsq_y_stats = dd(lambda:dd(lambda:dd(lambda:dd(list))))
                rsq_x_stats = dd(lambda:dd(lambda:dd(lambda:dd(list))))
                
                bootstrap_fits = dd(lambda:dd(lambda:dd(lambda:dd(list))))

                dict_lines = dd(lambda:dd(lambda:dd(lambda:dd(lambda:dd(list)))))

                ssj_group_stats_roi_means = dd(lambda:dd(lambda:dd(lambda:dd(list))))
                ssj_group_stats_roi_errs = dd(lambda:dd(lambda:dd(lambda:dd(list))))
                ssj_group_stats_roi_weights = dd(lambda:dd(lambda:dd(lambda:dd(list))))
                
                
                for analysis, analysis_res in analyses:       
                    if subject_ids == 'all':
                        subjects = [item for item in analysis_res.items()]
                    else:
                        subjects = [item for item in analysis_res.items() if item[0] in subject_ids]
                    
                    if len(subjects)>1:
                        for group in groups:
                            subjects.append((f'Group_{group}', {}))
                    
                    print([ee[0] for ee in subjects])
                        
                    figure_path = base_fig_path

                    if not os.path.exists(figure_path):
                        os.makedirs(figure_path)
                    figure_path = opj(figure_path, space)
                    if not os.path.exists(figure_path):
                        os.makedirs(figure_path)
                    figure_path = opj(figure_path, analysis)
                    if not os.path.exists(figure_path):
                        os.makedirs(figure_path)                            
                    
                    upsampling_corr_factors = dd(list)



                    for subj, subj_res in subjects:

                        if 'Group' not in subj:
                            group = self.find_group(subj)
                        else:
                            group = subj.split('_')[1]

                        print(f"{space} {analysis} {subj} {group}")

                        
                         
                        #upsampling correction: fsnative has approximately 3 times as many datapoints as original
                        if 'Group' not in subj:
                            if bold_voxel_volume != None:
                                
                                print("Make sure bold_voxel_volume is specified in mm^3")
                                
                                try:
                                    if subj.isdecimal() and space == 'HCP':
                                        pycortex_subj = '999999'
                                    elif 'fsaverage' in subj or 'fsaverage' in space:
                                        pycortex_subj = 'fsaverage'                                        
                                    else:
                                        pycortex_subj = subj.split('_')[0]
                                        
                                    aseg = nb.load(opj(cortex.database.default_filestore,pycortex_subj,'anatomicals','aseg.nii.gz'))
                                    anat_vox_vol = aseg.header.get_zooms()[0]*aseg.header.get_zooms()[1]*aseg.header.get_zooms()[2]
                                    cortex_volume = ((aseg.get_fdata()==42).sum()+(aseg.get_fdata()==3).sum())*anat_vox_vol
                                    nr_bold_voxels = cortex_volume/bold_voxel_volume
                                    nr_surf_vertices = cortex.db.get_surfinfo(pycortex_subj).data.shape[0]
            
                                    upsampling_corr_factor = nr_surf_vertices / nr_bold_voxels
                                    
              
                                except Exception as e:
                                    print(e)
                                    print("Unable to perform upsampling correction.")
                                    upsampling_corr_factor = 1
                                    pass
                                    
                            else:
                                print("BOLD voxel volume not specified. Not performing upsampling correction.")
                                upsampling_corr_factor = 1
                            print(upsampling_corr_factor)
                            upsampling_corr_factors[group].append(upsampling_corr_factor)
                        else:
                            upsampling_corr_factor = np.mean(upsampling_corr_factors[group])
                            
                        print(f"Upsampling correction factor: {upsampling_corr_factor}")
                        
                        x_ticks=[]
                        x_labels=[]    
                        bar_position = 0

                        if len(groups)>1:

                            if group == groups[1]:
                                bar_position += 1.5*bar_or_violin_width
                            elif group == groups[2]:
                                bar_position += 3*bar_or_violin_width

                        elif '-' in y_parameter or '-' in y_parameter_toplevel:
                            if group == '10mg':
                                bar_position += 1.5*bar_or_violin_width
                        
                        # binned eccentricity vs other parameters relationships       
            
                        #model_colors = {'Gauss':'blue','CSS':'orange','DoG':'green','Norm_abcd':'red','Norm_abc':'purple'}

                        model_colors = {'Gauss':'blue','CSS':'orange','DoG':'green','Norm_abcd':'red','Norm_abc':'purple'}

                        
                                                

    
                        for i, roi in enumerate(rois):                              
                            for model in self.only_models:                                

                                
                                if 'mean an' not in analysis or 'fsaverage' in subj:
                                    if 'sub' in subj or 'fsaverage' in subj or subj.isdecimal():
                                        if space == 'fsaverage':
                                            roi_subj = 'fsaverage'
                                        else:
                                            roi_subj = subj

                                        if 'rsq' in y_parameter.lower():
                                            #comparing same vertices for model performance
                                            curr_alpha = subj_res['Processed Results']['Alpha']['all']
                                        else:
                                            #otherwise model-specific alpha
                                            curr_alpha = (subj_res['Processed Results']['Alpha'][model])
                                            
                                        if roi in self.idx_rois[roi_subj]:
    
                                            alpha[analysis][subj][model][roi] = (roi_mask(self.idx_rois[roi_subj][roi], curr_alpha)) 
       
                                        else:
                                            #if ROI != defined
                                            #if Brain use all available vertices
                                            if roi == 'Brain':
                                                alpha[analysis][subj][model][roi] = curr_alpha
                                            elif roi == 'combined':
                                                alpha[analysis][subj][model][roi] = (roi_mask(np.concatenate(tuple([self.idx_rois[roi_subj][r] for r in rois if ('combined' not in r and 'Brain' not in r and r in self.idx_rois[roi_subj])])), curr_alpha))    
                                            elif roi == 'all_custom':
                                                alpha[analysis][subj][model][roi] = (roi_mask(np.concatenate(tuple([self.idx_rois[roi_subj][r] for r in self.idx_rois[roi_subj] if 'custom' in r])), curr_alpha))    
                                            # elif space == 'fsaverage' and roi in self.idx_rois['fsaverage']:
                                            #     alpha[analysis][subj][model][roi] = (roi_mask(self.idx_rois['fsaverage'][roi], curr_alpha))
                                            else:
                                                #, otherwise none
                                                print(f"{roi}: undefined ROI")
                                                alpha[analysis][subj][model][roi] = np.zeros_like(curr_alpha).astype('bool')
                                        
                                        
                                        #manual exclusion of outliers
                                        if quantile_exclusion == None:
                                            print("Using manual exclusion, see quant_plots function. Set quantile_exclusion=1 for no exclusion.")
                                            if y_parameter == 'Surround Size (fwatmin)' or x_parameter == 'Surround Size (fwatmin)':# and model == 'DoG':
                                                #exclude too large surround (no surround)
                                                alpha[analysis][subj][model][roi] *= (subj_res['Processed Results']['Surround Size (fwatmin)'][model]<w_max)
                                                if x_param_model != '':
                                                    alpha[analysis][subj][model][roi] *= (subj_res['Processed Results']['Surround Size (fwatmin)'][x_param_model]<w_max)
    
                                            
                                            if y_parameter == 'Surround/Centre Amplitude'  or x_parameter == 'Surround/Centre Amplitude' :# and model == 'DoG':
                                                #exclude too large surround (no surround)
                                                if x_param_model != '':
                                                    alpha[analysis][subj][model][roi] *= (subj_res['Processed Results']['Surround/Centre Amplitude'][x_param_model]<w_max)
                                                else:
                                                    alpha[analysis][subj][model][roi] *= (subj_res['Processed Results']['Surround/Centre Amplitude'][model]<w_max)
    
                                                    
                                            if 'Suppression' in y_parameter:
                                                #exclude nonsensical suppression index values
                                                alpha[analysis][subj][model][roi] *= (subj_res['Processed Results'][y_parameter][model]<supp_max)                                            
                                            if 'Suppression' in x_parameter:
                                                #exclude nonsensical suppression index values
                                                alpha[analysis][subj][model][roi] *= (subj_res['Processed Results'][x_parameter][model]<supp_max)
                                                if x_param_model != '':
                                                    alpha[analysis][subj][model][roi] *= (subj_res['Processed Results'][x_parameter][x_param_model]<supp_max)
                                                    
                                            if 'CSS Exponent' in x_parameter:
                                                alpha[analysis][subj][model][roi] *= (subj_res['Processed Results'][x_parameter][x_param_model]<css_max)
                                            
                                            if 'Norm Param.' in y_parameter:
                                                alpha[analysis][subj][model][roi] *= (subj_res['Processed Results'][y_parameter][model]<bd_max)                                            
                                            if 'Norm Param.' in x_parameter:
                                                alpha[analysis][subj][model][roi] *= (subj_res['Processed Results'][x_parameter][model]<bd_max)                                            
                                                if x_param_model != '':
                                                    alpha[analysis][subj][model][roi] *= (subj_res['Processed Results'][x_parameter][x_param_model]<bd_max)
                                        
                                        else:
                                            if y_parameter_toplevel == '':    
                                                alpha[analysis][subj][model][roi] *= (subj_res['Processed Results'][y_parameter][model]<np.nanquantile(subj_res['Processed Results'][y_parameter][model],quantile_exclusion))*(subj_res['Processed Results'][y_parameter][model]>np.nanquantile(subj_res['Processed Results'][y_parameter][model],1-quantile_exclusion))  

                                            if x_param_model != '':
                                                #nanquantile handles nans but will not work if there are infs
                                                alpha[analysis][subj][model][roi] *= (subj_res['Processed Results'][x_parameter][x_param_model]<np.nanquantile(subj_res['Processed Results'][x_parameter][x_param_model],quantile_exclusion))*(subj_res['Processed Results'][x_parameter][x_param_model]>np.nanquantile(subj_res['Processed Results'][x_parameter][x_param_model],1-quantile_exclusion))  

                                            if x_parameter_toplevel == '':
                                                alpha[analysis][subj][model][roi] *= (subj_res['Processed Results'][x_parameter][model]<np.nanquantile(subj_res['Processed Results'][x_parameter][model],quantile_exclusion))*(subj_res['Processed Results'][x_parameter][model]>np.nanquantile(subj_res['Processed Results'][x_parameter][model],1-quantile_exclusion))  

                                            
                                            
         
                                        #if 'ccrsq' in y_parameter.lower():
                                        #    alpha[analysis][subj][model][roi] *= (subj_res['Processed Results'][y_parameter][model]>0)
                                        if y_parameter_toplevel == '':
                                            alpha[analysis][subj][model][roi] *= np.isfinite(subj_res['Processed Results'][y_parameter][model])
                                        else:
                                            alpha[analysis][subj][model][roi] *= np.isfinite(subj_res['Processed Results'][y_parameter_toplevel][y_parameter])
                                       
                                        if x_param_model != '':
                                            
                                            alpha[analysis][subj][model][roi] *= np.isfinite(subj_res['Processed Results'][x_parameter][x_param_model])
                                            
                                            x_par[analysis][subj][model][roi] = subj_res['Processed Results'][x_parameter][x_param_model][alpha[analysis][subj][model][roi]>rsq_thresh]                                          
                                        
                                        
                                        elif x_parameter_toplevel != '':
                                            alpha[analysis][subj][model][roi] *= np.isfinite(subj_res['Processed Results'][x_parameter_toplevel][x_parameter])
                                            x_par[analysis][subj][model][roi] = (subj_res['Processed Results'][x_parameter_toplevel][x_parameter][alpha[analysis][subj][model][roi]>rsq_thresh])
                                            
                                        else:
                                            #remove nans and infinities
                                            alpha[analysis][subj][model][roi] *= np.isfinite(subj_res['Processed Results'][x_parameter][model])                                    

                                            x_par[analysis][subj][model][roi] = subj_res['Processed Results'][x_parameter][model][alpha[analysis][subj][model][roi]>rsq_thresh]
                                            
                                        #handling special case of plotting receptors as y-parameter, since they are not part of any model
                                        if y_parameter_toplevel == '':
                                            y_par[analysis][subj][model][roi] = (subj_res['Processed Results'][y_parameter][model][alpha[analysis][subj][model][roi]>rsq_thresh])
                                        else:
                                            y_par[analysis][subj][model][roi] = (subj_res['Processed Results'][y_parameter_toplevel][y_parameter][alpha[analysis][subj][model][roi]>rsq_thresh])
                                            
                                        
                                        
                                        if diff_gauss:
                                            y_par[analysis][subj][model][roi] -= subj_res['Processed Results'][y_parameter]['Gauss'][alpha[analysis][subj][model][roi]>rsq_thresh]
                                        
                                        if diff_gauss_x:
                                            x_par[analysis][subj][model][roi] -= subj_res['Processed Results'][x_parameter]['Gauss'][alpha[analysis][subj][model][roi]>rsq_thresh]
                                        
                                            

                                            #set negative ccrsq and rsq to zero
                                            #if 'ccrsq' in y_parameter.lower():
                                            #    y_par[analysis][subj][model][roi][y_par[analysis][subj][model][roi]<0] = 0
                 
    
                                        #r - squared weighting
                                        if weights != None:
                                            rsq_x[analysis][subj][model][roi] = np.copy(subj_res['Processed Results'][weights][model][alpha[analysis][subj][model][roi]>rsq_thresh])
                                            rsq_y[analysis][subj][model][roi] = np.copy(subj_res['Processed Results'][weights][model][alpha[analysis][subj][model][roi]>rsq_thresh])
                                        else:
                                            rsq_x[analysis][subj][model][roi] = np.ones_like(x_par[analysis][subj][model][roi])    
                                            rsq_y[analysis][subj][model][roi] = np.ones_like(y_par[analysis][subj][model][roi])

                                        #no need for rsq-weighting if plotting rsq
                                        if 'rsq' in y_parameter.lower():
                                            rsq_y[analysis][subj][model][roi] = np.ones_like(y_par[analysis][subj][model][roi])
                                        if 'rsq' in x_parameter.lower():
                                            rsq_x[analysis][subj][model][roi] = np.ones_like(x_par[analysis][subj][model][roi])      



                                        #if plotting different model parameters
                                        if x_param_model != '' and x_param_model in subj_res['Processed Results'][weights]:
                                            rsq_x = np.copy(subj_res['Processed Results'][weights][x_param_model][alpha[analysis][subj][model][roi]>rsq_thresh])
                                        
     
                                        #if plotting non-model stuff like receptors and noise ceiling and variance
                                        if x_parameter_toplevel != '':
                                            rsq_x[analysis][subj][model][roi] = np.ones_like(x_par[analysis][subj][model][roi])
                                            
                                        if y_parameter_toplevel != '':
                                            rsq_y[analysis][subj][model][roi] = np.ones_like(y_par[analysis][subj][model][roi])
                                        
                                        
        
                                    elif len(subjects)>1 and 'fsaverage' not in subjects:
                                        #group stats
                                        x_par_group = np.concatenate(tuple([x_par[analysis][sid][model][roi] for sid in x_par[analysis] if 'Group' not in sid and sid in self.groups_dict[group]]))
                                        
                                        y_par_group = np.concatenate(tuple([y_par[analysis][sid][model][roi] for sid in y_par[analysis] if 'Group' not in sid and sid in self.groups_dict[group]]))
                                        rsq_x_group = np.concatenate(tuple([rsq_x[analysis][sid][model][roi] for sid in rsq_x[analysis] if 'Group' not in sid and sid in self.groups_dict[group]]))
                                        rsq_y_group = np.concatenate(tuple([rsq_y[analysis][sid][model][roi] for sid in rsq_y[analysis] if 'Group' not in sid and sid in self.groups_dict[group]]))
                                       
                                        x_par[analysis][subj][model][roi] = np.copy(x_par_group)
                                        y_par[analysis][subj][model][roi] = np.copy(y_par_group)
                                        rsq_x[analysis][subj][model][roi] = np.copy(rsq_x_group)                                    
                                        rsq_y[analysis][subj][model][roi] = np.copy(rsq_y_group)

                                elif 'fsaverage' not in subjects:
                                    #mean analysis
                                    ans = [an[0] for an in analyses if 'mean an' not in an[0]]
                                    alpha[analysis][subj][model][roi] = np.hstack(tuple([alpha[an][subj][model][roi] for an in ans]))
                                    x_par[analysis][subj][model][roi] = np.hstack(tuple([x_par[an][subj][model][roi] for an in ans]))
                                    y_par[analysis][subj][model][roi] = np.hstack(tuple([y_par[an][subj][model][roi] for an in ans]))
                                    rsq_x[analysis][subj][model][roi] = np.hstack(tuple([rsq_x[an][subj][model][roi] for an in ans]))
                                    rsq_y[analysis][subj][model][roi] = np.hstack(tuple([rsq_y[an][subj][model][roi] for an in ans]))                        
                                    
                                
                                
                                
                                if zscore_xdata:
                                    x_par[analysis][subj][model][roi] = zscore(x_par[analysis][subj][model][roi])
                                if zscore_ydata:
                                    y_par[analysis][subj][model][roi] = zscore(y_par[analysis][subj][model][roi])
                                
    
                            

                        for i, roi in enumerate([r for r in rois]):# if 'all' not in r and 'combined' not in r and 'Brain' not in r]):
                            
                            if len(y_par[analysis][subj][model][roi])>10:
                                self.curr_rois_names.append(roi)

                            
                                samples_in_roi = len(y_par[analysis][subj][model][roi])
                                print(f"Samples in ROI {roi}: {samples_in_roi}")
                                
                                if i>0:
                                    bar_position += ((2+len(groups))*bar_or_violin_width)

                                if len(groups)>1:
                                    if group == groups[1]:
                                        
                                        label_position = bar_position+(0.5*bar_or_violin_width)*(len(self.only_models)-1)                     
            
                                        x_ticks.append(label_position)
                                        x_labels.append(roi.replace('custom.','').replace('HCPQ1Q6.','').replace('glasser_','')+'\n')
                                else:
                                    label_position = bar_position+(0.5*bar_or_violin_width)*(len(self.only_models)-1)                     
        
                                    x_ticks.append(label_position)
                                    x_labels.append(roi.replace('custom.','').replace('HCPQ1Q6.','').replace('glasser_','')+'\n')


                                for model in self.only_models:

                                    if len(rois)>15 or 'Group' in subj:
                                         figsize = (38, 18)
                                    else:
                                        figsize=(12, 8)



                                    if '-' in y_parameter:
                                        y_parameter_str = y_parameter.replace(' '+y_parameter.split(' ')[-1],'')
                                        y_parameter_toplevel_str = y_parameter_toplevel                                     
                                    elif '-' in y_parameter_toplevel:
                                        y_parameter_str = y_parameter
                                        y_parameter_toplevel_str = y_parameter_toplevel.replace(' '+y_parameter_toplevel.split(' ')[-1],'')
                                    else:
                                        y_parameter_str = y_parameter
                                        y_parameter_toplevel_str = y_parameter_toplevel

                                    figname = f"{analysis} {subj.split('_')[0]} Mean {y_parameter_toplevel_str} {y_parameter_str}"

                                    pl.figure(figname, figsize=figsize, frameon=True)

                                    
                                    if log_yaxis:
                                        pl.gca().set_yscale('log')
                                    
                                    if 'Eccentricity' in y_parameter or 'Size' in y_parameter:
                                        pl.ylabel(f"{subj.split('_')[0]} Mean {y_parameter_str} (°)")
                                    elif 'B/D' in y_parameter:
                                        pl.ylabel(f"{subj.split('_')[0]} Mean {y_parameter_str} (% signal change)")
                                    elif y_parameter_toplevel != '' and 'Receptor' in y_parameter_toplevel:
                                        pl.ylabel(f"{subj.split('_')[0]} Mean {y_parameter_str} (pmol/ml)")                                        
                                    else:
                                        pl.ylabel(f"{subj.split('_')[0]} Mean {y_parameter_str}")
                                        
                                    if 'Mean' in xlim:
                                        pl.xlim(xlim['Mean'])
                                        
                                    if 'Mean' in ylim:
                                        pl.ylim(ylim['Mean'])
                                        
                                    full_roi_stats = weightstats.DescrStatsW(y_par[analysis][subj][model][roi],
                                                                weights=rsq_y[analysis][subj][model][roi])
                                    
                                    bar_height = full_roi_stats.mean
                                    
                                    if zconfint_err_alpha != None:                                    
                                        bar_err = (np.abs(full_roi_stats.zconfint_mean(alpha=zconfint_err_alpha) - bar_height)).reshape(2,1)*upsampling_corr_factor**0.5                                    
                                    else:
                                        bar_err = full_roi_stats.std_mean*upsampling_corr_factor**0.5


                                    #horrible stuff out here
                                    if not only_stats:
                                        self.dict_stats_text_pos[analysis][subj][y_parameter_toplevel][y_parameter][model][roi]['barheight'] = bar_height+bar_err
                                        self.dict_stats_text_pos[analysis][subj][y_parameter_toplevel][y_parameter][model][roi]['barpos'] = bar_position
                                        
    
    
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
                                                       
                                                
                                            bar_height = y_par[analysis][subj][model][roi].max()
                                        else:
                                            if 'Group' in subj and each_subj_on_group:
                                                bar_err = None

                                            if not only_stats:
                                            
                                                pl.bar(bar_position, bar_height, width=bar_or_violin_width, yerr=bar_err, 
                                                   edgecolor=model_colors[model], label=model, color=model_colors[model])
                                            
                                            if 'Group' in subj and each_subj_on_group:
                                                ssj_datapoints_x = np.linspace(bar_position-0.33*bar_or_violin_width, bar_position+0.33*bar_or_violin_width, len(subjects)-1)
                                                for ssj_nr, ssj in enumerate([s for s in subjects if 'Group' not in s[0] and s[0] in self.groups_dict[group]]):
                                                    
                                                    ssj_stats = weightstats.DescrStatsW(y_par[analysis][ssj[0]][model][roi],
                                                                weights=rsq_y[analysis][ssj[0]][model][roi])
                                                    
                                                    if zconfint_err_alpha != None:
                                                        yerr_sj = (np.abs(ssj_stats.zconfint_mean(alpha=zconfint_err_alpha) - ssj_stats.mean)).reshape(2,1)*upsampling_corr_factor**0.5
                                                    else:
                                                        yerr_sj = ssj_stats.std_mean*upsampling_corr_factor**0.5

                                                    if not only_stats:
                                                       
                                                        pl.errorbar(ssj_datapoints_x[ssj_nr], ssj_stats.mean,
                                                        yerr=yerr_sj, alpha=np.nanmean(rsq_y[analysis][ssj[0]][model][roi]),
                                                        fmt='s',  mec='k', color=model_colors[model], ecolor='k')
                                        
    
                                                                               
                                        
                                        if stats_on_plot:
                                            if diff_gauss:
                                                base_model = 'Gauss'
                                            elif diff_norm:
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
                                                    
                                                    
                                                    diff = y_par[analysis][subj][mod][roi] - y_par[analysis][subj][base_model][roi]
                                                    
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
                                                            pvals.append(null_distrib.mean() >= observ.mean())
                                                        elif diff_norm:
                                                            #test whether norm improves over other models
                                                            pvals.append(null_distrib.mean() <= observ.mean())
                                                            
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
                                                        elif diff_norm:
                                                            text_color = model_colors[base_model]
                                                        pval_str+="*"
                                                        if pval<1e-4:
                                                            pval_str+="*"
                                                            if pval<1e-6:
                                                                pval_str+="*"
                                                    elif pval>0.99:
                                                        if diff_gauss:
                                                            text_color = model_colors[base_model]        
                                                        elif diff_norm:
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
                                                        #used to be bar_position-3*bar_or_violin_width
                                                        lx, ly = bar_position-bar_or_violin_width, text_height+3*text_distance  
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
                                                
                                                                               
                                                
                                                
                                            except:
                                                pass
                                                        
                                        else:
                                            if 'Group' in subj and each_subj_on_group:
                                                bar_err = None

                                            if not only_stats:
                                                
                                                pl.bar(bar_position, bar_height, width=bar_or_violin_width, yerr=bar_err, 
                                                   edgecolor=cmap_rois[i], color=cmap_rois[i])
                                            
                                            if i == (len(rois)-1):
                                                if not only_stats:
                                                    pl.plot(np.linspace(-bar_or_violin_width,bar_position+bar_or_violin_width,10),np.zeros(10),color='k',ls='--',alpha=1,lw=1)
                                            
                                            if 'Group' in subj and each_subj_on_group:

                                                ssj_datapoints_x = np.linspace(bar_position-0.3*bar_or_violin_width, bar_position+0.3*bar_or_violin_width, len([s for s in subjects if 'Group' not in s[0] and s[0] in self.groups_dict[group]]))



                                                for ssj_nr, ssj in enumerate([s for s in subjects if 'Group' not in s[0] and s[0] in self.groups_dict[group]]):
                                                    if len(y_par[analysis][ssj[0]][model][roi])>10:
                                                    
                                                        ssj_stats = weightstats.DescrStatsW(y_par[analysis][ssj[0]][model][roi],
                                                                    weights=rsq_y[analysis][ssj[0]][model][roi])
                                                        
                                                        ssj_group_stats_roi_means[analysis][group][model][roi].append(ssj_stats.mean)
                                                        ssj_group_stats_roi_weights[analysis][group][model][roi].append(rsq_y[analysis][ssj[0]][model][roi].mean())
                                                        
                                                        if zconfint_err_alpha != None:
                                                            yerr_sj = (np.abs(ssj_stats.zconfint_mean(alpha=zconfint_err_alpha) - ssj_stats.mean)).reshape(2,1)*upsampling_corr_factor**0.5
                                                        else:
                                                            yerr_sj = ssj_stats.std_mean*upsampling_corr_factor**0.5

                                                        ssj_group_stats_roi_errs[analysis][group][model][roi].append(yerr_sj)

                                                        if not only_stats:
                                                            
                                                            pl.errorbar(ssj_datapoints_x[ssj_nr], ssj_group_stats_roi_means[analysis][group][model][roi][ssj_nr],
                                                            yerr=ssj_group_stats_roi_errs[analysis][group][model][roi][ssj_nr],  alpha=np.nanmax((0,np.nanmean(rsq_y[analysis][ssj[0]][model][roi]))),
                                                            fmt='s',  mec='k', color=cmap_rois[i], ecolor='k')    

                                                        #if i == 0:
                                                        if  ssj_group_stats_roi_means[analysis][group][model][roi][ssj_nr]<0:
                                                            vanch_sj = 'top'
                                                        else:
                                                            vanch_sj = 'bottom'

                                                        #sj number on plot
                                                        #if space == 'fsnative'
                                                        sj_number_on_plot = True
                                                        if sj_number_on_plot:
                                                            if not only_stats:
                                                                pl.text(ssj_datapoints_x[ssj_nr], ssj_group_stats_roi_means[analysis][group][model][roi][ssj_nr], int(ssj[0].split('_')[0].split('-')[1]), fontsize=12, color='k', ha='center', va=vanch_sj,
                                                                    alpha=np.nanmax((0,np.nanmean(rsq_y[analysis][ssj[0]][model][roi]))))      

                                                        plot_lines = True
                                                        if plot_lines:
                                                            dict_lines[analysis][ssj[0].split('_')[0]][model][roi]['xpos'].append(ssj_datapoints_x[ssj_nr])      
                                                            dict_lines[analysis][ssj[0].split('_')[0]][model][roi]['ypos'].append(ssj_group_stats_roi_means[analysis][group][model][roi][ssj_nr])

                                                            

                                                            if group == groups[-1]:
                                                                if not only_stats:
                                                                    pl.plot(dict_lines[analysis][ssj[0].split('_')[0]][model][roi]['xpos'],dict_lines[analysis][ssj[0].split('_')[0]][model][roi]['ypos'],c='k',
                                                                        alpha=np.mean([np.nanmax((0,np.nanmean(rsq_y[analysis][s[0]][model][roi]))) for s in subjects if ssj[0].split('_')[0] in s[0]]))

                                        if stats_on_plot:
                                            #do stats once when subj is in last group

                                            #if 'Group' in subj and 'placebo' not in subj:   #group == groups[-1]:    
                                            
                                            if 'Group' in subj:
                                                diff = np.array(ssj_group_stats_roi_means[analysis][group][model][roi])
                                                diff_weights = np.array(ssj_group_stats_roi_weights[analysis][group][model][roi])
                                            else:
                                                diff = y_par[analysis][subj][model][roi]
                                                diff_weights = rsq_y[analysis][subj][model][roi]

                                            pvals = []
                                                
                                            for ccc in range(100000):                                                                                                                                 
                                                
                                                #correct for upsampling
                                                #ideally add rsq weighting, increase asterisks font size,

                                                if 'Group' in subj:
                                                    samp_idx = np.arange(len(diff))
                                                    
                                                else: 
                                                    samp_idx = np.random.randint(0, len(diff), int(len(diff)/upsampling_corr_factor))

                                                observ = diff[samp_idx]
                                                observ_weights = diff_weights[samp_idx]
                                                
                                                signs = np.sign(np.random.rand(len(observ))-0.5)

                                                null_distrib = signs*observ

                                                null_mean = (null_distrib * observ_weights).sum()/(observ_weights.sum())
                                                observ_mean = (observ * observ_weights).sum()/(observ_weights.sum())
                                                                                        
                                                pvals.append(np.abs(null_mean) >= np.abs(observ_mean))

                                                        
                                                    
                                            pval = np.mean(pvals) 
                                            print(f'roi {pval}')
                                            
                                            pval_str = ""
                                            
                                            #compute p values
                                            if pval<0.01:
                                                pval_str+="*"
                                                if pval<1e-3:
                                                    pval_str+="*"
                                                    if pval<1e-4:
                                                        pval_str+="*"
                                    
                                            if pval_str != '':
                                                if not only_stats:
                                                    if bar_height>0:
                                                        roi_pval_str = f"{roi.replace('custom.','').replace('HCPQ1Q6.','').replace('glasser_','')}\n{pval_str}"
                                                        vanch = 'bottom'
                                                    else:
                                                        roi_pval_str = f"{pval_str}\n{roi.replace('custom.','').replace('HCPQ1Q6.','').replace('glasser_','')}"
                                                        vanch = 'top'
                                                else:
                                                    roi_pval_str = f"{pval_str}"
                                                    vanch = 'bottom'                                                    

                                                text_x_pos = bar_position
                                                
                                                if bar_err != None:
                                                    text_y_pos = bar_height+(bar_err*np.sign(bar_height))
                                                else:
                                                    if each_subj_on_group:
                                                        text_y_pos = np.sign(bar_height)*np.max(np.abs(ssj_group_stats_roi_means[analysis][group][model][roi])) + np.sign(bar_height)*ssj_group_stats_roi_errs[analysis][group][model][roi][np.argmax(np.abs(ssj_group_stats_roi_means[analysis][group][model][roi]))]
                                                    else:
                                                        text_y_pos = bar_height
                                             
                                                if only_stats:
                                                    text_x_pos = self.dict_stats_text_pos[analysis][subj][y_parameter_toplevel_str][y_parameter_str][model][roi]['barpos']
                                                    text_y_pos = self.dict_stats_text_pos[analysis][subj][y_parameter_toplevel_str][y_parameter_str][model][roi]['barheight']

                                                if 'Group' in subj:
                                                    text_y_pos = 0
                                                    vanch = 'top'
                                                    fontsize=32
                                                else:
                                                    fontsize=16


                                                pl.text(text_x_pos, text_y_pos, roi_pval_str,
                                                    fontsize=fontsize, color='k', weight = 'bold', ha='center', va=vanch)
                                                                    
                                    #before was 0.4* bar_or_violin_width
                                    if len(self.only_models)>1:
                                        bar_position += (bar_or_violin_width)
                                    else:
                                        bar_position += (0.4*bar_or_violin_width)

                                    if not only_stats:
                                    
                                        if len(groups)>1:
                                            if group == groups[1]:
                                                pl.xticks(x_ticks,x_labels, rotation=90, ha='left')
                                        else:
                                            pl.xticks(x_ticks,x_labels, rotation=90, ha='left')
                                    
                                    # handles, labels = pl.gca().get_legend_handles_labels()
                                    # by_label = dict(zip(labels, handles))
                                    # if len(self.only_models) == 1:
                                    #     pl.legend(by_label.values(), by_label.keys())
                                        
                                    if save_figures:

                                        pl.savefig(opj(figure_path, f"{subj.split('_')[0]} {model} Mean {y_parameter_toplevel_str} {y_parameter_str.replace('/','')}.pdf"), dpi=600, bbox_inches='tight')

                                                              

                        
                        if not means_only:
                            for i, roi in enumerate(rois):
                                if len(y_par[analysis][subj][model][roi])>10:
                                    ###################
                                    #x vs y param by ROI
                                    pl.figure(f"{analysis} {subj} {roi.replace('custom.','').replace('HCPQ1Q6.','').replace('glasser_','')} {y_parameter} VS {x_parameter}", figsize=(8, 8), frameon=True)
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
    
                                            #set the weighting for regression and for plot alpha depending on the situation
                                            
                                            if np.allclose(rsq_y[analysis][subj][model][roi], rsq_x[analysis][subj][model][roi]):
                                                
                                                rsq_regr_weight[analysis][subj][model][roi] = rsq_y[analysis][subj][model][roi]
                                                if rsq_alpha_plot:
                                                    rsq_alpha_plots_all_rois = np.nan_to_num(np.array([np.nanmean(rsq_y[analysis][subj][model][r]) for r in rois if 'all' not in r and 'combined' not in r and 'Brain' not in r and len(y_par[analysis][subj][model][r])>10]))
    
    
                                            else:
    
                                                if y_parameter_toplevel != '':
                                                    rsq_regr_weight[analysis][subj][model][roi] = rsq_x[analysis][subj][model][roi]
                                                    
                                                    if rsq_alpha_plot:
                                                        rsq_alpha_plots_all_rois = np.nan_to_num(np.array([np.nanmean(rsq_x[analysis][subj][model][r]) for r in rois if 'all' not in r and 'combined' not in r and 'Brain' not in r and len(y_par[analysis][subj][model][r])>10]))

                                                if x_parameter_toplevel != '':
                                                    rsq_regr_weight[analysis][subj][model][roi] = rsq_y[analysis][subj][model][roi]
                                                    
                                                    if rsq_alpha_plot:
                                                        rsq_alpha_plots_all_rois = np.nan_to_num(np.array([np.nanmean(rsq_y[analysis][subj][model][r]) for r in rois if 'all' not in r and 'combined' not in r and 'Brain' not in r and len(y_par[analysis][subj][model][r])>10]))
    
                                                elif x_param_model != '':
                                                    rsq_regr_weight[analysis][subj][model][roi] = (rsq_y[analysis][subj][model][roi]+rsq_x[analysis][subj][model][roi])/2
                                                    if rsq_alpha_plot:
                                                        rsq_alpha_plots_all_rois = np.nan_to_num(np.array([np.nanmean((rsq_y[analysis][subj][model][roi]+rsq_x[analysis][subj][model][roi])/2) for r in rois if 'all' not in r and 'combined' not in r and 'Brain' not in r and len(y_par[analysis][subj][model][r])>10]))
                                                    
                                            
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
                                                                                                      weights=rsq_y[analysis][subj][model][roi][x_par_quantile]))
                        
                                                x_par_stats[analysis][subj][model][roi].append(weightstats.DescrStatsW(x_par[analysis][subj][model][roi][x_par_quantile],
                                                                                                      weights=rsq_x[analysis][subj][model][roi][x_par_quantile]))
                                                
                                                rsq_y_stats[analysis][subj][model][roi].append(weightstats.DescrStatsW(rsq_y[analysis][subj][model][roi][x_par_quantile],
                                                                                                      weights=np.ones_like(rsq_y[analysis][subj][model][roi][x_par_quantile])))
                                                rsq_x_stats[analysis][subj][model][roi].append(weightstats.DescrStatsW(rsq_x[analysis][subj][model][roi][x_par_quantile],
                                                                                                      weights=np.ones_like(rsq_x[analysis][subj][model][roi][x_par_quantile])))
    
    
                                            curr_x_bins = np.array([ss.mean for ss in x_par_stats[analysis][subj][model][roi]])
                                            curr_y_bins = np.array([ss.mean for ss in y_par_stats[analysis][subj][model][roi]])                                   
                
                                        except:
                                            pass    
        
                                        if not scatter:
                                            
                                            if len(self.only_models)>1:
                                                current_color = model_colors[model]
                                                current_label = model
                                            else:
                                                current_color = cmap_rois[i]
                                                current_label = roi.replace('custom.','').replace('HCPQ1Q6.','').replace('glasser_','')                                           
                                            if fit:
                                                try:
                                                    WLS = LinearRegression()
                                                    
    
                                                    WLS.fit(x_par[analysis][subj][model][roi].reshape(-1, 1), y_par[analysis][subj][model][roi], sample_weight=rsq_regr_weight[analysis][subj][model][roi])
        
                                                    wls_score = WLS.score(x_par[analysis][subj][model][roi].reshape(-1, 1),
                                                                          y_par[analysis][subj][model][roi].reshape(-1, 1),
                                                                        sample_weight=rsq_regr_weight[analysis][subj][model][roi].reshape(-1, 1))
                                                    
                                                    _, pval_lin = spearmanr(x_par[analysis][subj][model][roi],y_par[analysis][subj][model][roi])
                                                    
                                                    print(f"{roi} {model} WLS score {wls_score}")
                                                    
                                                    
                                                    if exp_fit:
                                                        #start points for sigmoid fits
                                                        x0s = [[1,1,0.5,1],[-1,1,0.5,1],[-10,10,1,10],[10,10,1,10],[1,100,0.5,1],[-1,100,0.5,1],[-5,20,0.28,56]]
                                                        curr_min_res = np.inf
                                                        
                                                        for x0 in x0s:
                                                            try:
                                                                res = minimize(lambda x,a,y:1-r2_score(y,x[3]+x[0]/(x[1]*np.exp(-x[2]*a)+1),sample_weight=rsq_regr_weight[analysis][subj][model][roi]), x0=x0,
                                                                               args=(x_par[analysis][subj][model][roi],
                                                                                     y_par[analysis][subj][model][roi]),
                                                                               method='Powell', options={'ftol':1e-8, 'xtol':1e-8})
                                                                if res['fun']<curr_min_res:
                                                                    exp_res = deepcopy(res)
                                                                    curr_min_res = deepcopy(res['fun'])
                                                            except Exception as e:
                                                                print(e)
                                                                x0s.append(np.random.rand(4))
                                                                pass
                                                        
                                                        
                                                        
                                                        exp_pred = exp_res['x'][3]+exp_res['x'][0]/(exp_res['x'][1]*np.exp(-exp_res['x'][2]*np.linspace(curr_x_bins.min(),curr_x_bins.max(),100))+1)
                                                        #full_exp_pred = exp_res['x'][3]+exp_res['x'][0]/(exp_res['x'][1]*np.exp(-exp_res['x'][2]*x_par[analysis][subj][model][roi])+1)
                                                        
                                                        rsq_pred=1-exp_res['fun']
                                                                
                                                    for c in range(200):
                                                        
                                                        sample = np.random.randint(0, len(x_par[analysis][subj][model][roi]), int(len(x_par[analysis][subj][model][roi])/upsampling_corr_factor))
                                                        
                                                        WLS_bootstrap = LinearRegression()
                                                        WLS_bootstrap.fit(x_par[analysis][subj][model][roi][sample].reshape(-1, 1), y_par[analysis][subj][model][roi][sample], sample_weight=rsq_regr_weight[analysis][subj][model][roi][sample])
                                                        
                                                        bootstrap_fits[analysis][subj][model][roi].append(WLS_bootstrap.predict(curr_x_bins.reshape(-1, 1)))
                                                
                                                                                                                
                                                        
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
                                                if zconfint_err_alpha != None:
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
                                                    current_label = roi.replace('custom.','').replace('HCPQ1Q6.','').replace('glasser_','')
                                                    
                                                scatter_cmap = colors.LinearSegmentedColormap.from_list(
                                                    'alpha_rsq', [(0, (*colors.to_rgb(current_color),0)), (1, current_color)])
                                                    
                                                pl.scatter(x_par[analysis][subj][model][roi], y_par[analysis][subj][model][roi], marker='o', s=0.01,
                                                    label=current_label, zorder=len(rois)-i, c=rsq_regr_weight[analysis][subj][model][roi], cmap=scatter_cmap)
    
                                            except Exception as e:
                                                print(e)
                                                pass                            
                                

                                if 'Eccentricity' in x_parameter or 'Size' in x_parameter:
                                    pl.xlabel(f"{x_parameter} (°)")
                                elif 'B/D' in x_parameter:
                                    pl.xlabel(f"{x_parameter} (% signal change)")  
                                elif x_parameter_toplevel != '' and 'Receptor' in x_parameter_toplevel:
                                    pl.xlabel(f"{x_parameter} (pmol/ml)")                                                 
                                else:
                                    pl.xlabel(f"{x_parameter}")  
        
                                if x_param_model != '':
                                    pl.xlabel(f"{x_param_model} {pl.gca().get_xlabel()}")
                                
                                if 'Eccentricity' in y_parameter or 'Size' in y_parameter:
                                    pl.ylabel(f"{subj} {roi.replace('custom.','').replace('HCPQ1Q6.','').replace('glasser_','')} {y_parameter} (°)")
                                elif 'B/D' in y_parameter:
                                    pl.ylabel(f"{subj} {roi.replace('custom.','').replace('HCPQ1Q6.','').replace('glasser_','')} {y_parameter} (% signal change)")
                                elif y_parameter_toplevel != '' and 'Receptor' in y_parameter_toplevel:
                                    pl.ylabel(f"{subj} {roi.replace('custom.','').replace('HCPQ1Q6.','').replace('glasser_','')} {y_parameter} (pmol/ml)")                                                                   
                                else:
                                    pl.ylabel(f"{subj} {roi.replace('custom.','').replace('HCPQ1Q6.','').replace('glasser_','')} {y_parameter}")
                                    
                                
                                    
                                if show_legend:
                                
                                    handles, labels = pl.gca().get_legend_handles_labels()
        
                                    legend_dict = dd(list)
                                    for cc, label in enumerate(labels):
                                        legend_dict[label].append(handles[cc])
                                        
                                    for label in legend_dict:
                                        legend_dict[label] = tuple(legend_dict[label])
        
                                    pl.legend([legend_dict[label] for label in legend_dict], legend_dict.keys())  
                                    
                                if save_figures:
                                    
                                    pl.savefig(opj(figure_path, f"{subj} {roi.replace('custom.','').replace('HCPQ1Q6.','').replace('glasser_','')} {y_parameter_toplevel} {y_parameter.replace('/','')} VS {x_parameter_toplevel} {x_param_model} {x_parameter.replace('/','')}.pdf"), dpi=600, bbox_inches='tight')
                                    

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
                                    
                                all_rois_x_means = []
                                all_rois_y_means = []
                                all_rois_rsq_means = []
                                all_rois_x = []
                                all_rois_y = []
                                all_rois_rsq = []
                                
                                all_rois_alpha = []
                                    
                                for i, roi in enumerate([r for r in rois if 'all' not in r and 'combined' not in r and 'Brain' not in r]):
                                    
                                    if len(y_par[analysis][subj][model][roi])>10:

                                        current_color = cmap_rois[i]
                                        current_label = roi.replace('custom.','').replace('HCPQ1Q6.','') .replace('glasser_','') 
                                        
                                        if not scatter:
                                            curr_x_bins = np.array([ss.mean for ss in x_par_stats[analysis][subj][model][roi]])
                                            curr_y_bins = np.array([ss.mean for ss in y_par_stats[analysis][subj][model][roi]])     
                                            
                                            if nr_bins == 1:
                                                all_rois_x.append(x_par[analysis][subj][model][roi])
                                                all_rois_y.append(y_par[analysis][subj][model][roi])
                                                all_rois_rsq.append(rsq_regr_weight[analysis][subj][model][roi])
                                                all_rois_x_means.append(curr_x_bins)
                                                all_rois_y_means.append(curr_y_bins)
                                                all_rois_rsq_means.append(np.nanmean(rsq_regr_weight[analysis][subj][model][roi]))
                                                
                                                #all_rois_alpha.append(np.where(alpha[analysis][subj][model][roi]>rsq_thresh)[0])
                                                all_rois_alpha.append(np.where(alpha[analysis][subj][model][roi]>rsq_thresh)[0])
                                                
                                            
                                            if fit:
                                                try:
                                                    WLS = LinearRegression()
                                                    WLS.fit(x_par[analysis][subj][model][roi].reshape(-1, 1), y_par[analysis][subj][model][roi], sample_weight=rsq_regr_weight[analysis][subj][model][roi])
                                                    
               
                                                    pl.plot(curr_x_bins,
                                                        WLS.predict(curr_x_bins.reshape(-1, 1)),
                                                        color=current_color, label=current_label)
                                                        #color=roi_colors[roi]) 
        
                                                                                      
                                                    # wls_score = WLS.score(curr_x_bins.reshape(-1, 1),
                                                    #                       curr_y_bins.reshape(-1, 1),
                                                    #                     sample_weight=np.array([ss.mean for ss in rsq_stats[analysis][subj][model][roi]]).reshape(-1, 1))
                                                    # print(f"{roi} {model} WLS score {wls_score}")
                                                except Exception as e:
                                                    print(e)
                                                    pass
                                                
                                                try:
                                                    #conf interval shading
                                                    pl.fill_between(curr_x_bins,
                                                                    np.min(bootstrap_fits[analysis][subj][model][roi], axis=0),
                                                                    np.max(bootstrap_fits[analysis][subj][model][roi], axis=0),
                                                                    alpha=0.2, color=current_color, label=current_label)
                                                    
                                                except Exception as e:
                                                    print(e)
                                                    pass
    
                                            #data points with errors
                                            if zconfint_err_alpha != None:
                                                curr_yerr = np.array([np.abs(ss.zconfint_mean(alpha=zconfint_err_alpha)-ss.mean) for ss in y_par_stats[analysis][subj][model][roi]]).T*upsampling_corr_factor**0.5
                                                curr_xerr = np.array([np.abs(ss.zconfint_mean(alpha=zconfint_err_alpha)-ss.mean) for ss in x_par_stats[analysis][subj][model][roi]]).T*upsampling_corr_factor**0.5
                                            else:
                                                curr_yerr = np.array([ss.std_mean for ss in y_par_stats[analysis][subj][model][roi]])*upsampling_corr_factor**0.5
                                                curr_xerr = np.array([ss.std_mean for ss in x_par_stats[analysis][subj][model][roi]])*upsampling_corr_factor**0.5
                                            
                                            if rsq_alpha_plot:
    
                                                rsq_alpha_plot_max = np.nanmax(rsq_alpha_plots_all_rois)
                                                rsq_alpha_plot_min = np.nanmin(rsq_alpha_plots_all_rois)   
                                                                                          
                                                alpha_plot = (np.nanmean(rsq_regr_weight[analysis][subj][model][roi])-rsq_alpha_plot_min)/(rsq_alpha_plot_max-rsq_alpha_plot_min)#np.nanmean(rsq_regr_weight[analysis][subj][model][roi])/rsq_alpha_plot_max#(1-rsq_alpha_plot_min)*(np.nanmean(rsq_regr_weight[analysis][subj][model][roi])-rsq_alpha_plot_min)/(rsq_alpha_plot_max-rsq_alpha_plot_min) + rsq_alpha_plot_min
                                                
                                            else:
                                                alpha_plot = 1
                                            #print(alpha_plot)
                                            alpha_plot = np.clip(alpha_plot,0,1)
                                            if np.isnan(alpha_plot) or not np.isfinite(alpha_plot):
                                                alpha_plot = 0
    
                                            plot_hexbins = False
                                            
                                            if plot_hexbins:
                                                roi_fill_color = 'k'
                                                roi_border_color = 'w'
                                            else:
                                                roi_fill_color = current_color
                                                roi_border_color = 'k'

                                            if nr_bins == 1:
                                                ms=16
                                                mew=2
                                            else:
                                                ms=5
                                                mew=1

                                                
                                            pl.errorbar(curr_x_bins,
                                                curr_y_bins,
                                                yerr=0,#curr_yerr,
                                                xerr=0,#curr_xerr,
                                            fmt='s ', mec=roi_border_color, color=roi_fill_color, label=current_label, alpha=alpha_plot, ms=ms, mew=mew)#color=current_color, ###mfc=roi_colors[roi], ecolor=roi_colors[roi])
                                            
                                            if rois_on_plot: 
                                                roi_name_txt = pl.text(curr_x_bins[-1], curr_y_bins[-1], current_label, fontsize=25, alpha=alpha_plot, color=roi_fill_color,  ha='left', va='bottom') #color=current_color,
                                                roi_name_txt.set_path_effects([peff.withStroke(linewidth=1, foreground=roi_border_color)])
                        
                                        
                                        else:
                                            
                                            try:
        
                                                
                                                scatter_cmap = colors.LinearSegmentedColormap.from_list(
                                                    'alpha_rsq', [(0, (*colors.to_rgb(current_color),0)), (1, current_color)])
                                                    
                                                pl.scatter(x_par[analysis][subj][model][roi], y_par[analysis][subj][model][roi], marker='o', s=0.01,
                                                    label=current_label, zorder=len(rois)-i, c=rsq_x[analysis][subj][model][roi], cmap=scatter_cmap)
        
                                            except Exception as e:
                                                print(e)
                                                pass       
 
      

                                if nr_bins == 1:
                                    
                                    all_rois_rsq_means = np.array(all_rois_rsq_means)
                                    all_rois_x_means = np.concatenate(all_rois_x_means)
                                    all_rois_y_means = np.concatenate(all_rois_y_means)
                                    all_rois_rsq = np.concatenate(all_rois_rsq)
                                    all_rois_x = np.concatenate(all_rois_x)
                                    all_rois_y = np.concatenate(all_rois_y)
                                    
                                    all_rois_alpha = np.concatenate(all_rois_alpha)
                                    
                                    
                                    normalize_rsq = False
                                    use_roi_means = True
                                    #NOTE if doing full_data: set plot hexbins to true. has been moved above to manage roi datapoints color
                                    use_full_data = False
                                    perm_yw = False
                                    
                                    n_perm = 1000000
                                    
                                    #np.save('/Users/marcoaqil/full_quantrois.npy', all_rois_alpha)
                                    #np.save(f'/Users/marcoaqil/{x_parameter}_alpha.npy', all_rois_alpha)
                                    #np.save(f'/Users/marcoaqil/{x_parameter}_quant.npy', all_rois_x)
                                    
                                    #an attempt to use a normalized r2 value as weight. correlations seems slightly stronger; pvals unaffected
                                    if normalize_rsq:                                        
                                        all_rois_rsq_means = (all_rois_rsq_means - all_rois_rsq_means.min()) / (all_rois_rsq_means.max() - all_rois_rsq_means.min())                                    
                                        all_rois_rsq = (all_rois_rsq - all_rois_rsq.min()) / (all_rois_rsq.max() - all_rois_rsq.min())
                                    
                                    
                                    if use_roi_means:
                                        CC_w = weightstats.DescrStatsW(np.stack((all_rois_x_means,all_rois_y_means)).T, weights=all_rois_rsq_means).corrcoef[0,1]
                                    if use_full_data:
                                        CC_w_full_data = weightstats.DescrStatsW(np.stack((all_rois_x,all_rois_y)).T, weights=all_rois_rsq).corrcoef[0,1]
                                        
                                        eigvecs_reduced, ft_x = reduced_graph_ft(all_rois_x, all_rois_alpha, 
                                                                                 eigenvectors_path='/Users/marcoaqil/1000eigvecs_full_quantrois.npy', 
                                                                                 eigenvectors_indices_path='/Users/marcoaqil/full_quantrois.npy', 
                                                                                 pycortex_subj=pycortex_subj)
                                        data_max_x = all_rois_x.max()
                                        data_min_x = all_rois_x.min()
                                        
                                        if perm_yw:
                                            _, ft_y = reduced_graph_ft(all_rois_y, all_rois_alpha, 
                                                                                     eigenvectors_path='/Users/marcoaqil/1000eigvecs_full_quantrois.npy', 
                                                                                     eigenvectors_indices_path='/Users/marcoaqil/full_quantrois.npy', 
                                                                                     pycortex_subj=pycortex_subj)
                                            data_max_y = all_rois_y.max()
                                            data_min_y = all_rois_y.min()   
                                            
                                            _, ft_w = reduced_graph_ft(all_rois_rsq, all_rois_alpha, 
                                                                                     eigenvectors_path='/Users/marcoaqil/1000eigvecs_full_quantrois.npy', 
                                                                                     eigenvectors_indices_path='/Users/marcoaqil/full_quantrois.npy', 
                                                                                     pycortex_subj=pycortex_subj)
                                            data_max_w = all_rois_rsq.max()
                                            data_min_w = all_rois_rsq.min()                                        
                                             
                                    
                                    CC_boot = []
                                    CC_boot_full_data = []
                                    
                                    for perm in tqdm(range(n_perm)):                                        
                                        if use_roi_means:
                                            #samp_idx = np.random.randint(0, len(all_rois_x_means), len(all_rois_x_means))
                                            perm_x = np.copy(all_rois_x_means)#[samp_idx]
                                            perm_y = np.copy(all_rois_y_means)#[samp_idx]
                                            perm_w = np.copy(all_rois_rsq_means)#[samp_idx] 
                                            
                                            np.random.shuffle(perm_x)
                                            if perm_yw:
                                                np.random.shuffle(perm_y)
                                                np.random.shuffle(perm_w)
                                            
                                            CC_boot.append(weightstats.DescrStatsW(np.stack((perm_x,perm_y)).T, weights=perm_w).corrcoef[0,1])
                                            
                                        if use_full_data:
                                            #samp_idx = np.random.randint(0, len(all_rois_x), int(len(all_rois_x)/upsampling_corr_factor))                                        
                                            # perm_x = all_rois_x[samp_idx]
                                            perm_y_full_data = np.copy(all_rois_y)#[samp_idx]
                                            perm_w_full_data = np.copy(all_rois_rsq)#[samp_idx]      
                                            
                                            perm_x_full_data = graph_randomization(data_max_x, data_min_x, eigvecs_reduced, ft_x)#[samp_idx]
                                            
                                            if perm_yw:
                                                perm_y_full_data = graph_randomization(data_max_y, data_min_y, eigvecs_reduced, ft_y)#[samp_idx]
                                                perm_w_full_data = graph_randomization(data_max_w, data_min_w, eigvecs_reduced, ft_w)#[samp_idx]
                                                
                                            CC_boot_full_data.append(weightstats.DescrStatsW(np.stack((perm_x_full_data,perm_y_full_data)).T, 
                                                                                             weights=perm_w_full_data).corrcoef[0,1])
                                            
                                            
                                    
                                    title_string = ""                                        
                                    if use_roi_means:
                                        print(f"null mean roi means CC (should be close to zero) {np.mean(CC_boot)}")
                                        pval_wcc = np.sum(np.abs(CC_w)<np.abs(np.array(CC_boot)))/len(CC_boot)
                                        
                                        if n_perm >= 100000:
                                            if pval_wcc<1e-2:
                                                pval_string = '*'
                                                if pval_wcc<1e-3:
                                                    pval_string = '**'
                                                    if pval_wcc<1e-4:
                                                        pval_string = '***'                                                                                        
                                            else:
                                                pval_string = 'n.s.'
                                        else:
                                            pval_string = ""
                                        
                                        title_string += f"wCC={CC_w:.2f} p={pval_wcc:.2e} ({pval_string})"
                                    
                                    if use_full_data:
                                        if use_roi_means:
                                            title_string += "; "
                                            
                                        print(f"null mean full data CC (should be close to zero) {np.mean(CC_boot_full_data)}")
                                        pval_wcc_full_data = np.sum(np.abs(CC_w_full_data)<np.abs(np.array(CC_boot_full_data)))/len(CC_boot_full_data)    
                                        
                                        if n_perm >= 100000:
                                            if pval_wcc_full_data<1e-2:
                                                pval_string_full_data = '*'
                                                if pval_wcc_full_data<1e-3:
                                                    pval_string_full_data = '**'
                                                    if pval_wcc_full_data<1e-4:
                                                        pval_string_full_data = '***'                                                                                        
                                            else:
                                                pval_string_full_data = 'n.s.'
                                        else:
                                            pval_string_full_data = ""
                                            
                                        title_string += f" wCCfd={CC_w_full_data:.2f} p={pval_wcc_full_data:.2e} ({pval_string_full_data})"                                        
                                    
                                    
                                    pl.title(title_string)
                                    


                                    #new_cmap = pl.get_cmap('cmr.rainforest')
                                    new_cmap = pl.get_cmap(cmr.get_sub_cmap('Reds', 0.05, 0.9))
                                    bins_b = 'log'#None #'log'
                                    

                                    if plot_hexbins:
                                        pl.hexbin(all_rois_x, all_rois_y, all_rois_rsq, gridsize=25, alpha=0.5, linewidths=0, bins=bins_b, edgecolors='none', cmap=new_cmap)#, cmap='binary') # mincnt=10,

                                    #regression line
                                                                       
                                    
                                    if use_roi_means:
                                        WLS_comb = LinearRegression()
                                        
                                        WLS_comb.fit(all_rois_x_means.reshape(-1, 1), all_rois_y_means, sample_weight=all_rois_rsq_means)  

                                        # if use_full_data:
                                        #     WLS_comb_prediction = WLS_comb.predict(all_rois_x.reshape(-1, 1))    
                                        #     pl.plot(all_rois_x[np.argsort(all_rois_x)],
                                        #     WLS_comb_prediction[np.argsort(all_rois_x)],
                                        #     color='w')            
                                        # else:
                                        WLS_comb_prediction = WLS_comb.predict(all_rois_x_means.reshape(-1, 1))                                          
                                        pl.plot(all_rois_x_means[np.argsort(all_rois_x_means)],
                                            WLS_comb_prediction[np.argsort(all_rois_x_means)],
                                            color='k', zorder=1003)    
                                        
                                    if use_full_data:     
                                        
                                        WLS_comb_full_data = LinearRegression()
                                        
                                        WLS_comb_full_data.fit(all_rois_x.reshape(-1, 1), all_rois_y, sample_weight=all_rois_rsq)                                           
                                        WLS_comb_full_data_prediction = WLS_comb_full_data.predict(all_rois_x.reshape(-1, 1))    
                                        pl.plot(all_rois_x[np.argsort(all_rois_x)],
                                            WLS_comb_full_data_prediction[np.argsort(all_rois_x)],
                                            color='k', zorder=1000)                                         

                                 
                                    #bootstrap conf intervals
                                    comb_boot_fits = []
                                    comb_boot_fits_full_data = []
                                    for bootc in range(200):
                                        if use_roi_means:
                                            samp_idx = np.random.randint(0, len(all_rois_x_means), len(all_rois_x_means))
                                            bootc_x = all_rois_x_means[samp_idx]
                                            bootc_y = all_rois_y_means[samp_idx]
                                            bootc_w = all_rois_rsq_means[samp_idx]
                                            
                                            WLS_comb_boot = LinearRegression()
                                            WLS_comb_boot.fit(bootc_x.reshape(-1, 1), bootc_y, sample_weight=bootc_w)  
                                        if use_full_data:                                            
                                            samp_idx_full_data = np.random.randint(0, len(all_rois_x), int(len(all_rois_x)/upsampling_corr_factor))
                                            bootc_x_full_data = all_rois_x[samp_idx_full_data]
                                            bootc_y_full_data = all_rois_y[samp_idx_full_data]
                                            bootc_w_full_data = all_rois_rsq[samp_idx_full_data]    
                                            WLS_comb_boot_full_data = LinearRegression()
                                            WLS_comb_boot_full_data.fit(bootc_x_full_data.reshape(-1, 1), bootc_y_full_data, sample_weight=bootc_w_full_data)                                       

                                        
                                        if use_roi_means:
                                            comb_boot_fits.append(WLS_comb_boot.predict(all_rois_x_means.reshape(-1, 1)))
                                        if use_full_data:
                                            comb_boot_fits_full_data.append(WLS_comb_boot_full_data.predict(all_rois_x.reshape(-1, 1)))
                                    
                                    
                                    if use_roi_means:
                                        comb_boot_fits = np.array(comb_boot_fits)
                                    
                                        conf_max = np.nanquantile(comb_boot_fits,0.95,axis=0)#WLS_comb_prediction + sem(comb_boot_fits,axis=0)#
                                        conf_min = np.nanquantile(comb_boot_fits,0.05,axis=0)#WLS_comb_prediction - sem(comb_boot_fits,axis=0)#
                                    
                                    
                                        pl.plot(all_rois_x_means[np.argsort(all_rois_x_means)],conf_max[np.argsort(all_rois_x_means)], color='k', ls = '--', zorder=1004)
                                        pl.plot(all_rois_x_means[np.argsort(all_rois_x_means)],conf_min[np.argsort(all_rois_x_means)], color='k', ls = '--', zorder=1005)
                                        
                                    if use_full_data:
                                        comb_boot_fits_full_data = np.array(comb_boot_fits_full_data)
                                    
                                        conf_max_full_data = np.nanquantile(comb_boot_fits_full_data,0.95,axis=0)#WLS_comb_prediction + sem(comb_boot_fits,axis=0)#
                                        conf_min_full_data = np.nanquantile(comb_boot_fits_full_data,0.05,axis=0)#WLS_comb_prediction - sem(comb_boot_fits,axis=0)#                                        
                                        
                                        pl.plot(all_rois_x[np.argsort(all_rois_x)],conf_max_full_data[np.argsort(all_rois_x)], color='k', ls = '--', zorder=1001)
                                        pl.plot(all_rois_x[np.argsort(all_rois_x)],conf_min_full_data[np.argsort(all_rois_x)], color='k', ls = '--', zorder=1002)                                    

                                
                                if 'Eccentricity' in x_parameter or 'Size' in x_parameter:
                                    pl.xlabel(f"{x_parameter} (°)")
                                elif 'B/D' in x_parameter:
                                    pl.xlabel(f"{x_parameter} (% signal change)")   
                                elif x_parameter_toplevel != '' and 'Receptor' in x_parameter_toplevel:
                                    pl.xlabel(f"{x_parameter} (pmol/ml)")                                                 
                                else:
                                    pl.xlabel(f"{x_parameter}")
                                    
                                if x_param_model != '':
                                    pl.xlabel(f"{x_param_model} {pl.gca().get_xlabel()}")                                    
                                
                                if 'Eccentricity' in y_parameter or 'Size' in y_parameter:
                                    pl.ylabel(f"{subj} {model} {y_parameter} (°)")
                                elif 'B/D' in y_parameter:
                                    pl.ylabel(f"{subj} {model} {y_parameter} (% signal change)")  
                                elif y_parameter_toplevel != '' and 'Receptor' in y_parameter_toplevel:
                                    pl.ylabel(f"{subj} {model} {y_parameter} (pmol/ml)")                                                 
                                else:
                                    pl.ylabel(f"{subj} {model} {y_parameter}")
                                    
                                if show_legend:
                                    handles, labels = pl.gca().get_legend_handles_labels()
        
                                    legend_dict = dd(list)
                                    for cc, label in enumerate(labels):
                                        legend_dict[label].append(handles[cc])
                                        
                                    for label in legend_dict:
                                        legend_dict[label] = tuple(legend_dict[label])
        
                                    pl.legend([legend_dict[label] for label in legend_dict], legend_dict.keys())                      
    
                                if save_figures:

                                    pl.savefig(opj(figure_path, f"{subj} {model} {y_parameter_toplevel} {y_parameter.replace('/','')} VS {x_parameter_toplevel} {x_param_model} {x_parameter.replace('/','')}.pdf"), dpi=600, bbox_inches='tight')


        return
    


    def quant_plots(self, x_parameter, y_parameter, rois, rsq_thresh, save_figures, figure_path,
                    space_names = 'fsnative', analysis_names = 'all', subject_ids='all', 
                    x_parameter_toplevel='', y_parameter_toplevel='', 
                    ylim={}, xlim={}, log_yaxis=False, log_xaxis = False, nr_bins = 8, weights='RSq',
                    x_param_model='', violin=False, scatter=False, diff_norm=False, diff_gauss=False, diff_gauss_x=False,
                    rois_on_plot = False, rsq_alpha_plot = False,
                    means_only=False, stats_on_plot=False, bin_by='size', zscore_ydata=False, zscore_xdata=False,
                    zconfint_err_alpha = None, fit = True,exp_fit = False, show_legend=False, each_subj_on_group=False,
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
            DESCRIPTION. The default == None.
        x_param_model : TYPE, optional
            DESCRIPTION. The default == None.
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
        
        cmap_values = list(np.linspace(0.9, 0.0, len([r for r in rois if r not in ['Brain', 'all_custom', 'combined']])))
        
        cmap_values += [0 for r in rois  if r in ['Brain', 'all_custom', 'combined']]
        
        cmap_rois = cm.get_cmap('nipy_spectral')(cmap_values)#

        self.curr_rois_names = []
        
        #making black into dark gray for visualization reasons
        cmap_rois[(cmap_rois == [0,0,0,1]).sum(1)==4] = [0.33,0.33,0.33,1]

        if space_names == 'all':
            spaces = [item for item in self.main_dict.items()]
        else:
            spaces = [item for item in self.main_dict.items() if item[0] in space_names] 

        for space, space_res in spaces:
            if 'fs' in space or 'HCP' in space:
                
                if quantile_exclusion == None:
                    #surrounds larger than this are excluded from surround size calculations
                    w_max=60
                    #remove absurd suppression index values
                    supp_max=1000
                    #css max
                    css_max=1
                    #max b and d
                    bd_max=1000
                #bar or violin width
                if len(self.only_models)>1:
                    bar_or_violin_width = 0.3
                else:
                    bar_or_violin_width = 1.5

                
                        
                if analysis_names == 'all':
                    analyses = [item for item in space_res.items()]
                else:
                    analyses = [item for item in space_res.items() if item[0] in analysis_names] 
                    
                if len(analyses)>1:
                    analyses.append(('Analyses mean', {sub:{} for sub in analyses[0][1]}))
                    

                alpha = dd(lambda:dd(lambda:dd(lambda:dd(dict))))
                x_par = dd(lambda:dd(lambda:dd(lambda:dd(dict))))
                y_par = dd(lambda:dd(lambda:dd(lambda:dd(dict))))
                rsq_y = dd(lambda:dd(lambda:dd(lambda:dd(dict))))
                rsq_x = dd(lambda:dd(lambda:dd(lambda:dd(dict))))
                rsq_regr_weight = dd(lambda:dd(lambda:dd(lambda:dd(list))))
    
                x_par_stats = dd(lambda:dd(lambda:dd(lambda:dd(list))))
                y_par_stats = dd(lambda:dd(lambda:dd(lambda:dd(list))))
                rsq_y_stats = dd(lambda:dd(lambda:dd(lambda:dd(list))))
                rsq_x_stats = dd(lambda:dd(lambda:dd(lambda:dd(list))))
                
                bootstrap_fits = dd(lambda:dd(lambda:dd(lambda:dd(list))))

                dict_lines = dd(lambda:dd(lambda:dd(lambda:dd(list))))
                
                
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
                            if bold_voxel_volume != None:
                                
                                print("Make sure bold_voxel_volume is specified in mm^3")
                                
                                try:
                                    if subj.isdecimal() and space == 'HCP':
                                        pycortex_subj = '999999'
                                    elif 'fsaverage' in subj or 'fsaverage' in space:
                                        pycortex_subj = 'fsaverage'                                        
                                    else:
                                        pycortex_subj = subj.split('_')[0]
                                        
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
            
                        model_colors = {'Gauss':'blue','CSS':'orange','DoG':'green','Norm_abcd':'red','Norm_abc':'purple'}
                                                

    
                        for i, roi in enumerate(rois):                              
                            for model in self.only_models:                                

                                
                                if 'mean an' not in analysis or 'fsaverage' in subj:
                                    if 'sub' in subj or 'fsaverage' in subj or subj.isdecimal():
                                        if space == 'fsaverage':
                                            roi_subj = 'fsaverage'
                                        else:
                                            roi_subj = subj

                                        if 'rsq' in y_parameter.lower():
                                            #comparing same vertices for model performance
                                            curr_alpha = subj_res['Processed Results']['Alpha']['all']
                                        else:
                                            #otherwise model-specific alpha
                                            curr_alpha = (subj_res['Processed Results']['Alpha'][model])
                                            
                                        if roi in self.idx_rois[roi_subj]:
    
                                            alpha[analysis][subj][model][roi] = (roi_mask(self.idx_rois[roi_subj][roi], curr_alpha)) 
       
                                        else:
                                            #if ROI != defined
                                            #if Brain use all available vertices
                                            if roi == 'Brain':
                                                alpha[analysis][subj][model][roi] = curr_alpha
                                            elif roi == 'combined':
                                                alpha[analysis][subj][model][roi] = (roi_mask(np.concatenate(tuple([self.idx_rois[roi_subj][r] for r in rois if ('combined' not in r and 'Brain' not in r and r in self.idx_rois[roi_subj])])), curr_alpha))    
                                            elif roi == 'all_custom':
                                                alpha[analysis][subj][model][roi] = (roi_mask(np.concatenate(tuple([self.idx_rois[roi_subj][r] for r in self.idx_rois[roi_subj] if 'custom' in r])), curr_alpha))    
                                            # elif space == 'fsaverage' and roi in self.idx_rois['fsaverage']:
                                            #     alpha[analysis][subj][model][roi] = (roi_mask(self.idx_rois['fsaverage'][roi], curr_alpha))
                                            else:
                                                #, otherwise none
                                                print(f"{roi}: undefined ROI")
                                                alpha[analysis][subj][model][roi] = np.zeros_like(curr_alpha).astype('bool')
                                        
                                        
                                        #manual exclusion of outliers
                                        if quantile_exclusion == None:
                                            print("Using manual exclusion, see quant_plots function. Set quantile_exclusion=1 for no exclusion.")
                                            if y_parameter == 'Surround Size (fwatmin)' or x_parameter == 'Surround Size (fwatmin)':# and model == 'DoG':
                                                #exclude too large surround (no surround)
                                                alpha[analysis][subj][model][roi] *= (subj_res['Processed Results']['Surround Size (fwatmin)'][model]<w_max)
                                                if x_param_model != '':
                                                    alpha[analysis][subj][model][roi] *= (subj_res['Processed Results']['Surround Size (fwatmin)'][x_param_model]<w_max)
    
                                            
                                            if y_parameter == 'Surround/Centre Amplitude'  or x_parameter == 'Surround/Centre Amplitude' :# and model == 'DoG':
                                                #exclude too large surround (no surround)
                                                if x_param_model != '':
                                                    alpha[analysis][subj][model][roi] *= (subj_res['Processed Results']['Surround/Centre Amplitude'][x_param_model]<w_max)
                                                else:
                                                    alpha[analysis][subj][model][roi] *= (subj_res['Processed Results']['Surround/Centre Amplitude'][model]<w_max)
    
                                                    
                                            if 'Suppression' in y_parameter:
                                                #exclude nonsensical suppression index values
                                                alpha[analysis][subj][model][roi] *= (subj_res['Processed Results'][y_parameter][model]<supp_max)                                            
                                            if 'Suppression' in x_parameter:
                                                #exclude nonsensical suppression index values
                                                alpha[analysis][subj][model][roi] *= (subj_res['Processed Results'][x_parameter][model]<supp_max)
                                                if x_param_model != '':
                                                    alpha[analysis][subj][model][roi] *= (subj_res['Processed Results'][x_parameter][x_param_model]<supp_max)
                                                    
                                            if 'CSS Exponent' in x_parameter:
                                                alpha[analysis][subj][model][roi] *= (subj_res['Processed Results'][x_parameter][x_param_model]<css_max)
                                            
                                            if 'Norm Param.' in y_parameter:
                                                alpha[analysis][subj][model][roi] *= (subj_res['Processed Results'][y_parameter][model]<bd_max)                                            
                                            if 'Norm Param.' in x_parameter:
                                                alpha[analysis][subj][model][roi] *= (subj_res['Processed Results'][x_parameter][model]<bd_max)                                            
                                                if x_param_model != '':
                                                    alpha[analysis][subj][model][roi] *= (subj_res['Processed Results'][x_parameter][x_param_model]<bd_max)
                                        
                                        else:
                                            if y_parameter_toplevel == '':    
                                                alpha[analysis][subj][model][roi] *= (subj_res['Processed Results'][y_parameter][model]<np.nanquantile(subj_res['Processed Results'][y_parameter][model],quantile_exclusion))*(subj_res['Processed Results'][y_parameter][model]>np.nanquantile(subj_res['Processed Results'][y_parameter][model],1-quantile_exclusion))  

                                            if x_param_model != '':
                                                #nanquantile handles nans but will not work if there are infs
                                                alpha[analysis][subj][model][roi] *= (subj_res['Processed Results'][x_parameter][x_param_model]<np.nanquantile(subj_res['Processed Results'][x_parameter][x_param_model],quantile_exclusion))*(subj_res['Processed Results'][x_parameter][x_param_model]>np.nanquantile(subj_res['Processed Results'][x_parameter][x_param_model],1-quantile_exclusion))  

                                            if x_parameter_toplevel == '':
                                                alpha[analysis][subj][model][roi] *= (subj_res['Processed Results'][x_parameter][model]<np.nanquantile(subj_res['Processed Results'][x_parameter][model],quantile_exclusion))*(subj_res['Processed Results'][x_parameter][model]>np.nanquantile(subj_res['Processed Results'][x_parameter][model],1-quantile_exclusion))  

                                            
                                            
         
                                        #if 'ccrsq' in y_parameter.lower():
                                        #    alpha[analysis][subj][model][roi] *= (subj_res['Processed Results'][y_parameter][model]>0)
                                        if y_parameter_toplevel == '':
                                            alpha[analysis][subj][model][roi] *= np.isfinite(subj_res['Processed Results'][y_parameter][model])
                                        else:
                                            alpha[analysis][subj][model][roi] *= np.isfinite(subj_res['Processed Results'][y_parameter_toplevel][y_parameter])
                                       
                                        if x_param_model != '':
                                            
                                            alpha[analysis][subj][model][roi] *= np.isfinite(subj_res['Processed Results'][x_parameter][x_param_model])
                                            
                                            x_par[analysis][subj][model][roi] = subj_res['Processed Results'][x_parameter][x_param_model][alpha[analysis][subj][model][roi]>rsq_thresh]                                          
                                        
                                        
                                        elif x_parameter_toplevel != '':
                                            alpha[analysis][subj][model][roi] *= np.isfinite(subj_res['Processed Results'][x_parameter_toplevel][x_parameter])
                                            x_par[analysis][subj][model][roi] = (subj_res['Processed Results'][x_parameter_toplevel][x_parameter][alpha[analysis][subj][model][roi]>rsq_thresh])
                                            
                                        else:
                                            #remove nans and infinities
                                            alpha[analysis][subj][model][roi] *= np.isfinite(subj_res['Processed Results'][x_parameter][model])                                    

                                            x_par[analysis][subj][model][roi] = subj_res['Processed Results'][x_parameter][model][alpha[analysis][subj][model][roi]>rsq_thresh]
                                            
                                        #handling special case of plotting receptors as y-parameter, since they are not part of any model
                                        if y_parameter_toplevel == '':
                                            y_par[analysis][subj][model][roi] = (subj_res['Processed Results'][y_parameter][model][alpha[analysis][subj][model][roi]>rsq_thresh])
                                        else:
                                            y_par[analysis][subj][model][roi] = (subj_res['Processed Results'][y_parameter_toplevel][y_parameter][alpha[analysis][subj][model][roi]>rsq_thresh])
                                            
                                        
                                        
                                        if diff_gauss:
                                            y_par[analysis][subj][model][roi] -= subj_res['Processed Results'][y_parameter]['Gauss'][alpha[analysis][subj][model][roi]>rsq_thresh]
                                        
                                        if diff_gauss_x:
                                            x_par[analysis][subj][model][roi] -= subj_res['Processed Results'][x_parameter]['Gauss'][alpha[analysis][subj][model][roi]>rsq_thresh]
                                        
                                            

                                            #set negative ccrsq and rsq to zero
                                            #if 'ccrsq' in y_parameter.lower():
                                            #    y_par[analysis][subj][model][roi][y_par[analysis][subj][model][roi]<0] = 0
                 
    
                                        #r - squared weighting
                                        if weights != None:
                                            rsq_x[analysis][subj][model][roi] = np.copy(subj_res['Processed Results'][weights][model][alpha[analysis][subj][model][roi]>rsq_thresh])
                                            rsq_y[analysis][subj][model][roi] = np.copy(subj_res['Processed Results'][weights][model][alpha[analysis][subj][model][roi]>rsq_thresh])
                                        else:
                                            rsq_x[analysis][subj][model][roi] = np.ones_like(x_par[analysis][subj][model][roi])    
                                            rsq_y[analysis][subj][model][roi] = np.ones_like(y_par[analysis][subj][model][roi])

                                        #no need for rsq-weighting if plotting rsq
                                        if 'rsq' in y_parameter.lower():
                                            rsq_y[analysis][subj][model][roi] = np.ones_like(y_par[analysis][subj][model][roi])
                                        if 'rsq' in x_parameter.lower():
                                            rsq_x[analysis][subj][model][roi] = np.ones_like(x_par[analysis][subj][model][roi])      



                                        #if plotting different model parameters
                                        if x_param_model != '' and x_param_model in subj_res['Processed Results'][weights]:
                                            rsq_x = np.copy(subj_res['Processed Results'][weights][x_param_model][alpha[analysis][subj][model][roi]>rsq_thresh])
                                        
     
                                        #if plotting non-model stuff like receptors and noise ceiling and variance
                                        if x_parameter_toplevel != '':
                                            rsq_x[analysis][subj][model][roi] = np.ones_like(x_par[analysis][subj][model][roi])
                                            
                                        if y_parameter_toplevel != '':
                                            rsq_y[analysis][subj][model][roi] = np.ones_like(y_par[analysis][subj][model][roi])
                                        
                                        
        
                                    elif len(subjects)>1 and 'fsaverage' not in subjects:
                                        #group stats
                                        x_par_group = np.concatenate(tuple([x_par[analysis][sid][model][roi] for sid in x_par[analysis] if 'Group' not in sid]))
                                        
                                        y_par_group = np.concatenate(tuple([y_par[analysis][sid][model][roi] for sid in y_par[analysis] if 'Group' not in sid]))
                                        rsq_x_group = np.concatenate(tuple([rsq_x[analysis][sid][model][roi] for sid in rsq_x[analysis] if 'Group' not in sid]))
                                        rsq_y_group = np.concatenate(tuple([rsq_y[analysis][sid][model][roi] for sid in rsq_y[analysis] if 'Group' not in sid]))
                                       
                                        x_par[analysis][subj][model][roi] = np.copy(x_par_group)
                                        y_par[analysis][subj][model][roi] = np.copy(y_par_group)
                                        rsq_x[analysis][subj][model][roi] = np.copy(rsq_x_group)                                    
                                        rsq_y[analysis][subj][model][roi] = np.copy(rsq_y_group)

                                elif 'fsaverage' not in subjects:
                                    #mean analysis
                                    ans = [an[0] for an in analyses if 'mean an' not in an[0]]
                                    alpha[analysis][subj][model][roi] = np.hstack(tuple([alpha[an][subj][model][roi] for an in ans]))
                                    x_par[analysis][subj][model][roi] = np.hstack(tuple([x_par[an][subj][model][roi] for an in ans]))
                                    y_par[analysis][subj][model][roi] = np.hstack(tuple([y_par[an][subj][model][roi] for an in ans]))
                                    rsq_x[analysis][subj][model][roi] = np.hstack(tuple([rsq_x[an][subj][model][roi] for an in ans]))
                                    rsq_y[analysis][subj][model][roi] = np.hstack(tuple([rsq_y[an][subj][model][roi] for an in ans]))                        
                                    
                                
                                
                                
                                if zscore_xdata:
                                    x_par[analysis][subj][model][roi] = zscore(x_par[analysis][subj][model][roi])
                                if zscore_ydata:
                                    y_par[analysis][subj][model][roi] = zscore(y_par[analysis][subj][model][roi])
                                
    
                            

                        for i, roi in enumerate([r for r in rois]):# if 'all' not in r and 'combined' not in r and 'Brain' not in r]):
                            
                            if len(y_par[analysis][subj][model][roi])>10:
                                self.curr_rois_names.append(roi)

                            
                                samples_in_roi = len(y_par[analysis][subj][model][roi])
                                print(f"Samples in ROI {roi}: {samples_in_roi}")
                                
                                if i>0:
                                    bar_position += (2*bar_or_violin_width)
                                    
                                label_position = bar_position+(0.5*bar_or_violin_width)*(len(self.only_models)-1)                     
                                if 'hV4' in roi and len(self.only_models)>1:
                                    label_position+=bar_or_violin_width
                                    
                                x_ticks.append(label_position)
                                x_labels.append(roi.replace('custom.','').replace('HCPQ1Q6.','').replace('glasser_','')+'\n')   
                                
                                for model in self.only_models:    
                                    if len(rois)>15:  
                                        pl.figure(f"{analysis} {subj} Mean {y_parameter}", figsize=(38, 18), frameon=True)
                                    else:
                                        pl.figure(f"{analysis} {subj} Mean {y_parameter}", figsize=(8, 8), frameon=True)

                                    
                                    if log_yaxis:
                                        pl.gca().set_yscale('log')
                                    
                                    if 'Eccentricity' in y_parameter or 'Size' in y_parameter:
                                        pl.ylabel(f"{subj} Mean {y_parameter} (°)")
                                    elif 'B/D' in y_parameter:
                                        pl.ylabel(f"{subj} Mean {y_parameter} (% signal change)")
                                    elif y_parameter_toplevel != '' and 'Receptor' in y_parameter_toplevel:
                                        pl.ylabel(f"{subj} Mean {y_parameter} (pmol/ml)")                                        
                                    else:
                                        pl.ylabel(f"{subj} Mean {y_parameter}")
                                        
                                    if 'Mean' in xlim:
                                        pl.xlim(xlim['Mean'])
                                        
                                    if 'Mean' in ylim:
                                        pl.ylim(ylim['Mean'])
                                        
                                    full_roi_stats = weightstats.DescrStatsW(y_par[analysis][subj][model][roi],
                                                                weights=rsq_y[analysis][subj][model][roi])
                                    bar_height = full_roi_stats.mean
                                    
                                    if zconfint_err_alpha != None:                                    
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
                                                ssj_datapoints_x = np.linspace(bar_position-0.33*bar_or_violin_width, bar_position+0.33*bar_or_violin_width, len(subjects)-1)
                                                for ssj_nr, ssj in enumerate(subjects[:-1]):
                                                    
                                                    ssj_stats = weightstats.DescrStatsW(y_par[analysis][ssj[0]][model][roi],
                                                                weights=rsq_y[analysis][ssj[0]][model][roi])
                                                    
                                                    if zconfint_err_alpha != None:
                                                        yerr_sj = (np.abs(ssj_stats.zconfint_mean(alpha=zconfint_err_alpha) - ssj_stats.mean)).reshape(2,1)*upsampling_corr_factor**0.5
                                                    else:
                                                        yerr_sj = ssj_stats.std_mean*upsampling_corr_factor**0.5
                                                       
                                                    pl.errorbar(ssj_datapoints_x[ssj_nr], ssj_stats.mean,
                                                    yerr=yerr_sj, alpha=np.nanmean(rsq_y[analysis][ssj[0]][model][roi]),
                                                    fmt='s',  mec='k', color=model_colors[model], ecolor='k')
                                        
    
                                                                               
                                        
                                        if stats_on_plot:
                                            if diff_gauss:
                                                base_model = 'Gauss'
                                            elif diff_norm:
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
                                                    
                                                    
                                                    diff = y_par[analysis][subj][mod][roi] - y_par[analysis][subj][base_model][roi]
                                                    
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
                                                            pvals.append(null_distrib.mean() >= observ.mean())
                                                        elif diff_norm:
                                                            #test whether norm improves over other models
                                                            pvals.append(null_distrib.mean() <= observ.mean())
                                                            
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
                                                        elif diff_norm:
                                                            text_color = model_colors[base_model]
                                                        pval_str+="*"
                                                        if pval<1e-4:
                                                            pval_str+="*"
                                                            if pval<1e-6:
                                                                pval_str+="*"
                                                    elif pval>0.99:
                                                        if diff_gauss:
                                                            text_color = model_colors[base_model]        
                                                        elif diff_norm:
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
                                                        #used to be bar_position-3*bar_or_violin_width
                                                        lx, ly = bar_position-bar_or_violin_width, text_height+3*text_distance  
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
                                            
                                            if i == (len(rois)-1):
                                                pl.plot(np.linspace(-bar_or_violin_width,bar_position+bar_or_violin_width,10),np.zeros(10),color='k',ls='--',alpha=1,lw=1)
                                            
                                            if subj == 'Group' and each_subj_on_group:
                                                ssj_datapoints_x = np.linspace(bar_position-0.15*bar_or_violin_width, bar_position+0.15*bar_or_violin_width, len(subjects)-1)

                                                ssj_group_stats_roi_means = []
                                                ssj_group_stats_roi_errs = []
                                                ssj_group_stats_roi_weights = []

                                                for ssj_nr, ssj in enumerate(subjects[:-1]):
                                                    if len(y_par[analysis][ssj[0]][model][roi])>10:
                                                    
                                                        ssj_stats = weightstats.DescrStatsW(y_par[analysis][ssj[0]][model][roi],
                                                                    weights=rsq_y[analysis][ssj[0]][model][roi])
                                                        
                                                        ssj_group_stats_roi_means.append(ssj_stats.mean)
                                                        ssj_group_stats_roi_weights.append(rsq_y[analysis][ssj[0]][model][roi].mean())
                                                        
                                                        if zconfint_err_alpha != None:
                                                            yerr_sj = (np.abs(ssj_stats.zconfint_mean(alpha=zconfint_err_alpha) - ssj_stats.mean)).reshape(2,1)*upsampling_corr_factor**0.5
                                                        else:
                                                            yerr_sj = ssj_stats.std_mean*upsampling_corr_factor**0.5

                                                        ssj_group_stats_roi_errs.append(yerr_sj)
                                                            
                                                        pl.errorbar(ssj_datapoints_x[ssj_nr], ssj_stats.mean,
                                                        yerr=yerr_sj,  alpha=np.nanmax((0,np.nanmean(rsq_y[analysis][ssj[0]][model][roi]))),
                                                        fmt='s',  mec='k', color=cmap_rois[i], ecolor='k')    

                                                        #if i == 0:
                                                        if  ssj_stats.mean<0:
                                                            vanch_sj = 'top'
                                                        else:
                                                            vanch_sj = 'bottom'

                                                        #sj number on plot
                                                        #if space == 'fsnative'
                                                        sj_number_on_plot = True
                                                        if sj_number_on_plot:
                                                            pl.text(ssj_datapoints_x[ssj_nr], ssj_stats.mean, int(ssj[0].split('_')[0].split('-')[1]), fontsize=12, color='k', ha='center', va=vanch_sj)      

                                                        plot_lines = False
                                                        if plot_lines:
                                                            dict_lines[analysis][ssj[0]][model]['xpos'].append(ssj_datapoints_x[ssj_nr])      
                                                            dict_lines[analysis][ssj[0]][model]['ypos'].append(ssj_stats.mean)

                                                            if i == (len(rois)-1):
                                                                pl.plot(dict_lines[analysis][ssj[0]][model]['xpos'],dict_lines[analysis][ssj[0]][model]['ypos'],c='k',alpha=0.5)

                                        if stats_on_plot:    
                                            
                                            if subj == 'Group':
                                                diff = np.array(ssj_group_stats_roi_means)
                                                diff_weights = np.array(ssj_group_stats_roi_weights)
                                            else:
                                                diff = y_par[analysis][subj][model][roi]
                                                diff_weights = rsq_y[analysis][subj][model][roi]

                                            pvals = []
                                                
                                            for ccc in range(10000):                                                                                                                                 
                                                
                                                #correct for upsampling
                                                #ideally add rsq weighting, increase asterisks font size,

                                                if subj == 'Group':
                                                    samp_idx = np.arange(len(diff))
                                                    
                                                else: 
                                                    samp_idx = np.random.randint(0, len(diff), int(len(diff)/upsampling_corr_factor))

                                                observ = diff[samp_idx]
                                                observ_weights = diff_weights[samp_idx]
                                                
                                                signs = np.sign(np.random.rand(len(observ))-0.5)

                                                null_distrib = signs*observ

                                                null_mean = (null_distrib * observ_weights).sum()/(observ_weights.sum())
                                                observ_mean = (observ * observ_weights).sum()/(observ_weights.sum())
                                                                                           
                                                pvals.append(np.abs(null_mean) >= np.abs(observ_mean))

                                                        
                                                    
                                            pval = np.mean(pvals) 
                                            print(f'roi {pval}')
                                            
                                            pval_str = ""
                                            
                                            #compute p values
                                            if pval<0.01:
                                                pval_str+="*"
                                                if pval<1e-3:
                                                    pval_str+="*"
                                                    if pval<1e-4:
                                                        pval_str+="*"
                                       
                                            if pval_str != '':
                                                if bar_height>0:
                                                    roi_pval_str = f"{roi.replace('custom.','').replace('HCPQ1Q6.','').replace('glasser_','')}\n{pval_str}"
                                                    vanch = 'bottom'
                                                else:
                                                    roi_pval_str = f"{pval_str}\n{roi.replace('custom.','').replace('HCPQ1Q6.','').replace('glasser_','')}"
                                                    vanch = 'top'
                                                
                                                if bar_err != None:
                                                    text_y_pos = bar_height+(bar_err*np.sign(bar_height))
                                                else:
                                                    if each_subj_on_group:
                                                        text_y_pos = np.sign(bar_height)*np.max(np.abs(ssj_group_stats_roi_means)) + np.sign(bar_height)*ssj_group_stats_roi_errs[np.argmax(np.abs(ssj_group_stats_roi_means))]
                                                    else:
                                                        text_y_pos = bar_height


                                                pl.text(bar_position, text_y_pos, roi_pval_str,
                                                     fontsize=16, color='k', weight = 'bold', ha='center', va=vanch)
                                                                 
                                    #before was 0.4* bar_or_violin_width
                                    if len(self.only_models)>1:
                                        bar_position += (bar_or_violin_width)
                                    else:
                                        bar_position += (0.4*bar_or_violin_width)


                                    pl.xticks(x_ticks,x_labels, rotation=90, ha='left')
                                    
                                    # handles, labels = pl.gca().get_legend_handles_labels()
                                    # by_label = dict(zip(labels, handles))
                                    # if len(self.only_models) == 1:
                                    #     pl.legend(by_label.values(), by_label.keys())
                                        
                                    if save_figures:
    
                                        pl.savefig(opj(figure_path, f"{subj} {model} Mean {y_parameter_toplevel} {y_parameter.replace('/','')}.pdf"), dpi=600, bbox_inches='tight')
                                                              

                        
                        if not means_only:
                            for i, roi in enumerate(rois):
                                if len(y_par[analysis][subj][model][roi])>10:
                                    ###################
                                    #x vs y param by ROI
                                    pl.figure(f"{analysis} {subj} {roi.replace('custom.','').replace('HCPQ1Q6.','').replace('glasser_','')} {y_parameter} VS {x_parameter}", figsize=(8, 8), frameon=True)
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
    
                                            #set the weighting for regression and for plot alpha depending on the situation
                                            
                                            if np.allclose(rsq_y[analysis][subj][model][roi], rsq_x[analysis][subj][model][roi]):
                                                
                                                rsq_regr_weight[analysis][subj][model][roi] = rsq_y[analysis][subj][model][roi]
                                                if rsq_alpha_plot:
                                                    rsq_alpha_plots_all_rois = np.nan_to_num(np.array([np.nanmean(rsq_y[analysis][subj][model][r]) for r in rois if 'all' not in r and 'combined' not in r and 'Brain' not in r and len(y_par[analysis][subj][model][r])>10]))
    
    
                                            else:
    
                                                if y_parameter_toplevel != '':
                                                    rsq_regr_weight[analysis][subj][model][roi] = rsq_x[analysis][subj][model][roi]
                                                    
                                                    if rsq_alpha_plot:
                                                        rsq_alpha_plots_all_rois = np.nan_to_num(np.array([np.nanmean(rsq_x[analysis][subj][model][r]) for r in rois if 'all' not in r and 'combined' not in r and 'Brain' not in r and len(y_par[analysis][subj][model][r])>10]))

                                                if x_parameter_toplevel != '':
                                                    rsq_regr_weight[analysis][subj][model][roi] = rsq_y[analysis][subj][model][roi]
                                                    
                                                    if rsq_alpha_plot:
                                                        rsq_alpha_plots_all_rois = np.nan_to_num(np.array([np.nanmean(rsq_y[analysis][subj][model][r]) for r in rois if 'all' not in r and 'combined' not in r and 'Brain' not in r and len(y_par[analysis][subj][model][r])>10]))
    
                                                elif x_param_model != '':
                                                    rsq_regr_weight[analysis][subj][model][roi] = (rsq_y[analysis][subj][model][roi]+rsq_x[analysis][subj][model][roi])/2
                                                    if rsq_alpha_plot:
                                                        rsq_alpha_plots_all_rois = np.nan_to_num(np.array([np.nanmean((rsq_y[analysis][subj][model][roi]+rsq_x[analysis][subj][model][roi])/2) for r in rois if 'all' not in r and 'combined' not in r and 'Brain' not in r and len(y_par[analysis][subj][model][r])>10]))
                                                    
                                            
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
                                                                                                      weights=rsq_y[analysis][subj][model][roi][x_par_quantile]))
                        
                                                x_par_stats[analysis][subj][model][roi].append(weightstats.DescrStatsW(x_par[analysis][subj][model][roi][x_par_quantile],
                                                                                                      weights=rsq_x[analysis][subj][model][roi][x_par_quantile]))
                                                
                                                rsq_y_stats[analysis][subj][model][roi].append(weightstats.DescrStatsW(rsq_y[analysis][subj][model][roi][x_par_quantile],
                                                                                                      weights=np.ones_like(rsq_y[analysis][subj][model][roi][x_par_quantile])))
                                                rsq_x_stats[analysis][subj][model][roi].append(weightstats.DescrStatsW(rsq_x[analysis][subj][model][roi][x_par_quantile],
                                                                                                      weights=np.ones_like(rsq_x[analysis][subj][model][roi][x_par_quantile])))
    
    
                                            curr_x_bins = np.array([ss.mean for ss in x_par_stats[analysis][subj][model][roi]])
                                            curr_y_bins = np.array([ss.mean for ss in y_par_stats[analysis][subj][model][roi]])                                   
                
                                        except:
                                            pass    
        
                                        if not scatter:
                                            
                                            if len(self.only_models)>1:
                                                current_color = model_colors[model]
                                                current_label = model
                                            else:
                                                current_color = cmap_rois[i]
                                                current_label = roi.replace('custom.','').replace('HCPQ1Q6.','').replace('glasser_','')                                           
                                            if fit:
                                                try:
                                                    WLS = LinearRegression()
                                                    
    
                                                    WLS.fit(x_par[analysis][subj][model][roi].reshape(-1, 1), y_par[analysis][subj][model][roi], sample_weight=rsq_regr_weight[analysis][subj][model][roi])
        
                                                    wls_score = WLS.score(x_par[analysis][subj][model][roi].reshape(-1, 1),
                                                                          y_par[analysis][subj][model][roi].reshape(-1, 1),
                                                                        sample_weight=rsq_regr_weight[analysis][subj][model][roi].reshape(-1, 1))
                                                    
                                                    _, pval_lin = spearmanr(x_par[analysis][subj][model][roi],y_par[analysis][subj][model][roi])
                                                    
                                                    pl.title(f"{roi} {model} pval unweighted {pval_lin:.3f}")
                                                    
                                                    
                                                    if exp_fit:
                                                        #start points for sigmoid fits
                                                        x0s = [[1,1,0.5,1],[-1,1,0.5,1],[-10,10,1,10],[10,10,1,10],[1,100,0.5,1],[-1,100,0.5,1],[-5,20,0.28,56]]
                                                        curr_min_res = np.inf
                                                        
                                                        for x0 in x0s:
                                                            try:
                                                                res = minimize(lambda x,a,y:1-r2_score(y,x[3]+x[0]/(x[1]*np.exp(-x[2]*a)+1),sample_weight=rsq_regr_weight[analysis][subj][model][roi]), x0=x0,
                                                                               args=(x_par[analysis][subj][model][roi],
                                                                                     y_par[analysis][subj][model][roi]),
                                                                               method='Powell', options={'ftol':1e-8, 'xtol':1e-8})
                                                                if res['fun']<curr_min_res:
                                                                    exp_res = deepcopy(res)
                                                                    curr_min_res = deepcopy(res['fun'])
                                                            except Exception as e:
                                                                print(e)
                                                                x0s.append(np.random.rand(4))
                                                                pass
                                                        
                                                        
                                                        
                                                        exp_pred = exp_res['x'][3]+exp_res['x'][0]/(exp_res['x'][1]*np.exp(-exp_res['x'][2]*np.linspace(curr_x_bins.min(),curr_x_bins.max(),100))+1)
                                                        #full_exp_pred = exp_res['x'][3]+exp_res['x'][0]/(exp_res['x'][1]*np.exp(-exp_res['x'][2]*x_par[analysis][subj][model][roi])+1)
                                                        
                                                        rsq_pred=1-exp_res['fun']
                                                                
                                                    for c in range(200):
                                                        
                                                        sample = np.random.randint(0, len(x_par[analysis][subj][model][roi]), int(len(x_par[analysis][subj][model][roi])/upsampling_corr_factor))
                                                        
                                                        WLS_bootstrap = LinearRegression()
                                                        WLS_bootstrap.fit(x_par[analysis][subj][model][roi][sample].reshape(-1, 1), y_par[analysis][subj][model][roi][sample], sample_weight=rsq_regr_weight[analysis][subj][model][roi][sample])
                                                        
                                                        bootstrap_fits[analysis][subj][model][roi].append(WLS_bootstrap.predict(curr_x_bins.reshape(-1, 1)))
                                                
                                                                                                                
                                                        
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
                                                if zconfint_err_alpha != None:
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
                                                    current_label = roi.replace('custom.','').replace('HCPQ1Q6.','').replace('glasser_','')
                                                    
                                                scatter_cmap = colors.LinearSegmentedColormap.from_list(
                                                    'alpha_rsq', [(0, (*colors.to_rgb(current_color),0)), (1, current_color)])
                                                    
                                                pl.scatter(x_par[analysis][subj][model][roi], y_par[analysis][subj][model][roi], marker='o', s=0.01,
                                                    label=current_label, zorder=len(rois)-i, c=rsq_regr_weight[analysis][subj][model][roi], cmap=scatter_cmap)
    
                                            except Exception as e:
                                                print(e)
                                                pass                            
                                

                                if 'Eccentricity' in x_parameter or 'Size' in x_parameter:
                                    pl.xlabel(f"{x_parameter} (°)")
                                elif 'B/D' in x_parameter:
                                    pl.xlabel(f"{x_parameter} (% signal change)")  
                                elif x_parameter_toplevel != '' and 'Receptor' in x_parameter_toplevel:
                                    pl.xlabel(f"{x_parameter} (pmol/ml)")                                                 
                                else:
                                    pl.xlabel(f"{x_parameter}")  
        
                                if x_param_model != '':
                                    pl.xlabel(f"{x_param_model} {pl.gca().get_xlabel()}")
                                
                                if 'Eccentricity' in y_parameter or 'Size' in y_parameter:
                                    pl.ylabel(f"{subj} {roi.replace('custom.','').replace('HCPQ1Q6.','').replace('glasser_','')} {y_parameter} (°)")
                                elif 'B/D' in y_parameter:
                                    pl.ylabel(f"{subj} {roi.replace('custom.','').replace('HCPQ1Q6.','').replace('glasser_','')} {y_parameter} (% signal change)")
                                elif y_parameter_toplevel != '' and 'Receptor' in y_parameter_toplevel:
                                    pl.ylabel(f"{subj} {roi.replace('custom.','').replace('HCPQ1Q6.','').replace('glasser_','')} {y_parameter} (pmol/ml)")                                                                   
                                else:
                                    pl.ylabel(f"{subj} {roi.replace('custom.','').replace('HCPQ1Q6.','').replace('glasser_','')} {y_parameter}")
                                    
                                
                                    
                                if show_legend:
                                
                                    handles, labels = pl.gca().get_legend_handles_labels()
        
                                    legend_dict = dd(list)
                                    for cc, label in enumerate(labels):
                                        legend_dict[label].append(handles[cc])
                                        
                                    for label in legend_dict:
                                        legend_dict[label] = tuple(legend_dict[label])
        
                                    pl.legend([legend_dict[label] for label in legend_dict], legend_dict.keys())  
                                    
                                if save_figures:
                                    
                                    pl.savefig(opj(figure_path, f"{subj} {roi.replace('custom.','').replace('HCPQ1Q6.','').replace('glasser_','')} {y_parameter_toplevel} {y_parameter.replace('/','')} VS {x_parameter_toplevel} {x_param_model} {x_parameter.replace('/','')}.pdf"), dpi=600, bbox_inches='tight')
                                    

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
                                    
                                all_rois_x_means = []
                                all_rois_y_means = []
                                all_rois_rsq_means = []
                                all_rois_x = []
                                all_rois_y = []
                                all_rois_rsq = []
                                
                                all_rois_alpha = []
                                    
                                for i, roi in enumerate([r for r in rois if 'all' not in r and 'combined' not in r and 'Brain' not in r]):
                                    
                                    if len(y_par[analysis][subj][model][roi])>10:

                                        current_color = cmap_rois[i]
                                        current_label = roi.replace('custom.','').replace('HCPQ1Q6.','') .replace('glasser_','') 
                                        
                                        if not scatter:
                                            curr_x_bins = np.array([ss.mean for ss in x_par_stats[analysis][subj][model][roi]])
                                            curr_y_bins = np.array([ss.mean for ss in y_par_stats[analysis][subj][model][roi]])     
                                            
                                            if nr_bins == 1:
                                                all_rois_x.append(x_par[analysis][subj][model][roi])
                                                all_rois_y.append(y_par[analysis][subj][model][roi])
                                                all_rois_rsq.append(rsq_regr_weight[analysis][subj][model][roi])
                                                all_rois_x_means.append(curr_x_bins)
                                                all_rois_y_means.append(curr_y_bins)
                                                all_rois_rsq_means.append(np.nanmean(rsq_regr_weight[analysis][subj][model][roi]))
                                                
                                                #all_rois_alpha.append(np.where(alpha[analysis][subj][model][roi]>rsq_thresh)[0])
                                                all_rois_alpha.append(np.where(alpha[analysis][subj][model][roi]>rsq_thresh)[0])
                                                
                                            
                                            if fit:
                                                try:
                                                    WLS = LinearRegression()
                                                    WLS.fit(x_par[analysis][subj][model][roi].reshape(-1, 1), y_par[analysis][subj][model][roi], sample_weight=rsq_regr_weight[analysis][subj][model][roi])
                                                    
               
                                                    pl.plot(curr_x_bins,
                                                        WLS.predict(curr_x_bins.reshape(-1, 1)),
                                                        color=current_color, label=current_label)
                                                        #color=roi_colors[roi]) 
        
                                                                                      
                                                    # wls_score = WLS.score(curr_x_bins.reshape(-1, 1),
                                                    #                       curr_y_bins.reshape(-1, 1),
                                                    #                     sample_weight=np.array([ss.mean for ss in rsq_stats[analysis][subj][model][roi]]).reshape(-1, 1))
                                                    # print(f"{roi} {model} WLS score {wls_score}")
                                                except Exception as e:
                                                    print(e)
                                                    pass
                                                
                                                try:
                                                    #conf interval shading
                                                    pl.fill_between(curr_x_bins,
                                                                    np.min(bootstrap_fits[analysis][subj][model][roi], axis=0),
                                                                    np.max(bootstrap_fits[analysis][subj][model][roi], axis=0),
                                                                    alpha=0.2, color=current_color, label=current_label)
                                                    
                                                except Exception as e:
                                                    print(e)
                                                    pass
    
                                            #data points with errors
                                            if zconfint_err_alpha != None:
                                                curr_yerr = np.array([np.abs(ss.zconfint_mean(alpha=zconfint_err_alpha)-ss.mean) for ss in y_par_stats[analysis][subj][model][roi]]).T*upsampling_corr_factor**0.5
                                                curr_xerr = np.array([np.abs(ss.zconfint_mean(alpha=zconfint_err_alpha)-ss.mean) for ss in x_par_stats[analysis][subj][model][roi]]).T*upsampling_corr_factor**0.5
                                            else:
                                                curr_yerr = np.array([ss.std_mean for ss in y_par_stats[analysis][subj][model][roi]])*upsampling_corr_factor**0.5
                                                curr_xerr = np.array([ss.std_mean for ss in x_par_stats[analysis][subj][model][roi]])*upsampling_corr_factor**0.5
                                            
                                            if rsq_alpha_plot:
    
                                                rsq_alpha_plot_max = np.nanmax(rsq_alpha_plots_all_rois)
                                                rsq_alpha_plot_min = np.nanmin(rsq_alpha_plots_all_rois)   
                                                                                          
                                                alpha_plot = (np.nanmean(rsq_regr_weight[analysis][subj][model][roi])-rsq_alpha_plot_min)/(rsq_alpha_plot_max-rsq_alpha_plot_min)#np.nanmean(rsq_regr_weight[analysis][subj][model][roi])/rsq_alpha_plot_max#(1-rsq_alpha_plot_min)*(np.nanmean(rsq_regr_weight[analysis][subj][model][roi])-rsq_alpha_plot_min)/(rsq_alpha_plot_max-rsq_alpha_plot_min) + rsq_alpha_plot_min
                                                
                                            else:
                                                alpha_plot = 1
                                            #print(alpha_plot)
                                            alpha_plot = np.clip(alpha_plot,0,1)
                                            if np.isnan(alpha_plot) or not np.isfinite(alpha_plot):
                                                alpha_plot = 0
    
                                            plot_hexbins = False
                                            
                                            if plot_hexbins:
                                                roi_fill_color = 'k'
                                                roi_border_color = 'w'
                                            else:
                                                roi_fill_color = current_color
                                                roi_border_color = 'k'

                                            if nr_bins == 1:
                                                ms=16
                                                mew=2
                                            else:
                                                ms=5
                                                mew=1

                                                
                                            pl.errorbar(curr_x_bins,
                                                curr_y_bins,
                                                yerr=0,#curr_yerr,
                                                xerr=0,#curr_xerr,
                                            fmt='s ', mec=roi_border_color, color=roi_fill_color, label=current_label, alpha=alpha_plot, ms=ms, mew=mew)#color=current_color, ###mfc=roi_colors[roi], ecolor=roi_colors[roi])
                                            
                                            if rois_on_plot: 
                                                roi_name_txt = pl.text(curr_x_bins[-1], curr_y_bins[-1], current_label, fontsize=25, alpha=alpha_plot, color=roi_fill_color,  ha='left', va='bottom') #color=current_color,
                                                roi_name_txt.set_path_effects([peff.withStroke(linewidth=1, foreground=roi_border_color)])
                        
                                        
                                        else:
                                            
                                            try:
        
                                                
                                                scatter_cmap = colors.LinearSegmentedColormap.from_list(
                                                    'alpha_rsq', [(0, (*colors.to_rgb(current_color),0)), (1, current_color)])
                                                    
                                                pl.scatter(x_par[analysis][subj][model][roi], y_par[analysis][subj][model][roi], marker='o', s=0.01,
                                                    label=current_label, zorder=len(rois)-i, c=rsq_x[analysis][subj][model][roi], cmap=scatter_cmap)
        
                                            except Exception as e:
                                                print(e)
                                                pass       
 
      

                                if nr_bins == 1:
                                    
                                    all_rois_rsq_means = np.array(all_rois_rsq_means)
                                    all_rois_x_means = np.concatenate(all_rois_x_means)
                                    all_rois_y_means = np.concatenate(all_rois_y_means)
                                    all_rois_rsq = np.concatenate(all_rois_rsq)
                                    all_rois_x = np.concatenate(all_rois_x)
                                    all_rois_y = np.concatenate(all_rois_y)
                                    
                                    all_rois_alpha = np.concatenate(all_rois_alpha)
                                    
                                    
                                    normalize_rsq = False
                                    use_roi_means = True
                                    #NOTE if doing full_data: set plot hexbins to true. has been moved above to manage roi datapoints color
                                    use_full_data = False
                                    perm_yw = False
                                    
                                    n_perm = 1000000
                                    
                                    #np.save('/Users/marcoaqil/full_quantrois.npy', all_rois_alpha)
                                    #np.save(f'/Users/marcoaqil/{x_parameter}_alpha.npy', all_rois_alpha)
                                    #np.save(f'/Users/marcoaqil/{x_parameter}_quant.npy', all_rois_x)
                                    
                                    #an attempt to use a normalized r2 value as weight. correlations seems slightly stronger; pvals unaffected
                                    if normalize_rsq:                                        
                                        all_rois_rsq_means = (all_rois_rsq_means - all_rois_rsq_means.min()) / (all_rois_rsq_means.max() - all_rois_rsq_means.min())                                    
                                        all_rois_rsq = (all_rois_rsq - all_rois_rsq.min()) / (all_rois_rsq.max() - all_rois_rsq.min())
                                    
                                    
                                    if use_roi_means:
                                        CC_w = weightstats.DescrStatsW(np.stack((all_rois_x_means,all_rois_y_means)).T, weights=all_rois_rsq_means).corrcoef[0,1]
                                    if use_full_data:
                                        CC_w_full_data = weightstats.DescrStatsW(np.stack((all_rois_x,all_rois_y)).T, weights=all_rois_rsq).corrcoef[0,1]
                                        
                                        eigvecs_reduced, ft_x = reduced_graph_ft(all_rois_x, all_rois_alpha, 
                                                                                 eigenvectors_path='/Users/marcoaqil/1000eigvecs_full_quantrois.npy', 
                                                                                 eigenvectors_indices_path='/Users/marcoaqil/full_quantrois.npy', 
                                                                                 pycortex_subj=pycortex_subj)
                                        data_max_x = all_rois_x.max()
                                        data_min_x = all_rois_x.min()
                                        
                                        if perm_yw:
                                            _, ft_y = reduced_graph_ft(all_rois_y, all_rois_alpha, 
                                                                                     eigenvectors_path='/Users/marcoaqil/1000eigvecs_full_quantrois.npy', 
                                                                                     eigenvectors_indices_path='/Users/marcoaqil/full_quantrois.npy', 
                                                                                     pycortex_subj=pycortex_subj)
                                            data_max_y = all_rois_y.max()
                                            data_min_y = all_rois_y.min()   
                                            
                                            _, ft_w = reduced_graph_ft(all_rois_rsq, all_rois_alpha, 
                                                                                     eigenvectors_path='/Users/marcoaqil/1000eigvecs_full_quantrois.npy', 
                                                                                     eigenvectors_indices_path='/Users/marcoaqil/full_quantrois.npy', 
                                                                                     pycortex_subj=pycortex_subj)
                                            data_max_w = all_rois_rsq.max()
                                            data_min_w = all_rois_rsq.min()                                        
                                             
                                    
                                    CC_boot = []
                                    CC_boot_full_data = []
                                    
                                    for perm in tqdm(range(n_perm)):                                        
                                        if use_roi_means:
                                            #samp_idx = np.random.randint(0, len(all_rois_x_means), len(all_rois_x_means))
                                            perm_x = np.copy(all_rois_x_means)#[samp_idx]
                                            perm_y = np.copy(all_rois_y_means)#[samp_idx]
                                            perm_w = np.copy(all_rois_rsq_means)#[samp_idx] 
                                            
                                            np.random.shuffle(perm_x)
                                            if perm_yw:
                                                np.random.shuffle(perm_y)
                                                np.random.shuffle(perm_w)
                                            
                                            CC_boot.append(weightstats.DescrStatsW(np.stack((perm_x,perm_y)).T, weights=perm_w).corrcoef[0,1])
                                            
                                        if use_full_data:
                                            #samp_idx = np.random.randint(0, len(all_rois_x), int(len(all_rois_x)/upsampling_corr_factor))                                        
                                            # perm_x = all_rois_x[samp_idx]
                                            perm_y_full_data = np.copy(all_rois_y)#[samp_idx]
                                            perm_w_full_data = np.copy(all_rois_rsq)#[samp_idx]      
                                            
                                            perm_x_full_data = graph_randomization(data_max_x, data_min_x, eigvecs_reduced, ft_x)#[samp_idx]
                                            
                                            if perm_yw:
                                                perm_y_full_data = graph_randomization(data_max_y, data_min_y, eigvecs_reduced, ft_y)#[samp_idx]
                                                perm_w_full_data = graph_randomization(data_max_w, data_min_w, eigvecs_reduced, ft_w)#[samp_idx]
                                                
                                            CC_boot_full_data.append(weightstats.DescrStatsW(np.stack((perm_x_full_data,perm_y_full_data)).T, 
                                                                                             weights=perm_w_full_data).corrcoef[0,1])
                                            
                                            
                                    
                                    title_string = ""                                        
                                    if use_roi_means:
                                        print(f"null mean roi means CC (should be close to zero) {np.mean(CC_boot)}")
                                        pval_wcc = np.sum(np.abs(CC_w)<np.abs(np.array(CC_boot)))/len(CC_boot)
                                        
                                        if n_perm >= 100000:
                                            if pval_wcc<1e-2:
                                                pval_string = '*'
                                                if pval_wcc<1e-3:
                                                    pval_string = '**'
                                                    if pval_wcc<1e-4:
                                                        pval_string = '***'                                                                                        
                                            else:
                                                pval_string = 'n.s.'
                                        else:
                                            pval_string = ""
                                        
                                        title_string += f"wCC={CC_w:.2f} p={pval_wcc:.2e} ({pval_string})"
                                    
                                    if use_full_data:
                                        if use_roi_means:
                                            title_string += "; "
                                            
                                        print(f"null mean full data CC (should be close to zero) {np.mean(CC_boot_full_data)}")
                                        pval_wcc_full_data = np.sum(np.abs(CC_w_full_data)<np.abs(np.array(CC_boot_full_data)))/len(CC_boot_full_data)    
                                        
                                        if n_perm >= 100000:
                                            if pval_wcc_full_data<1e-2:
                                                pval_string_full_data = '*'
                                                if pval_wcc_full_data<1e-3:
                                                    pval_string_full_data = '**'
                                                    if pval_wcc_full_data<1e-4:
                                                        pval_string_full_data = '***'                                                                                        
                                            else:
                                                pval_string_full_data = 'n.s.'
                                        else:
                                            pval_string_full_data = ""
                                            
                                        title_string += f" wCCfd={CC_w_full_data:.2f} p={pval_wcc_full_data:.2e} ({pval_string_full_data})"                                        
                                    
                                    
                                    pl.title(title_string)
                                    


                                    #new_cmap = pl.get_cmap('cmr.rainforest')
                                    new_cmap = pl.get_cmap(cmr.get_sub_cmap('Reds', 0.05, 0.9))
                                    bins_b = 'log'#None #'log'
                                    

                                    if plot_hexbins:
                                        pl.hexbin(all_rois_x, all_rois_y, all_rois_rsq, gridsize=25, alpha=0.5, linewidths=0, bins=bins_b, edgecolors='none', cmap=new_cmap)#, cmap='binary') # mincnt=10,

                                    #regression line
                                                                       
                                    
                                    if use_roi_means:
                                        WLS_comb = LinearRegression()
                                        
                                        WLS_comb.fit(all_rois_x_means.reshape(-1, 1), all_rois_y_means, sample_weight=all_rois_rsq_means)  

                                        # if use_full_data:
                                        #     WLS_comb_prediction = WLS_comb.predict(all_rois_x.reshape(-1, 1))    
                                        #     pl.plot(all_rois_x[np.argsort(all_rois_x)],
                                        #     WLS_comb_prediction[np.argsort(all_rois_x)],
                                        #     color='w')            
                                        # else:
                                        WLS_comb_prediction = WLS_comb.predict(all_rois_x_means.reshape(-1, 1))                                          
                                        pl.plot(all_rois_x_means[np.argsort(all_rois_x_means)],
                                            WLS_comb_prediction[np.argsort(all_rois_x_means)],
                                            color='k', zorder=1003)    
                                        
                                    if use_full_data:     
                                        
                                        WLS_comb_full_data = LinearRegression()
                                        
                                        WLS_comb_full_data.fit(all_rois_x.reshape(-1, 1), all_rois_y, sample_weight=all_rois_rsq)                                           
                                        WLS_comb_full_data_prediction = WLS_comb_full_data.predict(all_rois_x.reshape(-1, 1))    
                                        pl.plot(all_rois_x[np.argsort(all_rois_x)],
                                            WLS_comb_full_data_prediction[np.argsort(all_rois_x)],
                                            color='k', zorder=1000)                                         

                                 
                                    #bootstrap conf intervals
                                    comb_boot_fits = []
                                    comb_boot_fits_full_data = []
                                    for bootc in range(200):
                                        if use_roi_means:
                                            samp_idx = np.random.randint(0, len(all_rois_x_means), len(all_rois_x_means))
                                            bootc_x = all_rois_x_means[samp_idx]
                                            bootc_y = all_rois_y_means[samp_idx]
                                            bootc_w = all_rois_rsq_means[samp_idx]
                                            
                                            WLS_comb_boot = LinearRegression()
                                            WLS_comb_boot.fit(bootc_x.reshape(-1, 1), bootc_y, sample_weight=bootc_w)  
                                        if use_full_data:                                            
                                            samp_idx_full_data = np.random.randint(0, len(all_rois_x), int(len(all_rois_x)/upsampling_corr_factor))
                                            bootc_x_full_data = all_rois_x[samp_idx_full_data]
                                            bootc_y_full_data = all_rois_y[samp_idx_full_data]
                                            bootc_w_full_data = all_rois_rsq[samp_idx_full_data]    
                                            WLS_comb_boot_full_data = LinearRegression()
                                            WLS_comb_boot_full_data.fit(bootc_x_full_data.reshape(-1, 1), bootc_y_full_data, sample_weight=bootc_w_full_data)                                       

                                        
                                        if use_roi_means:
                                            comb_boot_fits.append(WLS_comb_boot.predict(all_rois_x_means.reshape(-1, 1)))
                                        if use_full_data:
                                            comb_boot_fits_full_data.append(WLS_comb_boot_full_data.predict(all_rois_x.reshape(-1, 1)))
                                    
                                    
                                    if use_roi_means:
                                        comb_boot_fits = np.array(comb_boot_fits)
                                    
                                        conf_max = np.nanquantile(comb_boot_fits,0.95,axis=0)#WLS_comb_prediction + sem(comb_boot_fits,axis=0)#
                                        conf_min = np.nanquantile(comb_boot_fits,0.05,axis=0)#WLS_comb_prediction - sem(comb_boot_fits,axis=0)#
                                    
                                    
                                        pl.plot(all_rois_x_means[np.argsort(all_rois_x_means)],conf_max[np.argsort(all_rois_x_means)], color='k', ls = '--', zorder=1004)
                                        pl.plot(all_rois_x_means[np.argsort(all_rois_x_means)],conf_min[np.argsort(all_rois_x_means)], color='k', ls = '--', zorder=1005)
                                        
                                    if use_full_data:
                                        comb_boot_fits_full_data = np.array(comb_boot_fits_full_data)
                                    
                                        conf_max_full_data = np.nanquantile(comb_boot_fits_full_data,0.95,axis=0)#WLS_comb_prediction + sem(comb_boot_fits,axis=0)#
                                        conf_min_full_data = np.nanquantile(comb_boot_fits_full_data,0.05,axis=0)#WLS_comb_prediction - sem(comb_boot_fits,axis=0)#                                        
                                        
                                        pl.plot(all_rois_x[np.argsort(all_rois_x)],conf_max_full_data[np.argsort(all_rois_x)], color='k', ls = '--', zorder=1001)
                                        pl.plot(all_rois_x[np.argsort(all_rois_x)],conf_min_full_data[np.argsort(all_rois_x)], color='k', ls = '--', zorder=1002)                                    

                                
                                if 'Eccentricity' in x_parameter or 'Size' in x_parameter:
                                    pl.xlabel(f"{x_parameter} (°)")
                                elif 'B/D' in x_parameter:
                                    pl.xlabel(f"{x_parameter} (% signal change)")   
                                elif x_parameter_toplevel != '' and 'Receptor' in x_parameter_toplevel:
                                    pl.xlabel(f"{x_parameter} (pmol/ml)")                                                 
                                else:
                                    pl.xlabel(f"{x_parameter}")
                                    
                                if x_param_model != '':
                                    pl.xlabel(f"{x_param_model} {pl.gca().get_xlabel()}")                                    
                                
                                if 'Eccentricity' in y_parameter or 'Size' in y_parameter:
                                    pl.ylabel(f"{subj} {model} {y_parameter} (°)")
                                elif 'B/D' in y_parameter:
                                    pl.ylabel(f"{subj} {model} {y_parameter} (% signal change)")  
                                elif y_parameter_toplevel != '' and 'Receptor' in y_parameter_toplevel:
                                    pl.ylabel(f"{subj} {model} {y_parameter} (pmol/ml)")                                                 
                                else:
                                    pl.ylabel(f"{subj} {model} {y_parameter}")
                                    
                                if show_legend:
                                    handles, labels = pl.gca().get_legend_handles_labels()
        
                                    legend_dict = dd(list)
                                    for cc, label in enumerate(labels):
                                        legend_dict[label].append(handles[cc])
                                        
                                    for label in legend_dict:
                                        legend_dict[label] = tuple(legend_dict[label])
        
                                    pl.legend([legend_dict[label] for label in legend_dict], legend_dict.keys())                      
    
                                if save_figures:

                                    pl.savefig(opj(figure_path, f"{subj} {model} {y_parameter_toplevel} {y_parameter.replace('/','')} VS {x_parameter_toplevel} {x_param_model} {x_parameter.replace('/','')}.pdf"), dpi=600, bbox_inches='tight')


        return
    
    def multidim_analysis(self, parameters, rois, rsq_thresh, save_figures, figure_path, space_names = 'fsnative',
                    analysis_names = 'all', subject_ids='all', parameter_toplevel=None, rsq_weights=True,
                    x_dims_idx=None, y_dims_idx=None, zscore_data=False, zscore_data_across_rois = False,
                    size_response_curves = False, third_dim_sr_curves = None,
                    plot_corr_matrix = False, perform_ols=False, polar_plots = False, cv_regression = False,
                    zconfint_err_alpha = None, bold_voxel_volume = None, quantile_exclusion=0.999,
                    perform_pca = False, perform_pls = False, vis_pls_pycortex = False, vis_pca_pycortex = False,
                    vis_pca_comps_rois = ['combined'], vis_pca_comps_axes = [0,1], rsq_alpha_pca_plot = True):
        
        np.set_printoptions(precision=4,suppress=True)
        self.curr_rois_names = []
        
        if not os.path.exists(figure_path) and save_figures:
            os.makedirs(figure_path)
       
        pl.rcParams.update({'font.size': 16})
        pl.rcParams.update({'pdf.fonttype':42})
        pl.rcParams.update({'figure.max_open_warning': 0})
        pl.rcParams['axes.spines.right'] = False
        pl.rcParams['axes.spines.top'] = False
        
        
        cmap_values = list(np.linspace(0.9, 0.0, len([r for r in rois if r not in ['Brain', 'all_custom', 'combined']])))
        
        cmap_values += [0 for r in rois  if r in ['Brain', 'all_custom', 'combined']]
        
        cmap_rois = cm.get_cmap('nipy_spectral')(cmap_values)#
        
        #making black into dark gray for visualization reasons
        cmap_rois[(cmap_rois == [0,0,0,1]).sum(1)==4] = [0.33,0.33,0.33,1]

        if space_names == 'all':
            spaces = [item for item in self.main_dict.items()]
        else:
            spaces = [item for item in self.main_dict.items() if item[0] in space_names] 

        for space, space_res in spaces:
        
            
            if quantile_exclusion == None:
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
                        if bold_voxel_volume != None:
                            
                            print("Make sure bold_voxel_volume is specified in mm^3")
                            
                            try:
                                if subj.isdecimal() and space == 'HCP':
                                    pycortex_subj = '999999'
                                elif 'fsaverage' in subj:
                                    pycortex_subj = 'fsaverage'
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
        
                    model_colors = {'Gauss':'blue','CSS':'orange','DoG':'green','Norm_abcd':'red','Norm_abc':'purple'}                                              

                    for i, roi in enumerate(rois):                              
                        if 'mean an' not in analysis or 'fsaverage' in subj:
                            if 'sub' in subj or 'fsaverage' in subj:
                                if len(self.only_models)>1:
                                    curr_alpha = subj_res['Processed Results']['Alpha']['all']
                                else:
                                    curr_alpha = subj_res['Processed Results']['Alpha'][self.only_models[0]]
                                    
                                                                

                                if roi in self.idx_rois[subj]:

                                    #comparing same vertices
                                    alpha[analysis][subj][roi] = roi_mask(self.idx_rois[subj][roi], curr_alpha)
                                    
                                
                                else:
                                    #if ROI != defined
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
                                        
                                for param in parameters:
                                            
                                    if parameter_toplevel != None:
                                        if param in subj_res['Processed Results'][parameter_toplevel] and alpha[analysis][subj][roi].sum()>0:  
                                            alpha[analysis][subj][roi] *= np.isfinite(subj_res['Processed Results'][parameter_toplevel][param])
                                    
                                    for model in self.only_models:
                                        if model in subj_res['Processed Results'][param] and alpha[analysis][subj][roi].sum()>0:
                                            alpha[analysis][subj][roi] *= np.isfinite(subj_res['Processed Results'][param][model])
                                            
                                            #manual exclusion of outliers
                                            if quantile_exclusion == None:    
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
                                                if 'rsq' not in param.lower() and 'eccentricity' not in param.lower() and 'polar angle' not in param.lower():

                                                    alpha[analysis][subj][roi] *= (subj_res['Processed Results'][param][model]<np.nanquantile(subj_res['Processed Results'][param][model],quantile_exclusion))*(subj_res['Processed Results'][param][model]>np.nanquantile(subj_res['Processed Results'][param][model],1-quantile_exclusion))
                                                    

                                                
                    for i, roi in enumerate(rois):                              
                        if 'mean an' not in analysis or 'fsaverage' in subj:
                            if 'sub' in subj or 'fsaverage' in subj:
                                for param in parameters:                                           
                                    
                                    for model in self.only_models:
                                        if model in subj_res['Processed Results'][param]:                       
                                            dimensions[analysis][subj][roi].append(f"{param} {model}")
                                            #print(param)
                                            #print(subj_res['Processed Results'][param][model][alpha[analysis][subj][roi]].max())

                                            if zscore_data and 'rsq' not in param.lower():
                                                multidim_param_array[analysis][subj][roi].append(zscore(subj_res['Processed Results'][param][model][alpha[analysis][subj][roi]>rsq_thresh]))      
                                            
                                            elif zscore_data_across_rois and 'rsq' not in param.lower():
                                                
                                                if 'Brain' in rois:
                                                    overall_roi = 'Brain'
                                                elif 'combined' in rois:
                                                    overall_roi = 'combined'
                                                elif 'all_custom' in rois:
                                                    overall_roi = 'all_custom'                                                
                                                                                       
                                                all_rois_zsc = zscore(subj_res['Processed Results'][param][model][alpha[analysis][subj][overall_roi]>rsq_thresh])
                                                
                                                # all_rois_data = np.copy(subj_res['Processed Results'][param][model][alpha[analysis][subj]['combined']>rsq_thresh])
                                                # all_rois_wstats = weightstats.DescrStatsW(all_rois_data,
                                                #                                           weights=subj_res['Processed Results']['RSq'][model][alpha[analysis][subj]['combined']>rsq_thresh])
                                                
                                                # all_rois_zsc = (all_rois_data-all_rois_wstats.mean)/all_rois_wstats.std_mean
                                                
                                                copy = np.zeros_like(alpha[analysis][subj][roi])
                                                copy[alpha[analysis][subj][overall_roi]>rsq_thresh] = np.copy(all_rois_zsc)
                                                multidim_param_array[analysis][subj][roi].append(copy[alpha[analysis][subj][roi]>rsq_thresh])

                                            else:

                                                multidim_param_array[analysis][subj][roi].append(subj_res['Processed Results'][param][model][alpha[analysis][subj][roi]>rsq_thresh])
                                                
                                    
                                    if parameter_toplevel != None:
                                        if param in subj_res['Processed Results'][parameter_toplevel]:
                                            dimensions[analysis][subj][roi].append(f"{param}")
                                            if zscore_data and 'rsq' not in param.lower():
                                                multidim_param_array[analysis][subj][roi].append(zscore(subj_res['Processed Results'][parameter_toplevel][param][alpha[analysis][subj][roi]>rsq_thresh]))
                                            
                                            elif zscore_data_across_rois and 'rsq' not in param.lower():
                                                
                                                if 'Brain' in rois:
                                                    overall_roi = 'Brain'
                                                elif 'combined' in rois:
                                                    overall_roi = 'combined'
                                                elif 'all_custom' in rois:
                                                    overall_roi = 'all_custom'
                                                
                                                all_rois_zsc = zscore(subj_res['Processed Results'][parameter_toplevel][param][alpha[analysis][subj][overall_roi]>rsq_thresh])
                                                copy = np.zeros_like(alpha[analysis][subj][roi])
                                                copy[alpha[analysis][subj][overall_roi]>rsq_thresh] = np.copy(all_rois_zsc)
                                                multidim_param_array[analysis][subj][roi].append(copy[alpha[analysis][subj][roi]>rsq_thresh])
                                                
                                            else:
                                                multidim_param_array[analysis][subj][roi].append(subj_res['Processed Results'][parameter_toplevel][param][alpha[analysis][subj][roi]>rsq_thresh])
                                                
                                
                                multidim_param_array[analysis][subj][roi] = np.array([x for _,x in sorted(zip(dimensions[analysis][subj][roi],multidim_param_array[analysis][subj][roi]))])

                                
                            elif len(subjects)>1 and space != 'fsaverage':
                                #group data
                                dimensions[analysis][subj][roi] = dimensions[analysis][subjects[0][0]][roi]
                                alpha[analysis][subj][roi] = np.hstack(tuple([alpha[analysis][sid][roi] for sid in alpha[analysis] if 'sub' in sid]))
                                multidim_param_array[analysis][subj][roi] = np.hstack(tuple([multidim_param_array[analysis][sid][roi] for sid in multidim_param_array[analysis] if 'sub' in sid]))
 
                                
                        elif space != 'fsaverage':
                            #all analyses data
                            ans = [an[0] for an in analyses if 'mean an' not in an[0]]
                            dimensions[analysis][subj][roi] = dimensions[analyses[0][0]][subj][roi]
                            alpha[analysis][subj][roi] = np.hstack(tuple([alpha[an][subj][roi] for an in ans]))
                            multidim_param_array[analysis][subj][roi] = np.hstack(tuple([multidim_param_array[an][subj][roi] for an in ans]))

                            
                    
                        ordered_dimensions = sorted(dimensions[analysis][subj][roi])
                        #print(f"Ordered dimensions: {ordered_dimensions}")
                
                #correlation matrices
                for subj, subj_res in subjects:  
                    for i, roi in enumerate(rois):                          
                        if ('mean' in analysis) or 'fsaverage' in subj:
             
                            #print(multidim_param_array[analysis][subj][roi].shape)
                            
                            if plot_corr_matrix and np.sum(alpha[analysis][subj][roi]>rsq_thresh)>10 and roi in vis_pca_comps_rois:
                                print(f"{analysis} {subj} {roi}")
                                pl.figure(figsize=(12,12))
                                
                                if rsq_weights:
                                    correlation_matrix = weightstats.DescrStatsW(multidim_param_array[analysis][subj][roi].T, weights=multidim_param_array[analysis][subj][roi][ordered_dimensions.index('RSq Norm_abcd')]).corrcoef
                                else:                                
                                    correlation_matrix = np.corrcoef(multidim_param_array[analysis][subj][roi])
                                    
                                np.fill_diagonal(correlation_matrix,0)
                                im = pl.imshow(correlation_matrix, vmin=-1, vmax=1, cmap='RdBu_r')
                                #pl.xlim((0,correlation_matrix.shape[0]))
                                #pl.ylim((0,correlation_matrix.shape[0]))
                                
                                for i in range(correlation_matrix.shape[0]):
                                    for j in range(correlation_matrix.shape[1]):
                                        if i != j:
                                            pl.text(i, j, f"{correlation_matrix[i,j]:.2f}", fontsize=14, color='black', weight = 'bold', ha='center', va='center')
                                    
                                
                                ticks = []
                                tick_colors = []
                                rec_dims = []
                                for dim, dim_name in enumerate(ordered_dimensions):
                                    for model in self.only_models:
                                        if model in dim_name:
                                            tick_colors.append(model_colors[model])
                                            ticks.append(dim_name.replace(model,''))
                                    if parameter_toplevel != None:
                                        if dim_name in subj_res['Processed Results'][parameter_toplevel]:
                                            tick_colors.append('black')
                                            ticks.append(dim_name)
                                            rec_dims.append(dim)
                                
                                pl.xticks(np.arange(len(ordered_dimensions)), ticks, rotation='vertical')
                                pl.yticks(np.arange(len(ordered_dimensions)), ticks)
                                colorbar(im)
                                
                                for tick_x, tick_y, tick_color in zip(pl.gca().get_xticklabels(), pl.gca().get_yticklabels(), tick_colors):
                                    tick_x.set_color(tick_color)
                                    tick_y.set_color(tick_color)

                #PLS/OLS regressions, PCA, orthoplots
                self.pls_result_dict = dd(lambda:dd(list))
                self.ols_result_dict = dd(lambda:dd(lambda:dd(list)))
                
                for subj, subj_res in subjects: 
                    if polar_plots:
                        fig_polarplot_allrois, ax_polarplot_allrois = pl.subplots(figsize=(11,8),subplot_kw={'projection': 'polar'})
                    
                    for i, roi in enumerate(rois):  
                        if ('mean' in analysis) or 'fsaverage' in subj:

                            print(f"{analysis} {subj} {roi}")
                            if x_dims_idx == None or y_dims_idx == None:    
                                if parameter_toplevel != None:
                                    print("X and Y dims not specified. Using receptors as X, everything else as Y.")
                                    x_dims = rec_dims
                                    y_dims = [dim for dim, _ in enumerate(ordered_dimensions) if dim not in rec_dims]
                                else:
                                    print("X and Y dims not specified. Using norm model parameters as X, everything else as Y.")
                                    x_dims = list(set([dim for dim, dim_name in enumerate(ordered_dimensions) if 'Norm_abcd' in dim_name]))
                                    y_dims = list(set([dim for dim, dim_name in enumerate(ordered_dimensions) if 'Norm_abcd' not in dim_name]))
                            else:
                                
                                x_dims = list(set([dim for dim, dim_name in enumerate(ordered_dimensions) for x in x_dims_idx if dim_name.startswith(parameters[x])]))
                                y_dims = list(set([dim for dim, dim_name in enumerate(ordered_dimensions) for y in y_dims_idx if dim_name.startswith(parameters[y])]))
                                
                            print(f"X-dims: {[ordered_dimensions[x_dim] for x_dim in x_dims]}")
                            print(f"Y-dims: {[ordered_dimensions[y_dim] for y_dim in y_dims]}")
                            
                            #xy_dims has the dimensions that go into X and Y. ordered_dims has everything that goes into the multidim_param_array
                            xy_dims = [ordered_dimensions[x_dim] for x_dim in x_dims]+[ordered_dimensions[y_dim] for y_dim in y_dims]
                            print(f"alldims {xy_dims}")
                            print(f"ord dims {ordered_dimensions}")
                            
                            X = multidim_param_array[analysis][subj][roi][x_dims].T
                            Y = multidim_param_array[analysis][subj][roi][y_dims].T
                            print(f"Sample size: {X.shape[0]}")
                            
                            try:
                                ############## polar plots
                                if polar_plots and np.sum(alpha[analysis][subj][roi]>rsq_thresh)>10:
                                    
                                    fig_polarplot, ax_polarplot = pl.subplots(figsize=(11,8),subplot_kw={'projection': 'polar'})
                                    print('drawing polar plots')

                                    full_dataset = np.concatenate((X,Y),axis=1)
                                    
                                    param_labels = [e.replace('Norm_abcd','') for e in xy_dims]
                                    
                                    par_stats = []
                                                                    
                                    for j in range(full_dataset.shape[1]):
                                        if rsq_weights and 'Norm_abcd' in xy_dims[j]:
                                            
                                            rsq_weights_polar = multidim_param_array[analysis][subj][roi][ordered_dimensions.index('RSq Norm_abcd')]
                                        else:
                                            rsq_weights_polar = np.ones(full_dataset.shape[0])  
                                            
                                        wstats = weightstats.DescrStatsW(full_dataset[:,j],
                                                            weights=rsq_weights_polar)
                                        par_stats.append(wstats)
                                        
                                    par_means = np.array([wstats.mean for wstats in par_stats])
                                        
                                    if zconfint_err_alpha != None:
                                        par_err = np.array([np.abs(wstats.zconfint_mean(alpha=zconfint_err_alpha)-wstats.mean) for wstats in par_stats])*upsampling_corr_factor**0.5
                                    else:
                                        par_err = np.array([wstats.std_mean for wstats in par_stats])*upsampling_corr_factor**0.5
                                        

                                    #ax_polarplot.set_title(roi)
                                    theta = np.linspace(0,2*np.pi,full_dataset.shape[1],endpoint=False)
                                    #angle for legend
                                    angle = np.deg2rad(0)
                                    
                                    ax_polarplot.plot(np.append(theta,0), np.append(par_means,par_means[0]), "o", color=cmap_rois[i], label=roi)
                                    ax_polarplot.errorbar(np.append(theta,0), np.append(par_means,par_means[0]), yerr=np.append(par_err,par_err[0]), capsize=0, color=cmap_rois[i])                                        
                                    ax_polarplot.set_xticks(theta)
                                    ax_polarplot.set_xticklabels(param_labels)
                                    ax_polarplot.xaxis.set_tick_params(pad=30)
                                    ax_polarplot.legend(loc="lower left",
                                              bbox_to_anchor=(.5 + np.cos(angle)/2, .5 + np.sin(angle)/2))  
                                    
                                    if roi != 'combined':
                                    
                                        ax_polarplot_allrois.plot(np.append(theta,0), np.append(par_means,par_means[0]), "o", color=cmap_rois[i], label=roi)
                                        ax_polarplot_allrois.errorbar(np.append(theta,0), np.append(par_means,par_means[0]), yerr=np.append(par_err,par_err[0]), capsize=0, color=cmap_rois[i])
                                        
                                        ax_polarplot_allrois.set_xticks(theta)
                                        ax_polarplot_allrois.set_xticklabels(param_labels)
                                        ax_polarplot_allrois.xaxis.set_tick_params(pad=30)
                                        
                                        ax_polarplot_allrois.legend(loc="lower left",
                                                  bbox_to_anchor=(.5 + np.cos(angle)/2, .5 + np.sin(angle)/2))                                     
                                        
                                ##########PCA
                                if perform_pca and np.sum(alpha[analysis][subj][roi]>rsq_thresh)>10:
                                    print('performing PCA')
                                    
                                    #always zscore pca data
                                    full_dataset = zscore(np.concatenate((X,Y),axis=1),axis=0)
    
                                    ncomp = full_dataset.shape[1]
                                    
                                    wpca_weights = np.ones_like(full_dataset)
                                    
                                    if rsq_weights:
                                        
                                        for p in range(full_dataset.shape[1]):
                                            if 'Norm_abcd' in xy_dims[p]:
                                                wpca_weights[:,p] = multidim_param_array[analysis][subj][roi][ordered_dimensions.index('RSq Norm_abcd')]
                                                
                                            
                                        
                                        pca = WPCA(n_components=ncomp).fit(full_dataset,weights=wpca_weights)
                                    else:
                                        pca = WPCA(n_components=ncomp).fit(full_dataset)
                                        
                                    if cv_regression:

                                        
                                        
                                        #cv_rsq_total = []
                                        
                                        for cv_subj in [s for s,_ in subjects if s!='fsaverage' and s!='Group' and s!=subj]:
                                            print(f"CV PCA (fit on {subj}, test on {cv_subj})")
                                            cv_rsq_comp = []
                                            
                                            #cv_subj = subj

                                            full_dataset_cv = zscore(np.concatenate((multidim_param_array[analysis][cv_subj][roi][x_dims].T,
                                                                              multidim_param_array[analysis][cv_subj][roi][y_dims].T),axis=1), axis=0)
                                        

                                            if rsq_weights:
                                                cv_wpca_weights = np.ones_like(full_dataset_cv)
                                                for p in range(full_dataset_cv.shape[1]):
                                                    if 'Norm_abcd' in xy_dims[p]:
                                                        cv_wpca_weights[:,p] = multidim_param_array[analysis][cv_subj][roi][ordered_dimensions.index('RSq Norm_abcd')]    
                                                        
                                            for p in range(ncomp):
                                                #weights of cv or fit subj?
                                                if rsq_weights:
                                                    X_trans = pca.transform(full_dataset_cv, weights=cv_wpca_weights)
                                                else:
                                                    X_trans = pca.transform(full_dataset_cv)
                                                    
                                                X_trans_ii = np.zeros_like(X_trans)
                                                X_trans_ii[:, p] = X_trans[:, p]
                                                X_approx_ii = pca.inverse_transform(X_trans_ii)
                                                
                                                
                                                if rsq_weights: 
                                                  
                                                    cv_rsq_comp.append(1 - (cv_wpca_weights*(X_approx_ii - full_dataset_cv)**2).sum() /\
                                                                   (cv_wpca_weights*(full_dataset_cv - pca.mean_)**2).sum())
                                                                                            
                                                else:
                                                    cv_rsq_comp.append( 1 - (np.linalg.norm(X_approx_ii - full_dataset_cv) /
                                                                  np.linalg.norm(full_dataset_cv - pca.mean_)) ** 2)
                                                    
                                                

                                    #new_cmap = pl.get_cmap('cmr.rainforest')
                                                                             
                                                                                     
                                    
                                    if roi in vis_pca_comps_rois:
                                        #new_cmap_rb = pl.get_cmap('cmr.fusion')   
                                        new_cmap_pca_r2bars = pl.get_cmap('Greens')   
                                                                                  
                                        ncomp_bar_plots = ncomp
                                        
                                        f_pca_r2, ax_pca_r2 = pl.subplots(num=f"PCA_dimensions_r2_{roi.replace('custom.','').replace('HCPQ1Q6.','').replace('glasser_','')}", figsize=(8,14))
                                        pl.ylabel("Variance")
                                        pl.xlabel("PCA component #")              
                                        
                                        color_r2_bars = new_cmap_pca_r2bars(pca.explained_variance_ratio_/np.max(pca.explained_variance_ratio_))
                                        


                                        ax_pca_r2.bar(np.arange(ncomp_bar_plots),pca.explained_variance_ratio_, color=color_r2_bars)
                                        ax_pca_r2.plot(np.linspace(-0.4, ncomp_bar_plots-0.6, ncomp_bar_plots), np.ones(ncomp_bar_plots)/ncomp_bar_plots, '--k', label='Equivariance threshold')                                        
                                        ax_pca_r2.plot(np.arange(ncomp_bar_plots),np.cumsum(pca.explained_variance_ratio_), '-ko', label='Cumulative variance explained')
                                        ax_pca_r2.plot(np.arange(ncomp_bar_plots),pca.explained_variance_ratio_, '-ro', label='Variance explained')
                                        ax_pca_r2.set_xticks(np.arange(ncomp_bar_plots))
                                        ax_pca_r2.set_xticklabels(np.arange(ncomp_bar_plots)+1)
                                        ax_pca_r2.legend()
                                        


                                        f, axes = pl.subplots(1, ncomp_bar_plots, figsize=(ncomp_bar_plots*6,7))
                                        
   
                                       
                                        for j in range(ncomp_bar_plots):  
                                            ax_pca_r2.text(j+0.35,pca.explained_variance_ratio_[j],f"{pca.explained_variance_ratio_[j]:.2f}",fontsize=18,ha='center', va='bottom')

                                            #color_bars = new_cmap_rb(pca.components_[j,:])
                                            
                                            if j == 0:
                                                axes[j].set_yticks(np.arange(full_dataset.shape[1])/2)
                                                axes[j].set_yticklabels([ordered_dimensions[x_dim].replace('Norm_abcd','') for x_dim in x_dims]+[ordered_dimensions[y_dim].replace('Norm_abcd','') for y_dim in y_dims])
                                            else:
                                                axes[j].set_yticks([])
                                                
                                            axes[j].barh(np.arange(full_dataset.shape[1])/2, np.array(pca.components_)[j,:], height=0.3, color=['red' if c>0 else 'blue' for c in np.sign(np.array(pca.components_)[j,:])], alpha=0.8)#, label=f"RSq {pca.explained_variance_ratio_[j]:.3f}")
                                             
                                            if cv_regression:
                                                print(f"PCA component {j+1} (R2={pca.explained_variance_ratio_[j]:.2f}, cvR2={cv_rsq_comp[j]:.2f})")
                                                axes[j].set_title(f"PCA comp {j+1} (R2={pca.explained_variance_ratio_[j]:.2f}, cvR2={cv_rsq_comp[j]:.2f})")
                                            else:
                                                print(f"PCA component {j+1} (R2={pca.explained_variance_ratio_[j]:.2f})")
                                                axes[j].set_title(f"PCA comp {j+1} (R2={pca.explained_variance_ratio_[j]:.2f})")
                                            #axes[j].legend()
                                            
                                            
                                        if save_figures: 
                                            f.savefig(opj(figure_path,f"PCA_dimensions_{roi.replace('custom.','').replace('HCPQ1Q6.','').replace('glasser_','')}.pdf"),dpi=600, bbox_inches='tight')
                                            f_pca_r2.savefig(opj(figure_path,f"PCA_dimensions_r2_{roi.replace('custom.','').replace('HCPQ1Q6.','').replace('glasser_','')}.pdf"),dpi=600, bbox_inches='tight')
                                    
                                    
                                        ds_pca_pycortex = dict()
                                        ds_pca_roi_mean = dd(lambda:dict())  
                                        
                                        
                                        compute_pca_on_roi_means = False
                                        #pca_params_correlations = False
                                        full_data_correlations = False
                                        

                                                                           
                                        #excluding rois based on position
                                        rr_rois = [r for r in rois if r != 'Brain' and r != 'combined' and np.sum(alpha[analysis][subj][r]>rsq_thresh)>10]
                                        
                                        if compute_pca_on_roi_means:
                       
                                            pca_roi_means_dims = x_dims+y_dims
    
                                            for pca_roi_means_dim in pca_roi_means_dims:
                                                pca_roi_means_dim_name = ordered_dimensions[pca_roi_means_dim]
                                                
                                                for rr in rr_rois:
                                                    
                                                    
                                                    data = multidim_param_array[analysis][subj][rr][pca_roi_means_dim]
                                                    
                                                    weights = alpha[analysis][subj][rr][alpha[analysis][subj][rr]>rsq_thresh]
                                                    
                                                    if not rsq_weights or 'Norm_abcd' not in pca_roi_means_dim_name:
                                                        weights = np.ones_like(weights)
    
                                                    roi_wstats = weightstats.DescrStatsW(data,
                                                                                        weights=weights)
    
                                                    ds_pca_roi_mean[rr][f"PCA data Dim {pca_roi_means_dim_name}"] = roi_wstats.mean
                                                    ds_pca_roi_mean[rr][f"PCA data Dim {pca_roi_means_dim_name} stdev"] = roi_wstats.std_mean   
                                                    
                                                    ds_pca_roi_mean[rr][f"Mean PCA weights {pca_roi_means_dim_name}"] = weights.mean()
                                                    
                                            pca_roi_means_array = np.array([[ds_pca_roi_mean[rr][f"PCA data Dim {pca_roi_means_dim_name}"] for rr in ds_pca_roi_mean] for pca_roi_means_dim_name in xy_dims]).T
                                            pca_roi_means_weights = np.array([[ds_pca_roi_mean[rr][f"Mean PCA weights {pca_roi_means_dim_name}"] for rr in ds_pca_roi_mean] for pca_roi_means_dim_name in xy_dims]).T
                                            
                                            
                                            pca_roi_means_array = zscore(pca_roi_means_array, axis=0)
                                            
        
                                            if rsq_weights:
                                                pca_on_roi_means = WPCA(n_components=ncomp).fit(pca_roi_means_array,weights=pca_roi_means_weights)
                                            else:
                                                pca_on_roi_means = WPCA(n_components=ncomp).fit(pca_roi_means_array)
    
    
    
                                            
                                            f_rm, axes_rm = pl.subplots(1, ncomp_bar_plots, figsize=(ncomp_bar_plots*8,8))
                                            
                                           
                                            for j in range(ncomp_bar_plots):                                           
                                                if j == 0:
                                                    axes_rm[j].set_yticks(np.arange(full_dataset.shape[1])/2)
                                                    axes_rm[j].set_yticklabels([ordered_dimensions[x_dim].replace('Norm_abcd','') for x_dim in x_dims]+[ordered_dimensions[y_dim].replace('Norm_abcd','') for y_dim in y_dims])
                                                else:
                                                    axes_rm[j].set_yticks([])
                                                    
                                                axes_rm[j].barh(np.arange(full_dataset.shape[1])/2, np.array(pca_on_roi_means.components_)[j,:], height=0.4, color=['red' if c>0 else 'blue' for c in np.sign(np.array(pca_on_roi_means.components_)[j,:])])#, label=f"RSq {pca.explained_variance_ratio_[j]:.3f}")
                                                 
                                                if cv_regression:
                                                    print(f"PCA RM comp {j+1} (R2={pca_on_roi_means.explained_variance_ratio_[j]:.2f}, cvR2={cv_rsq_comp[j]:.2f})")
                                                    axes_rm[j].set_title(f"PCA comp {j+1} (R2={pca_on_roi_means.explained_variance_ratio_[j]:.2f}, cvR2={cv_rsq_comp[j]:.2f})")
                                                else:
                                                    print(f"PCA RM comp {j} (R2={pca_on_roi_means.explained_variance_ratio_[j]:.2f})")
                                                    axes_rm[j].set_title(f"PCA RM comp {j+1} (R2={pca_on_roi_means.explained_variance_ratio_[j]:.2f})")
    
    



                                        pca_comp_corr_dims = list(set([dim for dim, dim_name in enumerate(ordered_dimensions) if 'Norm_abcd' in dim_name and 'RSq' not in dim_name]))
                                       

                                        for pca_comp_corr_dim in pca_comp_corr_dims:
                                            corr_dim_name = ordered_dimensions[pca_comp_corr_dim]
                                            
                                            for rr in rr_rois:
                                                
                                                
                                                data = multidim_param_array[analysis][subj][rr][pca_comp_corr_dim]
                                                
                                                weights = alpha[analysis][subj][rr][alpha[analysis][subj][rr]>rsq_thresh]
                                                
                                                if not rsq_weights or 'Norm_abcd' not in corr_dim_name:
                                                    weights = np.ones_like(weights)

                                                roi_wstats = weightstats.DescrStatsW(data,
                                                                                    weights=weights)

                                                ds_pca_roi_mean[rr][f"Corr Dim {corr_dim_name}"] = roi_wstats.mean
                                                ds_pca_roi_mean[rr][f"Corr Dim {corr_dim_name} stdev"] = roi_wstats.std_mean



                                            
                                                                                        
                                        for c in range(ncomp):
                                            
                                            zz = np.zeros_like(alpha[analysis][subj][roi]).astype(float)
                                            
                                            if rsq_weights:
                                                zz[alpha[analysis][subj][roi]>rsq_thresh] = pca.fit_transform(full_dataset,weights=wpca_weights)[:,c]
                                                                                
                                                
                                            else:
                                                zz[alpha[analysis][subj][roi]>rsq_thresh] = pca.fit_transform(full_dataset)[:,c]
                                                

                                                
                                            #saving pca result for later 1d plotting
                                            #subj_res['Processed Results']['Receptor Maps'][f"PCA Component {c}"] = np.copy(zz)
                                                
                                                
                                            for rr in rr_rois:
                                                
                                                data = zz[alpha[analysis][subj][rr]>rsq_thresh]
                                                
                                                weights = alpha[analysis][subj][rr][alpha[analysis][subj][rr]>rsq_thresh]
                                                
                                                ds_pca_roi_mean[rr]["Mean rsq"] = weights.mean()
                                                
                                                if not rsq_weights or np.all(wpca_weights == 1):
                                                    weights = np.ones_like(weights)


                                                roi_wstats = weightstats.DescrStatsW(data,
                                                                                    weights=weights)

                                                ds_pca_roi_mean[rr][f"Component {c}"] = roi_wstats.mean
                                                ds_pca_roi_mean[rr][f"Component {c} stdev"] = roi_wstats.std_mean
                                                
                                                
                                                
                                                
                                            pca_allrois_rsq_means = np.array([ds_pca_roi_mean[rr]["Mean rsq"] for rr in ds_pca_roi_mean])
                                            
                                            
                                            
                                            if compute_pca_on_roi_means:
                                                #using the pca done on roi means for correlations with other parameters
                                                if rsq_weights:
    
                                                    pca_allrois_means = pca_on_roi_means.fit_transform(pca_roi_means_array, weights=pca_roi_means_weights)[:,c]
                                                    
                                                else:
                                                    
                                                    pca_allrois_means = pca_on_roi_means.fit_transform(pca_roi_means_array)[:,c]    
                                            else:
                                                #computing the full data PCA result (roi means) for correlations with other parameters
                                                pca_allrois_means = np.array([ds_pca_roi_mean[rr][f"Component {c}"] for rr in ds_pca_roi_mean])
                                            

     
                                            
                                            for pca_comp_corr_dim in pca_comp_corr_dims:
                                                
                                                corr_dim_name = ordered_dimensions[pca_comp_corr_dim]
                                                if full_data_correlations:
                                                    corr_dim = multidim_param_array[analysis][subj][roi][pca_comp_corr_dim]
                                                    wCC_pca_params = weightstats.DescrStatsW(np.stack((zz[alpha[analysis][subj][roi]>rsq_thresh],corr_dim)).T, weights=alpha[analysis][subj][roi][alpha[analysis][subj][roi]>rsq_thresh]).corrcoef[0,1]
                                                else:
                                                    
                                                
                                                    pca_corr_dim_allrois_means = np.array([ds_pca_roi_mean[rr][f"Corr Dim {corr_dim_name}"] for rr in ds_pca_roi_mean])
                                                
                                                    wCC_pca_params = weightstats.DescrStatsW(np.stack((pca_allrois_means,pca_corr_dim_allrois_means)).T, weights=pca_allrois_rsq_means).corrcoef[0,1]
                                                

                                                CC_boot = []
                                                for perm in range(10):                                        

                                                    if full_data_correlations:
                                                        samp_idx = np.random.randint(0, len(corr_dim), int(len(corr_dim)/upsampling_corr_factor))                                        
                                                        perm_x = corr_dim[samp_idx]
                                                        perm_y = zz[alpha[analysis][subj][roi]>rsq_thresh][samp_idx]
                                                        perm_w = alpha[analysis][subj][roi][alpha[analysis][subj][roi]>rsq_thresh][samp_idx]     
                                                    else:
                                                        samp_idx = np.random.randint(0, len(pca_allrois_means), len(pca_allrois_means))
                                                        perm_x = pca_allrois_means[samp_idx]
                                                        perm_y = pca_corr_dim_allrois_means[samp_idx]
                                                        perm_w = pca_allrois_rsq_means[samp_idx] 
        
   
                                                                                 
                                                    np.random.shuffle(perm_x)
                                                    np.random.shuffle(perm_y)
                                                    np.random.shuffle(perm_w)
                                                    CC_boot.append(weightstats.DescrStatsW(np.stack((perm_x,perm_y)).T, weights=perm_w).corrcoef[0,1])
                                                
                                                pval_wcc = np.sum(np.abs(wCC_pca_params)<np.abs(np.array(CC_boot)))/len(CC_boot)
                                                
                                                
                                                if pval_wcc<1e-2:
                                                    pval_string = '*'
                                                    if pval_wcc<1e-3:
                                                        pval_string = '**'
                                                        if pval_wcc<1e-4:
                                                            pval_string = '***'                                                                                        
                                                else:
                                                    pval_string = 'n.s.'                                                
                                                
                                                if full_data_correlations:
                                                    print(f"wCC (full data) PCA comp {c+1} VS {corr_dim_name}: {wCC_pca_params:.4f} ---")
                                                else:
                                                    print(f"wCC (roi means) PCA comp {c+1} VS {corr_dim_name}: {wCC_pca_params:.4f} ---")
                         

                       
                                            
                                            if vis_pca_pycortex:
                                                if rsq_thresh<0.15:
                                                    print("rsq thresh<0.15. setting pca pycortex vmax2=0.15")
                                                    pca_vmax2 = 0.15
                                                else:
                                                    pca_vmax2 = np.nanquantile(alpha[analysis][subj][roi][alpha[analysis][subj][roi]>rsq_thresh],0.1)
                                                ds_pca_pycortex[f"Component {c}"] = Vertex2D_fix(zz, alpha[analysis][subj][roi], subject=pycortex_subj, 
                                                                vmin=np.nanquantile(zz[alpha[analysis][subj][roi]>rsq_thresh],0.1), vmax=np.nanquantile(zz[alpha[analysis][subj][roi]>rsq_thresh],0.9), 
                                                                 vmin2=rsq_thresh, vmax2=pca_vmax2, cmap='nipy_spectral')

                                        
                                        fig = pl.figure(f"{subj} PCA components {''.join(str(e) for e in vis_pca_comps_axes)} {roi.replace('custom.','').replace('HCPQ1Q6.','').replace('glasser_','')}", figsize=(8,8))
                                        
                                        if len(vis_pca_comps_axes)>2:
                                            ax = fig.add_subplot(111, projection='3d', azim=-45, elev=50)
                                            #ax.grid(False)
                                            ax.set_zlabel(f"PCA Component {vis_pca_comps_axes[2]+1}",labelpad=50) 
                                            ax.zaxis.set_tick_params(pad=20)
                                            ax.set_xlabel(f"PCA Component {vis_pca_comps_axes[0]+1}",labelpad=50)
                                            ax.set_ylabel(f"PCA Component {vis_pca_comps_axes[1]+1}",labelpad=50)                                        
                                            ax.xaxis.set_tick_params(pad=20)                                        
                                            ax.yaxis.set_tick_params(pad=20)       
                                            
                                            # make the panes 
                                            ax.xaxis.set_pane_color((0.5, 0.5, 0.5, 0.1))
                                            ax.yaxis.set_pane_color((0.5, 0.5, 0.5, 0.1))
                                            ax.zaxis.set_pane_color((0.5, 0.5, 0.5, 0.1))
                                            # make the grid lines 
                                            ax.xaxis._axinfo["grid"]['color'] =  (0.5, 0.5, 0.5, 0.2)
                                            ax.yaxis._axinfo["grid"]['color'] =  (0.5, 0.5, 0.5, 0.2)
                                            ax.zaxis._axinfo["grid"]['color'] =  (0.5, 0.5, 0.5, 0.2)                                           
                                                                                        
                                            
                                            
                                        else:
                                            ax = fig.add_subplot(111)
                                            

                                            ax.set_xlabel(f"PCA Component {vis_pca_comps_axes[0]+1}")
                                            ax.set_ylabel(f"PCA Component {vis_pca_comps_axes[1]+1}") 
                                        
                                        rsq_alpha_plots_all_rois = [ds_pca_roi_mean[rr]["Mean rsq"] for rr in ds_pca_roi_mean]
                                        
                                        
                                        #keep like this to preserve color order
                                        for rr_num,rr in enumerate([rr for rr in rois if rr != 'Brain' and rr != 'combined']):
                                            
                                            if rr in rr_rois:
                                                                                  
                                            
                                                if rsq_weights and rsq_alpha_pca_plot:    
                                                    rsq_alpha_plot_max = np.nanmax(rsq_alpha_plots_all_rois)
                                                    rsq_alpha_plot_min = np.nanmin(rsq_alpha_plots_all_rois)   
                                                                                              
                                                    alpha_plot = (ds_pca_roi_mean[rr]["Mean rsq"]-rsq_alpha_plot_min)/(rsq_alpha_plot_max-rsq_alpha_plot_min) #ds_pca_roi_mean[rr]["Mean rsq"]/rsq_alpha_plot_max#
                                                    
                                                else:
                                                    alpha_plot = 1
    
                                                alpha_plot = np.clip(alpha_plot,0,1)
                                                if np.isnan(alpha_plot) or not np.isfinite(alpha_plot):
                                                    alpha_plot = 0     
                                                    
                                                
                                                #
                                                #
                                                
                                                if len(vis_pca_comps_axes)>2:
                                                    #ax.set_xlim(-5.25,3.25)
                                                    #ax.set_ylim(-2.75,2.25)
                                                    #ax.set_zlim(-2.25,3.25)
                                                    ax.errorbar(ds_pca_roi_mean[rr][f"Component {vis_pca_comps_axes[0]}"], ds_pca_roi_mean[rr][f"Component {vis_pca_comps_axes[1]}"], ds_pca_roi_mean[rr][f"Component {vis_pca_comps_axes[2]}"],
                                                                xerr=0,#ds_pca_roi_mean[rr][f"Component {vis_pca_comps_axes[0]} stdev"]*upsampling_corr_factor**0.5, 
                                                                yerr=0,#ds_pca_roi_mean[rr][f"Component {vis_pca_comps_axes[1]} stdev"]*upsampling_corr_factor**0.5, 
                                                                zerr=0,#ds_pca_roi_mean[rr][f"Component {vis_pca_comps_axes[2]} stdev"]*upsampling_corr_factor**0.5,
                                                                color=cmap_rois[rr_num], fmt='s', mec='black', alpha=alpha_plot, ms=16, mew=2)
                                                    
                                                    roi_name_txt = ax.text(ds_pca_roi_mean[rr][f"Component {vis_pca_comps_axes[0]}"], ds_pca_roi_mean[rr][f"Component {vis_pca_comps_axes[1]}"], ds_pca_roi_mean[rr][f"Component {vis_pca_comps_axes[2]}"], 
                                                            rr.replace('custom.','').replace('HCPQ1Q6.','') .replace('glasser_','') , 
                                                            fontsize=25, color=cmap_rois[rr_num],  ha='left', va='bottom',alpha=alpha_plot)
                                                    roi_name_txt.set_path_effects([peff.withStroke(linewidth=1, foreground='k')])
                                                    
                                                else:
                                                    ax.errorbar(ds_pca_roi_mean[rr][f"Component {vis_pca_comps_axes[0]}"], ds_pca_roi_mean[rr][f"Component {vis_pca_comps_axes[1]}"], 
                                                                xerr=0,#ds_pca_roi_mean[rr][f"Component {vis_pca_comps_axes[0]} stdev"]*upsampling_corr_factor**0.5, 
                                                                yerr=0,#ds_pca_roi_mean[rr][f"Component {vis_pca_comps_axes[1]} stdev"]*upsampling_corr_factor**0.5, 
                                                                color=cmap_rois[rr_num], fmt='s', mec='black', alpha=alpha_plot, ms=16, mew=2)
                                                    
                                                    roi_name_txt = ax.text(ds_pca_roi_mean[rr][f"Component {vis_pca_comps_axes[0]}"], ds_pca_roi_mean[rr][f"Component {vis_pca_comps_axes[1]}"], 
                                                            rr.replace('custom.','').replace('HCPQ1Q6.','') .replace('glasser_','') , 
                                                            fontsize=25, color=cmap_rois[rr_num],  ha='left', va='bottom', alpha=alpha_plot)
                                                    
                                                    roi_name_txt.set_path_effects([peff.withStroke(linewidth=1, foreground='k')])


                                                

                                            
                                        if save_figures:
                                            pl.savefig(opj(figure_path,f"PCA_mean_roi_dims-{''.join(str(e) for e in vis_pca_comps_axes)}_{roi.replace('custom.','').replace('HCPQ1Q6.','').replace('glasser_','')}.pdf"),dpi=600, bbox_inches='tight')
                                    
                                            
                                        if vis_pca_pycortex:
                                            cortex.webgl.show(ds_pca_pycortex, with_curvature=False, with_labels=True, with_rois=True, with_borders=True, with_colorbar=True)    
                                    
                                ###########1D regressions
                                if perform_ols and np.sum(alpha[analysis][subj][roi]>rsq_thresh)>10:
                                    self.curr_rois_names.append(roi)
                                    print("performing OLS (1D y)")
                                    
                                    for y_dim in y_dims:
                                        
                                        y = multidim_param_array[analysis][subj][roi][y_dim].T
                                        
                                        if len(x_dims) == 2:
                                            ols_x0_dim = [ordered_dimensions[x_dim] for x_dim in x_dims][vis_pca_comps_axes[0]]
                                            ols_x1_dim = [ordered_dimensions[x_dim] for x_dim in x_dims][vis_pca_comps_axes[1]]
                                        ols_y_dim = ordered_dimensions[y_dim]
                                        
                                        
                                        ls1 = LinearRegression()
                                        if rsq_weights:
                                            ls1.fit(X,y,sample_weight = multidim_param_array[analysis][subj][roi][ordered_dimensions.index('RSq Norm_abcd')])
                                            rsq_prediction = ls1.score(X,y,sample_weight = multidim_param_array[analysis][subj][roi][ordered_dimensions.index('RSq Norm_abcd')])
                                        else: 
                                            ls1.fit(X,y)
                                            rsq_prediction = ls1.score(X,y)
                                            
                                        print(f"Estimated OLS betas (full data) {ordered_dimensions[y_dim]} {ls1.coef_}")
                                        print(f"wRSq (full data) {rsq_prediction}")
                                        corr_prediction = weightstats.DescrStatsW(np.stack((ls1.predict(X), y)).T, weights=multidim_param_array[analysis][subj][roi][ordered_dimensions.index('RSq Norm_abcd')]).corrcoef[0,1]
                                        print(f"wCC prediction (full data) {corr_prediction}")
                                        
                                        if roi in vis_pca_comps_rois:

                                            ds_ols_roi_mean = dd(lambda:dict())                          
                                                                                            
                                            for rr in [rr for rr in rois if rr != 'Brain' and rr != 'combined' and np.sum(alpha[analysis][subj][rr]>rsq_thresh)>10]:
                                                
                                                weights = alpha[analysis][subj][rr][alpha[analysis][subj][rr]>rsq_thresh]
                                                rr_y = multidim_param_array[analysis][subj][rr][y_dim][weights>rsq_thresh]
                                                
                                                
                                                for x_dim in x_dims:
                                                    rr_x = multidim_param_array[analysis][subj][rr][x_dim][weights>rsq_thresh]
                                                    
                                                    if not rsq_weights or 'Norm_abcd' not in ordered_dimensions[x_dim]:
                                                        weights_x = np.ones_like(weights)
                                                    else:
                                                        weights_x = weights
                                                    
                                                    rr_wstats_x = weightstats.DescrStatsW(rr_x,
                                                                                    weights=weights_x)
                                                    
                                                    ds_ols_roi_mean[rr][f"{ordered_dimensions[x_dim]} mean"] = rr_wstats_x.mean
                                                    ds_ols_roi_mean[rr][f"{ordered_dimensions[x_dim]} stdev"] = rr_wstats_x.std_mean                                                    
                                                    
                                                    
                                                if not rsq_weights or 'Norm_abcd' not in ols_y_dim:
                                                    weights_y = np.ones_like(weights)
                                                else:
                                                    weights_y = weights                                                    


                                                rr_wstats_y = weightstats.DescrStatsW(rr_y,
                                                                                    weights=weights_y)
                                                                                                                                                

                                                ds_ols_roi_mean[rr][f"{ols_y_dim} mean"] = rr_wstats_y.mean
                                                ds_ols_roi_mean[rr][f"{ols_y_dim} stdev"] = rr_wstats_y.std_mean
                                                
                                                ds_ols_roi_mean[rr]["Mean rsq"] = weights_y.mean()

    
                                            rsq_alpha_plots_all_rois = np.array([ds_ols_roi_mean[rr]["Mean rsq"] for rr in ds_ols_roi_mean])
                                            
                                            rois_ols_x_means = np.array([[ds_ols_roi_mean[rr][f"{ordered_dimensions[x_dim]} mean"] for x_dim in x_dims] for rr in ds_ols_roi_mean])

                                            rois_ols_y_means = np.array([ds_ols_roi_mean[rr][f"{ols_y_dim} mean"] for rr in ds_ols_roi_mean])                                                                                      
                                            
                                            rsq_on_roi_means = ls1.score(rois_ols_x_means, rois_ols_y_means, sample_weight = rsq_alpha_plots_all_rois)
                                            

                                            Zs_pred = ls1.predict(rois_ols_x_means)
                                                                                        
                                            
                                            print(f"wR2 (fit on full data, eval on roi means): {rsq_on_roi_means:.3f}")                                       

                                            corr_prediction_roi_means = weightstats.DescrStatsW(np.stack((Zs_pred,rois_ols_y_means)).T, weights=np.array(rsq_alpha_plots_all_rois)).corrcoef[0,1]
                                            print(f"wCC prediction (fit on full data, eval on roi means) {corr_prediction_roi_means:.4f}\n") 
                                            
                                            ls2 = LinearRegression()
                                            ls2.fit(rois_ols_x_means, rois_ols_y_means, sample_weight = rsq_alpha_plots_all_rois)
                                            Zs_pred_fit_roi_means = ls2.predict(rois_ols_x_means)
                                            
                                            print(f"wR2 (fit on roi means, eval on roi means): {ls2.score(rois_ols_x_means, rois_ols_y_means, sample_weight = rsq_alpha_plots_all_rois):.4f}")

                                            corr_prediction_fit_roi_means = weightstats.DescrStatsW(np.stack((Zs_pred_fit_roi_means,rois_ols_y_means)).T , weights=np.array(rsq_alpha_plots_all_rois)).corrcoef[0,1]

                                            print(f"wCC prediction (fit on roi means, eval on roi means) {corr_prediction_fit_roi_means:.4f}")    
                                            print(f"Estimated OLS betas (fit on roi means) {ordered_dimensions[y_dim]} {ls2.coef_}\n")
                                            
                                            
                                            if not cv_regression:
                                                for ddd in range(len(x_dims)):
                                                    
                                                    corr_prediction_fit_roi_means_perm = []
                                                    corr_prediction_perm_fd = []

                                                    w_fd = np.copy(multidim_param_array[analysis][subj][roi][ordered_dimensions.index('RSq Norm_abcd')])
                                                    x_perm_fd = np.copy(X)

                                                    eigvecs_reduced, ft_x = reduced_graph_ft(x_perm_fd[:,ddd], alpha[analysis][subj][roi]>rsq_thresh, 
                                                                                            eigenvectors_path='/Users/marcoaqil/1000eigvecs_full_quantrois.npy', 
                                                                                            eigenvectors_indices_path='/Users/marcoaqil/full_quantrois.npy', 
                                                                                            pycortex_subj=pycortex_subj)
                                                    data_max_x_fd = x_perm_fd[:,ddd].max()
                                                    data_min_x_fd = x_perm_fd[:,ddd].min()

                                                    

                                                    perms = 100
                                                    print(f'computing OLS stats with {perms} permutations')
                                                    
                                                    if perms<1000000:
                                                        print('WARNING: less than 10^6 permutations. stats probably unreliable. use for visualization only')
                                                    
                                                    for perm in tqdm(range(perms)):
                                                        # testing whether adding parameters significantly increases wCC (nocv here)
                                                        # the idea is asking how likely it is to get the same wCC improvement when adding a randomized version of the parameter, instead of the true one
                                                        
                                                        ls1_perm = LinearRegression()
                                                        #samp_idx = np.random.randint(0, len(y), int(len(y)/upsampling_corr_factor))

                                                        x_perm_curr_dim_fd = graph_randomization(data_max_x_fd, data_min_x_fd, eigvecs_reduced, ft_x)#[samp_idx]  
                                                        
                                                        x_perm_fd[:,ddd] = x_perm_curr_dim_fd

                                                        #y_perm_fd = np.copy(y)#[samp_idx]
                                                        #w_perm_fd = np.copy(multidim_param_array[analysis][subj][roi][ordered_dimensions.index('RSq Norm_abcd')])#[samp_idx]                                                     
                                                        
                                                        ls1_perm.fit(x_perm_fd, y, sample_weight = w_fd)
                                                        Zs_pred_perm_fd = ls1_perm.predict(x_perm_fd)
     
                                                        corr_prediction_perm_fd.append(weightstats.DescrStatsW(np.stack((Zs_pred_perm_fd,y)).T , weights=np.array(w_fd)).corrcoef[0,1])
                                                                                                           
                                                        
                                                        ls2_perm = LinearRegression()
                                                        #samp_idx = np.random.randint(0, len(rois_ols_x_means), len(rois_ols_x_means))
                                                        x_perm = np.copy(rois_ols_x_means)#[samp_idx])
                                                        y_perm = np.copy(rois_ols_y_means)#[samp_idx])
                                                        w_perm = np.copy(rsq_alpha_plots_all_rois)#[samp_idx])
                                                        
                                                        #x_perm[:,ddd] = np.copy(x_perm[samp_idx,ddd])
                                                        np.random.shuffle(x_perm[:,ddd])
                                                        #np.random.shuffle(y_perm)
                                                        #np.random.shuffle(w_perm)
                                                        ls2_perm.fit(x_perm, y_perm, sample_weight = w_perm)
                                                        Zs_pred_fit_roi_means_perm = ls2_perm.predict(x_perm)
     
                                                        corr_prediction_fit_roi_means_perm.append(weightstats.DescrStatsW(np.stack((Zs_pred_fit_roi_means_perm,y_perm)).T , weights=np.array(w_perm)).corrcoef[0,1])
                                                
                                                
                                                    pval_wcc_roimeans = np.sum(np.abs(corr_prediction_fit_roi_means)<np.abs(np.array(corr_prediction_fit_roi_means_perm)))/len(corr_prediction_fit_roi_means_perm)
                                                    self.ols_result_dict[ols_y_dim][f"{[ordered_dimensions[x_dim] for x_dim in x_dims]} {ordered_dimensions[x_dims[ddd]]} roi means"]['pval'].append(pval_wcc_roimeans)
                                                    print(f"{ordered_dimensions[x_dims[ddd]]} roi means pval={pval_wcc_roimeans:.7f}")                                            


                                                    pval_wcc_fd = np.sum(np.abs(corr_prediction)<np.abs(np.array(corr_prediction_perm_fd)))/len(corr_prediction_perm_fd)
                                                    self.ols_result_dict[ols_y_dim][f"{[ordered_dimensions[x_dim] for x_dim in x_dims]} {ordered_dimensions[x_dims[ddd]]} full data"]['pval'].append(pval_wcc_fd)
                                                    print(f"{ordered_dimensions[x_dims[ddd]]} full data pval={pval_wcc_fd:.7f}")     
                                            
                                                                                        
                                            
                                            print(f"wR2 (fit on full data, eval on full data): {rsq_prediction:.4f}")
                                            print(f"wCC prediction (fit on full data, eval on full data) {corr_prediction:.4f}")
                                            print(f"Estimated OLS betas (fit on full data) {ordered_dimensions[y_dim]} {ls1.coef_}")
                                            
                                            if len(x_dims) == 2:
                                                self.ols_3d_fig = pl.figure(f"3D OLS prediction {roi.replace('custom.','').replace('HCPQ1Q6.','').replace('glasser_','')}", figsize=(12,12))
                                                
                                                ax = self.ols_3d_fig.add_subplot(111, projection='3d', azim=-45, elev=50)
                                                #ax.grid(False)
    
                                                ax.set_xlabel(f"{ols_x0_dim.replace('Norm_abcd','')}",labelpad=20)
                                                ax.set_ylabel(f"{ols_x1_dim.replace('Norm_abcd','')}",labelpad=20)
                                                ax.set_zlabel(f"{ols_y_dim.replace('Norm_abcd','')}",labelpad=20) 
                                                ax.xaxis.set_tick_params(pad=10)
                                                ax.zaxis.set_tick_params(pad=10)
                                                ax.yaxis.set_tick_params(pad=10)
                                                
                                                # make the panes 
                                                ax.xaxis.set_pane_color((0.5, 0.5, 0.5, 0.05))
                                                ax.yaxis.set_pane_color((0.5, 0.5, 0.5, 0.05))
                                                ax.zaxis.set_pane_color((0.5, 0.5, 0.5, 0.05))
                                                # make the grid lines 
                                                ax.xaxis._axinfo["grid"]['color'] =  (0.5, 0.5, 0.5, 0.15)
                                                ax.yaxis._axinfo["grid"]['color'] =  (0.5, 0.5, 0.5, 0.15)
                                                ax.zaxis._axinfo["grid"]['color'] =  (0.5, 0.5, 0.5, 0.15)                                           
                                                
                                            
                                            

                                            for rr_num,rr in enumerate([rr for rr in rois if rr != 'Brain' and rr != 'combined']):
                                                
                                                if np.sum(alpha[analysis][subj][rr]>rsq_thresh)>10:
                                      
                                                    if rsq_weights and rsq_alpha_pca_plot:    
                                                        rsq_alpha_plot_max = np.nanmax(rsq_alpha_plots_all_rois)
                                                        rsq_alpha_plot_min = np.nanmin(rsq_alpha_plots_all_rois)   
                                                                                                  
                                                        alpha_plot = (ds_ols_roi_mean[rr]["Mean rsq"]-rsq_alpha_plot_min)/(rsq_alpha_plot_max-rsq_alpha_plot_min) #ds_ols_roi_mean[rr]["Mean rsq"]/rsq_alpha_plot_max#
                                                        
                                                    else:
                                                        alpha_plot = 1
        
                                                    alpha_plot = np.clip(alpha_plot,0,1)
                                                    if np.isnan(alpha_plot) or not np.isfinite(alpha_plot):
                                                        alpha_plot = 0                                            
                                                    
                                                    if len(x_dims) == 2:
                                                        ax.errorbar(ds_ols_roi_mean[rr][f"{ols_x0_dim} mean"], ds_ols_roi_mean[rr][f"{ols_x1_dim} mean"], ds_ols_roi_mean[rr][f"{ols_y_dim} mean"],
                                                                    xerr=0,#ds_ols_roi_mean[rr][f"{ols_x0_dim} stdev"]*upsampling_corr_factor**0.5, 
                                                                    yerr=0,#ds_ols_roi_mean[rr][f"{ols_x1_dim} stdev"]*upsampling_corr_factor**0.5, 
                                                                    zerr=0,#ds_ols_roi_mean[rr][f"{ols_y_dim} stdev"]*upsampling_corr_factor**0.5,
                                                                    color=cmap_rois[rr_num], fmt='s', mec='black', alpha=alpha_plot, ms=16, mew=2)
                                                        
                                                        roi_name_txt = ax.text(ds_ols_roi_mean[rr][f"{ols_x0_dim} mean"], ds_ols_roi_mean[rr][f"{ols_x1_dim} mean"], ds_ols_roi_mean[rr][f"{ols_y_dim} mean"],
                                                                rr.replace('custom.','').replace('HCPQ1Q6.','') .replace('glasser_','') , 
                                                                fontsize=25, color=cmap_rois[rr_num],  ha='left', va='bottom', alpha=alpha_plot)
                                                        roi_name_txt.set_path_effects([peff.withStroke(linewidth=1, foreground='k')])
                                            
                                            if len(x_dims) == 2:
                                                x0_min = np.min([ds_ols_roi_mean[rr][f"{ols_x0_dim} mean"] for rr in ds_ols_roi_mean])
                                                x0_max = np.max([ds_ols_roi_mean[rr][f"{ols_x0_dim} mean"] for rr in ds_ols_roi_mean])
                                                x1_min = np.min([ds_ols_roi_mean[rr][f"{ols_x1_dim} mean"] for rr in ds_ols_roi_mean])
                                                x1_max = np.max([ds_ols_roi_mean[rr][f"{ols_x1_dim} mean"] for rr in ds_ols_roi_mean])
                                                y_min = np.min([ds_ols_roi_mean[rr][f"{ols_y_dim} mean"] for rr in ds_ols_roi_mean])
                                                y_max = np.max([ds_ols_roi_mean[rr][f"{ols_y_dim} mean"] for rr in ds_ols_roi_mean])

                                                
                                                grid_points_x0 = np.linspace(x0_min,x0_max,50)
                                                grid_points_x1 = np.linspace(x1_min,x1_max,50)
    
                                                
                                                if 'Norm Param. D' in ols_y_dim:
                                                    ax.set_zlim(23,72)
                                                # if '5-HT1B' in ols_x0_dim:
                                                #     ax.set_xlim(13,32)
                                                #     grid_points_x0 = np.linspace(13,32,50)
                                                    
                                                # if '5-HT2A' in ols_x1_dim:
                                                #     ax.set_ylim(39,66)
                                                #     grid_points_x1 = np.linspace(39,66,50)
    
                                                if 'Norm Param. B' in ols_y_dim:
                                                    ax.set_zlim(-5,135)
                                                # if '5-HT1A' in ols_x0_dim:
                                                #     ax.set_xlim(7,43)
                                                #     grid_points_x0 = np.linspace(7,43,50)
                                                # if 'GABA_A' in ols_x1_dim:
                                                #     ax.set_ylim(720,1080)  
                                                #     grid_points_x1 = np.linspace(720,1080,50)
                                                
                                                
                                                xx0, xx1 = np.meshgrid(grid_points_x0,
                                                                       grid_points_x1)
                                                
                                                xx0xx1 = np.array([xx0.flatten(), xx1.flatten()]).T
                                                
                                                Z_pred = ls2.predict(xx0xx1)                                            
                                                
                                                
                                                #ax.set_title(f"$R^2$={rsq_on_roi_means:.2f}")
                                                if 'Param. D' in ols_y_dim:
                                                    pred_cmap = 'plasma_r'
                                                elif 'Param. B' in ols_y_dim:
                                                    pred_cmap = 'viridis_r'
                                                else:
                                                    pred_cmap = 'plasma_r'

                                                    
                                                ax.scatter(xx0.flatten(), xx1.flatten(), Z_pred,  s=4,  alpha=0.2, zorder=0, c=Z_pred, cmap=pred_cmap)#, vmin=np.nanquantile(Z_pred,0.1), vmax=np.nanquantile(Z_pred,0.95))#, label=f"R2 (full data) {rsq_prediction:.2f}")
                                                #pl.legend()
                                                
                                                
                                                # # make grid of points
                                                # x, y, z = np.mgrid[x0_min:x0_max:50, x1_min:x1_max:50, y_min:y_max:50]
                                                # points = np.random.normal(size=(3, 50))
                                                # kernel = gaussian_kde(points)
                                                # positions = np.vstack((x.ravel(), y.ravel(), z.ravel()))
                                                # density = np.reshape(kernel(positions).T, x.shape)                                                
                                                # # plot projection of density onto z-axis
                                                # plotdat = np.sum(density, axis=2)
                                                # plotdat = plotdat / np.max(plotdat)
                                                # plotx, ploty = np.mgrid[-4:4:100j, -4:4:100j]
                                                # ax.contour(plotx, ploty, plotdat, offset=-4, zdir='z')
                                                
                                                # #This is new
                                                # #plot projection of density onto y-axis
                                                # plotdat = np.sum(density, axis=1) #summing up density along y-axis
                                                # plotdat = plotdat / np.max(plotdat)
                                                # plotx, plotz = np.mgrid[-4:4:100j, -4:4:100j]
                                                # ax.contour(plotx, plotdat, plotz, offset=4, zdir='y')
                                                
                                                # #plot projection of density onto x-axis
                                                # plotdat = np.sum(density, axis=0) #summing up density along z-axis
                                                # plotdat = plotdat / np.max(plotdat)
                                                # ploty, plotz = np.mgrid[-4:4:100j, -4:4:100j]
                                                # ax.contour(plotdat, ploty, plotz, offset=-4, zdir='x')
                                                # #ax.set_title(f"R2 (full data) {rsq_prediction:.2f}")
                                                # #ax.set_zlim(15,40)                                                
                                        
                                        
                                            if cv_regression:
                                                for cv_subj in [s for s,_ in subjects if s!='fsaverage' and s!='Group' and s!=subj]:
                                                    print(f"CV regression (fit on {subj}, test on {cv_subj})")
    
                                                    ds_ols_roi_mean_cv = dd(lambda:dict())                          
                                                                                                    
                                                    for rr in [rr for rr in rois if rr != 'Brain' and rr != 'combined' and np.sum(alpha[analysis][cv_subj][rr]>rsq_thresh)>10]:



                                                        weights_cv = alpha[analysis][cv_subj][rr][alpha[analysis][cv_subj][rr]>rsq_thresh]
                                                        rr_y_cv = multidim_param_array[analysis][cv_subj][rr][y_dim][weights_cv>rsq_thresh]
                                                        
                                                        
                                                        for x_dim in x_dims:
                                                            rr_x_cv = multidim_param_array[analysis][cv_subj][rr][x_dim][weights_cv>rsq_thresh]
                                                            
                                                            if not rsq_weights or 'Norm_abcd' not in ordered_dimensions[x_dim]:
                                                                weights_x_cv = np.ones_like(weights_cv)
                                                            else:
                                                                weights_x_cv = weights_cv
                                                            
                                                            rr_wstats_x_cv = weightstats.DescrStatsW(rr_x_cv,
                                                                                            weights=weights_x_cv)
                                                            
                                                            ds_ols_roi_mean_cv[rr][f"{ordered_dimensions[x_dim]} mean"] = rr_wstats_x_cv.mean
                                                            ds_ols_roi_mean_cv[rr][f"{ordered_dimensions[x_dim]} stdev"] = rr_wstats_x_cv.std_mean                                                    
                                                            
                                                            
                                                        if not rsq_weights or 'Norm_abcd' not in ols_y_dim:
                                                            weights_y_cv = np.ones_like(weights_cv)
                                                        else:
                                                            weights_y_cv = weights_cv                                                    
        
        
                                                        rr_wstats_y_cv = weightstats.DescrStatsW(rr_y_cv,
                                                                                            weights=weights_y_cv)
                                                                                                                                                        
        
                                                        ds_ols_roi_mean_cv[rr][f"{ols_y_dim} mean"] = rr_wstats_y_cv.mean
                                                        ds_ols_roi_mean_cv[rr][f"{ols_y_dim} stdev"] = rr_wstats_y_cv.std_mean
                                                        
                                                        ds_ols_roi_mean_cv[rr]["Mean rsq"] = weights_y_cv.mean()



                                                    rsq_alpha_plots_all_rois_cv = np.array([ds_ols_roi_mean_cv[rr]["Mean rsq"] for rr in ds_ols_roi_mean_cv])
                                                    
                                                    rois_ols_x_means_cv = np.array([[ds_ols_roi_mean_cv[rr][f"{ordered_dimensions[x_dim]} mean"] for x_dim in x_dims] for rr in ds_ols_roi_mean_cv])
        
                                                    rois_ols_y_means_cv = np.array([ds_ols_roi_mean_cv[rr][f"{ols_y_dim} mean"] for rr in ds_ols_roi_mean_cv])                                                                                      
                                                    
                                                    rsq_on_roi_means_cv = ls1.score(rois_ols_x_means_cv, rois_ols_y_means_cv, sample_weight = rsq_alpha_plots_all_rois_cv)
                                                    
        
                                                    Zs_pred_cv = ls1.predict(rois_ols_x_means_cv)
                                                                                                
                                                    
                                                    print(f"wR2 (fit on {subj} full data, eval on {cv_subj} roi means): {rsq_on_roi_means_cv:.3f}")                                       
        
                                                    corr_prediction_roi_means_cv = weightstats.DescrStatsW(np.stack((Zs_pred_cv,rois_ols_y_means_cv)).T, weights=rsq_alpha_plots_all_rois_cv).corrcoef[0,1]
                                                    print(f"wCC prediction (fit on {subj} full data, eval on {cv_subj} roi means) {corr_prediction_roi_means_cv:.4f}\n") 
                                                    

                                                    Zs_pred_fit_roi_means_cv = ls2.predict(rois_ols_x_means_cv)
                                                    
                                                    print(f"wR2 (fit on {subj} roi means, eval on {cv_subj} roi means): {ls2.score(rois_ols_x_means_cv, rois_ols_y_means_cv, sample_weight = rsq_alpha_plots_all_rois_cv):.4f}")
                                                    
                                                    corr_prediction_fit_roi_means_cv = weightstats.DescrStatsW(np.stack((Zs_pred_fit_roi_means_cv,rois_ols_y_means_cv)).T , weights=rsq_alpha_plots_all_rois_cv).corrcoef[0,1]
                                                    print(f"wCC prediction (fit on {subj} roi means, eval on {cv_subj} roi means) {corr_prediction_fit_roi_means_cv:.4f}\n")    
                                                    
                                                    
                                                    # for ddd in range(len(x_dims)):
                                                        
                                                    #     corr_prediction_fit_roi_means_perm_cv = []
                                                    #     for perm in range(10000):
                                                    #         # testing whether adding parameters significantly increases wCC (nocv here)
                                                    #         # the idea is asking how likely it is to get the same wCC improvement when adding a randomized version of the parameter, instead of the true one
                                                    #         ls2_perm = LinearRegression()
                                                    #         #samp_idx = np.random.randint(0, len(rois_ols_x_means), len(rois_ols_x_means))
                                                    #         x_perm = np.copy(rois_ols_x_means)#[samp_idx])
                                                    #         y_perm = np.copy(rois_ols_y_means)#[samp_idx])
                                                    #         w_perm = np.copy(rsq_alpha_plots_all_rois)#[samp_idx])

                                                    #         np.random.shuffle(x_perm[:,ddd])#np.copy(x_perm[samp_idx,ddd])
                                                            
                                                            
                                                    #         #samp_idx_cv = np.random.randint(0, len(rois_ols_x_means_cv), len(rois_ols_x_means_cv))

                                                    #         x_perm_cv = np.copy(rois_ols_x_means_cv)#[samp_idx_cv])
                                                    #         y_perm_cv = np.copy(rois_ols_y_means_cv)#[samp_idx_cv])
                                                    #         w_perm_cv = np.copy(rsq_alpha_plots_all_rois_cv)#[samp_idx_cv])                                                                 
                                                    #         #np.random.shuffle(x_perm_cv[:,ddd])

                                                    #         ls2_perm.fit(x_perm, y_perm, sample_weight = w_perm)
                                                    #         Zs_pred_fit_roi_means_perm = ls2_perm.predict(x_perm_cv)
         
                                                    #         corr_prediction_fit_roi_means_perm_cv.append(weightstats.DescrStatsW(np.stack((Zs_pred_fit_roi_means_perm,y_perm_cv)).T , weights=w_perm_cv).corrcoef[0,1])
                                                    
                                                    
                                                    #     pval_wcc_roimeans_cv = np.sum(np.abs(corr_prediction_fit_roi_means_cv)<np.abs(np.array(corr_prediction_fit_roi_means_perm_cv)))/len(corr_prediction_fit_roi_means_perm_cv)
                                                        
                                                    #     print(f"{ordered_dimensions[x_dims[ddd]]} CV roimeans pval={pval_wcc_roimeans_cv:.5f}")     
                                                    #     self.ols_result_dict[ols_y_dim][f"{[ordered_dimensions[x_dim] for x_dim in x_dims]} {ordered_dimensions[x_dims[ddd]]}"]['pval_cv'].append(pval_wcc_roimeans_cv)
                                                    
                                                    

                                                    self.ols_result_dict[ols_y_dim][f"{[ordered_dimensions[x_dim] for x_dim in x_dims]}"]['wCC'].append(corr_prediction_fit_roi_means_cv)
                                                    self.ols_result_dict[ols_y_dim][f"{[ordered_dimensions[x_dim] for x_dim in x_dims]}"]['betas'].append(ls2.coef_)
                                                   
                                                    

                                                    X_cv = multidim_param_array[analysis][cv_subj][roi][x_dims].T
                                                    y_cv = multidim_param_array[analysis][cv_subj][roi][y_dim].T
                                                    
                                                    if rsq_weights:

                                                        rsq_prediction_cv = ls1.score(X_cv,y_cv,sample_weight = multidim_param_array[analysis][cv_subj][roi][ordered_dimensions.index('RSq Norm_abcd')])
                                                    else: 

                                                        rsq_prediction_cv = ls1.score(X_cv,y_cv)
                                                        

                                                    corr_prediction_cv = weightstats.DescrStatsW(np.stack((ls1.predict(X_cv), y_cv)).T, weights=multidim_param_array[analysis][cv_subj][roi][ordered_dimensions.index('RSq Norm_abcd')]).corrcoef[0,1]                                                    
                                                                                                            
                                                    
                                                    print(f"wR2 (fit on full data {subj}, eval on full data {cv_subj}): {rsq_prediction_cv:.4f}")
                                                    print(f"wCC prediction (fit on full data {subj}, eval on full data {cv_subj}) {corr_prediction_cv:.4f}")

                                               
                                                print(f"wCC(cv) predictions (fit on roi means, eval on roi means) {np.mean(self.ols_result_dict[ols_y_dim][f'{[ordered_dimensions[x_dim] for x_dim in x_dims]}']['wCC'])}")


                                                    
                                            
                                            pl.figure(f"{subj} OLS {ols_y_dim.replace('Norm_abcd','')} betas {roi.replace('custom.','').replace('HCPQ1Q6.','').replace('glasser_','')}",figsize=(8,8))
                                            pl.bar(np.arange(len(ls2.coef_)), ls2.coef_, label=f"wCC {corr_prediction_fit_roi_means:.3f}")
                                            pl.xticks(np.arange(len(ls2.coef_)),[ordered_dimensions[x_dim].replace('Norm_abcd','') for x_dim in x_dims],rotation='vertical')
                                            pl.ylabel(f"{ols_y_dim.replace('Norm_abcd','')} OLS betas")
                                            pl.legend()
                                            if save_figures:
                                                pl.savefig(opj(figure_path,f"{ols_y_dim.replace('Norm_abcd','')}_OLSbetas_{roi.replace('custom.','').replace('HCPQ1Q6.','').replace('glasser_','')}.pdf"),dpi=600, bbox_inches='tight')
                                    
                                ##########Multidim regression
                                if perform_pls and np.sum(alpha[analysis][subj][roi]>rsq_thresh)>10:                                     
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
                                            rsq_prediction.append(r2_score(Y[:,p], pred[:,p], sample_weight = multidim_param_array[analysis][subj][roi][ordered_dimensions.index('RSq Norm_abcd')]))
                                        
                                        if rsq_weights:
                                            rsqtot = pls.score(X,Y,sample_weight=multidim_param_array[analysis][subj][roi][ordered_dimensions.index('RSq Norm_abcd')])
                                        else:
                                            rsqtot = pls.score(X,Y)
                                            
                                        print(f"RSq total {rsqtot}")                                               
                                        print(f"RSq predictions {np.array(rsq_prediction)}")
                                        print(f"corr predictions {np.array(corr_prediction)}\n")
                                        
                                        if roi in vis_pca_comps_rois:
                                        
                                            for j,y_dim in enumerate(y_dims):
                                                pl.figure(figsize=(9,9))
                                                pl.bar(np.arange(X.shape[1]), np.array(pls.coef_)[:,j], color=['red' if c>0 else 'blue' for c in np.sign(np.array(pls.coef_)[:,j])], label=f"RSq {rsq_prediction[j]:.3f}")
                                                pl.xticks(np.arange(X.shape[1]),[ordered_dimensions[x_dim].replace('Norm_abcd','') for x_dim in x_dims],rotation='vertical')
                                                pl.ylabel(f"{ordered_dimensions[y_dim]} PLS betas")
                                                pl.legend()
                                                if save_figures:
                                                    pl.savefig(opj(figure_path,f"{ordered_dimensions[y_dim]}_PLSbetas_{n_components}comp_{roi.replace('custom.','').replace('HCPQ1Q6.','').replace('glasser_','')}.pdf"),dpi=600, bbox_inches='tight')
                                            
                                            pl.figure(figsize=(9,9))
                                            pl.bar(np.arange(X.shape[1]),np.abs(np.array(pls.coef_)).sum(1), label=f"RSq total {rsqtot:.3f}")
                                            pl.xticks(np.arange(X.shape[1]),[ordered_dimensions[x_dim].replace('Norm_abcd','') for x_dim in x_dims],rotation='vertical')
                                            pl.ylabel("Total PLS betas")
                                            pl.legend()
                                            if save_figures:
                                                pl.savefig(opj(figure_path,f"TotalPLSbetas_{n_components}comp_{roi.replace('custom.','').replace('HCPQ1Q6.','').replace('glasser_','')}.pdf"),dpi=600, bbox_inches='tight')
                                            
                                            
                                            if vis_pls_pycortex:
                                                ds_pls=dict()
                                                for c in range(n_components):
                                                    zz = np.zeros_like(alpha[analysis][subj][roi]).astype(float)
                                                    zz[alpha[analysis][subj][roi]>rsq_thresh] = pls.x_scores_[:,c]
                                                    ds_pls[f"x_score {c}"] = Vertex2D_fix(zz, alpha[analysis][subj][roi], subject=pycortex_subj, 
                                                                        vmin=np.nanquantile(zz[alpha[analysis][subj][roi]>rsq_thresh],0.1), vmax=np.nanquantile(zz[alpha[analysis][subj][roi]>rsq_thresh],0.9), 
                                                                      vmin2=rsq_thresh, vmax2=np.nanquantile(alpha[analysis][subj][roi][alpha[analysis][subj][roi]>rsq_thresh],0.1), cmap='nipy_spectral')                                                      
                                                    zz = np.zeros_like(alpha[analysis][subj][roi]).astype(float)
                                                    zz[alpha[analysis][subj][roi]>rsq_thresh] = pls.y_scores_[:,c]
                                                    ds_pls[f"y_score {c}"] = Vertex2D_fix(zz, alpha[analysis][subj][roi], subject=pycortex_subj, 
                                                                        vmin=np.nanquantile(zz[alpha[analysis][subj][roi]>rsq_thresh],0.1), vmax=np.nanquantile(zz[alpha[analysis][subj][roi]>rsq_thresh],0.9), 
                                                                      vmin2=rsq_thresh, vmax2=np.nanquantile(alpha[analysis][subj][roi][alpha[analysis][subj][roi]>rsq_thresh],0.1), cmap='nipy_spectral')               
                                                cortex.webgl.show(ds_pls, with_curvature=False, with_labels=True, with_rois=True, with_borders=True, with_colorbar=True)    

                                        
                                        
                                        if cv_regression:
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
                                                    
                                                if rsq_weights:
                                                    cv_rsq_total.append(pls.score(X_cv,Y_cv,sample_weight=multidim_param_array[analysis][cv_subj][roi][ordered_dimensions.index('RSq Norm_abcd')]))
                                                else:
                                                    cv_rsq_total.append(pls.score(X_cv,Y_cv))
                                                cv_corr_pred.append(corr_prediction)
                                                cv_rsq_pred.append(rsq_prediction)
                                                    
                                            print(f"CVRSq total {np.mean(cv_rsq_total,axis=0)}")
                                            print(f"CVRSq predictions {np.array(cv_rsq_pred).mean(0)}")
                                            print(f"CV corr predictions {np.array(cv_corr_pred).mean(0)}")
                                            
                                            self.pls_result_dict[roi][f"{n_components} components"].append(np.mean(cv_rsq_total,axis=0))
                                
                            except Exception as e:
                                import sys
                                print(e)
                                exc_type, exc_obj, exc_tb = sys.exc_info()
                                fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
                                print(exc_type, fname, exc_tb.tb_lineno) 
                                pass
                            
                for key in self.pls_result_dict:
                    for key2 in self.pls_result_dict[key]:
                        print(f"{key} {key2} CVRSq total {np.mean(self.pls_result_dict[key][key2]):.3f}")
                        
                #############size response curves
                if size_response_curves:
                    for subj, subj_res in subjects:
                        if 'analysis_info' in subj_res:
                            an_info = subj_res['analysis_info']
                            ss_deg = 2.0 * np.degrees(np.arctan(an_info['screen_size_cm'] /(2.0*an_info['screen_distance_cm'])))
                            n_pix = an_info['n_pix']
                            
                            ##correcting for a previous issue with non-divisors
                            if an_info["fitting_space"] == 'HCP' and n_pix == 54:
                                n_pix = 67
                        
                        if third_dim_sr_curves != None:
                            fig = pl.figure(f"3D sr curves by {third_dim_sr_curves}", figsize=(8,8))
                            
                            ax = fig.add_subplot(111, projection='3d', azim=-45, elev=50)
                            ax.grid(False)
                            ax.set_xlabel("Stimulus size (°)",labelpad=50)
                            ax.set_ylabel(f"{third_dim_sr_curves}",labelpad=60)
                            ax.set_zlabel("Response",labelpad=50) 
                            ax.xaxis.set_tick_params(pad=20)
                            ax.zaxis.set_tick_params(pad=20)
                            ax.yaxis.set_tick_params(pad=20)
                            
                            all_x = []
                            all_y = []
                            all_z = []
                            third_dim_ticks = []
                            third_dim_vals = []
    
                         
                            curve_par_dict = dict()
                            
                            dim_stim = 2
                            factr = 2 #4 for pnas (determines max stim size) (smaller factr larger stim)
                            bar_stim = False
                            center_prfs = True
                            #note plot_data option has some specifics that only apply to spinoza data
                            #TODO: would need to be edited to be more general
                            plot_data = False
                            plot_curves = True                               
                            resp_measure = 'Max (model-based)'
                            normalize_response = True
                            confint = True
                            log_stim_sizes = False
                            
                            x = np.linspace(-ss_deg/2,ss_deg/2,1000)
                            #correcting for the real size of design matrix VS smoother space used for better size-response curves
                            dx = n_pix/len(x)
                            
                            #1d stims
                            if dim_stim == 1:
                                stims = [np.zeros_like(x) for n in range(int(x.shape[-1]/4))]
                            else:
                                
                                stims = [np.zeros((x.shape[0],x.shape[0])) for n in range(int(x.shape[-1]/(2*factr)))]
                                stim_sizes=[]
                                
                            #print(len(stims))
                            for pp, stim in enumerate(stims):
                                
                                if dim_stim == 1 or bar_stim == True:
                                    #2d rectangle or 1d
                                    stim[int(stim.shape[0]/2)-pp:int(stim.shape[0]/2)+pp] = 1
                                
                                else:
                                    #2d circle
                                    xx,yy = np.meshgrid(x,x)
                                    stim[((xx**2+yy**2)**0.5)<(x.max()*pp/(len(stims)*factr))] = 1
                                    stim_sizes.append(2*(x.max()*pp/(len(stims)*factr)))
                            
                                
                            
                            if dim_stim == 1:
                                #1d stim sizes
                                stim_sizes = (np.max(x)-np.min(x))*np.sum(stims,axis=-1)/x.shape[0]
                            elif bar_stim:
                                #2d stim sizes (rectangle)
                                stim_sizes = (np.max(x)-np.min(x))*np.sum(stims,axis=(-1,-2))/x.shape[0]**2                                    
                            
                            
                            if log_stim_sizes:
                                stim_sizes = np.log10(stim_sizes[1:])
                                stims = stims[1:]
    
                        for i, roi in enumerate(rois):  
                            
                            if ('mean' in analysis) or 'fsaverage' in subj:
                                                          
                                
                                print(f"{analysis} {subj} {roi}")  
    
                                                
                                # for c in range(100):
                                #     sample = np.random.randint(0, len(multidim_param_array[analysis][subj][roi]), int(len(multidim_param_array[analysis][subj][roi])/upsampling_corr_factor))
                                        
                                #     for par in ['Amplitude', 'Norm Param. B', 'Surround Amplitude', 'Norm Param. D',
                                #             'Size (sigma_1)', 'Size (sigma_2)']:
                                                
                                
                                #         curve_par_dict[par] = weightstats.DescrStatsW(multidim_param_array[analysis][subj][roi][ordered_dimensions.index(f'{par} Norm_abcd')][sample],
                                #                         weights=multidim_param_array[analysis][subj][roi][ordered_dimensions.index(f'RSq Norm_abcd')][sample]).mean
                                        
                                #     response_functions.append(norm_1d_sr_function(curve_par_dict['Amplitude'],curve_par_dict['Norm Param. B'],curve_par_dict['Surround Amplitude'],
                                #                                             curve_par_dict['Norm Param. D'],curve_par_dict['Size (sigma_1)'],curve_par_dict['Size (sigma_2)'],x,stims))
    
                                if np.sum(alpha[analysis][subj][roi]>rsq_thresh)>10:
                                                                                                               
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
                                        mean_srf = norm_1d_sr_function(dx*curve_par_dict['Amplitude'].mean,curve_par_dict['Norm Param. B'].mean,dx*curve_par_dict['Surround Amplitude'].mean,
                                                                                      curve_par_dict['Norm Param. D'].mean,curve_par_dict['Size (sigma_1)'].mean,curve_par_dict['Size (sigma_2)'].mean,x,stims,
                                                                                      mu_x=ecc*np.sign(np.cos(curve_par_dict['Polar Angle'].mean)))
            
                                    else:
                                        mean_srf = norm_2d_sr_function(dx**2*curve_par_dict['Amplitude'].mean,curve_par_dict['Norm Param. B'].mean,
                                                                    dx**2*curve_par_dict['Surround Amplitude'].mean,
                                                                                    curve_par_dict['Norm Param. D'].mean,curve_par_dict['Size (sigma_1)'].mean,curve_par_dict['Size (sigma_2)'].mean,x,x,stims,
                                                                                    mu_x=ecc*np.cos(curve_par_dict['Polar Angle'].mean),
                                                                                    mu_y=ecc*np.sin(curve_par_dict['Polar Angle'].mean))
                                        
                                        #mean_srf = ((1 - np.exp(-np.array(stim_sizes)**2/(8*curve_par_dict['Size (sigma_1)'].mean**2)))*curve_par_dict['Amplitude'].mean*dx**2 + curve_par_dict['Norm Param. B'].mean)/((1 - np.exp(-np.array(stim_sizes)**2/(8*curve_par_dict['Size (sigma_2)'].mean**2)))*curve_par_dict['Surround Amplitude'].mean*dx**2 + curve_par_dict['Norm Param. D'].mean) - curve_par_dict['Norm Param. B'].mean/curve_par_dict['Norm Param. D'].mean
                                        
                                    
                                        
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
                                        
                                        
                                    #specific to spinoza data. needs update for more generality
                                    if plot_data:
                                        
                                        #actual_response_1R = weightstats.DescrStatsW(multidim_param_array['fit-task-1R_fit-runs-all'][subj][roi][ordered_dimensions.index(resp_measure)],
                                        #                                          weights=multidim_param_array['fit-task-1R_fit-runs-all'][subj][roi][ordered_dimensions.index('RSq Norm_abcd')])
                                        actual_response_2R = weightstats.DescrStatsW(multidim_param_array['fit-task-2R_fit-runs-all'][subj][roi][ordered_dimensions.index(resp_measure)],
                                                                                  weights=multidim_param_array['fit-task-2R_fit-runs-all'][subj][roi][ordered_dimensions.index('RSq Norm_abcd')])
                                        #actual_response_4R = weightstats.DescrStatsW(multidim_param_array['fit-task-4R_fit-runs-all'][subj][roi][ordered_dimensions.index(resp_measure)],
                                        #                                          weights=multidim_param_array['fit-task-4R_fit-runs-all'][subj][roi][ordered_dimensions.index('RSq Norm_abcd')])
                                        #actual_response_1S = weightstats.DescrStatsW(multidim_param_array['fit-task-1S_fit-runs-all'][subj][roi][ordered_dimensions.index(resp_measure)],
                                        #                                          weights=multidim_param_array['fit-task-1S_fit-runs-all'][subj][roi][ordered_dimensions.index('RSq Norm_abcd')])
                                        #actual_response_4F = weightstats.DescrStatsW(multidim_param_array['fit-task-4F_fit-runs-all'][subj][roi][ordered_dimensions.index(resp_measure)],
                                        #                                          weights=multidim_param_array['fit-task-4F_fit-runs-all'][subj][roi][ordered_dimensions.index('RSq Norm_abcd')])
                                       
                                        actual_response_1 = weightstats.DescrStatsW(np.concatenate((multidim_param_array['fit-task-1R_fit-runs-all'][subj][roi][ordered_dimensions.index(resp_measure)],
                                                                                                  multidim_param_array['fit-task-1S_fit-runs-all'][subj][roi][ordered_dimensions.index(resp_measure)])),
                                                                                  weights=np.concatenate((multidim_param_array['fit-task-1R_fit-runs-all'][subj][roi][ordered_dimensions.index('RSq Norm_abcd')],
                                                                                                          multidim_param_array['fit-task-1S_fit-runs-all'][subj][roi][ordered_dimensions.index('RSq Norm_abcd')])))
                                        actual_response_4 = weightstats.DescrStatsW(np.concatenate((multidim_param_array['fit-task-4R_fit-runs-all'][subj][roi][ordered_dimensions.index(resp_measure)],
                                                                                                  multidim_param_array['fit-task-4F_fit-runs-all'][subj][roi][ordered_dimensions.index(resp_measure)])),
                                                                                weights=np.concatenate((multidim_param_array['fit-task-4R_fit-runs-all'][subj][roi][ordered_dimensions.index('RSq Norm_abcd')],
                                                                                                          multidim_param_array['fit-task-4F_fit-runs-all'][subj][roi][ordered_dimensions.index('RSq Norm_abcd')])))
                                       
                                        
                                        data_sr = [actual_response_1.mean, actual_response_1.mean,actual_response_2R.mean,actual_response_4.mean,actual_response_4.mean]/np.max([actual_response_1.mean, actual_response_1.mean,actual_response_2R.mean,actual_response_4.mean,actual_response_4.mean])
                                        if zconfint_err_alpha != None:
                                            yerr_data_sr = np.array([np.abs(ss.zconfint_mean(alpha=zconfint_err_alpha)-ss.mean) for ss in [actual_response_1,actual_response_1,actual_response_2R,actual_response_4,actual_response_4]]).T*upsampling_corr_factor**0.5
                                        else:
                                            yerr_data_sr = np.array([ss.std_mean for ss in [actual_response_1,actual_response_1,actual_response_2R,actual_response_4,actual_response_4]])*upsampling_corr_factor**0.5
                                        
                                        #yerr_data_sr /= np.max([actual_response_1R.mean,actual_response_1S.mean,actual_response_2R.mean,actual_response_4R.mean,actual_response_4F.mean])
                                    
                                    if third_dim_sr_curves != None:
                                        if third_dim_sr_curves in ordered_dimensions:
                                            third_dim_data = multidim_param_array[analysis][subj][roi][ordered_dimensions.index(third_dim_sr_curves)]
                                        else:
                                            print("Third sr dim not found - check if it is in parameters, and/or append model (e.g. 'RSq Norm_abcd').")
                                            third_dim_data = multidim_param_array[analysis][subj][roi][ordered_dimensions.index(third_dim_sr_curves)]                  
                                                                                
                                        if 'Norm_abcd' in third_dim_sr_curves:
                                            third_dim = weightstats.DescrStatsW(third_dim_data,
                                                            weights=multidim_param_array[analysis][subj][roi][ordered_dimensions.index('RSq Norm_abcd')]).mean                                            
                                        
                                        else:
                                            third_dim = third_dim_data.mean()
                                        
                                        third_dim_ticks.append(f"{roi.replace('custom.','').replace('HCPQ1Q6.','').replace('glasser_','')} {third_dim:.2f}")
                                        third_dim_vals.append(third_dim)
                                        all_x.append(stim_sizes)
                                        all_y.append(third_dim*np.ones_like(stim_sizes))
                                        all_z.append(mean_srf)                                        
                                    
    
                                    if subj == 'Group' or subj == 'fsaverage':
                                        pl.figure(f"Size response {roi.replace('custom.','').replace('HCPQ1Q6.','').replace('glasser_','')}", figsize = (8,8))
                                        
                                        if plot_curves:
                                            pl.plot(stim_sizes, mean_srf, c=cmap_rois[i], label=roi.replace('custom.','').replace('HCPQ1Q6.','').replace('glasser_',''), linewidth=3)
                                            if third_dim_sr_curves != None:
                                                #pl.figure(f"3D sr curves by {third_dim_sr_curves}")
                                                
                                                ax.plot(stim_sizes, np.zeros_like(stim_sizes), mean_srf, c=cmap_rois[i], label=roi.replace('custom.','').replace('HCPQ1Q6.','').replace('glasser_',''), linewidth=2, zorder=100)
                                                
                                                #ax.plot_surface(np.tile(stim_sizes,(3,1)), np.array([third_dim*np.ones_like(stim_sizes)-0.1,third_dim*np.ones_like(stim_sizes),third_dim*np.ones_like(stim_sizes)+0.1]), np.tile(mean_srf,(3,1)), color=cmap_rois[i], alpha=1, zorder=1)
                                                #ax.plot_surface(np.tile(stim_sizes,(3,1)), np.tile(third_dim*np.ones_like(stim_sizes),(3,1)), np.array([mean_srf-0.02,mean_srf,mean_srf+0.02]), color=cmap_rois[i], alpha=1, zorder=1)
                                                
                                                #pl.figure(f"Size response {roi.replace('custom.','').replace('HCPQ1Q6.','').replace('glasser_','')}", figsize = (8,8))
                                        
                                        
                                        if plot_data:
                                            pl.errorbar([0.5, 1.25, 2.5], data_sr[np.r_[0,2,3]], yerr_data_sr[:,np.r_[0,2,3]], marker='s', mec='k', linestyle='-', c=cmap_rois[i], label=roi.replace('custom.','').replace('HCPQ1Q6.',''))# label='Same speed')
                                            #pl.errorbar([0.5, 1.25, 2.5], data_sr[np.r_[1,2,4]], yerr_data_sr[:,np.r_[1,2,4]], marker='v',  mec='k',linestyle='-', c=cmap_rois[i], label='Same STCE')
                                        
                                        #pl.plot(stim_sizes,np.zeros(len(stim_sizes)),linestyle='--',linewidth=0.8, alpha=0.8, color='black',zorder=0)
                                        pl.ylabel("Response")
                                        pl.xlabel("Stimulus size (°)")
                                        pl.legend(fontsize=28,loc=8)   
                                        if save_figures:
                                            pl.savefig(opj(figure_path,f"sr_functions_{roi.replace('custom.','').replace('HCPQ1Q6.','').replace('glasser_','')}.pdf"),dpi=600, bbox_inches='tight')
                                        
                                        
                                        pl.figure('Size response all rois', figsize = (8,8))
    
                                        
                                        if roi == 'all_custom':
                                            if plot_data:
                                                pl.errorbar([0.5, 1.25, 2.5], data_sr[np.r_[0,2,3]], yerr_data_sr[:,np.r_[0,2,3]], marker='s', mec='k', linestyle='-', c='grey', label=roi.replace('custom.','').replace('HCPQ1Q6.','').replace('glasser_',''))# label='Same speed')
                                                #pl.errorbar([0.5, 1.25, 2.5], data_sr[np.r_[1,2,4]], yerr_data_sr[:,np.r_[1,2,4]], marker='v',  mec='k',linestyle='', c='grey', label='Same STCE (all rois)')
                                                                                    
                                        else:
                                            if plot_data:
                                                pl.errorbar([0.5, 1.25, 2.5], data_sr[np.r_[0,2,3]], 0, marker='s', markersize=6, zorder=len(rois)-i, mec='k', linestyle='-', c=cmap_rois[i], label=roi.replace('custom.','').replace('HCPQ1Q6.','').replace('glasser_',''))# label='Same speed')
                                            if plot_curves:
                                                pl.plot(stim_sizes, mean_srf, c=cmap_rois[i], label=roi.replace('custom.','').replace('HCPQ1Q6.','').replace('glasser_',''), linewidth=2, zorder=len(rois))
                                        
                                        if confint:
                                            
                                            if normalize_response:
                                                response_functions[roi] /= mean_srf.max()
                                            pl.fill_between(stim_sizes, mean_srf+sem(response_functions[roi], axis=0),
                                                            mean_srf-sem(response_functions[roi], axis=0), color=cmap_rois[i], label=roi.replace('custom.','').replace('HCPQ1Q6.','').replace('glasser_',''), alpha=0.2, zorder=len(rois))
                                                                                    
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
                                        pl.figure(f"Size response {roi.replace('custom.','').replace('HCPQ1Q6.','').replace('glasser_','')}", figsize = (8,8))
    
                                        
                                        if plot_curves:
                                            pl.plot(stim_sizes, mean_srf, c=cmap_rois[i], alpha=0.5, linewidth=1)
                                        if plot_data:
                                            pl.errorbar([0.5, 1.25, 2.5], data_sr[np.r_[0,2,3]], 0, alpha=0.25,  linestyle='-', c=cmap_rois[i])#markersize=4, marker='s',
                                            #pl.errorbar([0.5, 1.25, 2.5], data_sr[np.r_[1,2,4]], yerr_data_sr[:,np.r_[1,2,4]], alpha=0.4, markersize=4, marker='v',linestyle='', c=cmap_rois[i])
                                                 
    
                        if third_dim_sr_curves != None:                
                            all_x = np.array(all_x)
                            all_y = np.array(all_y)
                            all_z = np.array(all_z)
                            third_dim_vals = np.array(third_dim_vals)
                            third_dim_ticks = np.array(third_dim_ticks)
                            
                            
                            pl.figure(f"3D sr curves by {third_dim_sr_curves}")
                            
                            ax.contourf(X=all_x[np.argsort(third_dim_vals)], Y=all_y[np.argsort(third_dim_vals)], Z=all_z[np.argsort(third_dim_vals)], offset=-0.25, zorder=0, cmap='jet', levels=50, vmin=0.25)
        
                            ax.plot_surface(X=all_x[np.argsort(third_dim_vals)], Y=all_y[np.argsort(third_dim_vals)], Z=all_z[np.argsort(third_dim_vals)], cmap='jet', vmin=0.25)
                            
                               
                            pl.yticks(np.sort(third_dim_vals),third_dim_ticks[np.argsort(third_dim_vals)])
                    




        return
      
   

                                
