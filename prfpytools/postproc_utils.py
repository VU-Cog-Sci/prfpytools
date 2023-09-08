import os
import numpy as np
from scipy import sparse
import matplotlib.pyplot as pl
from matplotlib import cm, colors
from tqdm import tqdm
import cortex
import yaml
from pathlib import Path
from collections import defaultdict as dd
import matplotlib.image as mpimg
from matplotlib.colors import Normalize
from copy import deepcopy
from prfpy.model import Iso2DGaussianModel, Norm_Iso2DGaussianModel, DoG_Iso2DGaussianModel, CSS_Iso2DGaussianModel
from prfpytools.preproc_utils import create_full_stim
from prfpy.rf import gauss2D_iso_cart
from prfpy.stimulus import PRFStimulus2D

opj = os.path.join

class results(object):
    def __init__(self):
        self.main_dict = dd(lambda:dd(lambda:dd(lambda:dd(dict))))
    
    def combine_results(self, results_folder,
                        timecourse_folder=None,
                        ref_volume_path=None,
                        cvfold_comb='median',
                        calculate_CCrsq=False,
                        calculate_noise_ceiling=False,
                        screenshot_paths = []):
        
        an_list = [path for path in os.listdir(results_folder) if 'analysis' in path]
        subjects = [path.split('_')[0] for path in an_list]
        an_infos = []
        for path in an_list:
            with open(opj(results_folder,path)) as f:
                an_infos.append(yaml.safe_load(f))
                
        an_names = []
        an_tasks = []
        an_runs = []
        for i, an_info in enumerate(an_infos):

            if 'subj' not in an_info:
                an_info['subj'] = subjects[i]

            if 'session' in an_info:
                ses_str = an_info['session']+'_'
            else:
                ses_str = ''
            
            if len(an_info['task_names'])>1:
                an_tasks.append('all')
            else:
                an_tasks.append(''.join(an_info['task_names']))
                
            if an_info['crossvalidate'] == True:
                an_runs.append(''.join([str(el) for el in an_info['fit_runs']]))#(len(an_info['fit_runs']))#
            else:
                an_runs.append('all')
            
            an_names.append(f"{subjects[i]}_{ses_str}fit-task-{an_tasks[i]}_fit-runs-{an_runs[i]}")
            
        unique_an_names = np.unique(np.array(an_names)).tolist()
        
        unique_an_results=dict()
        self.prf_stims=dict()
        
        #this is to combine multiple iterations (max) and different models fit on the same fold
        for an_name in tqdm(unique_an_names):
            current_an_infos = np.array(an_infos)[np.array(an_names)==an_name]

            if 'session' in current_an_infos[0]:
                ses_str = current_an_infos[0]['session']+'_'
            else:
                ses_str = ''

            print(current_an_infos)
            current_an_infos = np.array([cai for cai in current_an_infos if (os.path.exists(opj(results_folder,f"{cai['subj']}_{ses_str}mask_space-{cai['fitting_space']}{cai['analysis_time']}.npy"))\
                                or os.path.exists(opj(results_folder,f"{cai['subj']}_{ses_str}mask_space-{cai['fitting_space']}{cai['previous_analysis_time']}.npy")))])
            print(current_an_infos)

            merged_an_info = mergedict_AND(current_an_infos)


  
            r_r = dict()
            r_r_full=dict()
            
            try:            
                mask = np.load(opj(results_folder,f"{current_an_infos[0]['subj']}_{ses_str}mask_space-{current_an_infos[0]['fitting_space']}{current_an_infos[0]['analysis_time']}.npy"))
            except:
                mask = np.load(opj(results_folder,f"{current_an_infos[0]['subj']}_{ses_str}mask_space-{current_an_infos[0]['fitting_space']}{current_an_infos[0]['previous_analysis_time']}.npy"))
                
            for i, curr_an_info in enumerate(current_an_infos):


                if os.path.exists(opj(timecourse_folder,f"{curr_an_info['subj']}_{ses_str}timecourse_space-{curr_an_info['fitting_space']}{curr_an_info['analysis_time']}.npy")):
                    merged_an_info["timecourse_analysis_time"] = curr_an_info["analysis_time"]
                
                for model in curr_an_info["models_to_fit"]:
                    if model == 'norm':
                        for variant in curr_an_info['norm_model_variant']:
                            if os.path.exists(opj(results_folder,f"{curr_an_info['subj']}_{ses_str}iterparams-{model}{variant}_space-{curr_an_info['fitting_space']}{curr_an_info['analysis_time']}.npy")) or \
                            os.path.exists(opj(results_folder,f"{curr_an_info['subj']}_{ses_str}iterparams-{model}{variant}_space-{curr_an_info['fitting_space']}{curr_an_info['previous_analysis_time']}.npy")):
                            
                                try:
                                    raw_model_result = (np.load(opj(results_folder,f"{curr_an_info['subj']}_{ses_str}iterparams-{model}{variant}_space-{curr_an_info['fitting_space']}{curr_an_info['analysis_time']}.npy")))
                                except:
                                    raw_model_result = (np.load(opj(results_folder,f"{curr_an_info['subj']}_{ses_str}iterparams-{model}{variant}_space-{curr_an_info['fitting_space']}{curr_an_info['previous_analysis_time']}.npy")))
                                
                                if model == 'norm':
                                    if f"Norm_{variant}" not in r_r:
                                        r_r[f"Norm_{variant}"] = np.copy(raw_model_result)
                                    else:
                                        r_r[f"Norm_{variant}"][r_r[f"Norm_{variant}"][:,-1]<raw_model_result[:,-1]] = np.copy(raw_model_result[r_r[f"Norm_{variant}"][:,-1]<raw_model_result[:,-1]])

                    
                    else:
                        if os.path.exists(opj(results_folder,f"{curr_an_info['subj']}_{ses_str}iterparams-{model.lower()}_space-{curr_an_info['fitting_space']}{curr_an_info['analysis_time']}.npy")) or \
                        os.path.exists(opj(results_folder,f"{curr_an_info['subj']}_{ses_str}iterparams-{model.lower()}_space-{curr_an_info['fitting_space']}{curr_an_info['previous_analysis_time']}.npy")):
                        
                            try:
                                raw_model_result = (np.load(opj(results_folder,f"{curr_an_info['subj']}_{ses_str}iterparams-{model.lower()}_space-{curr_an_info['fitting_space']}{curr_an_info['analysis_time']}.npy")))
                            except:
                                raw_model_result = (np.load(opj(results_folder,f"{curr_an_info['subj']}_{ses_str}iterparams-{model.lower()}_space-{curr_an_info['fitting_space']}{curr_an_info['previous_analysis_time']}.npy")))
                            
                            if model == 'gauss':
                                if 'Gauss' not in r_r:
                                    r_r['Gauss'] = np.copy(raw_model_result)
                                else:
                                    r_r['Gauss'][r_r['Gauss'][:,-1]<raw_model_result[:,-1]] = np.copy(raw_model_result[r_r['Gauss'][:,-1]<raw_model_result[:,-1]])
                            
                            else:
                                if model not in r_r:
                                    r_r[model] = np.copy(raw_model_result)
                                else:
                                    r_r[model][r_r[model][:,-1]<raw_model_result[:,-1]] = np.copy(raw_model_result[r_r[model][:,-1]<raw_model_result[:,-1]])                            
                            
                    


            #move to full surface space so different masks can be handled when merging folds later
            for key in r_r:
                r_r_full[key] = np.zeros((mask.shape[0],r_r[key].shape[-1]))
                r_r_full[key][mask] = np.copy(r_r[key])   

            
                
            #housekeeping
            tc_paths = [str(path) for path in sorted(Path(timecourse_folder).glob(f"{merged_an_info['subj']}_{ses_str}timecourse_space-{merged_an_info['fitting_space']}_task-*_run-*.npy"))]    
            mask_paths = [tc_path.replace('timecourse_','mask_') for tc_path in tc_paths]
            all_task_names = np.unique(np.array([elem.replace('task-','') for path in tc_paths for elem in path.split('_')  if 'task' in elem]))
            all_runs = np.unique(np.array([int(elem.replace('run-','').replace('.npy','')) for path in tc_paths for elem in path.split('_')  if 'run' in elem]))
            
            if calculate_noise_ceiling and 'all' not in merged_an_info['session']:
                
                tc_test = dict()
                tc_fit = dict()
           
                
                for task in merged_an_info['task_names']:
                    tc_runs=[]

                    if task not in self.prf_stims:

                        self.prf_stims[task] = create_full_stim(screenshot_paths=screenshot_paths,
                                    n_pix=merged_an_info['n_pix'],
                                    discard_volumes=merged_an_info['discard_volumes'],
                                    dm_edges_clipping=merged_an_info['dm_edges_clipping'],
                                    screen_size_cm=merged_an_info['screen_size_cm'],
                                    screen_distance_cm=merged_an_info['screen_distance_cm'],
                                    TR=merged_an_info['TR'],
                                    task_names=[task],
                                    normalize_integral_dx=merged_an_info['normalize_integral_dx'])
                    
                    for run in all_runs:
                        mask_run = [np.load(mask_path) for mask_path in mask_paths if f"task-{task}" in mask_path and f"run-{run}" in mask_path][0]

                        tc_run = ([np.load(tc_path) for tc_path in tc_paths if f"task-{task}" in tc_path and f"run-{run}" in tc_path][0])
                        
                        #tc_run *= (100/tc_run.mean(-1))[...,np.newaxis]
                        #tc_run += (tc_run.mean(-1)-np.median(tc_run[...,prf_stim.late_iso_dict[task]], axis=-1))[...,np.newaxis]
                        
                        tc_runs_unmasked = np.zeros((mask_run.shape[0], tc_run.shape[-1]))
                        
                        if merged_an_info['fitting_space'] == 'HCP':
                            tc_runs_unmasked[mask_run] = np.copy(tc_run[:118584])
                        else:
                            tc_runs_unmasked[mask_run] = np.copy(tc_run)
                            
                        tc_runs.append(tc_runs_unmasked)
                                           
                        tc_runs[-1] -= np.median(tc_runs[-1][...,self.prf_stims[task].late_iso_dict[task]], axis=-1)[...,np.newaxis]
                   
                    if merged_an_info["crossvalidate"]:
                        tc_test[task] = np.mean([tc_runs[i] for i in all_runs if i not in merged_an_info['fit_runs']], axis=0)
                        tc_fit[task] = np.mean([tc_runs[i] for i in all_runs if i in merged_an_info['fit_runs']], axis=0)
                    else:                
                        tc_test[task] = np.mean([tc_runs[i] for i in all_runs if i not in np.arange(0,len(all_runs),2)], axis=0)
                        tc_fit[task] = np.mean([tc_runs[i] for i in all_runs if i in np.arange(0,len(all_runs),2)], axis=0)
                
                tc_all_test = np.concatenate(tuple([tc_test[task] for task in tc_test]), axis=-1)
                tc_all_fit = np.concatenate(tuple([tc_fit[task] for task in tc_fit]), axis=-1)
                
                #noise_ceiling = 1-np.sum((tc_all_test-tc_all_fit)**2, axis=-1)/(tc_all_test.shape[-1]*tc_all_test.var(-1))

                noise_ceiling = np.array([np.corrcoef(tc_t,tc_f)[0,1] for tc_t,tc_f in zip(tc_all_test,tc_all_fit)])

                #mean of the variance of single run timecourses
                r_r_full['Single run Variance'] = np.mean([np.var(tc,axis=-1) for tc in tc_runs],axis=0)
                #variance of the mean timecourse
                r_r_full['Variance of mean timecourse'] = np.var(np.mean(tc_runs,axis=0),axis=-1)
                
                r_r_full[f"Noise Ceiling (CC)"] = np.copy(noise_ceiling)
                
            
            #calculate cross-condition r-squared
            
            if not merged_an_info["crossvalidate"] and calculate_CCrsq:
                for task in [tsk for tsk in all_task_names if tsk not in merged_an_info['task_names']]:
                    if task not in self.prf_stims:
                        self.prf_stims[task] = create_full_stim(screenshot_paths=[s_p for s_p in screenshot_paths if task in s_p],
                                n_pix=merged_an_info['n_pix'],
                                discard_volumes=merged_an_info['discard_volumes'],
                                dm_edges_clipping=merged_an_info['dm_edges_clipping'],
                                screen_size_cm=merged_an_info['screen_size_cm'],
                                screen_distance_cm=merged_an_info['screen_distance_cm'],
                                TR=merged_an_info['TR'],
                                task_names=[task],
                                normalize_integral_dx=merged_an_info['normalize_integral_dx'])
                        

                    all_tcs_task = [np.load(tc_path) for tc_path in tc_paths if f"task-{task}" in tc_path]
                    all_masks_task = [np.load(mask_path) for mask_path in mask_paths if f"task-{task}" in mask_path]
                    
                    mask_task = np.product(all_masks_task, axis=0).astype('bool')
                    tc_full = np.zeros((mask_task.shape[0], all_tcs_task[0].shape[-1]))
                    
                    for i in range(len(all_tcs_task)):
                        tc_full[all_masks_task[i]] += all_tcs_task[i]

                    tc_full /= len(all_tcs_task)

                    common_mask = mask_task * mask
                    tc_comp = np.copy(tc_full[common_mask])              
                    
                    #tc_comp *= (100/tc_comp.mean(-1))[...,np.newaxis]
                    #tc_comp += (tc_comp.mean(-1)-np.median(tc_comp[...,prf_stims[task].late_iso_dict[task]], axis=-1))[...,np.newaxis]
                    

                    
                    for key in r_r:
                        gg = model_wrapper(key,
                                           stimulus=self.prf_stims[task],
                                           hrf=merged_an_info['hrf'],
                                           filter_predictions=merged_an_info['filter_predictions'],
                                           filter_type=merged_an_info['filter_type'],
                                           filter_params={x:merged_an_info[x] for x in ["first_modes_to_remove",
                                                                                        "last_modes_to_remove_percent",
                                                                                        "window_length",
                                                                                        "polyorder",
                                                                                        "highpass",
                                                                                        "add_mean"]},
                                           normalize_RFs=merged_an_info['normalize_RFs'])
                        
                        preds = gg.return_prediction(*list(r_r[key][:,:-1].T))
                        preds_full = np.zeros((mask.shape[0],tc_comp.shape[-1]))
                        preds_full[mask] = np.copy(preds)
                        
                        preds_comp = np.copy(preds_full[common_mask])
                        
                        cc_rsq = (1-np.sum((preds_comp-tc_comp)**2, axis=-1)/(tc_comp.var(-1)*tc_comp.shape[-1]))
                        
                        cc_rsq_full = np.zeros(mask.shape[0])
                        cc_rsq_full[common_mask] = np.copy(cc_rsq)
                        
                        r_r_full[f'CCrsq_task-{task}_model-{key}'] = np.copy(cc_rsq_full)
                        

                
             
            r_r_full["mask"] = np.copy(mask)                            
            r_r_full["analysis_info"] = deepcopy(merged_an_info)                          
            unique_an_results[an_name] = deepcopy(r_r_full)
            
            
            

            
        #this is to combine over folds (median)    
        folds = [key for key in unique_an_results if 'fit-runs-all' not in key]
        no_folds = [key for key in unique_an_results if 'fit-runs-all' in key]
        
        combined_results = dd(dict)
        for key in set(['_'.join(key.split('_')[:-1]) for key in folds]):
            current_fold_infos = [unique_an_results[fold]['analysis_info'] for fold in folds if key in fold]
            for res in unique_an_results[folds[0]]:
                if cvfold_comb == 'median':
                    if 'info' in res:
                        combined_results[key+'_fit-runs-CVmedian'][res] = mergedict_AND(current_fold_infos)
                    elif 'mask' in res:
                        combined_results[key+'_fit-runs-CVmedian'][res] = np.product([unique_an_results[fold][res] for fold in folds if key in fold], axis=0).astype('bool')
                    else:
                        combined_results[key+'_fit-runs-CVmedian'][res] = np.median([unique_an_results[fold][res] for fold in folds if key in fold], axis=0)
                else:
                    if 'info' in res:
                        combined_results[key+'_fit-runs-CVmean'][res] = mergedict_AND(current_fold_infos)
                    elif 'mask' in res:
                        combined_results[key+'_fit-runs-CVmean'][res] = np.product([unique_an_results[fold][res] for fold in folds if key in fold], axis=0).astype('bool')
                    else:
                        combined_results[key+'_fit-runs-CVmean'][res] = np.mean([unique_an_results[fold][res] for fold in folds if key in fold], axis=0)                                       
                                   
        for key in no_folds:
            combined_results[key] = deepcopy(unique_an_results[key])

        for key in folds:
            combined_results[key] = deepcopy(unique_an_results[key])            
 
        
        for an, an_res in combined_results.items():
            an_res['analysis_info']['timecourse_folder'] = timecourse_folder
            
            subj = an_res['analysis_info']['subj']

            if 'session' in an_res['analysis_info']:
                ses = an_res['analysis_info']['session']
                ses_str = '_'+an_res['analysis_info']['session']
            else:
                ses = ''
                ses_str = ''

            space = an_res['analysis_info']['fitting_space']
            reduced_an_name = an.replace(f"{subj}_",'')

            if ses_str != '':
                reduced_an_name = reduced_an_name.replace(ses+'_','')


            mask = an_res['mask']
            
            for res in an_res:
                if 'mask' not in res and 'info' not in res:
                    self.main_dict[space][reduced_an_name][subj+ses_str]['Results'][res] = deepcopy(an_res[res][mask])   
                else:
                    self.main_dict[space][reduced_an_name][subj+ses_str][res] = deepcopy(an_res[res])
                
            
                       
            raw_tc_stats = dd(lambda:np.zeros(mask.shape))

            
            if '999999' in subj:
                tc_raw = np.load(opj(timecourse_folder,f'999999_timecourse-raw_space-{space}.npy'))
                mask = np.load(opj(timecourse_folder,f'999999_mask-raw_space-{space}.npy'))
            else:
                try:
                    tc_raw = np.load(opj(timecourse_folder,f'{subj}{ses_str}_timecourse-raw_space-{space}.npy'))
                    mask = np.load(opj(timecourse_folder,f'{subj}{ses_str}_mask-raw_space-{space}.npy'))
                except:
                    tc_raw = np.load(opj(timecourse_folder,f'{subj}_ses-all_timecourse-raw_space-{space}.npy'))
                    mask = np.load(opj(timecourse_folder,f'{subj}_ses-all_mask-raw_space-{space}.npy'))              
                
            tc_mean = tc_raw.mean(-1)
            tc_mean_full = np.zeros(mask.shape)
            tc_mean_full[mask] = tc_mean
            raw_tc_stats['Mean'] = tc_mean_full
        
            tc_var = tc_raw.var(-1)
            tc_var_full = np.zeros(mask.shape)
            tc_var_full[mask] = tc_var
            raw_tc_stats['Variance'] = tc_var_full
        
            tc_tsnr_full = np.zeros(mask.shape)
            tc_tsnr_full[mask] = tc_mean/np.sqrt(tc_var)
            raw_tc_stats['TSNR'] = tc_tsnr_full
            

            self.main_dict[space][reduced_an_name][subj+ses_str]['Timecourse Stats'] = deepcopy(raw_tc_stats)
                
        return
        


    
    
    def process_results(self, results_dict, compute_suppression_index = False, return_norm_profiles=False, compute_fwhmax_fwatmin = False):
        for k, v in tqdm(results_dict.items()):
            if 'sub-' not in k and not k.isdecimal() and '999999' not in k and 'fsaverage' not in k:
                print(k)
                self.process_results(v, compute_suppression_index, return_norm_profiles)
            elif 'Results' in v and 'Processed Results' not in v:
                mask = v['mask']
                normalize_RFs = v['analysis_info']['normalize_RFs']
                
                if compute_suppression_index or return_norm_profiles:
                    #for suppression index computation
                    if not hasattr(self, 'prf_stim'):
                                            
                        self.prf_stim = PRFStimulus2D(screen_size_cm=v['analysis_info']['screen_size_cm'],
                                     screen_distance_cm=v['analysis_info']['screen_distance_cm'],
                                     design_matrix=np.zeros((v['analysis_info']['n_pix'],v['analysis_info']['n_pix'],10)),
                                     TR=1.0)
                                       
                        aperture = ((self.prf_stim.x_coordinates**2+self.prf_stim.y_coordinates**2)**0.5 < (self.prf_stim.screen_size_degrees/2))
                    else:
                        if [v['analysis_info']['screen_size_cm'],v['analysis_info']['screen_distance_cm'],v['analysis_info']['n_pix']] \
                            != [self.prf_stim.screen_size_cm,self.prf_stim.screen_distance_cm,self.prf_stim.design_matrix.shape[0]]:
                                
                            self.prf_stim = PRFStimulus2D(screen_size_cm=v['analysis_info']['screen_size_cm'],
                                         screen_distance_cm=v['analysis_info']['screen_distance_cm'],
                                         design_matrix=np.zeros((v['analysis_info']['n_pix'],v['analysis_info']['n_pix'],10)),
                                         TR=1.0)
                                           
                            aperture = ((self.prf_stim.x_coordinates**2+self.prf_stim.y_coordinates**2)**0.5 < (self.prf_stim.screen_size_degrees/2))
                
                #store processed results in nested default dictionary
                processed_results = dd(lambda:dd(lambda:np.zeros(mask.shape)))
    
                #loop over contents of single-subject analysis results (models and mask)
                for k2, v2 in v['Results'].items():
                    if isinstance(v2, np.ndarray) and v2.ndim == 2:
    
                        processed_results['RSq'][k2][mask] = np.copy(v2[:,-1])
                        processed_results['RSq'][k2][mask][np.all(np.isfinite(v2),axis=-1)] = 0
                        
                        processed_results['Eccentricity'][k2][mask] = np.sqrt(v2[:,0]**2+v2[:,1]**2)
                        processed_results['Polar Angle'][k2][mask] = np.arctan2(v2[:,1], v2[:,0])
                        processed_results['Amplitude'][k2][mask] = np.copy(v2[:,3])
                        processed_results['Size (sigma_1)'][k2][mask] = np.copy(v2[:,2])
                        
                        processed_results['x_pos'][k2][mask] = np.copy(v2[:,0])
                        processed_results['y_pos'][k2][mask] = np.copy(v2[:,1])
                        
                        if 'fit_hrf' in v['analysis_info']: #legacy (param only present if fit)
                            if v['analysis_info']['fit_hrf']:
                                processed_results['hrf_1'][k2][mask] = np.copy(v2[:,-3])
                                processed_results['hrf_2'][k2][mask] = np.copy(v2[:,-2])
                        else: #current (parm always present)
                            processed_results['hrf_1'][k2][mask] = np.copy(v2[:,-3])
                            processed_results['hrf_2'][k2][mask] = np.copy(v2[:,-2])                         
    
                        if k2 == 'CSS':
                            processed_results['CSS Exponent'][k2][mask] =  np.copy(v2[:,5])
    
                        if k2 == 'DoG':
                            processed_results['Surround Amplitude'][k2][mask] = np.copy(v2[:,5])
                            processed_results['Size (sigma_2)'][k2][mask] = np.copy(v2[:,6])
                            processed_results['Size ratio (sigma_2/sigma_1)'][k2][mask] = v2[:,6]/v2[:,2]

                            if compute_fwhmax_fwatmin:
                                (processed_results['Size (fwhmax)'][k2][mask],
                                processed_results['Surround Size (fwatmin)'][k2][mask]) = fwhmax_fwatmin(k2, v2, normalize_RFs)
                            
                            if compute_suppression_index:
                                processed_results['Suppression Index (full)'][k2][mask] = (v2[:,5] * v2[:,6]**2)/(v2[:,3] * v2[:,2]**2)
                                processed_results['Suppression Index'][k2][mask] = suppression_index(self.prf_stim, aperture, v2, normalize_RFs)
    
                        elif 'Norm' in k2:
                            processed_results['Surround Amplitude'][k2][mask] = np.copy(v2[:,5])
                            processed_results['Size (sigma_2)'][k2][mask] = np.copy(v2[:,6])    
                            processed_results['Size ratio (sigma_2/sigma_1)'][k2][mask] = v2[:,6]/v2[:,2]
                            processed_results['Norm Param. B'][k2][mask] = np.copy(v2[:,7])
                            processed_results['Norm Param. D'][k2][mask] = np.copy(v2[:,8])
                            processed_results['Ratio (B/D)'][k2][mask] = v2[:,7]/v2[:,8]
                            
                            
                            if compute_suppression_index:
                                processed_results['Suppression Index (full)'][k2][mask] = (v2[:,5] * v2[:,6]**2)/(v2[:,3] * v2[:,2]**2)
                                processed_results['Suppression Index'][k2][mask] = suppression_index(self.prf_stim, aperture, v2, normalize_RFs)
                            
    
                            if return_norm_profiles and len(mask.shape)<2:
                                processed_results['pRF Profiles'][k2] = np.zeros((mask.shape[0],1000))
                                if compute_fwhmax_fwatmin:
                                    ((processed_results['Size (fwhmax)'][k2][mask],
                                    processed_results['Surround Size (fwatmin)'][k2][mask]),
                                processed_results['pRF Profiles'][k2][mask]) = fwhmax_fwatmin(k2, v2, normalize_RFs, True)
                            else:
                                if compute_fwhmax_fwatmin:
                                    (processed_results['Size (fwhmax)'][k2][mask],
                                    processed_results['Surround Size (fwatmin)'][k2][mask]) = fwhmax_fwatmin(k2, v2, normalize_RFs, False)
    
                        else:
                            if compute_fwhmax_fwatmin:
                                processed_results['Size (fwhmax)'][k2][mask] = fwhmax_fwatmin(k2, v2, normalize_RFs)

                        ####copy beta and cross-cond rsq to processed results
                            #####################
                    elif isinstance(v2, np.ndarray) and v2.ndim == 1 and 'model' in k2:
                        processed_results[k2.split('_model-')[0]][k2.split('_model-')[1]][mask] = np.copy(v2)
                        
                    elif isinstance(v2, np.ndarray) and v2.ndim == 1 and 'Noise Ceiling' in k2:
                        processed_results['Noise Ceiling'][k2][mask] = np.copy(v2)

                    elif isinstance(v2, np.ndarray) and v2.ndim == 1 and 'Variance' in k2:
                        processed_results['Variance Stats'][k2][mask] = np.copy(v2)                        
    
                v['Processed Results'] = deepcopy(processed_results)
        return


def norm_1d_sr_function(a,b,c,d,s_1,s_2,x,stims,mu_x=0):
    sr_function = (a*np.sum(np.exp(-(x-mu_x)**2/(2*s_1**2))*stims, axis=-1) + b) / (c*np.sum(np.exp(-(x-mu_x)**2/(2*s_2**2))*stims, axis=-1) + d) - b/d
    return sr_function

def norm_2d_sr_function(a,b,c,d,s_1,s_2,x,y,stims,mu_x=0,mu_y=0):
    xx,yy = np.meshgrid(x,y)

    sr_function = (a*np.sum(np.exp(-((xx-mu_x)**2+(yy-mu_y)**2)/(2*s_1**2))*stims, axis=(-1,-2)) + b) / (c*np.sum(np.exp(-((xx-mu_x)**2+(yy-mu_y)**2)/(2*s_2**2))*stims, axis=(-1,-2)) + d) - b/d
    return sr_function             
            
def mergedict_OR(dict1, dict2):
    """
    Logical OR type merger (update or add elements of dictionaries)
    """
    for k, v in dict2.items():
        if k in dict1 and (isinstance(dict1[k], dict) or isinstance(dict1[k], dd)):
            mergedict_OR(dict1[k], dict2[k])
        else:
            dict1[k] = dict2[k]


def mergedict_AND(li):
    """
    Logical AND type merger (only keep common elements of dictionaries)
    """
    result = deepcopy(li[0])
    for element in li:
        for key in element:
            if key in result and result[key] != element[key]:
                del result[key]
            
    return result

def model_wrapper(model,**kwargs):
    if model == 'Gauss':
        return Iso2DGaussianModel(**kwargs)
    elif model == 'DoG':
        return DoG_Iso2DGaussianModel(**kwargs)
    elif model == 'CSS':
        return CSS_Iso2DGaussianModel(**kwargs)
    elif 'Norm' in model:
        return Norm_Iso2DGaussianModel(**kwargs)

def create_model_rf_wrapper(model,stim,params,normalize_RFs=False):
    prf = params[3][...,np.newaxis,np.newaxis]*np.rot90(gauss2D_iso_cart(x=stim.x_coordinates[...,np.newaxis],
                               y=stim.y_coordinates[...,np.newaxis],
                               mu=(params[0], params[1]),
                               sigma=params[2],
                              normalize_RFs=normalize_RFs).T, axes=(1,2))
    if model == 'CSS':
        prf **= params[5][...,np.newaxis,np.newaxis]
    elif model == 'DoG':
        prf -= params[5][...,np.newaxis,np.newaxis]*np.rot90(gauss2D_iso_cart(x=stim.x_coordinates[...,np.newaxis],
                               y=stim.y_coordinates[...,np.newaxis],
                               mu=(params[0], params[1]),
                               sigma=params[6],
                              normalize_RFs=normalize_RFs).T, axes=(1,2))
    elif 'Norm' in model:
        prf += params[7][...,np.newaxis,np.newaxis]
        prf /= (params[5][...,np.newaxis,np.newaxis]*np.rot90(gauss2D_iso_cart(x=stim.x_coordinates[...,np.newaxis],
                               y=stim.y_coordinates[...,np.newaxis],
                               mu=(params[0], params[1]),
                               sigma=params[6],
                              normalize_RFs=normalize_RFs).T, axes=(1,2)) + params[8][...,np.newaxis,np.newaxis])
        prf -= (params[7]/params[8])[...,np.newaxis,np.newaxis]

    return prf

def suppression_index(stim, aperture, params, normalize_RFs=False):
    prf = params[:,3][...,np.newaxis,np.newaxis]*np.rot90(gauss2D_iso_cart(x=stim.x_coordinates[...,np.newaxis],
                               y=stim.y_coordinates[...,np.newaxis],
                               mu=(params[:,0], params[:,1]),
                               sigma=params[:,2],
                              normalize_RFs=normalize_RFs).T, axes=(1,2))
    
    srf = params[:,5][...,np.newaxis,np.newaxis]*np.rot90(gauss2D_iso_cart(x=stim.x_coordinates[...,np.newaxis],
                               y=stim.y_coordinates[...,np.newaxis],
                               mu=(params[:,0], params[:,1]),
                               sigma=params[:,6],
                              normalize_RFs=normalize_RFs).T, axes=(1,2))
  
    prf[:,~aperture] = 0
    srf[:,~aperture] = 0
 
    return srf.sum(axis=(1,2)) / prf.sum(axis=(1,2))   


def fwhmax_fwatmin(model, params, normalize_RFs=False, return_profiles=False):
    model = model.lower()
    x=np.linspace(-50,50,1000).astype('float32')

    prf = params[...,3] * np.exp(-0.5*x[...,np.newaxis]**2 / params[...,2]**2)
    vol_prf =  2*np.pi*params[...,2]**2

    if 'dog' in model or 'norm' in model:
        srf = params[...,5] * np.exp(-0.5*x[...,np.newaxis]**2 / params[...,6]**2)
        vol_srf = 2*np.pi*params[...,6]**2

    if normalize_RFs==True:

        if model == 'gauss':
            profile =  prf / vol_prf
        elif model == 'css':
            #amplitude is outside exponent in CSS
            profile = (prf / vol_prf)**params[...,5] * params[...,3]**(1 - params[...,5])
        elif model =='dog':
            profile = prf / vol_prf - \
                       srf / vol_srf
        elif 'norm' in model:
            profile = (prf / vol_prf + params[...,7]) /\
                      (srf / vol_srf + params[...,8]) - params[...,7]/params[...,8]
    else:
        if model == 'gauss':
            profile = prf
        elif model == 'css':
            #amplitude is outside exponent in CSS
            profile = prf**params[...,5] * params[...,3]**(1 - params[...,5])
        elif model =='dog':
            profile = prf - srf
        elif 'norm' in model:
            profile = (prf + params[...,7])/(srf + params[...,8]) - params[...,7]/params[...,8]


    half_max = np.max(profile, axis=0)/2
    fwhmax = np.abs(2*x[np.argmin(np.abs(profile-half_max), axis=0)])


    if 'dog' in model or 'norm' in model:

        min_profile = np.min(profile, axis=0)
        fwatmin = np.abs(2*x[np.argmin(np.abs(profile-min_profile), axis=0)])

        result = fwhmax, fwatmin
    else:
        result = fwhmax

    if return_profiles:
        return result, profile.T
    else:
        return result
            
                                
def colorbar(mappable):
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    import matplotlib.pyplot as plt
    last_axes = plt.gca()
    ax = mappable.axes
    fig = ax.figure
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    cbar = fig.colorbar(mappable, cax=cax)
    plt.sca(last_axes)
    return cbar        



def Vertex2D_fix(data1, data2, subject, cmap, vmin, vmax, vmin2, vmax2, roi_borders=None):
    #this provides a nice workaround for pycortex opacity issues, at the cost of interactivity    
    # Get curvature
    curv = cortex.db.get_surfinfo(subject)
    # Adjust curvature contrast / color. Alternately, you could work
    # with curv.data, maybe threshold it, and apply a color map. 
    
    #standard
    curv.data = curv.data * .75 +0.1
    #alternative
    #curv.data = np.sign(curv.data) * .25
    #HCP adjustment
    #curv.data = curv.data * -2.5# 1.25 +0.1

    
    curv = cortex.Vertex(curv.data, subject, vmin=-1,vmax=1,cmap='gray')
    
    norm2 = Normalize(vmin2, vmax2)   
    
    vx = cortex.Vertex(data1, subject, cmap=cmap, vmin=vmin, vmax=vmax)
    
    # Map to RGB
    vx_rgb = np.vstack([vx.raw.red.data, vx.raw.green.data, vx.raw.blue.data])
    
    curv_rgb = np.vstack([curv.raw.red.data, curv.raw.green.data, curv.raw.blue.data])

    
    # Pick an arbitrary region to mask out
    # (in your case you could use np.isnan on your data in similar fashion)
    alpha = np.clip(norm2(data2), 0, 1)

    # Alpha mask
    display_data = (curv_rgb * (1-alpha)) + vx_rgb * alpha

    display_data /= 255


    #print(display_data.min())
    #print(display_data.max())
    
    if roi_borders is not None:
        display_data[:,roi_borders.astype('bool')] = 0#255-display_data[:,roi_borders.astype('bool')]#0#255
    
    # Create vertex RGB object out of R, G, B channels
    return cortex.VertexRGB(*display_data, subject)    



def simple_colorbar(vmin, vmax, cmap_name, ori, param_name):
    if ori == 'horizontal':
        fig, ax = pl.subplots(figsize=(8, 1))
        fig.subplots_adjust(bottom=0.5)
    elif ori == 'vertical':
        fig, ax = pl.subplots(figsize=(3 ,8))
        fig.subplots_adjust(right=0.5)
    elif ori == 'polar':
        fig, ax = pl.subplots(figsize=(3,3), subplot_kw={'projection': 'polar'})
        
    
    if cmap_name == 'hsvx2':
        top = cm.get_cmap('hsv', 256)
        bottom = cm.get_cmap('hsv', 256)

        newcolors = np.vstack((top(np.linspace(0, 1, 256)),
                       bottom(np.linspace(0, 1, 256))))
        cmap = colors.ListedColormap(newcolors, name='hsvx2')
        
    else:
        cmap = cm.get_cmap(cmap_name, 256)
    
    norm = colors.Normalize(vmin=vmin, vmax=vmax)
    
    if ori == 'polar':
        if 'Polar' in param_name:
            t = np.linspace(-np.pi,np.pi,200,endpoint=True)
            r = [0,1]
            rg, tg = np.meshgrid(r,t)
            ax.pcolormesh(t, r, tg.T, norm=norm, cmap=cmap)
            ax.set_yticklabels([])
            ax.set_xticklabels([])
            ax.set_theta_zero_location("W")
            ax.spines['polar'].set_visible(False)
        elif 'Ecc' in param_name:
            n = 200
            t = np.linspace(0,2*np.pi, n)
            r = np.linspace(0,1, n)
            rg, tg = np.meshgrid(r,t)
            c = tg
            ax.pcolormesh(t, r, c, norm = colors.Normalize(0, 2*np.pi), cmap=cmap)
            ax.tick_params(pad=1,labelsize=15)
            ax.spines['polar'].set_visible(False)
            box = ax.get_position()
            ax.set_yticklabels([])
            ax.set_xticklabels([])
            axl = fig.add_axes([0.97*box.xmin,0.5*(box.ymin+box.ymax), box.width/600,box.height*0.5])
            axl.spines['top'].set_visible(False)
            axl.spines['right'].set_visible(False)
            axl.spines['bottom'].set_visible(False)
            axl.yaxis.set_ticks_position('left')
            axl.xaxis.set_ticks_position('none')
            axl.set_xticklabels([])
            axl.set_yticklabels([f"{vmin:.1f}",f"{(vmin+vmax)/2:.1f}",f"{vmax:.1f}"],size = 'x-large')
            #axl.set_ylabel('$dva$\t\t', rotation=0, size='x-large')
            axl.yaxis.set_label_coords(box.xmax+30,0.4)
            axl.patch.set_alpha(0.5)            
    else:
        fig.colorbar(cm.ScalarMappable(norm=norm, cmap=cmap),
                 cax=ax, orientation=ori, label=param_name)
    
    fig.show()
    
    return fig


def reduced_graph_ft(param, param_indices, eigenvectors_path, eigenvectors_indices_path, pycortex_subj):

    pyc_eigenvectors = np.load(eigenvectors_path)
    
    nr_vertices = cortex.db.get_surfinfo(pycortex_subj).data.shape

    eigvecs_reduced = np.zeros((param.shape[0],pyc_eigenvectors.shape[1]))
    
    for eigvs in range(pyc_eigenvectors.shape[1]):
        zz = np.zeros(nr_vertices)
        zz[np.load(eigenvectors_indices_path)] = np.copy(pyc_eigenvectors[:,eigvs])
        eigvecs_reduced[:,eigvs] = np.copy(zz[param_indices])
        
    ft = np.dot(eigvecs_reduced.T,param)
    
    return eigvecs_reduced, ft 


def graph_randomization(data_max, data_min, eigvecs_reduced, ft):
        
    random_signs = np.sign(np.random.rand(len(ft))-0.5)
    perm_reduced = np.dot(eigvecs_reduced,random_signs*ft)
    
    return (data_max-data_min)*(perm_reduced - perm_reduced.min()) / (perm_reduced.max() - perm_reduced.min()) + data_min
    
    
    
    
    