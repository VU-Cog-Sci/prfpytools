import os
import numpy as np
import scipy as sp
import matplotlib.pyplot as pl
from matplotlib import colors
from tqdm import tqdm
import cortex
import yaml
from pathlib import Path
from collections import defaultdict as dd
import matplotlib.image as mpimg
from copy import deepcopy
from prfpy.model import Iso2DGaussianModel, Norm_Iso2DGaussianModel, DoG_Iso2DGaussianModel, CSS_Iso2DGaussianModel
from utils.preproc_utils import create_full_stim
from prfpy.rf import gauss2D_iso_cart
from prfpy.stimulus import PRFStimulus2D

opj = os.path.join

class results(object):
    def __init__(self):
        self.main_dict = dd(lambda:dd(lambda:dd(lambda:dd(dict))))
    
    def combine_results(self, results_folder,
                        timecourse_folder=None,
                        ref_volume_path=None,
                        calculate_CCrsq=False,
                        calculate_noise_ceiling=False):
        
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
            an_info['subj'] = subjects[i]
            
            if len(an_info['task_names'])>1:
                an_tasks.append('all')
            else:
                an_tasks.append(''.join(an_info['task_names']))
                
            if an_info['crossvalidate'] == True:
                an_runs.append(''.join([str(el) for el in an_info['fit_runs']]))#(len(an_info['fit_runs']))#
            else:
                an_runs.append('all')
            
            an_names.append(f"{subjects[i]}_fit-task-{an_tasks[i]}_fit-runs-{an_runs[i]}")
            
        unique_an_names = np.unique(np.array(an_names)).tolist()
        
        unique_an_results=dict()
        prf_stims=dict()
        
        #this is to combine multiple iterations (max) and different models fit on the same fold
        for an_name in tqdm(unique_an_names):
            current_an_infos = np.array(an_infos)[np.array(an_names)==an_name]
            merged_an_info = mergedict_AND(current_an_infos)
  
            r_r = dict()
            r_r_full=dict()
            
            try:            
                mask = np.load(opj(results_folder,f"{current_an_infos[0]['subj']}_mask_space-{current_an_infos[0]['fitting_space']}{current_an_infos[0]['analysis_time']}.npy"))
            except:
                mask = np.load(opj(results_folder,f"{current_an_infos[0]['subj']}_mask_space-{current_an_infos[0]['fitting_space']}{current_an_infos[0]['previous_analysis_time']}.npy"))
                
            for i, curr_an_info in enumerate(current_an_infos):
                if os.path.exists(opj(timecourse_folder,f"{curr_an_info['subj']}_timecourse_space-{curr_an_info['fitting_space']}{curr_an_info['analysis_time']}.npy")):
                    merged_an_info["timecourse_analysis_time"] = curr_an_info["analysis_time"]
                
                for model in curr_an_info["models_to_fit"]:
                    try:
                        raw_model_result = (np.load(opj(results_folder,f"{curr_an_info['subj']}_iterparams-{model.lower()}_space-{curr_an_info['fitting_space']}{curr_an_info['analysis_time']}.npy")))
                    except:
                        raw_model_result = (np.load(opj(results_folder,f"{curr_an_info['subj']}_iterparams-{model.lower()}_space-{curr_an_info['fitting_space']}{curr_an_info['previous_analysis_time']}.npy")))
                        
                    if i==0:
                        if model == 'norm':
                            r_r[f"Norm_{curr_an_info['norm_model_variant']}"] = np.copy(raw_model_result)
                        elif model == 'gauss':
                            r_r['Gauss'] = np.copy(raw_model_result)
                        else:
                            r_r[model] = np.copy(raw_model_result)
                    else:
                        if model == 'norm':
                            r_r[f"Norm_{curr_an_info['norm_model_variant']}"][r_r[f"Norm_{curr_an_info['norm_model_variant']}"][:,-1]<raw_model_result[:,-1]] = np.copy(raw_model_result[r_r[f"Norm_{curr_an_info['norm_model_variant']}"][:,-1]<raw_model_result[:,-1]])
                        elif model == 'gauss':
                            r_r['Gauss'][r_r['Gauss'][:,-1]<raw_model_result[:,-1]] = np.copy(raw_model_result[r_r['Gauss'][:,-1]<raw_model_result[:,-1]])
                        else:
                            r_r[model][r_r[model][:,-1]<raw_model_result[:,-1]] = np.copy(raw_model_result[r_r[model][:,-1]<raw_model_result[:,-1]])                            
                        
                    


            #move to full surface space so different masks can be handled when merging folds later
            for key in r_r:
                r_r_full[key] = np.zeros((mask.shape[0],r_r[key].shape[-1]))
                r_r_full[key][mask] = np.copy(r_r[key])   

            
                
            #housekeeping
            tc_paths = [str(path) for path in sorted(Path(timecourse_folder).glob(f"{merged_an_info['subj']}_timecourse_space-{merged_an_info['fitting_space']}_task-*_run-*.npy"))]    
            mask_paths = [tc_path.replace('timecourse_','mask_') for tc_path in tc_paths]
            all_task_names = np.unique(np.array([elem.replace('task-','') for path in tc_paths for elem in path.split('_')  if 'task' in elem]))
            all_runs = np.unique(np.array([int(elem.replace('run-','').replace('.npy','')) for path in tc_paths for elem in path.split('_')  if 'run' in elem]))
            
            if merged_an_info["crossvalidate"] and calculate_noise_ceiling:
                prf_stim = create_full_stim(screenshot_paths=[opj(timecourse_folder,f'task-{task}_screenshots') for task in merged_an_info['task_names']],
                                n_pix=merged_an_info['n_pix'],
                                discard_volumes=merged_an_info['discard_volumes'],
                                baseline_volumes_begin_end=merged_an_info['baseline_volumes_begin_end'],
                                dm_edges_clipping=merged_an_info['dm_edges_clipping'],
                                screen_size_cm=merged_an_info['screen_size_cm'],
                                screen_distance_cm=merged_an_info['screen_distance_cm'],
                                TR=merged_an_info['TR'],
                                task_names=merged_an_info['task_names'])
                
                tc_test = dict()
                tc_fit = dict()
           
                
                for task in merged_an_info['task_names']:
                    tc_runs=[]
                    
                    for run in all_runs:
                        mask_run = [np.load(mask_path) for mask_path in mask_paths if task in mask_path and f"run-{run}" in mask_path][0]

                        tc_run = ([np.load(tc_path) for tc_path in tc_paths if task in tc_path and f"run-{run}" in tc_path][0])
                        
                        tc_run *= (100/tc_run.mean(-1))[...,np.newaxis]
                        tc_run += (tc_run.mean(-1)-np.median(tc_run[...,prf_stim.late_iso_dict[task]], axis=-1))[...,np.newaxis]
                        
                        tc_runs_unmasked = np.zeros((mask_run.shape[0], tc_run.shape[-1]))
                        tc_runs_unmasked[mask_run] = np.copy(tc_run)
                        tc_runs.append(tc_runs_unmasked)
                                           
                    
                    
                    tc_test[task] = np.mean([tc_runs[i] for i in all_runs if i not in merged_an_info['fit_runs']], axis=0)
                    tc_fit[task] = np.mean([tc_runs[i] for i in all_runs if i in merged_an_info['fit_runs']], axis=0)                

                
                tc_all_test = np.concatenate(tuple([tc_test[task] for task in tc_test]), axis=-1)
                tc_all_fit = np.concatenate(tuple([tc_fit[task] for task in tc_fit]), axis=-1)
                
                noise_ceiling = 1-np.sum((tc_all_test-tc_all_fit)**2, axis=-1)/(tc_all_test.shape[-1]*tc_all_test.var(-1))
                
                r_r_full[f"Noise Ceiling (RSq)"] = np.copy(noise_ceiling)
                
            
            #calculate cross-condition r-squared
            
            if not merged_an_info["crossvalidate"] and calculate_CCrsq:
                for task in [tsk for tsk in all_task_names if tsk not in merged_an_info['task_names']]:
                    if task not in prf_stims:
                        prf_stims[task] = create_full_stim(screenshot_paths=[opj(timecourse_folder,f'task-{task}_screenshots')],
                                n_pix=merged_an_info['n_pix'],
                                discard_volumes=merged_an_info['discard_volumes'],
                                baseline_volumes_begin_end=merged_an_info['baseline_volumes_begin_end'],
                                dm_edges_clipping=merged_an_info['dm_edges_clipping'],
                                screen_size_cm=merged_an_info['screen_size_cm'],
                                screen_distance_cm=merged_an_info['screen_distance_cm'],
                                TR=merged_an_info['TR'],
                                task_names=[task])
                        

                    all_tcs_task = [np.load(tc_path) for tc_path in tc_paths if task in tc_path]
                    all_masks_task = [np.load(mask_path) for mask_path in mask_paths if task in mask_path]
                    
                    mask_task = np.product(all_masks_task, axis=0).astype('bool')
                    tc_full = np.zeros((mask_task.shape[0], all_tcs_task[0].shape[-1]))
                    
                    for i in range(len(all_tcs_task)):
                        tc_full[all_masks_task[i]] += all_tcs_task[i]

                    tc_full /= len(all_tcs_task)

                    common_mask = mask_task * mask
                    tc_comp = np.copy(tc_full[common_mask])              
                    
                    tc_comp *= (100/tc_comp.mean(-1))[...,np.newaxis]
                    tc_comp += (tc_comp.mean(-1)-np.median(tc_comp[...,prf_stims[task].late_iso_dict[task]], axis=-1))[...,np.newaxis]
                    

                    
                    for key in r_r:
                        gg = model_wrapper(key,
                                           stimulus=prf_stims[task],
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
        for key in set([key[:-13] for key in folds]):
            current_fold_infos = [unique_an_results[fold]['analysis_info'] for fold in folds if key in fold]
            for res in unique_an_results[folds[0]]:
                if 'info' in res:
                    combined_results[key+'_fit-runs-5050CVmedian'][res] = mergedict_AND(current_fold_infos)
                elif 'mask' in res:
                    combined_results[key+'_fit-runs-5050CVmedian'][res] = np.product([unique_an_results[fold][res] for fold in folds if key in fold], axis=0).astype('bool')
                else:
                    combined_results[key+'_fit-runs-5050CVmedian'][res] = np.median([unique_an_results[fold][res] for fold in folds if key in fold], axis=0)
                    
            #for res in unique_an_results[folds[0]]:     
                #final_mask = combined_results[key+'_fit-runs-5050CVmedian']['mask']
                #if 'Norm' in res:
                #    combined_results[key+'_fit-runs-5050CVmedian'][f'Size (fwhmax)_model-{res}'] = np.zeros(final_mask.shape)
                #    combined_results[key+'_fit-runs-5050CVmedian'][f'Surround Size (fwatmin)_model-{res}']= np.zeros(final_mask.shape)
                #    (combined_results[key+'_fit-runs-5050CVmedian'][f'Size (fwhmax)_model-{res}'][final_mask],
                #     combined_results[key+'_fit-runs-5050CVmedian'][f'Surround Size (fwatmin)_model-{res}'][final_mask]) = np.median([fwhmax_fwatmin(res, unique_an_results[fold][res][final_mask], False, False) for fold in folds if key in fold], axis=0)
                                
        for key in no_folds:
            combined_results[key] = deepcopy(unique_an_results[key])
            
 
        
        for an, an_res in combined_results.items():
            an_res['analysis_info']['timecourse_folder'] = timecourse_folder
            
            subj = an_res['analysis_info']['subj']
            space = an_res['analysis_info']['fitting_space']
            reduced_an_name = an.replace(f"{subj}_",'')
            mask = an_res['mask']
            
            for res in an_res:
                if 'mask' not in res and 'info' not in res:
                    self.main_dict[space][reduced_an_name][subj]['Results'][res] = deepcopy(an_res[res][mask])   
                else:
                    self.main_dict[space][reduced_an_name][subj][res] = deepcopy(an_res[res])
                
            
                       
        raw_tc_stats = dict()

        for subj in set(subjects):

            tc_raw = np.load(opj(timecourse_folder,f'{subj}_timecourse-raw_space-{space}.npy'))
            mask = np.load(opj(timecourse_folder,f'{subj}_mask-raw_space-{space}.npy'))
        
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
            
            for ke in self.main_dict[space]:
                self.main_dict[space][ke][subj]['Timecourse Stats'] = deepcopy(raw_tc_stats)
                
        return
        


    
    
    def process_results(self, results_dict, return_norm_profiles=False):
        for k, v in tqdm(results_dict.items()):
            if 'sub-' not in k and not k.isdecimal():
                self.process_results(v, return_norm_profiles)
            elif 'Results' in v and 'Processed Results' not in v:
                mask = v['mask']
                normalize_RFs = v['analysis_info']['normalize_RFs']
                #for suppression index computation
                prf_stim = PRFStimulus2D(screen_size_cm=v['analysis_info']['screen_size_cm'],
                             screen_distance_cm=v['analysis_info']['screen_distance_cm'],
                             design_matrix=np.zeros((v['analysis_info']['n_pix'],v['analysis_info']['n_pix'],10)),
                             TR=1.0)
                
                aperture = ((prf_stim.x_coordinates**2+prf_stim.y_coordinates**2)**0.5 < (prf_stim.screen_size_degrees/2))
    
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
    
                        if k2 == 'CSS':
                            processed_results['CSS Exponent'][k2][mask] =  np.copy(v2[:,5])
    
                        if k2 == 'DoG':
                            processed_results['Surround Amplitude'][k2][mask] = np.copy(v2[:,5])
                            processed_results['Size (sigma_2)'][k2][mask] = np.copy(v2[:,6])
                            processed_results['Size ratio (sigma_2/sigma_1)'][k2][mask] = v2[:,6]/v2[:,2]
                            (processed_results['Size (fwhmax)'][k2][mask],
                            processed_results['Surround Size (fwatmin)'][k2][mask]) = fwhmax_fwatmin(k2, v2, normalize_RFs)
                            processed_results['Suppression Index (full)'][k2][mask] = (v2[:,5] * v2[:,6]**2)/(v2[:,3] * v2[:,2]**2)
                            
                            processed_results['Suppression Index'][k2][mask] = suppression_index(prf_stim, aperture, v2, normalize_RFs)
    
                        elif 'Norm' in k2:
                            processed_results['Surround Amplitude'][k2][mask] = np.copy(v2[:,5])
                            processed_results['Size (sigma_2)'][k2][mask] = np.copy(v2[:,6])    
                            processed_results['Size ratio (sigma_2/sigma_1)'][k2][mask] = v2[:,6]/v2[:,2]
                            processed_results['Norm Param. B'][k2][mask] = np.copy(v2[:,7])
                            processed_results['Norm Param. D'][k2][mask] = np.copy(v2[:,8])
                            processed_results['Ratio (B/D)'][k2][mask] = v2[:,7]/v2[:,8]
                            processed_results['Suppression Index (full)'][k2][mask] = (v2[:,5] * v2[:,6]**2)/(v2[:,3] * v2[:,2]**2)
                            
                            processed_results['Suppression Index'][k2][mask] = suppression_index(prf_stim, aperture, v2, normalize_RFs)
                            
    
                            if return_norm_profiles and len(mask.shape)<2:
                                processed_results['pRF Profiles'][k2] = np.zeros((mask.shape[0],1000))
                                ((processed_results['Size (fwhmax)'][k2][mask],
                                processed_results['Surround Size (fwatmin)'][k2][mask]),
                                processed_results['pRF Profiles'][k2][mask]) = fwhmax_fwatmin(k2, v2, normalize_RFs, True)
                            else:
                                
                                (processed_results['Size (fwhmax)'][k2][mask],
                                processed_results['Surround Size (fwatmin)'][k2][mask]) = fwhmax_fwatmin(k2, v2, normalize_RFs, False)
    
                        else:
                            processed_results['Size (fwhmax)'][k2][mask] = fwhmax_fwatmin(k2, v2, normalize_RFs)

                        ####copy beta and cross-cond rsq to processed results
                            #####################
                    elif isinstance(v2, np.ndarray) and v2.ndim == 1 and 'model' in k2:
                        processed_results[k2.split('_model-')[0]][k2.split('_model-')[1]][mask] = np.copy(v2)
                        
                    elif isinstance(v2, np.ndarray) and v2.ndim == 1 and 'Noise Ceiling' in k2:
                        processed_results['Noise Ceiling'][k2][mask] = np.copy(v2)


                        
    
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


def make_2d_cmap(cmap_name):
    #####
    cmap = mpimg.imread(os.path.join(os.path.split(cortex.database.default_filestore)[
                          0], 'colormaps',f'{cmap_name}.png'))
    cmap = colors.rgb_to_hsv(cmap[...,:3])
    hue, alpha = np.meshgrid(cmap[...,0], 1-np.linspace(0, 1, 256))
    hsv = np.zeros(list(hue.shape)+[3])

    hsv[..., 0] = cmap[...,0]  
    hsv[..., 1] = np.ones_like(alpha)
    hsv[..., 2] = np.ones_like(alpha)
    hsv[-1,:,2] = 0

    rgb = colors.hsv_to_rgb(hsv)
    rgba = np.vstack((rgb.T, alpha[..., np.newaxis].T)).T
    pl.imshow(rgba)
    hsv_fn = os.path.join(os.path.split(cortex.database.default_filestore)[
                          0], 'colormaps', f'{cmap_name}_2D_alpha.png')
    mpimg.imsave(hsv_fn, rgba)
    

def create_retinotopy_colormaps():
    hue, alpha = np.meshgrid(np.linspace(
        0, 1, 256, endpoint=False), 1-np.linspace(0, 1, 256))

    hsv = np.zeros(list(hue.shape)+[3])

    hsv[..., 0] = hue 
    hsv[..., 1] = np.ones_like(alpha)
    hsv[..., 2] = np.ones_like(alpha)
    hsv[-1,:,2] = 0

    rgb = colors.hsv_to_rgb(hsv)
    rgba = np.vstack((rgb.T, alpha[..., np.newaxis].T)).T
    pl.imshow(rgba)
    hsv_fn = os.path.join(os.path.split(cortex.database.default_filestore)[
                          0], 'colormaps', 'Retinotopy_HSV_alpha.png')
    mpimg.imsave(hsv_fn, rgba)

    hue, alpha = np.meshgrid(
        np.fmod(np.linspace(0, 2, 256), 1.0), 1-np.linspace(0, 1, 256))
    hsv = np.zeros(list(hue.shape)+[3])

    hsv[..., 0] = hue
    hsv[..., 1] = np.ones_like(alpha)
    hsv[..., 2] = np.ones_like(alpha)
    hsv[-1,:,2] = 0

    rgb = colors.hsv_to_rgb(hsv)
    rgba = np.vstack((rgb.T, alpha[..., np.newaxis].T)).T
    pl.imshow(rgba)
    hsv_fn = os.path.join(os.path.split(cortex.database.default_filestore)[
                          0], 'colormaps', 'Retinotopy_HSV_2x_alpha.png')
    mpimg.imsave(hsv_fn, rgba)


