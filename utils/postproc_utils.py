import os
import numpy as np
import scipy as sp
import matplotlib.pyplot as pl
from matplotlib import colors
import cortex
from collections import defaultdict as dd
import matplotlib.image as mpimg


opj = os.path.join

class results(object):
    def __init__(self):
        self.main_dict = dd(lambda:dd(lambda:dd(dict)))
    
    def combine_results(self, subj, fitting_space, results_folder, suffix_list,
                        raw_timecourse_path=None, normalize_RFs=False, ref_img_path=None):
        try:
            mask = np.load(opj(results_folder, subj+'_mask_space-'+fitting_space+'.npy'))
        except Exception as e:
            print(e)
            pass
        
        g_l = []
        c_l = []
        d_l = []
        n_l = []
        masks = []
        nc_l = []
        
        for suf_list in suffix_list:
            for i, suffix in enumerate(suf_list):
                if i == 0:
                    try:
                        gauss = np.load(opj(results_folder,subj+'_iterparams-gauss_space-'+fitting_space+suffix+'.npy'))
                        css = np.load(opj(results_folder,subj+'_iterparams-css_space-'+fitting_space+suffix+'.npy'))
                        dog = np.load(opj(results_folder,subj+'_iterparams-dog_space-'+fitting_space+suffix+'.npy'))
                        norm = np.load(opj(results_folder,subj+'_iterparams-norm_space-'+fitting_space+suffix+'.npy'))
                        mask = np.load(opj(results_folder, subj+'_mask_space-'+fitting_space+suffix+'.npy'))
                        noise_ceiling = np.load(opj(results_folder, subj+'_noise-ceiling_space-'+fitting_space+suffix+'.npy'))
                        print(f"gauss iter {np.sum(gauss[:,-1]>0.5)}")
                    except Exception as e:
                        print(e)
                        pass
                else:
                    try:
                        gauss_it = np.load(opj(results_folder,subj+'_iterparams-gauss_space-'+fitting_space+suffix+'.npy'))
                        css_it = np.load(opj(results_folder,subj+'_iterparams-css_space-'+fitting_space+suffix+'.npy'))
                        dog_it = np.load(opj(results_folder,subj+'_iterparams-dog_space-'+fitting_space+suffix+'.npy'))
                        norm_it = np.load(opj(results_folder,subj+'_iterparams-norm_space-'+fitting_space+suffix+'.npy'))
                        gauss[(gauss[:,-1]<gauss_it[:,-1])] = np.copy(gauss_it[(gauss[:,-1]<gauss_it[:,-1])])
                        css[(css[:,-1]<css_it[:,-1])] = np.copy(css_it[(css[:,-1]<css_it[:,-1])])
                        dog[(dog[:,-1]<dog_it[:,-1])] = np.copy(dog_it[(dog[:,-1]<dog_it[:,-1])])
                        norm[(norm[:,-1]<norm_it[:,-1])] = np.copy(norm_it[(norm[:,-1]<norm_it[:,-1])])
                    except Exception as e:
                        print(e)
                        pass
            try:
    
                gauss_full = np.zeros((mask.shape[0],gauss.shape[-1]))
                css_full = np.zeros((mask.shape[0],css.shape[-1]))
                dog_full = np.zeros((mask.shape[0],dog.shape[-1]))
                norm_full = np.zeros((mask.shape[0],norm.shape[-1]))
                
                gauss_full[mask] = np.copy(gauss)
                css_full[mask] = np.copy(css)
                dog_full[mask] = np.copy(dog)
                norm_full[mask] = np.copy(norm)
                
    
                print(f"gauss fold {np.sum(gauss_full[:,-1]>0.5)}")
                print(mask.shape)
    
                
                g_l.append(gauss_full)
                c_l.append(css_full)
                d_l.append(dog_full)
                n_l.append(norm_full)
                masks.append(mask)
                noise_ceiling_full = np.zeros(mask.shape[0])
                noise_ceiling_full[mask] = np.copy(noise_ceiling)
                nc_l.append(noise_ceiling_full)
    
            except Exception as e:
                print(e)
                pass
    
        try:     
            gauss_full = np.median(g_l, axis=0)
            css_full = np.median(c_l, axis=0)
            dog_full = np.median(d_l, axis=0)
            norm_full = np.median(n_l, axis=0)
        
            noise_ceiling_full = np.median(nc_l, axis=0)
        except Exception as e:
            print(e)
            pass    
        print(f"{[np.sum(mask) for mask in masks]}")
        print(f"{np.sum(np.prod(masks, axis=0))}")
        print(f"gauss median {np.sum(gauss_full[:,-1]>0.5)}")
        print(f"norm median {np.sum(norm_full[:,-1]>0.5)}")
    
        try:
            mask = np.load(opj(results_folder, subj+'_mask_space-'+fitting_space+'.npy'))
        except Exception as e:
            print(e)
            mask = (gauss_full[:,-1]>0)
            pass
    
        try:     
            gauss = np.copy(gauss_full[mask])
            css = np.copy(css_full[mask])
            dog = np.copy(dog_full[mask])
            norm = np.copy(norm_full[mask])
            noise_ceiling = np.copy(noise_ceiling_full[mask])
            print(f"gauss in mask: {np.sum(gauss[:,-1]>0.5)}")
            print(f"norm in mask: {np.sum(norm[:,-1]>0.5)}")
        except Exception as e:
            print(e)
            noise_ceiling=0
            pass      
        
                    
        raw_tc_stats = dict()
        if raw_timecourse_path is not None:
            
            
            tc_raw = np.load(raw_timecourse_path)
            
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
        
        return {'Gauss':gauss, #'Gauss grid':gauss_grid, 'Norm grid':norm_grid, 
                'CSS':css, 'DoG':dog, 'Norm':norm,
                'mask':mask, 'noise_ceiling':noise_ceiling, 'normalize_RFs':normalize_RFs, 
                'Timecourse Stats':raw_tc_stats, 'ref_img_path':ref_img_path}
    
    
    def process_results(self, results_dict, return_norm_profiles):
        for k, v in results_dict.items():
            if 'sub-' not in k:
                self.process_results(v, return_norm_profiles)
            elif 'Results' in v and 'Processed Results' not in v:
                normalize_RFs = v['Results']['normalize_RFs']
                mask = v['Results']['mask']
    
                #store processed results in nested default dictionary
                processed_results = dd(lambda:dd(lambda:np.zeros(mask.shape)))
    
                #loop over contents of single-subject analysis results (models and mask)
                for k2, v2 in v['Results'].items():
                    if k2 != 'mask' and isinstance(v2, np.ndarray) and 'grid' not in k2 and 'Stats' not in k2 and k2 != 'noise_ceiling':
    
                        processed_results['RSq'][k2][mask] = np.copy(v2[:,-1])
                        processed_results['Eccentricity'][k2][mask] = np.sqrt(v2[:,0]**2+v2[:,1]**2)
                        processed_results['Polar Angle'][k2][mask] = np.arctan2(v2[:,1], v2[:,0])
                        processed_results['Amplitude'][k2][mask] = np.copy(v2[:,3])
    
                        if k2 == 'CSS':
                            processed_results['CSS Exponent'][k2][mask] =  np.copy(v2[:,5])
    
                        if k2 == 'DoG':
                            (processed_results['Size (fwhmax)'][k2][mask],
                            processed_results['Surround Size (fwatmin)'][k2][mask]) = fwhmax_fwatmin(k2, v2, normalize_RFs)
    
                        elif k2 == 'Norm':
                            processed_results['Norm Param. B'][k2][mask] = np.copy(v2[:,7])
                            processed_results['Norm Param. D'][k2][mask] = np.copy(v2[:,8])
                            processed_results['Ratio (B/D)'][k2][mask] = v2[:,7]/v2[:,8]
    
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
                    elif 'Stats' in k2:
                        v[k2] = v['Results'][k2]
                    elif k2 == 'noise_ceiling':
                        processed_results['Noise Ceiling']['Noise Ceiling'][mask] = np.copy(v2)
                        
    
                v['Processed Results'] = {ke : dict(va) for ke, va in processed_results.items()}
    




def create_retinotopy_colormaps():
    hue, alpha = np.meshgrid(np.linspace(
        0, 1, 256, endpoint=False), 1-np.linspace(0, 1, 256))

    hsv = np.zeros(list(hue.shape)+[3])


    # convert angles to colors, using correlations as weights
    hsv[..., 0] = hue  # angs_discrete  # angs_n
    # np.sqrt(rsq) #np.ones_like(rsq)  # np.sqrt(rsq)
    hsv[..., 1] = np.ones_like(alpha)
    # np.nan_to_num(rsq ** -3) # np.ones_like(rsq)#n
    hsv[..., 2] = np.ones_like(alpha)
    hsv[-1,:,2] = 0

    rgb = colors.hsv_to_rgb(hsv)
    rgba = np.vstack((rgb.T, alpha[..., np.newaxis].T)).T
    pl.imshow(rgba)
    hsv_fn = os.path.join(os.path.split(cortex.database.default_filestore)[
                          0], 'colormaps', 'Retinotopy_HSV_alpha.png')
    sp.misc.imsave(hsv_fn, rgba)

    hue, alpha = np.meshgrid(
        np.fmod(np.linspace(0, 2, 256), 1.0), 1-np.linspace(0, 1, 256))
    hsv = np.zeros(list(hue.shape)+[3])
    # convert angles to colors, using correlations as weights
    hsv[..., 0] = hue  # angs_discrete  # angs_n
    # np.sqrt(rsq) #np.ones_like(rsq)  # np.sqrt(rsq)
    hsv[..., 1] = np.ones_like(alpha)
    # np.nan_to_num(rsq ** -3) # np.ones_like(rsq)#n
    hsv[..., 2] = np.ones_like(alpha)
    hsv[-1,:,2] = 0

    rgb = colors.hsv_to_rgb(hsv)
    rgba = np.vstack((rgb.T, alpha[..., np.newaxis].T)).T
    pl.imshow(rgba)
    hsv_fn = os.path.join(os.path.split(cortex.database.default_filestore)[
                          0], 'colormaps', 'Retinotopy_HSV_2x_alpha.png')
    sp.misc.imsave(hsv_fn, rgba)
    #####
    jet = mpimg.imread('/Users/marcoaqil/pycortex/filestore/colormaps/jet_r.png')
    jet = colors.rgb_to_hsv(jet[...,:3])

    hue, alpha = np.meshgrid(jet[...,0], 1-np.linspace(0, 1, 256))

    hsv = np.zeros(list(hue.shape)+[3])

    # convert angles to colors, using correlations as weights
    hsv[..., 0] = hue  # angs_discrete  # angs_n
    # np.sqrt(rsq) #np.ones_like(rsq)  # np.sqrt(rsq)
    hsv[..., 1] = np.ones_like(alpha)
    # np.nan_to_num(rsq ** -3) # np.ones_like(rsq)#n
    hsv[..., 2] = np.ones_like(alpha)#rdbu[...,2]
    hsv[-1,:,2] = 0

    rgb = colors.hsv_to_rgb(hsv)
    rgba = np.vstack((rgb.T, alpha[..., np.newaxis].T)).T
    pl.imshow(rgba)
    hsv_fn = os.path.join(os.path.split(cortex.database.default_filestore)[
                          0], 'colormaps', 'Jet_r_2D_alpha.png')
    sp.misc.imsave(hsv_fn, rgba)



def fwhmax_fwatmin(model, params, normalize_RFs=False, return_profiles=False):
    model = model.lower()
    x=np.linspace(-50,50,1000).astype('float32')

    prf = params[...,3] * np.exp(-0.5*x[...,np.newaxis]**2 / params[...,2]**2)
    vol_prf =  2*np.pi*params[...,2]**2

    if model in ['dog', 'norm']:
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
        elif model == 'norm':
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
        elif model == 'norm':
            profile = (prf + params[...,7])/(srf + params[...,8]) - params[...,7]/params[...,8]


    half_max = np.max(profile, axis=0)/2
    fwhmax = np.abs(2*x[np.argmin(np.abs(half_max-profile), axis=0)])


    if model in ['dog', 'norm']:

        min_profile = np.min(profile, axis=0)
        fwatmin = np.abs(2*x[np.argmin(np.abs(min_profile-profile), axis=0)])

        result = fwhmax, fwatmin
    else:
        result = fwhmax

    if return_profiles:
        if model == 'norm':
            #accounting for the previous subtraction of baseline
            profile += params[...,7]/params[...,8]

        return result, profile.T
    else:
        return result
            
                                
        
