import numpy as np
from scipy.special import gamma
from scipy.special import gamma, gammainc
from scipy.optimize import brentq
import matplotlib.pyplot as plt

import lenstronomy.Util.kernel_util as kernel_util
import lenstronomy.Util.util as util
from astroObjectAnalyser.DataAnalysis.analysis import Analysis
from astroObjectAnalyser.astro_object_superclass import StrongLensSystem
from lenstronomy.Workflow.fitting_sequence import FittingSequence
#from lenstronomy.Plots.output_plots import ModelPlot
#import lenstronomy.Plots.output_plots as out_plot
from lenstronomy.Plots.model_plot import ModelPlot
from lenstronomy.Util import param_util

import paperfig as pf


def get_total_flux(kwargs):
    """
    Compute total flux from lenstronomy kwargs dictionary.
    """
    n_sersic = kwargs['n_sersic']
    e1 = kwargs['e1']
    e2 = kwargs['e2']
    amp = kwargs['amp']
    r_sersic = kwargs['R_sersic']
    
    phi, q = param_util.ellipticity2phi_q(e1, e2)
    
    b_n = 1.9992 * n_sersic - 0.3271
    flux = q * 2*np.pi * n_sersic * amp * r_sersic**2 * np.exp(b_n) * b_n**(-2*n_sersic) * gamma(2*n_sersic)
    
    return flux


def _enclosed_flux(r, amp, n, r_s):
    """
    Compute enclosed flux within r for a Sersic profile..
    """
    bn = 1.9992 * n - 0.3271
    x = bn * (r/r_s)**(1./n)
    
    return amp * r_s**2 * 2 * np.pi * n * np.exp(bn) / bn**(2*n) * gammainc(2*n, x) * gamma(2*n)


def _total_flux(amp, n, r_s):
    """
    Compute total flux for a Sersic profile.
    """
    bn = 1.9992 * n - 0.3271

    return amp * r_s**2 * 2 * np.pi * n * np.exp(bn) / bn**(2*n) * gamma(2*n)


def get_half_light_radius(kwargs_light):
    """
    Compute the half-light radius.
    """
    tot_flux = _total_flux(kwargs_light[0]['amp'], kwargs_light[0]['n_sersic'], 
                                  kwargs_light[0]['R_sersic']) + \
                        _total_flux(kwargs_light[1]['amp'], kwargs_light[1]['n_sersic'], 
                                  kwargs_light[1]['R_sersic'])
    def func(r):
        return _enclosed_flux(r, kwargs_light[0]['amp'], kwargs_light[0]['n_sersic'], 
                                  kwargs_light[0]['R_sersic']) + \
                        _enclosed_flux(r, kwargs_light[1]['amp'], kwargs_light[1]['n_sersic'], 
                                  kwargs_light[1]['R_sersic']) - tot_flux/2.
    
    return brentq(func, 0.01, 10) #min(kwargs_light[0]['R_sersic'], kwargs_light[1]['R_sersic']),
                  #max(kwargs_light[0]['R_sersic'], kwargs_light[1]['R_sersic'])
                 #)
        
        
def get_half_light_radius_triple_sersic(kwargs_light):
    """
    Compute the half-light radius.
    """
    tot_flux = _total_flux(kwargs_light[0]['amp'], kwargs_light[0]['n_sersic'], 
                                  kwargs_light[0]['R_sersic']) + \
                        _total_flux(kwargs_light[1]['amp'], kwargs_light[1]['n_sersic'], 
                                  kwargs_light[1]['R_sersic']) + \
                        _total_flux(kwargs_light[2]['amp'], kwargs_light[2]['n_sersic'], 
                                  kwargs_light[2]['R_sersic'])
    def func(r):
        return _enclosed_flux(r, kwargs_light[0]['amp'], kwargs_light[0]['n_sersic'], 
                                  kwargs_light[0]['R_sersic']) + \
                        _enclosed_flux(r, kwargs_light[1]['amp'], kwargs_light[1]['n_sersic'], 
                                  kwargs_light[1]['R_sersic']) + \
                        _enclosed_flux(r, kwargs_light[2]['amp'], kwargs_light[2]['n_sersic'], 
                                  kwargs_light[2]['R_sersic']) - tot_flux/2.
    
    return brentq(func, 0.01, 10) #min(kwargs_light[0]['R_sersic'], kwargs_light[1]['R_sersic']),
                  #max(kwargs_light[0]['R_sersic'], kwargs_light[1]['R_sersic'])
                 #)


def fit_galaxy_light(kwargs_data, kwargs_psf, mask, ra_offset, dec_offset, bound=0.5, 
                     fix_params=None, plot=True, do_mcmc=False, mcmc_steps=1000, custom_logL=None,
                     kwargs_light_init=None, linear_solver=True
                    ):
    """
    Fit galaxy light to data using lenstronomy.
    """
    lens_light_model_list = ['SERSIC_ELLIPSE', 'SERSIC_ELLIPSE']

    kwargs_model = {'lens_light_model_list': lens_light_model_list,
                    #'joint_len'
                   }
    kwargs_constraints = {'joint_lens_light_with_lens_light': [[0, 1, ['center_x', 'center_y', 
                                                                       #'e1', 'e2'
                                                                      ]
                                                               ]],
                          'linear_solver': linear_solver
                         }
    kwargs_numerics_galfit = {'supersampling_factor': 1}
    kwargs_likelihood = {'check_bounds': True, 
                         'image_likelihood_mask_list': [mask], 
                         'check_positive_flux': True,
                         'custom_logL_addition': custom_logL
                        }

    image_band = [kwargs_data, kwargs_psf, kwargs_numerics_galfit]
    multi_band_list = [image_band]
    kwargs_data_joint = {'multi_band_list': multi_band_list, 'multi_band_type': 'multi-linear'}

    # lens light model choices
    fixed_lens_light = []
    kwargs_lens_light_init = []
    kwargs_lens_light_sigma = []
    kwargs_lower_lens_light = []
    kwargs_upper_lens_light = []

    # first Sersic component
    fixed_lens_light.append({}) #'n_sersic': 4.})
    kwargs_lens_light_init.append({'amp': 50., 'R_sersic': .1, 'n_sersic': 4, 'e1': 0, 'e2': 0, 'center_x': ra_offset, 'center_y': dec_offset})
    kwargs_lens_light_sigma.append({'amp': 0.1, 'n_sersic': 0.5, 'R_sersic': 0.2, 'e1': 0.1, 'e2': 0.1, 'center_x': 0.5, 'center_y': 0.5})
    kwargs_lower_lens_light.append({'amp': .1, 'e1': -0.5, 'e2': -0.5, 'R_sersic': 0.01, 'n_sersic': 0.5, 'center_x': ra_offset-bound, 'center_y': dec_offset-bound})
    kwargs_upper_lens_light.append({'amp': 1000., 'e1': 0.5, 'e2': 0.5, 'R_sersic': 10, 'n_sersic': 8, 'center_x': ra_offset+bound, 'center_y': dec_offset+bound})

    # second Sersic component
    fixed_lens_light.append({}) #'n_sersic': 1.})
    kwargs_lens_light_init.append({'amp': 50., 'R_sersic': .5, 'n_sersic': 1, 'e1': 0, 'e2': 0, 'center_x': 0, 'center_y': 0})
    kwargs_lens_light_sigma.append({'amp': 0.1, 'n_sersic': 0.5, 'R_sersic': 0.2, 'e1': 0.1, 'e2': 0.1, 'center_x': 0.1, 'center_y': 0.1})
    kwargs_lower_lens_light.append({'amp': .1, 'e1': -0.5, 'e2': -0.5, 'R_sersic': 0.01, 'n_sersic': 0.5, 'center_x': -10, 'center_y': -10})
    kwargs_upper_lens_light.append({'amp': 500., 'e1': 0.5, 'e2': 0.5, 'R_sersic': 10, 'n_sersic': 8, 'center_x': 10, 'center_y': 10})
        
        
    if kwargs_light_init is not None:
        kwargs_lens_light_init = kwargs_light_init
        
    if fix_params is not None:
        fixed_lens_light = fix_params
        
    lens_light_params = [kwargs_lens_light_init, kwargs_lens_light_sigma, fixed_lens_light, kwargs_lower_lens_light, kwargs_upper_lens_light]
    
    kwargs_params = {'lens_light_model': lens_light_params}
    
    kwargs_result = {'kwargs_lens_light': kwargs_lens_light_init}

    fitting_seq = FittingSequence(kwargs_data_joint, 
                                  kwargs_model, kwargs_constraints, 
                                  kwargs_likelihood, kwargs_params)

    fitting_kwargs_list = [['PSO', {'sigma_scale': 1., 'n_particles': 100, 'n_iterations': 100}]]
    
    if do_mcmc:
        fitting_kwargs_list.append(['MCMC', {'n_burn': 0, 'n_run': mcmc_steps, 
                                           'walkerRatio': 8, 'sigma_scale': 0.05,
                                           'progress': True
                                           }])

    fit_output = fitting_seq.fit_sequence(fitting_kwargs_list)
    kwargs_result = fitting_seq.best_fit()
    
    lens_result = kwargs_result['kwargs_lens']
    lens_light_result = kwargs_result['kwargs_lens_light']
    source_result = kwargs_result['kwargs_source']
    ps_result = kwargs_result['kwargs_ps']
    
#     lensPlot = ModelPlot(multi_band_list, kwargs_model, kwargs_result, arrow_size=0.02, cmap_string=cmap,
#                              likelihood_mask_list=[mask], multi_band_type='multi-linear',
#                              bands_compute=None)
    
    if plot:
        plot_fitting(multi_band_list, kwargs_model, kwargs_result, mask, linear_solver=linear_solver)
    
    # flux1 = get_total_flux(lens_light_result[0])
    # flux2 = get_total_flux(lens_light_result[1])
    
    # return get_half_light_radius(kwargs_result['kwargs_lens_light']), flux2/(flux1+flux2), kwargs_result, multi_band_list, kwargs_model
    return kwargs_result, multi_band_list, kwargs_model, fit_output


def fit_galaxy_light_triple_sersic(kwargs_data, kwargs_psf, mask, ra_offset, dec_offset, bound=0.5, 
                     fix_params=None, plot=True, do_mcmc=False, mcmc_steps=1000, custom_logL=None,
                     kwargs_light_init=None, linear_solver=True
                    ):
    """
    Fit galaxy light to data using lenstronomy.
    """
    lens_light_model_list = ['SERSIC_ELLIPSE', 'SERSIC_ELLIPSE', 'SERSIC_ELLIPSE']

    kwargs_model = {'lens_light_model_list': lens_light_model_list,
                    #'joint_len'
                   }
    kwargs_constraints = {'joint_lens_light_with_lens_light': [[0, 1, ['center_x', 'center_y', #'e1', 'e2'
                                                                      ]],
                                                               [0, 2, ['center_x', 'center_y', #'e1', 'e2'
                                                                      ]]
                                                              ],
                          'linear_solver': linear_solver
                         }
    kwargs_numerics_galfit = {'supersampling_factor': 1}
    kwargs_likelihood = {'check_bounds': True, 
                         'image_likelihood_mask_list': [mask], 
                         'check_positive_flux': True,
                         'custom_logL_addition': custom_logL
                        }

    image_band = [kwargs_data, kwargs_psf, kwargs_numerics_galfit]
    multi_band_list = [image_band]
    kwargs_data_joint = {'multi_band_list': multi_band_list, 'multi_band_type': 'multi-linear'}

    # lens light model choices
    fixed_lens_light = []
    kwargs_lens_light_init = []
    kwargs_lens_light_sigma = []
    kwargs_lower_lens_light = []
    kwargs_upper_lens_light = []

    # first Sersic component
    fixed_lens_light.append({}) #'n_sersic': 4.})
    kwargs_lens_light_init.append({'amp': 50., 'R_sersic': .1, 'n_sersic': 4, 'e1': 0, 'e2': 0, 'center_x': ra_offset, 'center_y': dec_offset})
    kwargs_lens_light_sigma.append({'amp': 0.1, 'n_sersic': 0.5, 'R_sersic': 0.2, 'e1': 0.1, 'e2': 0.1, 'center_x': 0.5, 'center_y': 0.5})
    kwargs_lower_lens_light.append({'amp': .1, 'e1': -0.5, 'e2': -0.5, 'R_sersic': 0.01, 'n_sersic': 0.5, 'center_x': ra_offset-bound, 'center_y': dec_offset-bound})
    kwargs_upper_lens_light.append({'amp': 1000., 'e1': 0.5, 'e2': 0.5, 'R_sersic': 10, 'n_sersic': 8, 'center_x': ra_offset+bound, 'center_y': dec_offset+bound})

    # second Sersic component
    fixed_lens_light.append({}) #'n_sersic': 1.})
    kwargs_lens_light_init.append({'amp': 50., 'R_sersic': .5, 'n_sersic': 1, 'e1': 0, 'e2': 0, 'center_x': ra_offset, 'center_y': dec_offset})
    kwargs_lens_light_sigma.append({'amp': 0.1, 'n_sersic': 0.5, 'R_sersic': 0.2, 'e1': 0.1, 'e2': 0.1, 'center_x': 0.1, 'center_y': 0.1})
    kwargs_lower_lens_light.append({'amp': .1, 'e1': -0.5, 'e2': -0.5, 'R_sersic': 0.01, 'n_sersic': 0.5, 'center_x': ra_offset-bound, 'center_y': dec_offset-bound})
    kwargs_upper_lens_light.append({'amp': 500., 'e1': 0.5, 'e2': 0.5, 'R_sersic': 10, 'n_sersic': 8, 'center_x': ra_offset+bound, 'center_y': dec_offset+bound})
    
     # third Sersic component
    fixed_lens_light.append({}) #'n_sersic': 1.})
    kwargs_lens_light_init.append({'amp': 50., 'R_sersic': .5, 'n_sersic': 1, 'e1': 0, 'e2': 0, 'center_x': ra_offset, 'center_y': dec_offset})
    kwargs_lens_light_sigma.append({'amp': 0.1, 'n_sersic': 0.5, 'R_sersic': 0.2, 'e1': 0.1, 'e2': 0.1, 'center_x': 0.1, 'center_y': 0.1})
    kwargs_lower_lens_light.append({'amp': .1, 'e1': -0.5, 'e2': -0.5, 'R_sersic': 0.01, 'n_sersic': 0.5, 'center_x': ra_offset-bound, 'center_y': dec_offset-bound})
    kwargs_upper_lens_light.append({'amp': 500., 'e1': 0.5, 'e2': 0.5, 'R_sersic': 10, 'n_sersic': 8, 'center_x': ra_offset+bound, 'center_y': dec_offset+bound})
        
    if kwargs_light_init is not None:
        kwargs_lens_light_init = kwargs_light_init
        
    if fix_params is not None:
        fixed_lens_light = fix_params
        
    lens_light_params = [kwargs_lens_light_init, kwargs_lens_light_sigma, fixed_lens_light, kwargs_lower_lens_light, kwargs_upper_lens_light]
    
    kwargs_params = {'lens_light_model': lens_light_params}
    
    kwargs_result = {'kwargs_lens_light': kwargs_lens_light_init}

    fitting_seq = FittingSequence(kwargs_data_joint, 
                                  kwargs_model, kwargs_constraints, 
                                  kwargs_likelihood, kwargs_params)

    fitting_kwargs_list = [['PSO', {'sigma_scale': 1., 'n_particles': 100, 'n_iterations': 100}]]
    
    if do_mcmc:
        fitting_kwargs_list.append(['MCMC', {'n_burn': 0, 'n_run': mcmc_steps, 
                                           'walkerRatio': 8, 'sigma_scale': 0.05,
                                           'progress': True
                                           }])

    fit_output = fitting_seq.fit_sequence(fitting_kwargs_list)
    kwargs_result = fitting_seq.best_fit()
    
    lens_result = kwargs_result['kwargs_lens']
    lens_light_result = kwargs_result['kwargs_lens_light']
    source_result = kwargs_result['kwargs_source']
    ps_result = kwargs_result['kwargs_ps']
    
#     lensPlot = ModelPlot(multi_band_list, kwargs_model, kwargs_result, arrow_size=0.02, cmap_string=cmap,
#                              likelihood_mask_list=[mask], multi_band_type='multi-linear',
#                              bands_compute=None)
    
    if plot:
        plot_fitting(multi_band_list, kwargs_model, kwargs_result, mask, linear_solver=linear_solver)
    
    # flux1 = get_total_flux(lens_light_result[0])
    # flux2 = get_total_flux(lens_light_result[1])
    
    # return get_half_light_radius(kwargs_result['kwargs_lens_light']), flux2/(flux1+flux2), kwargs_result, multi_band_list, kwargs_model
    return kwargs_result, multi_band_list, kwargs_model, fit_output


def plot_fitting(multi_band_list, kwargs_model, kwargs_result, mask, linear_solver=True):
    """
    """
    cmap = pf.cmap
    lens_plot = ModelPlot(multi_band_list, kwargs_model, kwargs_result,
                          arrow_size=0.02, cmap_string=cmap,
                          image_likelihood_mask_list=[mask], #kwargs_likelihood['image_likelihood_mask_list'],
                          multi_band_type='multi-linear', linear_solver=linear_solver
                          )


    f, axes = plt.subplots(1, 3, figsize=(16, 8), sharex=False, sharey=False)

    lens_plot.data_plot(ax=axes[0], band_index=0, cmap=cmap, v_max=0.5)
    lens_plot.model_plot(ax=axes[1], band_index=0, cmap=cmap, v_max=0.5)
    lens_plot.normalized_residual_plot(ax=axes[2], v_min=-3, v_max=3, cmap=pf.msh_cmap2)
    f.tight_layout()
    #f.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0., hspace=0.05)
    plt.show()

    f, axes = plt.subplots(1, 3, figsize=(16, 8), sharex=False, sharey=False)

    lens_plot.decomposition_plot(ax=axes[0], text='Lens light', lens_light_add=True, unconvolved=True, cmap=cmap, v_max=0.5)
    lens_plot.decomposition_plot(ax=axes[1], text='Lens light convolved', lens_light_add=True, cmap=cmap, v_max=0.5)
    lens_plot.subtract_from_data_plot(ax=axes[2], text='Data - Lens Light', lens_light_add=True, v_max=0.5)
    f.tight_layout()
    #f.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0., hspace=0.05)
    plt.show()
    

def get_r_eff_from_kwargs_result(kwargs_result):
    """
    """
    return get_half_light_radius(kwargs_result['kwargs_lens_light'])
