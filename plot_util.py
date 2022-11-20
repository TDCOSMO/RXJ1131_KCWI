import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import corner

from getdist import plots
from getdist import MCSamples

import paperfig as pf
from kinematics_likelihood import KinematicLikelihood
from data_util import *

pf.set_fontscale(2.)


walker_ratio = 6

labels_pl = ['theta_E', 'gamma', 'q', 'pa', 'D_dt', 'inclination', 'lamda', 'ani_param_1', 
          'ani_param_2', 'ani_param_3'] #[:samples_mcmc.shape[1]]
latex_labels_pl = ['{\\theta}_{\\rm E} \ (^{\prime\prime})', 
                '{\\gamma}',
                'q', '{\\rm PA} {\ (^{\circ})}',
                'D_{\\Delta t}^{\prime}\ ({\\rm Mpc})',
                'i {\ (^{\circ})}',
                '{\\lambda_{\\rm MST}}',
                'a_{\\rm ani,1}' , 'a_{\\rm ani,2}', 'a_{\\rm ani,3}' 
               ]

labels_composite = ['kappa_s', 'r_scale', 'M/L', 'q', 'pa', 'D_dt', 'inclination', 'lamda', 'ani_param_1', 
          'ani_param_2', 'ani_param_3'] #[:samples_mcmc.shape[1]]

latex_labels_composite = ['{\\kappa}_{\\rm s}', 
                'r_{\\rm scale}\ (^{\prime\prime})',
                'M/L\ (M_{\\odot}/L_{\\odot})',
                'q', '{\\rm PA} {\ (^{\circ})}',
                'D_{\\Delta t}^{\prime}\ ({\\rm Mpc})',
                'i {\ (^{\circ})}',
                '{\\lambda_{\\rm MST}}',
                'a_{\\rm ani,1}' , 'a_{\\rm ani,2}', 'a_{\\rm ani,3}'
               ]

def get_init_pos(software, aperture_type, anisotropy_model, is_spherical, lens_model_type='powerlaw'):
    """
    """
    likelihood_class = KinematicLikelihood(lens_model_type=lens_model_type,
                                           software=software,
                                           anisotropy_model=anisotropy_model,
                                           aperture=aperture_type,
                                           snr_per_bin=15,
                                           is_spherical=is_spherical,
                                           mpi=False
                                           )

    walker_ratio = 6
    num_steps = 500
    num_param = 8 
    num_walker = num_param * walker_ratio * 1000

    init_lens_params = np.random.multivariate_normal(
        likelihood_class.lens_model_posterior_mean,
        cov=likelihood_class.lens_model_posterior_covariance,
        size=num_walker)

    init_pos = np.concatenate((
        init_lens_params,
        # lambda, ani_param, inclination (deg)
        np.random.normal(loc=[90, 1, 1], scale=[5, 0.05, 0.1],
                         size=(num_walker, 3))
    ), axis=1)
    
    return init_pos


def get_chain(software, aperture_type, anisotropy_model, is_spherical, lens_model_type='powerlaw'):
    """
    Get dynamics chain in right shape.
    """
    samples_mcmc = np.loadtxt(
        '../dynamics_chains/kcwi_dynamics_chain_{}_{}_{}_{}_{}_nl.txt'.format(
            software, aperture_type, anisotropy_model, str(is_spherical), lens_model_type
        )
    )
    
    n_params = samples_mcmc.shape[1]

    n_walkers = walker_ratio * n_params
    n_step = int(samples_mcmc.shape[0] / n_walkers)

    # print('N_step: {}, N_walkers: {}, N_params: {}'.format(n_step, n_walkers, n_params))

    chain = np.empty((n_walkers, n_step, n_params))

    for i in np.arange(n_params):
        samples = samples_mcmc[:, i].T
        chain[:,:,i] = samples.reshape((n_step, n_walkers)).T
       
    return chain


def plot_mcmc_trace(software, aperture_type, anisotropy_model, is_spherical, lens_model_type='powerlaw'):
    """
    """
    chain = get_chain(software, aperture_type, anisotropy_model, is_spherical, lens_model_type)
    
    n_params = chain.shape[2]
    n_walkers = chain.shape[0]
    n_step = chain.shape[1]
    
    mean_pos = np.zeros((n_params, n_step))
    median_pos = np.zeros((n_params, n_step))
    std_pos = np.zeros((n_params, n_step))
    q16_pos = np.zeros((n_params, n_step))
    q84_pos = np.zeros((n_params, n_step))
    
    if lens_model_type == 'powerlaw':
        labels = labels_pl
        latex_labels = latex_labels_pl
    else:
        labels = labels_composite
        latex_labels = latex_labels_composite
        
    for i in np.arange(n_params):
        for j in np.arange(n_step):
            mean_pos[i][j] = np.mean(chain[:, j, i])
            median_pos[i][j] = np.median(chain[:, j, i])
            std_pos[i][j] = np.std(chain[:, j, i])
            q16_pos[i][j] = np.percentile(chain[:, j, i], 16.)
            q84_pos[i][j] = np.percentile(chain[:, j, i], 84.)

    fig, ax = plt.subplots(n_params, sharex=True, figsize=(8, 6))

    burnin = -50
    last = n_step

    medians = []

    param_values = [median_pos[0][last-1], (q84_pos[0][last-1]-q16_pos[0][last-1])/2,
                    median_pos[1][last-1], (q84_pos[1][last-1]-q16_pos[1][last-1])/2]

    for i in range(n_params):
        print(labels[i], '{:.4f} ± {:.4f}'.format(median_pos[i][last-1], (q84_pos[i][last-1]-q16_pos[i][last-1])/2))

        ax[i].plot(median_pos[i][:last], c='g')
        ax[i].axhline(np.median(median_pos[i][burnin:last]), c='r', lw=1)
        ax[i].fill_between(np.arange(last), q84_pos[i][:last], q16_pos[i][:last], alpha=0.4)
        ax[i].set_ylabel(labels[i], fontsize=10)
        ax[i].set_xlim(0, last)

        medians.append(np.median(median_pos[i][burnin:last]))


    fig.set_size_inches((12., 2*n_params))
    plt.show()
    
    
def plot_corner(software, aperture_type, anisotropy_model, is_spherical, lens_model_type='powerlaw',
                fig=None, color='k', burnin=-100, plot_init=False
               ):
    """
    """
    if lens_model_type == 'powerlaw':
        labels = labels_pl
        latex_labels = latex_labels_pl
    else:
        labels = labels_composite
        latex_labels = latex_labels_composite
        
    chain = get_chain(software, aperture_type, anisotropy_model, is_spherical, lens_model_type,
                      burnin=burnin
                     )
    
    
    fig = corner.corner(chain[:, burnin:, :].reshape((-1, chain.shape[-1])),
                        color=color, labels=labels, scale_hist=False, fig=fig,
                       );
    if lens_model_type == 'powerlaw':
        chain[:, 4] = chain[:, 4] / chain[:, 6]
    else:
        chain[:, 5] = chain[:, 5] / chain[:, 7]
    
    if plot_init:
        init_pos = get_init_pos(software, aperture_type, anisotropy_model, is_spherical, lens_model_type)
        
        corner.corner(init_pos, color='k', labels=labels, 
                      scale_hist=False, fig=fig)
    
    return fig


def get_getdist_samples(software, aperture_type, anisotropy_model, is_spherical, lens_model_type='powerlaw',
                        burnin=-100, latex_labels=None, select_indices=None
                       ):
    """
    """
    if lens_model_type == 'powerlaw':
        labels = labels_pl
        if latex_labels is None:
            latex_labels = latex_labels_pl
    else:
        labels = labels_composite
        if latex_labels is None:
            latex_labels = latex_labels_composite
        
    chain = get_chain(software, aperture_type, anisotropy_model, is_spherical, lens_model_type)
    
    chain = chain[:, burnin:, :].reshape((-1, chain.shape[-1]))
    if lens_model_type == 'powerlaw':
        chain[:, 4] = chain[:, 4] / chain[:, 6]
    else:
        chain[:, 5] = chain[:, 5] / chain[:, 7]
    
    if select_indices is None:
        mc_samples= MCSamples(samples=chain, 
                          names=labels[:chain.shape[-1]], 
                          labels=latex_labels[:chain.shape[-1]]
                         )
    else:
        labels = labels[:chain.shape[-1]]
        latex_labels = latex_labels[:chain.shape[-1]]
        print(chain.shape, len(labels))
        mc_samples= MCSamples(samples=chain[:, select_indices], 
                              names=np.array(labels)[select_indices], 
                              labels=np.array(latex_labels)[select_indices]
                             )
    
    return mc_samples


def plot_dist(softwares, aperture_types, anisotropy_models, is_sphericals, lens_model_types,
              burnin=-100, legend_labels=[], save_fig=None, ani_param_latex=None, font_scale=1,
              select_indices=None
             ):
    """
    """
    if 'powerlaw' in lens_model_types:
        labels = labels_pl
        latex_labels = latex_labels_pl
    else:
        labels = labels_composite
        latex_labels = latex_labels_composite
        
    if ani_param_latex is not None:
        for i, a in enumerate(ani_param_latex):
            latex_labels[i-3] = a

    mc_samples_list = []
    
    for s, a, ani, sph, model in zip(softwares, aperture_types, 
                                     anisotropy_models, is_sphericals, lens_model_types):
        mc_samples_list.append(get_getdist_samples(s, a, ani, sph, model, burnin=burnin, 
                                                   latex_labels=latex_labels, select_indices=select_indices))
        
    g = plots.getSubplotPlotter(subplot_size=2.2)
    g.settings.lw_contour = 1.
    g.settings.alpha_factor_contour_lines = 2.
    g.settings.solid_contour_palefactor = 0.5
    g.settings.axes_fontsize = 16 * font_scale
    g.settings.lab_fontsize = 16 * font_scale

    g.settings.legend_fontsize = 18 * font_scale
    # g.settings.smooth_scale_2D = 4
    # g.settings.smooth_scale_1D = 4

    colors = [pf.cb2_blue, pf.cb2_orange, pf.cb2_emerald, pf.cb_grey]

    g.triangle_plot(mc_samples_list,
                    legend_labels=legend_labels,
                    filled=True, shaded=False,
                    alpha_filled_add=.5, 
                    contour_lws=[2 for l in legend_labels], 
                    contour_ls=['-' for l in legend_labels],
                    #filled=False,
                    #contour_colors=[sns.xkcd_rgb['emerald'], sns.xkcd_rgb['bright orange']], 
                    contour_args={'alpha': .5},
                    #line_args={'lw': 7., "zorder": 30},
                    #line_args={'lw': 1., 'alpha': 1.}
                    contour_colors=colors,
#                     param_limits={'dphi_AB': (-0.45, 0.15), 
#                                   'dphi_AC': (-0.45, 0.15), 
#                                   'dphi_AD': (-0.45, 0.15), 
#                                   'lambda_int': (0.3, 3), 'a_ani': (0.5, 5.)},
                   )

    #g.fig.tight_layout()
    if save_fig is not None:
        g.fig.savefig(save_fig, 
                      bbox_inches='tight')


def get_most_likely_value(software, aperture_type, anisotropy_model, is_spherical, 
                          lens_model_type='powerlaw', burnin=-100):
    """
    """
    chain = get_chain(software, aperture_type, anisotropy_model, is_spherical, lens_model_type)
    
    return np.mean(chain[:, burnin:, :].reshape((-1, chain.shape[-1])), 
                   axis=0)

def plot_residual(software, aperture_type, anisotropy_model, is_spherical, 
                  lens_model_type='powerlaw', burnin=-100, verbose=True, 
                  ax=None):
    """
    """
    likelihood_class = KinematicLikelihood(lens_model_type=lens_model_type,
                                           software=software,
                                           anisotropy_model=anisotropy_model,
                                           aperture=aperture_type,
                                           snr_per_bin=15,
                                           is_spherical=is_spherical,
                                           mpi=False
                                           )

    params = get_most_likely_value(software, aperture_type, anisotropy_model, is_spherical, 
                                   lens_model_type, burnin)
    
    v_rms = likelihood_class.get_v_rms(params)

    model_v_rms = get_kinematics_maps(v_rms, likelihood_class.voronoi_bin_mapping)
    data_v_rms = get_kinematics_maps(likelihood_class.velocity_dispersion_mean,
                                     likelihood_class.voronoi_bin_mapping
                                     )
    noise_v_rms = get_kinematics_maps(
        np.sqrt(np.diag(likelihood_class.velocity_dispersion_covariance)),
        likelihood_class.voronoi_bin_mapping
    )
    
    if verbose:
        print('reduced chi^2: {:.2f}'.format(-2 * likelihood_class.get_log_likelihood(params) 
                                             / len(likelihood_class.velocity_dispersion_mean)))

    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(111)
        
    im = ax.matshow((data_v_rms - model_v_rms) / noise_v_rms,
                vmax=3, vmin=-3, cmap='RdBu_r', origin='lower'
                )
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05) 
    plt.colorbar(im, cax=cax, label=r'(data$-$model)/noise')
    ax.set_title('{}, {}, {}'.format(software, anisotropy_model, is_spherical))
    
    return ax, fig

    
def get_bic(software, aperture_type, anisotropy_model, is_spherical, 
            lens_model_type='powerlaw', burnin=-100):
    """
    """
    likelihood_class = KinematicLikelihood(lens_model_type=lens_model_type,
                                           software=software,
                                           anisotropy_model=anisotropy_model,
                                           aperture=aperture_type,
                                           snr_per_bin=15,
                                           is_spherical=is_spherical,
                                           mpi=False
                                           )
    
    params = get_most_likely_value(software, aperture_type, anisotropy_model, is_spherical, 
                                   lens_model_type, burnin)
    np.random.seed(2)
    log_likelihood = likelihood_class.get_log_likelihood(params)
    num_params = len(params)
    num_data = len(likelihood_class.velocity_dispersion_mean)
    
    bic = num_params * np.log(num_data) - 2 * log_likelihood
    
    return bic