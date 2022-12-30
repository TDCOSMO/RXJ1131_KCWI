import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from astropy.cosmology import FlatLambdaCDM
import copy
import corner
import matplotlib.colors as colors
from getdist import plots
from getdist import MCSamples
import seaborn as sns
import palettable

import paperfig as pf
from kinematics_likelihood import KinematicLikelihood
from data_util import *

pf.set_fontscale(2.)

walker_ratio = 12

labels_pl = ['theta_E', 'gamma', 'q', 'D_dt', 'D_d',
             'inclination', 'lamda',
             'ani_param_1',
             'ani_param_2', 'ani_param_3']  # [:samples_mcmc.shape[1]]
latex_labels_pl = ['{\\theta}_{\\rm E} \ (^{\prime\prime})',
                   '{\\gamma}',
                   'q', #'{\\rm PA} {\ (^{\circ})}',
                   'D_{\\Delta t}^{\prime}\ ({\\rm Mpc})',
                   '{\\rm blinded}\ D_{\\rm d}',
                   'i {\ (^{\circ})}',
                   '{\\lambda}',
                   'a_{\\rm ani,1}', 'a_{\\rm ani,2}', 'a_{\\rm ani,3}'
                   ]

labels_composite = ['kappa_s', 'r_scale', 'M/L', 'q', 'D_dt', 'D_d',
                    'inclination', 'lamda', 'ani_param_1',
                    'ani_param_2', 'ani_param_3']  # [:samples_mcmc.shape[1]]

latex_labels_composite = ['{\\kappa}_{\\rm s}',
                          'r_{\\rm scale}\ (^{\prime\prime})',
                          'M/L\ (M_{\\odot}/L_{\\odot})',
                          'q', #'{\\rm PA} {\ (^{\circ})}',
                          'D_{\\Delta t}^{\prime}\ ({\\rm Mpc})',
                          '{\\rm blinded}\ D_{\\rm d}',
                          'i {\ (^{\circ})}',
                          '{\\lambda}',
                          'a_{\\rm ani,1}', 'a_{\\rm ani,2}', 'a_{\\rm ani,3}'
                          ]

cosmo = FlatLambdaCDM(H0=70, Om0=0.3)
Z_L = 0.295  # deflector redshift from Agnello et al. (2018)
Z_S = 0.657
D_d = cosmo.angular_diameter_distance(Z_L).value
D_ds = cosmo.angular_diameter_distance_z1z2(Z_L, Z_S).value
D_s = cosmo.angular_diameter_distance(Z_S).value


def get_init_pos(software, aperture_type, anisotropy_model, is_spherical,
                 lens_model_type='powerlaw', snr=23, shape='oblate'):
    """
    Get initial position of walkers
    :param software: 'jampy' or 'galkin'
    :param aperture_type: 'ifu' or 'single_slit'
    :param anisotropy_model: 'consant', 'step', 'free_step', 'Osipkov-Merritt'
    :param is_spherical: True or False
    :param lens_model_type: 'powerlaw' or 'composite'
    :param snr: signal-to-noise ratio per pixel in Voronoi bins
    :param shape: 'oblate' or 'prolate'
    :return: initial position of walkers
    """
    likelihood_class = KinematicLikelihood(lens_model_type=lens_model_type,
                                           software=software,
                                           anisotropy_model=anisotropy_model,
                                           aperture=aperture_type,
                                           snr_per_bin=snr,
                                           is_spherical=is_spherical,
                                           mpi=False, shape=shape
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


def load_samples_mcmc(software, aperture_type, anisotropy_model, is_spherical,
            lens_model_type='powerlaw', snr=23, shape='oblate'):
    """
    Load MCMC samples from file
    :param software: 'jampy' or 'galkin'
    :param aperture_type: 'ifu' or 'single_slit'
    :param anisotropy_model: 'consant', 'step', 'free_step', 'Osipkov-Merritt'
    :param is_spherical: True or False
    :param lens_model_type: 'powerlaw' or 'composite'
    :param snr: signal-to-noise ratio per pixel in Voronoi bins
    :param shape: 'oblate' or 'prolate'
    :return: MCMC samples
    """
    return np.loadtxt(
        '../dynamics_chains/kcwi_dynamics_chain_{}_{}_{}_{}_{}_{}_{}.txt'.format(
            software, aperture_type, anisotropy_model, str(is_spherical),
            lens_model_type, snr, shape
        )
    )


def load_likelihoods(software, aperture_type, anisotropy_model, is_spherical,
            lens_model_type='powerlaw', snr=23, shape='oblate'):
    """
    Load likelihoods from file
    :param software: 'jampy' or 'galkin'
    :param aperture_type: 'ifu' or 'single_slit'
    :param anisotropy_model: 'consant', 'step', 'free_step', 'Osipkov-Merritt'
    :param is_spherical: True or False
    :param lens_model_type: 'powerlaw' or 'composite'
    :param snr: signal-to-noise ratio per pixel in Voronoi bins
    :param shape: 'oblate' or 'prolate'
    :return: likelihoods
    """
    return np.loadtxt(
        '../dynamics_chains/kcwi_dynamics_chain_{}_{}_{}_{}_{}_{}_{}_logL.txt'.format(
            software, aperture_type, anisotropy_model, str(is_spherical),
            lens_model_type, snr, shape
        )
    )


def get_original_chain(software, aperture_type, anisotropy_model, is_spherical,
              lens_model_type='powerlaw', snr=23, shape='oblate', chain=None):
    """
    Get MCMC chain in original 3D shape as [walker, step, parameter]
    :param software: 'jampy' or 'galkin'
    :param aperture_type: 'ifu' or 'single_slit'
    :param anisotropy_model: 'consant', 'step', 'free_step', 'Osipkov-Merritt'
    :param is_spherical: True or False
    :param lens_model_type: 'powerlaw' or 'composite'
    :param snr: signal-to-noise ratio per pixel in Voronoi bins
    :param shape: 'oblate' or 'prolate'
    :return: original MCMC chain
    """
    if chain is None:
        samples_mcmc = load_samples_mcmc(software, aperture_type,
                                         anisotropy_model, is_spherical,
                                         lens_model_type, snr, shape)
    else:
        samples_mcmc = chain

    n_params = samples_mcmc.shape[1]

    n_walkers = walker_ratio * n_params
    n_step = int(samples_mcmc.shape[0] / n_walkers)

    # print('N_step: {}, N_walkers: {}, N_params: {}'.format(n_step, n_walkers, n_params))

    chain = np.empty((n_walkers, n_step, n_params))

    for i in np.arange(n_params):
        samples = samples_mcmc[:, i].T
        chain[:, :, i] = samples.reshape((n_step, n_walkers)).T

    return chain


def get_chain(software, aperture_type, anisotropy_model, is_spherical,
              lens_model_type='powerlaw', snr=23, shape='oblate', burnin=-100):
    """
    Get MCMC chain in original 2D shape as [..., parameter]
    :param software: 'jampy' or 'galkin'
    :param aperture_type: 'ifu' or 'single_slit'
    :param anisotropy_model: 'consant', 'step', 'free_step', 'Osipkov-Merritt'
    :param is_spherical: True or False
    :param lens_model_type: 'powerlaw' or 'composite'
    :param snr: signal-to-noise ratio per pixel in Voronoi bins
    :param shape: 'oblate' or 'prolate'
    :param burnin: number of steps to discard at the beginning of the chain
    :return: 2d MCMC chain after burnin
    """
    chain = get_original_chain(software, aperture_type, anisotropy_model,
                               is_spherical, lens_model_type, snr=snr,
                               shape=shape)
    chain = chain[:, burnin:, :].reshape((-1, chain.shape[-1]))
    
    return chain


def plot_mcmc_trace_walkers(software, aperture_type, anisotropy_model,
                            is_spherical, lens_model_type='powerlaw',
                            snr=23, shape='oblate', burnin=-50,
                            chain=None
                            ):
    """
    Plot MCMC trace in the emcee example style
    :param software: 'jampy' or 'galkin'
    :param aperture_type: 'ifu' or 'single_slit'
    :param anisotropy_model: 'consant', 'step', 'free_step', 'Osipkov-Merritt'
    :param is_spherical: True or False
    :param lens_model_type: 'powerlaw' or 'composite'
    :param snr: signal-to-noise ratio per pixel in Voronoi bins
    :param shape: 'oblate' or 'prolate'
    :return: None
    """
    if chain is None:
        chain = get_original_chain(software, aperture_type, anisotropy_model,
                                   is_spherical, lens_model_type, snr, shape)

    n_params = chain.shape[2]
    n_walkers = chain.shape[0]
    n_step = chain.shape[1]
    fig, axes = plt.subplots(n_params, figsize=(10, 7), sharex=True)

    if lens_model_type == 'powerlaw':
        labels = labels_pl
        latex_labels = latex_labels_pl
    else:
        labels = labels_composite
        latex_labels = latex_labels_composite

    for i in range(n_params):
        ax = axes[i]
        print(chain[:, :, i].shape)
        ax.plot(chain[:, :, i].T, "k", alpha=0.02)
        ax.set_xlim(0, n_step)
        ax.set_ylabel(labels[i])
        ax.yaxis.set_label_coords(-0.1, 0.5)

    axes[-1].set_xlabel("step number");
    plt.show()


def plot_mcmc_trace(software, aperture_type, anisotropy_model, is_spherical,
                    lens_model_type='powerlaw', snr=23, shape='oblate',
                    burnin=-50, chain=None):
    """
    Plot MCMC trace
    :param software: 'jampy' or 'galkin'
    :param aperture_type: 'ifu' or 'single_slit'
    :param anisotropy_model: 'consant', 'step', 'free_step', 'Osipkov-Merritt'
    :param is_spherical: True or False
    :param lens_model_type: 'powerlaw' or 'composite'
    :param snr: signal-to-noise ratio per pixel in Voronoi bins
    :param shape: 'oblate' or 'prolate'
    :return: None
    """
    if chain is None:
        chain = get_original_chain(software, aperture_type, anisotropy_model,
                                   is_spherical, lens_model_type, snr, shape)

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

    last = n_step

    medians = []

    param_values = [median_pos[0][last - 1],
                    (q84_pos[0][last - 1] - q16_pos[0][last - 1]) / 2,
                    median_pos[1][last - 1],
                    (q84_pos[1][last - 1] - q16_pos[1][last - 1]) / 2]

    for i in range(n_params):
        print(labels[i], '{:.4f} ± {:.4f}'.format(median_pos[i][last - 1], (
                    q84_pos[i][last - 1] - q16_pos[i][last - 1]) / 2))

        ax[i].plot(median_pos[i][:last], c='g')
        ax[i].axhline(np.median(median_pos[i][burnin:last]), c='r', lw=1)
        ax[i].fill_between(np.arange(last), q84_pos[i][:last],
                           q16_pos[i][:last], alpha=0.4)
        ax[i].set_ylabel(labels[i], fontsize=10)
        ax[i].set_xlim(0, last)

        medians.append(np.median(median_pos[i][burnin:last]))

    fig.set_size_inches((12., 2 * n_params))
    plt.show()


def plot_corner(software, aperture_type, anisotropy_model, is_spherical,
                lens_model_type='powerlaw', snr=23, shape='oblate',
                fig=None, color='k', burnin=-100, plot_init=False
                ):
    """
    Plot corner plot
    :param software: 'jampy' or 'galkin'
    :param aperture_type: 'ifu' or 'single_slit'
    :param anisotropy_model: 'consant', 'step', 'free_step', 'Osipkov-Merritt'
    :param is_spherical: True or False
    :param lens_model_type: 'powerlaw' or 'composite'
    :param snr: signal-to-noise ratio per pixel in Voronoi bins
    :param shape: 'oblate' or 'prolate'
    :param fig: figure to plot on
    :param color: color of the contours
    :param burnin: number of steps to discard at the beginning of the chain
    :param plot_init: if True, plot initial position of the walkers
    :return: None
    """
    if lens_model_type == 'powerlaw':
        labels = labels_pl
        latex_labels = latex_labels_pl
    else:
        labels = labels_composite
        latex_labels = latex_labels_composite

    chain = get_chain(software, aperture_type, anisotropy_model, is_spherical,
                      lens_model_type, snr, shape, burnin=burnin
                      )

    fig = corner.corner(chain, color=color, labels=labels, scale_hist=False,
                        fig=fig);

    # if lens_model_type == 'powerlaw':
    #     chain[:, 4] = chain[:, 4] / chain[:, 6]
    # else:
    #     chain[:, 5] = chain[:, 5] / chain[:, 7]

    if plot_init:
        init_pos = get_init_pos(software, aperture_type, anisotropy_model,
                                is_spherical, lens_model_type, snr, shape)

        corner.corner(init_pos, color='k', labels=labels,
                      scale_hist=False, fig=fig)

    return fig


def get_getdist_samples(software, aperture_type, anisotropy_model,
                        is_spherical, lens_model_type='powerlaw', snr=23,
                        shape='oblate',
                        oblate_fraction=None,
                        burnin=-100, latex_labels=None, select_indices=None,
                        blind_D=True, 
                        ):
    """
    Get samples from the chain in getdist format
    :param software: 'jampy' or 'galkin'
    :param aperture_type: 'ifu' or 'single_slit'
    :param anisotropy_model: 'consant', 'step', 'free_step', 'Osipkov-Merritt'
    :param is_spherical: True or False
    :param lens_model_type: 'powerlaw' or 'composite'
    :param snr: signal-to-noise ratio per pixel in Voronoi bins
    :param shape: 'oblate' or 'prolate'
    :param oblate_fraction: probability of oblateness
    :param burnin: number of steps to discard at the beginning of the chain
    :param latex_labels: list of latex labels
    :param select_indices: list of indices to select
    :param blind_D: if True, blind the distance parameter
    :return: getdist MCSamples object
    """
    if lens_model_type == 'powerlaw':
        labels = labels_pl
        if latex_labels is None:
            latex_labels = latex_labels_pl
    else:
        labels = labels_composite
        if latex_labels is None:
            latex_labels = latex_labels_composite

    if oblate_fraction is not None:
        chain_obl = get_chain(software, aperture_type, anisotropy_model, is_spherical,
                      lens_model_type, snr, 'oblate', burnin=burnin)
#         chain_obl = chain_obl[:, burnin:, :].reshape((-1, chain_obl.shape[-1]))
        chain_pro = get_chain(software, aperture_type, anisotropy_model, is_spherical,
                      lens_model_type, snr, 'prolate', burnin=burnin)
#         chain_pro = chain_pro[:, burnin:, :].reshape((-1, chain_pro.shape[-1]))

        indices = np.arange(chain_pro.shape[0])
        chain = np.concatenate(
            (chain_obl[:int(oblate_fraction * len(indices)), :],
             chain_pro[:int((1-oblate_fraction) * len(indices)), :]
            ),
            axis=0
        )
        print(chain_obl.shape, chain_pro.shape, chain.shape, oblate_fraction)
    else:
        chain = get_chain(software, aperture_type, anisotropy_model, is_spherical,
                          lens_model_type, snr, shape, burnin=burnin)

#         chain = chain[:, burnin:, :].reshape((-1, chain.shape[-1]))

    if lens_model_type == 'powerlaw':
        d_index = 4
    else:
        d_index = 5

    chain[:, d_index+2] = chain[:, d_index+2] * D_s / D_ds
    chain[:, d_index] = chain[:, d_index] / chain[:, d_index+2] / (1. + Z_L)

    if blind_D is None:
        mean_D = np.median(chain[:, d_index])
    else:
        if blind_D is True:
            mean_D = np.median(chain[:, d_index])
        else:
            mean_D = blind_D
        
        chain[:, d_index] -= mean_D
        chain[:, d_index] /= mean_D
        
    low, hi = np.percentile(chain[:, d_index], q=[16, 84])
    std_D = (hi - low) / 2.

    if select_indices is None:
        mc_samples = MCSamples(samples=chain,
                               names=labels[:chain.shape[-1]],
                               labels=latex_labels[:chain.shape[-1]]
                               )
    else:
        labels = labels[:chain.shape[-1]]
        latex_labels = latex_labels[:chain.shape[-1]]
        print(chain.shape, len(labels))
        mc_samples = MCSamples(samples=chain[:, select_indices],
                               names=np.array(labels)[select_indices],
                               labels=np.array(latex_labels)[select_indices]
                               )

#     if blind_D is True:
    return mc_samples, np.median(chain[:, d_index]), std_D, mean_D
#     else:
#         return mc_samples, np.median(chain[:, d_index]), std_D, 


def plot_dist(softwares, aperture_types, anisotropy_models, is_sphericals,
              lens_model_types, snrs=None, shapes=None,
              oblate_fractions=None,
              burnin=-100, legend_labels=[], save_fig=None,
              ani_param_latex=None, font_scale=1,
              select_indices=None, blind=True, print_difference=False
              ):
    """
    Plot the posterior distributions of the parameters using getdist
    :param softwares: list of 'jampy' or 'galkin'
    :param aperture_types: list of 'ifu' or 'single_slit'
    :param anisotropy_models: list of 'consant', 'step', 'free_step', 'Osipkov-Merritt'
    :param is_sphericals: list of True or False
    :param lens_model_types: list of 'powerlaw' or 'composite'
    :param snrs: list of signal-to-noise ratio per pixel in Voronoi bins
    :param shapes: list of 'oblate' or 'prolate'
    :param oblate_fractions: list of probability of oblateness
    :param burnin: number of steps to discard at the beginning of the chain
    :param legend_labels: list of legend labels
    :param save_fig: if not None, save the figure to this file
    :param ani_param_latex: list of latex labels for the anisotropy parameters
    :param font_scale: font scale
    :param select_indices: list of indices to select for plotting, if None plot all
    :param blind: if True, blind the distance parameter
    :param print_difference: if True, print the difference between the first and every other case
    :return: None
    """
    if 'powerlaw' in lens_model_types:
        labels = labels_pl
        latex_labels = latex_labels_pl
    else:
        labels = labels_composite
        latex_labels = latex_labels_composite

    if ani_param_latex is not None:
        for i, a in enumerate(ani_param_latex):
            latex_labels[i - 3] = a
            
    if snrs is None:
        snrs = [23] * len(softwares)
      
    if shapes is None:
        shapes = ['oblate'] * len(softwares)

    if oblate_fractions is None:
        oblate_fractions = [None] * len(softwares)

    mc_samples_list = []

    first = True
    for i, (s, a, ani, sph, model, snr, shape, f_obl) in enumerate(zip(softwares,
                                                      aperture_types,
                                     anisotropy_models, is_sphericals,
                                     lens_model_types, snrs, shapes, oblate_fractions)):
        if first:
            mc_samples, mean_D_first, std_D_first, mean_D_real = get_getdist_samples(s, a, ani, sph, model, snr, shape,
                                                   burnin=burnin,
                                                   oblate_fraction=f_obl,
                                                   latex_labels=latex_labels,
                                                   select_indices=None if
                                                   select_indices is None else
                                                   select_indices[i],
                                                   blind_D=blind
                                                   )
            first = False
            print('D uncertainty {:.2f}'.format(std_D_first))
        else:
            mc_samples, mean_D, std_D, _ = get_getdist_samples(s, a, ani, sph, model, snr, shape,
                                                burnin=burnin,
                                                oblate_fraction=f_obl,
                                                latex_labels=latex_labels,
                                                select_indices=None if
                                                select_indices is None else
                                                select_indices[i],
                                                blind_D=mean_D_real if blind is not None else None
                                                )
            if print_difference:
                if blind is None:
                    print('Difference: {:.2f}%, {:.2f} sigma, D uncertainty: {:.2f}'.format((mean_D_first - mean_D)/mean_D_first * 100,
                                                                 (mean_D_first - mean_D)/np.sqrt(std_D_first**2 + std_D**2), std_D))
                else:
                    print('Difference: {:.2f}%, {:.2f} sigma, D uncertainty: {:.2f}'.format((mean_D_first - mean_D) * 100,
                                                                 (mean_D_first - mean_D)/np.sqrt(std_D_first**2 + std_D**2), std_D))
            
        mc_samples_list.append(mc_samples)

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
                    # filled=False,
                    # contour_colors=[sns.xkcd_rgb['emerald'], sns.xkcd_rgb['bright orange']],
                    contour_args={'alpha': .5},
                    # line_args={'lw': 7., "zorder": 30},
                    # line_args={'lw': 1., 'alpha': 1.}
                    contour_colors=colors,
                    #                     param_limits={'dphi_AB': (-0.45, 0.15),
                    #                                   'dphi_AC': (-0.45, 0.15),
                    #                                   'dphi_AD': (-0.45, 0.15),
                    #                                   'lambda_int': (0.3, 3), 'a_ani': (0.5, 5.)},
                    )

    # g.fig.tight_layout()
    if save_fig is not None:
        g.fig.savefig(save_fig,
                      bbox_inches='tight')
        

def get_most_likely_value(software, aperture_type, anisotropy_model,
                          is_spherical,
                          lens_model_type='powerlaw', snr=23,
                          shape='oblate', burnin=-100):
    """
    Get the most likely value of the distance parameter
    :param software: software to use, 'jampy' or 'galkin'
    :param aperture_type: aperture type, 'ifu' or 'single_slit'
    :param anisotropy_model: anisotropy model
    :param is_spherical: if True, use spherical model
    :param lens_model_type: lens model type, 'powerlaw' or 'composite'
    :param snr: signal-to-noise ratio per A per Voronoi bins
    :param shape: shape of the galaxy axisymmetry, 'oblate' or 'prolate'
    :param burnin: number of burn-in steps
    """
    chain = load_samples_mcmc(software, aperture_type, anisotropy_model,
                              is_spherical, lens_model_type, snr, shape)

    likelihoods = load_likelihoods(software, aperture_type, anisotropy_model,
                                   is_spherical, lens_model_type, snr, shape)

    print(chain.shape, likelihoods.shape)
    print(np.argmax(likelihoods))
    return chain[np.argmax(likelihoods), :]


def plot_residual(software, aperture_type, anisotropy_model, sphericity,
                  lens_model_type='powerlaw', snr=23, shape='oblate',
                  burnin=-100, verbose=True, cmap='viridis',
                  #vmax=350,
                  # vmin=100,
                  norm=None
                  ):
    """
    Plot the residual between the observed and the predicted velocity dispersion
    :param software: software to use, 'jampy' or 'galkin'
    :param aperture_type: aperture type, 'ifu' or 'single_slit'
    :param anisotropy_model: anisotropy model
    :param sphericity: if True, use spherical model
    :param lens_model_type: lens model type, 'powerlaw' or 'composite'
    :param snr: signal-to-noise ratio per AA per Voronoi bins
    :param shape: shape of the galaxy axisymmetry, 'oblate' or 'prolate'
    :param burnin: number of burn-in steps
    :param verbose: if True, print the chi^2
    :param cmap: colormap
    #:param vmax: maximum value of the colorbar
    #:param vmin: minimum value of the colorbar
    :param norm: normalization of the colorbar
    :return: the figure
    """
    if sphericity == 'spherical':
        is_spherical = True
    elif sphericity == 'axisymmetric':
        is_spherical = False
    likelihood_class = KinematicLikelihood(lens_model_type=lens_model_type,
                                           software=software,
                                           anisotropy_model=anisotropy_model,
                                           aperture=aperture_type,
                                           snr_per_bin=snr,
                                           is_spherical=is_spherical,
                                           mpi=False, shape=shape
                                           )

    params = get_most_likely_value(software, aperture_type, anisotropy_model,
                                   sphericity,
                                   lens_model_type, snr, shape, burnin)
    
    v_rms = likelihood_class.get_v_rms(params)

    model_v_rms = get_kinematics_maps(v_rms,
                                      likelihood_class.voronoi_bin_mapping)
    data_v_rms = get_kinematics_maps(likelihood_class.velocity_dispersion_mean,
                                     likelihood_class.voronoi_bin_mapping
                                     )

    noise_v_rms = get_kinematics_maps(
        np.sqrt(np.diag(likelihood_class.velocity_dispersion_covariance)),
        likelihood_class.voronoi_bin_mapping
    )

    if verbose:
        print('reduced chi^2: {:.2f}'.format(
            -2 * likelihood_class.get_log_likelihood(params)
            / len(likelihood_class.velocity_dispersion_mean)))
    
    axes = []
    
#     vmax, vmin = 350, 100
#     if ax is None:
    likelihood_class.voronoi_bin_mapping = None
    v_rms_map = likelihood_class.get_v_rms(params)
    
    v_rms_map[data_v_rms == 0] = np.nan
    data_v_rms[data_v_rms == 0] = np.nan
    
    fig = plt.figure(figsize=(15, 5))
    ax = fig.add_subplot(131)
    
    im = ax.imshow(data_v_rms[9:36, 11:38],
                    #vmax=vmax, vmin=vmin, 
                      cmap=cmap, origin='lower',
                      norm=norm)
    
    ax.set_aspect('equal')
    
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(im, cax=cax, label=r'$\sigma_{\rm los}$ (km s$^{-1}$)', ticks=[350, 300, 250, 200, 0])
    ax.set_title('Data')
    ax.set_xticks([])
    ax.set_yticks([])
    axes.append(ax)
    
    ax = fig.add_subplot(132)
    #im = ax.matshow(v_rms_map[9:36, 11:38],
    im = ax.imshow(v_rms_map[9:36, 11:38],
                    #vmax=vmax, vmin=vmin, 
                      cmap=cmap, origin='lower',
                      norm=norm)
    ax.set_aspect('equal')
    
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(im, cax=cax, label=r'$\sigma_{\rm los}$ (km s$^{-1}$)', ticks=[350, 300, 250, 200, 0])
    ax.set_title('Model')
    ax.set_xticks([])
    ax.set_yticks([])                 
    axes.append(ax)                 
    
    ax = fig.add_subplot(133)
    im = ax.matshow(((data_v_rms - model_v_rms) / noise_v_rms)[9:36, 11:38],
                    vmax=3, vmin=-3, cmap='RdBu_r', origin='lower'
                    )
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(im, cax=cax, label=r'(data$-$model)/noise')
    ax.set_title('Residual')
    ax.set_xticks([])
    ax.set_yticks([])    
    axes.append(ax)
                     
    return fig, axes


def get_bic(software, aperture_type, anisotropy_model, sphericity,
            lens_model_type='powerlaw', snr=23, shape='oblate', burnin=-100):
    """
    Get the Bayesian information criterion
    :param software: software to use, 'jampy' or 'galkin'
    :param aperture_type: aperture type, 'ifu' or 'single_slit'
    :param anisotropy_model: anisotropy model
    :param sphericity: if True, 'spherical' or 'axisymmetric'
    :param lens_model_type: lens model type, 'powerlaw' or 'composite'
    :param snr: signal-to-noise ratio per AA per Voronoi bins
    :param shape: shape of the galaxy axisymmetry, 'oblate' or 'prolate'
    :param burnin: number of burn-in steps
    :return: the BIC
    """
    if sphericity == 'spherical':
        is_spherical = True
    elif sphericity == 'axisymmetric':
        is_spherical = False
    likelihood_class = KinematicLikelihood(lens_model_type=lens_model_type,
                                           software=software,
                                           anisotropy_model=anisotropy_model,
                                           aperture=aperture_type,
                                           snr_per_bin=snr,
                                           is_spherical=is_spherical,
                                           mpi=False, shape=shape
                                           )

    params = get_most_likely_value(software, aperture_type, anisotropy_model,
                                   sphericity,
                                   lens_model_type, snr, shape, burnin)
    np.random.seed(2)
    log_likelihood = likelihood_class.get_log_likelihood(params)
    num_params = len(params)
    num_data = len(likelihood_class.velocity_dispersion_mean)

    bic = num_params * np.log(num_data) - 2 * log_likelihood

    return bic