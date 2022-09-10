import astropy.io.fits as fits
import matplotlib.pyplot as plt
import numpy as np
from astropy.cosmology import FlatLambdaCDM
from pafit.fit_kinematic_pa import fit_kinematic_pa
from getdist import plots
from getdist import MCSamples
import seaborn as sns
from output_class import ModelOutput
import paperfig


paperfig.set_fontscale(2.)


class PostProcessKinematics(object):
    """

    """
    def __init__(self):
        """

        """
        self.bin_mapping = None

        self.dynamical_model_ifu = ModelOutput('../lens_model_chain/'
                        'multivariate_gaussian_resampled_chain.txt', cgd=True)
        self.dynamical_model_aperture = ModelOutput('../lens_model_chain/'
                        'multivariate_gaussian_resampled_chain.txt', cgd=True)


        fiducial_cosmo = FlatLambdaCDM(H0=70, Om0=0.3)

        z_s = self.dynamical_model_ifu.Z_S
        z_l = self.dynamical_model_ifu.Z_L

        self.fiducial_Dsds = fiducial_cosmo.angular_diameter_distance(
            z_s).value / fiducial_cosmo.angular_diameter_distance_z1z2(z_l,
                                                                    z_s).value

    def load_bin_mapping(self, url, plot=False):
        """

        """
        bins = np.loadtxt(url)
        # bins -= 1 # unbinned pixels set to -1

        self.bin_mapping = np.zeros((43, 43))

        for a in bins:
            self.bin_mapping[int(a[1])][int(a[0])] = int(a[2]) + 1

        self.bin_mapping -= 1

        self.bin_mapping[self.bin_mapping < 0] = np.nan

        if plot:
            cbar = plt.matshow(self.bin_mapping, cmap='turbo', origin='lower')
            plt.colorbar(cbar)
            plt.title('Bin mapping')
            plt.show()

        return self.bin_mapping

    def get_kinematics_maps(self, directory, name, radius_in_pixels,
                           vd_name=None,
                           vd_val=None, plot=False):
        """
        Remap the kinematics measurements above into 2D array
        :return: 2D velocity dispersion, uncertainty of the velocity
        dispersion, velocity, and the uncertainty of the velocity.
        """
        if vd_name == None:
            measurements = np.loadtxt(directory + 'VD.txt')
        else:
            measurements = np.loadtxt(directory + 'VD_%s.txt' % vd_name)

        # Vel, sigma, dv, dsigma
        output = np.loadtxt(
            directory + 'voronoi_2d_binning_' + name + '_output.txt')

        VD_array = np.zeros(output.shape[0])
        noise_array = np.zeros(output.shape[0])
        V_array = np.zeros(output.shape[0])
        dv_array = np.zeros(output.shape[0])

        for i in range(output.shape[0]):
            num = int(output.T[2][i])
            if vd_val is not None and measurements[num][1] > vd_val:
                results = np.nan
            else:
                results = measurements[num][1]
            sigma = measurements[num][3]
            v = measurements[num][0]
            dv = measurements[num][2]

            VD_array[i] = results
            noise_array[i] = sigma
            V_array[i] = v
            dv_array[i] = dv

        final = np.vstack((output.T, VD_array, noise_array, V_array, dv_array))

        dim = radius_in_pixels * 2 + 1

        VD_2d = np.zeros((dim, dim))
        VD_2d[:] = np.nan
        for i in range(final.shape[1]):
            VD_2d[int(final[1][i])][int(final[0][i])] = final[3][i]

        sigma_2d = np.zeros((dim, dim))
        sigma_2d[:] = np.nan
        for i in range(final.shape[1]):
            sigma_2d[int(final[1][i])][int(final[0][i])] = final[4][i]

        V_2d = np.zeros((dim, dim))
        V_2d[:] = np.nan
        for i in range(final.shape[1]):
            V_2d[int(final[1][i])][int(final[0][i])] = final[5][i]

        dv_2d = np.zeros((dim, dim))
        dv_2d[:] = np.nan
        for i in range(final.shape[1]):
            dv_2d[int(final[1][i])][int(final[0][i])] = final[6][i]

        if plot:
            fig = self.plot_kinematics_maps(VD_2d, sigma_2d, V_2d, dv_2d)
            return VD_2d, sigma_2d, V_2d, dv_2d, fig
        else:
            return VD_2d, sigma_2d, V_2d, dv_2d

    def plot_kinematics_maps(self, VD_2d, dVD_2d, V_2d,
                                              dV_2d):
        """
        """
        fig = plt.figure(figsize=(12, 4))
        ax = fig.add_subplot(121)
        im = ax.imshow(VD_2d, origin='lower', cmap='viridis')
        cbar = plt.colorbar(im)
        cbar.set_label(r'$\sigma$ [km/s]')

        ax = fig.add_subplot(122)
        im = ax.imshow(dVD_2d, origin='lower', cmap='viridis')
        cbar = plt.colorbar(im)
        cbar.set_label(r'$\sigma_{\sigma}$ [km/s]')
        plt.show()

        fig = plt.figure(figsize=(12, 4))
        ax = fig.add_subplot(121)
        im = plt.imshow(V_2d, origin='lower', cmap='RdBu_r')
        cbar = plt.colorbar(im)
        cbar.set_label(r'$v$ [km/s]')

        ax = fig.add_subplot(122)
        im = ax.imshow(dV_2d, origin='lower', cmap='viridis')
        cbar = plt.colorbar(im)
        cbar.set_label(r'$\sigma_{v}$ [km/s]')

        return fig

    def get_v_rms(self, directory, name, radius_in_pixels, vd_name=None,
                  vd_val=None, object_name="KCWI_RXJ1131_icubes_mosaic_0.1457",
                  plot=False, subtract_v_mean=True):
        """
        """
        x_grid, y_grid = np.meshgrid(
            np.arange(-3.0597, 3.1597, 0.1457),  # x-axis points to negative RA
            np.arange(-3.0597, 3.1597, 0.1457),
        )

        x_center = 0  # 0.35490234894050443 # fitted from the KCWI cube
        y_center = 2 * 0.1457  # 0.16776792706671506 # fitted from the KCWI cube

        x_grid -= x_center
        y_grid -= y_center

        VD_2d, dVD_2d, V_2d, dV_2d = self.get_kinematics_maps(directory,
                                                        object_name + '_targetSN_15',
                                                        radius_in_pixels=21,
                                                        vd_name=vd_name,
                                                        # 'targetSN_15_excluding_bins',
                                                        vd_val=1000)

        V_2d_flat = V_2d.flatten()
        x_grid_flat = x_grid.flatten()[~np.isnan(V_2d_flat)]
        y_grid_flat = y_grid.flatten()[~np.isnan(V_2d_flat)]
        V_2d_flat = V_2d_flat[~np.isnan(V_2d_flat)]

        if subtract_v_mean:
            v_mean = fit_kinematic_pa(x_grid_flat, y_grid_flat, V_2d_flat,
                                      quiet=~plot, plot=plot)[2]

            v_rms = np.sqrt(VD_2d ** 2 + (V_2d - v_mean) ** 2)

            sigma_rms = np.sqrt((VD_2d / v_rms) ** 2 * dVD_2d ** 2 + (
                        (V_2d - v_mean) / v_rms) ** 2 * dV_2d ** 2)
        else:
            v_rms = VD_2d
            sigma_rms = dVD_2d

        if plot:
            fig = plt.figure(figsize=(12, 4))

            ax = fig.add_subplot(131)
            im = ax.imshow(v_rms, origin='lower', cmap='viridis')
            cbar = plt.colorbar(im)
            cbar.set_label(r'$v_{\rm rms}$ [km/s]')

            ax = fig.add_subplot(132)
            im = ax.imshow(sigma_rms, origin='lower', cmap='viridis')
            cbar = plt.colorbar(im)
            cbar.set_label(r'$\sigma_{v, \rm rms}$ [km/s]')

            ax = fig.add_subplot(133)
            im = ax.imshow(V_2d - v_mean, origin='lower', cmap='RdBu')
            cbar = plt.colorbar(im)
            cbar.set_label(r'$v - v_{\rm mean}$ [km/s]')

        if plot:
            return v_rms, sigma_rms, fig
        else:
            return v_rms, sigma_rms

    def get_binned_v_rms_measurements_and_uncertainty(self,
                  directory, name, radius_in_pixels, vd_name=None,
                  vd_val=None, object_name="KCWI_RXJ1131_icubes_mosaic_0.1457",
                  plot=False):
        """

        """
        v_rms, sigma_rms = self.get_v_rms(directory, name,
                                    radius_in_pixels=radius_in_pixels,
                                    vd_name=vd_name,
                                    vd_val=vd_val,
                                    object_name=object_name,
                                    plot=False)

        bin_mapping = self.bin_mapping
        max_bin_num = int(np.nanmax(bin_mapping))
        vel_dis_measured = np.zeros(max_bin_num + 1)
        vel_dis_sigma = np.zeros(max_bin_num + 1)

        for i in range(len(vel_dis_measured)):
            vel_dis_measured[i] = np.mean(
                v_rms[bin_mapping == i])  # np.mean(VD_2d[binning == i+1])
            vel_dis_sigma[i] = np.mean(
                sigma_rms[bin_mapping == i])  # np.mean(dVD_2d[binning == i+1])

        # effectively masking out velocity dispersion bins where the error is zero
        vel_dis_sigma[vel_dis_sigma == 0.] = 1e10

        return vel_dis_measured, vel_dis_sigma

    def load_ifu_dynamical_models(self, software, anisotropy_model,
                                  ellipticity_model, directory,
                                  compute_chunk=500, num_samples=None
                                  ):
        """
        """
        if num_samples is None:
            num_samples = len(self.dynamical_model_ifu.samples_mcmc)

        dir_suffix = 'out.txt'

        self.dynamical_model_ifu.load_velocity_dispersion(
            '{}/{}_ifu_{}_{}_vd_'.format(directory, software,
                                         anisotropy_model,
                                         ellipticity_model),
            dir_suffix,
            compute_chunk,
            total_samples=num_samples)

        self.dynamical_model_ifu.load_ani_param(
            '{}/{}_ifu_{}_{}_ani_param_'.format(directory, software,
                                         anisotropy_model,
                                         ellipticity_model),
            dir_suffix,
            compute_chunk,
            total_samples=num_samples)

        self.dynamical_model_ifu.load_r_eff(
            '{}/{}_ifu_{}_{}_reff_'.format(directory, software,
                                                anisotropy_model,
                                                ellipticity_model),
            dir_suffix,
            compute_chunk,
            total_samples=num_samples)

        self.dynamical_model_ifu.load_intensity_map(
            '{}/{}_ifu_{}_{}_ir_'.format(directory, software,
                                           anisotropy_model,
                                           ellipticity_model),
            dir_suffix,
            compute_chunk,
            total_samples=num_samples)

        self.dynamical_model_ifu.bin_all_maps(self.bin_mapping.flatten())

    def load_aperture_dynamical_model(self,
                                      anisotropy_model,
                                      ellipticity_model,
                                      compute_chunk=1000,
                                      num_samples=None):
        """
        """
        dir_suffix = 'out.txt'
        if num_samples is None:
            num_samples = len(self.dynamical_model_aperture.samples_mcmc)

        self.dynamical_model_aperture.load_velocity_dispersion(
            '../galkin_models/galkin_aperture_{}_{}_vd_'.format(
                anisotropy_model, ellipticity_model),
            dir_suffix,
            compute_chunk,
            total_samples=num_samples)
        self.dynamical_model_aperture.load_ani_param(
            '../galkin_models/galkin_aperture_{}_{}_ani_param_'.format(
                anisotropy_model, ellipticity_model), dir_suffix,
            compute_chunk,
            total_samples=num_samples)
        self.dynamical_model_aperture.load_r_eff(
            '../galkin_models/galkin_aperture_{}_{}_reff_'.format(
                anisotropy_model, ellipticity_model), dir_suffix,
            compute_chunk,
            total_samples=num_samples)

    def binned_to_map(self, bins, voronoi_binning, size=43):
        """
        """
        unbinned_map = np.zeros((size, size))

        for i in range(size):
            for j in range(size):
                if np.isnan(voronoi_binning[i, j]):
                    continue
                else:
                    unbinned_map[i, j] = bins[int(voronoi_binning[i, j] - 1)]

        return unbinned_map

    def get_lambda_mst_posterior(self, v_rms_measured=None,
                                 v_rms_sigma=None, is_ifu=True, plot=False):
        """

        """
        data = v_rms_measured
        sigma = v_rms_sigma
        # sigma_matrix = np.diag(1 / v_rms_sigma ** 2)

        # lambda_int_prior = np.sqrt(np.random.uniform(0.5, 1.5, size=10000))

        # for i in tnrange(dyn_model.model_velocity_dispersion.shape[0]):
        if is_ifu:
            model = self.dynamical_model_ifu.binned_velocity_dispersion  # [i]

            mean_lambda_mst_sqrt = np.nansum(
                data[np.newaxis, :] * model / sigma[np.newaxis, :] ** 2,
                axis=1) / np.nansum(model * model / sigma[np.newaxis, :] ** 2,
                                    axis=1)
            # sigma_lambda_mst_sqrt = 1. / np.nansum(
            #     model * model / sigma[np.newaxis, :] ** 2, axis=1)
            #
            # lambda_mst_sqrt = np.random.normal(loc=mean_lambda_mst_sqrts,
            #                                scale=sigma_lambda_mst_sqrt)
            lambda_mst_sqrt = mean_lambda_mst_sqrt
            lambda_mst = lambda_mst_sqrt**2 # lambda_int_sqrt ** 2
            velocity_dispersion_likelihood = -0.5 * np.nansum(
                (model * lambda_mst_sqrt[:, np.newaxis] - data[np.newaxis,
                                                       :]) ** 2 / sigma[
                                                                  np.newaxis,
                                                                  :] ** 2,
                axis=1
            )
        else:
            data = 323 # km/s
            sigma = 20 # km/s

            model = self.dynamical_model_aperture.model_velocity_dispersion

            lambda_mst = (data / model)**2
            velocity_dispersion_likelihood = -0.5 * (model * np.sqrt(
                lambda_mst) - data)**2 / sigma**2

        velocity_dispersion_likelihood -= np.nanmax(velocity_dispersion_likelihood)
        velocity_dispersion_likelihood = np.exp(velocity_dispersion_likelihood)

        if plot:
            fig = plt.figure()
            ax = fig.add_subplot(111)
            sns.kdeplot(lambda_mst, label='unweighted', ax=ax)
            sns.kdeplot(lambda_mst, weights=velocity_dispersion_likelihood,
                        label='likelihood weighted', ax=ax)
            ax.set_xlabel(r'$\lambda_{\rm MST}$')

            mean = np.average(lambda_mst,
                               weights=velocity_dispersion_likelihood)
            std = np.sqrt(np.cov(lambda_mst,
                                 aweights=velocity_dispersion_likelihood))
            if is_ifu:
                ax.set_title(r'With IFU: {:.2f}$\pm${:.2f}'.format(mean, std))
            else:
                ax.set_title(r'With singer aperture: {:.2f}$\pm${'
                             r':.2f}'.format(mean, std))
            plt.legend()

        return lambda_mst, velocity_dispersion_likelihood

    def plot_v_rms_residual(self, lambda_mst,
                            velocity_dispersion_likelihood,
                            v_rms_measured,
                            v_rms_sigma,
                            model_index=None):
        """
        """
        if model_index is None:
            model_index = np.argmax(velocity_dispersion_likelihood)

        v_rms_model = self.dynamical_model_ifu.binned_velocity_dispersion[model_index]

        lamda = lambda_mst[model_index]

        binned_residual = self.binned_to_map(
            (v_rms_measured - np.sqrt(lamda) * v_rms_model) / v_rms_sigma,
            self.bin_mapping)
        binned_residual[self.bin_mapping == np.nan] = np.nan

        im = plt.matshow(binned_residual,
                         cmap='RdBu_r', origin='lower',
                         vmax=3, vmin=-3)
        plt.colorbar(im)
        plt.show()

    def plot_corner_plot(self, lambda_mst, velocity_dispersion_likelihood,
                         lambda_mst_aperture,
                         velocity_dispersion_likelihood_aperture
                         ):
        """

        """
        num_total_samples = len(self.dynamical_model_ifu.samples_mcmc)
        dyn_model_ifu = self.dynamical_model_ifu
        dyn_model_aperture = self.dynamical_model_aperture

        all_pl_samples = np.array([
            dyn_model_ifu.samples_mcmc[:num_total_samples, 1],
            dyn_model_ifu.samples_mcmc[:num_total_samples, 2],
            # lambda_mst*np.nan,
            #np.sqrt(np.random.uniform(0.5, 1.5, size=num_total_samples)) ** 2,
            dyn_model_ifu.ani_param,
            dyn_model_ifu.r_eff,
            # dyn_model_ifu.samples_mcmc[:num_total_samples, 4] * np.nan
        ])

        print('Power-law samples: {}'.format(all_pl_samples.shape))

        powerlaw_mc_samples = MCSamples(samples=all_pl_samples.T,
                                        names=[
                                            'theta_E', 'gamma', #'lambda_mst',
                                            'a_ani', 'r_eff', #'D_dt'
                                        ],
                                        labels=[
                                            '\\theta_{\\rm E}\ (^{\prime\prime})',
                                            '\\gamma',
                                            #'\\lambda_{\\rm MST}',
                                            'a_{\\rm ani}',
                                            'R_{\\rm eff}\ (^{\prime\prime})',
                                            #'D_{\\Delta t}'
                                        ],
                                        )

        smooth_factor = 1
        powerlaw_mc_samples.updateSettings({'smooth_scale_2D': smooth_factor})
        powerlaw_mc_samples.updateSettings({'smooth_scale_1D': smooth_factor})

        all_pl_samples = np.array([
            dyn_model_ifu.samples_mcmc[:num_total_samples, 1],
            dyn_model_ifu.samples_mcmc[:num_total_samples, 2],
            lambda_mst,
            dyn_model_ifu.ani_param,
            dyn_model_ifu.r_eff,
            dyn_model_ifu.samples_mcmc[:num_total_samples, 4] / lambda_mst
        ])

        importance_sampled_samples = all_pl_samples[:, np.random.choice(
            np.arange(all_pl_samples.shape[1]), size=1000,
            p=velocity_dispersion_likelihood / np.sum(
                velocity_dispersion_likelihood)
        )]

        importance_sampled_mc_samples = MCSamples(
            samples=importance_sampled_samples.T,
            names=[
                'theta_E', 'gamma', 'lambda_mst', 'a_ani', 'r_eff', 'D_dt'
            ],
            labels=[
                '\\theta_{\\rm E}',
                '\\gamma',
                '\\lambda_{\\rm MST}',
                'a_{\\rm ani}',
                'R_{\\rm eff}\ (^{\prime\prime})',
                'D_{\\Delta t}'
            ],
            )
        smooth_factor = 5
        importance_sampled_mc_samples.updateSettings(
            {'smooth_scale_2D': smooth_factor})
        importance_sampled_mc_samples.updateSettings(
            {'smooth_scale_1D': smooth_factor})

        ## single aperture velocity dispersion
        all_pl_samples_aperture = np.array([
            dyn_model_aperture.samples_mcmc[:num_total_samples, 1],
            dyn_model_aperture.samples_mcmc[:num_total_samples, 2],
            lambda_mst_aperture,
            dyn_model_aperture.ani_param,
            dyn_model_aperture.r_eff,
            dyn_model_aperture.samples_mcmc[:num_total_samples,
                                            4] / lambda_mst_aperture
        ])

        importance_sampled_samples_aperture = all_pl_samples_aperture[:,
                                          np.random.choice(
                                              np.arange(all_pl_samples_aperture.shape[
                                                      1]), size=1000,
                                              p=velocity_dispersion_likelihood_aperture / np.sum(
                                                      velocity_dispersion_likelihood_aperture)
                                          )]

        importance_sampled_mc_samples_aperture = MCSamples(
            samples=importance_sampled_samples_aperture.T,
            names=[
                'theta_E', 'gamma', 'lambda_mst', 'a_ani', 'r_eff', 'D_dt'
            ],
            labels=[
                '\\theta_{\\rm E}',
                '\\gamma',
                '\\lambda_{\\rm MST}',
                'a_{\\rm ani}',
                'R_{\\rm eff}\ (^{\prime\prime})',
                'D_{\\Delta t}'
            ],
            )
        smooth_factor = 3
        importance_sampled_mc_samples_aperture.updateSettings(
            {'smooth_scale_2D': smooth_factor})
        importance_sampled_mc_samples_aperture.updateSettings(
            {'smooth_scale_1D': smooth_factor})

        g = plots.getSubplotPlotter(subplot_size=2.2)
        g.settings.lw_contour = 1.
        g.settings.alpha_factor_contour_lines = 2.
        g.settings.solid_contour_palefactor = 0.5
        g.settings.axes_fontsize = 20
        g.settings.lab_fontsize = 20

        g.settings.legend_fontsize = 24
        # g.settings.smooth_scale_2D = 4
        # g.settings.smooth_scale_1D = 4

        colors = [paperfig.cb_red, paperfig.cb_blue, paperfig.cb_grey]

        g.triangle_plot(
            [importance_sampled_mc_samples,
             importance_sampled_mc_samples_aperture, powerlaw_mc_samples,
             ],
            legend_labels=['w/ IFU kinematics', 'w/ integrated kinematics',
                           'w/o kinematics'
                           ],
            filled=True, shaded=False,
            alpha_filled_add=.8,
            contour_lws=[1.6, 1.6],
            contour_ls=['-', '-'],
            # filled=False,
            # contour_colors=[sns.xkcd_rgb['emerald'], sns.xkcd_rgb['bright orange']],
            contour_args={'alpha': .5},
            # line_args={'lw': 7., "zorder": 30},
            # line_args={'lw': 1., 'alpha': 1.}
            contour_colors=colors,
            param_limits={'dphi_AD': (-0.6, 0.7),
                          'lambda_int': (0.6, 1.1),
                          'a_ani': (0.5, 3.),
                          }
            )

        return g.fig





