import numpy as np
import os
from tqdm import tqdm_notebook, tnrange
import joblib
from astropy.cosmology import FlatLambdaCDM
from astropy.io import fits
from scipy.interpolate import interp1d
from scipy import optimize
import matplotlib.pyplot as plt
import h5py as h5

from lenstronomy.Sampling.parameters import Param
from lenstronomy.Analysis.kinematics_api import KinematicsAPI
import lenstronomy.Util.class_creator as class_creator
from lenstronomy.Analysis.td_cosmography import TDCosmography
from lenstronomy.GalKin.galkin import Galkin
from lenstronomy.Cosmo.lens_cosmo import LensCosmo
from lenstronomy.Util.param_util import phi_q2_ellipticity
from lenstronomy.LightModel.light_model import LightModel
from lenstronomy.Analysis.light_profile import LightProfileAnalysis
from lenstronomy.Util.param_util import ellipticity2phi_q
import lenstronomy.Util.multi_gauss_expansion as mge
from lenstronomy.LensModel.lens_model import LensModel
from lenstronomy.Analysis.lens_profile import LensProfileAnalysis
from mgefit.mge_fit_1d import mge_fit_1d
from jampy.jam_axi_proj import jam_axi_proj
from jampy.jam_axi_intr import jam_axi_intr
from jampy.jam_sph_rms import jam_sph_rms
from astropy.convolution import Gaussian2DKernel
from astropy.convolution import convolve
from copy import deepcopy

cwd = os.getcwd()
base_path, _ = os.path.split(cwd)


class DynamicalModel(object):
    """
    Class to compute velocity dispersion in spherical symmetry for RXJ 1131.
    """
    PSF_FWHM = 0.7
    X_GRID, Y_GRID = np.meshgrid(
        np.arange(-3.0597, 3.1597, 0.1457), # x-axis points to negative RA
        np.arange(-3.0597, 3.1597, 0.1457),
    )

    PIXEL_SIZE = 0.1457
    X_CENTER = (21.5 - 21.5) * PIXEL_SIZE # 21.5
    Y_CENTER = (23.5 - 21.5) * PIXEL_SIZE # 23.5

    Z_L = 0.295 # deflector redshift from Agnello et al. (2018)
    Z_S = 0.657 # source redshift

    R_sersic_1 = lambda _: np.random.normal(2.49, 0.01) * np.sqrt(0.878)
    # np.sqrt(np.random.normal(0.912, 0.004))
    n_sersic_1 = lambda _: np.random.normal(0.93, 0.03)
    I_light_1 = lambda _: np.random.normal(0.091, 0.001)
    q_light_1 = lambda _: 0.878 #np.random.normal(0.921, 0.004)
    phi_light_1 = lambda _: 90 - 121.6 # np.random.normal(90-121.6, 0.5)

    R_sersic_2 = lambda _: np.random.normal(0.362, 0.009) * np.sqrt(0.849)
    # np.sqrt(np.random.normal(0.867, 0.002))
    n_sersic_2 = lambda _: np.random.normal(1.59, 0.03)
    I_light_2 = lambda _: np.random.normal(0.89, 0.03)
    q_light_2 = lambda _: 0.849 #np.random.normal(0.849, 0.004)

    R_EFF = 1.85

    def __init__(self,
                 mass_profile,
                 cosmo=None,
                 do_mge_light=True,
                 include_light_profile_uncertainty=False,
                 is_spherical_model=False
                 ):
        """
        Load the model output file and load the posterior chain and other model
        speification objects.
        """
        if mass_profile not in ['powerlaw', 'composite']:
            raise ValueError('Mass profile "{}" not recognized!'.format(
                mass_profile))

        self._do_mge_light = do_mge_light
        self._light_profile_uncertainty = include_light_profile_uncertainty
        self._is_spherical_model = is_spherical_model

        if self._do_mge_light:
            self.kwargs_model = {
                'lens_model_list': ['PEMD'],
                'lens_light_model_list': ['SERSIC', 'SERSIC'],
            }
        else:
            self.kwargs_model ={
                'lens_model_list': ['PEMD'],
                'lens_light_model_list': ['HERNQUIST'],
            }

        # numerical options to perform the numerical integrals
        self.kwargs_galkin_numerics = {#'sampling_number': 1000,
                                       'interpol_grid_num': 1000,
                                       'log_integration': True,
                                       'max_integrate': 100,
                                       'min_integrate': 0.001}

        self.lens_cosmo = LensCosmo(self.Z_L, self.Z_S, cosmo=cosmo)

        self._kwargs_cosmo = {'d_d': self.lens_cosmo.dd,
                              'd_s': self.lens_cosmo.ds,
                              'd_ds': self.lens_cosmo.dds}

        self.td_cosmography = TDCosmography(z_lens=self.Z_L, z_source=self.Z_S,
                                            kwargs_model=self.kwargs_model)

    def get_light_ellipticity_parameters(self):
        """

        """
        q1, phi = self.q_light_1(), self.phi_light_1()
        e11, e12 = phi_q2_ellipticity(phi, q1)

        q2 = self.q_light_2()
        e21, e22 = phi_q2_ellipticity(phi, q2)

        return e11, e12, e21, e22

    def get_double_sersic_kwargs(self, is_shperical=True):
        """

        """
        kwargs_lens_light = [
            {'amp': self.I_light_1(), 'R_sersic': self.R_sersic_1(),
             'n_sersic': self.n_sersic_1(),
             'center_x': self.X_CENTER, 'center_y': self.Y_CENTER},
            {'amp': self.I_light_2(), 'R_sersic': self.R_sersic_2(),
             'n_sersic': self.n_sersic_2(),
             'center_x': self.X_CENTER, 'center_y': self.Y_CENTER}
        ]

        if is_shperical:
            return kwargs_lens_light
        else:
            e11, e12, e21, e22 = self.get_light_ellipticity_parameters()

            kwargs_lens_light[0]['e1'] = e11
            kwargs_lens_light[0]['e2'] = e12

            kwargs_lens_light[1]['e1'] = e21
            kwargs_lens_light[1]['e2'] = e22

            return kwargs_lens_light

    @staticmethod
    def get_light_mge_2d(r_eff_multiplier=1):
        """
        Return light profile MGE for double Sersic fitted using mge_2d() in
        a separate notebook.
        """
        sigma_lum = np.array(
            [0.24344635, 0.36067945, 0.60628114, 1.13855552, 1.86364175,
             2.6761396, 3.55367977, 4.43848341, 5.33263851, 6.2222584,
             7.08440441, 8.12158143, 9.23341296, 10.29834757,
             19.07959706]) * r_eff_multiplier

        surf_lum = np.array(
            [4.73320859e-01, 4.72402840e-01, 5.58995798e-01, 1.02666580e+00,
             2.02419303e+00, 1.77472748e+00, 6.56147174e-01, 1.03750980e-01,
             6.98808495e-03, 1.86209035e-04, 2.16341896e-06, 2.11474312e-09,
             9.61877415e-14, 2.46144638e-15, 1.71561703e-25]) / np.sqrt(
            2 * np.pi * sigma_lum ** 2) ** 2

        qobs_lum = np.ones_like(sigma_lum) * 0.878

        return surf_lum, sigma_lum, qobs_lum

    def get_lenstronomy_light_kwargs(self, surf_lum, sigma_lum, qobs_lum):
        """

        """

    def get_light_profile_mge(self):
        """
        Get MGE of the double Sersic light profile.
        """
        if not self._light_profile_uncertainty:
            return self.get_light_mge_2d()
        else:
            pass
            # will compute light profile using mge_1d() below

        kwargs_light = self.get_double_sersic_kwargs(is_shperical=True)
        e11, e12, e21, e22 = self.get_light_ellipticity_parameters()

        light_model = LightModel(['SERSIC', 'SERSIC'])
        # x, y = np.meshgrid(np.arange(-5, 5, 0.01), np.arange(-5, 5, 0.01))
        # kwargs_light = self.get_double_sersic_kwargs(is_shperical=True)
        # model_image = light_model.surface_brightness(x, y, kwargs_light, )

        for i in range(2):
            kwargs_light[i]['center_x'] = 0
            kwargs_light[i]['center_y'] = 0

        n = 300
        rs_1 = np.logspace(-2.5, 2, n) * kwargs_light[0]['R_sersic']
        rs_2 = np.logspace(-2.5, 2, n) * kwargs_light[1]['R_sersic']

        flux_r_1 = light_model.surface_brightness(rs_1, 0 * rs_1, kwargs_light,
                                                  k=0)
        flux_r_2 = light_model.surface_brightness(rs_2, 0 * rs_2, kwargs_light,
                                                  k=1)

        mge_fit_1 = mge_fit_1d(rs_1, flux_r_1, ngauss=20, quiet=True)
        mge_fit_2 = mge_fit_1d(rs_2, flux_r_2, ngauss=20, quiet=True)

        mge_1 = (mge_fit_1.sol[0], mge_fit_1.sol[1])
        mge_2 = (mge_fit_2.sol[0], mge_fit_2.sol[1])

        sigma_lum = np.append(mge_1[1], mge_2[1])
        surf_lum = np.append(mge_1[0], mge_2[0]) / (np.sqrt(2*np.pi)*sigma_lum)

        _, q_1 = ellipticity2phi_q(e11,
                                   e12)  # kwargs_light[0]['e1'], kwargs_light[0]['e2'])
        _, q_2 = ellipticity2phi_q(e21,
                                   e22)  # kwargs_light[1]['e1'], kwargs_light[1]['e2'])

        qobs_lum = np.append(np.ones_like(mge_1[1]) * q_1,
                             np.ones_like(mge_2[1]) * q_2)

        sorted_indices = np.argsort(sigma_lum)
        sigma_lum = sigma_lum[sorted_indices]
        surf_lum = surf_lum[sorted_indices]
        qobs_lum = qobs_lum[sorted_indices]

        return surf_lum, sigma_lum, qobs_lum

    def get_mass_profile_mge(self, theta_e, gamma, q):
        """

        """
        lens_model = LensModel(['PEMD'])
        lens_cosmo = LensCosmo(z_lens=self.Z_L, z_source=self.Z_S)
        # 'q','$\theta_{E}$','$\gamma$','$\theta_{E,satellite}$','$\gamma_{ext}$','$\theta_{ext}$'

        n = 300
        r_array = np.logspace(-2.5, 2, n) * theta_e

        kwargs_lens = [{'theta_E': theta_e, 'gamma': gamma, 'e1': 0., 'e2': 0.,
                        'center_x': 0., 'center_y': 0.}]

        mass_r = lens_model.kappa(r_array, r_array * 0, kwargs_lens)

        # amps, sigmas, _ = mge.mge_1d(r_array, mass_r, N=20)
        mass_mge = mge_fit_1d(r_array, mass_r, ngauss=20, quiet=True,
                              plot=False)
        sigmas = mass_mge.sol[1]
        amps = mass_mge.sol[0] / (np.sqrt(2*np.pi) * sigmas)

        # mge_fit = mge_fit_1d(r_array, mass_r, ngauss=20)
        # print(mge_fit)

        surf_pot = lens_cosmo.kappa2proj_mass(amps) / 1e12 # M_sun / pc^2

        sigma_pot = sigmas
        qobs_pot = np.ones_like(sigmas) * q

        return surf_pot, sigma_pot, qobs_pot

    def get_anisotropy_bs(self, params, surf_lum, sigma_lum, model='Osipkov-Merritt',
                          plot=False):
        """
        """
        if model == 'Osipkov-Merritt':
            betas = sigma_lum**2 / (params**2 + sigma_lum**2)
        elif model == 'generalized-OM':
            betas = params[1] * sigma_lum**2 / (params[0]**2 + sigma_lum**2)
        elif model == 'constant':
            betas = (1 - params**2) * np.ones_like(sigma_lum)
        elif model == 'step':
            divider = 1. # arcsec
            betas = (1 - params[0]**2) * np.ones_like(sigma_lum)
            # betas[sigma_lum <= divider] = 1 - params[0]**2
            betas[sigma_lum > divider] = 1 - params[1]**2
        else:
            betas = sigma_lum * 0. # isotropic

        bs = betas
        # vs = np.zeros((len(sigma_lum), len(sigma_lum)))  # r, k
        #
        # for i in range(len(sigma_lum)):
        #     vs[i] = surf_lum[i] / (np.sqrt(2 * np.pi) * sigma_lum[i])**3 * \
        #             np.exp(
        #         -sigma_lum ** 2 / sigma_lum[i] ** 2 / 2.)
        #
        # ys = np.sum(vs, axis=0) / (1 - betas)
        # bs = optimize.nnls(vs, ys)[0] #bounds=(0, 1.)).x #.linalg.solve(
        # vs.T, ys)
        # #

        if plot:
            import matplotlib.pyplot as plt

            plt.loglog(sigma_lum, betas, 'o')
            plt.xlabel(r'$R$ (arcsec)')
            plt.ylabel(r'$\beta (R)$')
            plt.show()
            # plt.loglog(sigma_lum, 1 - (np.sum(vs, axis=0) / (vs.T @ bs)), 'o',
            #            ls='none', )

        return bs #betas #/ surf_lum * np.sum(surf_lum)

    @staticmethod
    def transform_pix_coords(xs, ys, x_center, y_center, angle):
        """
        """
        xs_ = xs - x_center
        ys_ = ys - y_center
        # angle *= np.pi / 180.

        xs_rotated = xs_ * np.cos(angle) + ys_ * np.sin(angle)
        ys_rotated = -xs_ * np.sin(angle) + ys_ * np.cos(angle)

        return xs_rotated, ys_rotated

    def get_jam_grid(self, phi=0., supersampling_factor=1):
        """
        """

        # n_pix = self.X_GRID.shape[0] * oversampling_factor
        # pix_size = self.PIXEL_SIZE / oversampling_factor
        #
        # pix_coordinates = np.arange(n_pix) * pix_size + pix_size / 2.
        # x_grid, y_grid = np.meshgrid(pix_coordinates, pix_coordinates)
        #
        # x_center_pix, y_center_pix = (n_pix - 1) / 2, (n_pix - 1) / 2
        # x_center_coord = x_center_pix * pix_size + pix_size / 2.
        # y_center_coord = y_center_pix * pix_size + pix_size / 2.

        delta_x = (self.X_GRID[0, 1] - self.X_GRID[0, 0])
        delta_y = (self.Y_GRID[1, 0] - self.Y_GRID[0, 0])
        assert np.abs(delta_x) == np.abs(delta_y)

        x_start = self.X_GRID[0, 0] - delta_x / 2. * (1 -
                                                      1 / supersampling_factor)
        x_end = self.X_GRID[0, -1] + delta_x / 2. * (1 -
                                                     1 / supersampling_factor)
        y_start = self.Y_GRID[0, 0] - delta_y / 2. * (1 -
                                                      1 / supersampling_factor)
        y_end = self.Y_GRID[-1, 0] + delta_y / 2. * (1 -
                                                     1 / supersampling_factor)

        xs = np.arange(x_start, x_end + delta_x / (10 + supersampling_factor),
                       delta_x / supersampling_factor)
        ys = np.arange(y_start, y_end + delta_y / (10 + supersampling_factor),
                       delta_y / supersampling_factor)

        x_grid_supersampled, y_grid_supersmapled = np.meshgrid(xs, ys)

        # x_grid = -(x_grid - x_center_coord)
        # y_grid = (y_grid - y_center_coord)

        x_grid, y_grid = self.transform_pix_coords(x_grid_supersampled,
                                            y_grid_supersmapled,
                                            self.X_CENTER, self.Y_CENTER, phi
                                          )

        return x_grid.flatten(), y_grid.flatten(), \
               x_grid_supersampled.flatten(), y_grid_supersmapled.flatten()

    def get_surface_brightness_image(self, surf_lum, sigma_lum, qobs_lum,
                                     x_grid,
                                     y_grid):
        """
        """
        e1, e2 = phi_q2_ellipticity(self.phi_light_1(), np.mean(qobs_lum))
        kwargs_light = [
            {'amp': surf_lum,
             'sigma': sigma_lum,
             'e1': e1,
             'e2': e2,
             'center_x': self.X_CENTER,
             'center_y': self.Y_CENTER
             }
        ]

        light_model = LightModel(['MULTI_GAUSSIAN_ELLIPSE'])

        model_image = light_model.surface_brightness(x_grid, y_grid,
                                                     kwargs_light)

        return model_image

    def compute_jampy_v_rms_model(self, theta_e, gamma,
                                  ani_param,
                                  q=1, pa=0,
                                  inclination=0,
                                  anisotropy_model='Oskipkov-Merritt',
                                  do_convolve=True,
                                  supersampling_factor=5,
                                  voronoi_bins=None,
                                  om_r_scale=None
                                  ):
        """
        :param pa: positoin angle in radian
        """
        surf_lum, sigma_lum, qobs_lum = self.get_light_profile_mge()

        surf_pot, sigma_pot, qobs_pot = self.get_mass_profile_mge(theta_e,
                                                                  gamma, q)
        if self._is_spherical_model:
            qobs_lum = np.ones_like(qobs_lum)
            qobs_pot = np.ones_like(qobs_pot)

        bs = self.get_anisotropy_bs(ani_param, surf_lum, sigma_lum,
                                    model=anisotropy_model
                                    )
        phi = 180. - pa/np.pi*180.

        x_grid_spaxel, y_grid_spaxel, _, _ = self.get_jam_grid(phi,
                                                        supersampling_factor=1)

        x_grid, y_grid, x_grid_original, y_grid_original = self.get_jam_grid(
                                    phi,
                                    supersampling_factor=supersampling_factor)

        # print(x_grid.shape, y_grid.shape)
        if do_convolve:
            sigma_psf = self.PSF_FWHM / 2.355
        else:
            sigma_psf = 0.
        norm_psf = 1.

        mbh = 0
        distance = self.lens_cosmo.dd

        if self._is_spherical_model:
            jam = jam_sph_rms(
                surf_lum, sigma_lum,  # qobs_lum,
                surf_pot, sigma_pot,  # qobs_pot,
                # inclination,
                mbh, distance,
                np.sqrt(x_grid_spaxel ** 2 + y_grid_spaxel ** 2),
                plot=False, pixsize=self.PIXEL_SIZE,
                # self.PIXEL_SIZE/supersampling_factor,
                # pixang=phi,
                quiet=1,
                sigmapsf=sigma_psf, normpsf=norm_psf,
                # moment='zz',
                # goodbins=goodbins,
                # align='sph',
                beta=bs if om_r_scale is None else None,
                scale=om_r_scale,
                # data=rms, errors=erms,
                ml=1
            )[0]
        else:
            jam = jam_axi_proj(
                surf_lum, sigma_lum, qobs_lum,
                surf_pot, sigma_pot, qobs_pot,
                inclination, mbh, distance,
                x_grid_spaxel, y_grid_spaxel,
                plot=False, pixsize=self.PIXEL_SIZE,
                # self.PIXEL_SIZE/supersampling_factor,
                pixang=phi, quiet=1,
                sigmapsf=sigma_psf, normpsf=norm_psf,
                moment='zz',
                # goodbins=goodbins,
                align='sph',
                beta=bs,
                # data=rms, errors=erms,
                ml=1).model

        num_pix = int(np.sqrt(len(x_grid)))
        num_pix_spaxel = int(np.sqrt(len(x_grid_spaxel)))
        vel_dis_model = jam.reshape((num_pix_spaxel, num_pix_spaxel))

        flux = self.get_surface_brightness_image(surf_lum * (2*np.pi*sigma_lum**2),
                                                 sigma_lum, qobs_lum,
                                                 x_grid_original,
                                                 y_grid_original).reshape((num_pix,num_pix))

        if do_convolve:
            sigma = self.PSF_FWHM / 2.355 / self.PIXEL_SIZE * supersampling_factor
            kernel = Gaussian2DKernel(x_stddev=sigma,
                                      x_size=4 * int(sigma) + 1,
                                      y_size=4 * int(sigma) + 1)

            convolved_flux = convolve(flux, kernel)
            # convolved_map = convolve(flux * vel_dis_map ** 2, kernel)
            # convolved_flux_spaxels = convolved_flux.reshape(
            #         len(self.X_GRID), supersampling_factor,
            #         len(self.Y_GRID), supersampling_factor
            #     ).sum(3).sum(1)
            # convolved_map = convolved_flux_spaxels * vel_dis_model ** 2
        else:
            convolved_flux = flux

        convolved_flux_spaxel = convolved_flux.reshape(
            len(self.X_GRID), supersampling_factor,
            len(self.Y_GRID), supersampling_factor
        ).sum(3).sum(1)
        # convolved_map = convolved_flux_spaxel * vel_dis_model**2

        if voronoi_bins is not None:
            # supersampled_voronoi_bins = voronoi_bins.repeat(
            #     supersampling_factor, axis=0).repeat(supersampling_factor,
            #                                          axis=1)

            # n_bins = int(np.max(voronoi_bins)) + 1
            #
            # binned_map = np.zeros(n_bins)
            # binned_IR = np.zeros(n_bins)
            # for n in range(n_bins):
            #     binned_map[n] = np.sum(
            #         convolved_map[voronoi_bins == n]
            #     )
            #     binned_IR[n] = np.sum(
            #         convolved_flux_spaxel[voronoi_bins == n]
            #     )
            # vel_dis_map = np.sqrt(binned_map / binned_IR)
            vel_dis_map, intensity_map = self.bin_map_in_voronoi_bins(vel_dis_model,
                                                                      convolved_flux_spaxel,
                                                                      voronoi_bins)
        else:
            # binned_map = convolved_map.reshape(
            #     len(self.X_GRID), supersampling_factor,
            #     len(self.Y_GRID), supersampling_factor
            # ).sum(3).sum(1)
            #
            # IR_integrated = convolved_flux.reshape(
            #     len(self.X_GRID), supersampling_factor,
            #     len(self.Y_GRID), supersampling_factor
            # ).sum(3).sum(1)
            vel_dis_map = vel_dis_model
            intensity_map = convolved_flux_spaxel

        return vel_dis_map, intensity_map

    def bin_map_in_voronoi_bins(self, vel_dis_map, IR_map, voronoi_bins):
        """

        """
        n_bins = int(np.nanmax(voronoi_bins)) + 1

        binned_map = np.zeros(n_bins)

        binned_IR = np.zeros(n_bins)

        for n in range(n_bins):
            binned_map[n] = np.sum(
                (vel_dis_map[voronoi_bins == n])**2 * IR_map[voronoi_bins == n]
            )
            binned_IR[n] = np.sum(
                IR_map[voronoi_bins == n]
            )

        binned_vel_dis_map = np.sqrt(binned_map / binned_IR)

        return binned_vel_dis_map, binned_IR

    def retrieve_anisotropy_profile(self, Rs,
                                    ani_param,
                                    surf_lum, sigma_lum,
                                    surf_pot, sigma_pot,
                                    print_step=None,
                                    bs=None,
                                    r_eff_uncertainty=0.02,
                                    analytic_kinematics=False,
                                    supersampling_factor=1,
                                    voronoi_bins=None,
                                    single_slit=False,
                                    do_convolve=True,
                                    anisotropy_model='Osipkov-Merritt',
                                    is_spherical=True,
                                    alignment='sph'
                                    ):
        """
        """

        qobs_lum = np.ones_like(sigma_lum)
        qobs_pot = np.ones_like(sigma_pot)

        if bs is None:
            bs = self.get_anisotropy_bs(ani_param, surf_lum, sigma_lum,
                                        model=anisotropy_model
                                        )

        # inc = 90
        mbh = 0
        distance = self.lens_cosmo.dd

        # for moment in ['xx', 'zz']:
        jam_moments = jam_axi_intr(
            surf_lum / (np.sqrt(2 * np.pi) * sigma_lum),
            sigma_lum, qobs_lum,
            surf_pot / (np.sqrt(2 * np.pi) * sigma_pot),
            sigma_pot, qobs_pot,
            #inclination,
            mbh=mbh,
            #distance,
            Rbin=Rs, zbin=Rs*0,
            plot=False, # pixsize=0.,
            # self.PIXEL_SIZE/supersampling_factor,
            # pixang=0., quiet=1,
            # sigmapsf=0., normpsf=norm_psf,
            #moment=moment,
            # goodbins=goodbins,
            align=alignment,
            beta=bs,
            proj_cyl=False,
            # data=rms, errors=erms,
            ml=1).model
        # print(jam_moments)
        # print(jam_moments[0])
        return 1 - jam_moments[1] / jam_moments[0]

    def get_galkin_kinematics_api(self, anisotropy_model='Osipkov-Merritt',
                                  single_slit=False
                                  ):
        """
        """
        if anisotropy_model == 'Osipkov-Merritt':
            anisotropy_type = 'OM'  # anisotropy model applied
        elif anisotropy_model == 'constant':
            anisotropy_type = 'const'
        else:
            raise ValueError('anisotropy model {} model not '
                             'supported!'.format(anisotropy_model))

        if single_slit:
            kwargs_aperture = {'aperture_type': 'slit',
                               'length': 1.,
                               'width': 0.81,
                               'center_ra': 0.,
                               # lens_light_result[0]['center_x'],
                               'center_dec': 0.,
                               # lens_light_result[0]['center_y'],
                               'angle': 0
                               }
            kwargs_seeing = {'psf_type': 'GAUSSIAN',
                             'fwhm': self.PSF_FWHM
                             # 'moffat_beta': self.MOFFAT_BETA[n]
                             }
        else:
            kwargs_aperture = {'aperture_type': 'IFU_grid',
                               'x_grid': self.X_GRID,
                               'y_grid': self.Y_GRID,
                               #'angle': 0
                               }
            kwargs_seeing = {'psf_type': 'GAUSSIAN',
                             'fwhm': self.PSF_FWHM,
                             # 'moffat_beta': self.MOFFAT_BETA[n]
                             }

        if self._do_mge_light:
            light_model_bool = [True, True]
        else:
            light_model_bool = [True]

        lens_model_bool = [True]

        kinematics_api = KinematicsAPI(z_lens=self.Z_L, z_source=self.Z_S,
                                       kwargs_model=self.kwargs_model,
                                       kwargs_aperture=kwargs_aperture,
                                       kwargs_seeing=kwargs_seeing,
                                       anisotropy_model=anisotropy_type,
                                       cosmo=None,
                                       lens_model_kinematics_bool=lens_model_bool,
                                       light_model_kinematics_bool=light_model_bool,
                                       multi_observations=False,
                                       kwargs_numerics_galkin=self.kwargs_galkin_numerics,
                                       analytic_kinematics=(not self._do_mge_light),
                                       Hernquist_approx=False,
                                       MGE_light=self._do_mge_light,
                                       MGE_mass=False,  #self._do_mge_light,
                                       kwargs_mge_light=None,
                                       kwargs_mge_mass=None,
                                       sampling_number=1000,
                                       num_kin_sampling=2000,
                                       num_psf_sampling=500,
                                       )

        return kinematics_api

    def compute_galkin_v_rms_model(self, kinematics_api,
                                   theta_E, gamma,
                                   ani_param,
                                   r_eff_multiplier=1,
                                   anisotropy_model='Osipkov-Merritt',
                                   aperture='ifu',
                                   voronoi_bins=None,
                                   supersampling_factor=5,
                                   ):
        """

        """
        kwargs_lens, kwargs_lens_light, kwargs_anisotropy = self.get_lens_mass_kwargs(
            theta_E, gamma,
            ani_param,
            r_eff_multiplier=r_eff_multiplier,
            anisotropy_model=anisotropy_model
        )

        if aperture == 'single_slit':
            v_rms = kinematics_api.velocity_dispersion(
                kwargs_lens,
                kwargs_lens_light,
                # kwargs_result['kwargs_lens_light'],
                kwargs_anisotropy,
                # r_eff=(1 + np.sqrt(2)) * r_eff, theta_E=theta_E,
                # gamma=gamma,
                kappa_ext=0,
            )
            intensity_map = None
        else:
            v_rms, intensity_map = kinematics_api.velocity_dispersion_map(
                kwargs_lens,
                kwargs_lens_light,
                # kwargs_result['kwargs_lens_light'],
                kwargs_anisotropy,
                # r_eff=(1 + np.sqrt(2)) * r_eff, theta_E=theta_E,
                # gamma=gamma,
                kappa_ext=0,
                direct_convolve=True,
                supersampling_factor=supersampling_factor,
                voronoi_bins=voronoi_bins,
                get_IR_map=True
            )

        return v_rms, intensity_map

    def compute_galkin_v_rms_radial_profile(self, rs, theta_E, gamma,
                                            ani_param,
                                            anisotropy_model='Osipkov-Merritt',
                                            ):
        """
        Compute v_rms using Galkin's numerical
        approach.
        """
        bins = np.zeros(len(rs)+1)
        bins[:-1] = rs - (rs[1] - rs[0])/2.
        bins[-1] = rs[-1] + (rs[1] - rs[0])/2.

        kwargs_aperture = {'aperture': 'IFU_shells',
                           'r_bins': bins,
                           }

        kwargs_seeing = {'psf_type': 'GAUSSIAN',
                         'fwhm': 1e-3,
                         }

        kwargs_galkin_numerics = {  # 'sampling_number': 1000,
            'interpol_grid_num': 1000,
            'log_integration': True,
            'max_integrate': 100,
            'min_integrate': 0.001}

        if anisotropy_model == 'Osipkov-Merritt':
            anisotropy_type = 'OM'  # anisotropy model applied
        elif anisotropy_model == 'constant':
            anisotropy_type = 'const'
        else:
            raise ValueError('anisotropy model {} model not '
                             'supported!'.format(anisotropy_model))

        kinematics_api = KinematicsAPI(z_lens=self.Z_L,
                                       z_source=self.Z_S,
                                       kwargs_model=self.kwargs_model,
                                       kwargs_aperture=kwargs_aperture,
                                       kwargs_seeing=kwargs_seeing,
                                       anisotropy_model=anisotropy_type,
                                       cosmo=None,
                                       multi_observations=False,
                                       kwargs_numerics_galkin=kwargs_galkin_numerics,
                                       analytic_kinematics=False,
                                       Hernquist_approx=False,
                                       MGE_light=True,
                                       MGE_mass=False,
                                       kwargs_mge_light=None,
                                       kwargs_mge_mass=None,
                                       sampling_number=1000,
                                       num_kin_sampling=2000,
                                       num_psf_sampling=500,
                                       )

        kwargs_lens, kwargs_lens_light, kwargs_anisotropy = \
            self.get_lens_mass_kwargs(theta_E, gamma, ani_param,
                                      anisotropy_model=anisotropy_model)

        vel_dis = kinematics_api.velocity_dispersion_map(
            kwargs_lens,
            kwargs_lens_light,
            kwargs_anisotropy,
            # r_eff=(1+np.sqrt(2))*r_eff, theta_E=theta_E,
            # gamma=gamma,
            kappa_ext=0,
            direct_convolve=False,
            #                     supersampling_factor=supersampling_factor,
            #                     voronoi_bins=voronoi_bins,
            #                     get_IR_map=True
        )

        return vel_dis

    def get_lens_mass_kwargs(self, theta_E, gamma,
                             ani_param,
                             anisotropy_model='Osipkov-Merritt',
                             r_eff_multiplier=None
                             ):
        """
        :param r_eff_multiplier: a factor for r_eff to account for uncertainty.
        """
        kwargs_lens = [{'theta_E': theta_E,
                        'gamma': gamma,
                        'e1': 0., 'e2': 0.,
                        'center_x': self.X_CENTER,
                        'center_y': self.Y_CENTER
                        }]

        if self._do_mge_light:
            kwargs_lens_light = self.get_double_sersic_kwargs(
                is_shperical=True)
        else:
            kwargs_lens_light = [{'amp': 1.,
                                  'Rs': self.R_EFF * r_eff_multiplier / (1 + np.sqrt(2)),
                                  'center_x': self.X_CENTER,
                                  'center_y': self.Y_CENTER
                                  }]

        # set the anisotropy radius. r_eff is pre-computed half-light
        # radius of the lens light
        if anisotropy_model == 'Osipkov-Merritt':
            kwargs_anisotropy = {'r_ani': ani_param}
        elif anisotropy_model == 'constant':
            kwargs_anisotropy = {'beta': ani_param}
        else:
            raise ValueError('anisotropy model {} not recognized!'.format(anisotropy_model))

        return kwargs_lens, kwargs_lens_light, kwargs_anisotropy


class KinematicLikelihood(object):

    def __init__(self,
                 lens_model_type='powerlaw',
                 software='jampy',
                 anisotropy='Osipkov-Meritt',
                 aperture='ifu',
                 snr_per_bin=15,
                 is_spherical=False
                 ):
        """

        """
        self.lens_model_type = lens_model_type
        self.software = software
        self.anistropy = anisotropy
        self.aperture = aperture
        self.snr_per_bin = snr_per_bin
        self.is_spherical = is_spherical

        self.bin_mapping = None
        self.load_bin_map(self.snr_per_bin)

        self.velocity_dispersion_mean = None
        self.velocity_dispersion_covariance = None

        if self.aperture == 'single_slit':
            self.velocity_dispersion_mean = 323 # km/s
            self.velocity_dispersion_covariance = 20**2 # km/s
        else:
            self.load_velocity_dispersion_data_ifu(self.snr_per_bin)

        self.lens_model_posterior_mean = None
        self.lens_model_posterior_covariance = None
        self.load_lens_model_posterior(self.lens_model_type)

        self._inclination_prior_interp = None

        if self.software == 'galkin':
            if True:
                self.kwargs_model = {
                    'lens_model_list': ['PEMD'],
                    'lens_light_model_list': ['SERSIC', 'SERSIC'],
                }
            else:
                self.kwargs_model = {
                    'lens_model_list': ['PEMD'],
                    'lens_light_model_list': ['HERNQUIST'],
                }

            # numerical options to perform the numerical integrals
            self.kwargs_galkin_numerics = {  # 'sampling_number': 1000,
                'interpol_grid_num': 1000,
                'log_integration': True,
                'max_integrate': 100,
                'min_integrate': 0.001}

            self.lens_cosmo = LensCosmo(self.Z_L, self.Z_S, cosmo=cosmo)

            self._kwargs_cosmo = {'d_d': self.lens_cosmo.dd,
                                  'd_s': self.lens_cosmo.ds,
                                  'd_ds': self.lens_cosmo.dds}

            self.td_cosmography = TDCosmography(z_lens=self.Z_L,
                                                z_source=self.Z_S,
                                                kwargs_model=self.kwargs_model)
            self.kinematics_api = None
            self.setup_galkin_api(self.anistropy, self.aperture)

    def setup_galkin_api(self, anisotropy, aperture):
        """

        """
        if anisotropy == 'Osipkov-Merritt':
            anisotropy_type = 'OM'  # anisotropy model applied
        elif anisotropy == 'constant':
            anisotropy_type = 'const'
        else:
            raise ValueError('anisotropy model {} model not '
                             'supported!'.format(anisotropy))

        if aperture == 'single_slit':
            kwargs_aperture = {'aperture': 'slit',
                               'length': 1.,
                               'width': 0.81,
                               'center_ra': 0.,
                               # lens_light_result[0]['center_x'],
                               'center_dec': 0.,
                               # lens_light_result[0]['center_y'],
                               'angle': 0
                               }
            kwargs_seeing = {'psf_type': 'GAUSSIAN',
                             'fwhm': 0.7,
                             # 'moffat_beta': self.MOFFAT_BETA[n]
                             }
        else:
            kwargs_aperture = {'aperture': 'IFU_grid',
                               'x_grid': self.X_GRID,
                               'y_grid': self.Y_GRID,
                               #'angle': 0
                               }
            kwargs_seeing = {'psf_type': 'GAUSSIAN',
                             'fwhm': self.PSF_FWHM,
                             # 'moffat_beta': self.MOFFAT_BETA[n]
                             }

        if self._cgd:
            light_model_bool = [True, True]
        else:
            light_model_bool = [True]
        lens_model_bool = [True]

        self.kinematics_api = KinematicsAPI(z_lens=self.Z_L, z_source=self.Z_S,
                                       kwargs_model=self.kwargs_model,
                                       kwargs_aperture=kwargs_aperture,
                                       kwargs_seeing=kwargs_seeing,
                                       anisotropy_model=anisotropy_type,
                                       cosmo=None,
                                       lens_model_kinematics_bool=lens_model_bool,
                                       light_model_kinematics_bool=light_model_bool,
                                       multi_observations=False,
                                       kwargs_numerics_galkin=self.kwargs_galkin_numerics,
                                       analytic_kinematics=(not self._cgd),
                                       Hernquist_approx=False,
                                       MGE_light=self._cgd,
                                       MGE_mass=False, #self._do_mge_light,
                                       kwargs_mge_light=None,
                                       kwargs_mge_mass=None,
                                       sampling_number=1000,
                                       num_kin_sampling=2000,
                                       num_psf_sampling=500,
                                       )

    def get_galkin_velocity_dispersion(self, theta_e, gamma):
        """
        """

    def get_lens_kwargs(self, theta_E, gamma,
                        anisotropy_model, ani_param
                        ):
        """

        """
        kwargs_lens = [{'theta_E': theta_E,
                        'gamma': gamma,
                        'e1': 0., 'e2': 0.,
                        'center_x': self.X_CENTER,
                        'center_y': self.Y_CENTER
                        }]

        if self._cgd:
            kwargs_lens_light = self.get_double_sersic_kwargs(
                is_shperical=True)
        else:
            kwargs_lens_light = [{'amp': 1., 'Rs': self.R_EFF / (1 + np.sqrt(
                2)),
                                  'center_x': self.X_CENTER,
                                  'center_y': self.Y_CENTER
                                  }]

        # set the anisotropy radius. r_eff is pre-computed half-light
        # radius of the lens light
        if anisotropy_model == 'Osipkov-Merritt':
            kwargs_anisotropy = {'r_ani': ani_param * r_eff}
        elif anisotropy_model == 'constant':
            kwargs_anisotropy = {'beta': ani_param}

        return kwargs_lens, kwargs_lens_light, kwargs_anisotropy


    def load_lens_model_posterior(self, lens_model_type):
        """

        """
        with h5.File('./data_products/lens_model_posterior_{}.h5'.format(
                lens_model_type)) as f:
            self.lens_model_posterior_mean = f['mean'][()]
            self.lens_model_posterior_covariance = f['covariance'][()]

    def load_velocity_dispersion_data_ifu(self, snr_per_bin):
        """

        """
        with h5.File('./data_products/'
                     'systematic_marginalized_velocity_dispersion'
                     '_snr_per_bin_{}.h5'.format(snr_per_bin)) as f:
            self.velocity_dispersion_mean = f['mean'][()]
            self.velocity_dispersion_covariance = f['covariance'][()]

    def load_bin_map(self, snr_per_bin, plot=False):
        """
        """
        bins = np.loadtxt('./data_products/binning_map_snr_per_bin_{'
                          '}.txt'.format(snr_per_bin))
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

    def log_prior(self, params):
        """
        """

    def inclination_prior(self, inclination):
        """

        """
        if self._inclination_prior_interp is None:
            scrapped_points = np.array([
                0, 0,
                0.05, 0,
                0.116, 0,
                0.17, 0.049,
                0.223, 0.223,
                0.272, 0.467,
                0.322, 0.652,
                0.376, 0.745,
                0.426, 0.842,
                0.475, 0.995,
                0.525, 1.109,
                0.577, 1.217,
                0.626, 1.337,
                0.676, 1.484,
                0.725, 1.516,
                0.776, 1.576,
                0.826, 1.489,
                0.876, 1.342,
                0.928, 1.076,
                0.976, 0.755,
            ])

            x = scrapped_points[::2]
            y = scrapped_points[1::2]

            self._inclination_prior_interp = interp1d(x, y, bounds_error=False,
                                   fill_value=0.)
        else:
            pass

        return np.log(self._inclination_prior_interp(inclination))

    def log_likelihood(self, params):
        """
        """

    def lens_model_likelihood(self, theta_e, gamma, q, PA, ):
        """

        """
    def log_probability(self, params):
        """
        """