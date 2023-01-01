import numpy as np
import os
from tqdm import tqdm_notebook, tnrange
import joblib
from astropy.cosmology import FlatLambdaCDM
from astropy.io import fits
from scipy.interpolate import interp1d
from scipy import optimize
import matplotlib.pyplot as plt

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
    Class to compute velocity dispersion in is_spherical symmetry for RXJ 1131.
    """
    # several static values defined as class attributes
    PSF_FWHM = 0.96 # arcsec
    PIXEL_SIZE = 0.1457  # KCWI datacube pixel size in arcsec

    # x and y coordinates of the KCWI datacube grid
    X_GRID, Y_GRID = np.meshgrid(
        -1 * np.arange(-3.0597, 3.1597, 0.1457),
        # x-axis points to negative RA
        np.arange(-3.0597, 3.1597, 0.1457),
    )

    # 23.29, 22.23, from notebook to find center of galaxy
    X_CENTER = -(23.29 - 21.) * PIXEL_SIZE
    Y_CENTER = (22.23 - 21.) * PIXEL_SIZE
    # X_CENTER = -(24 - 21.) * PIXEL_SIZE  # 23.5
    # Y_CENTER = (22 - 21.) * PIXEL_SIZE  # 21.5

    Z_L = 0.295  # deflector redshift from Agnello et al. (2018)
    Z_S = 0.657  # source redshift from Agnello et al. (2018)

    R_EFF = 1.91  # 1.8535 # arcsec

    # mean of light profile parameters using a double Sersic profile
    # [amp_1, R_sersic_1, n_sersic_1, PA, q_1, amp_2, R_sersic_2,
    # n_sersic_2, q_2]
    LIGHT_PROFILE_MEAN = np.array([4.40474442e+02, 3.00299605e-01,
                                   1.60397112e+00, 1.19135121e+02,
                                   8.47257348e-01, 3.28341603e+01,
                                   2.43706866e+00, 1.09647022e+00,
                                   8.65239555e-01])
    # covariance of light profile parameters using a double Sersic profile
    LIGHT_PROFILE_COVARIANCE = np.array(
        [[4.11515860e+01, -2.03208975e-02, -1.03104740e-01,
          1.09693956e-01, 1.48999694e-03, 8.56885752e-01,
          -3.13987606e-02, 5.36186134e-02, -1.17388270e-03],
         [-2.03208975e-02, 1.04182020e-05, 4.96588472e-05,
          -3.55194658e-05, -6.25591932e-07, -4.23764649e-04,
          1.59474533e-05, -2.95492629e-05, 6.36218169e-07],
         [-1.03104740e-01, 4.96588472e-05, 2.86411322e-04,
          -3.62377587e-04, -4.51689429e-06, -2.18435253e-03,
          7.89894920e-05, -1.28496654e-04, 2.92070310e-06],
         [1.09693956e-01, -3.55194658e-05, -3.62377587e-04,
          2.31043783e-01, 1.20140352e-05, 1.17187785e-03,
          2.99727402e-06, -4.30712951e-05, 3.22913188e-05],
         [1.48999694e-03, -6.25591932e-07, -4.51689429e-06,
          1.20140352e-05, 5.65701798e-06, 3.48519797e-05,
          -1.15319484e-06, 7.65111027e-07, -4.52028785e-07],
         [8.56885752e-01, -4.23764649e-04, -2.18435253e-03,
          1.17187785e-03, 3.48519797e-05, 2.75694523e-02,
          -9.68412872e-04, 8.38614112e-04, -2.01915206e-05],
         [-3.13987606e-02, 1.59474533e-05, 7.89894920e-05,
          2.99727402e-06, -1.15319484e-06, -9.68412872e-04,
          3.66265600e-05, -3.65495944e-05, 7.83519423e-07],
         [5.36186134e-02, -2.95492629e-05, -1.28496654e-04,
          -4.30712951e-05, 7.65111027e-07, 8.38614112e-04,
          -3.65495944e-05, 1.08987732e-04, -2.22861802e-06],
         [-1.17388270e-03, 6.36218169e-07, 2.92070310e-06,
          3.22913188e-05, -4.52028785e-07, -2.01915206e-05,
          7.83519423e-07, -2.22861802e-06, 2.05730034e-06]])

    def __init__(self,
                 mass_model,
                 cosmo=None,
                 do_mge_light=True,
                 n_gauss=20,
                 mass_profile_min=10 ** -2.5,
                 mass_profile_max=10 ** 2,
                 light_profile_min=10 ** -2.5,
                 light_profile_max=10 ** 2,
                 ):
        """
        Initialize the class
        :param mass_model: str, 'powerlaw' or 'composite'
        :param cosmo: astropy.cosmology object
        :param do_mge_light: bool, if True, use MGE to fit the light profile
        :param n_gauss: int, number of Gaussians to use in MGE
        :param mass_profile_min: float, minimum radius for mass profile MGE
        :param mass_profile_max: float, maximum radius for mass profile MGE
        :light_profile_min: float, minimum radius for light profile MGE
        :light_profile_max: float, maximum radius for light profile MGE
        """
        if mass_model not in ['powerlaw', 'composite']:
            raise ValueError('Mass profile "{}" not recognized!'.format(
                mass_model))

        self.mass_model = mass_model

        self._do_mge_light = do_mge_light
        # self._is_spherical_model = is_spherical_model
        self._n_gauss = n_gauss
        self._mass_profile_min = mass_profile_min
        self._mass_profile_max = mass_profile_max
        self._light_profile_min = light_profile_min
        self._light_profile_max = light_profile_max

        if self._do_mge_light:
            self.kwargs_model = {
                'lens_model_list': ['PEMD'],
                # 'lens_light_model_list': ['SERSIC', 'SERSIC'],
                'lens_light_model_list': ['MULTI_GAUSSIAN'],
            }
        else:
            self.kwargs_model = {
                'lens_model_list': ['PEMD'],
                'lens_light_model_list': ['SERSIC', 'SERSIC'],
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

        self.td_cosmography = TDCosmography(z_lens=self.Z_L, z_source=self.Z_S,
                                            kwargs_model=self.kwargs_model)

        self.interp_nfw_q = interp1d(_potential_qs, _nfw_qs)

    def get_light_ellipticity_parameters(self, params=None):
        """
        Get the ellipticity parameters for the 2 Sersic light profiles.
        :param params: list, [phi, q for sersic 1, q for sersic 2], if None, will call self.sample_from_double_sersic_fit()
        :return: e11, e12, e21, e22
        """
        if params is None:
            phi, q1, q2 = self.sample_from_double_sersic_fit(
                mode='ellipticities')
        else:
            phi, q1, q2 = params

        phi_radian = (90. - phi) / 180. * np.pi

        # q1, phi = self.q_light_1(), (90. - self.phi_light_1()) / 180. * np.pi
        e11, e12 = phi_q2_ellipticity(phi_radian, q1)

        # q2 = self.q_light_2()
        e21, e22 = phi_q2_ellipticity(phi_radian, q2)

        return e11, e12, e21, e22

    def get_double_sersic_kwargs(self, is_shperical=True, params=None):
        """
        Get the kwargs for the 2 Sersic light profiles
        :param is_shperical: bool, if True, will return kwargs for spherical case
        :return: kwargs_lens_light
        """
        if params is None:
            params = self.sample_from_double_sersic_fit()

        kwargs_lens_light = [
            {'amp': params[0], 'R_sersic': params[1],
             'n_sersic': params[2],
             'center_x': self.X_CENTER, 'center_y': self.Y_CENTER},
            {'amp': params[5], 'R_sersic': params[6],
             'n_sersic': params[7],
             'center_x': self.X_CENTER, 'center_y': self.Y_CENTER}
            # {'amp': self.I_light_1(), 'R_sersic': self.R_sersic_1(),
            #  'n_sersic': self.n_sersic_1(),
            #  'center_x': self.X_CENTER, 'center_y': self.Y_CENTER},
            # {'amp': self.I_light_2(), 'R_sersic': self.R_sersic_2(),
            #  'n_sersic': self.n_sersic_2(),
            #  'center_x': self.X_CENTER, 'center_y': self.Y_CENTER}
        ]

        if is_shperical:
            return kwargs_lens_light
        else:
            e11, e12, e21, e22 = self.get_light_ellipticity_parameters(
                params[[3, 4, 8]])

            kwargs_lens_light[0]['e1'] = e11
            kwargs_lens_light[0]['e2'] = e12

            kwargs_lens_light[1]['e1'] = e21
            kwargs_lens_light[1]['e2'] = e22

            return kwargs_lens_light

    @staticmethod
    def get_light_mge_2d_fit(r_eff_multiplier=1):
        """
        Return light profile MGE for double Sersic fitted using mge_2d() in
        a separate notebook
        :param r_eff_multiplier: float, multiply the effective radius by this
        :return: surf_lum, sigma_lum, qobs_lum
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
        Get lenstronomy light kwargs for given MGE parameters
        """
        raise NotImplementedError

    def get_light_mge(self, common_ellipticity=False,
                      is_spherical=False, set_q=None
                      ):
        """
        Get MGE of the double Sersic light profile
        :param common_ellipticity: bool, if True, will use the same ellipticity
        for both light profiles
        :param is_spherical: bool, if True, will return kwargs for spherical case
        :param set_q: float, if not None, will set the q of both light profiles
        :return: surf_lum, sigma_lum, qobs_lum, position angle
        """
        # if not self._light_profile_uncertainty:
        #     return self.get_light_mge_2d_fit()
        # else:
        #     pass
        # will compute light profile using mge_1d() below

        # get the kwargs light with circular profile for mge_1d
        params = self.sample_from_double_sersic_fit()
        kwargs_light = self.get_double_sersic_kwargs(is_shperical=True,
                                                     params=params)
        e11, e12, e21, e22 = self.get_light_ellipticity_parameters(
            params=params[[3, 4, 8]])

        light_model = LightModel(['SERSIC', 'SERSIC'])
        # x, y = np.meshgrid(np.arange(-5, 5, 0.01), np.arange(-5, 5, 0.01))
        # kwargs_light = self.get_double_sersic_kwargs(is_shperical=True)
        # model_image = light_model.surface_brightness(x, y, kwargs_light, )

        for i in range(2):
            kwargs_light[i]['center_x'] = 0
            kwargs_light[i]['center_y'] = 0

        n = 300

        if common_ellipticity:
            # rs = np.logspace(-2.5, 2, n) * kwargs_light[0]['R_sersic']
            rs = np.logspace(np.log10(self._light_profile_min),
                             np.log10(self._light_profile_max),
                             n)
            # taking profile along x-axis, but doesn't matter because light
            # has been set to spherical
            flux_r = light_model.surface_brightness(rs, 0 * rs,
                                                    kwargs_light)

            mge_fit = mge_fit_1d(rs, flux_r, ngauss=20, quiet=True)

            mge = (mge_fit.sol[0], mge_fit.sol[1])

            sigma_lum = mge[1]
            surf_lum = mge[0] / (np.sqrt(2 * np.pi) * sigma_lum)

            _, q_1 = ellipticity2phi_q(e11, e12)
            _, q_2 = ellipticity2phi_q(e21, e22)

            qobs_lum = np.ones_like(sigma_lum) * q_1
        else:
            rs_1 = np.logspace(-2.5, 2, n) * kwargs_light[0]['R_sersic']
            rs_2 = np.logspace(-2.5, 2, n) * kwargs_light[1]['R_sersic']
            flux_r_1 = light_model.surface_brightness(rs_1, 0 * rs_1,
                                                      kwargs_light, k=0)
            flux_r_2 = light_model.surface_brightness(rs_2, 0 * rs_2,
                                                      kwargs_light, k=1)

            mge_fit_1 = mge_fit_1d(rs_1, flux_r_1, ngauss=self._n_gauss,
                                   quiet=True)
            mge_fit_2 = mge_fit_1d(rs_2, flux_r_2, ngauss=self._n_gauss,
                                   quiet=True)

            mge_1 = (mge_fit_1.sol[0], mge_fit_1.sol[1])
            mge_2 = (mge_fit_2.sol[0], mge_fit_2.sol[1])

            sigma_lum = np.append(mge_1[1], mge_2[1])
            surf_lum = np.append(mge_1[0], mge_2[0]) / (
                    np.sqrt(2 * np.pi) * sigma_lum)

            _, q_1 = ellipticity2phi_q(e11, e12)
            _, q_2 = ellipticity2phi_q(e21, e22)

            qobs_lum = np.append(np.ones_like(mge_1[1]) * q_1,
                                 np.ones_like(mge_2[1]) * q_2)

        if is_spherical:
            qobs_lum = np.ones_like(qobs_lum)
        elif set_q is not None:
            qobs_lum = np.ones_like(qobs_lum) * set_q

        sorted_indices = np.argsort(sigma_lum)
        sigma_lum = sigma_lum[sorted_indices]
        surf_lum = surf_lum[sorted_indices]
        qobs_lum = qobs_lum[sorted_indices]

        # jampy wants sigmas along semi-major axis, lenstronomy definition
        # is along the intermediate axis, so dividing by sqrt(q)
        sigma_lum /= np.sqrt(qobs_lum)

        return surf_lum, sigma_lum, qobs_lum, params[3]

    def get_mass_mge(self, lens_params, lambda_int=1,
                     kappa_ext=0,
                     is_spherical=False,
                     plot=False,
                     r_c_mult=3.
                     ):
        """
        Get MGE of the mass profile
        :param lens_params: array, lens parameters
        :param lambda_int: float, MST parameter
        :param kappa_ext: float, external convergence
        :param is_spherical: bool, if True, will return kwargs for spherical case
        :param plot: bool, if True, will plot the MGE fitting for testing
        :param r_c_mult: float, multiple of the Einstein radius to use as
        the core radius
        :return: surf_mass, sigma_mass, qobs_mass
        """
        # core radius for approximate MST
        r_c = r_c_mult * 1.64

        if self.mass_model == 'powerlaw':
            theta_e, gamma, q = lens_params

            n = 300
            r_array = np.logspace(np.log10(self._mass_profile_min),
                                  np.log10(self._mass_profile_max),
                                  n)

            lens_model = LensModel(['PEMD'])

            # take spherical mass profile for mge_1d
            kwargs_lens = [
                {'theta_E': theta_e, 'gamma': gamma, 'e1': 0., 'e2': 0.,
                 'center_x': 0., 'center_y': 0.}]

            mass_r = lens_model.kappa(r_array, r_array * 0, kwargs_lens)

            mass_r = (1 - kappa_ext) * (lambda_int * mass_r
                     + (1. - lambda_int) * r_c ** 2 / (r_c**2 + r_array**2))

            # amps, sigmas, _ = mge.mge_1d(r_array, mass_r, N=20)
            mass_mge = mge_fit_1d(r_array, mass_r, ngauss=self._n_gauss,
                                  quiet=True,
                                  plot=False)
            sigmas = mass_mge.sol[1]
            amps = mass_mge.sol[0] / (np.sqrt(2 * np.pi) * sigmas)

            qobs_pot = np.ones_like(sigmas) * q
        else:
            kappa_s, r_s, mass_to_light, q_pot = lens_params
            q_nfw = self.interp_nfw_q(q_pot)

            n = 300
            r_array = np.logspace(np.log10(self._mass_profile_min),
                                  np.log10(self._mass_profile_max),
                                  n)

            lens_model_nfw = LensModel(['NFW'])
            lens_model_baryon = LensModel(['NIE', 'NIE'])

            alpha_rs = 4 * kappa_s * r_s * (np.log(0.5) + 1.)

            kwargs_nfw = [{
                'Rs': r_s, 'alpha_Rs': alpha_rs, 'center_x': 0.,
                'center_y': 0.
            }]

            q_1 = 0.882587
            q_2 = 0.847040
            kwargs_baryon_1 = [
                {'theta_E': mass_to_light * 5.409 * q_1,
                 's_scale': 2.031239,
                 'e1': 0., 'e2': 0., 'center_x': 0., 'center_y': 0.},
                {'theta_E': -mass_to_light * 5.409 * q_1,
                 's_scale': 2.472729,
                 'e1': 0., 'e2': 0., 'center_x': 0., 'center_y': 0.},
            ]

            kwargs_baryon_2 = [
                {'theta_E': mass_to_light * 1.26192 * q_2,
                 's_scale': 0.063157,
                 'e1': 0., 'e2': 0., 'center_x': 0., 'center_y': 0.},
                {'theta_E': -mass_to_light * 1.26192 * q_2,
                 's_scale': 0.667333,
                 'e1': 0., 'e2': 0., 'center_x': 0., 'center_y': 0.},
            ]

            mass_nfw = lens_model_nfw.kappa(r_array, r_array * 0, kwargs_nfw)

            piemd_1 = 1 / np.sqrt(r_array ** 2 + 2.031239 ** 2 * 4 * q_1 ** 2 /
                                  (1 + q_1) ** 2)
            piemd_2 = 1 / np.sqrt(r_array ** 2 + 2.472729 ** 2 * 4 * q_1 ** 2 /
                                  (1 + q_1) ** 2)
            piemd_3 = 1 / np.sqrt(r_array ** 2 + 0.063157 ** 2 * 4 * q_2 ** 2 /
                                  (1 + q_2) ** 2)
            piemd_4 = 1 / np.sqrt(r_array ** 2 + 0.667333 ** 2 * 4 * q_2 ** 2 /
                                  (1 + q_2) ** 2)
            mass_baryon_1 = mass_to_light * 5.409 / (1 + q_1) * (q_1) * \
                            (piemd_1 - piemd_2)
            mass_baryon_2 = mass_to_light * 1.26192 / (1 + q_2) * (q_2) \
                            * (piemd_3 - piemd_4)
            # mass_baryon_1 = lens_model_baryon.kappa(r_array, r_array*0,
            #                                         kwargs_baryon_1)
            # mass_baryon_2 = lens_model_baryon.kappa(r_array, r_array*0,
            #                                         kwargs_baryon_2)

            # amps, sigmas, _ = mge.mge_1d(r_array, mass_r, N=20)

            mass_nfw = lambda_int * mass_nfw + (1. - lambda_int) * r_c ** 2 / (
                    r_c**2 + r_array**2)
            mass_nfw = (1 - kappa_ext) * mass_nfw

            mass_baryon_1 = lambda_int * mass_baryon_1
            mass_baryon_1 = (1 - kappa_ext) * mass_baryon_1

            mass_baryon_2 = lambda_int * mass_baryon_2
            mass_baryon_2 = (1 - kappa_ext) * mass_baryon_2

            mass_mge_nfw = mge_fit_1d(r_array, mass_nfw, ngauss=self._n_gauss,
                                      quiet=True,
                                      plot=False)
            mass_mge_baryon_1 = mge_fit_1d(r_array, mass_baryon_1,
                                           ngauss=self._n_gauss,
                                           quiet=True,
                                           plot=False)
            mass_mge_baryon_2 = mge_fit_1d(r_array, mass_baryon_2,
                                           ngauss=self._n_gauss,
                                           quiet=True,
                                           plot=False)
            sigmas = np.concatenate((mass_mge_nfw.sol[1],
                                     mass_mge_baryon_1.sol[1],
                                     mass_mge_baryon_2.sol[1]))
            amps = np.concatenate((mass_mge_nfw.sol[0],
                                   mass_mge_baryon_1.sol[0],
                                   mass_mge_baryon_2.sol[0]
                                   )) / (np.sqrt(2 * np.pi) * sigmas)

            qobs_pot = np.concatenate((
                np.ones_like(mass_mge_nfw.sol[0]) * q_nfw,
                np.ones_like(mass_mge_baryon_1.sol[0]) * 0.882587,
                np.ones_like(mass_mge_baryon_2.sol[0]) * 0.847040))

        if plot:
            plt.figure()
            if self.mass_model == 'composite':
                mass_r = mass_nfw + mass_baryon_1 + mass_baryon_2
                plt.loglog(r_array, mass_nfw, 'b--', label='NFW')
                mge = 0.
                for a, s in zip(mass_mge_nfw.sol[0], mass_mge_nfw.sol[1]):
                    mge += a / np.sqrt(2*np.pi*s**2) * np.exp(
                        -r_array ** 2 / (2 * s ** 2))
                plt.loglog(r_array, mge, 'r--', label='NFW MGE')

            plt.loglog(r_array, mass_r, 'b', label='total')
            mge = 0
            for a, s in zip(amps, sigmas):
                mge += a * np.exp(
                    -r_array ** 2 / (2 * s ** 2))
            plt.loglog(r_array, mge, 'r', label='total MGE')
            plt.legend()
            plt.show()

        surf_pot = amps  # lens_cosmo.kappa2proj_mass(amps) / 1e12 # M_sun/pc^2
        sigma_pot = sigmas

        if is_spherical:
            qobs_pot = np.ones_like(sigmas)

        # jampy wants sigmas along semi-major axis, lenstronomy definition
        # is along the intermediate axis, so dividing by sqrt(q)
        sigma_pot /= np.sqrt(qobs_pot)
        return surf_pot, sigma_pot, qobs_pot

    def get_anisotropy_bs(self, params, surf_lum, sigma_lum,
                          model='Osipkov-Merritt',
                          plot=False):
        """
        Calculate the anisotropy b values for a given set of parameters
        :param params: list of anisotropy parameters
        :param surf_lum: surface brightness amplitudes
        :param sigma_lum: surface brightness sigmas
        :param model: anisotropy model
        :param plot: if True, plot the anisotropy profile
        :return: b values
        """
        if model == 'Osipkov-Merritt':
            betas = sigma_lum ** 2 / (
                    (params * self.R_EFF) ** 2 + sigma_lum ** 2)
        elif model == 'generalized-OM':
            betas = params[1] * sigma_lum ** 2 / (
                    (params[0] * self.R_EFF) ** 2 +
                    sigma_lum ** 2)
        elif model == 'constant':
            betas = (1 - params ** 2) * np.ones_like(sigma_lum)
        elif model == 'step':
            divider = self.R_EFF  # arcsec
            betas = (1 - params[0] ** 2) * np.ones_like(sigma_lum)
            # betas[sigma_lum <= divider] = 1 - params[0]**2
            betas[sigma_lum > divider] = 1 - params[1] ** 2
        elif model == 'free_step':
            divider = params[2]  # arcsec
            betas = (1 - params[0] ** 2) * np.ones_like(sigma_lum)
            # betas[sigma_lum <= divider] = 1 - params[0]**2
            betas[sigma_lum > divider] = 1 - params[1] ** 2
        else:
            betas = sigma_lum * 0.  # isotropic

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

        return bs  # betas #/ surf_lum * np.sum(surf_lum)

    @staticmethod
    def transform_pix_coords(xs, ys, x_center, y_center, angle):
        """
        Transform pixel coordinates to a new coordinate system
        :param xs: x coordinates
        :param ys: y coordinates
        :param x_center: x coordinate of the center of the new coordinate system
        :param y_center: y coordinate of the center of the new coordinate system
        :param angle: angle of rotation of new coordinate system
        :return: transformed x and y coordinates
        """
        xs_ = xs - x_center
        ys_ = ys - y_center
        angle *= np.pi / 180.

        xs_rotated = xs_ * np.cos(angle) + ys_ * np.sin(angle)
        ys_rotated = -xs_ * np.sin(angle) + ys_ * np.cos(angle)

        return xs_rotated, ys_rotated

    def get_supersampled_grid(self, supersampling_factor=1):
        """
        Get a supersampled grid for KCWI coordinates
        :param supersampling_factor: supersampling factor
        :return: x and y coordinates of the supersampled grid
        """
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

        n_x, n_y = self.X_GRID.shape
        xs = np.linspace(x_start, x_end, n_x * supersampling_factor)
        ys = np.linspace(y_start, y_end, n_y * supersampling_factor)

        x_grid_supersampled, y_grid_supersmapled = np.meshgrid(xs, ys)

        return x_grid_supersampled, y_grid_supersmapled

    def get_jam_grid(self, phi=0., supersampling_factor=1):
        """
        Get supersampled grid that is aligned with major and minor axes of
        the galaxy for JAM
        :param phi: angle of rotation of the major axis
        :param supersampling_factor: supersampling factor
        :return: flattened x and y coordinates of the JAM grid aligned with
        major axis, flattened x and y coordinates of the non-rotated
        supersampled grid
        """
        x_grid_supersampled, y_grid_supersmapled = \
            self.get_supersampled_grid(
                supersampling_factor=supersampling_factor)

        x_grid, y_grid = self.transform_pix_coords(x_grid_supersampled,
                                                   y_grid_supersmapled,
                                                   self.X_CENTER,
                                                   self.Y_CENTER, phi
                                                   )

        return x_grid.flatten(), y_grid.flatten(), \
               x_grid_supersampled.flatten(), y_grid_supersmapled.flatten()

    def get_surface_brightness_image(self, surf_lum, sigma_lum, qobs_lum,
                                     x_grid,
                                     y_grid,
                                     phi=None,
                                     x_center=None,
                                     y_center=None,
                                     ):
        """
        Get surface brightness image corrsponding to the given MGE parameters
        :param surf_lum: surface brightness of the MGE components
        :param sigma_lum: sigma of the MGE components
        :param qobs_lum: axis ratio of the MGE components
        :param x_grid: x coordinates of the grid
        :param y_grid: y coordinates of the grid
        :param phi: angle of rotation of the major axis
        :x_center: x coordinate of the center of the grid
        :y_center: y coordinate of the center of the grid
        :return: surface brightness image
        """
        if phi is None:
            phi = 90. - self.LIGHT_PROFILE_MEAN[3]
        if x_center is None:
            x_center = self.X_CENTER
        if y_center is None:
            y_center = self.Y_CENTER

        e1, e2 = phi_q2_ellipticity(phi / 180. * np.pi, np.mean(qobs_lum))
        kwargs_light = [
            {'amp': surf_lum,
             'sigma': sigma_lum,
             'e1': e1,
             'e2': e2,
             'center_x': x_center,
             'center_y': y_center
             }
        ]

        light_model = LightModel(['MULTI_GAUSSIAN_ELLIPSE'])
        model_image = light_model.surface_brightness(x_grid, y_grid,
                                                     kwargs_light)

        return model_image

    def compute_jampy_v_rms_model(self, lens_params,
                                  cosmo_params,
                                  ani_param,
                                  inclination=90,
                                  anisotropy_model='Oskipkov-Merritt',
                                  do_convolve=True,
                                  supersampling_factor=5,
                                  voronoi_bins=None,
                                  om_r_scale=None,
                                  aperture_type='ifu',
                                  is_spherical=False,
                                  q_light=None,
                                  moment='zz',
                                  x_points=None, y_points=None,
                                  shape='oblate',
                                  ):
        """
        Compute the jampy sigma_los for the given lens parameters for deault
        `moment='zz'`. Can be used to compute other moments as well.
        :param lens_params: lens mass model parameters, [theta_E, gamma,
        q] for power law; [kappa_s, r_s, m2l, q]  for composite
        :param cosmo_params: cosmology parameters, [lambda, D_dt/Mpc, D_d/Mpc]
        :param ani_param: anisotropy parameters
        :param inclination: inclination, degrees
        :param anisotropy_model: anisotropy model
        :param do_convolve: convolve with the PSF
        :param supersampling_factor: supersampling factor
        :param voronoi_bins: Voronoi binning map
        :param om_r_scale: scale of the Oskipkov-Merritt anisotropy
        :param aperture_type: aperture type, 'ifu' or 'single_slit'
        :param is_spherical: if True, will override any q for mass or light
        :param q_light: if provided, will override the axis ratio of the light
        :param moment: moment to compute, 'zz' for LOS velocity dispersion,
        'z' for LOS velocity
        :param x_points: to get kinematic profile directly
        :param y_points: to get kinematic profile directly
        :param shape: axisymmetric shape of the galaxy, 'oblate' or 'prolate'
        :return: velocity moment map, surface brightness map
        """
        surf_lum, sigma_lum, qobs_lum, pa = self.get_light_mge(
            is_spherical=is_spherical, set_q=q_light)

        lambda_int, kappa_ext, D_dt, D_d = cosmo_params

        # surf_pot is in dimension of convergence
        surf_pot, sigma_pot, qobs_pot = self.get_mass_mge(
            lens_params, lambda_int, kappa_ext,
            is_spherical=is_spherical)

        c2_over_4piG = 1.6624541593797972e+6
        # the passed D_dt is true D_dt, so not needed to divide by
        # lambda_int
        sigma_crit = c2_over_4piG * D_dt / D_d ** 2 / (1 + self.Z_L)
        surf_pot *= sigma_crit  # from convergence to M_sun / pc^2

        bs = self.get_anisotropy_bs(ani_param, surf_lum, sigma_lum,
                                    model=anisotropy_model
                                    )
        phi = 90. - pa  # /np.pi*180.
        if inclination > 90.:
            inc = 180. - inclination
        else:
            inc = inclination

        # these grids are in the rotated frame that is passed to jampy
        x_grid_spaxel, y_grid_spaxel, _, _ = self.get_jam_grid(
            phi, supersampling_factor=1)

        # thses supersampled grids are used to compute the surface brightness
        x_grid, y_grid, x_grid_original, y_grid_original = self.get_jam_grid(
            phi, supersampling_factor=supersampling_factor)

        if do_convolve:
            sigma_psf = self.PSF_FWHM / 2.355
        else:
            sigma_psf = 0.
        norm_psf = 1.

        mbh = 0
        distance = D_d  # self.lens_cosmo.dd

        if aperture_type == 'single_slit':
            pixel_size = 0.01
            phi = -90.
            supersampling_factor = 1

            x_grid_spaxel, y_grid_spaxel = np.meshgrid(
                np.arange(-.695, .7, 0.01),
                np.arange(-.7, .7, 0.01))

            voronoi_bins = np.zeros_like(x_grid_spaxel)
            voronoi_bins[30:-29, 35:-35] = 1
            voronoi_bins -= 1
            # voronoi_bins = None

            x_grid, y_grid = self.transform_pix_coords(x_grid_spaxel,
                                                       y_grid_spaxel,
                                                       0,
                                                       0, phi
                                                       )

            _x_grid_spaxel = x_grid.flatten()
            _y_grid_spaxel = y_grid.flatten()
        else:
            pixel_size = self.PIXEL_SIZE
            _x_grid_spaxel = x_grid_spaxel
            _y_grid_spaxel = y_grid_spaxel

        if x_points is not None:
            if is_spherical and moment != 'zz':
                raise ValueError
            voronoi_bins = None
            _x_grid_spaxel = x_points
            _y_grid_spaxel = y_points
            pixel_size = np.sqrt((x_points[1] - x_points[0]) ** 2
                                 + (y_points[1] - y_points[0]) ** 2) / 2.
            phi = np.arctan2(y_points[1] - y_points[0],
                             x_points[1] - x_points[0]) * 180. / np.pi

        if is_spherical:
            jam = jam_sph_rms(
                surf_lum, sigma_lum,  # qobs_lum,
                surf_pot, sigma_pot,  # qobs_pot,
                # intrinsic_q,
                mbh, distance,
                np.sqrt(_x_grid_spaxel ** 2 + _y_grid_spaxel ** 2),
                plot=False, pixsize=pixel_size,
                # self.PIXEL_SIZE/supersampling_factor,
                # pixang=phi,
                quiet=1,
                sigmapsf=sigma_psf, normpsf=norm_psf,
                # moment='zz',
                # goodbins=goodbins,
                # align='sph',
                beta=bs if om_r_scale is None else None,
                scale=(om_r_scale * self.R_EFF) if om_r_scale is not None
                else None,
                # data=rms, errors=erms,
                ml=1
            )[0]
        else:
            jam = jam_axi_proj(
                surf_lum, sigma_lum, qobs_lum,
                surf_pot, sigma_pot, qobs_pot,
                inc, mbh, distance,
                _x_grid_spaxel, _y_grid_spaxel,
                plot=False, pixsize=pixel_size,
                # self.PIXEL_SIZE/supersampling_factor,
                pixang=phi, quiet=1,
                sigmapsf=sigma_psf, normpsf=norm_psf,
                moment=moment,
                # goodbins=goodbins,
                align='sph',
                beta=bs,
                # data=rms, errors=erms,
                ml=1, shape=shape).model

        # if aperture_type == 'single_slit':
        #    return jam[0], None

        if x_points is not None:
            # for debugging and plotting profiles
            return jam

        num_pix = int(np.sqrt(len(x_grid)))
        num_pix_spaxel = int(np.sqrt(len(_x_grid_spaxel)))
        vel_dis_model = jam.reshape((num_pix_spaxel, num_pix_spaxel))

        if aperture_type == 'single_slit':
            flux = self.get_surface_brightness_image(
                surf_lum * (2 * np.pi * sigma_lum ** 2),
                sigma_lum, qobs_lum,
                x_grid_spaxel,
                y_grid_spaxel, phi=phi, x_center=0, y_center=0).reshape((
                num_pix_spaxel,
                num_pix_spaxel))
        else:
            flux = self.get_surface_brightness_image(
                surf_lum * (2 * np.pi * sigma_lum ** 2),
                sigma_lum, qobs_lum,
                x_grid_original,
                y_grid_original).reshape((num_pix, num_pix))

        if do_convolve:
            sigma = self.PSF_FWHM / 2.355 / self.PIXEL_SIZE * supersampling_factor
            kernel = Gaussian2DKernel(x_stddev=sigma,
                                      x_size=6*int(sigma)+1,
                                      y_size=6*int(sigma)+1)

            convolved_flux = convolve(flux, kernel)
            # convolved_map = convolve(flux * vel_dis_map ** 2, kernel)
            # convolved_flux_spaxels = convolved_flux.reshape(
            #         len(self.X_GRID), supersampling_factor,
            #         len(self.Y_GRID), supersampling_factor
            #     ).sum(3).sum(1)
            # convolved_map = convolved_flux_spaxels * vel_dis_model ** 2
        else:
            convolved_flux = flux

        if aperture_type == 'ifu':
            convolved_flux_spaxel = convolved_flux.reshape(
                len(self.X_GRID), supersampling_factor,
                len(self.Y_GRID), supersampling_factor
            ).sum(3).sum(1)
        else:
            convolved_flux_spaxel = convolved_flux.reshape(
                int(np.sqrt(len(_x_grid_spaxel))), supersampling_factor,
                int(np.sqrt(len(_x_grid_spaxel))), supersampling_factor
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
            vel_dis_map, intensity_map = self.bin_map_in_voronoi_bins(
                vel_dis_model,
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

        if aperture_type == 'ifu':
            return vel_dis_map, intensity_map
        else:
            return vel_dis_map[0], intensity_map[0]

    @staticmethod
    def bin_map_in_voronoi_bins(vel_dis_map, IR_map, voronoi_bins):
        """
        Bin a map in Voronoi bins
        :param vel_dis_map: velocity dispersion map
        :param IR_map: surface brightness map
        :param voronoi_bins: Voronoi bins
        :return: binned velocity dispersion map and binned surface brightness map
        """
        n_bins = int(np.nanmax(voronoi_bins)) + 1

        binned_map = np.zeros(n_bins)

        binned_IR = np.zeros(n_bins)

        for n in range(n_bins):
            binned_map[n] = np.sum(
                (vel_dis_map[voronoi_bins == n]) ** 2 * IR_map[
                    voronoi_bins == n]
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
                                    bs=None,
                                    anisotropy_model='constant',
                                    alignment='sph',
                                    **kwargs
                                    ):
        """
        Retrieve the anisotropy profile that is effectively applied in the
        3D case
        :param Rs: radii
        :param ani_param: anisotropy parameter
        :param surf_lum: surface brightness MGE amplitude
        :param sigma_lum: surface brightness MGE sigma
        :param surf_pot: mass MGE amplitude
        :param sigma_pot: mass MGE sigma
        :param print_step: print step
        :param anisotropy_model: anisotropy model
        :param alignment: alignment, 'sph' or 'ell'
        :param kwargs: keyword arguments
        :return: anisotropy profile
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
            # intrinsic_q,
            mbh=mbh,
            # distance,
            Rbin=Rs, zbin=Rs * 0,
            plot=False,  # pixsize=0.,
            # self.PIXEL_SIZE/supersampling_factor,
            # pixang=0., quiet=1,
            # sigmapsf=0., normpsf=norm_psf,
            # moment=moment,
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
        Get the kinematics_api object from lenstronomy's galkin
        :param anisotropy_model: anisotropy model
        :param single_slit: single slit or not
        :return: kinematics_api object
        """
        if anisotropy_model == 'Osipkov-Merritt':
            anisotropy_type = 'OM'  # anisotropy_model model applied
        elif anisotropy_model == 'constant':
            anisotropy_type = 'const'
        else:
            raise ValueError('anisotropy_model model {} model not '
                             'supported!'.format(anisotropy_model))

        if single_slit:
            kwargs_aperture = {'aperture_type': 'slit',
                               'length': 0.7,
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
                               # 'angle': 0
                               }
            kwargs_seeing = {'psf_type': 'GAUSSIAN',
                             'fwhm': self.PSF_FWHM,
                             # 'moffat_beta': self.MOFFAT_BETA[n]
                             }

        # if self._do_mge_light:
        #     light_model_bool = [True, True]
        # else:
        #     light_model_bool = [True]
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
                                       analytic_kinematics=False,
                                       # (not self._do_mge_light),
                                       Hernquist_approx=False,
                                       MGE_light=False,  # self._do_mge_light,
                                       MGE_mass=False,  # self._do_mge_light,
                                       kwargs_mge_light=None,
                                       kwargs_mge_mass=None,
                                       sampling_number=1000,
                                       num_kin_sampling=2000,
                                       num_psf_sampling=500,
                                       )

        return kinematics_api

    def compute_galkin_v_rms_model(self, kinematics_api,
                                   lens_params,
                                   ani_param,
                                   r_eff_multiplier=1,
                                   anisotropy_model='Osipkov-Merritt',
                                   aperture_type='ifu',
                                   voronoi_bins=None,
                                   supersampling_factor=5,
                                   ):
        """
        Compute the galkin v_rms model
        :param kinematics_api: kinematics_api object
        :param lens_params: lens parameters, for power-law [theta_E, gamma,
        q], for composite [kappa, r_s, m2l, q]
        :param ani_param: anisotropy parameter
        :param r_eff_multiplier: effective radius multiplier
        :param anisotropy_model: anisotropy model
        :param aperture_type: aperture type, 'ifu' or 'single_slit'
        :param voronoi_bins: Voronoi binning map
        :param supersampling_factor: supersampling factor
        :return: v_rms model
        """
        if self.mass_model == 'powerlaw':
            theta_E, gamma, _ = lens_params
        else:
            raise ValueError('Other mass model is not supported!')

        kwargs_lens, kwargs_lens_light, kwargs_anisotropy = self.get_lenstronomy_kwargs(
            theta_E, gamma,
            ani_param,
            r_eff_multiplier=r_eff_multiplier,
            anisotropy_model=anisotropy_model
        )

        if aperture_type == 'single_slit':
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
        Compute v_rms profile as a function of radius using Galkin's numerical
        approach
        :param rs: radius array
        :param theta_E: Einstein radius
        :param gamma: power-law slope
        :param ani_param: anisotropy parameter
        :param anisotropy_model: anisotropy model
        :return: v_rms array
        """
        bins = np.zeros(len(rs) + 1)
        bins[:-1] = rs - (rs[1] - rs[0]) / 2.
        bins[-1] = rs[-1] + (rs[1] - rs[0]) / 2.

        kwargs_aperture = {'aperture_type': 'IFU_shells',
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
            anisotropy_type = 'OM'  # anisotropy_model model applied
        elif anisotropy_model == 'constant':
            anisotropy_type = 'const'
        else:
            raise ValueError('anisotropy_model model {} model not '
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
            self.get_lenstronomy_kwargs(theta_E, gamma, ani_param,
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

    def get_lenstronomy_kwargs(self, theta_E, gamma,
                               ani_param,
                               anisotropy_model='Osipkov-Merritt'
                               ):
        """
        Get lenstronomy kwargs for lens model, lens light model and anisotropy
        :param theta_E: Einstein radius
        :param gamma: power-law slope
        :param ani_param: anisotropy parameter
        :param anisotropy_model: anisotropy model
        :return: kwargs_lens, kwargs_lens_light, kwargs_anisotropy
        """
        kwargs_lens = [{'theta_E': theta_E,
                        'gamma': gamma,
                        'e1': 0., 'e2': 0.,
                        'center_x': self.X_CENTER,
                        'center_y': self.Y_CENTER
                        }]

        if self._do_mge_light:
            # kwargs_lens_light = self.get_double_sersic_kwargs(
            #     is_shperical=True)
            amp, sigma, _ = self.get_light_mge()
            kwargs_lens_light = [{
                'amp': amp * (2 * np.pi * sigma ** 2),
                'sigma': sigma,
                'center_x': self.X_CENTER,
                'center_y': self.Y_CENTER
            }]
        else:
            kwargs_lens_light = self.get_double_sersic_kwargs(
                is_shperical=True)
            # [
            # {'amp': 1.,
            #  'R_s': self.R_EFF * r_eff_multiplier / (1 + np.sqrt(2)),
            #  'center_x': self.X_CENTER,
            #  'center_y': self.Y_CENTER
            #  }]

        # set the anisotropy_model radius. r_eff is pre-computed half-light
        # radius of the lens light
        if anisotropy_model == 'Osipkov-Merritt':
            kwargs_anisotropy = {'r_ani': ani_param * self.R_EFF}
        elif anisotropy_model == 'constant':
            kwargs_anisotropy = {'beta': 1 - ani_param ** 2}
        else:
            raise ValueError(
                'anisotropy_model model {} not recognized!'.format(
                    anisotropy_model))

        return kwargs_lens, kwargs_lens_light, kwargs_anisotropy

    def sample_from_double_sersic_fit(self, mode='all'):
        """
        Sample one point from the mean and covariance of double Sersic
        profile fitting
        """
        sample = np.random.multivariate_normal(
            mean=self.LIGHT_PROFILE_MEAN,
            cov=self.LIGHT_PROFILE_COVARIANCE
        )

        if mode == 'all':
            return sample
        elif mode == 'pa':
            return sample[3]
        elif mode == 'ellipticities':
            return sample[[3, 4, 8]]
        else:
            raise ValueError('mode not recognized!')


# interpolation grid to get convergence ellipticity from potential
# ellipticity for the NFW profile
_nfw_qs = np.array(
    [0.2955890109246895, 0.29729484339412804, 0.2993046678215645,
     0.30148731960847175, 0.3037781629679952, 0.30656181091958756,
     0.30944101265858914, 0.3128449794390762, 0.31637187278135614,
     0.32057430074818316, 0.32525183117191303, 0.3304782178184469,
     0.33659904664300166, 0.34335555305827037, 0.3512799441223804,
     0.36060271442019076, 0.37271593485449106, 0.3853688300669107,
     0.39850632018999277, 0.4123670033003701, 0.426085510857797,
     0.4405323411004247, 0.4549318199526641, 0.4703884811354565,
     0.48553317127707907, 0.5009183370787333, 0.5168215047382049,
     0.5334143339607527, 0.5499198443785049, 0.5666119560282026,
     0.5839594642663198, 0.6018951710415525, 0.6198941913485861,
     0.6385840422577401, 0.6573510262452367, 0.6761701874779739,
     0.6958430091624018, 0.7161996832176144, 0.7364049180144907,
     0.7566471634820947, 0.7777133348551333, 0.7987235524003334,
     0.8198882232045268, 0.8422204547523987, 0.8646648092383161,
     0.8868192016046453, 0.9103444426772708, 0.9322329992014406,
     0.9567554192086308, 0.9796534205266368])
_potential_qs = np.linspace(0.4, 0.99, 50)
