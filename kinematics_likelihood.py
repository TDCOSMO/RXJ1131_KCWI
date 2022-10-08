import numpy as np
import matplotlib.pyplot as plt
import h5py as h5
from scipy.interpolate import interp1d
import pickle

from dynamical_model import DynamicalModel
from data_util import *


class KinematicLikelihood(object):

    def __init__(self,
                 lens_model_type='powerlaw',
                 software='jampy',
                 anisotropy_model='constant',
                 aperture='ifu',
                 snr_per_bin=15,
                 is_spherical=False,
                 mpi=False
                 ):
        """

        """
        self.lens_model_type = lens_model_type
        self.software = software
        self.anistropy_model = anisotropy_model
        self.aperture_type = aperture
        self.snr_per_bin = snr_per_bin
        self.is_spherical = is_spherical
        self._mpi = mpi

        self.voronoi_bin_mapping = self.load_voronoi_bin_map(self.snr_per_bin)

        self.velocity_dispersion_mean = None
        self.velocity_dispersion_covariance = None
        if self.aperture_type == 'single_slit':
            self.velocity_dispersion_mean = 323 # km/s
            self.velocity_dispersion_covariance = 20**2 # km/s
            self.velocity_dispersion_inverse_covariance = 1.\
                                        / self.velocity_dispersion_covariance
        else:
            self.load_velocity_dispersion_data_ifu(self.snr_per_bin)
            self.velocity_dispersion_inverse_covariance = np.linalg.inv(
                self.velocity_dispersion_covariance)

        self.lens_model_posterior_mean = None
        self.lens_model_posterior_covariance = None
        self.load_lens_model_posterior(self.lens_model_type)
        self.lens_model_posterior_inverse_covariance = np.linalg.inv(
                                        self.lens_model_posterior_covariance)

        self._intrinsic_q_prior_interp = None

        self.dynamical_model = DynamicalModel(
                mass_profile=self.lens_model_type,
                include_light_profile_uncertainty=True,
            )
        self.galkin_kinematics_api = self.dynamical_model.get_galkin_kinematics_api(
            anisotropy_model=self.anistropy_model,
            single_slit=(True if self.aperture_type == 'single_slit' else False)
        )

    def get_galkin_velocity_dispersion(self, theta_e, gamma, ani_param):
        """
        """
        self.dynamical_model.compute_galkin_v_rms_model(
            self.galkin_kinematics_api,
            theta_e, gamma, ani_param,
            anisotropy_model=self.anistropy_model,
            aperture_type=self.aperture_type,
            voronoi_bins=self.voronoi_bin_mapping
        )

    def get_lens_kwargs(self, theta_E, gamma, r_eff,
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

        # set the anisotropy_model radius. r_eff is pre-computed half-light
        # radius of the lens light
        if anisotropy_model == 'Osipkov-Merritt':
            kwargs_anisotropy = {'r_ani': ani_param * r_eff}
        elif anisotropy_model == 'constant':
            kwargs_anisotropy = {'beta': ani_param}

        return kwargs_lens, kwargs_lens_light, kwargs_anisotropy

    @staticmethod
    def get_mean_covariance_from_file(file_id):
        """

        """
        mean = np.loadtxt(file_id+'_mean.txt')
        covariance = np.loadtxt(file_id+'_covariance.txt')

        return mean, covariance

    def load_lens_model_posterior(self, lens_model_type):
        """

        """
        mean, covariance = self.get_mean_covariance_from_file(
            './data_products/lens_model_posterior_{}'.format(lens_model_type))
        self.lens_model_posterior_mean = mean
        self.lens_model_posterior_covariance = covariance

    def load_velocity_dispersion_data_ifu(self, snr_per_bin):
        """

        """
        mean, covariance = self.get_mean_covariance_from_file(
            './data_products/'
            'systematic_marginalized_velocity_dispersion'
            '_snr_per_bin_{}'.format(snr_per_bin))
        self.velocity_dispersion_mean = mean
        self.velocity_dispersion_covariance = covariance

    @staticmethod
    def load_voronoi_bin_map(snr_per_bin, plot=False):
        """
        """
        voronoi_bin_mapping = load_bin_mapping(snr_per_bin, plot=plot)
        # self.voronoi_bin_mapping[self.voronoi_bin_mapping < 0] = np.nan

        return voronoi_bin_mapping

    def intrinsic_q_prior(self, intrinsic_q):
        """

        """
        if self._intrinsic_q_prior_interp is None:
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

            self._intrinsic_q_prior_interp = interp1d(x, y, bounds_error=False,
                                                      fill_value=0.)
        else:
            pass

        return np.log(self._intrinsic_q_prior_interp(intrinsic_q))

    @staticmethod
    def normalized_delta_squared(delta, inverse_covariance):
        """

        """
        return -0.5 * (delta.T @ (inverse_covariance @ delta))

    def lens_model_likelihood(self, lens_params):
        """
        """
        return self.normalized_delta_squared(
            self.lens_model_posterior_mean - lens_params,
            self.lens_model_posterior_inverse_covariance
        )

    def kinematic_likelihood(self, v_rms):
        """
        """
        return self.normalized_delta_squared(
            v_rms - self.velocity_dispersion_mean,
            self.velocity_dispersion_inverse_covariance
        )

    def log_prior(self, params):
        """
        """
        if self.lens_model_type == 'powerlaw':
            theta_E, gamma, q, pa, D_dt, lamda, ani_param, inclination = params

            if not 1.0 < theta_E < 2.2:
                return -np.inf

            if not 1.5 < gamma < 2.5:
                return -np.inf

            lens_model_params = np.array([theta_E, gamma, q, pa, D_dt])
        else:
            raise NotImplementedError

        if not 0.5 <= q <= 1.:
            return -np.inf

        if not 70 < pa < 170:
            return -np.inf

        if inclination > 90:
            inclination = 180 - inclination

        intrinsic_q = np.sqrt(q**2 - np.cos(inclination*np.pi/180.)**2)
        intrinsic_q_lum = np.sqrt(self.dynamical_model.q_light_2()**2 -
                                  np.cos(inclination * np.pi / 180.) ** 2)
        if np.isinf(intrinsic_q) or np.isnan(intrinsic_q) or intrinsic_q**2\
                < 0.05:
            return -np.inf

        if np.isinf(intrinsic_q_lum) or np.isnan(intrinsic_q_lum) or \
                intrinsic_q_lum**2 < 0.05:
            return -np.inf

        return self.lens_model_likelihood(lens_model_params) + \
               self.intrinsic_q_prior(intrinsic_q)

    def log_likelihood(self, params):
        """
        """
        if self.lens_model_type == 'powerlaw':
            theta_E, gamma, q, pa, D_dt, lamda, ani_param, inclination = params

            if self.software == 'jampy':
                v_rms, _ = self.dynamical_model.compute_jampy_v_rms_model(
                    theta_E, gamma, ani_param, q, pa, inclination,
                    anisotropy_model=self.anistropy_model,
                    voronoi_bins=self.voronoi_bin_mapping,
                )

                return self.kinematic_likelihood(np.sqrt(lamda) * v_rms)
        else:
            raise NotImplementedError

    def log_probability(self, params):
        """
        """
        log_prior = self.log_prior(params)

        if np.isinf(log_prior) or np.isnan(log_prior):
            return -np.inf
        else:
            log_likelihood = self.log_likelihood(params)

            if np.isinf(log_likelihood) or np.isnan(log_likelihood):
                return -np.inf

        return log_prior + log_likelihood