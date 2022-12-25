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
                 mpi=False,
                 shape='oblate'
                 ):
        """

        """
        self.lens_model_type = lens_model_type
        self.software = software
        self.anisotropy_model = anisotropy_model
        self.aperture_type = aperture
        self.snr_per_bin = snr_per_bin
        self.is_spherical = is_spherical
        self._mpi = mpi
        self.shape = shape
        self.voronoi_bin_mapping = self.load_voronoi_bin_map(self.snr_per_bin)

        self.velocity_dispersion_mean = None
        self.velocity_dispersion_covariance = None
        if self.aperture_type == 'single_slit':
            self.velocity_dispersion_mean = 293 #323 # km/s
            self.velocity_dispersion_covariance = (20 * 293/323)**2 # km/s
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
                mass_model=self.lens_model_type,
                include_light_profile_uncertainty=True,
            )
        if self.software == 'galkin':
            self.galkin_kinematics_api = self.dynamical_model.get_galkin_kinematics_api(
                anisotropy_model=self.anisotropy_model,
                single_slit=(True if self.aperture_type == 'single_slit' else
                             False)
            )
        else:
            self.galkin_kinematics_api = None

    def get_galkin_velocity_dispersion(self, theta_e, gamma, ani_param):
        """
        """
        self.dynamical_model.compute_galkin_v_rms_model(
            self.galkin_kinematics_api,
            theta_e, gamma, ani_param,
            anisotropy_model=self.anisotropy_model,
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

    def get_intrinsic_q_prior(self, intrinsic_q):
        """

        """
        if self._intrinsic_q_prior_interp is None:
            if self.shape == 'oblate':
                scrapped_points = np.array([
                    0.16240221274286662, 0.0,
                    0.2029264579533636, 0.1911580724417048,
                    0.2407752136623306, 0.39252068793994876,
                    0.2777045053735887, 0.5686809459927957,
                    0.30559742550080704, 0.6692868878404652,
                    0.34066141115113935, 0.7656308879610525,
                    0.3775002637806548, 0.8409515698717298,
                    0.41704475227228144, 0.9330562380356633,
                    0.45573006948736117, 1.0671849328489817,
                    0.48546945420014154, 1.2265988875992946,
                    0.5152389852735029, 1.4196260343969964,
                    0.5468549809324269, 1.671461194097342,
                    0.5712697647076556, 1.8939451034773822,
                    0.5938493887825392, 2.070225946972551,
                    0.6155246220400041, 2.2381110290460176,
                    0.63809294122967, 2.4017869255234157,
                    0.6606085042883197, 2.5066397359178803,
                    0.6803958217144235, 2.5694986660235433,
                    0.7019467012344933, 2.598729330901526,
                    0.7252272281928764, 2.556516889498514,
                    0.7413103115626366, 2.4891548467811213,
                    0.7555846132975597, 2.405001281220324,
                    0.7698438418521923, 2.304041119635831,
                    0.7805231900878766, 2.2115144024237665,
                    0.7894012932788689, 2.110599460380145,
                    0.7991687141069892, 2.00127368373453,
                    0.815131212034427, 1.7994588728275769,
                    0.8364145426043439, 1.530372458284973,
                    0.8567633359962618, 1.2192770902732764,
                    0.8833298162579323, 0.8409025820357834,
                    0.9063239527907992, 0.4793648161825659,
                    0.9275922101804258, 0.19347180561626676,
                    0.9435697812881539, 0.0,
                ])
            else:
                scrapped_points = np.array([
                    0.16868396062885307, 0.010773555612499486,
                    0.2117706464887026, 0.052428289344770285,
                    0.2477164131860181, 0.13195815685151402,
                    0.28546342492802557, 0.21987624918981652,
                    0.3214204965105588, 0.31201106371433385,
                    0.3537826145938532, 0.39577272658758345,
                    0.3951321164252446, 0.5004672685889999,
                    0.4311155660732857, 0.6220136261549825,
                    0.46082103613041325, 0.7436127398519812,
                    0.4869503941636647, 0.877846946927332,
                    0.5067942360158569, 1.0037306121218514,
                    0.5284543960930316, 1.154809098171623,
                    0.5456152118535489, 1.2891186711484255,
                    0.5636879550216299, 1.4402273035587774,
                    0.5826726255972747, 1.6081349954026796,
                    0.6016648327630647, 1.7844459852584293,
                    0.6215689673364183, 1.977556034547729,
                    0.6423699561370454, 2.170658547246883,
                    0.6676627526641847, 2.372126675007159,
                    0.6902273035587777, 2.5316009224786327,
                    0.7181503700465761, 2.6658200563736933,
                    0.7576647121776222, 2.724311532490239,
                    0.795204467690638, 2.5811389295027345,
                    0.8210398987082287, 2.3876445141160323,
                    0.8432653030462894, 2.168970351054368,
                    0.8645410970260615, 1.891480638499916,
                    0.8822520838671751, 1.6392309663415872,
                    0.8990737530711602, 1.3953921287852515,
                    0.915865275914565, 1.1179400991815256,
                    0.9317750177109868, 0.8573022021916397,
                    0.9485590039642465, 0.5714468745760661,
                    0.9662398444447794, 0.28558401037034775,
                    0.9884049560616796, 0.0,
                ])

            x = scrapped_points[::2]
            y = scrapped_points[1::2]

            self._intrinsic_q_prior_interp = interp1d(x, y, bounds_error=False,
                                                      fill_value=0.)
        else:
            pass

        return np.log(self._intrinsic_q_prior_interp(intrinsic_q))

    @staticmethod
    def get_normalized_delta_squared(delta, inverse_covariance):
        """

        """
        return delta.T @ (inverse_covariance @ delta)

    def get_lens_model_likelihood(self, lens_params):
        """
        """
        return -0.5 * self.get_normalized_delta_squared(
            self.lens_model_posterior_mean - lens_params,
            self.lens_model_posterior_inverse_covariance
        )

    def get_kinematic_likelihood(self, v_rms):
        """
        """
        if self.aperture_type == 'ifu':
            return -0.5 * self.get_normalized_delta_squared(
                v_rms - self.velocity_dispersion_mean,
                self.velocity_dispersion_inverse_covariance
            )
        elif self.aperture_type == 'single_slit':
            return -0.5 * (v_rms - self.velocity_dispersion_mean)**2 \
                   * self.velocity_dispersion_inverse_covariance
        else:
            raise ValueError('Aperture type not recognized!')

    def get_anisotropy_prior(self, ani_param):
        """
        """
        if self.anisotropy_model == 'constant':
            if not 0.87 < ani_param < 1.12:
                return -np.inf
        elif self.anisotropy_model == 'Osipkov-Merritt':
            if not 0.1 < ani_param < 5.:
                return -np.inf
        elif self.anisotropy_model == 'step':
            if not 0.87 < ani_param[0] < 1.12:
                return -np.inf
            if not 0.87 < ani_param[1] < 1.12:
                return -np.inf
        elif self.anisotropy_model == 'free_step':
            if not 0.87 < ani_param[0] < 1.12:
                return -np.inf
            if not 0.87 < ani_param[1] < 1.12:
                return -np.inf
            if not 0.5*1.91 < ani_param[2] < 100.:
                return -np.inf

            # return -np.log(ani_param[2])

        return 0.

    def get_log_prior(self, params):
        """
        """
        if self.lens_model_type == 'powerlaw':
            theta_e, gamma, q, D_dt, inclination, lamda, *ani_param = params

            if not 1.0 < theta_e < 2.2:
                return -np.inf

            if not 1.5 < gamma < 2.5:
                return -np.inf

            lens_model_params = np.array([theta_e, gamma, q, D_dt])
        elif self.lens_model_type == 'composite':
            kappa_s, r_s, m2l, q, D_dt, inclination, lamda, *ani_param = \
                params
            lens_model_params = np.array([kappa_s, r_s, m2l, q, D_dt])
        else:
            raise NotImplementedError

        if len(ani_param) == 1:
            ani_param = ani_param[0]

        if not 0.5 < q < 0.99:
            return -np.inf

        # if not 70 < pa < 170:
        #     return -np.inf

        if inclination > 90:
            inclination = 180 - inclination

        if self.lens_model_type == 'powerlaw':
            intrinsic_q = np.sqrt(q**2 - np.cos(inclination*np.pi/180.)**2) \
                          / np.sin(inclination*np.pi/180.)
        elif self.lens_model_type == 'composite':
            q_nfw = self.dynamical_model.interp_nfw_q(q)
            intrinsic_q = np.sqrt(q_nfw**2 - np.cos(inclination*np.pi/180.)**2) \
                          / np.sin(inclination*np.pi/180.)
        else:
            raise ValueError("Lens model type {} not recognized!".format(
                self.lens_model_type))
        intrinsic_q_lum = np.sqrt(self.dynamical_model.q_light_2()**2 -
                                  np.cos(inclination * np.pi / 180.) ** 2) \
                          / np.sin(inclination*np.pi/180.)
        if np.isinf(intrinsic_q) or np.isnan(intrinsic_q) or intrinsic_q**2\
                < 0.1:
            return -np.inf

        if np.isinf(intrinsic_q_lum) or np.isnan(intrinsic_q_lum) or \
                intrinsic_q_lum**2 < 0.1:
            return -np.inf

        if not 0. < lamda < 2.:
            return -np.inf

        return self.get_anisotropy_prior(ani_param) + \
            self.get_lens_model_likelihood(lens_model_params) + \
            self.get_intrinsic_q_prior(intrinsic_q_lum)

    def get_v_rms(self, params):
        """
        """
        if self.lens_model_type == 'powerlaw':
            theta_e, gamma, q, D_dt, inclination, lamda, *ani_param = params
            lens_params = [theta_e, gamma, q]
        elif self.lens_model_type == 'composite':
            kappa_s, r_s, m2l, q, D_dt, inclination, lamda, *ani_param = \
                params
            lens_params = [kappa_s, r_s, m2l, q]
        else:
            raise ValueError('lens model type not recognized!')

        if len(ani_param) == 1:
            ani_param = ani_param[0]

        if self.software == 'jampy':
            v_rms, _ = self.dynamical_model.compute_jampy_v_rms_model(
                lens_params, ani_param,
                inclination,
                anisotropy_model=self.anisotropy_model,
                voronoi_bins=self.voronoi_bin_mapping,
                om_r_scale=ani_param if self.anisotropy_model ==
                'Osipkov-Merritt' else None,
                is_spherical=self.is_spherical,
                aperture_type=self.aperture_type,
                shape=self.shape
            )
        elif self.software == 'galkin':
            v_rms, _ = self.dynamical_model.compute_galkin_v_rms_model(
                self.galkin_kinematics_api,
                lens_params, ani_param,
                anisotropy_model=self.anisotropy_model,
                aperture_type=self.aperture_type,
                voronoi_bins=self.voronoi_bin_mapping,
                supersampling_factor=5,
            )
        else:
            raise ValueError('Software not recognized!')

        return np.sqrt(lamda) * v_rms

    def get_log_likelihood(self, params):
        """

        """
        v_rms = self.get_v_rms(params)
        return self.get_kinematic_likelihood(v_rms)

    def get_log_probability(self, params):
        """
        """
        log_prior = self.get_log_prior(params)

        if np.isinf(log_prior) or np.isnan(log_prior):
            return -np.inf
        else:
            log_likelihood = self.get_log_likelihood(params)

            if np.isinf(log_likelihood) or np.isnan(log_likelihood):
                return -np.inf

        return log_prior + log_likelihood

    def plot_residual(self, params):
        """
        """
        v_rms = self.get_v_rms(params)

        model_v_rms = get_kinematics_maps(v_rms, self.voronoi_bin_mapping)
        data_v_rms = get_kinematics_maps(self.velocity_dispersion_mean,
                                         self.voronoi_bin_mapping
                                         )
        noise_v_rms = get_kinematics_maps(
            np.sqrt(np.diag(self.velocity_dispersion_covariance)),
            self.voronoi_bin_mapping
        )

        im = plt.matshow((data_v_rms - model_v_rms) / noise_v_rms,
                    vmax=3, vmin=-3, cmap='RdBu_r'
                    )
        plt.colorbar(im)
        plt.show()