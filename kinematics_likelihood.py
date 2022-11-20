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
        self.anisotropy_model = anisotropy_model
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
            scrapped_points = np.array([
                # these values are for brightest ellipticals from
                # Padilla & Strauss08
                0., 0.,
                0.09378378378378378, 0.000,
                0.09378378378378378, 0.006169665809769409,
                0.11081081081081094, 0.006169665809768077,
                0.12844594594594594, 0.002827763496143998,
                0.1497297297297297, 0.009511568123393488,
                0.17283783783783785, 0.022879177377892468,
                0.19351351351351354, 0.032904884318766925,
                0.21418918918918922, 0.04627249357326502,
                0.24337837837837833, 0.06298200514138808,
                0.2628378378378378, 0.07634961439588706,
                0.2829054054054054, 0.0997429305912596,
                0.31270270270270273, 0.12982005141388164,
                0.3321621621621621, 0.14987146529563056,
                0.3516216216216215, 0.16992287917737814,
                0.36439189189189164, 0.20000000000000018,
                0.37959459459459455, 0.23341902313624674,
                0.3978378378378377, 0.2701799485861187,
                0.4160810810810811, 0.30694087403598935,
                0.4312837837837837, 0.34035989717223636,
                0.444662162162162, 0.37043701799485795,
                0.461081081081081, 0.4305912596401029,
                0.47567567567567554, 0.5007712082262215,
                0.4878378378378378, 0.5676092544987146,
                0.4987837837837837, 0.6210796915167096,
                0.5133783783783783, 0.6946015424164527,
                0.5291891891891891, 0.7614395886889462,
                0.5389189189189189, 0.8149100257069408,
                0.5498648648648647, 0.8750642673521849,
                0.5602027027027026, 0.965295629820051,
                0.5711486486486486, 1.0688946015424166,
                0.5796621621621619, 1.1424164524421596,
                0.5863513513513512, 1.2059125964010282,
                0.596689189189189, 1.2994858611825193,
                0.6076351351351351, 1.4030848329048844,
                0.6143243243243242, 1.4732647814910027,
                0.6216216216216216, 1.5367609254498715,
                0.6319594594594594, 1.6303341902313624,
                0.6416891891891892, 1.7239074550128535,
                0.6502027027027025, 1.8041131105398458,
                0.6581081081081079, 1.8709511568123394,
                0.6690540540540539, 1.9645244215938302,
                0.6787837837837837, 2.051413881748072,
                0.6885135135135134, 2.131619537275064,
                0.697027027027027, 2.1984575835475577,
                0.7073648648648648, 2.282005141388175,
                0.7189189189189188, 2.3789203084832904,
                0.7274324324324322, 2.4524421593830334,
                0.7353378378378377, 2.5192802056555266,
                0.7450675675675675, 2.6028277634961436,
                0.7614864864864865, 2.639588688946015,
                0.7906756756756748, 2.639588688946015,
                0.8119594594594594, 2.6329048843187657,
                0.832027027027027, 2.6262210796915166,
                0.8484459459459456, 2.629562982005141,
                0.855135135135135, 2.559383033419023,
                0.8630405405405404, 2.4591259640102825,
                0.871554054054054, 2.3588688946015424,
                0.8776351351351348, 2.2853470437017993,
                0.8843243243243242, 2.2118251928020563,
                0.8928378378378378, 2.118251928020565,
                0.9013513513513515, 2.017994858611825,
                0.9074324324324322, 1.937789203084833,
                0.9147297297297297, 1.8676092544987146,
                0.9232432432432429, 1.7640102827763495,
                0.9305405405405405, 1.6670951156812341,
                0.9372297297297298, 1.5935732647814909,
                0.944527027027027, 1.5133676092544988,
                1., 0.
                # these values are for all ellipticals luminosity weighted
                # from Padilla & Strauss 08
                # 0, 0,
                # 0.05, 0,
                # 0.116, 0,
                # 0.17, 0.049,
                # 0.223, 0.223,
                # 0.272, 0.467,
                # 0.322, 0.652,
                # 0.376, 0.745,
                # 0.426, 0.842,
                # 0.475, 0.995,
                # 0.525, 1.109,
                # 0.577, 1.217,
                # 0.626, 1.337,
                # 0.676, 1.484,
                # 0.725, 1.516,
                # 0.776, 1.576,
                # 0.826, 1.489,
                # 0.876, 1.342,
                # 0.928, 1.076,
                # 0.976, 0.755,
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
            if not 0.5 < ani_param < 2.:
                return -np.inf
        elif self.anisotropy_model == 'Osipkov-Merritt':
            if not 0.1 < ani_param < 5.:
                return -np.inf
        elif self.anisotropy_model == 'step':
            if not 0.5 < ani_param[0] < 2.:
                return -np.inf
            if not 0.5 < ani_param[1] < 2.:
                return -np.inf
        elif self.anisotropy_model == 'free_step':
            if not 0.5 < ani_param[0] < 2.:
                return -np.inf
            if not 0.5 < ani_param[1] < 2.:
                return -np.inf
            if not 0. < ani_param[2] < 100.:
                return -np.inf

            return -np.log(ani_param[2])

        return 0.

    def get_log_prior(self, params):
        """
        """
        if self.lens_model_type == 'powerlaw':
            theta_e, gamma, q, pa, D_dt, inclination, lamda, *ani_param = params

            if not 1.0 < theta_e < 2.2:
                return -np.inf

            if not 1.5 < gamma < 2.5:
                return -np.inf

            lens_model_params = np.array([theta_e, gamma, q, pa, D_dt])
        elif self.lens_model_type == 'composite':
            kappa_s, r_s, m2l, q, pa, D_dt, inclination, lamda, *ani_param = \
                params
            lens_model_params = np.array([kappa_s, r_s, m2l, q, pa, D_dt])
        else:
            raise NotImplementedError

        if len(ani_param) == 1:
            ani_param = ani_param[0]

        if not 0.5 < q < 0.99:
            return -np.inf

        if not 70 < pa < 170:
            return -np.inf

        if inclination > 90:
            inclination = 180 - inclination

        if self.lens_model_type == 'powerlaw':
            intrinsic_q = np.sqrt(q**2 - np.cos(inclination*np.pi/180.)**2)
        elif self.lens_model_type == 'composite':
            q_nfw = self.dynamical_model.interp_nfw_q(q)
            intrinsic_q = np.sqrt(
                q_nfw ** 2 - np.cos(inclination * np.pi / 180.) ** 2)
        else:
            raise ValueError("Lens model type {} not recognized!".format(
                self.lens_model_type))
        intrinsic_q_lum = np.sqrt(self.dynamical_model.q_light_2()**2 -
                                  np.cos(inclination * np.pi / 180.) ** 2)
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
            self.get_intrinsic_q_prior(intrinsic_q)

    def get_v_rms(self, params):
        """
        """
        if self.lens_model_type == 'powerlaw':
            theta_e, gamma, q, pa, D_dt, inclination, lamda, *ani_param = params
            lens_params = [theta_e, gamma, q]
        elif self.lens_model_type == 'composite':
            kappa_s, r_s, m2l, q, pa, D_dt, inclination, lamda, *ani_param = \
                params
            lens_params = [kappa_s, r_s, m2l, q]
        else:
            raise ValueError('lens model type not recognized!')

        if len(ani_param) == 1:
            ani_param = ani_param[0]

        if self.software == 'jampy':
            v_rms, _ = self.dynamical_model.compute_jampy_v_rms_model(
                lens_params, ani_param, pa, inclination,
                anisotropy_model=self.anisotropy_model,
                voronoi_bins=self.voronoi_bin_mapping,
                om_r_scale=ani_param if self.anisotropy_model ==
                'Osipkov-Merritt' else None,
                is_spherical=self.is_spherical,
                aperture_type=self.aperture_type
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