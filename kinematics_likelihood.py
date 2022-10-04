import numpy as np
import matplotlib.pyplot as plt

from dynamical_model import DynamicalModel


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