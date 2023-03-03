from scipy.interpolate import interp1d
from scipy.stats import gaussian_kde

from dynamical_model import DynamicalModel
from data_util import *


class KinematicLikelihood(object):

    def __init__(self,
                 lens_model_type='powerlaw',
                 software='jampy',
                 anisotropy_model='constant',
                 aperture='ifu',
                 snr_per_bin=23,
                 is_spherical=False,
                 mpi=False,
                 shape='oblate'
                 ):
        """
        Initialize the kinematic likelihood object
        :param lens_model_type: string, 'powerlaw' or 'composite'
        :param software: string, 'jampy' or 'galkin'
        :param anisotropy_model: string, 'constant', 'step', 'free_step,
        or 'Osipkov-Merritt'
        :param aperture: string, 'ifu' or 'single_slit'
        :param snr_per_bin: float, signal-to-noise ratio per bin
        :param is_spherical: bool, if True, use spherical lens model
        :param mpi: bool, if True, use MPI
        :param shape: string, 'oblate' or 'prolate' for the lens galaxy 3D shape
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
            self.velocity_dispersion_mean = 288 # 323 # km/s
            self.velocity_dispersion_covariance = (20 * self.velocity_dispersion_mean / 323) ** 2  # km/s
            self.velocity_dispersion_inverse_covariance = 1. \
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
            mass_model=self.lens_model_type
        )
        if self.software == 'galkin':
            self.galkin_kinematics_api = self.dynamical_model.get_galkin_kinematics_api(
                anisotropy_model=self.anisotropy_model,
                single_slit=(True if self.aperture_type == 'single_slit' else
                             False)
            )
        else:
            self.galkin_kinematics_api = None

        # self.kappa_ext_array = np.loadtxt('./data_products/hst_imaging_and'
        #                                   '_lens_model_products/kappa_powerlaw_rxj.dat')
        # self.kappa_ext_kde = gaussian_kde(self.kappa_ext_array)
        self.kappa_ext_prior_interp_func = self.get_kappa_ext_prior_interp()

    def get_kappa_ext_prior_interp(self):
        """
        Get the prior for kappa_ext
        :return: scipy interp1d object for kappa_ext prior
        """
        if self.lens_model_type == 'powerlaw':
            points = kappa_ext_pl_points
        elif self.lens_model_type == 'composite':
            points = kappa_ext_comp_points

        return self.interp_points(points)

    def get_kappa_ext_prior(self, kappa_ext):
        """
        Get the prior for kappa_ext
        :param kappa_ext: float, kappa_ext
        :return: float, prior
        """
        return self.log(self.kappa_ext_prior_interp_func(kappa_ext))

    def get_galkin_velocity_dispersion(self, theta_e, gamma, ani_param):
        """
        Get the velocity dispersion from Galkin
        :param theta_e: Einstein radius
        :param gamma: power-law slope
        :param ani_param: anisotropy parameter
        :return: velocity dispersion
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
        Get the kwargs for GalKin
        :param theta_E: Einstein radius
        :param gamma: power-law slope
        :param r_eff: effective radius
        :param anisotropy_model: string, 'constant', 'Osipkov-Merritt'
        :param ani_param: anisotropy parameter
        :return: kwargs_lens, kwargs_light, kwargs_anisotropy
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
        else:
            raise NotImplementedError

        return kwargs_lens, kwargs_lens_light, kwargs_anisotropy

    @staticmethod
    def get_mean_covariance_from_file(file_id):
        """
        Read mean and covariance from a file for a given file_id
        :param file_id: string, file ID
        """
        mean = np.loadtxt(file_id + '_mean.txt')
        covariance = np.loadtxt(file_id + '_covariance.txt')

        return mean, covariance

    def load_lens_model_posterior(self, lens_model_type):
        """
        Load the lens model posterior
        :param lens_model_type: string, lens model type, 'powerlaw' or 'composite'
        """
        mean, covariance = self.get_mean_covariance_from_file(
            './data_products/lens_model_posterior_{}'.format(lens_model_type))
        self.lens_model_posterior_mean = mean
        self.lens_model_posterior_covariance = covariance

    def load_velocity_dispersion_data_ifu(self, snr_per_bin):
        """
        Load the velocity dispersion data for IFU
        :param snr_per_bin: signal-to-noise ratio per bin
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
        Load the Voronoi bin map
        :param snr_per_bin: signal-to-noise ratio per bin
        :param plot: bool, plot the Voronoi bin map if True
        """
        voronoi_bin_mapping = load_bin_mapping(snr_per_bin, plot=plot)
        # self.voronoi_bin_mapping[self.voronoi_bin_mapping < 0] = np.nan

        return voronoi_bin_mapping

    @staticmethod
    def interp_points(points):
        """
        Interpolate the points
        :param points: 2D array, points to be interpolated
        :return: scipy interp1d object
        """
        x = points[::2]
        y = points[1::2]

        return interp1d(x, y, bounds_error=False, fill_value=0.)

    def get_intrinsic_q_prior(self, intrinsic_q):
        """
        Get the intrinsic q prior. The values are taken from Chang et al.
        2013 (figure 7, https://ui.adsabs.harvard.edu/abs/2013ApJ...773
        ..149C/abstract)
        :param intrinsic_q: intrinsic q value
        """
        if self._intrinsic_q_prior_interp is None:
            if self.shape == 'oblate':
                points = oblate_shape_prior_points
            else:
                points = prolate_shape_prior_points

            self._intrinsic_q_prior_interp = self.interp_points(points)
        else:
            pass

        return self.log(self._intrinsic_q_prior_interp(intrinsic_q))

    @staticmethod
    def log(x):
        """
        Log function
        :param x: input
        :return: log(x)
        """
        if isinstance(x, np.ndarray):
            return np.log(x)
        elif x > 0:
            return np.log(x)
        else:
            return -np.inf

    @staticmethod
    def get_normalized_delta_squared(delta, inverse_covariance):
        """
        Compute v * Sigma^{-1} * v, where v is a vector and Sigma is the
        covariance matrix
        :param delta: vector
        :param inverse_covariance: inverse covariance matrix \Sigma^{-1}
        :return: v * Sigma^{-1} * v
        """
        return delta.T @ (inverse_covariance @ delta)

    def get_lens_model_likelihood(self, lens_params):
        """
        Get the likelihood of the lens model parameters
        :param lens_params: lens model parameters, for power-law [theta,
        gamma, q, D_dt], for composite [kappa, r_s, m2l, q, D_dt]
        :return: log likelihood of the lens model parameters
        """
        return -0.5 * self.get_normalized_delta_squared(
            self.lens_model_posterior_mean - lens_params,
            self.lens_model_posterior_inverse_covariance
        )

    def get_kinematic_likelihood(self, v_rms):
        """
        Get the likelihood of the kinematic data
        :param v_rms: RMS velocity
        :return: log likelihood of the kinematic data
        """
        if self.aperture_type == 'ifu':
            return -0.5 * self.get_normalized_delta_squared(
                v_rms - self.velocity_dispersion_mean,
                self.velocity_dispersion_inverse_covariance
            )
        elif self.aperture_type == 'single_slit':
            return -0.5 * (v_rms - self.velocity_dispersion_mean) ** 2 \
                   * self.velocity_dispersion_inverse_covariance
        else:
            raise ValueError('Aperture type not recognized!')

    def get_anisotropy_prior(self, ani_param):
        """
        Get uniform prior on the anisotropy parameter
        :param ani_param: anisotropy parameter
        :return: log prior of the anisotropy parameter
        """
        low = 0.78  # 0.87
        hi = 1.14 # 1.12
        if self.anisotropy_model == 'constant':
            if not low < ani_param < hi:
                return -np.inf
        elif self.anisotropy_model == 'Osipkov-Merritt':
            if not 0.1 < ani_param < 5.:
                return -np.inf
        elif self.anisotropy_model == 'step':
            if not low < ani_param[0] < hi:
                return -np.inf
            if not low < ani_param[1] < hi:
                return -np.inf
        elif self.anisotropy_model == 'free_step':
            if not low < ani_param[0] < hi:
                return -np.inf
            if not low < ani_param[1] < hi:
                return -np.inf
            if not 0.5 * 1.91 < ani_param[2] < 100.:
                return -np.inf
            # return -np.log(ani_param[2]) # log uniform prior on the break

        return 0.

    def get_log_prior(self, params):
        """
        Get the log prior of the parameters
        :param params: parameters
        """
        if self.lens_model_type == 'powerlaw':
            theta_e, gamma, q, D_dt_model, inclination, \
            kappa_ext, lambda_int, D_d, *ani_param = params

            if not 1.0 < theta_e < 2.2:
                return -np.inf

            if not 1.5 < gamma < 2.5:
                return -np.inf

            lens_model_params = np.array([theta_e, gamma, q, D_dt_model])
        elif self.lens_model_type == 'composite':
            kappa_s, r_s, m2l, q, D_dt_model, inclination, \
            kappa_ext, lambda_int, D_d, *ani_param = params
            lens_model_params = np.array([kappa_s, r_s, m2l, q, D_dt_model])
        else:
            raise NotImplementedError

        if len(ani_param) == 1:
            ani_param = ani_param[0]

        if not 0.5 < q < 0.99:
            return -np.inf

        if not 1000. < D_dt_model < 3000.:
            return -np.inf

        # if not 70 < pa < 170:
        #     return -np.inf

        if not 0 < inclination < 180:
            return -np.inf

        if inclination > 90:
            inclination = 180 - inclination

        if self.lens_model_type == 'powerlaw':
            if not 0.5 < lambda_int < 1.13:
                return -np.inf
        elif self.lens_model_type == 'composite':
            if not 0.5 < lambda_int < 1.13:
                return -np.inf

        if not -0.1 < kappa_ext < 0.4:
            return -np.inf

        if not 500 < D_d < 1300:
            return -np.inf

        if self.lens_model_type == 'powerlaw':
            intrinsic_q = np.sqrt(
                q ** 2 - np.cos(inclination * np.pi / 180.) ** 2) \
                          / np.sin(inclination * np.pi / 180.)
        elif self.lens_model_type == 'composite':
            q_nfw = self.dynamical_model.interp_nfw_q(q)
            intrinsic_q = np.sqrt(
                q_nfw ** 2 - np.cos(inclination * np.pi / 180.) ** 2) \
                          / np.sin(inclination * np.pi / 180.)
        else:
            raise ValueError("Lens model type {} not recognized!".format(
                self.lens_model_type))
        intrinsic_q_lum = np.sqrt(self.dynamical_model.LIGHT_PROFILE_MEAN[4]**2
                                  - np.cos(inclination * np.pi / 180.) ** 2) \
            / np.sin(inclination * np.pi / 180.)
        if np.isinf(intrinsic_q) or np.isnan(intrinsic_q) or intrinsic_q ** 2 \
                < 0.1:
            return -np.inf

        if np.isinf(intrinsic_q_lum) or np.isnan(intrinsic_q_lum) or \
                intrinsic_q_lum ** 2 < 0.1:
            return -np.inf

        return self.get_anisotropy_prior(ani_param) + \
               self.get_lens_model_likelihood(lens_model_params) + \
               self.get_intrinsic_q_prior(intrinsic_q_lum) + \
               self.get_kappa_ext_prior(kappa_ext)

    def get_v_rms(self, params):
        """
        Get the RMS velocity
        :param params: parameters in the MCMC chain
        :return: RMS velocity
        """
        if self.lens_model_type == 'powerlaw':
            theta_e, gamma, q, D_dt_model, inclination, \
            kappa_ext, lamda_int, D_d, *ani_param = params
            lens_params = [theta_e, gamma, q]
        elif self.lens_model_type == 'composite':
            kappa_s, r_s, m2l, q, D_dt_model, inclination, \
            kappa_ext, lamda_int, D_d, *ani_param = params
            lens_params = [kappa_s, r_s, m2l, q]
        else:
            raise ValueError('lens model type not recognized!')

        cosmo_params = [lamda_int, kappa_ext,
                        D_dt_model / (1.-kappa_ext) / lamda_int,
                        D_d]

        if len(ani_param) == 1:
            ani_param = ani_param[0]

        if self.software == 'jampy':
            v_rms, _ = self.dynamical_model.compute_jampy_v_rms_model(
                lens_params,
                cosmo_params,
                ani_param,
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
            raise ValueError('Galkin is not compatible anymore with the '
                             'sampled parameter set!')
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

        return v_rms

    def get_log_likelihood(self, params):
        """
        Get the log likelihood of the parameters
        :param params: parameters in the MCMC chain
        :return: log likelihood
        """
        v_rms = self.get_v_rms(params)
        return self.get_kinematic_likelihood(v_rms)

    def get_log_probability(self, params):
        """
        Get the log probability of the parameters
        :param params: parameters in the MCMC chain
        :return: log probability
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
        Plot the residuals of the model
        :param params: parameters in the MCMC chain
        :return: None
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


kappa_ext_pl_points = np.array([
    -0.005026022304832713, 0,
    0.0027063197026022667, 0.00047910141867291145,
    0.0075390334572490725, 0.12965682142825585,
    0.012758364312267609, 0.32329763732022876,
    0.01778438661710037, 1.0002200317626517,
    0.021457249070631956, 1.741497724268262,
    0.024743494423791812, 2.6599590989344435,
    0.02841635687732344, 3.7395422807001975,
    0.03034944237918215, 4.36794367896656,
    0.032862453531598515, 5.238027787882281,
    0.03440892193308549, 5.8986248014834395,
    0.035375464684014854, 6.3336488896381,
    0.03634200743494424, 6.833112118604218,
    0.037695167286245335, 7.509806939872767,
    0.039628252788104085, 8.412074686587822,
    0.04136802973977696, 9.1371228185359,
    0.04407434944237919, 10.023328690189953,
    0.04600743494423791, 10.619510518050589,
    0.04774721189591076, 11.183460797970028,
    0.05103345724907063, 12.134141743041939,
    0.05315985130111521, 12.504798555597946,
    0.05489962825278796, 12.617674849837194,
    0.060892193308550224, 12.779144005465303,
    0.0632118959107808, 12.666519239470857,
    0.06495167286245371, 12.634407466884332,
    0.06611152416356897, 12.473381480068493,
    0.0692044609665429, 12.248036127795867,
    0.07345724907063195, 11.636127795867306,
    0.07867657992565041, 10.89540106999317,
    0.0829293680297398, 10.202943812050288,
    0.08486245353159855, 9.687550460913307,
    0.08679553903345727, 9.284925606196378,
    0.08892193308550189, 8.898422514217774,
    0.09085501858736059, 8.576346585515166,
    0.09259479553903346, 8.205929323668496,
    0.09414126394052047, 7.771060943474905,
    0.09588104089219332, 7.432863252033962,
    0.09878066914498143, 6.981968929385772,
    0.10071375464684024, 6.788771282306072,
    0.10284014869888483, 6.337829049516019,
    0.10670631970260225, 5.613128266096478,
    0.10921933085501861, 5.371637196014586,
    0.11153903345724911, 5.146243933600095,
    0.11656505576208176, 4.324956304176171,
    0.1190780669144981, 4.051245663688549,
    0.12275092936802973, 3.6003992511822265,
    0.1260371747211896, 3.4072854468507927,
    0.1297100371747212, 3.0047683899530675,
    0.1349293680297398, 2.6667862941505263,
    0.14014869888475834, 2.232145487130806,
    0.1457546468401487, 1.94251670200779,
    0.15058736059479552, 1.6528400067429079,
    0.15599999999999997, 1.4598579553016098,
    0.1633457249070632, 1.2186663236063922,
    0.1683717472118959, 1.1223190283113453,
    0.1747509293680297, 0.8810675089387932,
    0.18113011152416353, 0.8170236267977398,
    0.186542750929368, 0.6884807161679003,
    0.19311524163568772, 0.6405585967651799,
    0.19794795539033444, 0.5603091091375294,
    0.20626022304832714, 0.4963850023511469,
    0.21437918215613383, 0.4485587032321625,
    0.21921189591078066, 0.3844190008073767,
    0.2242379182156134, 0.3364010611209203,
    0.2313903345724907, 0.2885148743246013,
    0.2387360594795539, 0.2567504502666118,
    0.24453531598513006, 0.2248902059248863,
    0.25033457249070634, 0.20913974678602898,
    0.25845353159851303, 0.1774232328699057,
    0.2642527881040892, 0.1616727737310466,
    0.3, 0.0
])

kappa_ext_comp_points = np.array([
    -0.005026022304832713, 0,
    0.0027063197026022667, 0.00047910141867291145,
    0.0075390334572490725, 0.016888325008212135,
    0.012758364312267609, 0.08165085927726601,
    0.01817100371747213, 0.21086451189325572,
    0.021457249070631956, 0.42049533763341707,
    0.024743494423791812, 0.662345733779313,
    0.028609665427509312, 1.0975494849659757,
    0.03092936802973978, 1.3232302082316778,
    0.03440892193308555, 1.8550687155645829,
    0.03634200743494423, 2.3707016174109032,
    0.03846840148698881, 2.886346496792683,
    0.04117472118959106, 3.530905590403771,
    0.04368773234200743, 4.159342921276531,
    0.04658736059479551, 5.029450985263194,
    0.05006691449814131, 5.8995949818562465,
    0.055286245353159846, 7.269250117557289,
    0.05973234200743496, 8.364990994667776,
    0.06089219330855017, 8.638929208329266,
    0.06301858736059485, 8.912927309668087,
    0.06495167286245349, 9.074144937051393,
    0.06688475836431201, 9.074264712406062,
    0.06978438661710035, 9.29998136827816,
    0.07191078066914516, 9.57397946961698,
    0.07287732342007452, 9.735137209322957,
    0.07461710037174737, 9.880233073967933,
    0.07674349442379189, 9.977023538075251,
    0.07848327137546471, 9.896582409880136,
    0.08215613382899609, 9.929029553459735,
    0.08292936802973988, 9.912967678398736,
    0.08486245353159855, 10.041965735376316,
    0.08814869888475832, 9.897181286653476,
    0.08969516728624533, 9.800618395720026,
    0.09278810408921934, 9.543053473041674,
    0.09588104089219335, 9.140500483537544,
    0.0976208178438662, 8.91507128851665,
    0.09955390334572492, 8.689654071031221,
    0.10090706319702603, 8.625298772968033,
    0.10438661710037184, 8.077781671709062,
    0.10612639405204471, 7.7556937654709825,
    0.10689962825278812, 7.594643823584208,
    0.10883271375464687, 7.498104887721698,
    0.11076579925650565, 7.449895307467778,
    0.11250557620817844, 7.321124823664069,
    0.11443866171003708, 7.047378250570041,
    0.1169516728624535, 6.660899113662376,
    0.11985130111524167, 6.467761354260009,
    0.12410408921933094, 6.065280229968684,
    0.12700371747211897, 5.856032685363452,
    0.13125650557620822, 5.340783064652079,
    0.13473605947955394, 5.14768123785611,
    0.13860223048327136, 4.648517447276658,
    0.14111524163568767, 4.455355732803364,
    0.1447881040892193, 4.101168031514229,
    0.1486542750929368, 3.6342238113405116,
    0.15097397769516727, 3.3605011933174254,
    0.15329368029739776, 3.3445351385401576,
    0.15599999999999994, 3.103056045993732,
    0.15889962825278806, 2.813259575374188,
    0.16295910780669134, 2.7490719628075393,
    0.16605204460966558, 2.523726610534906,
    0.1718513011152419, 2.1535608769330405,
    0.1782304832713758, 1.8800897871547662,
    0.18286988847583693, 1.6387304699630025,
    0.18866914498141307, 1.5424310848098184,
    0.1906022304832719, 1.4297823637444438,
    0.1960148698884765, 1.3173492383174619,
    0.200847583643123, 1.2693193210955425,
    0.20510037174721274, 1.1245947600500408,
    0.20877323420074434, 1.0764929776153203,
    0.21225278810408932, 0.9478302916308046,
    0.21515241635687743, 0.8674610286484867,
    0.21689219330855025, 0.8997883968734186,
    0.22095167286245382, 0.883930139915357,
    0.2246245353159853, 0.7713892166691831,
    0.23100371747211917, 0.7395649049338537,
    0.23351672862453565, 0.6591716868806063,
    0.23815613382899647, 0.6111297921232133,
    0.24260223048327173, 0.5308563494246332,
    0.2532342007434946, 0.5154053286724505,
    0.25884014869888466, 0.4513135363895273,
    0.26444609665427515, 0.40333152930947236,
    0.30, 0.
])

oblate_shape_prior_points = np.array([
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

prolate_shape_prior_points = np.array([
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