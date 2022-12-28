"""
For testing on the local machine, run from command line:

```
python run_mcmc.py jampy constant ifu axisymmetric powerlaw 15 oblate
```
"""

import numpy as np
import sys
from schwimmbad import MPIPool
from kinematics_likelihood import KinematicLikelihood
import emcee

is_cluster = True
resume = False
#run_name = str(sys.argv[1])
#run_type = str(sys.argv[2])

software = sys.argv[1] #'galkin', 'jampy'][1]
anisotropy_model = sys.argv[2] #['om', 'constant', 'step'][0]
aperture = sys.argv[3] #['slit', 'ifu'][1]
sphericity = sys.argv[4]
lens_model_type = sys.argv[5]
snr = int(sys.argv[6])
shape = sys.argv[7]

print(software, anisotropy_model, aperture,
      sphericity, lens_model_type)

if sphericity == 'spherical':
    is_spherical = True
elif sphericity == 'axisymmetric':
    is_spherical = False

if software == 'galkin':
    anisotropy_model = anisotropy_model
    
if software == 'jampy':
    aperture == 'ifu'

if not is_cluster:
    base_dir = '/Users/ajshajib/Research/RXJ1131_KCWI/RXJ1131_KCWI_kinematics/'
    out_dir = '/Users/ajshajib/Research/RXJ1131_KCWI/temp/'
else:
    base_dir = '/u/home/a/ajshajib/RXJ1131_kinematics/'
    out_dir = '/u/scratch/a/ajshajib/RXJ1131_kinematics_chains/'

if anisotropy_model == 'step':
    additional_ani_param_num = 1
    ani_param_init_mean = [1, 1]
    ani_param_init_sigma = [0.05, 0.05]
elif anisotropy_model == 'free_step':
    additional_ani_param_num = 2
    ani_param_init_mean = [1., 1, 30]
    ani_param_init_sigma = [0.05, 0.05, 10]
else:
    additional_ani_param_num = 0
    ani_param_init_mean = [.85]
    ani_param_init_sigma = [0.05]

if anisotropy_model == 'om':
    anisotropy_type = 'Osipkov-Merritt'
else:
    anisotropy_type = anisotropy_model

if lens_model_type == 'powerlaw':
    num_param = 8 + additional_ani_param_num
elif lens_model_type == 'composite':
    num_param = 9 + additional_ani_param_num
else:
    raise NotImplementedError

walker_ratio = 16

if anisotropy_model in ['step', 'free_step']:
    num_steps = 1250
else:
    num_steps = 1500

num_walker = num_param * walker_ratio

likelihood_class = KinematicLikelihood(lens_model_type=lens_model_type,
                                       software=software,
                                       anisotropy_model=anisotropy_type,
                                       aperture=aperture,
                                       snr_per_bin=snr,
                                       is_spherical=is_spherical,
                                       mpi=True,
                                       shape=shape
                                       )

init_lens_params = np.random.multivariate_normal(
    likelihood_class.lens_model_posterior_mean,
    cov=likelihood_class.lens_model_posterior_covariance,
    size=num_walker)

init_pos = np.concatenate((
    init_lens_params,
    # lambda, ani_param, inclination (deg)
    np.random.normal(loc=[900, 90, 0.9, *ani_param_init_mean],
                     scale=[10, 5, 0.05, *ani_param_init_sigma],
                     size=(num_walker, 4+additional_ani_param_num))
), axis=1)

# divide lens model predicted D_dt by lambda array as the sampled D_dt is
# taken as the true D_dt
init_pos[:, 3] /= init_pos[:, -2]


def likelihood_function(params):
    """
    Wrapper around the `KinematicLikelihood.get_log_probability` method
    :param params: array of parameters in the MCMC chain
    """
    return likelihood_class.get_log_probability(params)


if is_cluster:
    with MPIPool(use_dill=True) as pool:
        if not pool.is_master():
            pool.wait()
            sys.exit(0)

        filename = out_dir + 'kcwi_dynamics_backend_{}_{}_{}_{}_{}_{}_{}.txt'.format(
                                        software,
                                        aperture, anisotropy_model, sphericity,
                                        lens_model_type, snr, shape)
        backend = emcee.backends.HDFBackend(filename)
        if not resume:
            backend.reset(num_walker, num_param)

        print(software, anisotropy_model, aperture, sphericity, lens_model_type)
        sampler = emcee.EnsembleSampler(num_walker,
                                        num_param,
                                        likelihood_function,
                                        pool=pool
                                        )

        sampler.run_mcmc(init_pos, num_steps,
                         progress=False)

        chain = sampler.get_chain(flat=True)
        likelihood_chain = sampler.get_log_prob(flat=True)

        np.savetxt(out_dir + 'kcwi_dynamics_chain_{}_{}_{}_{}_{}_{}_{}.txt'.format(
                                        software,
                                        aperture, anisotropy_model, sphericity,
                                        lens_model_type, snr, shape),
                   chain)
        np.savetxt(
            out_dir + 'kcwi_dynamics_chain_{}_{}_{}_{}_{}_{}_{}_logL.txt'.format(
                software,
                aperture, anisotropy_model, sphericity,
                lens_model_type, snr, shape),
            likelihood_chain)

        print('finished computing velocity dispersions', chain.shape)
else:
    walker_ratio = 2
    num_steps = 2

    sampler = emcee.EnsembleSampler(num_walker,
                                    num_param,
                                    likelihood_class.get_log_prior
                                    )

    sampler.run_mcmc(init_pos, num_steps,
                     progress=True)

    chain = sampler.get_chain(flat=True)
    likelihood_chain = sampler.get_log_prob(flat=True)

    np.savetxt(out_dir+'kcwi_dynamics_chain_{}_{}_{}_{}_{}_{}_{}.txt'.format(
        software, aperture, anisotropy_model, sphericity, lens_model_type,
        snr, shape),
        chain)
    np.savetxt(out_dir + 'kcwi_dynamics_chain_{}_{}_{}_{}_{}_{}_{}_logL.txt'.format(
        software, aperture, anisotropy_model, sphericity, lens_model_type,
        snr, shape),
        likelihood_chain)

    print('finished computing velocity dispersions', chain.shape)
