"""
For testing on the local machine, run from command line:

```
python run_mcmc.py <run_id> powerlaw 0 5
```

This will compute velocity dispersion for the first 5 samples from the chain
of <run_id>. The <run_id>_out.txt file needs to be in `base_dir`.
"""

import numpy as np
import sys
from schwimmbad import MPIPool
from kinematics_likelihood import KinematicLikelihood
import emcee

is_cluster = True

#run_name = str(sys.argv[1])
#run_type = str(sys.argv[2])

software = sys.argv[1] #'galkin', 'jampy'][1]
anisotropy_model = sys.argv[2] #['om', 'constant', 'step'][0]
aperture = sys.argv[3] #['slit', 'ifu'][1]
sphericity = sys.argv[4]
lens_model_type = sys.argv[5]

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
    ani_param_init_mean = [1, 1, 1]
    ani_param_init_sigma = [0.05, 0.05, 0.1]
else:
    additional_ani_param_num = 0
    ani_param_init_mean = [1]
    ani_param_init_sigma = [0.05]
if anisotropy_model == 'om':
    anisotropy_model = 'Osipkov-Merritt'

if lens_model_type == 'powerlaw':
    num_param = 8 + additional_ani_param_num
elif lens_model_type == 'composite':
    num_param = 9 + additional_ani_param_num
else:
    raise NotImplementedError

walker_ratio = 6
num_steps = 500
num_walker = num_param * walker_ratio

likelihood_class = KinematicLikelihood(lens_model_type=lens_model_type,
                                       software=software,
                                       anisotropy_model=anisotropy_model,
                                       aperture=aperture,
                                       snr_per_bin=15,
                                       is_spherical=is_spherical,
                                       mpi=True
                                       )

init_lens_params = np.random.multivariate_normal(
    likelihood_class.lens_model_posterior_mean,
    cov=likelihood_class.lens_model_posterior_covariance,
    size=num_walker)

init_pos = np.concatenate((
    init_lens_params,
    # lambda, ani_param, inclination (deg)
    np.random.normal(loc=[90, 1, *ani_param_init_mean],
                     scale=[5, 0.05, *ani_param_init_sigma],
                     size=(num_walker, 3+additional_ani_param_num))
), axis=1)


def likelihood_function(params):
    """
    Wrapper around the `KinematicLikelihood.get_log_probability` method.
    """
    return likelihood_class.get_log_probability(params)


if is_cluster:
    with MPIPool(use_dill=True) as pool:
        if not pool.is_master():
            pool.wait()
            sys.exit(0)

        print(software, anisotropy_model, aperture, sphericity, lens_model_type)
        sampler = emcee.EnsembleSampler(num_walker,
                                        num_param,
                                        likelihood_function,
                                        pool=pool
                                        )

        sampler.run_mcmc(init_pos, num_steps,
                         progress=False)

        chain = sampler.get_chain(flat=True)

        np.savetxt(out_dir + 'kcwi_dynamics_chain_{}_{}_{}_{}_{}_nl'.format(
            software,
                                        aperture, anisotropy_model, is_spherical,
                                        lens_model_type) + '.txt',
                   chain)

        print('finished computing velocity dispersions', chain.shape)
else:
    sampler = emcee.EnsembleSampler(num_walker,
                                    num_param,
                                    likelihood_class.get_log_prior
                                    )

    sampler.run_mcmc(init_pos, num_steps,
                     progress=True)

    chain = sampler.get_chain(flat=True)

    np.savetxt(out_dir+'kcwi_dynamics_chain_{}_{}_{}_{}_{}_nl.txt'.format(
        software, aperture, anisotropy_model, sphericity, lens_model_type),
        chain)

    print('finished computing velocity dispersions', chain.shape)