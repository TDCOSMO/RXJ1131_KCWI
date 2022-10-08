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
aperture = sys.argv[3] #['single_slit', 'ifu'][1]
is_spherical = str(sys.argv[4])
lens_model_type = str(sys.argv[5])

if is_spherical == 'True':
    is_spherical = True
else:
    is_spherical = False

if software == 'galkin':
    anisotropy_model = anisotropy_model
    
if software == 'jampy':
    aperture == 'ifu'

if not is_cluster:
    base_dir = '/Users/ajshajib/Research/RXJ1131 KCWI kinematics/model_chain/'
    out_dir = '/Users/ajshajib/Research/RXJ1131 KCWI kinematics/vel_dis_test/'
else:
    base_dir = '/home/ajshajib/RXJ1131_kinematics/' #'/u/home/a/ajshajib/RXJ1131_kinematics/'
    out_dir = '/scratch/midway2/ajshajib/'# '/u/scratch/a/ajshajib/RXJ1131_kinematics_chains/'

likelihood_class = KinematicLikelihood(lens_model_type=lens_model_type,
                                       software=software,
                                       anisotropy_model=anisotropy_model,
                                       aperture=aperture,
                                       snr_per_bin=15,
                                       is_spherical=is_spherical,
                                       mpi=True
                                       )

if lens_model_type == 'powerlaw':
    num_param = 8
elif lens_model_type == 'composite':
    num_param = 9
else:
    raise NotImplementedError

walker_ratio = 6
num_steps = 500
num_walker = num_param * walker_ratio

init_lens_params = np.random.multivariate_normal(
    likelihood_class.lens_model_posterior_mean,
    cov=likelihood_class.lens_model_posterior_covariance,
    size=num_walker)

init_pos = np.concatenate((
    init_lens_params,
    # lambda, ani_param, inclination (deg)
    np.random.normal(loc=[1, 1, 90], scale=[0.1, 0.05, 5],
                     size=(num_walker, 3))
), axis=1)


def likelihood_function(params):
    """
    Wrapper around the `KinematicLikelihood.log_probability` method.
    """
    return likelihood_class.log_probability(params)


with MPIPool(use_dill=True) as pool:
    print(software, anisotropy_model, aperture, is_spherical, lens_model_type)

    if not pool.is_master():
        pool.wait()
        sys.exit(0)
    sampler = emcee.EnsembleSampler(num_walker,
                                    num_param,
                                    likelihood_function,
                                    pool=pool
                                    )

    sampler.run_mcmc(init_pos, num_steps,
                     progress=False)

    chain = sampler.get_chain(flat=True)

    np.savetxt(out_dir + 'kcwi_dynamics_chain_{}_{}_{}_{}_{}'.format(software,
                                    aperture, anisotropy_model, is_spherical,
                                    lens_model_type) + '.txt',
               chain)

    print('finished computing velocity dispersions', chain.shape)
