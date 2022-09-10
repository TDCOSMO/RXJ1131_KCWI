"""
For testing on the local machine, run from command line:

```
python process_output.py <run_id> powerlaw 0 5
```

This will compute velocity dispersion for the first 5 samples from the chain
of <run_id>. The <run_id>_out.txt file needs to be in `base_dir`.
"""

import numpy as np
import sys
from astropy.io import fits
from output_class import ModelOutput

is_cluster = True

#run_name = str(sys.argv[1])
#run_type = str(sys.argv[2])

output_file = 'multivariate_gaussian_resampled_chain.txt' #'combined_chain_HST.txt' #+ '_out.txt'

start_index = int(sys.argv[1])
num_compute = int(sys.argv[2])

print(output_file, start_index)

software = sys.argv[3] #'galkin', 'jampy'][1]
ani_model = sys.argv[4] #['om', 'constant', 'step'][0]
slit = sys.argv[5] #['single_slit', 'ifu'][1]
spherical = str(sys.argv[6])

if spherical == 'True':
    spherical = True
else:
    spherical = False

print(software, ani_model, slit, spherical)

if software == 'galkin':
    ani_model = ani_model
    
if software == 'jampy':
    slit == 'ifu'

if not is_cluster:
    base_dir = '/Users/ajshajib/Research/RXJ1131 KCWI kinematics/model_chain/'
    out_dir = '/Users/ajshajib/Research/RXJ1131 KCWI kinematics/vel_dis_test/'
else:
    base_dir = '/u/home/a/ajshajib/RXJ1131_kinematics/'
    out_dir = '/u/flashscratch/a/ajshajib/RXJ1131_vel_dis_{}/'.format(software)

output = ModelOutput(base_dir + output_file, cgd=True
                     #is_test=False
                    )

# binning = fits.getdata('voronoi_binning_KCWI_RXJ1131_icubes_mosaic_0'
#                        '.1457_bin_number.fits') - 1
binning = np.loadtxt('binning_map_sn15.txt') - 1
binning = None

print('loaded {}'.format(base_dir + output_file))
# print('model type: {}'.format(run_type))

#output.compute_model_time_delays()
#output.save_time_delays()
#print('finished computing time delays', output.model_time_delays.shape)

if software == 'jampy':
    output.compute_jampy_velocity_dispersion(start_index=start_index,
                                         num_compute=num_compute,
                                         supersampling_factor=5,
                                         analytic_kinematics=False,
                                         voronoi_bins=binning,
                                         anisotropy_model='Osipkov-Merritt' if ani_model=='om' else ani_model,
                                         print_step=5, is_spherical=spherical)
else:
    output.compute_galkin_velocity_dispersion(
                                          start_index=start_index,
                                          num_compute=num_compute,
                                          analytic_kinematics=False,
                                          supersampling_factor=5,
                                          voronoi_bins=binning,
                                          single_slit=False if slit == 'ifu' else True,
                                          anisotropy_model='Osipkov-Merritt' if ani_model=='om' else ani_model
                                          )

if spherical:
    sph = 'sph'
else:
    sph = 'ell'

if slit == 'single_slit':
    slit = 'aperture'

np.savetxt(out_dir+'{}_{}_{}_{}_vd_{}'.format(software, slit, ani_model, sph, start_index)+'_out.txt',
           output.model_velocity_dispersion)
np.savetxt(out_dir+'{}_{}_{}_{}_ani_param_{}'.format(software, slit,
                                                     ani_model, sph, start_index)+'_out.txt',
           output.ani_param)
np.savetxt(out_dir+'{}_{}_{}_{}_reff_{}'.format(software, slit, ani_model, sph, start_index)+'_out.txt',
           output.r_eff)
np.savetxt(out_dir+'{}_{}_{}_{}_inc_{}'.format(software, slit, ani_model, sph, start_index)+'_out.txt',
           output.inclination)
if slit == 'ifu':
    np.savetxt(out_dir+'{}_{}_{}_{}_ir_{}'.format(software, slit, ani_model, sph, start_index)+'_out.txt',
           output.IR_map)

print('finished computing velocity dispersions', output.model_velocity_dispersion.shape)
