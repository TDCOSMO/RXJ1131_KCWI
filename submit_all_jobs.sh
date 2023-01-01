# baseline, powerlaw
./idre_submit_job.sh jampy constant ifu axisymmetric powerlaw 23 oblate
./idre_submit_job.sh jampy constant ifu axisymmetric powerlaw 23 prolate

## axisymmetric vs spherical
./idre_submit_job.sh jampy constant ifu spherical powerlaw 23 oblate
#
## ifu vs single slit, constant, both spherical and axisymmetric
./idre_submit_job.sh jampy constant single_slit spherical powerlaw 23 oblate
./idre_submit_job.sh jampy constant single_slit axisymmetric powerlaw 23 oblate
./idre_submit_job.sh jampy constant single_slit axisymmetric powerlaw 23 prolate
#
## ifu vs single slit, Osipkov-Merritt, spherical
#./idre_submit_job.sh jampy om single_slit spherical powerlaw 23 oblate
#./idre_submit_job.sh jampy om ifu spherical powerlaw 23 oblate
#
## anisotropy
./idre_submit_job.sh jampy step ifu axisymmetric powerlaw 23 oblate
./idre_submit_job.sh jampy step ifu axisymmetric powerlaw 23 prolate
#./idre_submit_job.sh jampy free_step ifu axisymmetric powerlaw 23 oblate
#./idre_submit_job.sh jampy free_step ifu axisymmetric powerlaw 23 prolate
#
## snr 23 vs 28
./idre_submit_job.sh jampy constant ifu axisymmetric powerlaw 28 oblate
./idre_submit_job.sh jampy constant ifu axisymmetric powerlaw 28 prolate

# composite
./idre_submit_job.sh jampy constant ifu axisymmetric composite 23 oblate
./idre_submit_job.sh jampy constant ifu axisymmetric composite 23 prolate

