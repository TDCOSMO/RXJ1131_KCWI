# RXJ1131_KCWI

Notebooks and codes to extract spatially resolved velocity dispersion from Keck/KCWI for RXJ1131 and incorporate it with time-delay $H_0$ measurement, that was used 
in [TDCOSMO-XII, Shajib et al. (2023)](https://ui.adsabs.harvard.edu/abs/2023A%26A...673A...9S/abstract).

The notebooks and codes in an approximate order of the workflow are listed below.

- [Kinematic extraction](https://github.com/TDCOSMO/RXJ1131_KCWI/blob/main/kinematic_extraction/Extract%20kinematics%20from%20KCWI%20data.ipynb)
- - [kcwi_util_modified.py](https://github.com/TDCOSMO/RXJ1131_KCWI/blob/bd7508f10d71332b0454a204cc5e7bd93d12f06e/kinematic_extraction/kcwi_util_modified.py)
-  [Estimate PSF FWHM](https://github.com/TDCOSMO/RXJ1131_KCWI/blob/736f1457c75ab14793eeaff5dd97e166cedb6d83/Find%20PSF%20FWHM.ipynb)
- [Extract mean and covariance from lens model posterior](https://github.com/TDCOSMO/RXJ1131_KCWI/blob/bd7508f10d71332b0454a204cc5e7bd93d12f06e/Process%20lens%20model%20posterior.ipynb)
- [Half-light radius fitting from HST imaging](https://github.com/TDCOSMO/RXJ1131_KCWI/blob/bd7508f10d71332b0454a204cc5e7bd93d12f06e/Half-light%20radius%20by%20profile%20fitting%20(RXJ1131)%20double%20Sersic%20to%20real%20arc-subtracted%20data.ipynb)
- - [fit_light_util.py](https://github.com/TDCOSMO/RXJ1131_KCWI/blob/bd7508f10d71332b0454a204cc5e7bd93d12f06e/fit_light_utils.py)
- [Create covariance matrix for velocity dispersion map](https://github.com/TDCOSMO/RXJ1131_KCWI/blob/bd7508f10d71332b0454a204cc5e7bd93d12f06e/Create%20covariance%20matrix%20for%20velocity%20dispersion%20map.ipynb)
- [Radial binning of the velocity dispersion map](https://github.com/TDCOSMO/RXJ1131_KCWI/blob/main/Radially%20bin%20and%20create%20covariance%20matrix.ipynb)
- [Running MCMC chains for dynamical models](https://github.com/TDCOSMO/RXJ1131_KCWI/blob/main/run_mcmc.py)
- - [kinematics_likelihood.py](https://github.com/TDCOSMO/RXJ1131_KCWI/blob/bd7508f10d71332b0454a204cc5e7bd93d12f06e/kinematics_likelihood.py)
- - [dynamical_model.py](https://github.com/TDCOSMO/RXJ1131_KCWI/blob/bd7508f10d71332b0454a204cc5e7bd93d12f06e/dynamical_model.py)
- - [data_util.py](https://github.com/TDCOSMO/RXJ1131_KCWI/blob/bd7508f10d71332b0454a204cc5e7bd93d12f06e/data_util.py)
- [Notebook to make plots for kinematics maps, radial profiles, oblateness etc.](https://github.com/TDCOSMO/RXJ1131_KCWI/blob/bd7508f10d71332b0454a204cc5e7bd93d12f06e/Plot%20kinematic%20maps,%20get%20oblateness%20probability,%20radial%20profile.ipynb)
- [Notebook to make corner plots](https://github.com/TDCOSMO/RXJ1131_KCWI/blob/2ac86b65dd6bd90bb546081026012a61f3df7295/Make%20residual%20figures%20and%20corner%20plots.ipynb)
- - [plot_util.py](https://github.com/TDCOSMO/RXJ1131_KCWI/blob/bd7508f10d71332b0454a204cc5e7bd93d12f06e/plot_util.py)
- [Fit cosmological parameters](https://github.com/TDCOSMO/RXJ1131_KCWI/blob/bd7508f10d71332b0454a204cc5e7bd93d12f06e/Fit%20cosmology.ipynb)
