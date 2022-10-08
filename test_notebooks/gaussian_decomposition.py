from scipy.special import comb
import numpy as np


P = 10

## Euler nodes and weights
kes = np.arange(2*P+1)
betas = np.sqrt(2*P * np.log(10) / 3. + 1j * 2*np.pi * kes)
epsilons = np.zeros(2*P+1)

epsilons[0] = 0.5
epsilons[1:P+1] = 1.
epsilons[-1] = 1/2**P

for k in range(1, P):
    epsilons[2*P-k] = epsilons[2*P-k+1] + 1/2**P * comb(P, k)

etas = (-1)**kes * epsilons * 2 * np.sqrt(2*np.pi) * 10**(P/3)


def transform(func, sigmas, **kwargs):
    """
    """
    f_sigmas = np.zeros_like(sigmas)

    f_sigmas = np.sum(etas * func(sigmas[:, np.newaxis] * betas[np.newaxis, :],
                                  **kwargs).real,
                      axis=1
                     )

    return f_sigmas


def decompose(func, sigma_start=0.02, sigma_end=15, N_sigma=15, **kwargs):
    """
    Compute the amplitudes and sigmas of Gaussian components using the
    integral transform with Gaussian kernel. The returned values are in the 
    convention of eq. (2.13).
    """
    sigmas = np.logspace(np.log10(sigma_start), np.log10(sigma_end), N_sigma)

    f_sigmas = transform(func, sigmas, **kwargs)

    # weighting for trapezoid method integral
    f_sigmas[0] *= 0.5
    f_sigmas[-1] *= 0.5

    del_log_sigma = np.abs(np.diff(np.log(sigmas)).mean())

    f_sigmas *= del_log_sigma / np.sqrt(2.*np.pi)

    return f_sigmas, sigmas


def sersic_func(R, n_sersic=1):
    """
    """
    b_n = 1.9992 * n_sersic - 0.3271

    return np.exp(-b_n*R**(1./n_sersic) + b_n)


def sum_gauss_components(R, A_sigmas, sigmas):
    """
    """
    total = np.zeros_like(R)

    for i, r in enumerate(R):
        total[i] = np.sum(A_sigmas * np.exp(-r*r/2/sigmas**2))

    return total