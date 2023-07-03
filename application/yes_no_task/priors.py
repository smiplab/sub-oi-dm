from scipy.stats import truncnorm, beta
import numpy as np


def sample_scale(loc=default_scale_prior_loc, scale=default_scale_prior_scale):
    """Generates 3 random draws from a half-normal prior over the
    scale of the random walk.

    Parameters:
    -----------
    loc    : tuple, optional, default: ``configuration.default_scale_prior_loc``
        The location parameters of the half-normal distribution.
    scale  : tuple, optional, default: ``configuration.default_scale_prior_scale``
        The scale parameters of the half-normal distribution.

    Returns:
    --------
    scales : np.array
        The randomly drawn scale parameters.
    """

    return halfnorm.rvs(loc=loc, scale=scale)


def sample_ddm_params():
    """Generates random draws from a half-normal prior over the
    diffusion decision parameters, v, a, tau.

    Parameters:
    -----------
    loc        : list, optional, default: [0.0, 0.0, 0.0]
        The shapes of the half-normal distribution.
    scale      : list, optional, default: [2.5, 2.5, 1.0]
        The scales of the half-normal distribution.

    Returns:
    --------
    ddm_params : np.array
        The randomly drawn DDM parameters, v, a, tau.
    """

    v_1 = truncnorm.rvs(a=0, b=np.inf, loc=0.0, scale=2.5)
    v_2 = truncnorm.rvs(a=-np.inf, b=0.0, loc=0.0, scale=2.5)
    a = truncnorm.rvs(a=0, b=np.inf, loc=0.0, scale=2.5)
    tau = truncnorm.rvs(a=0, b=np.inf, loc=0.0, scale=1)
    bias = beta.rvs(a=25, b=25)

    return np.concatenate(v_1, v_2, a, tau, bias)