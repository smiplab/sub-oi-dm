from scipy.stats import halfnorm
import numpy as np

from configuration import default_prior_settings, default_lower_bounds, default_upper_bounds

def sample_scale(loc=default_prior_settings['scale_loc'], scale=default_prior_settings['scale_scale']):
    """Generates 3 random draws from a half-normal prior over the
    scale of the random walk.

    Parameters:
    -----------
    loc    : tuple, optional, default: ``configuration.default_prior_settings.scale_loc``
        The location parameters of the half-normal distribution.
    scale  : tuple, optional, default: ``configuration.default_prior_settings.scale_scale``
        The scale parameters of the half-normal distribution.

    Returns:
    --------
    scales : np.array
        The randomly drawn scale parameters.
    """

    return halfnorm.rvs(loc=loc, scale=scale)

def sample_ddm_params(loc=default_prior_settings['ddm_loc'], scale=default_prior_settings['ddm_scale']):
    """Generates random draws from a half-normal prior over the
    diffusion decision parameters, v, a, tau.

    Parameters:
    -----------
    loc    : tuple, optional, default: ``configuration.default_prior_settings.ddm_loc``
        The location parameters of the half-normal distribution.
    scale  : tuple, optional, default: ``configuration.default_prior_settings.ddm_scale``
        The scale parameters of the half-normal distribution.

    Returns:
    --------
    ddm_params : np.array
        The randomly drawn DDM parameters, v, a, tau.
    """

    return halfnorm.rvs(loc=loc, scale=scale)

def sample_random_walk(sigma, num_steps=80, lower_bounds=default_lower_bounds, upper_bounds=default_upper_bounds, rng=None):
    """Generates a single simulation from a random walk transition model.

    Parameters:
    -----------
    sigma           : np.array
        The standard deviations of the random walk process.
    num_steps       : int, optional, default: 80
        The number of time steps to take for the random walk. Default
        corresponds to the maximal number of trials in the 2afc dataset.
    lower_bounds    : tuple, optional, default: ``configuration.default_lower_bounds``
        The minimum values the parameters can take.
    upper_bound     : tuple, optional, default: ``configuration.default_upper_bounds``
        The maximum values the parameters can take.
    rng             : np.random.Generator or None, default: None
        An optional random number generator to use, if fixing the seed locally.

    Returns:
    --------
    theta_t : np.ndarray of shape (num_steps, num_params)
        The array of time-varying parameters
    """

    # Configure RNG, if not provided
    if rng is None:
        rng = np.random.default_rng()
    # Sample initial parameters
    theta_t = np.zeros((num_steps, 3))
    theta_t[0] = sample_ddm_params()
    # Run random walk from initial
    z = rng.normal(size=(num_steps - 1, 3))
    for t in range(1, num_steps):
        theta_t[t] = np.clip(
            theta_t[t - 1] + sigma * z[t - 1], lower_bounds, upper_bounds
        )
    return theta_t