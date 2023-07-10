from scipy.stats import halfnorm, truncnorm, beta
import numpy as np

from configuration import default_prior_settings, default_lower_bounds, default_upper_bounds

def sample_scale(loc=default_prior_settings['scale_loc'], scale=default_prior_settings['scale_scale']):
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
    """Generates random draws from truncated normal and beta priors over the
    diffusion decision parameters, v_1, v_2, a, tau, bias.

    Returns:
    --------
    ddm_params : np.array
        The randomly drawn DDM parameters, v_1, v_2, a, tau, bias.
    """

    v_1 = truncnorm.rvs(a=0, b=np.inf, loc=0.0, scale=2.5)
    v_2 = truncnorm.rvs(a=-np.inf, b=0.0, loc=0.0, scale=2.5)
    a = truncnorm.rvs(a=0, b=np.inf, loc=0.0, scale=2.5)
    tau = truncnorm.rvs(a=0, b=np.inf, loc=0.0, scale=1)
    bias = beta.rvs(a=50, b=50)

    return np.concatenate(([v_1], [v_2], [a], [tau], [bias]))

def sample_random_walk(sigma, num_steps=112, lower_bounds=default_lower_bounds, upper_bounds=default_upper_bounds, rng=None):
    """Generates a single simulation from a random walk transition model.

    Parameters:
    -----------
    sigma           : np.array
        The standard deviations of the random walk process.
    num_steps       : int, optional, default: 112
        The number of time steps to take for the random walk. Default
        corresponds to the maximal number of trials in the yes no dataset.
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
    theta_t = np.zeros((num_steps, 5))
    theta_t[0] = sample_ddm_params()
    # Run random walk from initial
    z = rng.normal(size=(num_steps - 1, 5))
    for t in range(1, num_steps):
        theta_t[t] = np.clip(
            theta_t[t - 1] + sigma * z[t - 1], lower_bounds, upper_bounds
        )
    return theta_t