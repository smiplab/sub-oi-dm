import numpy as np
from numba import njit
from configuration import default_prior_settings
from scipy.stats import halfnorm, truncnorm

@njit
def _sample_diffusion_trial(v, a, tau, beta=0.5, dt=0.001, s=1.0, max_iter=1e5):
    """Generates a single response time from a Diffusion Decision process.

    Parameters:
    -----------
    v        : float
        The drift rate parameter.
    a        : float
        The boundary separation parameter.
    tau      : float
        The non-decision time parameter.
    beta     : float, optional, default: 0.5
        The starting point parameter. The default corresponds to
        no a priori bias.
    dt       : float, optional, default: 0.001
        Time resolution of the process. Default corresponds to
        a precision of 1 millisecond.
    s        : float, optional, default: 1
        Scaling factor of the Wiener noise.
    max_iter : int, optional, default: 1e5
        Maximum iterations of the process. Default corresponds to
        100 seconds.

    Returns:
    --------
    rt : float
        A response time samples from the static Diffusion decision process.
        Reaching the lower boundary results in a negative rt.
    """
    n_iter = 0
    x = a * beta
    c = np.sqrt(dt * s)
    while x > 0 and x < a and n_iter < max_iter:
        x += v*dt + c * np.random.randn()
        n_iter += 1
    rt = n_iter * dt
    return rt+tau if x >= 0 else -(rt+tau)

@njit
def sample_random_walk_diffusion_process(theta_t, beta=0.5, dt=0.001, s=1.0, max_iter=1e5):
    """Generates a single simulation from a non-stationary
    Diffusion decision process with a parameters following a random walk.

    Parameters:
    -----------
    theta_t : np.ndarray of shape (theta_t, 3)
        The trajectory of the 3 latent DDM parameters, v, a, tau.
    beta     : float, optional, default: 0.5
        The starting point parameter. The default corresponds to
        no a priori bias.
    dt       : float, optional, default: 0.001
        Time resolution of the process. Default corresponds to
        a precision of 1 millisecond.
    s        : float, optional, default: 1
        Scaling factor of the Wiener noise.
    max_iter : int, optional, default: 1e5
        Maximum iterations of the process. Default corresponds to
        100 seconds.

    Returns:
    --------
    rt : np.array of shape (num_steps, )
        Response time samples from the Random Walk Diffusion decision process.
        Reaching the lower boundary results in negative rt's.
    """

    num_steps = theta_t.shape[0]
    rt = np.zeros(num_steps)
    
    for t in range(num_steps):
        rt[t] = _sample_diffusion_trial(
            theta_t[t, 0], theta_t[t, 1], theta_t[t, 2], beta,
            dt=dt, s=s, max_iter=max_iter)
    return rt

# @njit
def sample_random_walk_mixture_diffusion_process(params, beta=0.5, dt=0.001, s=1.0, max_iter=1e5):
    """Generates a single simulation from a mixture model. Response times are generated as a guess or as a 
    non-stationary Diffusion decision process with parameters following a random walk. Probability to guess
    also follows a random walk.

    Parameters:
    -----------
    params   : tuple of shape (theta_t, gamma)
        theta_t: The trajectory of the 3 latent DDM parameters, v, a, tau, and the trajectory of the probability of guessing, p.
        gamma  : The prior draws for the guessing rt distribution.
    beta     : float, optional, default: 0.5
        The starting point parameter. The default corresponds to
        no a priori bias.
    dt       : float, optional, default: 0.001
        Time resolution of the process. Default corresponds to
        a precision of 1 millisecond.
    s        : float, optional, default: 1
        Scaling factor of the Wiener noise.
    max_iter : int, optional, default: 1e5
        Maximum iterations of the process. Default corresponds to
        100 seconds.

    Returns:
    --------
    rt : np.array of shape (num_steps, )
        Response time samples from the Random Walk Mixture Diffusion decision process.
        Reaching the lower boundary results in negative rt's.
    """
    theta_t = params[0]
    gamma = params[1]
    num_steps = theta_t.shape[0]
    rt = np.zeros(num_steps)

    for t in range(num_steps):
        guessing_state = np.random.binomial(1, theta_t[t, 3])
        if guessing_state == 1:
            guessing_direction = np.random.binomial(1, 0.5)
            if guessing_direction == 1:
                myclip_a, myclip_b = 0, 1
                a, b = (myclip_a - gamma[0]) / gamma[1], (myclip_b - gamma[0]) / gamma[1]
                rt[t] = truncnorm.rvs(a, b, loc=gamma[0], scale=gamma[1])
            else:
                myclip_a, myclip_b = 0, 1
                a, b = (myclip_a - gamma[0]) / gamma[1], (myclip_b - gamma[0]) / gamma[1]
                rt[t] = -truncnorm.rvs(a, b, loc=gamma[0], scale=gamma[1])
        else:
            rt[t] = _sample_diffusion_trial(
                theta_t[t, 0], theta_t[t, 1], theta_t[t, 2], beta,
                dt=dt, s=s, max_iter=max_iter)
    return rt