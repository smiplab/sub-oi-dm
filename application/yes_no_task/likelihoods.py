import numpy as np
from numba import njit

@njit
def _sample_diffusion_trial(v, a, tau, beta, dt=0.001, s=1.0, max_iter=1e5):
    """Generates a single response time from a Diffusion Decision process.

    Parameters:
    -----------
    v        : float
        The drift rate parameter.
    a        : float
        The boundary separation parameter.
    tau      : float
        The non-decision time parameter.
    beta     : float
        The starting point parameter.
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
def sample_random_walk_diffusion_process(theta_t, context, dt=0.001, s=1.0, max_iter=1e5):
    """Generates a single simulation from a non-stationary
    Diffusion decision process with a parameters following a random walk.

    Parameters:
    -----------
    theta_t  : np.ndarray of shape (num_steps, 5)
        The trajectory of the 5 latent DDM parameters, v_1, v_2, a, tau, bias.
    context  : np.ndarray of shape (num_steps, 1)
        The experimental context which determines if v_1 or v_2 should be used.
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
            theta_t[t, context[t]], theta_t[t, 2], theta_t[t, 3], theta_t[t, 4],
            dt=dt, s=s, max_iter=max_iter)
    return rt
