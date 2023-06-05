import numpy as np
from scipy.stats import halfnorm

LOWER_BOUND = [0.0, 0.0, 0.0]
UPPER_BOUND = [8.0, 6.0, 1.0]

def rwddm_local_prior(hypers, T):
    """
    Draws a random local parameters trajectory following a Random Walk
    given a set of randomly sampled starting values and hyper parameters.
    """
    # starting values for local parameters
    params = halfnorm.rvs(loc=[0.0, 0.5, 0.1], scale=[2.0, 1.5, 0.2])
    params_t = np.zeros((T, 3))
    params_t[0, :] = params

    # randomness
    z = np.random.normal(loc=0.0, scale=1.0, size=(T-1, 3))
    
    # transition model
    for t in range(1, T):
        params_t[t, 0] = np.clip(params_t[t-1, :] + hypers[0] * z[t-1, :],
                                 a_min=LOWER_BOUND, a_max=UPPER_BOUND)
    return params_t

def rwddm_shared_prior():
    


def rwddm_hyper_prior():
    """
    Draws a random sample for the high-level parameter of the Random Walk transition model.
    sigma : Std. deviation of the random walk.
    """
    sigmas = halfnorm.rvs(loc=0.0, scale=[0.1, 0.1, 0.005])
    return np.concatenate([sigmas, q])