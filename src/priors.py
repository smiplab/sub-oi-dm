import numpy as np
from scipy.stats import halfnorm

LOWER_BOUND = [0.0, 0.1]
UPPER_BOUND = [8.0, 6.0]

def rwddm_local_prior(hypers, T=150):
    """
    Draws a random local parameters trajectory following a Random Walk
    given a set of randomly sampled starting values and hyper parameters.
    """
    # starting values for local parameters
    params = halfnorm.rvs(loc=[0.0, 0.5], scale=[2.0, 1.5])
    params_t = np.zeros((T, 2))
    params_t[0, :] = params

    # randomness
    z = np.random.normal(loc=0.0, scale=1.0, size=(T-1, 2))
    
    # transition model
    for t in range(1, T):
        params_t[t, :] = np.clip(params_t[t-1, :] + hypers * z[t-1, :],
                                 a_min=LOWER_BOUND, a_max=UPPER_BOUND)
    return params_t

def rwddm_shared_prior():
    """
    Draws a random sample for the shared ndt parameter of the DDM.
    """
    return np.array([halfnorm.rvs(loc=0.1, scale=0.2)])


def rwddm_hyper_prior():
    """
    Draws a random sample for the high-level parameter of the Random Walk transition model.
    sigma : Std. deviation of the random walk.
    """
    return halfnorm.rvs(loc=0.0, scale=0.1, size=2)
   