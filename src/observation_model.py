import numpy as np
from numpy.random import default_rng
from numba import njit

RNG = np.random.default_rng(2023)

@njit
def ddm_trial(v, a, ndt, zr=0.5, dt=0.001, s=1.0, max_iter=1e4):
    n_iter = 0
    x = a * zr
    c = np.sqrt(dt * s)
    while x > 0 and x < a and n_iter < max_iter:
        x += v*dt + c * np.random.randn()
        n_iter += 1
    rt = n_iter * dt
    return rt+ndt if x >= 0 else -(rt+ndt)

@njit
def dynamic_ddm(theta_t, shared):
    T = theta_t.shape[0]
    rt = np.zeros(T)
    for t in range(T):
        rt[t] = ddm_trial(theta_t[t, 0], theta_t[t, 1], shared)
    return np.atleast_2d(rt).T

@njit
def batched_dynamic_ddm(params):
    local_params = params[0]
    shared_params = params[1]
    B, T = local_params.shape[0], local_params.shape[1]
    rt = np.zeros((B, T, 1))
    for b in range(B):
        rt[b] = dynamic_ddm(local_params[b], shared_params[b, 0])
    return rt