from abc import ABC, abstractmethod
from functools import partial
import numpy as np
import bayesflow as bf
from scipy.stats import halfnorm

from likelihoods import sample_random_walk_mixture_diffusion_process    # replaced with newly written function for mixture model
from priors import sample_scale, sample_random_walk, sample_mixture_ddm_params
from configuration import default_prior_settings

class DiffusionModel(ABC):
    """An interface for running a standardized simulated experiment."""

    @abstractmethod
    def __init__(self, *args, **kwargs):
        pass

    @abstractmethod
    def generate(self, batch_size, *args, **kwargs):
        pass

    @abstractmethod
    def configure(self, raw_dict, *args, **kwargs):
        pass

class RandomWalkDiffusion(DiffusionModel):
    """A wrapper for a Non-Stationary Diffusion Decision process with
    a Gaussian random walk transition model."""

    def __init__(self, rng=None):
        """Creates an instance of the Non-Stationary Diffusion Decision model with
        given configuration. When used in a BayesFlow pipeline,
        only the attribute ``self.generator`` and the method ``self.configure``
        should be used.

        Parameters:
        -----------
        rng : np.random.Generator or None, default: None
            An optional random number generator to use, if fixing the seed locally.
        """

        self.hyper_prior_mean = halfnorm(
            loc=default_prior_settings['scale_loc'],
            scale=default_prior_settings['scale_scale']
            ).mean()
        self.hyper_prior_std = halfnorm(
            loc=default_prior_settings['scale_loc'],
            scale=default_prior_settings['scale_scale']
            ).std()
        self.local_prior_means = np.array([2.0, 1.5, 0.7])
        self.local_prior_stds = np.array([1.5, 1.2, 0.5])

        # Store local RNG instance
        if rng is None:
            rng = np.random.default_rng()
        self._rng = rng

        # Create prior wrapper
        self.prior = bf.simulation.TwoLevelPrior(
            hyper_prior_fun=sample_scale,
            local_prior_fun=partial(sample_random_walk, rng=self._rng),
        )

        # Create simulator wrapper
        self.likelihood = bf.simulation.Simulator(
            simulator_fun=sample_random_walk_diffusion_process,
        )

        # Create generative model wrapper. Will generate 3D tensors
        self.generator = bf.simulation.TwoLevelGenerativeModel(
            prior=self.prior,
            simulator=self.likelihood,
            name="random_walk_diffusion_model",
        )

    def generate(self, batch_size, *args, **kwargs):
        """Wraps the call function of ``bf.simulation.TwoLevelGenerativeModel``.

        Parameters:
        -----------
        batch_size : int
            The number of simulations to generate per training batch
        **kwargs   : dict, optional, default: {}
            Optional keyword arguments passed to the call function of
            ``bf.simulation.TwoLevelGenerativeModel``

        Returns:
        --------
        raw_dict   : dict
            The simulation dictionary configured for ``bayesflow.amortizers.TwoLevelAmortizer``
        """

        return self.generator(batch_size, *args, **kwargs)

    def configure(self, raw_dict, transform=True):
        """Configures the output of self.generator for a BayesFlow pipeline.

        1. Converts float64 to float32 (for TensorFlow)
        2. Appends a trailing dimensions of 1 to data
        3. Scale the model parameters if tranform=True

        Parameters:
        -----------
        raw_dict  : dict
            A simulation dictionary as returned by ``bayesflow.simulation.TwoLevelGenerativeModel``
        transform : boolean, optional, default: True
            An indicator to standardize the parameter and log-transform the data samples. 

        Returns:
        --------
        input_dict : dict
            The simulation dictionary configured for ``bayesflow.amortizers.TwoLevelAmortizer``
        """

        # Extract relevant simulation data, convert to float32, and add extra dimension
        theta_t = raw_dict.get("local_prior_draws").astype(np.float32)
        scales = raw_dict.get("hyper_prior_draws").astype(np.float32)
        rt = raw_dict.get("sim_data").astype(np.float32)[..., None]

        if transform:
            out_dict = dict(
                local_parameters=(theta_t - self.local_prior_means) / self.local_prior_stds,
                hyper_parameters=((scales - self.hyper_prior_mean) / self.hyper_prior_std).astype(np.float32),
                summary_conditions=rt,
            )
        else:
            out_dict = dict(
                local_parameters=theta_t,
                hyper_parameters=scales,
                summary_conditions=rt
            )

        return out_dict


class RandomWalkMixtureDiffusion(DiffusionModel):
    """A wrapper for a Non-Stationary Diffusion Decision process with
    a Gaussian random walk transition model."""

    def __init__(self, rng=None):
        """Creates an instance of the Non-Stationary Diffusion Decision model with
        given configuration. When used in a BayesFlow pipeline,
        only the attribute ``self.generator`` and the method ``self.configure``
        should be used.

        Parameters:
        -----------
        rng : np.random.Generator or None, default: None
            An optional random number generator to use, if fixing the seed locally.
        """

        self.hyper_prior_mean = halfnorm(
            loc=default_prior_settings['scale_loc'],
            scale=default_prior_settings['scale_scale']
            ).mean()
        self.hyper_prior_std = halfnorm(
            loc=default_prior_settings['scale_loc'],
            scale=default_prior_settings['scale_scale']
            ).std()
        self.local_prior_means = np.array([2.0, 2.0, 0.8, 0.05])  # previous values: [2.0, 1.5, 0.7, ]
        self.local_prior_stds = np.array([1.5, 1.5, 0.6, 0.05])   # previous values: [1.5, 1.2, 0.5, ]

        # Store local RNG instance
        if rng is None:
            rng = np.random.default_rng()
        self._rng = rng

        # Create prior wrapper
        self.prior = bf.simulation.TwoLevelPrior(
            hyper_prior_fun=sample_scale,
            local_prior_fun=partial(sample_random_walk, init_fun=sample_mixture_ddm_params, rng=self._rng), # added init_fun argument
        )

        # Create simulator wrapper
        self.likelihood = bf.simulation.Simulator(
            simulator_fun=sample_random_walk_mixture_diffusion_process,
        )

        # Create generative model wrapper. Will generate 3D tensors
        self.generator = bf.simulation.TwoLevelGenerativeModel(
            prior=self.prior,
            simulator=self.likelihood,
            name="random_walk_mixture_diffusion_model",
        )

    def generate(self, batch_size, *args, **kwargs):
        """Wraps the call function of ``bf.simulation.TwoLevelGenerativeModel``.

        Parameters:
        -----------
        batch_size : int
            The number of simulations to generate per training batch
        **kwargs   : dict, optional, default: {}
            Optional keyword arguments passed to the call function of
            ``bf.simulation.TwoLevelGenerativeModel``

        Returns:
        --------
        raw_dict   : dict
            The simulation dictionary configured for ``bayesflow.amortizers.TwoLevelAmortizer``
        """

        return self.generator(batch_size, *args, **kwargs)

    def configure(self, raw_dict, transform=True):
        """Configures the output of self.generator for a BayesFlow pipeline.

        1. Converts float64 to float32 (for TensorFlow)
        2. Appends a trailing dimensions of 1 to data
        3. Scale the model parameters if tranform=True

        Parameters:
        -----------
        raw_dict  : dict
            A simulation dictionary as returned by ``bayesflow.simulation.TwoLevelGenerativeModel``
        transform : boolean, optional, default: True
            An indicator to standardize the parameter and log-transform the data samples. 

        Returns:
        --------
        input_dict : dict
            The simulation dictionary configured for ``bayesflow.amortizers.TwoLevelAmortizer``
        """

        # Extract relevant simulation data, convert to float32, and add extra dimension
        theta_t = raw_dict.get("local_prior_draws").astype(np.float32)
        scales = raw_dict.get("hyper_prior_draws").astype(np.float32)
        rt = raw_dict.get("sim_data").astype(np.float32)[..., None]

        if transform:
            out_dict = dict(
                local_parameters=(theta_t - self.local_prior_means) / self.local_prior_stds,
                hyper_parameters=((scales - self.hyper_prior_mean) / self.hyper_prior_std).astype(np.float32),
                summary_conditions=rt,
            )
        else:
            out_dict = dict(
                local_parameters=theta_t,
                hyper_parameters=scales,
                summary_conditions=rt
            )

        return out_dict