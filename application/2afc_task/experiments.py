from abc import ABC, abstractmethod
import bayesflow as bf
import tensorflow as tf

from tensorflow.keras.layers import LSTM, Bidirectional
from tensorflow.keras.models import Sequential

from configuration import default_bayesflow_settings

class Experiment(ABC):
    """An interface for running a standardized simulated experiment."""

    @abstractmethod
    def __init__(self, model, *args, **kwargs):
        pass

    def run(self, epochs=100, iterations_per_epoch=1000, batch_size=32):
        """Wrapper for online training

        Parameters:
        -----------
        epochs: int, optional, default: 100
            Number of trainig epochs.
        iterations_per_epoch, int, optional, default: 1000
            Number of iterations per epoch.
        batch_size: int, optional, default: 32
            Number of simulated data sets per batch.

        Returns:
        --------
        history : dict
            A dictionary with the training history/
        """

        history = self.trainer.train_online(epochs, iterations_per_epoch, batch_size)
        return history


class RandomWalkDiffusionExperiment(Experiment):
    """Wrapper for estimating the Non-Stationary Diffusion Decision Model with
    a Gaussian random walk transition model neural superstatistics method."""

    def __init__(self, model, config=default_bayesflow_settings):
        """Creates an instance of the model with given configuration. When used in a BayesFlow pipeline,
        only the attribute ``self.generator`` and the method ``self.configure`` should be used.

        Parameters:
        -----------
        model   : an instance of models.RandomWalkDiffusion
            The model wrapper, should include a callable attribute ``generator`` and a method
            ``configure()``
        config  : dict, optional, default: ``configuration.default_settings``
            A configuration dictionary with the following keys:
            ``lstm1_hidden_units``        - The dimensions of the first LSTM of the first summary net
            ``lstm2_hidden_units``        - The dimensions of the second LSTM of the first summary net
            ``lstm3_hidden_units``        - The dimensions of the third LSTM of the second summary net
            ``trainer``                   - The settings for the ``bf.trainers.Trainer``, not icnluding
                the ``amortizer``, ``generative_model``, and ``configurator`` keys,
                as these will be provided internaly by the Experiment instance
        """

        self.model = model

        # Two-level summary network -> reduce 3D into 3D and 2D
        # for local and global amortizer, respectively
        self.summary_network = bf.networks.HierarchicalNetwork(
            [
                tf.keras.Sequential(
                    [
                        tf.keras.layers.LSTM(
                            config["lstm1_hidden_units"],
                            return_sequences=True
                        ),
                        tf.keras.layers.LSTM(
                            config["lstm2_hidden_units"],
                            return_sequences=True
                        ),
                    ]
                ),
                tf.keras.Sequential(
                    [
                        tf.keras.layers.LSTM(
                            config["lstm3_hidden_units"]
                        )
                    ]
                )
            ]
        )

        self.local_net = bf.amortizers.AmortizedPosterior(
            bf.networks.InvertibleNetwork(
                num_params=3,
                **config.get("local_amortizer_settings")
            ))

        self.global_net = bf.amortizers.AmortizedPosterior(
            bf.networks.InvertibleNetwork(
                num_params=3,
                **config.get("global_amortizer_settings")
            ))

        self.amortizer = bf.amortizers.TwoLevelAmortizedPosterior(
            self.local_net,
            self.global_net,
            self.summary_network
            )

        # Trainer setup
        self.trainer = bf.trainers.Trainer(
            amortizer=self.amortizer,
            generative_model=self.model.generate,
            configurator=self.model.configure,
            **config.get("trainer")
        )


class RandomWalkMixtureDiffusionExperiment(Experiment):
    """Wrapper for estimating the Non-Stationary Diffusion Decision Model with
    a Gaussian random walk transition model neural superstatistics method."""

    def __init__(self, model, summary_network_type="smoothing", checkpoint_path=None, config=default_bayesflow_settings):
        """Creates an instance of the model with given configuration. When used in a BayesFlow pipeline,
        only the attribute ``self.generator`` and the method ``self.configure`` should be used.

        Parameters:
        -----------
        model   : an instance of models.RandomWalkDiffusion
            The model wrapper, should include a callable attribute ``generator`` and a method
            ``configure()``
        config  : dict, optional, default: ``configuration.default_settings``
            A configuration dictionary with the following keys:
            ``lstm1_hidden_units``        - The dimensions of the first LSTM of the first summary net
            ``lstm2_hidden_units``        - The dimensions of the second LSTM of the first summary net
            ``lstm3_hidden_units``        - The dimensions of the third LSTM of the second summary net
            ``trainer``                   - The settings for the ``bf.trainers.Trainer``, not icnluding
                the ``amortizer``, ``generative_model``, and ``configurator`` keys,
                as these will be provided internaly by the Experiment instance
        """

        self.model = model

        if summary_network_type == "smoothing":
            self.summary_network = bf.networks.HierarchicalNetwork([
                Sequential([
                    Bidirectional(LSTM(config["lstm1_hidden_units"], return_sequences=True)),
                    Bidirectional(LSTM(config["lstm2_hidden_units"], return_sequences=True)),
                ]),
                Sequential([Bidirectional(LSTM(config["lstm3_hidden_units"]))])
            ])
        if summary_network_type == "filtering":
            self.summary_network = bf.networks.HierarchicalNetwork([
                Sequential([
                    LSTM(config["lstm1_hidden_units"], return_sequences=True),
                    LSTM(config["lstm2_hidden_units"], return_sequences=True),
                ]),
                Sequential([LSTM(config["lstm3_hidden_units"])])
            ])

        self.local_net = bf.amortizers.AmortizedPosterior(
            bf.networks.InvertibleNetwork(
                num_params=4,
                **config.get("local_amortizer_settings")
            ))

        self.global_net = bf.amortizers.AmortizedPosterior(
            bf.networks.InvertibleNetwork(
                num_params=4 + 2,
                **config.get("global_amortizer_settings")
            ))

        self.amortizer = bf.amortizers.TwoLevelAmortizedPosterior(
            self.local_net,
            self.global_net,
            self.summary_network
            )

        # Trainer setup
        self.trainer = bf.trainers.Trainer(
            amortizer=self.amortizer,
            generative_model=self.model.generate,
            configurator=self.model.configure,
            checkpoint_path=checkpoint_path,
            **config.get("trainer")
        )
