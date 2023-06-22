import numpy as np

default_bayesflow_settings = {
    "lstm1_hidden_units": 512,
    "lstm2_hidden_units": 128,
    "transformer_template_dim": 128,
    "transformer_summary_dim": 32,
    "trainer": {
        "checkpoint_path": "../checkpoints/yes_no_task",
        "max_to_keep": 1,
        "default_lr": 5e-4,
        "memory": False,
    },
    "local_amortizer_settings": {
        "num_coupling_layers": 8,
        "coupling_design": 'interleaved'
    },
    "global_amortizer_settings": {
        "num_coupling_layers": 6,
        "coupling_design": 'interleaved'
    },
}

default_prior_settings = {
    # v, a, tau
    "ddm_loc": (0.0, 0.0, 0.0),
    "ddm_scale": (2.5, 2.5, 1.0),
    "scale_loc": (0.0, 0.0, 0.0),
    "scale_scale": (0.1, 0.1, 0.01)
}

default_lower_bounds = (0.0, 0.0, 0.0)
default_upper_bounds = (8.0, 6.0, 4.0)