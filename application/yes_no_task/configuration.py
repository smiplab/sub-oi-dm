default_bayesflow_settings = {
    "lstm1_hidden_units": 512,
    "lstm2_hidden_units": 256,
    "lstm3_hidden_units": 128,
    "trainer": {
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
    "scale_loc": (0.0, 0.0, 0.0, 0.0, 0.0, 0.0),       # added value like in 2afc
    "scale_scale": (0.1, 0.1, 0.1, 0.01, 0.01, 0.01)   # added value like in 2afc
}

default_lower_bounds = (0.0, -6.0, 0.0, 0.0, 0.0, 0.0) # added lower bound 0 for guessing probability
default_upper_bounds = (6.0, 0.0, 4.0, 2.0, 1.0, 1.0)  # added upper bound 1 for guessing probability