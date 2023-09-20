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
    # v, a, tau, p
    "ddm_loc": (0.0, 0.0, 0.0, 1.0),        # TO DO: test values for p - trial and error - remember: a and b of beta distribution
    "ddm_scale": (2.5, 2.5, 1.0, 30),       # TO DO: test values for p - trial and error - remember: a and b of beta distribution
    "scale_loc": (0.0, 0.0, 0.0, 0.0),      # TO DO: test values for p - trial and error
    "scale_scale": (0.1, 0.1, 0.01, 0.01)   # TO DO: test values for p - trial and error
}

default_lower_bounds = (0.0, 0.0, 0.0, 0.0)
default_upper_bounds = (6.0, 4.0, 2.0, 1.0)