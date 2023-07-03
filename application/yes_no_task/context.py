import numpy as np

def generate_context(num_steps=112):
    context = np.concatenate([np.repeat(1, num_steps/2), np.repeat(0, num_steps/2)])
    return np.random.choice(context, num_steps, replace=False)