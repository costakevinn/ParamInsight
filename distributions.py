# distributions.py
# Distributions and likelihood functions for ParamInsight (2D)
# Generic 2-parameter model (a, b)
# Log-likelihood via chi²
# Random number generators: uniform and normal (Box-Muller)

import math
import numpy as np

# --------------------------
# PDF / Likelihood functions
# --------------------------

def log_likelihood_2d(a, b, x_data, y_data, dy_data, model_func):
    """
    Compute log-likelihood via chi² for two parameters (a, b).
    L ~ exp(-chi²/2)

    Parameters:
        a (float): model parameter 1
        b (float): model parameter 2
        x_data (array): observed x values
        y_data (array): observed y values
        dy_data (array): uncertainty of each y point
        model_func (callable): model function f(x, a, b)

    Returns:
        float: log-likelihood value
    """
    chi2 = 0.0
    for xi, yi, dyi in zip(x_data, y_data, dy_data):
        model_val = model_func(xi, a, b)
        chi2 += ((model_val - yi) ** 2) / (dyi ** 2)
    return -0.5 * chi2

# --------------------------
# Random number generators
# --------------------------

def rand_uniform(a=0.0, b=1.0):
    """
    Generate a random number uniformly in [a, b)
    """
    return a + (b - a) * np.random.random()


def rand_normal(mu=0.0, sigma=1.0):
    """
    Generate a normal random number using Box-Muller transform
    """
    u1 = np.random.random()
    u2 = np.random.random()
    z0 = math.sqrt(-2 * math.log(u1)) * math.cos(2 * math.pi * u2)
    return mu + z0 * sigma
