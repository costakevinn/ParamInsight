# distributions.py
"""
Distributions and likelihood functions for ParamInsight (2D)

Assumptions:
- Independent Gaussian observational errors
- Heteroscedastic uncertainties provided by the user
- dy_data is assumed to be strictly positive and correctly specified

This module intentionally avoids automatic validation of uncertainties.
"""

import numpy as np

# -------------------------------------------------
# Log-likelihood
# -------------------------------------------------

def log_likelihood_2d(a, b, x_data, y_data, dy_data, model_func):
    """
    Compute Gaussian log-likelihood via chi-square statistic:

        log L = -1/2 * sum_i [ (y_i - f(x_i; a, b))^2 / dy_i^2 ]

    Parameters:
        a, b (float): model parameters
        x_data (array-like): observed x values
        y_data (array-like): observed y values
        dy_data (array-like): observational uncertainties (assumed valid)
        model_func (callable): model function f(x, a, b)

    Returns:
        float: log-likelihood value
    """
    x = np.asarray(x_data)
    y = np.asarray(y_data)
    dy = np.asarray(dy_data)

    model_y = model_func(x, a, b)

    if not np.all(np.isfinite(model_y)):
        return -np.inf

    chi2 = np.sum(((y - model_y) / dy) ** 2)

    return -0.5 * chi2

# -------------------------------------------------
# Random number generators (Box–Muller)
# -------------------------------------------------

def rand_uniform(a=0.0, b=1.0):
    """
    Generate a uniform random number in [a, b).
    """
    return a + (b - a) * np.random.random()


def rand_normal(mu=0.0, sigma=1.0):
    """
    Generate a normal random number using the Box–Muller transform.

    Notes:
        Implemented explicitly for educational and transparency purposes.
        No reliance on NumPy's normal RNG.
    """
    u1 = max(np.random.random(), 1e-12)
    u2 = np.random.random()


    z0 = np.sqrt(-2.0 * np.log(u1)) * np.cos(2.0 * np.pi * u2)
    return mu + sigma * z0
