# mcmc.py
"""
Momentum Metropolis–Hastings MCMC for 2-parameter models (a, b)

- Gaussian likelihood
- Persistent momentum
- Detailed balance preserved
- Returns parameter chains and log-likelihoods
"""

import numpy as np
from distributions import rand_normal, log_likelihood_2d

# -------------------------------------------------
# Momentum MCMC (2D)
# -------------------------------------------------

def mcmc_2d(x_data, y_data, dy_data, model_func,
            a_init, b_init,
            n_steps=5000, scale=0.1, rho=0.9):
    """
    2-parameter Momentum MCMC sampler.

    Parameters:
        x_data, y_data, dy_data (array-like): observed data
        model_func (callable): model f(x, a, b)
        a_init, b_init (float): initial parameter values
        n_steps (int): number of MCMC steps
        scale (float): momentum noise scale
        rho (float): momentum persistence (0 < rho < 1)

    Returns:
        chain (np.ndarray): shape (n_steps, 2)
        loglikes (np.ndarray): log-likelihood values
    """

    chain = np.zeros((n_steps, 2))
    loglikes = np.zeros(n_steps)

    # Initial state
    a, b = a_init, b_init
    v_a, v_b = 0.0, 0.0

    loglike = log_likelihood_2d(
        a, b, x_data, y_data, dy_data, model_func
    )

    for i in range(n_steps):

        # -----------------------------
        # Momentum update (persistent)
        # -----------------------------
        v_a = rho * v_a + rand_normal(0.0, scale)
        v_b = rho * v_b + rand_normal(0.0, scale)

        # Proposed parameters
        a_prop = a + v_a
        b_prop = b + v_b

        # Compute likelihood
        loglike_prop = log_likelihood_2d(
            a_prop, b_prop, x_data, y_data, dy_data, model_func
        )

        # -----------------------------
        # Metropolis acceptance
        # -----------------------------
        log_alpha = loglike_prop - loglike

        if np.log(np.random.random()) < log_alpha:
            # Accept
            a, b = a_prop, b_prop
            loglike = loglike_prop
        else:
            # Reject → reverse momentum (reversibility)
            v_a = -v_a
            v_b = -v_b

        chain[i, 0] = a
        chain[i, 1] = b
        loglikes[i] = loglike

    return chain, loglikes
