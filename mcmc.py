# mcmc.py
"""
MCMC module for ParamInsight (2D)
- Metropolis-Hastings with n-1/n-2 memory
- Returns chains for later analysis
"""

import numpy as np
from distributions import rand_normal, log_likelihood_2d

def mcmc_2d(x_data, y_data, dy_data, model_func,
            a_init, b_init, n_steps=5000, scale=0.1):
    """
    2-parameter MCMC (a, b) with n-1/n-2 memory.

    Parameters:
        x_data, y_data, dy_data (arrays): observed data
        model_func (callable): model function f(x, a, b)
        a_init, b_init (float): initial parameter values
        n_steps (int): number of MCMC steps
        scale (float): proposal step scale

    Returns:
        chain (np.array): shape (n_steps, 2) parameter chain
        loglikes (np.array): corresponding log-likelihoods
    """
    # Initialize arrays
    chain = np.zeros((n_steps, 2))
    loglikes = np.zeros(n_steps)
    
    # Initial step
    chain[0, 0] = a_init
    chain[0, 1] = b_init
    loglikes[0] = log_likelihood_2d(a_init, b_init, x_data, y_data, dy_data, model_func)
    
    # Second step (no n-2 memory yet)
    a_new = a_init + rand_normal(0, scale)
    b_new = b_init + rand_normal(0, scale)
    loglike_new = log_likelihood_2d(a_new, b_new, x_data, y_data, dy_data, model_func)
    
    alpha = min(1, np.exp(loglike_new - loglikes[0]))
    if np.random.random() < alpha:
        chain[1, 0] = a_new
        chain[1, 1] = b_new
        loglikes[1] = loglike_new
    else:
        chain[1, :] = chain[0, :]
        loglikes[1] = loglikes[0]
    
    # Subsequent steps with n-1/n-2 memory
    for i in range(2, n_steps):
        a_prev1, b_prev1 = chain[i-1]
        a_prev2, b_prev2 = chain[i-2]
        
        # Proposal using memory inertia
        a_prop = a_prev1 + 0.5*(a_prev1 - a_prev2) + rand_normal(0, scale)
        b_prop = b_prev1 + 0.5*(b_prev1 - b_prev2) + rand_normal(0, scale)
        
        loglike_prop = log_likelihood_2d(a_prop, b_prop, x_data, y_data, dy_data, model_func)
        
        # Metropolis-Hastings acceptance
        alpha = min(1, np.exp(loglike_prop - loglikes[i-1]))
        if np.random.random() < alpha:
            chain[i, 0] = a_prop
            chain[i, 1] = b_prop
            loglikes[i] = loglike_prop
        else:
            chain[i, :] = chain[i-1, :]
            loglikes[i] = loglikes[i-1]
    
    return chain, loglikes
