# examples.py
"""
MCMC application examples for ParamInsight (2D)

Uses Momentum Metropolis–Hastings MCMC with heteroscedastic
observational uncertainties.

Four test models:
- Linear
- Logarithmic
- Quadratic
- Inverse

For each example, the script:
- Generates synthetic observations with realistic uncertainties
- Runs Momentum MCMC
- Saves observational data, chains, plots, and final statistics
"""

import numpy as np
import os

from mcmc import mcmc_2d
from utils import (
    save_chain,
    plot_trace,
    plot_histogram,
    plot_scatter,
    save_final_results
)

# -------------------------------------------------
# Reproducibility
# -------------------------------------------------

np.random.seed(42)

# -------------------------------------------------
# Helper function: generate observational data
# -------------------------------------------------

def generate_observational_data(model_func, x_data, true_a, true_b,
                                instrument_error, trend_coeff, noise_sigma):
    """
    Generate synthetic observations with heteroscedastic uncertainties.

    Uncertainty model:
        dy = instrument_error + trend_coeff * x + Gaussian noise

    Parameters:
        model_func (callable): f(x, a, b)
        x_data (array-like): independent variable
        true_a, true_b (float): true model parameters
        instrument_error (float): baseline measurement uncertainty
        trend_coeff (float): linear trend in uncertainty with x
        noise_sigma (float): standard deviation of random noise in dy

    Returns:
        y_data (np.ndarray): observed values
        dy_data (np.ndarray): observational uncertainties
    """
    trend_error = trend_coeff * x_data
    gaussian_noise = np.random.normal(0.0, noise_sigma, size=len(x_data))

    # Ensure dy > 0
    dy_data = np.abs(instrument_error + trend_error + gaussian_noise)

    # Observed y values
    y_data = model_func(x_data, true_a, true_b) \
             + np.random.normal(0.0, dy_data)

    return y_data, dy_data

# -------------------------------------------------
# Models
# -------------------------------------------------

def linear_model(x, a, b):
    return a * x + b

def log_model(x, a, b):
    return a * np.log(b * x)

def quadratic_model(x, a, b):
    return a * x + b * x**2

def inverse_model(x, a, b):
    return a / x + b

# -------------------------------------------------
# Synthetic data for each example
# -------------------------------------------------

# Linear
x_linear = np.linspace(0, 10, 20)
true_a_linear, true_b_linear = 2.0, 1.0
y_linear, dy_linear = generate_observational_data(
    linear_model, x_linear,
    true_a_linear, true_b_linear,
    instrument_error=0.2,
    trend_coeff=0.05,
    noise_sigma=0.05
)

# Logarithmic
x_log = np.linspace(1, 10, 20)
true_a_log, true_b_log = 1.5, 0.5
y_log, dy_log = generate_observational_data(
    log_model, x_log,
    true_a_log, true_b_log,
    instrument_error=0.3,
    trend_coeff=0.02,
    noise_sigma=0.05
)

# Quadratic
x_quad = np.linspace(0, 5, 20)
true_a_quad, true_b_quad = 1.0, 0.2
y_quad, dy_quad = generate_observational_data(
    quadratic_model, x_quad,
    true_a_quad, true_b_quad,
    instrument_error=0.2,
    trend_coeff=0.05,
    noise_sigma=0.03
)

# Inverse
x_inv = np.linspace(1, 10, 20)
true_a_inv, true_b_inv = 5.0, 1.0
y_inv, dy_inv = generate_observational_data(
    inverse_model, x_inv,
    true_a_inv, true_b_inv,
    instrument_error=0.3,
    trend_coeff=0.01,
    noise_sigma=0.02
)

# -------------------------------------------------
# Generic runner
# -------------------------------------------------

def run_example(x_data, y_data, dy_data, model_func,
                true_a, true_b, name,
                n_steps=int(1e5), burn_in=1000,
                scale=0.1, rho=0.9):
    """
    Run a Momentum MCMC example and save all outputs.

    Parameters:
        burn_in (int): number of initial samples to discard
        scale (float): proposal / momentum noise scale
        rho (float): momentum persistence parameter
    """

    data_path = f"data/{name}"
    plots_path = f"plots/{name}"
    results_path = f"results/{name}"

    os.makedirs(data_path, exist_ok=True)
    os.makedirs(plots_path, exist_ok=True)
    os.makedirs(results_path, exist_ok=True)

    print(f"\nRunning '{name}' example (Momentum MCMC)...")

    # -------------------------------------------------
    # Save observational data
    # -------------------------------------------------

    obs_file = f"{data_path}/observations.txt"
    with open(obs_file, "w") as f:
        f.write("x\t y\t dy\n")
        for xi, yi, dyi in zip(x_data, y_data, dy_data):
            f.write(f"{xi:.6f}\t{yi:.6f}\t{dyi:.6f}\n")

    # -------------------------------------------------
    # Run MCMC
    # -------------------------------------------------

    chain, loglikes = mcmc_2d(
        x_data, y_data, dy_data, model_func,
        a_init=0.0,
        b_init=0.0,
        n_steps=n_steps,
        scale=scale,
        rho=rho
    )

    # -------------------------------------------------
    # Burn-in removal
    # -------------------------------------------------

    chain_post = chain[burn_in:]
    loglikes_post = loglikes[burn_in:]

    # -------------------------------------------------
    # Save chain and plots
    # -------------------------------------------------

    save_chain(chain, loglikes, f"{data_path}/chain.npz")

    plot_trace(chain, ["a", "b"], f"{plots_path}/trace.png")
    plot_histogram(chain_post, ["a", "b"], f"{plots_path}/histogram.png")
    plot_scatter(chain_post, ["a", "b"], f"{plots_path}/scatter.png")

    # -------------------------------------------------
    # Statistics
    # -------------------------------------------------

    mean_a, mean_b = np.mean(chain_post[:, 0]), np.mean(chain_post[:, 1])
    std_a = np.std(chain_post[:, 0], ddof=1)
    std_b = np.std(chain_post[:, 1], ddof=1)

    abs_err_a = abs(mean_a - true_a)
    abs_err_b = abs(mean_b - true_b)

    perc_err_a = abs_err_a / abs(true_a) * 100.0
    perc_err_b = abs_err_b / abs(true_b) * 100.0

    # -------------------------------------------------
    # Save final results
    # -------------------------------------------------

    result_file = f"{results_path}/final_results.txt"
    with open(result_file, "w") as f:
        f.write("Momentum MCMC results\n")
        f.write("--------------------\n\n")

        f.write("MCMC configuration:\n")
        f.write(f"Steps     = {n_steps}\n")
        f.write(f"Burn-in   = {burn_in}\n")
        f.write(f"Scale     = {scale}\n")
        f.write(f"Rho       = {rho}\n\n")

        f.write("Estimated parameters (posterior mean ± std):\n")
        f.write(f"a = {mean_a:.6f} ± {std_a:.6f} (true: {true_a})\n")
        f.write(f"b = {mean_b:.6f} ± {std_b:.6f} (true: {true_b})\n\n")

        f.write("Errors:\n")
        f.write(f"|a - true| = {abs_err_a:.6f} ({perc_err_a:.2f} %)\n")
        f.write(f"|b - true| = {abs_err_b:.6f} ({perc_err_b:.2f} %)\n")

    # -------------------------------------------------
    # Console summary
    # -------------------------------------------------

    print("Estimated parameters:")
    print(f"a = {mean_a:.4f} ± {std_a:.4f} (true: {true_a})")
    print(f"b = {mean_b:.4f} ± {std_b:.4f} (true: {true_b})")

    print(f"Absolute errors: a = {abs_err_a:.4e}, b = {abs_err_b:.4e}")
    print(f"Percentage errors: a = {perc_err_a:.2f} %, b = {perc_err_b:.2f} %")

    print(f"Results saved in '{results_path}/'")
    print(f"Plots saved in '{plots_path}/'")

# -------------------------------------------------
# Specific runners
# -------------------------------------------------

def run_linear_example():
    run_example(
        x_linear, y_linear, dy_linear,
        linear_model,
        true_a_linear, true_b_linear,
        name="linear"
    )

def run_log_example():
    run_example(
        x_log, y_log, dy_log,
        log_model,
        true_a_log, true_b_log,
        name="logarithmic"
    )

def run_quadratic_example():
    run_example(
        x_quad, y_quad, dy_quad,
        quadratic_model,
        true_a_quad, true_b_quad,
        name="quadratic"
    )

def run_inverse_example():
    run_example(
        x_inv, y_inv, dy_inv,
        inverse_model,
        true_a_inv, true_b_inv,
        name="inverse"
    )
