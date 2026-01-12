# examples.py
# MCMC examples for ParamInsight (2D)
# Four test models: Linear, Logarithmic, Quadratic, Inverse
# Observational errors (dy) are generated with statistical justification:
# - instrument_error: base error from measurement device
# - trend_error: variation with x
# - gaussian_noise: random observational noise
# Saves: observations, MCMC chain, plots, final results

import numpy as np
import os
from mcmc import mcmc_2d
from utils import save_chain, plot_trace, plot_histogram, plot_scatter, save_final_results

# -------------------------------
# Helper function to generate observational data
# -------------------------------

def generate_observational_data(model_func, x_data, true_a, true_b,
                                instrument_error, trend_coeff, noise_sigma):
    """
    Generate y_data and dy_data with realistic observational errors
    """
    # Trend in uncertainty with x
    trend_error = trend_coeff * x_data

    # Random Gaussian noise for uncertainty
    gaussian_noise = np.random.normal(0, noise_sigma, size=len(x_data))

    # Total uncertainty per point (dy must be positive)
    dy_data = np.abs(instrument_error + trend_error + gaussian_noise)

    # Observed y with noise proportional to dy
    y_data = model_func(x_data, true_a, true_b) + np.random.normal(0, dy_data)

    return y_data, dy_data

# -------------------------------
# Models
# -------------------------------

def linear_model(x, a, b):
    return a*x + b

def log_model(x, a, b):
    return a * np.log(b*x)

def quadratic_model(x, a, b):
    return a*x + b*x**2

def inverse_model(x, a, b):
    return a/x + b

# -------------------------------
# Data for each example
# -------------------------------

x_linear = np.linspace(0, 10, 20)
true_a_linear, true_b_linear = 2.0, 1.0
y_linear, dy_linear = generate_observational_data(
    linear_model, x_linear, true_a_linear, true_b_linear,
    instrument_error=0.2, trend_coeff=0.05, noise_sigma=0.05
)

x_log = np.linspace(1, 10, 20)
true_a_log, true_b_log = 1.5, 0.5
y_log, dy_log = generate_observational_data(
    log_model, x_log, true_a_log, true_b_log,
    instrument_error=0.3, trend_coeff=0.02, noise_sigma=0.05
)

x_quad = np.linspace(0, 5, 20)
true_a_quad, true_b_quad = 1.0, 0.2
y_quad, dy_quad = generate_observational_data(
    quadratic_model, x_quad, true_a_quad, true_b_quad,
    instrument_error=0.2, trend_coeff=0.05, noise_sigma=0.03
)

x_inv = np.linspace(1, 10, 20)
true_a_inv, true_b_inv = 5.0, 1.0
y_inv, dy_inv = generate_observational_data(
    inverse_model, x_inv, true_a_inv, true_b_inv,
    instrument_error=0.3, trend_coeff=0.01, noise_sigma=0.02
)

# -------------------------------
# Generic runner for examples
# -------------------------------

def run_example(x_data, y_data, dy_data, model_func, true_a, true_b, name):
    """
    Run a single MCMC example
    Saves: observations, chain, plots, final results
    """
    data_path = f"data/{name}"
    plots_path = f"plots/{name}"
    results_path = f"results/{name}"
    os.makedirs(data_path, exist_ok=True)
    os.makedirs(plots_path, exist_ok=True)
    os.makedirs(results_path, exist_ok=True)

    print(f"Running {name} example with variable uncertainty...")

    # Save observational data
    obs_file = f"{data_path}/observations.txt"
    with open(obs_file, "w") as f:
        f.write("x\t y\t dy\n")
        for xi, yi, dyi in zip(x_data, y_data, dy_data):
            f.write(f"{xi:.4f}\t{yi:.4f}\t{dyi:.4f}\n")
    print(f"Observational data saved in {obs_file}")

    # Run MCMC
    chain, loglikes = mcmc_2d(x_data, y_data, dy_data, model_func,
                               a_init=0.0, b_init=0.0, n_steps=5000, scale=0.1)

    # Save chain and plots
    save_chain(chain, loglikes, f"{data_path}/chain.npz")
    plot_trace(chain, ["a","b"], f"{plots_path}/trace.png")
    plot_histogram(chain, ["a","b"], f"{plots_path}/histogram.png")
    plot_scatter(chain, ["a","b"], f"{plots_path}/scatter.png")

    # Compute statistics
    mean_a, mean_b = np.mean(chain[:,0]), np.mean(chain[:,1])
    std_a, std_b = np.std(chain[:,0]), np.std(chain[:,1])
    abs_err_a = abs(mean_a - true_a)
    abs_err_b = abs(mean_b - true_b)
    perc_err_a = abs_err_a / true_a * 100
    perc_err_b = abs_err_b / true_b * 100

    # Save final results
    result_file = f"{results_path}/final_results.txt"
    with open(result_file, "w") as f:
        f.write("Observational data (x, y, dy):\n")
        f.write("x\t y\t dy\n")
        for xi, yi, dyi in zip(x_data, y_data, dy_data):
            f.write(f"{xi:.4f}\t{yi:.4f}\t{dyi:.4f}\n")
        f.write("\nEstimated parameters vs True values:\n")
        f.write(f"a: {mean_a:.4f} ± {std_a:.4f} (true: {true_a})\n")
        f.write(f"b: {mean_b:.4f} ± {std_b:.4f} (true: {true_b})\n\n")
        f.write("Absolute errors:\n")
        f.write(f"|a - true_a| = {abs_err_a:.4f}\n")
        f.write(f"|b - true_b| = {abs_err_b:.4f}\n\n")
        f.write("Percentage errors:\n")
        f.write(f"% error a = {perc_err_a:.2f}%\n")
        f.write(f"% error b = {perc_err_b:.2f}%\n")
    print(f"Final results saved in {result_file}")

    # Print summary
    print(f"\nEstimated parameters vs True values:")
    print(f"a: {mean_a:.4f} ± {std_a:.4f} (true: {true_a})")
    print(f"b: {mean_b:.4f} ± {std_b:.4f} (true: {true_b})\n")
    print(f"Absolute errors: |a - true_a| = {abs_err_a:.4f}, |b - true_b| = {abs_err_b:.4f}")
    print(f"Percentage errors: %a = {perc_err_a:.2f}%, %b = {perc_err_b:.2f}%\n")
    print(f"Example finished! Check {data_path}/, {plots_path}/ and {results_path}/ folders.")

# -------------------------------
# Specific runners
# -------------------------------

def run_linear_example():
    run_example(x_linear, y_linear, dy_linear, linear_model,
                true_a_linear, true_b_linear, "linear")

def run_log_example():
    run_example(x_log, y_log, dy_log, log_model,
                true_a_log, true_b_log, "logarithmic")

def run_quadratic_example():
    run_example(x_quad, y_quad, dy_quad, quadratic_model,
                true_a_quad, true_b_quad, "quadratic")

def run_inverse_example():
    run_example(x_inv, y_inv, dy_inv, inverse_model,
                true_a_inv, true_b_inv, "inverse")
