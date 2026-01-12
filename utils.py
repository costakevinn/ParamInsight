# utils.py
"""
Utility functions for ParamInsight

- Save MCMC chains
- Plot trace, histogram, and scatter plots
- Save final parameter estimates

Plots are styled for professional / market-facing reports.
"""

import numpy as np
import matplotlib.pyplot as plt
import os

# -------------------------------------------------
# Internal plotting style (local)
# -------------------------------------------------

_MARKET_STYLE = {
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "axes.edgecolor": "#333333",
    "axes.labelcolor": "#333333",
    "axes.titlesize": 14,
    "axes.titleweight": "semibold",
    "axes.labelsize": 12,
    "xtick.color": "#555555",
    "ytick.color": "#555555",
    "grid.color": "#dddddd",
    "grid.linestyle": "--",
    "grid.linewidth": 0.6,
    "legend.frameon": False,
    "font.size": 11,
}

# -------------------------------------------------
# Save MCMC chain
# -------------------------------------------------

def save_chain(chain, loglikes, filename):
    np.savez(filename, chain=chain, loglikes=loglikes)
    print(f"Chain saved in {filename}")

# -------------------------------------------------
# Trace plot
# -------------------------------------------------

def plot_trace(chain, param_names, save_path, burn_in=0):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    chain_plot = chain[burn_in:]

    with plt.rc_context(_MARKET_STYLE):
        plt.figure(figsize=(12, 5))
        for i, name in enumerate(param_names):
            plt.plot(chain_plot[:, i], label=name, linewidth=1.2)

        plt.xlabel("MCMC step")
        plt.ylabel("Parameter value")
        plt.title("MCMC Trace")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.savefig(save_path, dpi=200)
        plt.close()

    print(f"Trace plot saved in {save_path}")

# -------------------------------------------------
# Histogram
# -------------------------------------------------

def plot_histogram(chain, param_names, save_path, burn_in=0, bins=30):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    chain_plot = chain[burn_in:]

    with plt.rc_context(_MARKET_STYLE):
        plt.figure(figsize=(8, 5))

        colors = ["#1f77b4", "#ff7f0e"]

        for i, name in enumerate(param_names):
            plt.hist(
                chain_plot[:, i],
                bins=bins,
                density=True,
                histtype="stepfilled",
                alpha=0.6,
                linewidth=1.2,
                label=name,
                color=colors[i]
            )

        plt.xlabel("Parameter value")
        plt.ylabel("Posterior density")
        plt.title("Posterior Distributions")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.savefig(save_path, dpi=200)
        plt.close()

    print(f"Histogram saved in {save_path}")

# -------------------------------------------------
# Scatter plot
# -------------------------------------------------

def plot_scatter(chain, param_names, save_path, burn_in=0):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    chain_plot = chain[burn_in:]

    with plt.rc_context(_MARKET_STYLE):
        plt.figure(figsize=(6, 6))
        plt.scatter(
            chain_plot[:, 0],
            chain_plot[:, 1],
            s=10,
            alpha=0.35,
            color="#1f77b4",
            rasterized=True
        )

        plt.xlabel(param_names[0])
        plt.ylabel(param_names[1])
        plt.title("Parameter Correlation")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(save_path, dpi=200)
        plt.close()

    print(f"Scatter plot saved in {save_path}")

# -------------------------------------------------
# Save final results
# -------------------------------------------------

def save_final_results(chain, param_names, save_path, burn_in=0):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    chain_post = chain[burn_in:]

    with open(save_path, "w") as f:
        f.write("Final estimated parameters (posterior mean ± std):\n\n")
        for i, name in enumerate(param_names):
            mean = np.mean(chain_post[:, i])
            std = np.std(chain_post[:, i], ddof=1)
            f.write(f"{name}: {mean:.6f} ± {std:.6f}\n")

    print(f"Final results saved in {save_path}")
