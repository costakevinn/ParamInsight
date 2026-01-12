# utils.py
"""
Utility functions for ParamInsight
- Save MCMC chains
- Plot trace, histogram, and scatter plots with modern aesthetics
- Save final results with analysis
"""

import numpy as np
import matplotlib.pyplot as plt
import os

# --------------------------
# Save MCMC chain
# --------------------------
def save_chain(chain, loglikes, filename):
    """Save MCMC chain and log-likelihoods as a .npz file"""
    np.savez(filename, chain=chain, loglikes=loglikes)
    print(f"Chain saved in {filename}")

# --------------------------
# Trace plot
# --------------------------
def plot_trace(chain, param_names, save_path):
    """Trace plot for each parameter across MCMC steps"""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    plt.style.use("ggplot")
    plt.figure(figsize=(12,5))
    
    for i, name in enumerate(param_names):
        plt.plot(chain[:,i], label=name, linewidth=1.5)
    
    plt.xlabel("Step", fontsize=12)
    plt.ylabel("Parameter value", fontsize=12)
    plt.title("MCMC Trace", fontsize=14)
    plt.legend(frameon=True)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Trace plot saved in {save_path}")

# --------------------------
# Histogram
# --------------------------
def plot_histogram(chain, param_names, save_path):
    """Histogram of posterior samples for each parameter"""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    plt.style.use("ggplot")
    plt.figure(figsize=(8,5))
    
    colors = ["#1f77b4", "#ff7f0e"]  # blue, orange
    
    for i, name in enumerate(param_names):
        plt.hist(chain[:,i], bins=30, alpha=0.7, label=name, color=colors[i], edgecolor='black')
    
    plt.xlabel("Parameter value", fontsize=12)
    plt.ylabel("Frequency", fontsize=12)
    plt.title("Posterior Distribution", fontsize=14)
    plt.legend(frameon=True)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Histogram saved in {save_path}")

# --------------------------
# Scatter plot (a vs b)
# --------------------------
def plot_scatter(chain, param_names, save_path):
    """Scatter plot showing correlation between parameters"""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    plt.style.use("ggplot")
    plt.figure(figsize=(6,6))
    
    plt.scatter(chain[:,0], chain[:,1], s=20, alpha=0.6, color="#2ca02c")
    plt.xlabel(param_names[0], fontsize=12)
    plt.ylabel(param_names[1], fontsize=12)
    plt.title("Parameter Correlation", fontsize=14)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Scatter plot saved in {save_path}")

# --------------------------
# Save final results
# --------------------------
def save_final_results(chain, param_names, save_path):
    """
    Save mean ± std for each parameter in a text file
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    with open(save_path, "w") as f:
        f.write("Final estimated parameters:\n")
        for i, name in enumerate(param_names):
            mean = np.mean(chain[:,i])
            std = np.std(chain[:,i])
            f.write(f"{name}: {mean:.4f} ± {std:.4f}\n")
    
    print(f"Final results saved in {save_path}")
