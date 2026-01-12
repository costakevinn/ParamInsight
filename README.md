# ParamInsight: Bayesian Parameter Inference Tool (2D)

ParamInsight is a Python tool for **Bayesian parameter inference** using **Metropolis–Hastings MCMC** on 2-parameter models.
It provides a fully reproducible pipeline for generating observational data, evaluating likelihoods, sampling posterior distributions, and producing statistical visualizations.

---

## Key Features

* Probabilistic modeling with **custom 2D models**: linear, logarithmic, quadratic, inverse.
* **Bayesian inference** via Metropolis–Hastings MCMC with memory (momentum) for improved chain exploration.
* **Uncertainty quantification** using heteroscedastic observational errors and Gaussian noise (Box–Muller).
* Automatic generation of:

  * MCMC chains (`.npz`)
  * Trace plots
  * Posterior histograms
  * Parameter correlation scatter plots
  * Final parameter estimates with mean ± std and error metrics
* Optimized for **numerical analysis**, **statistical modeling**, and **data-driven insights**.

---

## Mathematical Model

Observational data `(x_i, y_i)` with uncertainties `dy_i` are modeled with a function `F(x; a, b)` under **independent Gaussian noise**.

The log-likelihood is:

```
log L(a, b) = -1/2 * sum_i [ (y_i - F(x_i; a, b))^2 / dy_i^2 ]
```

Maximizing this is equivalent to minimizing the **chi-square statistic**.

---

## MCMC Algorithm

The sampler implements **Metropolis–Hastings** with optional **n−1 / n−2 memory**:

```
a' = a_(n-1) + 0.5 * (a_(n-1) - a_(n-2)) + N(0, σ)
b' = b_(n-1) + 0.5 * (b_(n-1) - b_(n-2)) + N(0, σ)
α = min(1, exp(logL_new - logL_old))
```

This enhances exploration in **correlated parameter spaces**, producing robust posterior estimates.

---

## Example: Logarithmic Model

**Model:** `F(x) = a * log(b * x)`

**True parameters:** `a = 1.5, b = 0.5`

**Estimated parameters (example run):**

| Parameter | Mean ± Std  | True | % Error |
| --------- | ----------- | ---- | ------- |
| a         | 1.51 ± 0.05 | 1.50 | 0.67%   |
| b         | 0.48 ± 0.03 | 0.50 | 4.00%   |

**Posterior diagnostics:**

| Trace                            | Histogram                            | Scatter                            |
| -------------------------------- | ------------------------------------ | ---------------------------------- |
| ![](plots/logarithmic/trace.png) | ![](plots/logarithmic/histogram.png) | ![](plots/logarithmic/scatter.png) |

---

## Usage

```bash
python3 main.py
```

Outputs are stored in:

* `data/` — observations and chains
* `plots/` — trace, histogram, and scatter visualizations
* `results/` — numerical summaries and parameter estimates

---

## Applications

* **Statistical modeling** and parameter estimation.
* **Data analysis** pipelines with uncertainty quantification.
* **Probabilistic modeling** and stochastic simulations.
* **Reproducible research** in Bayesian inference.

---

## License

MIT License — see `LICENSE` for details.
