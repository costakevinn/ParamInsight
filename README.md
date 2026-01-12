# ParamInsight: 2D Bayesian Parameter Inference with MCMC

**ParamInsight** is a Python framework for performing **Bayesian parameter inference** on custom 2-parameter models using **Metropolis-Hastings MCMC**.  
It integrates **statistical modeling, uncertainty quantification, and data visualization** in a reproducible pipeline.

---

## Mathematical Model

Observational data `(x_i, y_i)` with measurement uncertainties `dy_i` are modeled assuming **independent Gaussian errors**.  

The **log-likelihood function** is:

```
log L(a, b) = -1/2 * sum_i [ (y_i - F(x_i; a, b))^2 / dy_i^2 ]
```

Optimizing this function corresponds to standard **chi-square minimization** in statistical parameter estimation.

---

## MCMC Sampling

A **Metropolis-Hastings sampler with momentum (n−1 / n−2 memory)** explores the parameter space:

```
a' = a_(n-1) + 0.5 * (a_(n-1) - a_(n-2)) + N(0, σ)
b' = b_(n-1) + 0.5 * (b_(n-1) - b_(n-2)) + N(0, σ)
```

**Acceptance probability:**

```
α = min(1, exp(logL_new - logL_old))
```

This approach improves sampling efficiency for **correlated parameters**.

---

## Example: Logarithmic Model

**Model:** `F(x) = a * log(b * x)`  
**True parameters:** `a = 1.5`, `b = 0.5`

**Estimated parameters (example run):**

| Parameter | Mean ± Std | True | % Error |
|-----------|------------|------|---------|
| a         | 1.51 ± 0.05 | 1.50 | 0.67%  |
| b         | 0.48 ± 0.03 | 0.50 | 4.00%  |

**Posterior diagnostics:**

| Trace | Histogram | Scatter |
|-------|-----------|---------|
| ![](plots/logarithmic/trace.png) | ![](plots/logarithmic/histogram.png) | ![](plots/logarithmic/scatter.png) |

---

## Features

- Custom 2-parameter models: linear, logarithmic, quadratic, inverse  
- Fully implemented **Metropolis-Hastings MCMC**  
- Momentum-based proposals for correlated parameters  
- Gaussian noise generation using Box-Muller transform  
- Heteroscedastic observational uncertainties  
- Automatic generation of:
  - MCMC chains (`.npz`)  
  - Trace plots  
  - Posterior histograms  
  - Parameter correlation plots  
- Reproducible results via fixed random seed

---

## Usage

```bash
python3 main.py
```

Outputs are stored in:

- `data/` — observational data and MCMC chains  
- `plots/` — visual diagnostics of parameter sampling  
- `results/` — final parameter estimates and error metrics  

---

## Applications

- Bayesian parameter estimation  
- Statistical modeling and uncertainty quantification  
- Analysis of experimental or synthetic data  
- Exploration of correlated parameter spaces  

---

## License

MIT License — see `LICENSE` for details.

