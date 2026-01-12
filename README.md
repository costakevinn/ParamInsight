# ParamInsight: MCMC Parameter Inference Tool (2D)

ParamInsight is a lightweight Python tool for performing **Bayesian parameter inference**
using **MCMC (Metropolis–Hastings)** on custom 2-parameter models.

The project is designed for **clarity, transparency, and reproducibility**, showcasing
a full MCMC pipeline implemented **from scratch**, including likelihood evaluation,
Gaussian noise generation (Box–Muller), and statistical visualization.

---

## Mathematical Model

ParamInsight assumes **independent Gaussian observational errors**.

Given:
- observed data points `(x_i, y_i)`
- uncertainties `dy_i`
- a model function `F(x; a, b)`

The log-likelihood is defined as:

```
log L(a, b) = -1/2 * sum_i [ (y_i - F(x_i; a, b))^2 / dy_i^2 ]
```

Maximizing this quantity is equivalent to minimizing the chi-square statistic.

---

## MCMC Algorithm

ParamInsight uses a **Metropolis–Hastings sampler with memory (momentum)**.

At step `n`, a proposal is generated using:

```
a' = a_(n-1) + 0.5 * (a_(n-1) - a_(n-2)) + N(0, σ)
b' = b_(n-1) + 0.5 * (b_(n-1) - b_(n-2)) + N(0, σ)
```

The proposal is accepted with probability:

```
α = min(1, exp(logL_new - logL_old))
```

This simple inertia term improves exploration of correlated parameter spaces.

---

## Logarithmic Example

**Model**
```
F(x) = a * log(b * x)
```

**True parameters**
```
a = 1.5
b = 0.5
```

**Estimated parameters (example run)**

| Parameter | Mean ± Std | True | % Error |
|----------|------------|------|---------|
| a | 1.51 ± 0.05 | 1.50 | 0.67% |
| b | 0.48 ± 0.03 | 0.50 | 4.00% |

**Posterior diagnostics**

| Trace | Histogram | Scatter |
|------|-----------|---------|
| ![](plots/logarithmic/trace.png) | ![](plots/logarithmic/histogram.png) | ![](plots/logarithmic/scatter.png) |

---

## Features

- Custom 2-parameter models (linear, logarithmic, quadratic, inverse)
- Metropolis–Hastings MCMC implemented from scratch
- Optional momentum (n−1 / n−2 memory)
- Box–Muller Gaussian noise generator
- Heteroscedastic observational uncertainties
- Automatic generation of:
  - MCMC chains
  - Trace plots
  - Posterior histograms
  - Parameter correlation plots
- Reproducible results via fixed random seed

---

## Usage

```
python3 main.py
```

All outputs are saved to:
- `data/` — observations and chains
- `plots/` — visual diagnostics
- `results/` — numerical summaries

---

## Purpose

ParamInsight is intended as:
- an **educational tool** for Bayesian inference
- a **portfolio project** demonstrating MCMC literacy
- a **transparent alternative** to black-box fitting tools

---

## License

MIT License — see `LICENSE` for details.
