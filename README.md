# ParamInsight: MCMC Parameter Inference Tool (2D)

ParamInsight is a Python tool for performing **Bayesian parameter inference** using **MCMC (Metropolis–Hastings)** on custom 2-parameter models.  
It provides a complete workflow for **statistical modeling, data analysis, and uncertainty quantification** with reproducible outputs.

---

## Statistical Model

Assuming independent Gaussian observational errors:

- Observed data points: `(x_i, y_i)`
- Measurement uncertainties: `dy_i`
- Model function: `F(x; a, b)`

The log-likelihood is:

log L(a, b) = -1/2 * sum_i [ (y_i - F(x_i; a, b))^2 / dy_i^2 ]

Maximizing this corresponds to minimizing the chi-square statistic, a standard approach in statistical inference.

---

## MCMC Sampling

Metropolis–Hastings algorithm with momentum (n−1 / n−2 memory) for efficient exploration:

a' = a_(n-1) + 0.5 * (a_(n-1) - a_(n-2)) + N(0, σ)  
b' = b_(n-1) + 0.5 * (b_(n-1) - b_(n-2)) + N(0, σ)

Acceptance probability:

α = min(1, exp(logL_new - logL_old))

This approach handles correlated parameter spaces and improves convergence in 2D parameter inference.

---

## Example: Logarithmic Model

**Model:**  
F(x) = a * log(b * x)

**True parameters:**  
a = 1.5, b = 0.5

**Estimated parameters (sample run):**

| Parameter | Mean ± Std | True | % Error |
|-----------|------------|------|---------|
| a         | 1.51 ± 0.05 | 1.50 | 0.67% |
| b         | 0.48 ± 0.03 | 0.50 | 4.00% |

**Posterior Diagnostics:**

| Trace | Histogram | Scatter |
|-------|-----------|---------|
| ![](plots/logarithmic/trace.png) | ![](plots/logarithmic/histogram.png) | ![](plots/logarithmic/scatter.png) |

---

## Features

- Custom 2-parameter models: linear, logarithmic, quadratic, inverse  
- Bayesian inference via Metropolis–Hastings MCMC  
- Memory-based proposal to enhance exploration of correlated parameters  
- Gaussian noise generation (Box–Muller) for realistic observations  
- Handles heteroscedastic measurement uncertainties  
- Automatic generation of:
  - MCMC chains (`.npz`)
  - Trace plots
  - Posterior histograms
  - Parameter correlation plots  
- Fully reproducible using fixed random seed

---

## Usage

```bash
python3 main.py
```

Outputs are saved to:

- `data/` — observations and MCMC chains  
- `plots/` — visual diagnostics  
- `results/` — numerical summaries of inferred parameters

---

## License

MIT License — see `LICENSE` for details.

