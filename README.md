# ParamInsight: MCMC Parameter Inference Tool (2D)

**ParamInsight** is a lightweight, fully transparent Python framework for **Bayesian parameter inference** using **Markov Chain Monte Carlo (MCMC)** methods.

The project is designed to:
- Implement **Metropolis–Hastings MCMC from scratch**
- Infer parameters of **generic 2-parameter models**
- Generate realistic observational data with **heteroscedastic uncertainties**
- Provide clean statistical analysis and publication-ready visualizations

This repository demonstrates solid skills in **statistical modeling, stochastic processes, numerical methods, and scientific Python**.

---

## Core Methodology

### Bayesian Inference

Given observational data \((x_i, y_i, \delta y_i)\) and a model \(f(x; a, b)\), ParamInsight evaluates the Gaussian log-likelihood:

\[
\log L(a, b) = -\frac{1}{2} \sum_i \left( \frac{y_i - f(x_i; a, b)}{\delta y_i} \right)^2
\]

This likelihood is explored using **Metropolis–Hastings MCMC**, producing samples from the posterior distribution of parameters \((a, b)\).

---

### MCMC Algorithm

- Metropolis–Hastings sampler
- Proposal distribution: Gaussian (Box–Muller)
- **n−1 / n−2 memory (inertia)** to improve exploration efficiency
- Outputs:
  - Parameter chains
  - Log-likelihood evolution

Implemented in `mcmc.py`.

---

### Random Number Generation

Gaussian noise is generated explicitly using the **Box–Muller transform**, ensuring full transparency and avoiding black-box RNG behavior.

Implemented in `distributions.py`.

---

## Observational Data Model

ParamInsight generates **realistic observational datasets** with heteroscedastic uncertainties:

\[
\delta y_i = |\text{instrument error} + \text{trend}(x_i) + \text{Gaussian noise}|
\]

Observed values are then generated as:

\[
y_i = f(x_i; a_{\text{true}}, b_{\text{true}}) + \mathcal{N}(0, \delta y_i)
\]

This approach mimics real experimental and observational conditions.

---

## Implemented Models

The project includes four example models:

- **Linear:** \( f(x) = a x + b \)
- **Logarithmic:** \( f(x) = a \log(bx) \)
- **Quadratic:** \( f(x) = ax + bx^2 \)
- **Inverse:** \( f(x) = \frac{a}{x} + b \)

Each example:
- Generates observational data
- Runs MCMC inference
- Saves chains, plots, and statistical summaries

---

## Example: Logarithmic Model

**Model**
\[
f(x) = a \log(bx)
\]

**True parameters**
- \(a = 1.5\)
- \(b = 0.5\)

**Outputs**
- Trace plot (chain evolution)
- Posterior histograms
- Parameter correlation scatter plot

| Trace | Histogram | Scatter |
|------|-----------|---------|
| ![Trace](plots/logarithmic/trace.png) | ![Histogram](plots/logarithmic/histogram.png) | ![Scatter](plots/logarithmic/scatter.png) |

---

## Project Structure

```text
ParamInsight/
├── distributions.py   # Likelihood and RNG (Box–Muller)
├── mcmc.py            # Metropolis–Hastings with memory
├── examples.py        # Models and data generation
├── utils.py           # Plotting and result saving
├── main.py            # Entry point
├── data/              # Observational data and chains
├── plots/             # Trace, histogram, scatter plots
├── results/           # Final parameter analysis
└── docs/              # Technical documentation (PDF)
