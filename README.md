# ParamInsight

ParamInsight is a Python tool for **Bayesian parameter inference** using **MCMC (Metropolis-Hastings)**.  
It supports generic 2-parameter models and produces **synthetic observational data**, MCMC chains, **plots**, and **summary statistics**.

## Features

- **2D MCMC inference from scratch**  
- **Custom Gaussian noise** (Box-Muller)  
- **Multiple models supported**: Linear, Logarithmic, Quadratic, Inverse  
- **Comprehensive outputs**:  
  - Observational data (`x, y, dy`)  
  - MCMC chains  
  - Trace, histogram, and scatter plots  
  - Final parameter estimates with absolute & percentage errors  

## Usage

Run all examples:

```bash
python3 main.py
```

Outputs are saved in:

- `data/<model>/` → observations and chain  
- `plots/<model>/` → trace, histogram, scatter  
- `results/<model>/` → final parameter estimates  

### Example: Logarithmic Model

**Model:** `F(x) = a * log(b * x)`  
**True parameters:** `a = 1.5, b = 0.5`  

**Observational data with Gaussian noise:**

| x    | y        | dy      |
|------|----------|---------|
| ...  | ...      | ...     |

**MCMC results:**

- Estimated `a`: 1.48 ± 0.12 (true 1.5)  
- Estimated `b`: 0.52 ± 0.08 (true 0.5)  
- Absolute errors: |a - true_a| = 0.02, |b - true_b| = 0.02  
- Percentage errors: 1.33%, 4.00%  

**Plots generated:**

**Trace Plot:**  
![Trace](plots/logarithmic/trace.png)

**Histogram:**  
![Histogram](plots/logarithmic/histogram.png)

**Scatter (a vs b):**  
![Scatter](plots/logarithmic/scatter.png)

---

Repeat similarly for other models: **Linear, Quadratic, Inverse**.  
This shows **posterior distributions, convergence, and correlations**.

## Objective

Demonstrates **statistical modeling, Bayesian inference, and data analysis from scratch**.  
Helps the **community explore parameter uncertainty and model fitting** with reproducible code.

## Extendability

- Add new models by defining `f(x, a, b)`  
- Run inference with the generic `run_example()` framework  

---

**Stack:** Python, NumPy, Matplotlib, MCMC
