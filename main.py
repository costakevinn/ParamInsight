# main.py
# Entry point to run all MCMC parameter inference examples

import numpy as np

# Set global random seed for reproducibility
np.random.seed(40)

from examples import run_linear_example, run_log_example, run_quadratic_example, run_inverse_example

def main():
    # List of examples: (display name, function)
    examples = [
        ("Linear", run_linear_example),
        ("Logarithmic", run_log_example),
        ("Quadratic", run_quadratic_example),
        ("Inverse", run_inverse_example)
    ]

    # Run each example sequentially
    for name, func in examples:
        print(f"\n==== Running {name} example ====")
        func()  # execute the MCMC example
        print(f"==== {name} example completed ====\n")

if __name__ == "__main__":
    main()
