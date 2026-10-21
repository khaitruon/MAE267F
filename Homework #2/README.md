# MAE-263F Homework 2 simulator

## Run

From this directory, run:

```bash
python Homework2.py
```

The default run prints the analytical checks for Part 1 and runs the loaded-beam simulations required for Parts 2 questions 6-11. It saves shape/time plots for the 2,000 N and 20,000 N cases. To also run the longer Part 1 numerical checks, use:

```bash
python Homework2.py --part1-full
```

To make a denser logarithmic load-sweep plot for Question 10, use:

```bash
python Homework2.py --sweep
```

Add `--no-plots` to suppress the two default diagnostic plots.

## Files

- `Homework2.py`: simulation setup, implicit-Euler/Newton solver, both homework models, analytical checks, and run options.
- `gradEb.py`: supplied gradient of the discrete bending energy.
- `hessEb.py`: supplied Hessian of the discrete bending energy, with the required `crossMat` helper defined.
- `gradEs.py`: gradient of the discrete stretching energy.
- `hessEs.py`: Hessian of the discrete stretching energy.

Dependencies: Python 3, NumPy, and Matplotlib.
