# Homework 1 – Modular Multi-Mass-Spring-Damper Simulator

## How to run
```
pip install numpy matplotlib
python Homework1.py
```
This runs all 11 test cases (assertions), then every simulation for the Gradescope questions,
prints the results, and saves `homework1_plots.png`.

## Files
- `Homework1.py` – main file: `gradEs`, `hessEs`, `dampingForce`, `dampingJacobian`, `getFJ`
  (element assembly), `implicitEulerStep`, `explicitEulerStep`, a `simulate` driver, the tests, and the question runs.
- `README.md` – this file.
- `homework1_plots.png` – output plots (generated on run).
