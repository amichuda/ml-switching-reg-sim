# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

This repo (`ml-switching-reg-sim`) contains Monte Carlo simulation infrastructure for Michuda (2023), testing a switching regression estimator that corrects for ML-based regime misclassification. It depends on a companion package `ml-switching-reg` (found at `/Users/lordflaron/Documents/ml-switching-reg`) which contains the MLE estimator itself.

## Environment & Setup

The project uses `uv` for package management (see `.python-version`; Python ≥3.9, <3.11). Dependencies are declared in `pyproject.toml` (poetry format). Key dependencies: `statsmodels`, `linearmodels`, `scipy`, `scikit-learn`, `tqdm`, `matplotlib`, and `ml-switching-reg`.

```bash
# Install dependencies
uv sync

# Run a notebook
jupyter notebook analytical_mat.ipynb
```

## Two-Repo Architecture

**`ml-switching-reg-sim`** (this repo) — simulation harness:
- `ml_switching_reg_sim/data_creation.py` — `UberDatasetCreator` generates synthetic panel datasets with configurable misclassification. Regime assignment is random per driver; the "drought index" serves as the covariate. `MisclassificationCreator` converts true regimes into soft probability vectors via `noisify_matrix`.
- `ml_switching_reg_sim/monte_carlo.py` — `MonteCarlo` orchestrates data generation and estimation across parameter grids using `multiprocessing.Pool`. `SimulationVisualizer` / `DirectorySimulationVisualizer` loads pickled/CSV results and plots mean estimates with 95% CIs.
- `ml_switching_reg_sim/regression.py` — OLS baseline (`uber_regression`) used as the starting point for MLE.
- `ml_switching_reg_sim/utils.py` — helpers for lagged drought columns and covariance matrix construction.

**`ml-switching-reg`** (companion package) — the estimator:
- `ml_switching_reg/mle.py` — `DriverSpecificProbUberMLE` (extends `GenericLikelihoodModel`); uses driver-specific regime probabilities and `numba_stats` for fast likelihood evaluation. `get_mle_betas` / `get_mle_sigmas` extract parameters from fitted results.
- `ml_switching_reg/patches.py` — patched `PanelData` class from `linearmodels` supporting 3-level MultiIndex (required for within-group demeaning with regime as third level).
- `ml_switching_reg/cm.py` — empirical confusion matrices (`cm_4`, `cm_10`) from the real Uganda Uber dataset, used for initializing MLE.

## Key Data Structures

- Simulated datasets are `(driver, time)` MultiIndex DataFrames.
- Columns follow naming conventions: `drought_{i}` (covariates), `regime_{i}` (true regime dummies), `misclass_regime_{i}` (soft ML prediction weights for regime i).
- `MonteCarlo.simulate()` returns results in a 3-level MultiIndex column: `(estimator, parameter, regime)` where estimator ∈ `{reg, mle, true}`.

## Simulation Workflow

```python
from ml_switching_reg_sim.monte_carlo import MonteCarlo

mc = MonteCarlo(drivers=275, time_periods=10, regimes=2)
dc_list = mc.change_param(
    n_jobs_within=5,
    N_within=100,
    param_dict={"weight": [0.1, 0.5, 0.9]},
    construct_dict={...}
)
results = mc.simulate(dc_list, n_jobs=5)
```

## Notebooks

- `analytical_mat.ipynb` — active development notebook (currently modified per git status)
- `examples/mle_comparison.ipynb` — comparing OLS vs MLE estimates
- `examples/montecarlo_visualizations.ipynb` — plotting simulation outputs

## Literature

`literature/` stores reference papers on misclassification and switching regression:
- `BinaryBasianEL.pdf` — binary Bayesian empirical likelihood
- `yoel.pdf` — related misclassification/switching regression reference
- `J of Applied Econometrics - 2024 - Aronsson - A maximum likelihood bunching estimator of the elasticity of taxable income (1).pdf` — Aronsson (2024), ML bunching estimator for elasticity of taxable income
- `michuda-uganda-uber-jde-reduced.pdf` — Michuda Uganda Uber paper (JDE, reduced version)
