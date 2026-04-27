# AGENTS.md

## Setup
- This repo is the simulation harness. The estimator lives in the editable companion repo at `../ml-switching-reg`; estimator fixes belong there.
- Python version is `3.11` from `.python-version`. Package manager is `uv`.
- Install this repo with `uv sync`.
- If you need estimator tests or estimator-side edits, also run `uv sync` in `../ml-switching-reg`.

## Verification
- There is no CI, no lint/typecheck config, and no pytest suite in this repo.
- Main verification entrypoints here are notebook-based:
  - `marimo run examples/comprehensive_verification.py`
  - `marimo run examples/monte_carlo_coverage.py`
- The only checked-in automated tests are in the companion repo:
  - `PYTHONPATH=. uv run pytest tests/test_score_bootstrap.py` from `../ml-switching-reg`

## Architecture
- `ml_switching_reg_sim/data_creation.py` is the DGP. `extract_estimator_inputs()` is the main bridge from simulated panels to IRLS/MLE arrays.
- `construct()` returns `mw[true, predicted] = P(predicted | true)` with rows summing to 1.
- Estimators use `cm = mw / mw.sum(axis=0)` so `cm[true, predicted] = P(true | predicted)`.
- `ml_switching_reg_sim/classifier.py` simulates TF-IDF-like surname features and XGBoost out-of-fold probabilities. It does not use simulated `y` or drought summaries directly.

## Likelihood Gotchas
- If latent regime is fixed within driver, pass `driver_ids` into `DriverSpecificProbUberMLE.from_arrays(...)`. Observation-level and driver-level likelihoods differ when `T > 1`.
- Driver-level likelihood requires classifier-implied regime weights to be constant within driver.
- `examples/monte_carlo_coverage.py` is the main inference workbench for coverage and bootstrap experiments.
- `ml_switching_reg_sim/monte_carlo.py` has legacy interfaces alongside current arrays-mode code; check it against `../ml-switching-reg/ml_switching_reg/mle.py` before extending it.

## Scope
- The built-in `MonteCarlo` path is centered on the simple no-lag, no-FE workflow.
- Lagged-covariate and FE workflows are handled manually through `extract_estimator_inputs(..., lags=...)` and `demean_arrays(...)`.
- Marimo `.py` notebooks are preferred for new notebook work; legacy `.ipynb` files still exist.
