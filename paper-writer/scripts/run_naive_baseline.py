"""Naive hard-classification baseline for the Monte Carlo coverage study.

For each Monte Carlo cell, regenerate the same simulated dataset (matching the
seed scheme in ``examples/monte_carlo_coverage.py``), assign each observation
to its modal classifier prediction, and estimate regime-specific OLS without
the confusion-matrix correction. Standard errors are cluster-robust at the
driver level, matching the clustering used by the corrected MLE bootstrap.

Outputs land in ``mc_naive_results/cell_*.parquet`` and have the same key
columns as the MLE outputs so the analysis script can join on
``(weight, R, rho, rep, regime, param_idx)``.
"""

from __future__ import annotations

import hashlib
import os
import sys
import time
import warnings
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
PAPER_DIR = SCRIPT_DIR.parent
ROOT_DIR = PAPER_DIR.parent
RESULTS_DIR = ROOT_DIR / "mc_naive_results"

sys.path.insert(0, str(ROOT_DIR))

from ml_switching_reg_sim.data_creation import (  # noqa: E402
    UberDatasetCreatorHet,
    extract_estimator_inputs,
)

WEIGHTS = [0.0, 0.1, 0.3, 0.5, 0.7, 0.9]
REGIMES_LIST = [2, 4, 10]
RHOS = [0.0, 0.3, 0.6, 0.9]
N_REPS = 200
DRIVERS = 200
TIME_PERIODS = 15

TRUE_PARAMS = {
    2: {
        "beta0": [2.0, -1.0],
        "beta1": [3.0, -2.0],
        "y_sd": [1.0, 1.0],
    },
    4: {
        "beta0": [2.0, -1.0, 0.5, -0.5],
        "beta1": [3.0, -2.0, 1.5, -1.0],
        "y_sd": [1.0, 1.0, 1.0, 1.0],
    },
    10: {
        "beta0": np.linspace(2.0, -1.0, 10).tolist(),
        "beta1": np.linspace(3.0, -2.0, 10).tolist(),
        "y_sd": [1.0] * 10,
    },
}


def _make_drought_cov(R: int, rho: float):
    if rho == 0.0:
        return None
    cov = np.full((R, R), rho)
    np.fill_diagonal(cov, 1.0)
    return cov.tolist()


def _cell_key(weight: float, R: int, rho: float, rep: int) -> str:
    return f"w{weight}_R{R}_rho{rho}_rep{rep}"


def result_path(weight: float, R: int, rho: float, rep: int) -> Path:
    return RESULTS_DIR / f"cell_{_cell_key(weight, R, rho, rep)}.parquet"


def _seed(weight: float, R: int, rho: float, rep: int) -> int:
    return int(
        hashlib.md5(f"seed_{weight}_{R}_{rho}_{rep}".encode()).hexdigest(), 16
    ) % (2**31)


def _cluster_robust_ols(
    y: np.ndarray, X: np.ndarray, cluster_ids: np.ndarray
) -> tuple[np.ndarray, np.ndarray, float]:
    """OLS with Liang-Zeger cluster-robust SE.

    Returns (beta, se, sigma2). Uses the standard small-sample correction
    ``G/(G-1) * (n-1)/(n-k)``.
    """
    n, k = X.shape
    XtX_inv = np.linalg.inv(X.T @ X)
    beta = XtX_inv @ X.T @ y
    resid = y - X @ beta
    sigma2 = float(resid @ resid) / max(n - k, 1)

    unique_clusters, cluster_idx = np.unique(cluster_ids, return_inverse=True)
    G = len(unique_clusters)
    meat = np.zeros((k, k))
    for g in range(G):
        mask = cluster_idx == g
        Xg = X[mask]
        ug = resid[mask]
        s = Xg.T @ ug
        meat += np.outer(s, s)
    correction = (G / max(G - 1, 1)) * ((n - 1) / max(n - k, 1)) if G > 1 else 1.0
    V = correction * XtX_inv @ meat @ XtX_inv
    se = np.sqrt(np.maximum(np.diag(V), 0.0))
    return beta, se, sigma2


def run_single_rep(args: tuple[float, int, float, int]) -> str:
    weight, R, rho, rep = args
    key = _cell_key(weight, R, rho, rep)
    path = result_path(weight, R, rho, rep)
    if path.exists():
        return key

    seed = _seed(weight, R, rho, rep)
    tp = TRUE_PARAMS[R]
    drought_cov = _make_drought_cov(R, rho)

    t_start = time.perf_counter()
    try:
        dc = UberDatasetCreatorHet(
            drivers=DRIVERS, time_periods=TIME_PERIODS, regimes=R, seed=seed
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            df, mw = dc.construct(
                seed=seed,
                beta0=tp["beta0"],
                beta1=tp["beta1"],
                y_sd=tp["y_sd"],
                weight=weight,
                drought_cov=drought_cov,
            )

        y, Xl, cp, _, driver_ids = extract_estimator_inputs(df, mw, R)
        p = Xl[0].shape[1]

        # Hard classification: each obs assigned to its modal predicted regime.
        pred = cp.argmax(axis=1)

        beta_hat = np.full((R, p), np.nan)
        se_hat = np.full((R, p), np.nan)
        sigma2_hat = np.full(R, np.nan)
        n_per_regime = np.zeros(R, dtype=int)

        for r in range(R):
            mask = pred == r
            n_per_regime[r] = int(mask.sum())
            if mask.sum() <= p:
                # Underidentified — leave as NaN; this is the realistic failure mode
                # of the naive baseline when the classifier rarely picks regime r.
                continue
            beta_r, se_r, sigma2_r = _cluster_robust_ols(
                y[mask], Xl[r][mask], driver_ids[mask]
            )
            beta_hat[r] = beta_r
            se_hat[r] = se_r
            sigma2_hat[r] = sigma2_r

        t_naive = time.perf_counter() - t_start

        rows = []
        for r in range(R):
            for j in range(p):
                est = beta_hat[r, j]
                se = se_hat[r, j]
                if np.isfinite(est) and np.isfinite(se):
                    ci_low = est - 1.96 * se
                    ci_high = est + 1.96 * se
                else:
                    ci_low = np.nan
                    ci_high = np.nan
                rows.append(
                    {
                        "weight": weight,
                        "R": R,
                        "rho": rho,
                        "rep": rep,
                        "regime": r,
                        "param_idx": j,
                        "param_name": f"beta_{r}_{j}",
                        "true_val": (
                            tp["beta0"][r] if j == 0 else tp["beta1"][r]
                        ),
                        "naive_est": est,
                        "naive_se": se,
                        "naive_ci_low": ci_low,
                        "naive_ci_high": ci_high,
                        "naive_s2": sigma2_hat[r],
                        "naive_n_regime": n_per_regime[r],
                        "naive_identified": bool(np.isfinite(est)),
                        "t_naive": t_naive,
                    }
                )

        res_df = pd.DataFrame(rows)
        res_df.to_parquet(path, index=False)
        return key

    except Exception as e:
        err_path = RESULTS_DIR / f"err_{key}.parquet"
        pd.DataFrame(
            [{"weight": weight, "R": R, "rho": rho, "rep": rep, "error": str(e)}]
        ).to_parquet(err_path, index=False)
        return key


def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    args_list = [
        (weight, R, rho, rep)
        for R in REGIMES_LIST
        for rho in RHOS
        for weight in WEIGHTS
        for rep in range(N_REPS)
    ]
    pending = [a for a in args_list if not result_path(*a).exists()]
    print(f"Total cells: {len(args_list)}; pending: {len(pending)}")

    if not pending:
        print("Nothing to do.")
        return

    n_workers = max(1, (os.cpu_count() or 2) - 1)
    print(f"Using {n_workers} workers")
    t0 = time.perf_counter()
    with Pool(processes=n_workers) as pool:
        for i, _ in enumerate(pool.imap_unordered(run_single_rep, pending), 1):
            if i % 200 == 0:
                elapsed = time.perf_counter() - t0
                rate = i / elapsed
                remaining = (len(pending) - i) / max(rate, 1e-9)
                print(
                    f"  {i}/{len(pending)} done — {elapsed:.1f}s elapsed, "
                    f"~{remaining:.1f}s remaining"
                )
    elapsed = time.perf_counter() - t0
    print(f"Done in {elapsed:.1f}s")


if __name__ == "__main__":
    main()
