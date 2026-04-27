import hashlib
import os
import sys
import time
import warnings
from multiprocessing import Pool

import numpy as np
import pandas as pd


sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from ml_switching_reg_sim.data_creation import UberDatasetCreatorHet, extract_estimator_inputs
from ml_switching_reg.irls import MLSwitchingRegIRLS
from ml_switching_reg.mle import DriverSpecificProbUberMLE, ScoreWildBootstrap


warnings.filterwarnings("ignore")

RESULTS_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "mc_coverage_results"
)
WEIGHTS = [0.0, 0.1, 0.3, 0.5, 0.7, 0.9]
RHOS = [0.0, 0.3, 0.6, 0.9]
N_REPS = 200
R = 10
DRIVERS = 200
TIME_PERIODS = 15
N_BOOT = 199
TRUE_PARAMS = {
    "beta0": np.linspace(2.0, -1.0, 10).tolist(),
    "beta1": np.linspace(3.0, -2.0, 10).tolist(),
    "y_sd": [1.0] * 10,
}


def _make_drought_cov(rho):
    if rho == 0.0:
        return None
    cov = np.full((R, R), rho)
    np.fill_diagonal(cov, 1.0)
    return cov.tolist()


def _cell_key(weight, rho, rep):
    return f"w{weight}_R{R}_rho{rho}_rep{rep}"


def _result_path(weight, rho, rep):
    return os.path.join(RESULTS_DIR, f"cell_{_cell_key(weight, rho, rep)}.parquet")


def _run_one(task):
    weight, rho, rep = task
    key = _cell_key(weight, rho, rep)
    path = _result_path(weight, rho, rep)
    if os.path.exists(path):
        return key, "skip"

    seed = int(hashlib.md5(f"seed_{weight}_{R}_{rho}_{rep}".encode()).hexdigest(), 16) % (
        2**31
    )
    boot_seed = int(
        hashlib.md5(f"boot_{weight}_{R}_{rho}_{rep}".encode()).hexdigest(), 16
    ) % (2**31)

    try:
        dc = UberDatasetCreatorHet(
            drivers=DRIVERS,
            time_periods=TIME_PERIODS,
            regimes=R,
            seed=seed,
        )
        df, mw = dc.construct(
            seed=seed,
            beta0=TRUE_PARAMS["beta0"],
            beta1=TRUE_PARAMS["beta1"],
            y_sd=TRUE_PARAMS["y_sd"],
            weight=weight,
            drought_cov=_make_drought_cov(rho),
        )

        y, Xl, cp, cm, driver_ids = extract_estimator_inputs(df, mw, R)
        p = Xl[0].shape[1]

        t_irls_start = time.perf_counter()
        mod_irls = MLSwitchingRegIRLS(y, Xl, cp, cm, driver_ids=driver_ids)
        bi = np.zeros((R, p))
        for r in range(R):
            mask = cp.argmax(axis=1) == r
            if mask.sum() > p:
                bi[r] = np.linalg.lstsq(Xl[r][mask], y[mask], rcond=None)[0]
        irls_b, irls_s2 = mod_irls.fit(
            bi, float(np.var(y)), tol=1e-8, max_iter=500, verbose=False
        )
        t_irls = time.perf_counter() - t_irls_start

        t_mle_start = time.perf_counter()
        start = np.append(irls_b.flatten(), np.sqrt(max(irls_s2, 1e-6)))
        mle_mod = DriverSpecificProbUberMLE.from_arrays(
            y,
            Xl,
            cp,
            cm,
            start_params=start,
            driver_ids=driver_ids,
        )
        mle_res, _ = mle_mod.fit(disp=0)
        t_mle = time.perf_counter() - t_mle_start

        mle_params = (
            mle_res.params
            if isinstance(mle_res.params, np.ndarray)
            else mle_res.params.values
        )
        mle_b = mle_params[:-1].reshape(R, p)
        mle_s2 = float(mle_params[-1]) ** 2

        mle_retvals = getattr(mle_res, "mle_retvals", None)
        if isinstance(mle_retvals, dict):
            mle_converged = bool(mle_retvals.get("converged", False))
        else:
            mle_converged = bool(getattr(mle_retvals, "converged", False))

        try:
            bse_vals = mle_res.bse
            mle_bse = (
                bse_vals if isinstance(bse_vals, np.ndarray) else bse_vals.values
            )
        except Exception:
            mle_bse = np.full(len(mle_params), np.nan)

        t_boot_start = time.perf_counter()
        try:
            swb = ScoreWildBootstrap(mle_res)
            swb.get_pvalue(bootstrap_num=N_BOOT, n_jobs=1, seed=boot_seed)
            mle_bse_boot = swb.bse_boot_
            mle_ci_low_boot = swb.conf_int_boot_[:, 0]
            mle_ci_high_boot = swb.conf_int_boot_[:, 1]
            mle_pvalue_boot = swb.pvalues_
        except Exception:
            mle_bse_boot = np.full(len(mle_params), np.nan)
            mle_ci_low_boot = np.full(len(mle_params), np.nan)
            mle_ci_high_boot = np.full(len(mle_params), np.nan)
            mle_pvalue_boot = np.full(len(mle_params), np.nan)
        t_boot = time.perf_counter() - t_boot_start

        true_beta = np.array(
            [
                val
                for r in range(R)
                for val in [TRUE_PARAMS["beta0"][r], TRUE_PARAMS["beta1"][r]]
            ]
        )

        rows = []
        for r in range(R):
            for j in range(p):
                idx = r * p + j
                rows.append(
                    {
                        "weight": weight,
                        "R": R,
                        "rho": rho,
                        "rep": rep,
                        "regime": r,
                        "param_idx": j,
                        "param_name": f"beta_{r}_{j}",
                        "true_val": true_beta[idx],
                        "irls_est": irls_b[r, j],
                        "mle_est": mle_b[r, j],
                        "mle_bse": mle_bse[idx],
                        "mle_bse_boot": mle_bse_boot[idx],
                        "mle_ci_low_boot": mle_ci_low_boot[idx],
                        "mle_ci_high_boot": mle_ci_high_boot[idx],
                        "mle_pvalue_boot": mle_pvalue_boot[idx],
                        "mle_converged": mle_converged,
                        "irls_s2": irls_s2,
                        "mle_s2": mle_s2,
                        "true_s2": TRUE_PARAMS["y_sd"][r] ** 2,
                        "t_irls": t_irls,
                        "t_mle": t_mle,
                        "t_boot": t_boot,
                    }
                )

        pd.DataFrame(rows).to_parquet(path, index=False)
        return key, "ok"
    except Exception as exc:
        err_path = os.path.join(RESULTS_DIR, f"err_{key}.parquet")
        pd.DataFrame(
            [
                {
                    "weight": weight,
                    "R": R,
                    "rho": rho,
                    "rep": rep,
                    "error": str(exc),
                }
            ]
        ).to_parquet(err_path, index=False)
        return key, f"err: {exc}"


def main():
    tasks = [
        (weight, rho, rep)
        for rho in RHOS
        for weight in WEIGHTS
        for rep in range(N_REPS)
    ]
    total = len(tasks)
    workers = min(8, os.cpu_count() or 1)
    print(f"Starting R=10 run with {total} tasks on {workers} workers", flush=True)
    done = 0
    started = time.perf_counter()
    with Pool(processes=workers) as pool:
        for key, status in pool.imap_unordered(_run_one, tasks, chunksize=4):
            done += 1
            if done % 25 == 0 or status.startswith("err"):
                elapsed = time.perf_counter() - started
                rate = done / elapsed if elapsed > 0 else 0.0
                remaining = (total - done) / rate if rate > 0 else float("inf")
                print(
                    f"[{done}/{total}] {key} -> {status} | elapsed={elapsed/60:.1f}m | eta={remaining/60:.1f}m",
                    flush=True,
                )
    print("R=10 run complete", flush=True)


if __name__ == "__main__":
    main()
