import marimo

__generated_with = "0.20.4"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import sys
    import os
    import hashlib
    import time
    import warnings

    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

    import numpy as np
    import pandas as pd

    warnings.filterwarnings("ignore")

    RESULTS_DIR = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "..", "mc_coverage_results"
    )
    os.makedirs(RESULTS_DIR, exist_ok=True)

    WEIGHTS = [0.0, 0.1, 0.3, 0.5, 0.7, 0.9]
    REGIMES_LIST = [2, 4, 10]
    RHOS = [0.0, 0.3, 0.6, 0.9]
    N_REPS = 200

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

    DRIVERS = 200
    TIME_PERIODS = 15
    N_BOOT = 199  # score wild bootstrap draws per MC replication
    return (
        DRIVERS,
        N_BOOT,
        N_REPS,
        REGIMES_LIST,
        RESULTS_DIR,
        RHOS,
        TIME_PERIODS,
        TRUE_PARAMS,
        WEIGHTS,
        hashlib,
        mo,
        np,
        os,
        pd,
        time,
        warnings,
    )


@app.cell
def _(mo):
    mo.md(r"""
    # Monte Carlo Coverage Study

    This notebook runs a full factorial Monte Carlo simulation across:

    - **Misclassification weight** $\in \{0.0, 0.1, 0.3, 0.5, 0.7, 0.9\}$
    - **Regimes** $R \in \{2, 4, 10\}$
    - **Shock correlation** $\rho \in \{0.0, 0.3, 0.6, 0.9\}$
    - **200 replications** per cell

    For each replication we:
    1. Generate data via `UberDatasetCreatorHet`
    2. Fit IRLS for starting values, then MLE
    3. Record estimates, standard errors, convergence, and timing

    The `R=10` design is meant to sit closer to the Uganda application while keeping the
    same Monte Carlo grid over misclassification severity and shock correlation.

    Results are saved to disk for resumability.
    """)
    return


@app.cell
def _(
    DRIVERS,
    N_BOOT,
    N_REPS,
    REGIMES_LIST,
    RESULTS_DIR,
    RHOS,
    TIME_PERIODS,
    TRUE_PARAMS,
    WEIGHTS,
    hashlib,
    np,
    os,
    pd,
    time,
    warnings,
):
    from ml_switching_reg_sim.data_creation import (
        UberDatasetCreatorHet,
        extract_estimator_inputs,
    )
    from ml_switching_reg.irls import MLSwitchingRegIRLS
    from ml_switching_reg.mle import DriverSpecificProbUberMLE, ScoreWildBootstrap

    def _make_drought_cov(R, rho):
        if rho == 0.0:
            return None
        cov = np.full((R, R), rho)
        np.fill_diagonal(cov, 1.0)
        return cov.tolist()

    def _cell_key(weight, R, rho, rep):
        return f"w{weight}_R{R}_rho{rho}_rep{rep}"

    def result_path(weight, R, rho, rep):
        return os.path.join(
            RESULTS_DIR, f"cell_{_cell_key(weight, R, rho, rep)}.parquet"
        )

    def run_single_rep(weight, R, rho, rep):
        key = _cell_key(weight, R, rho, rep)
        path = result_path(weight, R, rho, rep)
        if os.path.exists(path):
            return key

        seed = int(
            hashlib.md5(f"seed_{weight}_{R}_{rho}_{rep}".encode()).hexdigest(), 16
        ) % (2**31)

        tp = TRUE_PARAMS[R]
        drought_cov = _make_drought_cov(R, rho)

        try:
            dc = UberDatasetCreatorHet(
                drivers=DRIVERS, time_periods=TIME_PERIODS, regimes=R, seed=seed
            )
            df, mw = dc.construct(
                seed=seed,
                beta0=tp["beta0"],
                beta1=tp["beta1"],
                y_sd=tp["y_sd"],
                weight=weight,
                drought_cov=drought_cov,
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
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
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

            # Score wild bootstrap — cluster by driver for valid inference
            boot_seed = int(
                hashlib.md5(f"boot_{weight}_{R}_{rho}_{rep}".encode()).hexdigest(), 16
            ) % (2**31)
            t_boot_start = time.perf_counter()
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    swb = ScoreWildBootstrap(mle_res)
                    swb.get_pvalue(
                        bootstrap_num=N_BOOT, n_jobs=1, seed=boot_seed
                    )
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

            # Order must match mle_params: [b0_r0, b1_r0, b0_r1, b1_r1, ...]
            # (regime-major, coefficient-minor — same as irls_b.flatten())
            true_beta = np.array(
                [val for r in range(R) for val in [tp["beta0"][r], tp["beta1"][r]]]
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
                            "true_s2": tp["y_sd"][r] ** 2,
                            "t_irls": t_irls,
                            "t_mle": t_mle,
                            "t_boot": t_boot,
                        }
                    )

            res_df = pd.DataFrame(rows)
            res_df.to_parquet(path, index=False)
            return key

        except Exception as e:
            err_df = pd.DataFrame(
                [
                    {
                        "weight": weight,
                        "R": R,
                        "rho": rho,
                        "rep": rep,
                        "error": str(e),
                    }
                ]
            )
            err_path = os.path.join(RESULTS_DIR, f"err_{key}.parquet")
            err_df.to_parquet(err_path, index=False)
            return key

    _total_cells = len(WEIGHTS) * len(REGIMES_LIST) * len(RHOS) * N_REPS
    _total_cells
    return result_path, run_single_rep


@app.cell
def _(
    N_REPS,
    REGIMES_LIST,
    RHOS,
    WEIGHTS,
    mo,
    os,
    result_path,
    run_single_rep,
    time,
):
    t_global_start = time.perf_counter()

    _total = len(WEIGHTS) * len(REGIMES_LIST) * len(RHOS) * N_REPS

    with mo.status.progress_bar(
        total=_total,
        title="Monte Carlo Simulation",
        subtitle='Waiting...'

    ) as _progress:
        for R in REGIMES_LIST:
            for rho in RHOS:
                for weight in WEIGHTS:
                    for rep in range(N_REPS):
                        _progress.subtitle = f"Running R: {R}, rho: {rho}, weight: {weight}, rep: {rep}"
                        if os.path.exists(result_path(weight, R, rho, rep)):
                            _progress.total -= 1
                            continue
                        run_single_rep(weight, R, rho, rep)
                        _progress.update()

    t_total = time.perf_counter() - t_global_start

    t_total_minutes = t_total / 60
    f"Simulation complete in {t_total_minutes:.1f} minutes"
    return


@app.cell
def _(RESULTS_DIR, TRUE_PARAMS, np, os, pd):
    all_dfs = []
    err_dfs = []

    import glob as _glob

    for f in sorted(_glob.glob(os.path.join(RESULTS_DIR, "cell_*.parquet"))):
        try:
            all_dfs.append(pd.read_parquet(f))
        except Exception:
            pass

    for f in sorted(_glob.glob(os.path.join(RESULTS_DIR, "err_*.parquet"))):
        try:
            err_dfs.append(pd.read_parquet(f))
        except Exception:
            pass

    results = pd.concat(all_dfs, ignore_index=True) if all_dfs else pd.DataFrame()
    errors = pd.concat(err_dfs, ignore_index=True) if err_dfs else pd.DataFrame()

    # Recompute true_val from regime/param_idx so cached results with the old
    # (wrong) true_beta ordering are repaired automatically.
    if len(results) > 0 and "regime" in results.columns and "param_idx" in results.columns:
        def _correct_true_val(row):
            tp = TRUE_PARAMS.get(int(row["R"]), {})
            r, j = int(row["regime"]), int(row["param_idx"])
            if j == 0:
                return tp.get("beta0", [np.nan])[r] if r < len(tp.get("beta0", [])) else np.nan
            elif j == 1:
                return tp.get("beta1", [np.nan])[r] if r < len(tp.get("beta1", [])) else np.nan
            return np.nan
        results["true_val"] = results.apply(_correct_true_val, axis=1)

    n_results = len(results)
    n_errors = len(errors)
    f"Loaded {n_results} result rows, {n_errors} errors"
    return errors, results


@app.cell
def _(mo, results):
    mo.md("## Bias Plots")

    _display = mo.md("No results yet.")
    if len(results) > 0:
        bias_df = results.copy()
        bias_df["mle_bias"] = bias_df["mle_est"] - bias_df["true_val"]
        bias_df["irls_bias"] = bias_df["irls_est"] - bias_df["true_val"]

        pivot_mle = (
            bias_df.groupby(["weight", "rho", "R", "param_name"])["mle_bias"]
            .mean()
            .reset_index()
        )

        for _R_val in sorted(bias_df["R"].unique()):
            _display = mo.md(f"### R = {_R_val}")
            for _pname in sorted(
                bias_df.loc[bias_df["R"] == _R_val, "param_name"].unique()
            ):
                _sub = pivot_mle[
                    (pivot_mle["R"] == _R_val) & (pivot_mle["param_name"] == _pname)
                ]
                _heat = _sub.pivot(index="weight", columns="rho", values="mle_bias")
                _display = mo.vstack(
                    [
                        _display,
                        mo.md(f"**MLE Bias: {_pname} (R={_R_val})**"),
                        mo.ui.table(_heat.reset_index()),
                    ]
                )

    _display
    return


@app.cell
def _(mo, results):
    mo.md("## 95% Coverage Table")

    _display = mo.md("No results yet.")
    if len(results) > 0 and results["mle_bse"].notna().any():
        cov_df = results.copy()
        cov_df["mle_ci_low"] = cov_df["mle_est"] - 1.96 * cov_df["mle_bse"]
        cov_df["mle_ci_high"] = cov_df["mle_est"] + 1.96 * cov_df["mle_bse"]
        cov_df["covered"] = (cov_df["true_val"] >= cov_df["mle_ci_low"]) & (
            cov_df["true_val"] <= cov_df["mle_ci_high"]
        )

        coverage = (
            cov_df.groupby(["weight", "rho", "R"])["covered"]
            .mean()
            .reset_index()
            .rename(columns={"covered": "coverage_rate"})
        )

        _display = mo.md("### Analytical Wald 95% CI")
        for _R_val in sorted(coverage["R"].unique()):
            _sub = coverage[coverage["R"] == _R_val]
            _heat = _sub.pivot(index="weight", columns="rho", values="coverage_rate")
            _display = mo.vstack(
                [
                    _display,
                    mo.md(f"**95% CI Coverage Rate (R={_R_val})**"),
                    mo.ui.table(_heat.reset_index()),
                ]
            )

        if "mle_ci_low_boot" in cov_df.columns and cov_df["mle_ci_low_boot"].notna().any():
            cov_df["covered_boot"] = (
                cov_df["true_val"] >= cov_df["mle_ci_low_boot"]
            ) & (cov_df["true_val"] <= cov_df["mle_ci_high_boot"])
            coverage_boot = (
                cov_df.groupby(["weight", "rho", "R"])["covered_boot"]
                .mean()
                .reset_index()
                .rename(columns={"covered_boot": "coverage_rate_boot"})
            )
            _display = mo.vstack([_display, mo.md("### Score Wild Bootstrap 95% CI")])
            for _R_val in sorted(coverage_boot["R"].unique()):
                _sub = coverage_boot[coverage_boot["R"] == _R_val]
                _heat = _sub.pivot(
                    index="weight", columns="rho", values="coverage_rate_boot"
                )
                _display = mo.vstack(
                    [
                        _display,
                        mo.md(f"**Bootstrap 95% CI Coverage Rate (R={_R_val})**"),
                        mo.ui.table(_heat.reset_index()),
                    ]
                )

    _display
    return


@app.cell
def _(mo, np, results):
    mo.md("## RMSE Decomposition")

    _display = mo.md("No results yet.")
    if len(results) > 0:
        rmse_df = results.copy()
        rmse_df["mle_bias"] = rmse_df["mle_est"] - rmse_df["true_val"]
        rmse_df["irls_bias"] = rmse_df["irls_est"] - rmse_df["true_val"]

        decomp = (
            rmse_df.groupby(["weight", "rho", "R", "param_name"])
            .agg(
                mle_bias2=("mle_bias", lambda x: (x**2).mean()),
                mle_var=("mle_est", "var"),
                irls_bias2=("irls_bias", lambda x: (x**2).mean()),
                irls_var=("irls_est", "var"),
            )
            .reset_index()
        )
        decomp["mle_rmse"] = np.sqrt(decomp["mle_bias2"] + decomp["mle_var"])
        decomp["irls_rmse"] = np.sqrt(decomp["irls_bias2"] + decomp["irls_var"])
        decomp["mle_bias_pct"] = decomp["mle_bias2"] / (
            decomp["mle_bias2"] + decomp["mle_var"] + 1e-300
        )
        decomp["irls_bias_pct"] = decomp["irls_bias2"] / (
            decomp["irls_bias2"] + decomp["irls_var"] + 1e-300
        )

        _display = mo.ui.table(decomp, page_size=20)

    _display
    return


@app.cell
def _(mo, results):
    mo.md("## Timing Comparison: IRLS vs MLE vs Bootstrap")

    _display = mo.md("No results yet.")
    if len(results) > 0:
        _agg = {
            "mean_t_irls": ("t_irls", "mean"),
            "median_t_irls": ("t_irls", "median"),
            "mean_t_mle": ("t_mle", "mean"),
            "median_t_mle": ("t_mle", "median"),
            "n_reps": ("rep", "nunique"),
        }
        if "t_boot" in results.columns and results["t_boot"].notna().any():
            _agg["mean_t_boot"] = ("t_boot", "mean")
            _agg["median_t_boot"] = ("t_boot", "median")
        timing = (
            results.groupby(["weight", "rho", "R"])
            .agg(**_agg)
            .reset_index()
        )
        timing["mle_irls_ratio"] = timing["mean_t_mle"] / (
            timing["mean_t_irls"] + 1e-300
        )
        if "mean_t_boot" in timing.columns:
            timing["boot_mle_ratio"] = timing["mean_t_boot"] / (
                timing["mean_t_mle"] + 1e-300
            )

        _display = mo.ui.table(timing, page_size=20)

    _display
    return


@app.cell
def _(errors, mo, results):
    mo.md("## Summary")

    _lines = []
    if len(results) > 0:
        conv_rate = results.groupby("R")["mle_converged"].mean()
        _lines.append(f"**Total rows:** {len(results)}")
        _lines.append(f"**Convergence rates:** {conv_rate.to_dict()}")
        _lines.append(f"**Errors:** {len(errors)}")

        avg_irls = results["t_irls"].mean()
        avg_mle = results["t_mle"].mean()
        _lines.append(
            f"**Avg IRLS time:** {avg_irls:.3f}s  |  **Avg MLE time:** {avg_mle:.3f}s  |  **Ratio:** {avg_mle / (avg_irls + 1e-300):.1f}x"
        )
        if "t_boot" in results.columns and results["t_boot"].notna().any():
            avg_boot = results["t_boot"].mean()
            _lines.append(
                f"**Avg bootstrap time:** {avg_boot:.3f}s  |  **Boot/MLE ratio:** {avg_boot / (avg_mle + 1e-300):.1f}x"
            )
    else:
        _lines.append("No results loaded yet. Run the simulation cell above.")

    mo.md("\n\n".join(_lines))
    return


if __name__ == "__main__":
    app.run()
