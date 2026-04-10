import marimo

__generated_with = "0.22.4"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import sys, os

    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import contextlib, io

    from ml_switching_reg_sim.data_creation import (
        UberDatasetCreatorHet,
        extract_estimator_inputs,
    )
    from ml_switching_reg.irls import MLSwitchingRegIRLS

    return (
        MLSwitchingRegIRLS,
        UberDatasetCreatorHet,
        contextlib,
        extract_estimator_inputs,
        io,
        mo,
        np,
        pd,
        plt,
    )


@app.cell
def _(mo):
    mo.md("""
    # Classifier Comparison: `noisify_matrix` vs. XGBoost ML Classifier

    **Aleksandr Michuda** · Simulation Study

    ---

    ## Overview

    This notebook compares two approaches to generating synthetic misclassification for
    the switching regression estimator:

    - **`noisify_matrix`** — deterministic, per-driver soft probabilities parameterised by
      `weight ∈ (0,1)`.  At `weight=0` the classifier is perfect; at `weight=1` predictions
      are uniform (P=1/R each regime).
    - **XGBoost ML classifier** — trained on synthetic TF-IDF-like name features (regime mean
      embedded in a `n_features`-dim vector plus Gaussian noise), controlled by `noise_scale`.

    Both approaches now return a `mw` that is **exactly consistent** with the per-driver
    `classifier_pred` columns, eliminating the previous bug where `mw` was drawn from a
    different random process.

    ### Design

    | Parameter | Value |
    |-----------|-------|
    | Drivers | 200 |
    | Time periods | 15 |
    | Regimes | 2 |
    | True β₀ | [1.0, 3.0] |
    | True β₁ | [−1.5, 2.0] |
    | True σ | [1.0, 1.0] |
    | Monte Carlo replicates | 50 |

    A **random-chance reference line** (P = 1/R = 0.5) is shown on every panel,
    representing the bias/RMSE expected when the classifier is fully uninformative.
    """)
    return


@app.cell
def _(mo):
    mo.md("""
    ## Shared setup
    """)
    return


@app.cell
def _():
    DRIVERS = 200
    PERIODS = 15
    REGIMES = 2
    B0 = [1.0, 3.0]
    B1 = [-1.5, 2.0]
    SD = [1.0, 1.0]
    N_SIM = 50
    TRUE = {"beta0_0": 1.0, "beta1_0": -1.5, "beta0_1": 3.0, "beta1_1": 2.0, "sigma": 1.0}
    PARAMS = ["beta0_0", "beta1_0", "beta0_1", "beta1_1"]
    LABELS = {
        "beta0_0": "β₀ reg 0",
        "beta1_0": "β₁ reg 0",
        "beta0_1": "β₀ reg 1",
        "beta1_1": "β₁ reg 1",
    }
    return B0, B1, DRIVERS, LABELS, N_SIM, PARAMS, PERIODS, REGIMES, SD, TRUE


@app.cell
def _(mo):
    mo.md("""
    ## Helper: IRLS runner
    """)
    return


@app.cell
def _(MLSwitchingRegIRLS, contextlib, io, np):
    def run_irls(y, Xl, cp, cm):
        R = len(Xl)
        p = Xl[0].shape[1]
        b0 = np.zeros((R, p))
        for r in range(R):
            mask = cp.argmax(axis=1) == r
            if mask.sum() > p:
                b0[r] = np.linalg.lstsq(Xl[r][mask], y[mask], rcond=None)[0]
        mod = MLSwitchingRegIRLS(y, Xl, cp, cm)
        with contextlib.redirect_stdout(io.StringIO()):
            beta, s2 = mod.fit(beta_0=b0, sigma2_0=float(np.var(y)), tol=1e-7, max_iter=500)
        return {
            "beta0_0": float(beta[0, 0]),
            "beta1_0": float(beta[0, 1]),
            "beta0_1": float(beta[1, 0]),
            "beta1_1": float(beta[1, 1]),
            "sigma": float(np.sqrt(max(s2, 1e-10))),
        }

    return (run_irls,)


@app.cell
def _(mo):
    mo.md("""
    ## Random-chance baseline

    Computed once: simulate with `weight=0.99` (near-uniform classifier) to get the
    bias/RMSE expected when the classifier provides no information.
    """)
    return


@app.cell
def _(
    B0,
    B1,
    DRIVERS,
    N_SIM,
    PARAMS,
    PERIODS,
    REGIMES,
    SD,
    TRUE,
    UberDatasetCreatorHet,
    extract_estimator_inputs,
    np,
    pd,
    run_irls,
):
    _rows_rc = []
    for _sim in range(N_SIM):
        _u = UberDatasetCreatorHet(
            drivers=DRIVERS, time_periods=PERIODS, regimes=REGIMES, seed=9000 + _sim
        )
        _df, _mw = _u.construct(seed=9000 + _sim, beta0=B0, beta1=B1, y_sd=SD, weight=0.99)
        _y, _Xl, _cp, _cm = extract_estimator_inputs(_df, _mw, REGIMES)
        _r = run_irls(_y, _Xl, _cp, _cm)
        _rows_rc.append(_r)

    df_rc = pd.DataFrame(_rows_rc)
    random_chance_bias = {p: float(np.mean(df_rc[p].values) - TRUE[p]) for p in PARAMS}
    random_chance_rmse = {
        p: float(np.sqrt(np.mean((df_rc[p].values - TRUE[p]) ** 2))) for p in PARAMS
    }
    print("Random-chance bias:", {k: round(v, 3) for k, v in random_chance_bias.items()})
    print("Random-chance RMSE:", {k: round(v, 3) for k, v in random_chance_rmse.items()})
    df_rc, random_chance_bias, random_chance_rmse
    return


@app.cell
def _(mo):
    mo.md("""
    ---
    ## Section A — `noisify_matrix` sensitivity

    Grid: `weight ∈ [0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 0.9, 0.95]`.
    After the `mw` consistency fix, bias and RMSE should be **monotonically increasing** —
    the spurious peak from the old inconsistent `mw` is eliminated.
    """)
    return


@app.cell
def _(
    B0,
    B1,
    DRIVERS,
    N_SIM,
    PERIODS,
    REGIMES,
    SD,
    UberDatasetCreatorHet,
    extract_estimator_inputs,
    pd,
    run_irls,
):
    _WEIGHT_GRID = [0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 0.9, 0.95]
    _rows_a = []
    for _w in _WEIGHT_GRID:
        for _sim in range(N_SIM):
            _u = UberDatasetCreatorHet(
                drivers=DRIVERS, time_periods=PERIODS, regimes=REGIMES, seed=1000 + _sim
            )
            _df, _mw = _u.construct(
                seed=1000 + _sim, beta0=B0, beta1=B1, y_sd=SD, weight=_w
            )
            _y, _Xl, _cp, _cm = extract_estimator_inputs(_df, _mw, REGIMES)
            _r = run_irls(_y, _Xl, _cp, _cm)
            _r["weight"] = _w
            _rows_a.append(_r)
        print(f"  noisify weight={_w:.2f}  done")
    df_a = pd.DataFrame(_rows_a)
    df_a
    return (df_a,)


@app.cell
def _(LABELS, PARAMS, TRUE, df_a, mo, np, plt):
    _weight_vals = sorted(df_a["weight"].unique())
    _COLORS = ["#4878d0", "#d65f5f", "#6acc65", "#ee854a"]
    _fig_a, _axes_a = plt.subplots(2, len(PARAMS), figsize=(14, 7))

    for _ci, _pk in enumerate(PARAMS):
        _bias_vals, _rmse_vals, _std_vals = [], [], []
        for _w in _weight_vals:
            _v = df_a[df_a["weight"] == _w][_pk].values
            _bias_vals.append(float(np.mean(_v) - TRUE[_pk]))
            _rmse_vals.append(float(np.sqrt(np.mean((_v - TRUE[_pk]) ** 2))))
            _std_vals.append(float(np.std(_v)))

        # Bias row
        _ax = _axes_a[0, _ci]
        _ax.plot(_weight_vals, _bias_vals, marker="o", color=_COLORS[_ci], lw=2)
        _lo = [b - s for b, s in zip(_bias_vals, _std_vals)]
        _hi = [b + s for b, s in zip(_bias_vals, _std_vals)]
        _ax.fill_between(_weight_vals, _lo, _hi, alpha=0.15, color=_COLORS[_ci])
        _ax.axhline(0, color="black", lw=0.8, linestyle="--", alpha=0.5)
        # weight=1.0 is where P(correct) = 1/R (random chance for noisify_matrix)
        _ax.axvline(
            1.0, color="gray", lw=1.2, linestyle=":", label="Random chance (weight=1)"
        )
        _ax.set_title(f"{LABELS[_pk]} — Bias", fontsize=9)
        _ax.set_xlabel("weight")
        _ax.grid(True, alpha=0.3)
        if _ci == 0:
            _ax.legend(fontsize=7)

        # RMSE row
        _ax2 = _axes_a[1, _ci]
        _ax2.plot(_weight_vals, _rmse_vals, marker="o", color=_COLORS[_ci], lw=2)
        _ax2.axvline(
            1.0, color="gray", lw=1.2, linestyle=":", label="Random chance (weight=1)"
        )
        _ax2.set_title(f"{LABELS[_pk]} — RMSE", fontsize=9)
        _ax2.set_xlabel("weight")
        _ax2.grid(True, alpha=0.3)
        if _ci == 0:
            _ax2.legend(fontsize=7)

    plt.suptitle(
        "Section A: noisify_matrix sensitivity (N=3 000, 50 simulations)\n"
        "Shaded = ±1 std  ·  Dotted vertical = weight where P(correct)=1/R",
        fontweight="bold",
        fontsize=10,
    )
    plt.tight_layout()
    mo.as_html(_fig_a)
    return


@app.cell
def _(mo):
    mo.md("""
    ---
    ## Section B — XGBoost ML classifier sensitivity

    Grid: `noise_scale ∈ [0.1, 0.5, 1.0, 2.0, 5.0, 10.0]`.
    Features are `n_features=50`-dim TF-IDF-like vectors: `x_i = μ[regime_i] + noise_scale·N(0,I)`.
    The secondary x-axis shows the mean P(correct) = CM diagonal mean.
    """)
    return


@app.cell
def _(
    B0,
    B1,
    DRIVERS,
    N_SIM,
    PERIODS,
    REGIMES,
    SD,
    UberDatasetCreatorHet,
    extract_estimator_inputs,
    np,
    pd,
    run_irls,
):
    _NS_GRID = [0.1, 0.5, 1.0, 2.0, 5.0, 10.0]
    _rows_b = []
    for _ns in _NS_GRID:
        for _sim in range(N_SIM):
            _u = UberDatasetCreatorHet(
                drivers=DRIVERS, time_periods=PERIODS, regimes=REGIMES, seed=2000 + _sim
            )
            _df, _mw = _u.construct(
                seed=2000 + _sim,
                beta0=B0,
                beta1=B1,
                y_sd=SD,
                classifier_mode="ml",
                noise_scale=_ns,
            )
            _r = run_irls(*extract_estimator_inputs(_df, _mw, REGIMES))
            _r["noise_scale"] = _ns
            _r["p_correct"] = float(np.diag(_mw).mean())
            _rows_b.append(_r)
        print(f"  ml noise_scale={_ns:.2f}  done")
    df_b = pd.DataFrame(_rows_b)
    df_b
    return (df_b,)


@app.cell
def _(LABELS, PARAMS, REGIMES, TRUE, df_b, mo, np, plt):
    _ns_vals = sorted(df_b["noise_scale"].unique())
    _p_correct_vals = [
        float(df_b[df_b["noise_scale"] == ns]["p_correct"].mean()) for ns in _ns_vals
    ]
    # Interpolate noise_scale where P(correct) crosses the random-chance threshold 1/R
    _random_chance_pc = 1.0 / REGIMES
    _ns_random_chance = float(
        np.interp(
            _random_chance_pc, list(reversed(_p_correct_vals)), list(reversed(_ns_vals))
        )
    )

    _COLORS = ["#4878d0", "#d65f5f", "#6acc65", "#ee854a"]
    _fig_b, _axes_b = plt.subplots(2, len(PARAMS), figsize=(14, 7))

    for _ci, _pk in enumerate(PARAMS):
        _bias_vals, _rmse_vals, _std_vals = [], [], []
        for _ns in _ns_vals:
            _v = df_b[df_b["noise_scale"] == _ns][_pk].values
            _bias_vals.append(float(np.mean(_v) - TRUE[_pk]))
            _rmse_vals.append(float(np.sqrt(np.mean((_v - TRUE[_pk]) ** 2))))
            _std_vals.append(float(np.std(_v)))

        _ax = _axes_b[0, _ci]
        _ax.plot(_ns_vals, _bias_vals, marker="o", color=_COLORS[_ci], lw=2)
        _lo = [b - s for b, s in zip(_bias_vals, _std_vals)]
        _hi = [b + s for b, s in zip(_bias_vals, _std_vals)]
        _ax.fill_between(_ns_vals, _lo, _hi, alpha=0.15, color=_COLORS[_ci])
        _ax.axhline(0, color="black", lw=0.8, linestyle="--", alpha=0.5)
        _ax.axvline(
            _ns_random_chance,
            color="gray",
            lw=1.2,
            linestyle=":",
            label=f"Random chance (P=1/{REGIMES})",
        )
        _ax.set_title(f"{LABELS[_pk]} — Bias", fontsize=9)
        _ax.set_xlabel("noise_scale")
        _ax.grid(True, alpha=0.3)
        if _ci == 0:
            _ax.legend(fontsize=7)

        _ax2 = _axes_b[1, _ci]
        _ax2.plot(_ns_vals, _rmse_vals, marker="o", color=_COLORS[_ci], lw=2)
        _ax2.axvline(
            _ns_random_chance,
            color="gray",
            lw=1.2,
            linestyle=":",
            label=f"Random chance (P=1/{REGIMES})",
        )
        _ax2.set_title(f"{LABELS[_pk]} — RMSE", fontsize=9)
        _ax2.set_xlabel("noise_scale")
        _ax2.grid(True, alpha=0.3)
        if _ci == 0:
            _ax2.legend(fontsize=7)

    plt.suptitle(
        "Section B: XGBoost ML classifier sensitivity (N=3 000, 50 simulations)\n"
        f"Shaded = ±1 std  ·  Dotted vertical = noise_scale where P(correct)=1/{REGIMES}",
        fontweight="bold",
        fontsize=10,
    )
    plt.tight_layout()
    mo.as_html(_fig_b)
    return


@app.cell
def _(mo):
    mo.md("""
    ---
    ## Section C — Direct comparison at matched P(correct)

    Both grids are re-indexed to a common x-axis: **mean P(correct classification) = CM diagonal mean**.
    noisify (solid) and ML (dashed) lines are overlaid on the same bias/RMSE panels.
    A vertical dashed line marks **P(correct) = 1/R = 0.5** (random chance).
    """)
    return


@app.cell
def _(LABELS, PARAMS, TRUE, df_a, df_b, mo, np, plt):
    _COLORS = ["#4878d0", "#d65f5f", "#6acc65", "#ee854a"]
    _fig_c, _axes_c = plt.subplots(2, len(PARAMS), figsize=(14, 7))

    for _ci, _pk in enumerate(PARAMS):
        # noisify: compute p_correct from mw (= 1 - weight/2 for R=2)
        _weight_vals_a = sorted(df_a["weight"].unique())
        _pc_a = [1.0 - w / 2.0 for w in _weight_vals_a]
        _bias_a, _rmse_a = [], []
        for _w in _weight_vals_a:
            _v = df_a[df_a["weight"] == _w][_pk].values
            _bias_a.append(float(np.mean(_v) - TRUE[_pk]))
            _rmse_a.append(float(np.sqrt(np.mean((_v - TRUE[_pk]) ** 2))))

        # ml: p_correct from data
        _ns_vals_b = sorted(df_b["noise_scale"].unique())
        _pc_b = [
            float(df_b[df_b["noise_scale"] == ns]["p_correct"].mean()) for ns in _ns_vals_b
        ]
        _bias_b, _rmse_b = [], []
        for _ns in _ns_vals_b:
            _v = df_b[df_b["noise_scale"] == _ns][_pk].values
            _bias_b.append(float(np.mean(_v) - TRUE[_pk]))
            _rmse_b.append(float(np.sqrt(np.mean((_v - TRUE[_pk]) ** 2))))

        for _row, (_y_a, _y_b, _ylabel) in enumerate([
            (_bias_a, _bias_b, "Bias"),
            (_rmse_a, _rmse_b, "RMSE"),
        ]):
            _ax = _axes_c[_row, _ci]
            _ax.plot(_pc_a, _y_a, marker="o", color=_COLORS[_ci], lw=2, label="noisify")
            _ax.plot(
                _pc_b,
                _y_b,
                marker="s",
                color=_COLORS[_ci],
                lw=2,
                linestyle="--",
                alpha=0.7,
                label="ML",
            )
            _ax.axvline(
                0.5, color="gray", lw=1.2, linestyle=":", label="Random chance (P=1/R)"
            )
            if _row == 0:
                _ax.axhline(0, color="black", lw=0.8, linestyle="--", alpha=0.5)
            _ax.set_title(f"{LABELS[_pk]} — {_ylabel}", fontsize=9)
            _ax.set_xlabel("P(correct classification)")
            _ax.grid(True, alpha=0.3)
            if _ci == 0:
                _ax.legend(fontsize=7)

    plt.suptitle(
        "Section C: noisify vs. ML at matched P(correct)  (N=3 000, 50 simulations)\n"
        "Vertical dotted = random chance  ·  noisify solid, ML dashed",
        fontweight="bold",
        fontsize=10,
    )
    plt.tight_layout()
    mo.as_html(_fig_c)
    return


@app.cell
def _(mo):
    mo.md("""
    ---
    ## Section D — Coverage rates

    95% confidence intervals from Hessian SEs (`DriverSpecificProbUberMLE`).
    Both noisify (varying `weight`) and ML (varying `noise_scale`) grids are shown.
    Reference lines: 95% nominal coverage and the random-chance coverage level.
    """)
    return


@app.cell
def _(
    B0,
    B1,
    DRIVERS,
    N_SIM,
    PARAMS,
    PERIODS,
    REGIMES,
    SD,
    TRUE,
    UberDatasetCreatorHet,
    contextlib,
    extract_estimator_inputs,
    io,
    np,
    pd,
):
    from ml_switching_reg.mle import DriverSpecificProbUberMLE


    def _run_mle_coverage(
        weight=None, noise_scale=None, classifier_mode="noisify", seed_base=3000
    ):
        hits = {p: [] for p in PARAMS}
        for _sim in range(N_SIM):
            _u = UberDatasetCreatorHet(
                drivers=DRIVERS,
                time_periods=PERIODS,
                regimes=REGIMES,
                seed=seed_base + _sim,
            )
            _kw = dict(
                seed=seed_base + _sim,
                beta0=B0,
                beta1=B1,
                y_sd=SD,
                classifier_mode=classifier_mode,
            )
            if classifier_mode == "noisify":
                _kw["weight"] = weight
            else:
                _kw["noise_scale"] = noise_scale
            _df, _mw = _u.construct(**_kw)
            _y, _Xl, _cp, _cm = extract_estimator_inputs(_df, _mw, REGIMES)
            try:
                with contextlib.redirect_stdout(io.StringIO()):
                    _mod = DriverSpecificProbUberMLE.from_arrays(_y, _Xl, _cp, _cm)
                    _res, _ = _mod.fit()
                _p = _res.params.values[:-1].reshape(REGIMES, 2)
                _se = _res.bse.values[:-1].reshape(REGIMES, 2)
                for _pk, (_r, _c) in zip(PARAMS, [(0, 0), (0, 1), (1, 0), (1, 1)]):
                    _lo = _p[_r, _c] - 1.96 * _se[_r, _c]
                    _hi = _p[_r, _c] + 1.96 * _se[_r, _c]
                    hits[_pk].append(int(_lo <= TRUE[_pk] <= _hi))
            except Exception:
                pass
        return {pk: np.mean(v) if v else np.nan for pk, v in hits.items()}


    _WEIGHT_GRID_D = [0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 0.9, 0.95]
    _NS_GRID_D = [0.1, 0.5, 1.0, 2.0, 5.0, 10.0]

    cov_noisify = []
    for _w in _WEIGHT_GRID_D:
        _c = _run_mle_coverage(weight=_w, classifier_mode="noisify", seed_base=3000)
        _c["weight"] = _w
        cov_noisify.append(_c)
        print(f"  coverage noisify weight={_w:.2f}  {_c}")

    cov_ml = []
    for _ns in _NS_GRID_D:
        _c = _run_mle_coverage(noise_scale=_ns, classifier_mode="ml", seed_base=4000)
        _c["noise_scale"] = _ns
        cov_ml.append(_c)
        print(f"  coverage ml noise_scale={_ns:.2f}  {_c}")

    df_cov_n = pd.DataFrame(cov_noisify)
    df_cov_ml = pd.DataFrame(cov_ml)
    df_cov_n, df_cov_ml
    return df_cov_ml, df_cov_n


@app.cell
def _(LABELS, PARAMS, REGIMES, df_cov_ml, df_cov_n, mo, np, plt):
    _COLORS = ["#4878d0", "#d65f5f", "#6acc65", "#ee854a"]
    _fig_d, _axes_d = plt.subplots(1, 2, figsize=(12, 5))

    for _ci, _pk in enumerate(PARAMS):
        _axes_d[0].plot(
            df_cov_n["weight"],
            df_cov_n[_pk],
            marker="o",
            color=_COLORS[_ci],
            lw=2,
            label=LABELS[_pk],
        )
        _axes_d[1].plot(
            df_cov_ml["noise_scale"],
            df_cov_ml[_pk],
            marker="o",
            color=_COLORS[_ci],
            lw=2,
            label=LABELS[_pk],
        )

    # noisify: vertical line at weight=1 (where P(correct)=1/R)
    _axes_d[0].axvline(
        1.0, color="gray", lw=1.0, linestyle=":", label=f"Random chance (weight=1)"
    )
    _axes_d[0].axhline(0.95, color="black", lw=1.2, linestyle="--", label="95% nominal")

    # ml: vertical line at interpolated noise_scale where P(correct)=1/R
    _ns_vals_d = sorted(df_cov_ml["noise_scale"].unique())
    _pc_d = [
        float(df_cov_ml[df_cov_ml["noise_scale"] == ns]["p_correct"].mean())
        if "p_correct" in df_cov_ml.columns
        else float("nan")
        for ns in _ns_vals_d
    ]
    _ns_rc = (
        float(np.interp(1.0 / REGIMES, list(reversed(_pc_d)), list(reversed(_ns_vals_d))))
        if not any(np.isnan(_pc_d))
        else _ns_vals_d[-1]
    )
    _axes_d[1].axvline(
        _ns_rc, color="gray", lw=1.0, linestyle=":", label=f"Random chance (P=1/{REGIMES})"
    )
    _axes_d[1].axhline(0.95, color="black", lw=1.2, linestyle="--", label="95% nominal")

    for _ax, _xlabel in zip(_axes_d, ["weight", "noise_scale"]):
        _ax.set_ylim(0, 1.05)
        _ax.set_xlabel(_xlabel)
        _ax.set_ylabel("Coverage rate")
        _ax.grid(True, alpha=0.3)
        _ax.legend(fontsize=7)

    _axes_d[0].set_title("noisify_matrix", fontsize=10)
    _axes_d[1].set_title("XGBoost ML classifier", fontsize=10)

    plt.suptitle(
        "Section D: 95% CI coverage rates  (N=3 000, 50 simulations)\n"
        "Dashed = 95% nominal  ·  Dotted vertical = random-chance threshold",
        fontweight="bold",
        fontsize=10,
    )
    plt.tight_layout()
    mo.as_html(_fig_d)
    return


if __name__ == "__main__":
    app.run()
