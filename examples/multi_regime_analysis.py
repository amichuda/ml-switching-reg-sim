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
    # Multi-Regime Analysis: Scaling and Misclassification

    **Aleksandr Michuda** · Simulation Study

    ---

    ## Overview

    This notebook tests the IRLS switching regression estimator for **R > 2 regimes**,
    examining how performance scales with:

    - **N×T** (total observations, fixing N/T ratio)
    - **N** (number of drivers, fixing T)
    - **T** (time periods per driver, fixing N)
    - **Misclassification severity** (`weight`), comparing R ∈ {2, 3, 4}

    All experiments use `noisify_matrix`.  A **random-chance reference line** is shown
    on each panel: the bias/RMSE expected when the classifier is fully uninformative
    (P = 1/R for each regime).  This threshold differs by R:
    - R=2: P(correct) = 0.50
    - R=3: P(correct) = 0.33
    - R=4: P(correct) = 0.25
    """)
    return


@app.cell
def _(mo):
    mo.md("""
    ## IRLS runner and helpers
    """)
    return


@app.cell
def _(MLSwitchingRegIRLS, contextlib, io, np, pd):
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
            beta, s2 = mod.fit(
                beta_0=b0, sigma2_0=float(np.var(y)), tol=1e-7, max_iter=500
            )
        return beta, float(np.sqrt(max(s2, 1e-10)))

    def make_true(R):
        """Generate fixed true parameters for R regimes."""
        beta0 = [float(1.0 + 2.0 * r) for r in range(R)]
        beta1 = [float(-1.5 + r) for r in range(R)]
        sd = [1.0] * R
        return beta0, beta1, sd

    def simulate_grid(
        grid_param,
        grid_vals,
        R,
        N_SIM,
        fixed_drivers=None,
        fixed_periods=None,
        weight=0.3,
        seed_base=0,
        UberDatasetCreatorHet=None,
        extract_estimator_inputs=None,
        np=None,
    ):
        """Run Monte Carlo over a 1D grid.

        grid_param : 'drivers' | 'periods'
        grid_vals  : list of values to sweep
        R          : number of regimes
        """
        beta0, beta1, sd = make_true(R)
        rows = []
        for val in grid_vals:
            drv = val if grid_param == "drivers" else fixed_drivers
            prd = val if grid_param == "periods" else fixed_periods
            for sim in range(N_SIM):
                u = UberDatasetCreatorHet(
                    drivers=drv, time_periods=prd, regimes=R, seed=seed_base + sim
                )
                df, mw = u.construct(
                    seed=seed_base + sim,
                    beta0=beta0,
                    beta1=beta1,
                    y_sd=sd,
                    weight=weight,
                )
                y, Xl, cp, cm, _ = extract_estimator_inputs(df, mw, R)
                try:
                    beta_hat, _ = run_irls(y, Xl, cp, cm)
                    row = {grid_param: val, "sim": sim, "NT": drv * prd}
                    for r in range(R):
                        row[f"beta0_{r}"] = float(beta_hat[r, 0])
                        row[f"beta1_{r}"] = float(beta_hat[r, 1])
                    rows.append(row)
                except Exception:
                    pass
        return pd.DataFrame(rows), beta0, beta1

    return make_true, run_irls, simulate_grid


@app.cell
def _(mo):
    mo.md("""
    ## Random-chance baselines by R

    Computed at `weight=0.99` for R ∈ {2, 3, 4}.
    """)
    return


@app.cell
def _(
    UberDatasetCreatorHet,
    extract_estimator_inputs,
    make_true,
    np,
    pd,
    run_irls,
):
    _N_RC = 20
    _DRV_RC = 100
    _PD_RC = 10
    rc_bias = {}
    rc_rmse = {}
    for _R in [2, 3, 4]:
        _b0, _b1, _sd = make_true(_R)
        _rows = []
        for _sim in range(_N_RC):
            _u = UberDatasetCreatorHet(
                drivers=_DRV_RC, time_periods=_PD_RC, regimes=_R, seed=8000 + _sim
            )
            _df, _mw = _u.construct(
                seed=8000 + _sim, beta0=_b0, beta1=_b1, y_sd=_sd, weight=0.99
            )
            _y, _Xl, _cp, _cm, _ = extract_estimator_inputs(_df, _mw, _R)
            try:
                _beta, _ = run_irls(_y, _Xl, _cp, _cm)
                _r = {}
                for _r_idx in range(_R):
                    _r[f"beta0_{_r_idx}"] = float(_beta[_r_idx, 0])
                    _r[f"beta1_{_r_idx}"] = float(_beta[_r_idx, 1])
                _rows.append(_r)
            except Exception:
                pass
        _df_rc = pd.DataFrame(_rows)
        rc_bias[_R] = {}
        rc_rmse[_R] = {}
        for _r_idx in range(_R):
            for _ptype, _true in [("beta0", _b0[_r_idx]), ("beta1", _b1[_r_idx])]:
                _pk = f"{_ptype}_{_r_idx}"
                _v = _df_rc[_pk].values
                rc_bias[_R][_pk] = float(np.mean(_v) - _true)
                rc_rmse[_R][_pk] = float(np.sqrt(np.mean((_v - _true) ** 2)))
        print(
            f"  R={_R} random-chance RMSE beta1: {[round(rc_rmse[_R][f'beta1_{r}'], 3) for r in range(_R)]}"
        )

    rc_bias, rc_rmse
    return rc_bias, rc_rmse


@app.cell
def _(mo):
    mo.md("""
    ---
    ## Section A — N×T scaling  (R=3, weight=0.3)

    Fix T=15, vary N ∈ {25, 50, 100, 200, 400}.  Total N×T scales proportionally.
    RMSE should shrink as 1/√N.
    """)
    return


@app.cell
def _(UberDatasetCreatorHet, extract_estimator_inputs, np, simulate_grid):
    _R_A = 3
    _N_GRID = [25, 50, 100, 200, 400]
    _T_A = 15
    _N_SIM_A = 50
    df_sec_a, _b0_a, _b1_a = simulate_grid(
        "drivers",
        _N_GRID,
        _R_A,
        _N_SIM_A,
        fixed_periods=_T_A,
        weight=0.3,
        seed_base=10000,
        UberDatasetCreatorHet=UberDatasetCreatorHet,
        extract_estimator_inputs=extract_estimator_inputs,
        np=np,
    )
    true_a = (_b0_a, _b1_a)
    print("Section A done, rows:", len(df_sec_a))
    df_sec_a, true_a
    return df_sec_a, true_a


@app.cell
def _(df_sec_a, mo, np, plt, rc_bias, rc_rmse, true_a):
    _R = 3
    _N_GRID = sorted(df_sec_a["drivers"].unique())
    _b0_a, _b1_a = true_a
    _COLORS = ["#4878d0", "#d65f5f", "#6acc65"]

    _fig_a, _axes_a = plt.subplots(2, _R, figsize=(12, 7))

    for _r in range(_R):
        for _row, (_ptype, _true_vals) in enumerate(
            [("beta0", _b0_a), ("beta1", _b1_a)]
        ):
            _pk = f"{_ptype}_{_r}"
            _true_v = _true_vals[_r]
            _bias_v, _rmse_v = [], []
            for _n in _N_GRID:
                _v = df_sec_a[df_sec_a["drivers"] == _n][_pk].values
                _bias_v.append(float(np.mean(_v) - _true_v))
                _rmse_v.append(float(np.sqrt(np.mean((_v - _true_v) ** 2))))
            _vals = _bias_v if _row == 0 else _rmse_v
            _ax = _axes_a[_row, _r]
            _ax.plot(
                _N_GRID,
                _vals,
                marker="o",
                color=_COLORS[_r],
                lw=2,
                label=f"Regime {_r} (β{'₀' if _ptype == 'beta0' else '₁'}={_true_v:.1f})",
            )
            if _row == 0:
                _ax.axhline(0, color="black", lw=0.8, linestyle="--", alpha=0.5)
                _ax.axhline(
                    rc_bias[_R][_pk],
                    color="gray",
                    lw=1.2,
                    linestyle=":",
                    label=f"Random chance (1/R={1.0 / _R:.2f})",
                )
            else:
                _ax.axhline(
                    rc_rmse[_R][_pk],
                    color="gray",
                    lw=1.2,
                    linestyle=":",
                    label=f"Random chance (1/R={1.0 / _R:.2f})",
                )
            _ax.set_title(
                f"Regime {_r} {'β₀' if _ptype == 'beta0' else 'β₁'} — {'Bias' if _row == 0 else 'RMSE'}",
                fontsize=9,
            )
            _ax.set_xlabel("N (drivers, T=15 fixed)")
            _ax.grid(True, alpha=0.3)
            if _r == 0:
                _ax.legend(fontsize=7)

    plt.suptitle(
        "Section A: N×T scaling  (R=3, weight=0.3, 50 simulations)\n"
        "Dotted = random-chance level (P=1/3)",
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
    ## Section B — N scaling  (R=3, T=15, weight=0.3)

    Vary N ∈ {25, 50, 100, 200, 400} with T=15 fixed — same as Section A but the x-axis
    is drivers (not N×T) to isolate the driver-count effect.
    """)
    return


@app.cell
def _(df_sec_a, mo, np, plt, rc_bias, rc_rmse, true_a):
    _R = 3
    _N_GRID = sorted(df_sec_a["drivers"].unique())
    _b0_a, _b1_a = true_a
    _COLORS = ["#4878d0", "#d65f5f", "#6acc65"]

    _fig_b, _axes_b = plt.subplots(2, 2, figsize=(10, 7))

    for _col, (_ptype, _true_vals, _sym) in enumerate(
        [
            ("beta0", _b0_a, "β₀"),
            ("beta1", _b1_a, "β₁"),
        ]
    ):
        for _row, _ylabel in enumerate(["Bias", "RMSE"]):
            _ax = _axes_b[_row, _col]
            for _r in range(_R):
                _pk = f"{_ptype}_{_r}"
                _true_v = _true_vals[_r]
                _vals = []
                for _n in _N_GRID:
                    _v = df_sec_a[df_sec_a["drivers"] == _n][_pk].values
                    if _row == 0:
                        _vals.append(float(np.mean(_v) - _true_v))
                    else:
                        _vals.append(float(np.sqrt(np.mean((_v - _true_v) ** 2))))
                _ax.plot(
                    _N_GRID,
                    _vals,
                    marker="o",
                    color=_COLORS[_r],
                    lw=2,
                    label=f"Regime {_r} (true={_true_v:.1f})",
                )
                if _row == 0:
                    _ax.axhline(
                        rc_bias[_R][_pk],
                        color=_COLORS[_r],
                        lw=0.8,
                        linestyle=":",
                        alpha=0.6,
                    )
                else:
                    _ax.axhline(
                        rc_rmse[_R][_pk],
                        color=_COLORS[_r],
                        lw=0.8,
                        linestyle=":",
                        alpha=0.6,
                    )
            if _row == 0:
                _ax.axhline(0, color="black", lw=0.8, linestyle="--", alpha=0.5)
            _ax.set_title(f"{_sym} — {_ylabel}", fontsize=9)
            _ax.set_xlabel("N (drivers, T=15)")
            _ax.grid(True, alpha=0.3)
            _ax.legend(fontsize=7)

    plt.suptitle(
        "Section B: N scaling  (R=3, T=15, weight=0.3, 50 simulations)\n"
        "Dotted lines = random-chance level per regime",
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
    ## Section C — T scaling  (R=3, N=100, weight=0.3)

    Vary T ∈ {5, 10, 15, 25, 40} with N=100 fixed to isolate the time-periods effect.
    """)
    return


@app.cell
def _(UberDatasetCreatorHet, extract_estimator_inputs, np, simulate_grid):
    _R_C = 3
    _T_GRID = [5, 10, 15, 25, 40]
    _N_C = 100
    _N_SIM_C = 50
    df_sec_c, _b0_c, _b1_c = simulate_grid(
        "periods",
        _T_GRID,
        _R_C,
        _N_SIM_C,
        fixed_drivers=_N_C,
        weight=0.3,
        seed_base=20000,
        UberDatasetCreatorHet=UberDatasetCreatorHet,
        extract_estimator_inputs=extract_estimator_inputs,
        np=np,
    )
    true_c = (_b0_c, _b1_c)
    print("Section C done, rows:", len(df_sec_c))
    df_sec_c, true_c
    return df_sec_c, true_c


@app.cell
def _(df_sec_c, mo, np, plt, rc_bias, rc_rmse, true_c):
    _R = 3
    _T_GRID = sorted(df_sec_c["periods"].unique())
    _b0_c, _b1_c = true_c
    _COLORS = ["#4878d0", "#d65f5f", "#6acc65"]

    _fig_c, _axes_c = plt.subplots(2, 2, figsize=(10, 7))

    for _col, (_ptype, _true_vals, _sym) in enumerate(
        [
            ("beta0", _b0_c, "β₀"),
            ("beta1", _b1_c, "β₁"),
        ]
    ):
        for _row, _ylabel in enumerate(["Bias", "RMSE"]):
            _ax = _axes_c[_row, _col]
            for _r in range(_R):
                _pk = f"{_ptype}_{_r}"
                _true_v = _true_vals[_r]
                _vals = []
                for _t in _T_GRID:
                    _v = df_sec_c[df_sec_c["periods"] == _t][_pk].values
                    if _row == 0:
                        _vals.append(float(np.mean(_v) - _true_v))
                    else:
                        _vals.append(float(np.sqrt(np.mean((_v - _true_v) ** 2))))
                _ax.plot(
                    _T_GRID,
                    _vals,
                    marker="o",
                    color=_COLORS[_r],
                    lw=2,
                    label=f"Regime {_r} (true={_true_v:.1f})",
                )
                if _row == 0:
                    _ax.axhline(
                        rc_bias[_R][_pk],
                        color=_COLORS[_r],
                        lw=0.8,
                        linestyle=":",
                        alpha=0.6,
                    )
                else:
                    _ax.axhline(
                        rc_rmse[_R][_pk],
                        color=_COLORS[_r],
                        lw=0.8,
                        linestyle=":",
                        alpha=0.6,
                    )
            if _row == 0:
                _ax.axhline(0, color="black", lw=0.8, linestyle="--", alpha=0.5)
            _ax.set_title(f"{_sym} — {_ylabel}", fontsize=9)
            _ax.set_xlabel("T (time periods, N=100)")
            _ax.grid(True, alpha=0.3)
            _ax.legend(fontsize=7)

    plt.suptitle(
        "Section C: T scaling  (R=3, N=100, weight=0.3, 50 simulations)\n"
        "Dotted lines = random-chance level per regime",
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
    ## Section D — Misclassification severity  (N=200, T=15, R ∈ {2, 3, 4})

    For each R, sweep `weight ∈ [0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 0.9, 0.95]`.
    R=2, 3, 4 are overlaid on the same panels (different line styles).
    Annotated random-chance lines: 1/R = 0.50, 0.33, 0.25.
    """)
    return


@app.cell
def _(
    UberDatasetCreatorHet,
    extract_estimator_inputs,
    make_true,
    pd,
    run_irls,
):
    _WEIGHT_GRID_D = [0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 0.9, 0.95]
    _N_D, _T_D, _N_SIM_D = 200, 15, 50

    rows_d = {R: [] for R in [2, 3, 4]}
    true_d = {}

    for _R in [2, 3, 4]:
        _b0, _b1, _sd = make_true(_R)
        true_d[_R] = (_b0, _b1)
        for _w in _WEIGHT_GRID_D:
            for _sim in range(_N_SIM_D):
                _u = UberDatasetCreatorHet(
                    drivers=_N_D, time_periods=_T_D, regimes=_R, seed=30000 + _sim
                )
                _df, _mw = _u.construct(
                    seed=30000 + _sim, beta0=_b0, beta1=_b1, y_sd=_sd, weight=_w
                )
                _y, _Xl, _cp, _cm, _ = extract_estimator_inputs(_df, _mw, _R)
                try:
                    _beta, _ = run_irls(_y, _Xl, _cp, _cm)
                    _row = {"weight": _w}
                    for _r in range(_R):
                        _row[f"beta0_{_r}"] = float(_beta[_r, 0])
                        _row[f"beta1_{_r}"] = float(_beta[_r, 1])
                    rows_d[_R].append(_row)
                except Exception:
                    pass
        print(f"  Section D R={_R} done")

    df_d = {R: pd.DataFrame(rows_d[R]) for R in [2, 3, 4]}
    df_d, true_d
    return df_d, true_d


@app.cell
def _(df_d, mo, np, plt, true_d):
    _WEIGHT_GRID_D = [0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 0.9, 0.95]
    _R_LIST = [2, 3, 4]
    _LINESTYLES = {2: "-", 3: "--", 4: ":"}
    _COLORS_R = {2: "#4878d0", 3: "#d65f5f", 4: "#6acc65"}
    _RC_COLORS = {2: "#4878d0", 3: "#d65f5f", 4: "#6acc65"}

    _fig_d, _axes_d = plt.subplots(2, 2, figsize=(12, 8))

    for _col, _ptype in enumerate(["beta0", "beta1"]):
        for _row, _ylabel in enumerate(["Bias", "RMSE"]):
            _ax = _axes_d[_row, _col]
            for _R in _R_LIST:
                _b0, _b1 = true_d[_R]
                _true_vals = _b0 if _ptype == "beta0" else _b1
                # Average bias/RMSE across all regimes for this R
                _avg_bias, _avg_rmse = [], []
                for _w in _WEIGHT_GRID_D:
                    _grp = df_d[_R][df_d[_R]["weight"] == _w]
                    _biases = [
                        float(np.mean(_grp[f"{_ptype}_{r}"].values) - _true_vals[r])
                        for r in range(_R)
                    ]
                    _rmses = [
                        float(
                            np.sqrt(
                                np.mean(
                                    (_grp[f"{_ptype}_{r}"].values - _true_vals[r]) ** 2
                                )
                            )
                        )
                        for r in range(_R)
                    ]
                    _avg_bias.append(
                        float(np.mean(np.abs(_biases)))
                    )  # mean abs bias across regimes
                    _avg_rmse.append(float(np.mean(_rmses)))

                _vals = _avg_bias if _row == 0 else _avg_rmse
                _ax.plot(
                    _WEIGHT_GRID_D,
                    _vals,
                    marker="o",
                    color=_COLORS_R[_R],
                    lw=2,
                    linestyle=_LINESTYLES[_R],
                    label=f"R={_R}",
                )

            if _row == 0:
                _ax.axhline(0, color="black", lw=0.8, linestyle="--", alpha=0.5)
            # weight=1.0 is where P(correct)=1/R for all R with noisify_matrix
            _ax.axvline(
                1.0,
                color="gray",
                lw=1.2,
                linestyle=":",
                label="Random chance (weight=1, P=1/R)",
            )
            _ax.set_title(
                f"{'β₀' if _ptype == 'beta0' else 'β₁'} — mean |{_ylabel}| across regimes",
                fontsize=9,
            )
            _ax.set_xlabel("weight")
            _ax.grid(True, alpha=0.3)
            _ax.legend(fontsize=7)

    plt.suptitle(
        "Section D: Misclassification severity  (N=200, T=15, R ∈ {2,3,4}, 50 simulations)\n"
        "Solid R=2  ·  Dashed R=3  ·  Dotted R=4  ·  Dotted vertical = random-chance weight",
        fontweight="bold",
        fontsize=10,
    )
    plt.tight_layout()
    mo.as_html(_fig_d)
    return


if __name__ == "__main__":
    app.run()
