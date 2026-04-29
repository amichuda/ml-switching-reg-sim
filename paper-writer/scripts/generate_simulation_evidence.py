"""Generate paper figures and appendix tables from Monte Carlo outputs.

This script is intentionally self-contained under paper-writer. It reads the
cached simulation outputs in ../mc_coverage_results and writes all paper-facing
artifacts under paper-writer/results.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
PAPER_DIR = SCRIPT_DIR.parent
ROOT_DIR = PAPER_DIR.parent
INPUT_DIR = ROOT_DIR / "mc_coverage_results"
NAIVE_DIR = ROOT_DIR / "mc_naive_results"
FIGURE_DIR = PAPER_DIR / "results" / "figures"
TABLE_DIR = PAPER_DIR / "results" / "tables"

WEIGHTS = [0.0, 0.1, 0.3, 0.5, 0.7, 0.9]
RHOS = [0.0, 0.3, 0.6, 0.9]
REGIMES = [2, 4, 10]
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


def _ensure_dirs() -> None:
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    TABLE_DIR.mkdir(parents=True, exist_ok=True)


def _true_val(row: pd.Series) -> float:
    params = TRUE_PARAMS[int(row["R"])]
    regime = int(row["regime"])
    param_idx = int(row["param_idx"])
    if param_idx == 0:
        return float(params["beta0"][regime])
    if param_idx == 1:
        return float(params["beta1"][regime])
    return float("nan")


def load_results() -> pd.DataFrame:
    files = sorted(INPUT_DIR.glob("cell_*.parquet"))
    if not files:
        raise FileNotFoundError(f"No Monte Carlo parquet files found in {INPUT_DIR}")

    columns = [
        "weight",
        "R",
        "rho",
        "rep",
        "regime",
        "param_idx",
        "param_name",
        "true_val",
        "irls_est",
        "mle_est",
        "mle_bse",
        "mle_bse_boot",
        "mle_ci_low_boot",
        "mle_ci_high_boot",
        "mle_pvalue_boot",
        "mle_converged",
        "irls_s2",
        "mle_s2",
        "true_s2",
        "t_irls",
        "t_mle",
        "t_boot",
    ]
    parts = [pd.read_parquet(path, columns=columns) for path in files]
    data = pd.concat(parts, ignore_index=True)

    # Some early cached files used an obsolete true-beta ordering. Recompute from
    # the design constants so every artifact is internally consistent.
    data["true_val"] = data.apply(_true_val, axis=1)
    data["param_type"] = np.where(data["param_idx"].eq(0), "Intercept", "Slope")
    data["classification_error"] = data["weight"] * (1.0 - 1.0 / data["R"])
    data["p_correct"] = 1.0 - data["classification_error"]
    data["mle_error"] = data["mle_est"] - data["true_val"]
    data["irls_error"] = data["irls_est"] - data["true_val"]
    data["mle_ci_low"] = data["mle_est"] - 1.96 * data["mle_bse"]
    data["mle_ci_high"] = data["mle_est"] + 1.96 * data["mle_bse"]
    data["covered_wald"] = data["true_val"].between(
        data["mle_ci_low"], data["mle_ci_high"]
    )
    data["covered_boot"] = data["true_val"].between(
        data["mle_ci_low_boot"], data["mle_ci_high_boot"]
    ).astype(float)
    data["covered_wald"] = data["covered_wald"].astype(float)
    data.loc[data["mle_ci_low_boot"].isna(), "covered_boot"] = np.nan
    return data


def load_naive_results() -> pd.DataFrame | None:
    """Load naive hard-classification baseline results, if available."""
    if not NAIVE_DIR.exists():
        return None
    files = sorted(NAIVE_DIR.glob("cell_*.parquet"))
    if not files:
        return None
    parts = [pd.read_parquet(path) for path in files]
    naive = pd.concat(parts, ignore_index=True)
    # Recompute true_val from the design constants for parity with the MLE table.
    naive["true_val"] = naive.apply(_true_val, axis=1)
    naive["naive_error"] = naive["naive_est"] - naive["true_val"]
    naive["param_type"] = np.where(naive["param_idx"].eq(0), "Intercept", "Slope")
    naive["classification_error"] = naive["weight"] * (1.0 - 1.0 / naive["R"])
    naive["covered_naive"] = naive["true_val"].between(
        naive["naive_ci_low"], naive["naive_ci_high"]
    ).astype(float)
    naive.loc[~naive["naive_identified"], "covered_naive"] = np.nan
    return naive


def estimator_long(
    data: pd.DataFrame, naive: pd.DataFrame | None = None
) -> pd.DataFrame:
    shared = [
        "weight",
        "classification_error",
        "p_correct",
        "R",
        "rho",
        "rep",
        "regime",
        "param_idx",
        "param_name",
        "param_type",
        "true_val",
    ]
    irls = data[shared + ["irls_est", "irls_error"]].rename(
        columns={"irls_est": "estimate", "irls_error": "error"}
    )
    irls["estimator"] = "IRLS"
    mle = data[shared + ["mle_est", "mle_error"]].rename(
        columns={"mle_est": "estimate", "mle_error": "error"}
    )
    mle["estimator"] = "MLE"
    frames = [irls, mle]
    if naive is not None and len(naive) > 0:
        naive_shared = naive.copy()
        naive_shared["p_correct"] = 1.0 - naive_shared["classification_error"]
        naive_long = naive_shared[shared + ["naive_est", "naive_error"]].rename(
            columns={"naive_est": "estimate", "naive_error": "error"}
        )
        naive_long["estimator"] = "Naive"
        frames.append(naive_long)
    return pd.concat(frames, ignore_index=True)


def summarize_performance(long: pd.DataFrame) -> pd.DataFrame:
    summary = (
        long.groupby(["estimator", "R", "rho", "weight", "classification_error", "param_type"])
        .agg(
            bias=("error", "mean"),
            mean_abs_bias=("error", lambda x: float(np.mean(np.abs(x)))),
            rmse=("error", lambda x: float(np.sqrt(np.mean(np.square(x))))),
            sd_estimate=("estimate", "std"),
            n_param_rep=("error", "size"),
        )
        .reset_index()
    )
    return summary


def summarize_coverage(
    data: pd.DataFrame, naive: pd.DataFrame | None = None
) -> pd.DataFrame:
    slopes = data[data["param_type"].eq("Slope")]
    coverage = (
        slopes.groupby(["R", "rho", "weight", "classification_error"])
        .agg(
            wald_coverage=("covered_wald", "mean"),
            bootstrap_coverage=("covered_boot", "mean"),
            mean_wald_se=("mle_bse", "mean"),
            mean_bootstrap_se=("mle_bse_boot", "mean"),
            n_param_rep=("mle_est", "size"),
        )
        .reset_index()
    )
    if naive is not None and len(naive) > 0:
        naive_slopes = naive[naive["param_type"].eq("Slope")]
        naive_cov = (
            naive_slopes.groupby(["R", "rho", "weight", "classification_error"])
            .agg(
                naive_coverage=("covered_naive", "mean"),
                mean_naive_se=("naive_se", "mean"),
                naive_identified_rate=("naive_identified", "mean"),
            )
            .reset_index()
        )
        coverage = coverage.merge(
            naive_cov,
            on=["R", "rho", "weight", "classification_error"],
            how="left",
        )
    return coverage


def summarize_timing(data: pd.DataFrame) -> pd.DataFrame:
    per_rep = data.drop_duplicates(["R", "rho", "weight", "rep"])
    timing = (
        per_rep.groupby(["R", "rho", "weight"])
        .agg(
            convergence_rate=("mle_converged", "mean"),
            mean_irls_seconds=("t_irls", "mean"),
            mean_mle_seconds=("t_mle", "mean"),
            mean_bootstrap_seconds=("t_boot", "mean"),
            reps=("rep", "nunique"),
        )
        .reset_index()
    )
    timing["mle_to_irls_time_ratio"] = timing["mean_mle_seconds"] / timing[
        "mean_irls_seconds"
    ]
    timing["bootstrap_to_mle_time_ratio"] = timing["mean_bootstrap_seconds"] / timing[
        "mean_mle_seconds"
    ]
    return timing


def save_table_outputs(
    data: pd.DataFrame,
    performance: pd.DataFrame,
    coverage: pd.DataFrame,
    timing: pd.DataFrame,
) -> None:
    design = pd.DataFrame(
        [
            {
                "design_axis": "Regimes",
                "values": ", ".join(str(x) for x in REGIMES),
                "notes": "Number of latent switching-regression regimes.",
            },
            {
                "design_axis": "Misclassification weight",
                "values": ", ".join(str(x) for x in WEIGHTS),
                "notes": "0 is perfect classification; 1 would be uniform classification.",
            },
            {
                "design_axis": "Shock correlation",
                "values": ", ".join(str(x) for x in RHOS),
                "notes": "Equicorrelation in regime-specific drought shocks.",
            },
            {
                "design_axis": "Drivers and periods",
                "values": f"{DRIVERS} drivers, {TIME_PERIODS} periods",
                "notes": "Each Monte Carlo replication has 3,000 observations.",
            },
            {
                "design_axis": "Replications",
                "values": str(N_REPS),
                "notes": "Per cell in the full factorial design.",
            },
        ]
    )

    design.to_csv(TABLE_DIR / "table_A0_design_summary.csv", index=False)
    design.to_latex(TABLE_DIR / "table_A0_design_summary.tex", index=False, escape=True)

    performance.to_csv(TABLE_DIR / "table_A1_performance_all_cells.csv", index=False)
    coverage.to_csv(TABLE_DIR / "table_A2_coverage_all_cells.csv", index=False)
    timing.to_csv(TABLE_DIR / "table_A3_convergence_timing_all_cells.csv", index=False)

    # Compact appendix tables: central shock-correlation case, slope coefficients.
    perf_compact = performance[
        performance["param_type"].eq("Slope") & performance["rho"].eq(0.6)
    ].copy()
    perf_compact = perf_compact[
        ["estimator", "R", "weight", "classification_error", "bias", "mean_abs_bias", "rmse"]
    ]
    for col in ["classification_error", "bias", "mean_abs_bias", "rmse"]:
        perf_compact[col] = perf_compact[col].map(lambda x: f"{x:.3f}")
    perf_compact.to_latex(
        TABLE_DIR / "table_A1_performance_rho_0_6.tex", index=False, escape=True
    )

    cov_compact_cols = ["R", "weight", "classification_error", "wald_coverage", "bootstrap_coverage"]
    if "naive_coverage" in coverage.columns:
        cov_compact_cols.append("naive_coverage")
    cov_compact = coverage[coverage["rho"].eq(0.6)][cov_compact_cols].copy()
    fmt_cols = [c for c in cov_compact_cols if c not in {"R", "weight"}]
    for col in fmt_cols:
        cov_compact[col] = cov_compact[col].map(
            lambda x: f"{x:.3f}" if pd.notna(x) else "—"
        )
    cov_compact.to_latex(
        TABLE_DIR / "table_A2_coverage_rho_0_6.tex", index=False, escape=True
    )

    timing_compact = timing.groupby(["R", "weight"]).agg(
        convergence_rate=("convergence_rate", "mean"),
        mean_irls_seconds=("mean_irls_seconds", "mean"),
        mean_mle_seconds=("mean_mle_seconds", "mean"),
        mean_bootstrap_seconds=("mean_bootstrap_seconds", "mean"),
    )
    timing_compact = timing_compact.reset_index()
    for col in [
        "convergence_rate",
        "mean_irls_seconds",
        "mean_mle_seconds",
        "mean_bootstrap_seconds",
    ]:
        timing_compact[col] = timing_compact[col].map(lambda x: f"{x:.3f}")
    timing_compact.to_latex(
        TABLE_DIR / "table_A3_timing_by_regime_weight.tex", index=False, escape=True
    )

    inventory = pd.DataFrame(
        [
            {
                "artifact": "result rows",
                "count": len(data),
            },
            {
                "artifact": "Monte Carlo cells",
                "count": data.groupby(["R", "rho", "weight"]).ngroups,
            },
            {
                "artifact": "completed replications per cell",
                "count": int(data.groupby(["R", "rho", "weight"])["rep"].nunique().min()),
            },
        ]
    )
    inventory.to_csv(TABLE_DIR / "artifact_inventory.csv", index=False)


def _save_current_figure(name: str) -> None:
    plt.savefig(FIGURE_DIR / f"{name}.png", dpi=300, bbox_inches="tight")
    plt.savefig(FIGURE_DIR / f"{name}.pdf", bbox_inches="tight")
    plt.close()


def plot_coverage_landscape(coverage: pd.DataFrame) -> None:
    fig, axes = plt.subplots(
        len(REGIMES),
        2,
        figsize=(12.5, 10.5),
        sharex=True,
        sharey=True,
        constrained_layout=True,
    )
    methods = [
        ("wald_coverage", "Analytical Wald"),
        ("bootstrap_coverage", "Score wild bootstrap"),
    ]
    vmin, vmax = -0.25, 0.25
    image = None
    for row, R in enumerate(REGIMES):
        for col, (metric, label) in enumerate(methods):
            ax = axes[row, col]
            pivot = (
                coverage[coverage["R"].eq(R)]
                .pivot(index="rho", columns="weight", values=metric)
                .reindex(index=RHOS, columns=WEIGHTS)
            )
            pivot = pivot.astype(float)
            gap = pivot - 0.95
            image = ax.imshow(
                gap.values,
                cmap="RdBu",
                vmin=vmin,
                vmax=vmax,
                origin="lower",
                aspect="auto",
            )
            for i, rho in enumerate(RHOS):
                for j, weight in enumerate(WEIGHTS):
                    value = pivot.loc[rho, weight]
                    ax.text(j, i, f"{value:.2f}", ha="center", va="center", fontsize=8)
            ax.set_xticks(range(len(WEIGHTS)))
            ax.set_xticklabels([str(w) for w in WEIGHTS])
            ax.set_yticks(range(len(RHOS)))
            ax.set_yticklabels([str(r) for r in RHOS])
            if row == 0:
                ax.set_title(label, fontsize=12, fontweight="bold")
            if col == 0:
                ax.set_ylabel(f"R={R}\nshock correlation", fontsize=10)
            if row == len(REGIMES) - 1:
                ax.set_xlabel("Misclassification weight")
    fig.suptitle(
        "Coverage stays near nominal except in the hardest classification designs",
        fontsize=15,
        fontweight="bold",
    )
    if image is not None:
        cbar = fig.colorbar(image, ax=axes.ravel().tolist(), shrink=0.75)
        cbar.set_label("Coverage minus 0.95 target")
    _save_current_figure("figure_1_coverage_landscape")


def plot_rmse_frontier(performance: pd.DataFrame) -> None:
    slopes = performance[performance["param_type"].eq("Slope")].copy()
    estimators = [e for e in ["MLE", "IRLS", "Naive"] if e in slopes["estimator"].unique()]
    fig, axes = plt.subplots(1, len(RHOS), figsize=(15, 4.2), sharey=True)
    colors = {2: "#2A6FBB", 4: "#D95F02", 10: "#1B9E77"}
    linestyles = {"MLE": "-", "IRLS": "--", "Naive": ":"}
    for ax, rho in zip(axes, RHOS):
        subset = slopes[slopes["rho"].eq(rho)]
        for estimator in estimators:
            for R in REGIMES:
                line = subset[subset["estimator"].eq(estimator) & subset["R"].eq(R)]
                line = line.sort_values("classification_error")
                ax.plot(
                    line["classification_error"],
                    line["rmse"],
                    marker="o",
                    color=colors[R],
                    linestyle=linestyles[estimator],
                    linewidth=2,
                    markersize=4,
                    label=f"{estimator}, R={R}",
                )
        ax.set_title(f"rho = {rho}")
        ax.set_xlabel("Classifier error probability")
        ax.grid(True, alpha=0.25)
    axes[0].set_ylabel("Slope RMSE")
    handles, labels = axes[-1].get_legend_handles_labels()
    fig.legend(handles, labels, loc="center left", bbox_to_anchor=(1.0, 0.5), frameon=False)
    fig.suptitle(
        "Sampling error rises smoothly with classification error and regime count",
        fontsize=15,
        fontweight="bold",
    )
    fig.tight_layout(rect=[0, 0, 0.86, 0.9])
    _save_current_figure("figure_2_rmse_frontier")


def plot_naive_vs_corrected(performance: pd.DataFrame, coverage: pd.DataFrame) -> None:
    """Side-by-side comparison: corrected MLE vs naive hard-classification baseline.

    Panel (a): slope bias as a function of classification error, by R. Shows the
    naive estimator's bias inflating with misclassification while the corrected
    MLE stays centered.

    Panel (b): 95% CI coverage as a function of classification error, by R. Shows
    the naive estimator's coverage collapsing while the corrected MLE retains
    near-nominal coverage.
    """
    if "Naive" not in performance["estimator"].unique():
        return
    if "naive_coverage" not in coverage.columns:
        return

    slopes = performance[performance["param_type"].eq("Slope")].copy()
    # Average bias (signed) and coverage across rho for clarity.
    bias_avg = (
        slopes.groupby(["estimator", "R", "classification_error"])["bias"]
        .mean()
        .reset_index()
    )
    cov_avg = (
        coverage.groupby(["R", "classification_error"])
        .agg(
            mle_wald=("wald_coverage", "mean"),
            mle_boot=("bootstrap_coverage", "mean"),
            naive=("naive_coverage", "mean"),
        )
        .reset_index()
    )

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.6), constrained_layout=True)
    colors = {2: "#2A6FBB", 4: "#D95F02", 10: "#1B9E77"}

    ax = axes[0]
    for R in REGIMES:
        for estimator, ls, marker in [("MLE", "-", "o"), ("Naive", ":", "s")]:
            line = bias_avg[
                bias_avg["estimator"].eq(estimator) & bias_avg["R"].eq(R)
            ].sort_values("classification_error")
            ax.plot(
                line["classification_error"],
                line["bias"],
                color=colors[R],
                linestyle=ls,
                marker=marker,
                markersize=5,
                linewidth=2,
                label=f"{estimator}, R={R}",
            )
    ax.axhline(0, color="black", linewidth=1.0, alpha=0.6)
    ax.set_xlabel("Classifier error probability")
    ax.set_ylabel("Mean slope bias")
    ax.set_title("(a) Bias: naive baseline vs corrected MLE")
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=8, ncol=2, frameon=False)

    ax = axes[1]
    for R in REGIMES:
        sub = cov_avg[cov_avg["R"].eq(R)].sort_values("classification_error")
        ax.plot(
            sub["classification_error"],
            sub["mle_boot"],
            color=colors[R],
            linestyle="-",
            marker="o",
            markersize=5,
            linewidth=2,
            label=f"MLE bootstrap, R={R}",
        )
        ax.plot(
            sub["classification_error"],
            sub["naive"],
            color=colors[R],
            linestyle=":",
            marker="s",
            markersize=5,
            linewidth=2,
            label=f"Naive, R={R}",
        )
    ax.axhline(0.95, color="black", linewidth=1.0, alpha=0.6, linestyle="--")
    ax.set_xlabel("Classifier error probability")
    ax.set_ylabel("Slope 95% CI coverage")
    ax.set_title("(b) Coverage: naive baseline vs corrected MLE")
    ax.set_ylim(0.0, 1.05)
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=8, ncol=2, frameon=False)

    fig.suptitle(
        "Confusion-matrix correction yields large bias and coverage gains over a naive baseline",
        fontsize=14,
        fontweight="bold",
    )
    _save_current_figure("figure_5_naive_vs_corrected")


def plot_bias_distributions(data: pd.DataFrame) -> None:
    subset = data[
        data["param_type"].eq("Slope") & data["R"].eq(2) & data["rho"].eq(0.0)
    ].copy()
    weights = WEIGHTS
    positions = np.arange(len(weights))
    fig, ax = plt.subplots(figsize=(10.5, 5.2))
    arrays = [subset[subset["weight"].eq(weight)]["mle_error"].values for weight in weights]
    box = ax.boxplot(
        arrays,
        positions=positions,
        widths=0.55,
        patch_artist=True,
        showfliers=False,
        medianprops={"color": "black", "linewidth": 1.5},
    )
    for patch, weight in zip(box["boxes"], weights):
        patch.set_facecolor(plt.cm.viridis(0.15 + 0.75 * weight / max(weights)))
        patch.set_alpha(0.8)
    rng = np.random.default_rng(20260427)
    for idx, weight in enumerate(weights):
        values = arrays[idx]
        if len(values) > 800:
            values = rng.choice(values, size=800, replace=False)
        jitter = rng.normal(0, 0.055, size=len(values))
        ax.scatter(
            np.full(len(values), idx) + jitter,
            values,
            s=7,
            alpha=0.18,
            color="black",
            linewidths=0,
        )
    ax.axhline(0, color="black", linewidth=1.2, alpha=0.8)
    ax.set_xticks(positions)
    ax.set_xticklabels([str(w) for w in weights])
    ax.set_xlabel("Misclassification weight")
    ax.set_ylabel("MLE slope estimation error")
    ax.set_title(
        "Baseline design: slope estimates remain centered as classification degrades",
        fontsize=14,
        fontweight="bold",
    )
    ax.grid(True, axis="y", alpha=0.25)
    fig.tight_layout()
    _save_current_figure("figure_3_baseline_bias_distribution")


def plot_computation_tradeoffs(timing: pd.DataFrame) -> None:
    avg = timing.groupby("R").agg(
        convergence_rate=("convergence_rate", "mean"),
        mean_irls_seconds=("mean_irls_seconds", "mean"),
        mean_mle_seconds=("mean_mle_seconds", "mean"),
        mean_bootstrap_seconds=("mean_bootstrap_seconds", "mean"),
    )
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.2), constrained_layout=True)
    x = np.arange(len(avg.index))
    width = 0.24
    axes[0].bar(x - width, avg["mean_irls_seconds"], width, label="IRLS", color="#6BAED6")
    axes[0].bar(x, avg["mean_mle_seconds"], width, label="MLE", color="#756BB1")
    axes[0].bar(
        x + width,
        avg["mean_bootstrap_seconds"],
        width,
        label="Bootstrap",
        color="#31A354",
    )
    axes[0].set_yscale("log")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels([str(v) for v in avg.index])
    axes[0].set_xlabel("Regimes")
    axes[0].set_ylabel("Mean seconds per replication, log scale")
    axes[0].set_title("Runtime by estimator stage")
    axes[0].legend(frameon=False)
    axes[0].grid(True, axis="y", alpha=0.25)

    axes[1].plot(
        avg.index.astype(str),
        avg["convergence_rate"],
        color="#D95F02",
        marker="o",
        linewidth=2.5,
    )
    axes[1].set_ylim(0.95, 1.002)
    axes[1].set_xlabel("Regimes")
    axes[1].set_ylabel("MLE convergence rate")
    axes[1].set_title("Convergence remains high as R grows")
    axes[1].grid(True, alpha=0.25)
    fig.suptitle("Computational feasibility of the simulation estimator", fontweight="bold")
    _save_current_figure("figure_4_computation_tradeoffs")


def write_machine_summary(
    data: pd.DataFrame,
    performance: pd.DataFrame,
    coverage: pd.DataFrame,
    timing: pd.DataFrame,
    naive: pd.DataFrame | None = None,
) -> None:
    def to_markdown_table(frame: pd.DataFrame) -> str:
        cols = list(frame.columns)
        rows = ["| " + " | ".join(cols) + " |", "| " + " | ".join(["---"] * len(cols)) + " |"]
        for _, row in frame.iterrows():
            rows.append("| " + " | ".join(str(row[col]) for col in cols) + " |")
        return "\n".join(rows)

    slopes = performance[performance["param_type"].eq("Slope")]
    mle_slopes = slopes[slopes["estimator"].eq("MLE")]
    irls_slopes = slopes[slopes["estimator"].eq("IRLS")]
    naive_slopes = slopes[slopes["estimator"].eq("Naive")]
    hardest = coverage.sort_values("bootstrap_coverage").head(5)
    best = coverage.sort_values("bootstrap_coverage", ascending=False).head(5)

    lines = [
        "# Simulation Evidence Summary",
        "",
        f"Generated from {len(data):,} parameter-replication rows.",
        f"Full factorial cells: {data.groupby(['R', 'rho', 'weight']).ngroups}.",
        f"Replications per cell: {data.groupby(['R', 'rho', 'weight'])['rep'].nunique().min()}.",
        "",
        "## Headline Metrics",
        "",
        f"Mean MLE slope RMSE across all cells: {mle_slopes['rmse'].mean():.3f}.",
        f"Mean IRLS slope RMSE across all cells: {irls_slopes['rmse'].mean():.3f}.",
    ]
    if len(naive_slopes) > 0:
        lines.append(
            f"Mean naive slope RMSE across all cells: {naive_slopes['rmse'].mean():.3f}."
        )
        lines.append(
            f"Mean naive slope absolute bias across all cells: {naive_slopes['mean_abs_bias'].mean():.3f}."
        )
    lines.extend(
        [
            f"Mean analytical Wald slope coverage: {coverage['wald_coverage'].mean():.3f}.",
            f"Mean score wild bootstrap slope coverage: {coverage['bootstrap_coverage'].mean():.3f}.",
        ]
    )
    if "naive_coverage" in coverage.columns:
        lines.append(
            f"Mean naive slope coverage: {coverage['naive_coverage'].mean():.3f}."
        )
    lines.extend(
        [
            f"Mean MLE convergence rate: {timing['convergence_rate'].mean():.3f}.",
            "",
            "## Hardest Coverage Cells",
            "",
            to_markdown_table(
                hardest[["R", "rho", "weight", "wald_coverage", "bootstrap_coverage"]].round(3)
            ),
            "",
            "## Best Coverage Cells",
            "",
            to_markdown_table(
                best[["R", "rho", "weight", "wald_coverage", "bootstrap_coverage"]].round(3)
            ),
        ]
    )
    if naive is not None and len(naive) > 0 and "naive_coverage" in coverage.columns:
        compare = (
            coverage.groupby(["R", "classification_error"])
            .agg(
                mle_boot=("bootstrap_coverage", "mean"),
                naive=("naive_coverage", "mean"),
            )
            .reset_index()
            .round(3)
        )
        bias_compare = (
            slopes.groupby(["estimator", "R", "classification_error"])
            .agg(rmse=("rmse", "mean"), abs_bias=("mean_abs_bias", "mean"))
            .reset_index()
            .round(3)
        )
        lines.extend(
            [
                "",
                "## Naive Baseline vs Corrected MLE — Slope Coverage",
                "",
                to_markdown_table(compare),
                "",
                "## Naive Baseline vs Corrected MLE — Slope RMSE and |bias|",
                "",
                to_markdown_table(bias_compare),
            ]
        )
    (TABLE_DIR / "simulation_evidence_summary.md").write_text("\n".join(lines) + "\n")


def main() -> None:
    _ensure_dirs()
    data = load_results()
    naive = load_naive_results()
    long = estimator_long(data, naive)
    performance = summarize_performance(long)
    coverage = summarize_coverage(data, naive)
    timing = summarize_timing(data)

    save_table_outputs(data, performance, coverage, timing)
    plot_coverage_landscape(coverage)
    plot_rmse_frontier(performance)
    plot_bias_distributions(data)
    plot_computation_tradeoffs(timing)
    plot_naive_vs_corrected(performance, coverage)
    write_machine_summary(data, performance, coverage, timing, naive)

    print(f"Loaded rows: {len(data):,}")
    if naive is not None:
        print(f"Loaded naive baseline rows: {len(naive):,}")
    print(f"Wrote figures to {FIGURE_DIR}")
    print(f"Wrote tables to {TABLE_DIR}")


if __name__ == "__main__":
    main()
