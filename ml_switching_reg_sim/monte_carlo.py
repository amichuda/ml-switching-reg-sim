from multiprocessing import Pool
from typing import Any
from .data_creation import UberDatasetCreator, extract_estimator_inputs
from ml_switching_reg.mle import (
    DriverSpecificProbUberMLE,
    get_mle_betas,
    get_mle_sigmas,
)
from .regression import uber_regression, get_reg_effect
from sklearn.model_selection import ParameterGrid
import statsmodels.api as sm

from multiprocessing import Pool
from functools import partial
import pandas as pd
import matplotlib.pyplot as plt

from dataclasses import make_dataclass, field, fields
import numpy as np
from scipy.special import logit

from tqdm import tqdm

from pathlib import Path
import re
import ast

import scipy.stats as ss
# A list of post simulation stats to add to the


class MonteCarlo:
    def __init__(self, drivers=275, time_periods=10, regimes=2, seed=1):
        """Creates different instances of data for estimation"""

        self.drivers = drivers
        self.time_periods = time_periods
        self.regimes = regimes
        self.seed = seed

        uber_dataset_kwds = {
            "seed": self.seed,
            "time_periods": self.time_periods,
            "regimes": self.regimes,
            "drivers": self.drivers,
        }

        self.u = UberDatasetCreator(**uber_dataset_kwds)

    def create_data_instances(self, N, n_jobs=5, **kwargs):
        """Creates `N` instances of data based on parameters
        from UberDatasetCreation

        This basically just takes range(N) and uses it as a seed

        Args:
            N_dist (int): Number of datasets to create, **within** a particular parameter choice
            uber_dataset_kwds (dict): keywords to pass the UberDatasetConstructor
            kwargs: arguments passed to `construct`
        """

        construct_kwds = {
            "y_sd": None,
            "drought_mean": None,
            "drought_cov": None,
            "beta0": [1, 2],
            "beta1": [-1, -2],
            "y_name": "y",
            "weight": 0.1,
            "reg_ready": False,
            "output_true_beta": True,
            "output_sigma": True,
        }
        construct_kwds.update(kwargs)

        # Create N instances of the data
        partial_construct = partial(self.u.construct, **construct_kwds)

        p = Pool(n_jobs)
        with p:
            results = p.map(partial_construct, range(N))

        df_list = [f[0] for f in results]
        weight_matrices = [f[1] for f in results]

        true_betas = [f[2] for f in results]

        true_sigmas = [f[3] for f in results]

        # Check that weight_matrices is the same across all

        return df_list, weight_matrices, true_betas, true_sigmas

    def change_param(
        self,
        n_jobs_within,
        N_within,
        param_dict,
        construct_dict=None,
        save=False,
        overwrite=False,
    ):
        """Creates instances of `create_data_instances` on
        different parameter changes based on `param_change_dict`

        Args:
            param_dict (dict): dictionary of parameters to pass to
            `UberDatasetCreator.construct`

        """

        param_dataclass_list = []

        param_dataclass = make_dataclass(
            "param_dataclass",
            list(param_dict.keys())
            + [
                ("df_list", list, field(repr=False)),
                ("weight_matrices", Any, field(repr=False)),
                ("uber_data_creator", Any, field(repr=False, default=None)),
                ("true_betas", Any, field(repr=False, default=None)),
                ("true_sigmas", Any, field(repr=False, default=None)),
                ("sim_results", Any, field(repr=False, default=None)),
            ],
            repr=True,
        )

        if construct_dict is None:
            construct_dict = {
                "y_sd": None,
                "drought_mean": None,
                "drought_cov": None,
                "beta0": [1, 2],
                "beta1": [-1, -2],
                "y_name": "y",
                "weight": 0.1,
                "reg_ready": False,
                "output_true_beta": True,
                "output_sigma": True,
            }
        else:
            construct_dict = construct_dict.copy()
            construct_dict.setdefault("output_true_beta", True)
            construct_dict.setdefault("output_sigma", True)

        for p_d in ParameterGrid(param_dict):
            print(f"Running {N_within} replicates for {p_d}")

            construct_dict.update(p_d)

            df_list, weight_matrices, true_betas, true_sigmas = (
                self.create_data_instances(
                    N=N_within, n_jobs=n_jobs_within, **construct_dict
                )
            )

            p_d_class = param_dataclass(
                **p_d,
                df_list=df_list,
                weight_matrices=weight_matrices,
                uber_data_creator=self.u,
                true_betas=true_betas,
                true_sigmas=true_sigmas,
            )

            param_dataclass_list.append(p_d_class)

        if save:
            # Creates a folder and saves the param_data_list, and weight matrix
            for p in param_dataclass_list:
                save_folder = Path(
                    f"UgandaUber/Estimation/modules/saved_models/param_dataclasses/{p}"
                )
                save_folder.mkdir(exist_ok=overwrite)

                # First save data to csv
                [
                    df.to_csv(save_folder / f"df_{i}.csv")
                    for i, df in enumerate(p.df_list)
                ]

                # Save weight matrix
                for i, wm in enumerate(p.weight_matrices):
                    with open(save_folder / f"weight_matrix_{i}.npy", "wb") as file:
                        np.save(file, wm)

        return param_dataclass_list

    def drought_cols(self, data):

        return data.columns[data.columns.str.fullmatch(r"drought_\d+")].tolist()

    def classifier_cols(self, data):

        return data.columns[
            data.columns.str.fullmatch(r"misclass_regime_\d+")
        ].tolist()

    def reg_fit(self, data, endog_col="y"):

        classifier_cols = self.classifier_cols(data=data)
        drought_cols = self.drought_cols(data=data)

        mod = uber_regression(
            data=data,
            endog_col=endog_col,
            classifier_cols=classifier_cols,
            drought_cols=drought_cols,
        )

        res = mod.fit()

        bse_beta0 = res.bse.loc[lambda s: ~s.index.str.contains("drought_")].values
        bse_beta1 = res.bse.loc[
            lambda s: s.index.str.contains("misclass_regime_[0-9]:drought_")
        ].values
        bse_both = np.append(bse_beta0, bse_beta1)

        return get_reg_effect(res), [np.nan] * len(drought_cols), bse_both

    def mle_fit(
        self,
        data,
        beta_start,
        mw,
        endog_col="y",
        estimator=None,
        p_mat_start=None,
        panel_likelihood=True,
    ):

        y, X_list, classifier_pred, cm, driver_ids = extract_estimator_inputs(
            data, mw, self.regimes, lags=None
        )

        ids = driver_ids if panel_likelihood else None

        if estimator is None:
            us = DriverSpecificProbUberMLE.from_arrays(
                y=y,
                X_list=X_list,
                classifier_pred=classifier_pred,
                cm=cm,
                driver_ids=ids,
            )
        else:
            us = estimator.from_arrays(
                y=y,
                X_list=X_list,
                classifier_pred=classifier_pred,
                cm=cm,
                driver_ids=ids,
            )

        p = X_list[0].shape[1]
        expected_beta_count = self.regimes * p
        beta_start = np.asarray(beta_start, dtype=float)

        if beta_start.size == expected_beta_count:
            if p == 2:
                beta0_start = beta_start[: self.regimes]
                beta1_start = beta_start[self.regimes :]
                start_params = np.array(
                    [
                        coef
                        for r in range(self.regimes)
                        for coef in (beta0_start[r], beta1_start[r])
                    ],
                    dtype=float,
                )
            else:
                start_params = beta_start.copy()
            start_params = np.append(start_params, max(float(data[endog_col].std()), 1e-6))
        elif beta_start.size == expected_beta_count + 1:
            start_params = beta_start.copy()
        else:
            raise ValueError(
                f"beta_start has length {beta_start.size}, expected {expected_beta_count} betas or {expected_beta_count + 1} full start params"
            )

        res_smle, _ = us.fit(
            start_params=start_params,
            method="bfgs",
            disp=False,
            sigma_bound=data[endog_col].std(),
        )

        mle_retvals = getattr(res_smle, "mle_retvals", None)
        if isinstance(mle_retvals, dict):
            success = int(bool(mle_retvals.get("converged", False)))
        else:
            success = int(bool(getattr(mle_retvals, "converged", False)))

        bse_array = np.asarray(res_smle.bse, dtype=float)
        beta_bse_mat = bse_array[:-1].reshape(self.regimes, p)
        bse_beta0 = beta_bse_mat[:, 0]
        bse_beta1 = (
            beta_bse_mat[:, 1] if p > 1 else np.full(self.regimes, np.nan, dtype=float)
        )
        bse_mle_both = np.append(bse_beta0, bse_beta1)

        bse_sigma = float(np.abs(bse_array[-1])) if len(bse_array) else np.nan
        bse_mle_sigmas = np.full(self.regimes, bse_sigma)

        return (
            get_mle_betas(res_smle, regimes=self.regimes),
            get_mle_sigmas(res_smle, regimes=self.regimes),
            success,
            res_smle,
            bse_mle_both,
            bse_mle_sigmas,
        )

    def append_reg_mle_results(
        self, df, mw, return_estimator=False, estimator=None, p_mat_start=None
    ):

        (_, _, reg_both), fake_sigmas, bse_reg_both = self.reg_fit(df)

        (
            (_, _, mle_both),
            sigmas,
            success,
            estimator_obj,
            bse_mle_both,
            bse_mle_sigmas,
        ) = self.mle_fit(
            df, beta_start=reg_both, mw=mw, estimator=estimator, p_mat_start=p_mat_start
        )

        # bse_reg_part: [bse_beta0s(R), bse_beta1s(R), NaN_sigmas(R)] = 3R
        # bse_mle_part: [bse_beta0s(R), bse_beta1s(R), bse_sigmas(R)] = 3R
        bse_reg_part = np.concatenate([bse_reg_both, np.full(self.regimes, np.nan)])
        bse_mle_part = np.concatenate([bse_mle_both, bse_mle_sigmas])

        # flat layout: reg_est(3R), mle_est(3R), bse_reg(3R), bse_mle(3R)
        result = np.concatenate(
            [
                reg_both,
                fake_sigmas,
                mle_both,
                sigmas,
                bse_reg_part,
                bse_mle_part,
            ]
        )

        if return_estimator:
            return np.append(np.append(result, success), estimator_obj)

        return np.append(result, success)

    def simulate(
        self,
        dataclass_list,
        show_progress=True,
        n_jobs=5,
        directory=None,
        overwrite=False,
        estimator=None,
        p_mat_start=None,
        return_estimator=True,
    ):
        """A function that takes a list of param dataclasses and simulated reg and mle on it
        And output various statistics

        Args:
            dataclass_list (list of dataclasses)
            show_progress (bool): Whether to show a progress bar
            n_jobs (int): number of jobs for `create_data_instances within each parameter change
            fit (str,bool): Whether to fit both "reg" and "mle" (True) or one of them
        """

        # if isinstance(dataclass_list[0], )

        # Get parameter settings from dataclasses
        if show_progress:
            range_iterable = tqdm(dataclass_list)
        else:
            range_iterable = dataclass_list

        results_multiindex = pd.MultiIndex.from_product(
            [
                ["reg", "mle", "true"],
                [f"beta_0", "beta_1", "sigma"],
                [f"regime_{i}" for i in range(self.regimes)],
            ]
        )

        for dc in range_iterable:
            # Create empty multiindex dataframe
            results_df = pd.DataFrame(
                index=range(len(dc.df_list)), columns=results_multiindex
            ).assign(success=np.nan, estimator=np.nan)

            p = Pool(n_jobs)

            with p:
                pooled_sim_results = p.starmap(
                    self.append_reg_mle_results,
                    zip(
                        dc.df_list,
                        dc.weight_matrices,
                        [return_estimator] * len(dc.df_list),
                        [estimator] * len(dc.df_list),
                        [p_mat_start] * len(dc.df_list),
                    ),
                )

            true_beta_array = np.array(dc.true_betas).reshape(
                len(dc.df_list), 2 * self.regimes
            )

            true_sigma_array = np.array(dc.true_sigmas).reshape(
                len(dc.df_list), self.regimes
            )

            true_array = np.append(true_beta_array, true_sigma_array, axis=1)

            pooled_sim_results_array = np.array(pooled_sim_results)

            R = self.regimes
            # flat layout: reg_est(3R), mle_est(3R), bse_reg(3R), bse_mle(3R), success, estimator
            estimates_array = pooled_sim_results_array[:, : 6 * R]
            bse_reg_part = pooled_sim_results_array[:, 6 * R : 9 * R]  # (n, 3R)
            bse_mle_part = pooled_sim_results_array[:, 9 * R : 12 * R]  # (n, 3R)

            sim_results_with_true_beta = np.append(estimates_array, true_array, axis=1)

            estimators = pooled_sim_results_array[:, -1]

            results_df.loc[:, :] = np.append(
                np.append(
                    sim_results_with_true_beta,
                    pooled_sim_results_array[:, -2][:, np.newaxis],
                    axis=1,
                ),
                estimators[:, np.newaxis],
                axis=1,
            )

            # Compute per-rep coverage indicators: 1 if true param is in 95% CI, else 0
            for param_idx, param in enumerate(["beta_0", "beta_1"]):
                for r_idx in range(R):
                    regime = f"regime_{r_idx}"
                    flat_idx = param_idx * R + r_idx
                    true_vals = results_df[("true", param, regime)].values.astype(float)

                    mle_est = results_df[("mle", param, regime)].values.astype(float)
                    mle_bse = bse_mle_part[:, flat_idx].astype(float)
                    results_df[f"coverage_mle_{param}_{regime}"] = (
                        (true_vals >= mle_est - 1.96 * mle_bse)
                        & (true_vals <= mle_est + 1.96 * mle_bse)
                    ).astype(float)

                    reg_est = results_df[("reg", param, regime)].values.astype(float)
                    reg_bse = bse_reg_part[:, flat_idx].astype(float)
                    results_df[f"coverage_reg_{param}_{regime}"] = (
                        (true_vals >= reg_est - 1.96 * reg_bse)
                        & (true_vals <= reg_est + 1.96 * reg_bse)
                    ).astype(float)

            for r_idx in range(R):
                regime = f"regime_{r_idx}"
                true_sigma = results_df[("true", "sigma", regime)].values.astype(float)
                mle_sigma = results_df[("mle", "sigma", regime)].values.astype(float)
                mle_sigma_bse = bse_mle_part[:, 2 * R + r_idx].astype(float)
                results_df[f"coverage_mle_sigma_{regime}"] = (
                    (true_sigma >= mle_sigma - 1.96 * mle_sigma_bse)
                    & (true_sigma <= mle_sigma + 1.96 * mle_sigma_bse)
                ).astype(float)

            # Now get parameters and add them as columns
            change_params = [f.name for f in fields(dc) if f.repr]

            param_setting = [str(getattr(dc, f)) for f in change_params]

            dc.sim_results = results_df.assign(
                **{k: v for k, v in zip(change_params, param_setting)}
            )

            # dc.sim_results.assign(estimator_saved =lambda df: f"est_{df.index}.pickle")
            if directory is not None:
                root = Path(
                    "UgandaUber/Estimation/modules/saved_models/param_dataclasses"
                )

                save_path = root / directory

                save_path.mkdir(exist_ok=overwrite)

                dc.sim_results.to_pickle(save_path / f"sim_results_{dc}.pickle")

        return dataclass_list


class SimulationVisualizer:
    def __init__(self, path=None, regimes=2):
        """This class loads the data files from the
        simulations together and graphs them
        in various ways.

        """
        self.regimes = regimes

        if path is None:
            path = "UgandaUber/Estimation/modules/saved_models/param_dataclasses"

        self.data_path = Path(path)

    def _load_files(self, ranger, param_name):

        file_stubs = [
            f"sim_results_param_dataclass({param_name}={repr(s)}).csv" for s in ranger
        ]

        df_list = [
            pd.read_csv(self.data_path / f, header=[0, 1, 2]) for f in file_stubs
        ]

        return df_list

    def _concat_files(self, ranger, param_name):

        df_list = self._load_files(ranger=ranger, param_name=param_name)

        return pd.concat(df_list)

    def _apply_x_var(
        self,
        x,
        f=None,
        num_replace=1,
        regex=False,
        comma_repl=False,
        matrix_repl=False,
        newline_repl=False,
        str_replace=True,
    ):

        if str_replace:
            if regex:
                within_list_sub = re.sub(r"(?<=[\d.-])\s+(?=[\d-])", ",", x)
                if matrix_repl:
                    within_list_sub = re.sub(r"\]\s", "],", within_list_sub)

                if newline_repl:
                    within_list_sub = within_list_sub.replace("\n", ",")

                x = ast.literal_eval(within_list_sub)
            elif comma_repl:
                x = ast.literal_eval(
                    x.replace(". ", ", ", num_replace).replace("\n", "", num_replace)
                )
            else:
                x = ast.literal_eval(
                    x.replace(" ", ",", num_replace).replace("\n", "", num_replace)
                )

        if f is not None:
            return f(x)

        return x

    def create_data(self, ranger, param_name, other_var=None, **kwargs):

        df = self._concat_files(ranger=ranger, param_name=param_name).assign(
            x_var=lambda df: (
                df[param_name].iloc[:, 0].apply(self._apply_x_var, **kwargs)
            )
        )

        if other_var is not None:
            df = df.assign(other_var=lambda df: df[other_var].iloc[:, 0])

        return df

    @staticmethod
    def calculate_statistics(data, only_success=False, other_var=False):
        """Turns concatenated dataframe into frame of statistics

        Args:
            data (pd.DataFrame)

        """

        if other_var:
            other_var = ["other_var"]
        else:
            other_var = []

        success_df = data[["success", "x_var"] + other_var]

        if only_success:
            data = data[(data["success"] == 1).values]

        # Separate coverage columns (flat string names) from MultiIndex estimate columns
        coverage_col_names = [
            c for c in data.columns if isinstance(c, str) and c.startswith("coverage_")
        ]
        drop_col = ("Unnamed: 0_level_0", "Unnamed: 0_level_1", "Unnamed: 0_level_2")
        estimate_data = data.drop(
            columns=[drop_col] + coverage_col_names, errors="ignore"
        )

        df = estimate_data.groupby(["x_var"] + other_var).agg(
            [
                "mean",
                "std",
                (
                    "ci_low",
                    lambda x: ss.t.interval(0.95, len(x) - 1, x.mean(), x.std())[0],
                ),
                (
                    "ci_high",
                    lambda x: ss.t.interval(0.95, len(x) - 1, x.mean(), x.std())[1],
                ),
            ]
        )

        success_df = success_df.groupby(["x_var"] + other_var).mean()

        if coverage_col_names:
            coverage_df = (
                data[coverage_col_names + ["x_var"] + other_var]
                .groupby(["x_var"] + other_var)
                .mean()
            )
        else:
            coverage_df = pd.DataFrame()

        return df, success_df, coverage_df

    def plot(
        self,
        stats_data,
        to_plot="beta_1",
        level_subset=None,
        ax=None,
        fig=None,
        hide_true=False,
        xlabel=None,
    ):
        """The plotter function for the simulations results.

        Args:
            stats (pd.DataFrame): The statistics dataframe to be plotted
            regimes (int, optional): The number of regimes. Defaults to 2.
            to_plot (str, optional): which parameter to plot. Defaults to 'beta_1'.
            level_subset (): A tuple of the level in the
            row index and how to subset it. Defaults to None.

        Returns:
            [type]: [description]
        """
        subset_title = ""
        other_var_label = ""

        stats, success_df, *_ = stats_data

        if isinstance(stats.index, pd.MultiIndex) and level_subset is None:
            raise Exception("Multindex in the data, but no subset was given.")

        if level_subset is not None:
            # Get index of what we want
            new_index = stats.index[
                stats.index.get_level_values(level_subset[0]) == level_subset[1]
            ]

            stats = stats.loc[new_index, :].reset_index(level=level_subset[0])

            success_df = success_df.loc[new_index, :].reset_index(level=level_subset[0])

            other_var_label = f", Misclass: {level_subset[1]}"

            # subset_title = f"Other Var at {level_subset[1]}"
        ax_none = False
        if ax is None:
            ax_none = True

            fig, ax = plt.subplots(
                3,
                self.regimes,
                figsize=(12, 10),
                gridspec_kw={"height_ratios": [3, 3, 1]},
                sharey="row",
            )

        def get_subset(
            regime, data=stats, model="reg", agg="mean", hide_true=hide_true
        ):

            if hide_true:
                plotting_slice = [model]
            else:
                plotting_slice = [model, "true"]

            return data.loc[:, (plotting_slice, [to_plot], [regime], [agg])]

        for r in range(self.regimes):
            labels_reg = ["Regression" + other_var_label]
            labels_mle = ["MLE" + other_var_label]

            if not hide_true:
                labels_reg.append("True")
                labels_mle.append("True")

            get_subset(f"regime_{r}").plot(
                ax=ax[0, r], title=f"{to_plot}, Regime {r}, OLS", legend=None
            )

            get_subset(f"regime_{r}", model="mle").plot(
                ax=ax[1, r], title=f"{to_plot}, Regime {r}, MLE", legend=None
            )

            ax[0, r].legend(labels_reg)
            ax[1, r].legend(labels_mle)

            # if not ax_none:
            #     ax[0,r].legend(ax[0,r].get_legend().texts + labels_reg)
            #     ax[1,r].legend(ax[1,r].get_legend().texts + labels_mle)

            # else:

            (
                success_df["success"].plot(
                    ax=ax[2, r], title="Successful Convergence", legend=None
                )
            )

            ax[0, r].fill_between(
                get_subset(f"regime_{r}", agg="ci_low").index,
                get_subset(f"regime_{r}", agg="ci_low")["reg"].values.flatten(),
                get_subset(f"regime_{r}", agg="ci_high")["reg"].values.flatten(),
                alpha=0.3,
                label=None,
            )

            ax[1, r].fill_between(
                get_subset(f"regime_{r}", model="mle", agg="ci_low")["mle"].index,
                get_subset(f"regime_{r}", model="mle", agg="ci_low")[
                    "mle"
                ].values.flatten(),
                get_subset(f"regime_{r}", model="mle", agg="ci_high")[
                    "mle"
                ].values.flatten(),
                alpha=0.3,
                label=None,
            )

        for r in range(self.regimes):
            ax[2, r].set_ylim([0, 1])

        if xlabel is not None:
            for a in ax.flatten():
                a.set_xlabel(xlabel)

        fig.suptitle(f"Plots of {to_plot} with " + subset_title)

        plt.tight_layout()


class DirectorySimulationVisualizer(SimulationVisualizer):
    def _load_files(self, directory):

        data_location = self.data_path / directory

        return [pd.read_csv(f, header=[0, 1, 2]) for f in data_location.glob("*.csv")]

    def _concat_files(self, directory):

        df_list = self._load_files(directory=directory)

        return pd.concat(df_list)

    def create_data(self, directory, param_name, other_var=None, **kwargs):

        df = self._concat_files(directory=directory).assign(
            x_var=lambda df: (
                df[param_name].iloc[:, 0].apply(self._apply_x_var, **kwargs)
            )
        )

        if other_var is not None:
            df = df.assign(other_var=lambda df: df[other_var].iloc[:, 0])

        return df
