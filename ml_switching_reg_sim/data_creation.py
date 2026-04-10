
import pandas as pd
import numpy as np
from functools import reduce
from .utils import lagged_drought_df

from mockseries.trend import LinearTrend, Switch
from mockseries.seasonality import SinusoidalSeasonality
from mockseries.noise import RedNoise
from mockseries.transition import LinearTransition
from datetime import timedelta


# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------

def extract_estimator_inputs(df, mw, regimes, lags=None):
    """Extract the unified estimator interface arrays from dataset output.

    Both ``MLSwitchingRegIRLS`` and ``DriverSpecificProbUberMLE.from_arrays``
    accept the four arrays returned here.

    Parameters
    ----------
    df : DataFrame
        Output of ``UberDatasetCreatorHet.construct()``.
    mw : (R, R) ndarray
        Row-stochastic confusion matrix: ``mw[true, predicted] = P(predicted | true)``.
    regimes : int
    lags : list of int or None
        If None uses contemporaneous ``drought_r``; otherwise ``lagged_{lag}_drought_r``.

    Returns
    -------
    y : (N,)
    X_list : list of R arrays each (N, p)
    classifier_pred : (N, R)
    cm : (R, R)  column-normalised: ``cm[k, j] = P(true=k | predicted=j)``
    """
    N = len(df)
    y = df["y"].values.astype(float)
    if lags is None:
        X_list = [
            np.column_stack([np.ones(N), df[f"drought_{r}"].values.astype(float)])
            for r in range(regimes)
        ]
    else:
        X_list = [
            np.column_stack(
                [np.ones(N)]
                + [df[f"lagged_{lag}_drought_{r}"].values.astype(float) for lag in lags]
            )
            for r in range(regimes)
        ]
    classifier_pred = df[
        [f"misclass_regime_{r}" for r in range(regimes)]
    ].values.astype(float)
    cm = mw / mw.sum(axis=0, keepdims=True)   # column-normalise → P(true | pred)
    return y, X_list, classifier_pred, cm


def demean_arrays(y, X_list, entity_ids=None, time_ids=None):
    """Within-transform ``y`` and each array in ``X_list`` for fixed effects.

    The intercept column (column 0) is dropped from each X array since it is
    collinear with the fixed effects.

    Parameters
    ----------
    y : (N,) array
    X_list : list of R arrays each (N, p) — column 0 must be the intercept
    entity_ids : (N,) array of entity labels, or None
    time_ids   : (N,) array of time labels, or None

    Returns
    -------
    y_demeaned : (N,)
    X_list_demeaned : list of R arrays each (N, p-1)
    """
    y_s = pd.Series(y)
    X_frames = [pd.DataFrame(X[:, 1:]) for X in X_list]

    if entity_ids is not None:
        g = pd.Series(entity_ids)
        y_s = y_s - y_s.groupby(g).transform('mean')
        X_frames = [X.subtract(X.groupby(g).transform('mean')) for X in X_frames]
    if time_ids is not None:
        g = pd.Series(time_ids)
        y_s = y_s - y_s.groupby(g).transform('mean')
        X_frames = [X.subtract(X.groupby(g).transform('mean')) for X in X_frames]

    return y_s.values, [X.values for X in X_frames]


# ---------------------------------------------------------------------------
# Misclassification helper
# ---------------------------------------------------------------------------

class MisclassificationCreator:
    """Per-driver soft misclassification probability vectors via ``noisify_matrix``.

    ``noisify_matrix(extent, index)`` returns a length-R probability vector for a
    driver whose true regime is ``index``.

    - ``extent=0``  → perfect classifier (one-hot)
    - ``extent=1``  → fully uninformative (each entry ≈ 1/R)

    The mapping from ``extent`` to off-diagonal mass is
    ``off_diag = extent * (1 - 1/R)``, so P(correct) = ``1 - extent*(1-1/R)``.
    """

    def __init__(self, regimes, seed=None):
        if seed is None:
            seed = 1234
        self.regimes = regimes
        self.random = np.random.default_rng(seed=seed)

    def _index_to_misclassification(self, extent):
        return extent * (1.0 - 1.0 / self.regimes)

    def noisify_matrix(self, extent, index):
        extent = self._index_to_misclassification(extent)

        if self.regimes == 2:
            return np.insert(np.array([extent]), index, 1.0 - extent)

        def _recursive_fill_in(ext):
            if ext.shape == ():
                ext_diff = float(ext)
            else:
                ext_diff = reduce(lambda x, y: x - y, ext)
            if ext.shape != () and ext.shape[0] == self.regimes - 1:
                return np.append(ext, ext_diff)
            return _recursive_fill_in(np.append(ext, np.random.uniform(0, ext_diff)))

        new_vec = np.delete(_recursive_fill_in(np.array(extent)), 0)
        self.random.shuffle(new_vec)
        return np.insert(new_vec, index, 1.0 - extent)


# ---------------------------------------------------------------------------
# Dataset creator
# ---------------------------------------------------------------------------

class UberDatasetCreatorHet:
    """Synthetic panel dataset creator for switching-regression simulations.

    Generates a ``(driver, time)`` MultiIndex DataFrame with:
    - regime-specific drought covariates
    - true one-hot regime dummies
    - soft misclassification predictions (``misclass_regime_*``)
    - a consistent row-stochastic confusion matrix ``mw``
    - an outcome ``y`` generated from the true switching regression DGP

    Two misclassification modes are supported via ``construct(classifier_mode=...)``:

    ``'noisify'`` (default)
        Deterministic per-driver soft probabilities from
        ``MisclassificationCreator.noisify_matrix``, controlled by ``weight``.

    ``'ml'``
        An XGBoost classifier trained on driver-level summaries of ``y`` and
        drought variables, controlled by ``noise_scale``.
    """

    def __init__(self,
                 drivers=275,
                 regimes=4,
                 time_periods=10,
                 seed=None,
                 lags=None):

        self.regimes = regimes
        self.drivers = drivers
        self.time_periods = time_periods
        self.N = drivers * regimes * time_periods
        self.seed = seed
        self.lags = lags
        self.random = np.random.default_rng(seed=seed)
        self.dates = pd.date_range(start="2016-01-01",
                                   periods=self.time_periods,
                                   freq='ME')

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _create_drought_index(self,
                              mean=None,
                              cov=None,
                              mock=False,
                              mock_dict=None):
        """Create a (time,) × R drought DataFrame, optionally via mockseries."""

        if mock:
            if mock_dict is None:
                raise TypeError("Need mock_dict in order to use mockseries")
            bases = mock_dict.get('bases')
            amplitudes = mock_dict.get('amplitudes')
            offsets = mock_dict.get('offsets')
            shock_times = mock_dict.get('shock_times')
            if len(bases) != len(amplitudes) or len(bases) != self.regimes:
                raise ValueError(
                    "mock_dict parameters must be equal in length and equal to number of regimes."
                )
            alpha_r = dict(zip(range(self.regimes),
                               self.random.normal(0, 1, size=self.regimes)))
            df = pd.DataFrame(index=pd.DatetimeIndex(self.dates, name='time'),
                              columns=[f'drought_{i}' for i in range(self.regimes)])
            for shock, base, amplitude, offset, r, ar in zip(
                    shock_times, bases, amplitudes, offsets, range(self.regimes), alpha_r.values()):
                trend = LinearTrend(coefficient=0, time_unit=timedelta(days=30), flat_base=base)
                seasonality = SinusoidalSeasonality(
                    amplitude=amplitude, period=timedelta(days=90), offset=offset)
                noise_ts = RedNoise(mean=0, std=1, correlation=.5)
                trans = LinearTransition(transition_window=timedelta(days=30),
                                         stop_window=timedelta(days=5))
                switch = Switch(start_time=self.dates[shock[0]], base_value=0,
                                switch_value=-5, stop_time=self.dates[shock[1]],
                                transition=trans)
                df[f"drought_{r}"] = (trend + seasonality + noise_ts + switch).generate(
                    self.dates.tolist()) + ar
            return df

        if mean is None:
            mean = [-1] * self.regimes
        if cov is None:
            cov = [1] * self.regimes
        if hasattr(cov[0], "__len__"):
            df = pd.DataFrame(
                self.random.multivariate_normal(mean, cov, size=self.time_periods,
                                                check_valid='raise', method='eigh'),
                columns=[f"drought_{i}" for i in range(self.regimes)]
            )
        else:
            df = pd.DataFrame(
                {f'drought_{i}': self.random.normal(m, sd, size=self.time_periods)
                 for i, (m, sd) in enumerate(zip(mean, cov))}
            )
        df.index.names = ['time']
        return df

    def _create_drought_index_with_driver(self, mean=None, cov=None, mock=False, mock_dict=None):
        """Broadcast the time-level drought DataFrame to all drivers."""
        drought_df = self._create_drought_index(mean=mean, cov=cov, mock=mock, mock_dict=mock_dict)
        drought_cols = [f"drought_{i}" for i in range(self.regimes)]

        driver_time_df = (
            pd.concat([drought_df.assign(driver=i) for i in range(self.drivers)])
            .set_index('driver', append=True)
        )
        driver_time_df.index.names = ['time', 'driver']

        return (
            driver_time_df
            .reorder_levels(['driver', 'time'])
            .reset_index()
            .pipe(lagged_drought_df, drought_cols,
                  shift=self.lags, groupby_index='driver', date_col='time',
                  dropna=(self.lags is not None))
            .set_index(['driver', 'time'])
            .reorder_levels(['driver', 'time'])
        )

    def _create_driver_index(self):
        return pd.DataFrame({'driver': list(range(self.drivers))})

    def _create_regime_index(self, p=None):
        return pd.DataFrame({'regime': self.random.choice(
            list(range(self.regimes)), size=self.drivers, p=p)})

    def _create_y(self, data, beta1, beta0=None, name='y', sd=None):
        """Generate per-regime outcome columns and attach to *data*."""
        if sd is None:
            sd = [1] * self.regimes
        if isinstance(beta0, (int, float)):
            beta0 = [beta0] * self.regimes
        if isinstance(beta1, (int, float)):
            beta1 = [beta1] * self.regimes

        def y_lambda(i):
            if self.lags is not None:
                b1 = np.asarray(beta1)
                return lambda df: (
                    beta0[i]
                    + df.filter(regex=rf"lagged_\d+_drought_{i}").values @ b1
                    + self.random.normal(0, sd[i], len(df))
                )
            else:
                return lambda df: (
                    beta0[i] + beta1[i] * df[f'drought_{i}']
                    + self.random.normal(0, sd[i], len(df))
                )

        return data.assign(**{name + f'_{i}': y_lambda(i) for i in range(self.regimes)})

    def _build_regime_dummies_noisify(self, regime, weight):
        """Build the misclassification columns using ``noisify_matrix``.

        Returns
        -------
        regime_dummies : DataFrame  (per-driver, indexed 0..N_drivers-1)
            Columns: ``regime_r``, ``misclass_regime_r``, ``max_misclass_regime``.
        mw : (R, R) ndarray  row-stochastic, consistent with ``misclass_regime_*``.
        """
        m = MisclassificationCreator(self.regimes)
        regime_with_mc = regime.assign(
            misclass_regime=lambda df: df['regime'].apply(
                lambda x: m.noisify_matrix(extent=weight, index=x))
        )
        rd = (
            pd.concat([
                pd.get_dummies(regime_with_mc, columns=['regime']),
                regime_with_mc.apply(lambda x: x['misclass_regime'],
                                     result_type='expand', axis=1),
            ], axis=1)
            .rename({r: f'misclass_regime_{r}' for r in range(self.regimes)}, axis=1)
            .assign(max_misclass_regime=lambda df:
                    df['misclass_regime'].apply(lambda x: x.argmax()))
            .drop(['misclass_regime'], axis=1)
        )
        # mw[r, :] = E[predicted proba | true_regime == r]  ← matches classifier_pred exactly
        mc_cols = [f'misclass_regime_{r}' for r in range(self.regimes)]
        mw = np.array([
            rd.loc[regime['regime'] == r, mc_cols].mean().values
            for r in range(self.regimes)
        ])
        return rd, mw

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def construct(self,
                  seed=None,
                  y_sd=None,
                  drought_mean=None,
                  drought_cov=None,
                  beta0=12,
                  beta1=2,
                  y_name='y',
                  weight=0.9,
                  reg_ready=False,
                  output_true_beta=False,
                  output_sigma=False,
                  driver_fe=False,
                  time_fe=False,
                  month_year_fe=False,
                  mock=False,
                  mock_dict=None,
                  classifier_mode='noisify',
                  noise_scale=1.0,
                  ):
        """Construct a synthetic switching-regression panel dataset.

        Parameters
        ----------
        seed : int or None
        y_sd : list of float  per-regime residual std
        drought_mean, drought_cov : passed to ``_create_drought_index``
        beta0, beta1 : per-regime intercepts and slopes (lists or scalars)
        y_name : str
        weight : float in (0, 1)
            Misclassification severity for ``classifier_mode='noisify'``.
            0 = perfect, 1 = fully uninformative (P=1/R).
        reg_ready : bool  drop regime + y_r columns before returning
        output_true_beta : bool
        output_sigma : bool
        driver_fe : bool  add random driver fixed effects to y
        time_fe : bool  add random time fixed effects to y
        month_year_fe : bool  add random month/year fixed effects to y
        mock, mock_dict : mockseries options
        classifier_mode : {'noisify', 'ml'}
            ``'noisify'`` uses deterministic ``MisclassificationCreator.noisify_matrix``
            controlled by ``weight``.
            ``'ml'`` trains an XGBoost classifier on driver-level features of y
            and drought; severity controlled by ``noise_scale``.
        noise_scale : float
            Noise added to XGBoost features when ``classifier_mode='ml'``.
            Higher → worse classifier → more misclassification.

        Returns
        -------
        df : MultiIndex (driver, time) DataFrame
        mw : (R, R) ndarray  row-stochastic confusion matrix consistent with
             the ``misclass_regime_*`` columns in df.
        Optionally appends [beta0, beta1] and y_sd to the return tuple.
        """
        if seed is not None:
            self.random = np.random.default_rng(seed=seed)

        drought = self._create_drought_index_with_driver(
            mean=drought_mean, cov=drought_cov, mock=mock, mock_dict=mock_dict)
        regime = pd.concat([self._create_driver_index(), self._create_regime_index()], axis=1)

        if classifier_mode == 'noisify':
            regime_dummies, mw = self._build_regime_dummies_noisify(regime, weight)

            df = (
                drought
                .join(regime_dummies.set_index('driver'))
                .join(regime.set_index('driver'))            # adds integer 'regime' column
                .pipe(self._create_y, beta0=beta0, beta1=beta1, name=y_name, sd=y_sd)
                .pipe(pd.get_dummies, columns=['max_misclass_regime'])
            )
            for r in range(self.regimes):
                df.loc[lambda df: df['regime'] == r, y_name] = df[f'{y_name}_{r}']

        elif classifier_mode == 'ml':
            from .classifier import XGBoostRegimeClassifier

            # Generate y first so the classifier can use it as a signal feature
            partial_df = (
                drought
                .join(regime.set_index('driver'))
                .pipe(self._create_y, beta0=beta0, beta1=beta1, name=y_name, sd=y_sd)
            )
            for r in range(self.regimes):
                partial_df.loc[lambda df: df['regime'] == r, y_name] = partial_df[f'{y_name}_{r}']

            true_regimes = regime.set_index('driver')['regime']
            clf_seed = seed if seed is not None else (self.seed if self.seed is not None else 42)
            # fit_predict returns (N_drivers, R) — driver-level, which is what regime_dummies needs
            driver_proba, mw = XGBoostRegimeClassifier(
                noise_scale=noise_scale, seed=clf_seed
            ).fit_predict(self.drivers, true_regimes)

            regime_dummies = pd.get_dummies(regime, columns=['regime']).copy()
            for r in range(self.regimes):
                regime_dummies[f'misclass_regime_{r}'] = driver_proba[:, r]
            regime_dummies['max_misclass_regime'] = np.argmax(driver_proba, axis=1)

            df = (
                partial_df
                .join(regime_dummies.set_index('driver'))
                .pipe(pd.get_dummies, columns=['max_misclass_regime'])
            )
        else:
            raise ValueError(
                f"classifier_mode must be 'noisify' or 'ml', got {classifier_mode!r}"
            )

        # Fixed effects
        driver_ids = df.index.get_level_values('driver').unique().tolist()
        time_ids = df.index.get_level_values('time').unique().tolist()
        time_level = pd.DatetimeIndex(df.index.get_level_values('time'))
        month_ids = time_level.month.unique().tolist()
        year_ids = time_level.year.unique().tolist()

        alpha_i = ({k: v for k, v in zip(driver_ids, self.random.normal(0, 1, size=len(driver_ids)))}
                   if driver_fe else {k: 0 for k in driver_ids})
        alpha_t = ({k: v for k, v in zip(time_ids, self.random.normal(0, 1, size=len(time_ids)))}
                   if time_fe else {k: 0 for k in time_ids})
        alpha_m = ({k: v for k, v in zip(month_ids, self.random.normal(0, 1, size=len(month_ids)))}
                   if month_year_fe else {k: 0 for k in month_ids})
        alpha_y = ({k: v for k, v in zip(year_ids, self.random.normal(0, 1, size=len(year_ids)))}
                   if month_year_fe else {k: 0 for k in year_ids})

        df = df.assign(
            alpha_i=df.index.get_level_values('driver').map(alpha_i),
            alpha_t=df.index.get_level_values('time').map(alpha_t),
            alpha_m=pd.DatetimeIndex(df.index.get_level_values('time')).month.map(alpha_m),
            alpha_y=pd.DatetimeIndex(df.index.get_level_values('time')).year.map(alpha_y),
            y=lambda df: df['y'] + df['alpha_i'] + df['alpha_t'] + df['alpha_m'] + df['alpha_y'],
        )

        if reg_ready:
            regime_cols = df.columns[df.columns.str.contains("^regime")].tolist()
            y_cols = df.columns[df.columns.str.contains(f"{y_name}_")].tolist()
            df = df.drop(regime_cols + y_cols + ['regime'], axis=1)

        if output_true_beta:
            if output_sigma:
                return df, mw, [beta0, beta1], y_sd
            return df, mw, [beta0, beta1]

        return df, mw


# Backward-compatible alias used by monte_carlo.py
UberDatasetCreator = UberDatasetCreatorHet
