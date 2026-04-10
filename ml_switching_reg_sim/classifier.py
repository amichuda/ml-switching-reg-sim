"""XGBoost-based regime classifier with synthetic TF-IDF-like name features.

In Michuda (2023) the ML classifier predicts driver regimes from surnames
decomposed into TF-IDF character n-gram features.  This module reproduces
that structure synthetically so we can control misclassification severity.

**Synthetic feature DGP**

Each driver ``i`` with true regime ``r`` receives a ``n_features``-dimensional
feature vector::

    x_i = mu[r] + noise_scale * N(0, I_{n_features})

where ``mu[r]`` are regime-specific mean vectors drawn once at construction
time.  This mirrors the TF-IDF setting: regime-correlated name patterns are
observed through a noisy channel.

- ``noise_scale = 0``  → features cluster perfectly by regime  → near-perfect classifier
- ``noise_scale → ∞``  → features dominated by noise           → near-random (P ≈ 1/R)

The classifier uses k-fold out-of-fold (OOF) predictions so every driver
receives a prediction without data leakage.  The returned ``mw`` is computed
from those OOF predictions and is therefore exactly consistent with the
returned ``classifier_pred``.
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold


class XGBoostRegimeClassifier:
    """Simulate an ML regime classifier with controllable misclassification.

    Parameters
    ----------
    noise_scale : float
        Standard deviation of feature noise.  At 0 the classifier is
        near-perfect; larger values degrade it toward random guessing.
    n_features : int
        Dimensionality of the synthetic TF-IDF feature space.
    seed : int
    n_folds : int
        Number of CV folds for OOF predictions.
    xgb_params : dict or None
        Extra keyword arguments forwarded to ``xgboost.XGBClassifier``.
    """

    _default_xgb_params = dict(
        n_estimators=100,
        max_depth=3,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric='mlogloss',
        verbosity=0,
    )

    def __init__(self, noise_scale=1.0, n_features=50, seed=42, n_folds=5,
                 xgb_params=None):
        self.noise_scale = noise_scale
        self.n_features = n_features
        self.seed = seed
        self.n_folds = n_folds
        self.xgb_params = {**self._default_xgb_params, **(xgb_params or {})}
        self.xgb_params['random_state'] = seed
        self.rng = np.random.default_rng(seed)

    def _make_features(self, true_regimes_array, R):
        """Draw synthetic TF-IDF-like features for each driver.

        Each regime ``r`` has a fixed random mean vector ``mu[r]`` in
        ``R^{n_features}``.  Driver features are::

            x_i = mu[regime_i] + noise_scale * N(0, I)

        The mean vectors are spaced apart by construction (drawn from a
        unit sphere scaled by ``sqrt(n_features)`` so that at ``noise_scale=1``
        the SNR is roughly 1).

        Returns
        -------
        X : (N_drivers, n_features) ndarray
        """
        # Regime means: fixed, well-separated — orthogonal basis scaled by sqrt(n_features)
        rng_means = np.random.default_rng(seed=0)  # fixed seed so means are reproducible
        mu = rng_means.normal(0, np.sqrt(self.n_features), size=(R, self.n_features))

        signal = mu[true_regimes_array]                                     # (N, n_features)
        noise = self.rng.normal(0, self.noise_scale, signal.shape)
        return signal + noise

    def fit_predict(self, n_drivers, true_regimes):
        """Generate synthetic features, train XGBoost, and return OOF predictions.

        Parameters
        ----------
        n_drivers : int
            Number of drivers (rows in the feature matrix).
        true_regimes : pd.Series indexed by *driver* (int labels 0..R-1)
            Must have exactly ``n_drivers`` entries.

        Returns
        -------
        classifier_pred : (N_obs, R) ndarray  — one row per **observation**
            Each driver's OOF probability is broadcast to all its time periods.
            Pass the original MultiIndex DataFrame as ``df`` to ``fit_predict_from_df``
            to get observation-level output directly.
        mw : (R, R) ndarray
            Row-stochastic confusion matrix consistent with ``classifier_pred``.
            ``mw[r, :] = mean(OOF proba | true_regime == r)``.

        Notes
        -----
        This method returns driver-level predictions (shape ``(n_drivers, R)``).
        Use ``fit_predict_from_df`` to broadcast to the full observation panel.
        """
        import xgboost as xgb

        y_labels = true_regimes.values.astype(int)
        R = len(np.unique(y_labels))

        X = self._make_features(y_labels, R)

        # ------------------------------------------------------------------
        # k-fold OOF predictions
        # ------------------------------------------------------------------
        oof_proba = np.zeros((n_drivers, R))
        kf = StratifiedKFold(n_splits=self.n_folds, shuffle=True, random_state=self.seed)

        for train_idx, val_idx in kf.split(X, y_labels):
            params = {**self.xgb_params}
            if R > 2:
                params['objective'] = 'multi:softprob'
                params['num_class'] = R
            else:
                params['objective'] = 'binary:logistic'
                params.pop('num_class', None)
            clf = xgb.XGBClassifier(**params)
            clf.fit(X[train_idx], y_labels[train_idx])
            proba = clf.predict_proba(X[val_idx])
            if proba.ndim == 1:
                proba = np.column_stack([1.0 - proba, proba])
            oof_proba[val_idx] = proba

        # mw[r, :] = E[predicted proba | true_regime == r]
        mw = np.array([
            oof_proba[y_labels == r].mean(axis=0) for r in range(R)
        ])

        return oof_proba, mw

    def fit_predict_from_df(self, df, true_regimes):
        """Convenience wrapper that broadcasts driver predictions to all observations.

        Parameters
        ----------
        df : MultiIndex (driver, time) DataFrame
        true_regimes : pd.Series indexed by *driver* (int labels 0..R-1)

        Returns
        -------
        classifier_pred : (N_obs, R) ndarray
        mw : (R, R) ndarray
        """
        driver_order = true_regimes.index.tolist()
        n_drivers = len(driver_order)
        driver_proba, mw = self.fit_predict(n_drivers, true_regimes)

        driver_to_row = {d: i for i, d in enumerate(driver_order)}
        obs_drivers = df.index.get_level_values('driver')
        classifier_pred = driver_proba[
            np.array([driver_to_row[d] for d in obs_drivers])
        ]
        return classifier_pred, mw
