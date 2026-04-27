# Plan: Observation-Level to Driver-Level Likelihood Fix

## Problem

The DGP assigns each **driver** $i$ to a fixed regime across all time periods. The correct likelihood marginalizes over regimes at the driver level:

$$\ell_{\text{driver}} = \sum_{i=1}^{N_{\text{drivers}}} \log\!\left[\sum_r w_i(r)\,\prod_{t=1}^{T_i} f_{it}^{(r)}\right]$$

Both estimators currently use the observation-level likelihood:

$$\ell_{\text{obs}} = \sum_{n=1}^{N} \log\!\left[\sum_r w_n(r)\,f_n^{(r)}\right]$$

These are **equivalent when T=1** (confirmed numerically in `theory_derivations.py`). For T>1, the observation-level form is a composite likelihood: consistent but inefficient, with incorrect standard errors that overstate information by treating within-driver observations as independent.

**Key insight:** the M-step (WLS for $\beta_r$, pooled variance for $\sigma^2$) does NOT change. Only the E-step posteriors (IRLS) and the likelihood function (MLE) change. The WLS equation $(X_r'W_rX_r)\hat\beta_r = X_r'W_ry$ remains valid — the only difference is that the weights $\tau_{nr}$ are now constant within each driver.

---

## Step 1: Data Pipeline — Return `driver_ids`

**File:** `ml_switching_reg_sim/data_creation.py`, function `extract_estimator_inputs` (line 18)

**Current return:** `(y, X_list, classifier_pred, cm)` — all arrays indexed by flat observation $n$, no driver identity.

**Change:** Add `driver_ids` to the return tuple.

```python
def extract_estimator_inputs(df, mw, regimes, lags=None):
    # ... existing code unchanged ...
    driver_ids = df.index.get_level_values('driver').values  # (N,) integer array
    return y, X_list, classifier_pred, cm, driver_ids
```

The `driver_ids` array is `(N,)` where `N = n_drivers × T`. Each entry is the integer driver index for that observation, extracted from the DataFrame's `(driver, time)` MultiIndex.

**Breaking change:** all call sites must be updated. Callers that don't need `driver_ids` can use `_, _, _, _, _ = extract_estimator_inputs(...)` or `*_, driver_ids = ...`.

**Call sites to update:**
- `examples/theory_derivations.py` — every cell that calls `extract_estimator_inputs`
- `ml_switching_reg_sim/monte_carlo.py` — the `simulate` and `mle_fit` methods
- Any notebooks or scripts that import `extract_estimator_inputs`

---

## Step 2: IRLS Constructor — Accept `driver_ids`

**File:** `ml_switching_reg/irls.py`, method `__init__` (line 37)

Add optional `driver_ids` parameter and precompute the driver structure:

```python
def __init__(self, y, X_list, classifier_pred, cm, driver_ids=None):
    # ... existing code unchanged ...
    self.driver_ids = driver_ids  # (N,) or None
    if driver_ids is not None:
        _ids = np.asarray(driver_ids)
        self._unique_drivers = np.unique(_ids)
        self._n_drivers = len(self._unique_drivers)
        # Map each observation to a contiguous driver index 0..D-1
        _map = {d: idx for idx, d in enumerate(self._unique_drivers)}
        self._obs_to_driver = np.array([_map[d] for d in _ids])
        # Index lookup: for each driver, which observation indices belong to it
        self._driver_indices = {
            d: np.where(_ids == d)[0] for d in self._unique_drivers
        }
    else:
        self._n_drivers = None
```

When `driver_ids is None`, the estimator falls back to observation-level behavior (backward compatible).

---

## Step 3: IRLS E-step — Driver-Level Posteriors

**File:** `ml_switching_reg/irls.py`, method `_estep` (line 49)

This is the core algorithmic change. Replace the per-observation posterior computation with a driver-level one.

### Current (observation-level):
```
τ_{nr} = w_n(r) · f_n^(r) / Σ_j w_n(j) · f_n^(j)     — one τ per observation
```

### New (driver-level):
```
τ_{ir} = w_i(r) · Π_t f_{it}^(r) / Σ_j w_i(j) · Π_t f_{it}^(j)     — one τ per driver, broadcast
```

### Algorithm:
1. Compute per-observation log-densities `log f_{it}^{(r)}` — same as before, but stay in log space
2. Sum log-densities within each driver: `log_prod_i(r) = Σ_{t ∈ driver(i)} log f_{it}^{(r)}`
3. Add log-weights: `log_unnorm_i(r) = log w_i(r) + log_prod_i(r)`
4. Normalize using log-sum-exp for numerical stability → `τ_i(r)`
5. Broadcast `τ_i` back to all observations of driver $i$

### Implementation:

```python
def _estep(self, beta, sigma2):
    sigma2 = max(float(sigma2), 1e-8)

    if self.driver_ids is None:
        # Observation-level (original behavior, unchanged)
        regimes = np.array([
            np.exp(-(self.y - self.X_list[r] @ beta[r]) ** 2 / (2 * sigma2))
            for r in range(self.n_regimes)
        ])
        unnorm = regimes * self.weighted_cm.T
        return unnorm / (unnorm.sum(axis=0) + 1e-300)

    # --- Driver-level E-step ---
    R = self.n_regimes
    D = self._n_drivers

    # Step 1: Per-observation log-densities (R, N)
    # Omit -0.5*log(2*pi*sigma2) since it cancels in the ratio
    log_f = np.array([
        -(self.y - self.X_list[r] @ beta[r]) ** 2 / (2 * sigma2)
        for r in range(R)
    ])  # (R, N)

    # Step 2: Sum log-densities within each driver → (R, D)
    log_prod = np.zeros((R, D))
    np.add.at(log_prod.T, self._obs_to_driver, log_f.T)

    # Step 3: Add log-weights → (R, D)
    # weighted_cm is (N, R); constant within driver, take first obs
    first_obs = np.array([self._driver_indices[d][0]
                          for d in self._unique_drivers])
    log_w = np.log(self.weighted_cm[first_obs, :].T + 1e-300)  # (R, D)

    unnorm_log = log_w + log_prod  # (R, D)

    # Step 4: Log-sum-exp normalization
    max_log = unnorm_log.max(axis=0, keepdims=True)  # (1, D)
    log_denom = max_log + np.log(
        np.exp(unnorm_log - max_log).sum(axis=0, keepdims=True) + 1e-300
    )
    tau_driver = np.exp(unnorm_log - log_denom)  # (R, D)

    # Step 5: Broadcast back to (R, N)
    tau_obs = tau_driver[:, self._obs_to_driver]  # (R, N)

    return tau_obs
```

### Why the M-step doesn't change:

The M-step solves $(X_r'W_rX_r)\beta_r = X_r'W_ry$ where $W_r = \text{diag}(\tau_{\cdot r})$. With the driver-level E-step, the only difference is that all observations of the same driver get the same weight $\tau_{ir}$. The WLS algebra is identical — `_beta_h` and `_sigma2_h` are unchanged.

---

## Step 4: MLE Constructor — Accept `driver_ids`

**File:** `ml_switching_reg/mle.py`, method `from_arrays` (around line 69)

Add `driver_ids=None` parameter. Store it and precompute driver structure:

```python
@classmethod
def from_arrays(cls, y, X_list, classifier_pred, cm, start_params=None,
                soft=True, driver_ids=None, **kwargs):
    # ... existing code ...
    obj._driver_ids = driver_ids
    if driver_ids is not None:
        _ids = np.asarray(driver_ids)
        _unique = np.unique(_ids)
        obj._unique_drivers = _unique
        obj._n_drivers = len(_unique)
        _map = {d: idx for idx, d in enumerate(_unique)}
        obj._obs_to_driver = np.array([_map[d] for d in _ids])
        obj._driver_indices = {d: np.where(_ids == d)[0] for d in _unique}

        # Precompute driver-level weights (D, R) — constant within driver
        _first_obs = np.array([obj._driver_indices[d][0] for d in _unique])
        obj._w_driver = obj._weighted_cm_arr[_first_obs, :]      # (D, R)
        obj._log_w_driver = np.log(obj._w_driver + 1e-300)       # (D, R)

        # Correct nobs for standard errors: D independent units, not N
        obj._n_obs_total = len(y)
        obj.nobs = float(obj._n_drivers)
        obj.df_resid = obj._n_drivers - len(start_params)
    else:
        obj._n_drivers = None
    return obj
```

**Critical:** setting `nobs = D` (number of drivers) ensures statsmodels computes correct degrees of freedom. The Hessian is computed from the scalar `loglike`, so it is unaffected by `nobs`.

---

## Step 5: MLE Likelihood — Driver-Level `_ll`

**File:** `ml_switching_reg/mle.py`, method `_ll` (line 583)

The current method returns `(N,)` — one log-likelihood per observation. For driver-level, return `(D,)` — one per driver.

### Key change (after existing `rnl` computation, replacing line 611):

```python
    # rnl is (N, R) at this point — per-obs densities

    if getattr(self, '_driver_ids', None) is None:
        # Observation-level (original)
        return np.log((rnl * self._weighted_cm).sum(axis=1))

    # --- Driver-level likelihood ---
    D = self._n_drivers
    R = self.n_regimes

    # Compute log-densities directly (avoids log(pdf) roundtrip)
    log_f = np.log(rnl + 1e-300)  # (N, R)

    # Sum log-densities within each driver → (D, R)
    log_prod = np.zeros((D, R))
    np.add.at(log_prod, self._obs_to_driver, log_f)

    # log L_i = log[ Σ_r w_i(r) · exp(log_prod_i(r)) ]
    # = log[ Σ_r exp( log w_i(r) + log_prod_i(r) ) ]
    log_terms = self._log_w_driver + log_prod  # (D, R)

    # Log-sum-exp across regimes
    max_log = log_terms.max(axis=1, keepdims=True)  # (D, 1)
    ll_driver = max_log.squeeze() + np.log(
        np.exp(log_terms - max_log).sum(axis=1) + 1e-300
    )  # (D,)

    return ll_driver
```

### `nloglikeobs` (line 623):
No code change needed — it returns `-self._ll(params)`. The return shape changes from `(N,)` to `(D,)` automatically.

### Standard errors — the key payoff:
With driver-level likelihood:
- `loglike(params)` returns the correct scalar (base class sums `loglikeobs`)
- `hessian(params)` uses numerical differentiation of `loglike` → correct driver-level Hessian
- `score(params)` uses numerical differentiation of `loglike` → correct
- Standard errors from `√(diag(−H⁻¹))` are correct — **no clustering needed**
- `score_obs` returns `(D, k)` Jacobian — equivalent to driver-clustered robust SEs

### Performance note — avoid `log(norm.pdf(...))`:

The current code computes `norm.pdf(...)` then takes `log(...)`. For the driver-level case, this loses precision when densities are very small (large T makes products tiny). Better to compute log-densities directly:

```python
log_f = -0.5 * np.log(2 * np.pi) - np.log(np.abs(sigma)) \
        - 0.5 * ((self.y[None, :] - Xb) / sigma) ** 2   # (R, N)
log_f = log_f.T  # (N, R)
```

This avoids the `norm.pdf → log` roundtrip and prevents underflow for large T.

---

## Step 6: Update `from_formula` Methods

**File:** `ml_switching_reg/irls.py`, method `from_formula` (line 138)

Add `panel_likelihood=True` parameter:

```python
@classmethod
def from_formula(cls, formula, data, classifier_pred, cm,
                 entity_effects=False, time_effects=False,
                 panel_likelihood=True, **kwargs):
    # ... existing formula parsing ...
    driver_ids = data.index.get_level_values(0).values if panel_likelihood else None
    return cls(y, X_list, cp, np.asarray(cm, dtype=float),
               driver_ids=driver_ids, **kwargs)
```

Similarly for `mle.py` `from_formula_arrays` if it exists.

---

## Step 7: Monte Carlo Simulation Updates

**File:** `ml_switching_reg_sim/monte_carlo.py`

### 7.1 Pass `driver_ids` through to estimators

Update the `mle_fit` method (around line 209) to accept and forward `driver_ids`:

```python
def mle_fit(self, data, beta_start, mw, endog_col="y",
            estimator=None, p_mat_start=None, panel_likelihood=True):
    # After extracting arrays:
    y, X_list, cp, cm, driver_ids = extract_estimator_inputs(...)
    ids = driver_ids if panel_likelihood else None
    # Pass to estimator construction
```

### 7.2 Add simulation axis

Add `panel_likelihood` as a simulation axis so the Monte Carlo framework can compare observation-level vs driver-level in the same run — useful for the paper.

---

## Step 8: Verification Plan

### 8.1 T=1 equivalence test (backward compatibility)

Generate data with `time_periods=1`. Verify that:
- IRLS with `driver_ids` produces identical `τ`, `β`, `σ²` to IRLS without
- MLE with `driver_ids` produces identical log-likelihood, parameters, and standard errors
- **This test must pass before proceeding.**

### 8.2 T>1 difference test

Generate data with `time_periods=10`. Verify that:
- Driver-level log-likelihood < observation-level (obs-level has more free cross-regime terms)
- Driver-level E-step produces posteriors constant within each driver
- Driver-level MLE standard errors are **larger** than observation-level (obs-level overstates information)
- Point estimates are similar (both are consistent)

### 8.3 Coverage simulation (the definitive test)

Run a Monte Carlo with `time_periods=10`, `n_reps=1000`:

| Estimator | Expected 95% coverage |
|---|---|
| Observation-level MLE | Below 95% (SEs too small) |
| Driver-level MLE | Near 95% (correct SEs) |

This is the key validation that the fix matters for the paper.

### 8.4 Add verification cells to `theory_derivations.py`

Extend the existing notebook with a cell that:
1. Creates a small panel dataset
2. Fits both observation-level and driver-level IRLS
3. Fits both observation-level and driver-level MLE
4. Compares log-likelihoods, point estimates, and standard errors
5. Confirms equivalence at T=1

---

## Step 9: Potential Pitfalls

### 9.1 Numerical stability (most important)

The product of $T$ Gaussian densities underflows to zero for large $T$ when computed naively. The log-sum-exp approach in Steps 3 and 5 handles this. Additionally, compute log-densities directly rather than `log(norm.pdf(...))` — the latter loses precision for very small densities.

### 9.2 `weighted_cm` constancy within driver

The implementation assumes `weighted_cm` is constant within a driver. This is true by construction: `MisclassificationCreator` assigns misclassification probabilities per driver, then broadcasts to all time periods. Add a runtime assertion:

```python
if driver_ids is not None:
    for d in self._unique_drivers:
        idx = self._driver_indices[d]
        assert np.allclose(self.weighted_cm[idx[0]], self.weighted_cm[idx]), \
            f"weighted_cm not constant within driver {d}"
```

### 9.3 Unbalanced panels

If drivers have different numbers of time periods, `_obs_to_driver` and `np.add.at` handle this correctly — no special code needed.

### 9.4 statsmodels `nobs` vs `endog` length

`GenericLikelihoodModel` stores `self.endog` as `(N,)`. Some internal methods use `len(self.endog)` instead of `self.nobs`. If this causes issues, override `nobs` as a `@property`. Monitor for mismatches during testing.

### 9.5 Legacy DataFrame-mode constructor

The non-`arrays_mode` `__init__` path in `mle.py` (lines 236-408) is complex legacy code. Adding driver-level support there is optional — the `from_arrays` path is sufficient for the simulation framework.

---

## Implementation Sequence

Execute in this order to maintain a working system at each step:

| Step | File | Change | Test |
|------|------|--------|------|
| 1 | `data_creation.py` | Add `driver_ids` to return | All existing code still runs (ignore extra return) |
| 2 | `irls.py` | Add `driver_ids` to constructor | `driver_ids=None` preserves old behavior |
| 3 | `irls.py` | Implement driver-level E-step | T=1 equivalence test |
| 4 | `mle.py` | Add `driver_ids` to `from_arrays` | `driver_ids=None` preserves old behavior |
| 5 | `mle.py` | Implement driver-level `_ll` | T=1 equivalence test |
| 6 | `mle.py` | Verify SEs are correct | Compare SEs at T=1 and T>1 |
| 7 | `monte_carlo.py` | Pass `driver_ids` through | Existing MC still runs |
| 8 | `theory_derivations.py` | Add verification cells | Visual + numerical checks |
| 9 | Monte Carlo | Run coverage simulation | Coverage near 95% for driver-level |

---

## Critical Files

| File | Repo | Action |
|------|------|--------|
| `ml_switching_reg/irls.py` | `ml-switching-reg` | Modify `__init__`, `_estep`, `from_formula` |
| `ml_switching_reg/mle.py` | `ml-switching-reg` | Modify `from_arrays`, `_ll` |
| `ml_switching_reg_sim/data_creation.py` | `ml-switching-reg-sim` | Modify `extract_estimator_inputs` |
| `ml_switching_reg_sim/monte_carlo.py` | `ml-switching-reg-sim` | Pass `driver_ids` through |
| `examples/theory_derivations.py` | `ml-switching-reg-sim` | Add verification cells |
