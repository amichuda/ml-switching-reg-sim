# Optimization Opportunities

This document catalogs performance bottlenecks and optimization strategies for the
`ml-switching-reg` estimator and `ml-switching-reg-sim` simulation harness.
Items are ordered from highest expected impact to lowest.

---

## 1. Replace `jax.grad`-free optimization with `jaxopt` + analytic gradient

**File:** `ml-switching-reg/ml_switching_reg/mle.py` — `fit()` via `super().fit()`

**Current:** `statsmodels.GenericLikelihoodModel.fit()` computes the Jacobian numerically
(finite differences via `approx_fprime`). Each likelihood evaluation is already vectorized,
but the optimizer calls `_ll` O(p) extra times per gradient step.

**Fix:** Rewrite `nloglikeobs` to return a JAX-traced scalar, then use `jaxopt.LBFGSB` with
`jax.grad` for an analytic gradient:

```python
import jax
import jax.numpy as jnp
import jaxopt

@jax.jit
def neg_log_likelihood(params):
    ...  # same computation as _ll, but in jnp
    return -jnp.sum(log_likes)

solver = jaxopt.LBFGSB(fun=neg_log_likelihood)
result = solver.run(init_params)
```

**Expected gain:** 5–20× per optimization run (eliminates O(p) extra likelihood evaluations
per gradient step). Most impactful for large N or many parameters.

---

## 2. Default `hdfe=True` in `demean_data`

**File:** `ml-switching-reg/ml_switching_reg/mle.py` — `demean_data()` (line 310)

**Current:** When `hdfe=False` (the default), entity/time demeaning uses
`groupby().transform(lambda x: x - x.mean())` — a Python-level loop over groups:

```python
# Current (slow):
new_data = data.groupby(self.entity).transform(lambda x: x - x.mean())
```

`pyhdfe` is already imported and used in the `hdfe=True` path (`_hdfe_data`). It applies
the Frisch-Waugh-Lovell projection using sparse algebra and is substantially faster for
large panels.

**Fix:** Make `hdfe=True` the default, or replace the non-hdfe path with a vectorized
numpy alternative:

```python
# Vectorized alternative (no pyhdfe dependency):
entity_means = data.groupby(self.entity).transform('mean')
new_data = data - entity_means
```

Using `transform('mean')` instead of `transform(lambda x: x - x.mean())` avoids the
Python function call overhead per group.

**Expected gain:** 2–5× for the demeaning step, which runs once per regime per likelihood
evaluation via `_regime_list`.

---

## 3. Full JAX JIT of IRLS iteration loop

**File:** `ml-switching-reg-sim/analytical_mat.ipynb` — `MLSwitchingRegIRLS._irls()`

**Current:** The outer iteration loop (`_irls`) is pure Python. The `jacobian` method is
called once per iteration; `beta_h` uses `@partial(jit, ...)` on the inner linear solve,
but the loop structure, convergence check, and Python overhead dominate at scale.

**Fix:** Rewrite `_irls` as a `jax.lax.while_loop` so the entire iteration compiles to
XLA and runs without Python re-entry:

```python
import jax.lax as lax

def body_fn(state):
    theta, i = state
    _, theta_new = self.jacobian(theta, l2_lambda)
    return theta_new, i + 1

def cond_fn(state):
    theta, i = state
    _, theta_new = self.jacobian(theta, l2_lambda)
    return (jnp.linalg.norm(theta_new - theta) > tol) & (i < max_iter)

theta_final, _ = lax.while_loop(cond_fn, body_fn, (theta_0, 0))
```

**Note:** `cond_fn` calls `jacobian` twice per iteration as written above; fold norm
computation into `body_fn` via a carry variable to avoid the duplication.

**Expected gain:** 10–50× for large N (eliminates Python loop overhead and enables
XLA fusion across the E- and M-steps within a single iteration).

---

## 4. Eliminate intermediate array in `_ll` einsum

**File:** `ml-switching-reg/ml_switching_reg/mle.py` — `_ll()` (line 413–418)

**Current:**

```python
beta_vec = beta_df.values[:, np.newaxis]
Xb = np.einsum("ijk,kl -> ijl", X, beta_vec)[..., 0]  # (R, N, 1) → (R, N)
rnl = norm.pdf(self.y - Xb, 0, scale=np.abs(sigma)).T  # (N, R)
return np.log((rnl * (self.p @ self.cm.T)).sum(axis=1))
```

`X` has shape `(R, N, p)` and `beta_vec` has shape `(p, 1)`. The einsum allocates an
`(R, N, 1)` intermediate before slicing to `(R, N)`. Since `beta_vec` is the same for
all regimes, this is equivalent to a batched matrix-vector product.

**Fix:** Use `np.tensordot` or reshape to avoid the extra dimension:

```python
# X: (R, N, p), beta_df.values: (p,)
Xb = X @ beta_df.values  # (R, N) via broadcasting — no intermediate allocation
```

**Expected gain:** Modest (avoids one `(R, N, 1)` allocation per likelihood call), but
accumulates over thousands of optimizer iterations.

---

## 5. Cache `self.p @ self.cm.T` *(implemented — always on)*

**File:** `ml-switching-reg/ml_switching_reg/mle.py` — `_weighted_cm` cached property

`self.p` (the classifier probability matrix) and `self.cm` (the confusion matrix) are
both constant across all optimizer iterations. The product is now cached on first access
via `@functools.cached_property` and reused in every `_ll` call:

```python
@functools.cached_property
def _weighted_cm(self):
    return self.p @ self.cm.T  # (N, R), computed once

# In _ll:
return np.log((rnl * self._weighted_cm).sum(axis=1))
```

**Gain:** Eliminates one `(N, R)` matrix multiply per likelihood evaluation.
For N=30,000 (full Uganda dataset), R=10, this accumulates meaningfully over thousands
of optimizer iterations. No flag needed — always correct behavior.

---

## 6. Use `numpyro` / NUTS for posterior uncertainty (Bayesian extension)

**Scope:** Optional extension, not a performance fix for the existing MLE.

The IRLS algorithm (after convergence) provides point estimates but no standard errors
without an additional bootstrap or observed-Fisher step. If the paper moves toward
reporting posterior credible intervals rather than frequentist CIs, `numpyro` with the
NUTS sampler provides posterior uncertainty naturally and handles the confusion matrix
correction within a generative model:

```python
import numpyro
import numpyro.distributions as dist

def model(W, p, cm, y):
    beta = numpyro.sample("beta", dist.Normal(0, 10).expand([R, p]))
    sigma = numpyro.sample("sigma", dist.HalfNormal(1))
    weights = p @ cm.T                    # (N, R)
    means = jnp.einsum("rni,ri->rn", W, beta)  # (R, N)
    mixture = dist.MixtureSameFamily(
        dist.Categorical(probs=weights),
        dist.Normal(means.T, sigma)
    )
    numpyro.sample("y", mixture, obs=y)
```

**Tradeoff:** NUTS is slower per iteration than L-BFGS-B but provides calibrated
uncertainty without a separate bootstrap step.

---

## Full-Likelihood vs IRLS: Recommendation

For the JAE submission, **prefer `DriverSpecificProbUberMLE` (full likelihood)** as the
primary estimator:

| Criterion | Full likelihood | IRLS |
|---|---|---|
| Standard errors | Correct (Fisher information) | Requires bootstrap or observed Fisher |
| Convergence guarantee | BFGS (may fail) | Monotone EM (always improves) |
| Gradient computation | Numerical (slow) | Closed-form M-step |
| Sigma2 estimate | Correct | Bug fixed — now correct |
| Soft probability use | Correct after fix | Not implemented |
| Speed | Slow for large N | Fast per iteration, slow near optimum |

**Recommended workflow:**
1. Use IRLS (fast) to get good starting values for the optimizer.
2. Pass IRLS estimates as `start_params_res` to `DriverSpecificProbUberMLE.fit()`.
3. Report MLE standard errors from the Fisher information matrix.
4. Use IRLS vs MLE agreement (Step 5 in the implementation plan) as a convergence
   diagnostic — large disagreement signals optimizer failure, not model mismatch.

This combines the stability of EM initialization with the inferential correctness of
maximum likelihood, and directly mirrors the workflow in Aronsson (2024).
