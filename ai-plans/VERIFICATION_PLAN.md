# Verification Plan: Econometric Soundness of Simulation Notebooks

## Overview

This plan outlines a systematic verification of the switching regression estimators (MLE + IRLS) and their theoretical foundations. The goal is to ensure all simulation notebooks are econometrically sound before paper writing begins.

**File locations:**
- Simulation harness: `ml_switching_reg_sim/` (this repo)
- MLE/IRLS estimator: `../ml-switching-reg/ml_switching_reg/` (companion repo, editable install)
- Notebooks: `examples/`

---

## Phase 1: Code-to-Theory Alignment

*Do this first — you cannot trust notebook outputs until the underlying code is verified.*

### 1.1 Likelihood Function Verification

**Theory:** L_n = Σ_j (1/σ)φ((y_n - X_j'β_j)/σ) × w_n(j)

**Code:** `../ml-switching-reg/ml_switching_reg/mle.py:583-611` (`_ll` method)

**Checks:**
- [ ] The einsum computation `np.einsum('rni,ri->rn', X, beta_mat)` correctly computes X_j'β_j for all regimes
- [ ] `norm.pdf(self.y[None, :] - Xb, 0, scale=np.abs(sigma)).T` gives (N, R) matrix of φ_j/σ
- [ ] `(rnl * self._weighted_cm).sum(axis=1)` correctly computes Σ_j φ_j × w_n(j)
- [ ] The log is taken after summing, not before: `np.log((rnl * self._weighted_cm).sum(axis=1))`

**Verdict:** Code matches theory.

### 1.2 E-Step Verification

**Theory:** τ_{nj} = φ_j × w_n(j) / L_n

**Code:** `../ml-switching-reg/ml_switching_reg/irls.py:49-63` (`_estep` method)

**Checks:**
- [ ] `regimes[r] = exp(-(y - X_list[r] @ beta[r])² / (2σ²))` — correct (normalizing constant dropped)
- [ ] `unnorm = regimes * self.weighted_cm.T` — correctly multiplies by w_n(j)
- [ ] `unnorm / (unnorm.sum(axis=0) + 1e-300)` — correct normalization

**Verdict:** Code matches theory.

### 1.3 M-Step Verification

**Theory:** β_j = (X_j'W_jX_j)⁻¹ X_j'W_jy, σ² = (1/N) Σ_n Σ_j τ_{nj}(y_n - X_j'β_j)²

**Code:** `../ml-switching-reg/ml_switching_reg/irls.py:69-83` (`_beta_h`, `_sigma2_h`)

**Checks:**
- [ ] `(Wr * wt[:, None]).T @ Wr + l2_lambda * np.eye(...)` computes X_j'W_jX_j + λI
- [ ] `(Wr * wt[:, None]).T @ self.y` computes X_j'W_jy
- [ ] `resid2.sum() / (rrh.sum() + 2 * l2_lambda)` — correct (rrh.sum() = N when l2=0)

**Verdict:** Code matches theory.

### 1.4 DGP Verification

**Theory:** y_{it} = X_{jt}'β_j + ε_{it}, ε_{it} ~ N(0, σ²), j = r_i

**Code:** `ml_switching_reg_sim/data_creation.py:273-296` (`_create_y`)

**Checks:**
- [ ] Per-regime outcome: `beta0[i] + beta1[i] * df[f'drought_{i}'] + N(0, sd[i])`
- [ ] Single σ shared across regimes when `y_sd=[1.0, 1.0]`
- [ ] Regime assignment is random per driver via `_regime_index`
- [ ] Fixed effects added additively to y

**Verdict:** DGP matches theory.

### 1.5 Confusion Matrix Construction

**Theory:** π_{kj} = P(true=k | predicted=j), column-normalized

**Code:** `ml_switching_reg_sim/data_creation.py:59` — `cm = mw / mw.sum(axis=0, keepdims=True)`

**Checks:**
- [ ] `noisify_matrix` (line 122-139) produces row-stochastic matrices (rows sum to 1)
- [ ] `mw` is row-stochastic: mw[r, :] = E[predicted proba | true_regime == r]
- [ ] Column normalization gives P(true | predicted)
- [ ] `weighted_cm = classifier_pred @ cm.T` computes w_n(j) = Σ_k p_k(n) × π_{jk}
- [ ] For R>2, off-diagonal mass distribution is asymmetric by design (document at line 122)

**Verdict:** Confusion matrix construction is correct.

### 1.6 Score Function Verification

**Theory:** ∂ℓ/∂σ = (1/σ) Σ_n Σ_j τ_{nj}[(y_n - X_j'β_j)²/σ² - 1]

**Code:** `examples/theory_derivations.py:277-280`

**Checks:**
- [ ] `score_sigma = sum(tau[r] * ((y - Xβ)²/σ² - 1)).sum() / (-sigma)` — **BUG: should divide by `+sigma`, not `-sigma`**. The negative sign flips the score direction.

**Verdict:** Bug found — P3 (High severity).

---

## Phase 2: Audit Existing Notebooks

### 2.1 `examples/theory_derivations.py` — Theory Verification Notebook

**Status:** Exists, 1134 lines, 8 sections covering THEORY_PLAN.md derivations.

**Verification checks:**
- [ ] **Score ≈ 0 at MLE solution** (Section 2) — after fixing P3, verify `_compute_score` returns near-zero
- [ ] **EM monotone convergence** (Section 3) — confirm all log-likelihood increments are ≥ 0
- [ ] **Constant-beta = pooled OLS** (Section 4) — verify precision-weighted average formula
- [ ] **OLS bias vs analytic prediction** (Section 5) — confirm plim formula matches simulation at ρ=0
- [ ] **Sigma bias** (Section 6) — verify OLS σ̂ inflates, IRLS σ̂ consistent
- [ ] **Multi-regime scaling** (Section 7) — verify bias behavior as R grows
- [ ] **Correlated shocks** (Section 8) — verify analytic plim tracks simulation across ρ sweep

**Action:** After applying fixes from Phase 3, run with `marimo run --no-browser examples/theory_derivations.py` and verify all cells execute without errors.

### 2.2 `examples/irls_vs_mle_comparison.py` — Estimator Comparison

**Status:** Exists.

**Verification checks:**
- [ ] IRLS and MLE converge to same parameter values (‖β_IRLS - β_MLE‖ < 1e-4)
- [ ] MLE standard errors correctly computed from Fisher information
- [ ] Both estimators recover true β as misclassification → 0

### 2.3 `examples/multi_regime_analysis.py` — Multi-Regime Behavior

**Status:** Exists.

**Verification checks:**
- [ ] Estimator performance degrades gracefully as R increases
- [ ] Identification issues flagged when effective sample per regime is too small

### 2.4 `examples/misclassification_monte_carlo.py` — Monte Carlo Study

**Status:** Exists.

**Verification checks:**
- [ ] Monte Carlo mean estimates converge to true values as N → ∞
- [ ] Coverage rates of confidence intervals close to nominal (95%)
- [ ] Bias direction matches theoretical predictions

### 2.5 `examples/classifier_comparison.py` — Classifier Impact

**Status:** Exists.

**Verification checks:**
- [ ] Both classifier modes (noisify vs ML) produce consistent confusion matrices
- [ ] Results robust to classifier choice

### 2.6 Legacy Notebooks

**Files:** `examples/mle_comparison.ipynb`, `examples/montecarlo_visualizations.ipynb`, `examples/quick_test.py`

**Action:**
- Review `.ipynb` files for consistency with marimo notebooks. If content is fully covered by marimo equivalents, mark for deletion; otherwise convert.
- `quick_test.py` imports from `UgandaUber.Estimation.modules` which no longer exists. Delete or archive.

---

## Phase 3: Issues Found & Fixes

| # | Severity | Location | Issue | Fix |
|---|----------|----------|-------|-----|
| **P1** | **High** | `examples/theory_derivations.py:1058` | `_pi_jj7 = 1.0 - _WEIGHT7 * 0.5` — hardcoded for R=2. Should be `1 - weight*(1-1/R)` for general R. | Replace `0.5` with `(1 - 1/R)` where R is the number of regimes in that section |
| **P2** | **High** | `examples/theory_derivations.py:706` | `_pi_jj4 = 1.0 - _weights4 * 0.5` — same issue | Same fix as P1 |
| **P3** | **High** | `examples/theory_derivations.py:280` | `score_sigma` divides by `(-sigma)` instead of `sigma`. Flips sign of sigma score. | Change `/ (-sigma)` to `/ sigma` |
| **P4** | Low | `../ml-switching-reg/ml_switching_reg/irls.py:83` | Ridge term `+ 2 * l2_lambda` in sigma2 denominator is undocumented | Add comment explaining the ridge regularization term |
| **P5** | Low | `../ml-switching-reg/ml_switching_reg/mle.py:704` | Arrays-mode sigma init uses marginal `np.std(y)` not residual std | Use residual-based sigma initialization |
| **P6** | Low | `ml_switching_reg_sim/data_creation.py:122-139` | `noisify_matrix` for R>2 distributes off-diagonal mass randomly | Add docstring documenting asymmetric misclassification behavior |

---

## Phase 4: New Deliverables

### 4.1 `examples/comprehensive_verification.py` (marimo notebook)

A single pass/fail verification notebook with 12 checks:

| # | Check | Expected Result |
|---|-------|-----------------|
| 1 | DGP sanity: regime-specific means match β | Means within 2 SE of true β |
| 2 | Score at MLE: |score| < 1e-6 | PASS (after P3 fix) |
| 3 | EM convergence: all Δℓ ≥ 0 | PASS |
| 4 | IRLS vs MLE: ‖β_IRLS - β_MLE‖ < 1e-4 | PASS |
| 5 | OLS bias direction: bias < 0 when β > 0, ρ = 0 | PASS |
| 6 | OLS bias magnitude: matches analytic plim within 5% | PASS (after P1/P2 fix) |
| 7 | Sigma consistency: |σ̂_MLE - σ_true| < 0.1 | PASS |
| 8 | Sigma bias: σ̂_OLS > σ_true at high misclassification | PASS |
| 9 | Constant-beta: precision-weighted = pooled OLS | ‖diff‖ < 1e-10 |
| 10 | MC coverage: 95% CI covers true β in ~95% of reps | Coverage within 2 binomial SE of 95% (≈ [92%, 98%] for N=200) |
| 11 | Multi-regime: variance ∝ R/N | Variance ratio matches theory |
| 12 | Correlated shocks: bias monotone in ρ | PASS |

After creation, run `marimo check --fix examples/comprehensive_verification.py`.

### 4.2 `examples/monte_carlo_coverage.py` (marimo notebook)

Dedicated MC study across parameter grid:
- **Misclassification weight:** [0.0, 0.1, 0.3, 0.5, 0.7, 0.9]
- **Regimes:** [2, 4]
- **Shock correlation ρ:** [0.0, 0.3, 0.6, 0.9]
- **Replications:** 200 per cell (9,600 total)

**Computational notes:**
- Use `multiprocessing.Pool` (as in `monte_carlo.py`) for parallelism
- Estimate: ~2-4 hours on 5-core machine; consider reducing reps to 100 for initial pass
- Save intermediate results to disk for resumability

**Outputs:** bias plots, coverage tables, RMSE decomposition, timing comparison.

After creation, run `marimo check --fix examples/monte_carlo_coverage.py`.

### 4.3 Reproducibility Check

- [ ] Run `examples/theory_derivations.py` twice with same seed — verify identical outputs
- [ ] Verify `np.random.default_rng(seed=...)` is used consistently (not legacy `np.random.seed()`)

### 4.4 `INSIGHTS.md`

Final document summarizing:
- All verification results (pass/fail with numbers)
- Issues found (P1-P6) and fixes applied (F1-F6)
- Remaining concerns or open questions
- Confidence level for paper writing

---

## Phase 5: Execution Order

1. **Verify code-to-theory alignment** (Phase 1) — read and validate all core functions
2. **Apply fixes P1, P2, P3, P4, P5, P6** — fix bugs and add documentation
3. **Run `examples/theory_derivations.py`** — `marimo run --no-browser`, verify all cells pass
4. **Run remaining audit notebooks** (Phase 2.2-2.5) — verify each executes without errors
5. **Handle legacy files** (Phase 2.6) — delete `quick_test.py`, decide on `.ipynb` files
6. **Create `examples/comprehensive_verification.py`** — then `marimo check --fix`
7. **Create `examples/monte_carlo_coverage.py`** — then `marimo check --fix`
8. **Run comprehensive verification notebook** — collect pass/fail results
9. **Run Monte Carlo coverage study** — use parallelism, save intermediate results
10. **Run reproducibility check** (Phase 4.3)
11. **Write `INSIGHTS.md`**

---

## Sub-Agent Delegation

| Agent | Task |
|-------|------|
| **econometric-theorist** | Verify score equations (Phase 1.6), derive correct π_{jj} for general R, confirm P3 fix |
| **general** | Apply fixes P1-P6, create marimo notebooks, run them, collect results, write INSIGHTS.md |

---

## Success Criteria

- All existing notebooks execute without errors
- Score ≈ 0 at MLE (|score| < 1e-6)
- EM log-likelihood monotone (all Δℓ ≥ 0)
- IRLS ≈ MLE (‖β_IRLS - β_MLE‖ < 1e-4)
- OLS bias matches analytic plim within 5%
- Monte Carlo 95% CI coverage within 2 binomial SE of 95%
- Reproducibility: identical results across runs with same seed
- `INSIGHTS.md` documents all findings
