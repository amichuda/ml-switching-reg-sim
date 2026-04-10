# Plan: Theory Section Derivations for Switching Regression Paper

## Context

The JAE simulation paper needs a formal theory section deriving the estimators, their properties, and bias expressions under misclassification. Section 5 of the Michuda JDE paper (pages 37-43) defines the model and likelihood but does not include formal derivations of the estimators, bias expressions, or the IRLS/EM connection. No symbolic derivations exist in the repo — this fills that gap.

The deliverables: (1) formal derivations via the `econometric-theorist` agent, (2) a marimo notebook rendering them in LaTeX with simulation verification, and (3) an updated PAPER_ROADMAP.md.

---

## Step 1: Add SymPy dependency

- Add `"sympy>=1.12"` to `pyproject.toml` under `[project.dependencies]`
- Run `uv sync`

**File:** `pyproject.toml`

---

## Step 2: Derive MLE estimators (regime-specific betas + sigma)

Use the `econometric-theorist` agent. The model from the paper:

**Likelihood per observation:**
```
L_n = Σ_j φ((y_n - X_j'β_j)/σ)/σ × w_n(j)
```
where `w_n(j) = Σ_k p_k(n) × π_{jk}` (precomputed from classifier + confusion matrix).

**Derivation steps:**
1. Write log-likelihood: `ℓ = Σ_n log L_n`
2. Define posterior weights: `τ_{nj} = φ_j × w_n(j) / L_n`
3. FOC w.r.t. `β_j`: yields weighted normal equation `(X_j'W_jX_j)β_j = X_j'W_jy` where `W_j = diag(τ_{·j})`
4. FOC w.r.t. `σ`: yields `σ² = (1/N) Σ_n Σ_j τ_{nj}(y_n - X_j'β_j)²`
5. Note: can't be solved in closed form since `τ_{nj}` depends on `β_j` — motivates EM

**Verify against code:** `mle.py:583` (`_ll` method), `irls.py:69` (`_beta_h`), `irls.py:78` (`_sigma2_h`)

---

## Step 3: Derive constant-beta restriction

Impose `β_j = β ∀j`. The key result:

```
β_const = (Σ_j X_j'W_jX_j)⁻¹ × Σ_j X_j'W_jX_j × β_j = (Σ_j A_j)⁻¹ Σ_j A_j β_j
```

This is a **precision-weighted average** of regime-specific betas. When all regimes share the same design matrix (as in the simulation DGP where `X_list = [X]*R` at `data_creation.py:229`), simplifies to:

```
β_const = Σ_j n_j^eff × β_j / Σ_j n_j^eff
```

where `n_j^eff = Σ_n τ_{nj}` is the effective sample size per regime.

---

## Step 4: Derive IRLS as EM algorithm

**Complete-data log-likelihood** (treating true regime `r_n` as observed):
```
ℓ_c = Σ_n Σ_j I(r_n=j) × [log φ((y_n - X_j'β_j)/σ) - log σ + log w_n(j)]
```

**E-step:** Replace `I(r_n=j)` with posterior `τ_{nj}^(t)` = code's `_estep()` at `irls.py:49`

**M-step:** Maximize Q-function → WLS per regime (matches `_beta_h`) + pooled variance (matches `_sigma2_h`)

**Convergence:** EM monotonicity theorem guarantees `ℓ(θ^(t+1)) ≥ ℓ(θ^(t))`

---

## Step 5: Library assessment

**Result: No canned Python library works.** The key blocker is observation-specific mixing weights `w_n(j)` from an external classifier with a known (not estimated) confusion matrix.

- `statsmodels MarkovRegression` — Markov transition probs, not cross-sectional misclassification
- `sklearn GaussianMixture` — estimates mixing weights as parameters, can't fix to pre-computed values
- `pomegranate`, `hmmlearn` — same limitations

The custom IRLS and MLE implementations are necessary. Note this in the notebook.

---

## Step 6: Derive OLS bias under misclassification

**Key result** (for slope, predicted regime `j`, equicorrelated shocks with correlation `ρ`):
```
plim β̂₁ⱼ^OLS = π_{jj} × β_{1,j} + (1 - π_{jj}) × ρ × β̄_{-j}
```

where `β̄_{-j} = (1/(R-1)) Σ_{k≠j} β_{1,k}`.

**Two bias sources:**
1. **Attenuation:** `π_{jj} < 1` shrinks the true effect
2. **Contamination:** `(1-π_{jj}) × ρ × β̄_{-j}` injects other regimes' effects, modulated by shock correlation

**Limiting cases:**
- `ρ = 0`: slope bias = `-(1-π_{jj})β_{1j}` (attenuation toward zero; contamination vanishes but attenuation persists — slope is NOT unbiased at ρ=0)
- `ρ = 1`: maximum contamination, `Bias = (1-π_{jj})(β̄_{-j} - β_{1,j})`
- `Π → uniform`: all regime estimates collapse to the same value

**Sigma bias:** `E[σ̂²_OLS] = σ² + Σ_k π_{kj} × (β_k - β̂_j^OLS)'E[X_kX_k'](β_k - β̂_j^OLS)` — always upward biased.

**Multi-regime scaling:** For symmetric CM with `π_{jj}` fixed, as R grows, each contaminant contributes less (`π_{kj} = (1-π_{jj})/(R-1)`) but there are R-1 of them. Net bias depends on dispersion of true betas.

**Correlated shocks:** When `ρ_{kj} → 1`, contamination from regime k enters at full strength. When `ρ_{kj} → 0`, contamination vanishes regardless of misclassification. This is the key insight: **misclassification bias is modulated by shock correlation**.

---

## Step 7: Create marimo notebook `examples/theory_derivations.py`

**Cell structure (17 cells):**

| Cell | Content | Type |
|------|---------|------|
| 1 | Imports (marimo, sympy, numpy, existing estimators) | Code |
| 2 | Title + overview | `mo.md()` |
| 3 | DGP definition in LaTeX | `mo.md()` |
| 4 | Symbolic likelihood (R=2), FOCs via SymPy | Code + LaTeX |
| 5 | Score equations display | `mo.md()` |
| 6 | **Verification:** Generate data, fit MLE, evaluate analytic score ≈ 0 | Simulation |
| 7 | EM/IRLS derivation in LaTeX | `mo.md()` |
| 8 | **Verification:** IRLS convergence plot (monotone log-likelihood) | Simulation |
| 9 | Constant-beta derivation + precision-weighted average | `mo.md()` |
| 10 | **Verification:** Identical betas → regime-specific = constant | Simulation |
| 11 | OLS bias derivation (full LaTeX) | `mo.md()` |
| 12 | **Interactive:** Sliders for extent, ρ, R → analytic + simulated bias | Interactive |
| 13 | Sigma bias derivation | `mo.md()` + Simulation |
| 14 | Multi-regime scaling (bias vs R plot) | Simulation |
| 15 | Correlated shocks bias derivation (equicorrelated case) | `mo.md()` |
| 16 | **Verification:** Sweep ρ, plot analytic vs simulated bias | Simulation |
| 17 | Summary and key takeaways | `mo.md()` |

**Reuse existing infrastructure:**
- `UberDatasetCreatorHet` from `data_creation.py` for DGP
- `extract_estimator_inputs()` for (y, X_list, classifier_pred, cm) tuple
- `MLSwitchingRegIRLS` from `irls.py` for IRLS fits
- `DriverSpecificProbUberMLE.from_arrays()` from `mle.py` for MLE fits
- `noisify_matrix()` for controlled misclassification
- Follow patterns from `examples/irls_vs_mle_comparison.py` for cell structure

---

## Step 8: Update PAPER_ROADMAP.md

Add "Phase 0.5: Theory Section" between Phase 0 and Phase 1:

```markdown
## Phase 0.5: Theory Section
**Depends on:** Phase 0 decisions
**Who:** econometric-theorist agent + manual

### Steps
1. Formal MLE derivation (score equations, sigma estimator)
2. EM/IRLS as proper EM algorithm (E-step, M-step, convergence)
3. Constant-beta restriction → precision-weighted average of regime betas
4. OLS bias = f(Π, β_true, ρ) — closed-form for equicorrelated case
5. Sigma bias under misclassification (always upward)
6. Multi-regime scaling and correlated-shocks analysis
7. All derivations verified in `examples/theory_derivations.py` marimo notebook
```

Update dependency graph: Phase 0.5 feeds into Phase 1 (simulation designs should test analytic predictions) and Phase 4 (outline needs the theory).

---

## Execution Order

1. Step 1 — add SymPy dependency (`pyproject.toml` edit + `uv sync`)
2. Steps 2-6 — launch `econometric-theorist` agent for all derivations (produces LaTeX + SymPy expressions)
3. Step 7 — create the marimo notebook, integrating derivations + simulation verification cells
4. Step 5 — library assessment note included in notebook
5. Step 8 — update PAPER_ROADMAP.md
6. Verify — run `marimo run examples/theory_derivations.py` to confirm notebook executes

---

## Critical Files

| File | Action |
|------|--------|
| `pyproject.toml` | Add sympy dependency |
| `examples/theory_derivations.py` | **Create** — new marimo notebook |
| `paper-writer/PAPER_ROADMAP.md` | Update with Phase 0.5 |
| `ml_switching_reg/mle.py` | Read-only reference (likelihood at line 583) |
| `ml_switching_reg/irls.py` | Read-only reference (E-step line 49, M-step line 69) |
| `ml_switching_reg_sim/data_creation.py` | Read-only reference (DGP, noisify_matrix) |
| `examples/irls_vs_mle_comparison.py` | Read-only reference (marimo patterns) |
