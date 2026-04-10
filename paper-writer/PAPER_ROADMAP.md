# Plan: From Code-Complete to Submitted Paper

## Context

The `ml-switching-reg-sim` project has a **complete simulation codebase** (DGP, IRLS estimator, MLE estimator, Monte Carlo harness, three marimo analysis notebooks) and a **complete paper-writing pipeline** (8 Claude Code agents, JAE template) but **zero paper outputs** — no saved simulation results, no literature review, no outline, no draft. The goal is to take this to a submitted manuscript at the Journal of Applied Econometrics.

The paper's contribution: a switching regression estimator that corrects for ML-based regime misclassification, validated via Monte Carlo simulation and (potentially) applied to Uganda ride-sharing data.

---

## Phase 0: Pre-Simulation Decisions (User Required)

These cascade through every downstream phase — resolve first.

### Decision 1: Estimator Presentation -- RESOLVED: MLE primary, IRLS for initialization
MLE provides Fisher-information SEs for coverage. IRLS used for warm-start and convergence diagnostic.

### Decision 2: Paper Structure -- RESOLVED: Simulation only
Reference Michuda JDE for empirical context. No empirical application section.

### Decision 3: MC Replicate Count -- RESOLVED: 1000 replicates
Standard for published MC studies. ~0.7pp simulation SE on coverage rates.

### Decision 4: Fixed Effects -- RESOLVED: Yes, include FE variation
Add designs with/without driver and time FE in the DGP as a simulation axis.

---

## Phase 0.5: Theory Section
**Depends on:** Phase 0 decisions
**Who:** `econometric-theorist` agent + manual

### Steps
1. Formal MLE derivation (score equations, sigma estimator)
2. EM/IRLS as proper EM algorithm (E-step, M-step, EM monotonicity convergence)
3. Constant-beta restriction → precision-weighted average of regime betas
4. OLS bias = f(Π, β_true, ρ) — closed-form for equicorrelated case
5. Sigma bias under misclassification (always upward biased)
6. Multi-regime scaling and correlated-shocks analysis
7. All derivations verified in `examples/theory_derivations.py` marimo notebook

**Output:** `examples/theory_derivations.py` ✅ DONE

---

## Phase 1: Production Simulation Runs
**Depends on:** Phase 0 decisions (Phase 0.5 informs simulation designs)
**Duration:** 2-5 days (including compute time)
**Who:** `analysis-coder` agent or manual scripting

### Steps
1. Create `scripts/run_paper_simulations.py` — non-interactive batch script that:
   - Defines 5 simulation designs (1000 reps each):
     - **Design 1 (Baseline):** R=2, N=200, T=15, no FE. Vary misclassification weight. OLS vs MLE.
     - **Design 2 (Classifier comparison):** Noisify vs XGBoost at matched P(correct). No FE.
     - **Design 3 (Sample size):** R=2, fix weight=0.3, no FE. Vary N and T separately.
     - **Design 4 (Multi-regime):** R ∈ {2,3,4}, N=200, T=15, no FE. Vary weight.
     - **Design 5 (Fixed effects):** R=2, N=200, T=15, driver FE + time FE in DGP. Vary weight. Compare demeaned MLE vs naive MLE.
   - For each replicate: OLS → IRLS → MLE (IRLS-initialized). Record estimates, SEs, coverage.
   - Saves per-design results to `results/*.parquet`
   - Supports resumption (skip existing grid points)

2. Create `scripts/generate_paper_figures.py` — reads saved results, produces:
   - **Fig 1:** Bias/RMSE of OLS vs MLE as misclassification increases (hero figure)
   - **Fig 2:** Noisify vs XGBoost comparison
   - **Fig 3:** RMSE vs sample size (shows √N convergence)
   - **Fig 4:** Multi-regime scaling
   - Saved as PDF + PNG to `results/figures/`

3. Create `scripts/generate_paper_tables.py` — produces LaTeX tables:
   - **Table 1:** Simulation design summary
   - **Table 2:** Bias, RMSE, coverage for OLS vs MLE (central evidence)
   - **Table 3:** Coverage across all designs
   - Saved as `.tex` to `results/tables/`

4. Run simulations, verify outputs

### Critical files
- `ml_switching_reg_sim/monte_carlo.py` — existing harness to build on
- `examples/classifier_comparison.py` — has the bias/RMSE/coverage patterns to replicate
- `ml_switching_reg/mle.py` — `DriverSpecificProbUberMLE.from_arrays` for estimation

---

## Phase 2: Literature Review
**Depends on:** Nothing (run in parallel with Phase 1)
**Who:** `literature-gatherer` agent (Sonnet)

Search scope:
- Switching/mixture regression (Quandt 1972, Goldfeld-Quandt 1973, Hamilton 1989)
- Misclassification correction (Aigner 1973, Bollinger 1996, Mahajan 2006, Lewbel 2007)
- ML predictions in econometrics (Athey-Imbens, Chernozhukov DML, Aronsson 2024)
- EM algorithm (Dempster-Laird-Rubin 1977)
- Ethnicity prediction from names (Imai-Khanna 2016)
- Gig economy labor supply (if empirical section included)

**Output:** `paper-writer/literature/literature_review.md`

---

## Phase 3: Results Summary
**Depends on:** Phase 1 complete
**Who:** `results-reader` agent (Sonnet)

- Reads simulation outputs from `results/`
- Produces `paper-writer/outline/results_summary.md`

---

## Phase 4: Paper Outline
**Depends on:** Phase 2 + Phase 3 complete + Phase 0.5 theory results
**Who:** `outline-planner` agent (Sonnet)

Proposed structure:
1. Introduction
2. The Switching Regression Model with Misclassification
3. Theory: Bias under Misclassification (draws from Phase 0.5)
4. Estimation (MLE primary, IRLS initialization)
4. Monte Carlo Evidence (5 designs: baseline, classifier type, sample size, multi-regime, FE)
5. Conclusion

**USER CHECKPOINT:** Review outline before proceeding to draft.

---

## Phase 5: Paper Draft (v1)
**Depends on:** Phase 4 approved
**Who:** `paper-writer` agent (Opus)

- Reads outline, literature review, results summary
- Writes full draft to `paper-writer/draft/paper_draft_v1.md`
- Logs missing tables/figures in `analysis_requests.md`
- Target: 6,000-10,000 words

---

## Phase 6: Analysis Gap Fill (if needed)
**Depends on:** Phase 5 identifies gaps
**Who:** `analysis-coder` agent (Sonnet)

---

## Phase 7: Adversarial Review
**Depends on:** Phase 5 (+ Phase 6 if triggered)
**Who:** `adversarial-critic` agent (Sonnet)

**Outputs:** `paper-writer/review/referee_report.md`, `review/revision_checklist.md`

Likely critique areas:
- Coverage at high misclassification — does the estimator work when you need it most?
- Stylized DGP — sensitivity to DGP misspecification?
- Known confusion matrix assumption — how realistic?
- Comparison to alternatives (bootstrap naive OLS, latent class models)

---

## Phase 8: Revision (v2)
**Depends on:** Phase 7
**Who:** `paper-reviser` agent (Opus)

**Outputs:** `paper-writer/draft/paper_draft_v2.md`, `review/response_to_reviewer.md`

**USER CHECKPOINT:** Approve prose before formatting.

---

## Phase 9: PDF Formatting
**Depends on:** Phase 8 approved
**Who:** `pdf-formatter` agent (Sonnet)

- Scaffolds `paper-writer/draft/paper.qmd` from `templates/jae_template.qmd`
- Creates `references.bib`, downloads `jae.bst`
- Renders to PDF, iterates on formatting
- **Output:** `paper-writer/draft/paper.pdf`

---

## Phase 10: Final Review and Submit
**Who:** User
- Read PDF end-to-end, verify numbers
- Write cover letter
- Submit to JAE

---

## Dependency Graph

```
Phase 0 (decisions) ✅ DONE
  │
Phase 0.5 (theory derivations) ✅ DONE → examples/theory_derivations.py
  ├── Phase 1 (simulations, 1000 reps) ── Phase 3 (results digest) ──┐
  └── Phase 2 (lit review) ── parallel with Phase 1 ─────────────────┤
                                                                      v
                                                             Phase 4 (outline, + theory)
                                                                      │
                                                             [USER CHECKPOINT]
                                                                      │
                                                             Phase 5 (draft v1)
                                                                      │
                                                             Phase 6 (gap fill, if needed)
                                                                      │
                                                             Phase 7 (adversarial review)
                                                                      │
                                                             Phase 8 (revision v2)
                                                                      │
                                                             [USER CHECKPOINT]
                                                                      │
                                                             Phase 9 (PDF formatting)
                                                                      │
                                                             Phase 10 (submit)
```

---

## Risks

1. **MLE convergence failures at high misclassification** — Track `success` flag; report convergence rates alongside coverage; use IRLS warm-start aggressively
2. **Coverage under-coverage** — Report honestly; consider profile likelihood CIs as robustness
3. **Thin literature folder** — Only 4 papers currently; literature-gatherer must find classical references
4. **Reviewer demands empirical application** — Simulation-only framing may face "where's the application?" pushback. Mitigation: reference Michuda JDE prominently; frame paper as the methodological complement

---

## Setup: Create Directory Structure

Before starting Phase 1, create:
```
paper-writer/literature/
paper-writer/outline/
paper-writer/draft/
paper-writer/review/
results/
results/figures/
results/tables/
scripts/
```
