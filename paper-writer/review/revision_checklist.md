# Revision Checklist

## Must Fix (blocks publication)

- [ ] Remove all disclaimers stating the naive baseline is absent (Section 6 paragraph 4 lines 145-147; Draft Notes block lines 179-181; implicit framing in abstract and introduction).
- [ ] Add a new Section 5.4 (or extend 5.1) introducing the naive hard-classification baseline. Use Figure 5 (`paper-writer/results/figures/figure_5_naive_vs_corrected.{pdf,png}`) and the new naive coverage column from `table_A2_coverage_all_cells.csv`.
- [ ] Update the abstract to report the naive vs corrected coverage comparison (e.g., `R = 4, w = 0.9` naive coverage 0.001 vs corrected bootstrap 0.471).
- [ ] Update the introduction's contribution paragraph to claim the quantified correction payoff over the naive baseline, not just "validates finite-sample behavior."
- [ ] Update the conclusion to lead with the correction-payoff finding.
- [ ] Add an identification paragraph (in Section 2 or Section 3) tying parameter identification to (i) the rank of regime-specific covariates, (ii) the dispersion of `w_i(·)` across units, and (iii) shock correlation as a separate threat. The current "weak classifier" intuition is correct but vague.
- [ ] Defend the driver-level mixture likelihood in Section 2 by explicitly stating the assumption "regime is fixed within driver" and the implication for the likelihood.
- [ ] Discuss the asymmetry of naive failure: the naive baseline matches MLE for `R = 2` (e.g., naive 0.955 vs bootstrap 0.960 at `w = 0.9`) but collapses for `R = 4` and `R = 10`. The paper should explain why hard classification works for two regimes but breaks for many.

## Should Fix (important for quality)

- [ ] Tighten the redundancy between Section 1 paragraph 5 and Section 4. Either compress the introduction preview or compress Section 4.
- [ ] Promote the IRLS-MLE near-identity result (Section 5.2 paragraph 3) — currently buried, but it is one of the paper's least-expected findings.
- [ ] Reconcile notation: equation in Section 2 uses `x_{r_i t}` while prose uses `x_rt`. Pick one convention and apply it consistently.
- [ ] Make the likelihood in equation (4) explicit about the role of `x_{rt}`: write `f(y_{it} | alpha_r + beta_r x_{rt}, sigma)`.
- [ ] Resolve "EM-style IRLS" inconsistent usage. Define once, then reuse one phrase.
- [ ] Address Section 6 paragraph 2 ("validation data are not optional"). Either run a small robustness check perturbing the confusion matrix, or move the recommendation to a clearly labeled future-work paragraph.
- [ ] Cross-reference Figure 5 from Section 5.1 paragraph 4 ("weakly identified high-regime cells").
- [ ] Add Hu (2008) to the literature placement paragraph as the foundational identification reference for mismeasured discrete variables.

## Nice to Have (improves paper)

- [ ] Add Hyslop and Imbens (2001) and Mullainathan and Spiess (2017) to the literature framing.
- [ ] Define or remove "conditional optimism" in the conclusion.
- [ ] Update the word count footer after revision.
- [ ] Reorder Section 5 subsections so the naive comparison appears before the RMSE/bias deep dive — the comparison is now the most important reader-facing result.
- [ ] Consider a one-paragraph summary table at the end of Section 5 listing the four headline findings (corrected coverage, correction payoff, IRLS-MLE near-identity, computation tradeoffs).
