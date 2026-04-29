# Referee Report

Paper: Correcting Machine-Learned Regime Misclassification in Switching Regressions
Reviewed: 2026-04-28
Recommendation: Major Revision

## Summary

The paper studies a switching-regression estimator that corrects for ML-classifier misclassification by combining soft predictions with a confusion matrix in a driver-level mixture likelihood. It evaluates the estimator in a 72-cell Monte Carlo, reporting near-nominal coverage when classifier signal is moderate and graceful failure when many-regime designs are paired with weak classifiers. The contribution is methodological and the evidence is honest, but the draft has not yet been updated to reflect the naive hard-classification baseline that is now available in `mc_naive_results/`. Once the naive comparison is integrated, the paper will have a quantitative horse-race result that materially strengthens the contribution. There are also several smaller issues with notation, identification framing, and consistency across sections.

## Major Concerns

1. **Out-of-date treatment of the naive baseline.** The draft repeatedly disclaims the absence of a naive hard-classification baseline (Section 6 paragraph 4, the Draft Notes block at the end, and implicitly in the abstract and introduction). That disclaimer is now obsolete: a full naive baseline matching the same Monte Carlo grid sits in `mc_naive_results/` and is integrated into Figure 5, the compact `table_A2_coverage_rho_0_6.tex` table, and the machine summary. The v2 must (a) remove the disclaimers, (b) introduce the naive baseline in Section 5 and Figure 5, and (c) use the naive vs corrected comparison in the abstract, introduction contribution paragraph, and conclusion. The new evidence is the strongest argument the paper has and currently sits unused.

2. **Headline contribution is understated.** The current abstract and introduction frame the paper as validating finite-sample behavior of the corrected estimator. With the naive baseline available, the paper can go further and report a quantitative correction payoff. The naive coverage at `R = 4, w = 0.9` is essentially zero (0.001) versus corrected MLE bootstrap coverage 0.471, and at `R = 10, w = 0.9` naive coverage is 0.145 versus 0.483. These are unusually clean numbers and should be the headline of the paper, not a future-revision note.

3. **Asymmetry of the naive failure deserves a substantive treatment.** The new evidence shows that the naive baseline matches MLE for `R = 2` even at `w = 0.9` (naive 0.955 vs bootstrap 0.960), but collapses for `R = 4` and `R = 10`. This is informative and not yet discussed: hard classification is fine when correct-class probability stays well above 0.5 in a binary problem, but breaks down quickly when regime count grows because the modal class is wrong on a large share of observations. The discussion should explicitly draw out this regime-count dependence rather than reporting a single average gap.

4. **Identification language is loose.** Section 2 introduces `w_i(k)` and the marginal likelihood but does not articulate when the slope parameters are identified. The draft reaches for "weak classifier" intuition in Section 5.1 paragraph 4 but does not formally connect identification to (i) the rank of the matrix of regime-specific covariates, (ii) the rank of the matrix of `w_i(·)` weights across units, or (iii) shock correlation as a separate identification threat. A short identification paragraph in Section 2 or Section 3 would strengthen the paper. As written, "the likelihood cannot manufacture identification" is rhetorically right but analytically vague.

5. **Driver-level vs observation-level likelihood is mentioned but not defended.** Section 2 distinguishes the driver-level mixture from an observation-level mixture but does not explain why the driver-level form is correct under the simulation DGP. The DGP keeps regime fixed within driver, so the likelihood is correct, but a reader without simulation-code context cannot tell. One paragraph is enough: state the assumption (regime is fixed within driver), state the implication for the likelihood, and flag what changes if regime can switch within driver across periods. This will matter for applied uptake.

## Minor Concerns

1. The abstract should report the naive comparison number (e.g., "naive 95% coverage drops to 0.001 at the hardest four-regime design while corrected coverage holds at 0.47"), not just the corrected coverage.

2. Section 1 paragraph 5 ("The Monte Carlo varies three features...") and Section 4 are largely redundant. Tighten one or the other.

3. Section 5.1 paragraph 4 says "weakly identified high-regime cells" — but those are also the cells where the naive baseline fails most. Cross-reference Figure 5 here.

4. Section 5.2 paragraph 3 ("IRLS and MLE are nearly indistinguishable in the RMSE frontier") is correct but buried. Consider promoting it; it is one of the paper's least-expected results and currently reads as a side note.

5. Equation 2 uses inline math notation `x_{r_i t}` while the prose uses `x_rt`. Reconcile.

6. The likelihood in equation 4 abstracts the role of `x_{rt}`. Make it explicit, e.g., `f(y_{it} | alpha_r + beta_r x_{rt}, sigma)`.

7. "EM-style IRLS" is used five times. Pick one term and define it once.

8. Section 6 paragraph 2 ("Validation data are not optional") is correct but currently dangles: the paper does not vary confusion-matrix accuracy in the simulation. Either commit to listing it as future work explicitly, or run a quick robustness check perturbing the confusion matrix.

9. The phrase "conditional optimism" in the conclusion is good but undefined. Either drop it or define it.

10. The Draft Notes block at line 179 should be deleted in v2 (the naive baseline is no longer absent).

11. Approximate word count claim (line 183) should be updated after revision.

## Specific Comments

- Line 9 (abstract): consider replacing "use the classifier's soft predictions and a confusion matrix to correct for regime misclassification" with "uses the classifier's soft predictions and a confusion matrix to correct for misclassification in regime assignment, and we show that the correction yields large bias and coverage gains over hard-classification OLS in many-regime designs."
- Line 25 ("The lesson is not that the estimator fails generally..."): this is the right framing. Repeat the structure when introducing Figure 5.
- Line 99-101: the paragraph mixing "first result" with the moderate-design averages is well-organized; reuse the structure for the naive comparison.
- Line 103 ("two-regime designs remain well behaved"): tie back to the naive result that R=2 also remains well-behaved naively.
- Line 117 ("IRLS and MLE are nearly indistinguishable"): consider promoting this to a separate subsection or numbered finding.
- Line 145-147: this paragraph should be removed (the naive baseline now exists).
- Line 179-181: this entire block should be removed.

## Assessment of Contribution

The contribution is real but currently underclaimed. With the naive baseline integrated, the paper has three findings of interest: (1) the corrected likelihood holds nominal coverage under moderate misclassification, (2) corrected coverage and bias are dramatically better than naive in many-regime settings, and (3) all estimators fail visibly when classifier signal collapses, which is itself useful guidance. The first finding alone is methodological housekeeping; the combination is a clean methods paper. The literature placement (post-Quandt switching regression, post-Mahajan/Lewbel misclassification, post-Athey-Imbens prediction-policy work) is appropriate and not overclaimed.

## Identification / Empirical Strategy Critique

The simulation correctly varies classifier informativeness, regime count, and shock correlation. Two extensions would strengthen the design:

- Misspecified confusion matrix: the current design assumes the estimator has access to the true confusion matrix. A robustness panel where the estimator uses a slightly perturbed confusion matrix would address Section 6 paragraph 2. This may be a separate paper but should be noted explicitly as future work, not buried.
- Within-driver regime switching: the current design fixes regime within driver. Settings where ethnicity is fixed but occupation can change make this assumption nontrivial. Either restrict claims to fixed-regime applications or note the extension.

## Literature Gaps

- Hu (2008) on identification with mismeasured discrete variables would tighten the identification discussion. It is foundational for the approach taken here.
- Hyslop and Imbens (2001) on classical-error measurement in regressors is a useful classical reference.
- For the "ML predictions as data" framing in the introduction, Mullainathan and Spiess (2017) is a natural addition.

These are not blocking; the existing references are adequate. They would strengthen the framing.

## Empirical-Accuracy Checks

I cross-checked all numerical claims against `paper-writer/results/tables/table_A1_performance_all_cells.csv` and `table_A2_coverage_all_cells.csv`:

- 0.936 Wald, 0.959 bootstrap for `w <= 0.7`: matches.
- 0.884 Wald, 0.906 bootstrap across all cells: matches.
- RMSE 0.029 / 0.041 / 0.092 by R for `w <= 0.7`: matches.
- 0.913 RMSE for `R = 4, rho = 0, w = 0.9`: matches.
- 0.138 / 0.142 Wald/bootstrap at `R = 4, rho = 0.9, w = 0.9`: matches.
- 0.230 / 0.242 Wald/bootstrap at `R = 10, rho = 0.9, w = 0.9`: matches.
- 0.998 / 0.995 / 0.984 convergence by R: matches.
- IRLS-MLE RMSE near-identity (0.264 vs 0.263): matches.

No numerical errors detected in v1. The naive numbers cited above are sourced from the same tables.
