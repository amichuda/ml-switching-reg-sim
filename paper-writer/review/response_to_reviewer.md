# Response to Referee Report

Paper: Correcting Machine-Learned Regime Misclassification in Switching Regressions
Draft: v2 (2026-04-28)

## Overall Response

The revision integrates the naive hard-classification baseline that was completed between v1 and v2 (request logged in `paper-writer/analysis_requests.md`, fulfilled by `paper-writer/scripts/run_naive_baseline.py` against the same Monte Carlo grid). The naive comparison is now the second numbered finding in the abstract, the centerpiece of the contribution paragraph in the introduction, the dedicated Section 5.2, and the headline of the conclusion. Section 2 adds an explicit identification subsection and defends the driver-level mixture likelihood. The IRLS-MLE near-identity finding is promoted out of a side note into Section 5.3 and Discussion item 4. Notation is reconciled, and obsolete disclaimers are removed.

## Response to Major Concerns

**Concern 1 (Out-of-date treatment of the naive baseline).**
Response: Addressed in full. The disclaimer paragraph at v1 lines 145-147 and the Draft Notes block at v1 lines 179-181 are removed. The naive baseline now appears in (i) abstract sentence 4, (ii) introduction contribution paragraph (paragraph 5 of Section 1), (iii) the new Section 5.2, (iv) Discussion item 1, and (v) the conclusion's headline-finding paragraph. Figure 5 is referenced from Section 5.1 paragraph 4 and Section 5.2. The compact `table_A2_coverage_rho_0_6.tex` and the machine summary in `paper-writer/results/tables/simulation_evidence_summary.md` already include the naive comparison.

**Concern 2 (Headline contribution understated).**
Response: Addressed. The introduction now states three numbered contributions, with the naive comparison numbers (`R = 4, w = 0.9` naive 0.001 vs corrected 0.471; `R = 10, w = 0.9` naive 0.145 vs corrected 0.483) appearing in the abstract and the contribution paragraph. The conclusion now leads with the correction-payoff finding rather than the validation-of-coverage finding.

**Concern 3 (Asymmetry of naive failure).**
Response: Addressed. Section 5.2 now devotes a dedicated paragraph to the asymmetry: at `R = 2`, naive coverage averages 0.955 versus corrected 0.960 even at `w = 0.9`; at `R = 4` and `R = 10`, the gap opens dramatically. A new closing paragraph in Section 5.2 explains the mechanism: in binary problems the modal class is correct whenever correct-class probability exceeds 0.5, while in many-regime problems the modal class is wrong on a large share of observations once correct-class probability falls below `1/R + epsilon`. Discussion item 1 reuses this framing and tells applied researchers to use soft output specifically when `R > 2`.

**Concern 4 (Identification language is loose).**
Response: Addressed. New Section 2.1 ("Identification") states three explicit identification requirements: (i) variation in the unit-level posterior weights `w_i(·)`, (ii) within-cell variation in the regime-specific covariates `x_{rt}` (correlated regimes weaken slope identification), and (iii) the interaction with regime count via the gap between top entries of `cm`. Section 4 paragraph 3 explicitly ties shock correlation to identification threat (ii). Section 5.1 paragraph 4 cross-references identification when describing the hardest cells. The phrase "the likelihood cannot manufacture identification" in v1's discussion is retained but no longer carries the analytical load — the new Section 2.1 does.

**Concern 5 (Driver-level vs observation-level likelihood is mentioned but not defended).**
Response: Addressed. Section 2 paragraph 5 now states the assumption ("regime membership is fixed within unit"), gives the likelihood implication, and flags the extension to within-unit regime switching as appropriate when targeting attributes such as occupation transitions or location transitions over a long panel. Discussion paragraph 4 reiterates this as a limitation.

## Response to Minor Concerns

1. Abstract — Updated to include the naive comparison (e.g., "naive coverage 0.001 versus corrected 0.471").
2. Section 1 paragraph 5 vs Section 4 — Tightened. v1's introduction-paragraph preview of the design is shortened; Section 4 retains the substantive design statement. Total word count has gone up, but redundancy is reduced.
3. Section 5.1 paragraph 4 cross-reference to Figure 5 — Added.
4. Section 5.2 (IRLS-MLE near-identity) promoted — Now Section 5.3 paragraph 3 with explicit framing as an unexpected finding plus a diagnostic implication. Also added as Discussion item 4.
5. Notation: `x_{r_i, t}` in the model equation, `x_{rt}` in prose — Made consistent. The model equation uses `x_{r_i, t}` because it indexes the realized true-regime shock, while the likelihood and Section 2.1 use `x_{rt}` because they range over regimes inside the mixture.
6. Likelihood made explicit — Equation (4) now writes `f(y_{it} | alpha_r + beta_r x_{rt}, sigma)`.
7. "EM-style IRLS" — Defined once at first use in Section 1; reused as "IRLS" thereafter.
8. Validation-data robustness — Section 6 paragraph 2 now explicitly labels the misspecified-confusion-matrix robustness as future work rather than a present-paper claim.
9. "Conditional optimism" — Removed from the conclusion.
10. Draft Notes block — Removed.
11. Word count — Updated to approximate 4,050.

Each minor concern is closed.

## Response to Specific Comments

- Line 9 abstract suggestion — Adopted in spirit; abstract sentence 4 reports naive coverage 0.001 vs corrected 0.471.
- Line 25 framing structure — Reused for Section 5.2.
- Line 99-101 paragraph structure — Reused as the structural template for Section 5.2's three numbered findings.
- Line 103 cross-reference — Reused: Section 5.2 explicitly contrasts the `R = 2` near-tie with the `R = 4`/`R = 10` collapse.
- Line 117 promotion — Done: now Section 5.3 paragraph 3.
- Line 145-147 deletion — Done.
- Line 179-181 deletion — Done.

## Response to Identification / Empirical Strategy Critique

The two extensions flagged (misspecified confusion matrix, within-unit regime switching) are now identified explicitly as future work in Section 6 paragraph 5, rather than buried.

## Response to Literature Gaps

Hu (2008) added to the literature placement paragraph (Section 1, last paragraph) as the foundational identification reference. Mullainathan and Spiess (2017) added in the same paragraph. Hyslop and Imbens (2001) was considered but not added; the classical-error connection is tangential to the discrete-misclassification framework here, and adding it would dilute the placement paragraph.

## Response to Empirical-Accuracy Checks

All numerical claims in v1 were independently verified against the appendix CSVs and matched. v2 retains those numbers and adds the naive numbers, all sourced from `paper-writer/results/tables/table_A2_coverage_all_cells.csv` and `table_A1_performance_all_cells.csv`.

## Changes Not Made

1. **Misspecified-confusion-matrix robustness simulation.** The referee suggested either running this or labeling it explicit future work. v2 does the latter. Running the robustness simulation requires re-running the Monte Carlo grid with perturbed confusion matrices, which is a separate scope and not justified for v2.

2. **Hyslop and Imbens (2001).** Considered but not added. Reason: the paper's framework is discrete misclassification with a known matrix-based correction, not classical measurement error. Adding the reference would dilute the placement.

3. **Reordering Section 5 so the naive comparison appears before the RMSE/bias deep dive.** The referee suggested this in the Nice-to-Have list. v2 keeps the original order: Section 5.1 reports corrected-coverage results, Section 5.2 reports the naive comparison, Section 5.3 reports RMSE and the IRLS-MLE near-identity, Section 5.4 reports computation. This places the naive comparison adjacent to the coverage result it should be read against, which matches the referee's primary recommendation, while keeping the corrected-coverage figure as the lead visual. If the editor disagrees, the order can be flipped without other content changes.

4. **One-paragraph end-of-Section-5 summary table.** Considered. Decided against because Section 6 already has a four-item numbered list that performs the same function and a summary table would be redundant.

## Outstanding Analysis Requests

`paper-writer/analysis_requests.md` shows no pending requests. The naive baseline request is now ✅ Complete. No new requests were generated by this revision.

## Quality Bar Check

The revised draft is meaningfully better than v1: (a) the naive comparison is integrated and now drives the contribution claim, (b) identification has an explicit subsection, (c) the driver-level likelihood assumption is defended, (d) the IRLS-MLE near-identity is promoted from buried to explicit, and (e) obsolete disclaimers are removed. A referee re-reviewing v2 with the same standards as v1 should recommend Minor Revision or Accept with Revisions, conditional on the misspecified-confusion-matrix simulation appearing in a follow-up rather than v2.
