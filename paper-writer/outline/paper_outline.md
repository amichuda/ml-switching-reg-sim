# Paper Outline: Correcting Machine-Learned Regime Misclassification in Switching Regressions

## Core Contribution

The paper develops and evaluates a likelihood-based switching-regression estimator for settings in which latent regime membership is observed only through a noisy machine-learning classifier. The estimator combines soft classifier predictions with a confusion matrix to form observation- or driver-level regime probabilities, uses an EM/IRLS routine for stable initialization, and then estimates the full model by maximum likelihood. The paper's contribution is finite-sample evidence about when this strategy delivers reliable point estimates, standard errors, and coverage as classification quality, regime count, and shock correlation vary.

## Target Journal / Venue

Journal of Applied Econometrics or a methods-oriented applied econometrics outlet. This first draft is simulation-only and should be framed as a methodological complement to the motivating Uganda ride-hailing application.

## Argument Arc

Empirical researchers increasingly use machine-learning predictions to assign observations to economic states, places, ethnic groups, occupations, or other latent categories. If those predicted categories are treated as true, switching-regression estimates can inherit classifier error as econometric bias and invalid inference. The paper proposes a practical correction: treat classifier output as a noisy measurement of latent regime membership and estimate the switching regression by likelihood. The Monte Carlo shows that the method performs well when classifier probabilities retain meaningful information, including under moderate misclassification and many regimes, but fails gracefully and visibly when the classifier is nearly uninformative in high-regime settings.

## Section Plan

### Abstract

- State the problem: switching regressions with ML-predicted regimes.
- State the estimator: classifier probabilities plus confusion matrix enter a likelihood; IRLS initializes MLE.
- State the evidence: 72-cell Monte Carlo, 200 replications per cell, 153,600 parameter-replication rows.
- State the result: moderate misclassification yields near-nominal coverage; near-uninformative multi-regime classifiers produce undercoverage.

### 1. Introduction

- Motivate with ML-predicted group membership in applied economics.
- Explain why predicted regimes are not ordinary covariates: classification error changes the composition of observations assigned to each regime.
- Connect to ride-hailing/weather-shock motivation but keep the paper simulation-only.
- Preview the likelihood estimator and the IRLS initializer.
- Preview main results using Figures 1 and 2.

### 2. Model

- Define panel units `i`, periods `t`, latent regimes `r`.
- Define regime-specific outcome equation `y_it = alpha_r + beta_r x_rt + epsilon_it`.
- Define soft classifier output and confusion matrix.
- Define implied regime weights and clarify the distinction between row-stochastic `P(predicted | true)` and column-normalized `P(true | predicted)`.
- Explain driver-level likelihood when regime is fixed within driver.

### 3. Estimation and Inference

- Present IRLS/EM updates conceptually.
- Present full MLE objective.
- Explain why IRLS is used for warm starts and MLE is used for standard errors.
- Explain analytical Wald intervals and score wild bootstrap intervals.

### 4. Monte Carlo Design

- Describe regimes, drivers, periods, misclassification weights, shock correlations, and true coefficients.
- Explain mapping from misclassification weight to correct-class probability.
- Emphasize the hardest cells: many regimes, high shock correlation, high misclassification.
- Refer to Table A0.

### 5. Results

- Main visual: Figure 1 coverage heatmaps.
- Main visual: Figure 2 RMSE frontier.
- Supporting visual: Figure 3 baseline bias distribution.
- Supporting visual: Figure 4 computational feasibility.
- Appendix tables provide exact table versions.

### 6. Discussion

- Practical guidance: the estimator is reliable when classifier signal remains informative.
- Warning: many-regime settings with near-uniform classifier probabilities are weakly identified.
- Mention absent naive OLS/hard-classification comparison as an extension.
- Explain how the method applies to the motivating ride-hailing setting.

### 7. Conclusion

- Restate contribution.
- Give applied recommendation: report classifier quality, use soft probabilities, use the confusion matrix, and stress-test regime count and classifier informativeness.

## Evidence Map

| Claim | Evidence | Caveat |
|---|---|---|
| The corrected MLE has good coverage under moderate misclassification. | For `w <= 0.7`, average slope coverage is 0.936 for Wald and 0.959 for score wild bootstrap. | This averages across simulated DGPs with known confusion matrices. |
| Near-uninformative classifiers undermine many-regime inference. | `R = 4`, `rho = 0.9`, `w = 0.9` has bootstrap coverage 0.142; `R = 10`, `rho = 0.9`, `w = 0.9` has bootstrap coverage 0.242. | These are extreme designs but informative stress tests. |
| IRLS is an effective initializer but not a distinct finite-sample improvement over MLE. | Mean slope RMSE is 0.263 for MLE and 0.264 for IRLS. | Current cached results compare IRLS and MLE, not naive OLS. |
| Computation is feasible but scales with regimes. | Mean MLE time rises from 0.135 seconds at `R = 2` to 6.315 seconds at `R = 10`; convergence remains 0.984 at `R = 10`. | These runtimes are machine- and implementation-specific. |

## Anticipated Reviewer Objections

1. **No naive baseline**: The current cached Monte Carlo results omit a hard-classification OLS baseline. The draft should be honest and avoid claims that require that comparison. A future revision should add it.
2. **Known confusion matrix**: The simulation treats classifier error structure as known or externally calibrated. The paper should discuss the need for validation data in applications.
3. **Extreme undercoverage at high misclassification**: This should not be hidden. It is the paper's practical warning: many-regime models require classification signal.
4. **Simulation-only scope**: The introduction should frame the paper as a methods/simulation complement to a real empirical setting, not as a standalone empirical contribution.
