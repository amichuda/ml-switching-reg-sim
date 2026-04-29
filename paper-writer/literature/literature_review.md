# Literature Review: Switching Regressions with Machine-Learned Misclassification

Generated: 2026-04-27

## Search Strategy

This literature note supports a simulation-only methods paper on estimating switching regressions when regime membership is not observed directly but is predicted by a machine-learning classifier. The search scope combines four literatures: switching and mixture regression, misclassification in econometric regressions, EM and maximum-likelihood estimation with latent data, and the use of machine-learning predictions as inputs to empirical economic models.

The local repository already contains papers on the motivating Uganda ride-hailing application and related likelihood methods. Web search was used to verify core bibliographic facts for several canonical references. Semantic Scholar search through the local helper timed out during this pass, so citation counts and complete DOI metadata should be checked before final submission.

## Key Sources

**Quandt, Richard E. 1972. "A New Approach to Estimating Switching Regressions." Journal of the American Statistical Association.**
- Summary: Introduces switching-regression estimation as a way to model observations drawn from distinct latent regimes. The paper is a canonical precursor to finite-mixture and regime-switching regression methods.
- Relevance: Provides the econometric ancestry for the paper's model: the conditional mean function differs by unobserved regime.

**Goldfeld, Stephen M., and Richard E. Quandt. 1973. "A Markov Model for Switching Regressions." Journal of Econometrics, 1, 3-15.**
- Summary: Extends switching regressions to Markov regime dynamics. This is a foundational paper for likelihood-based regime-switching models.
- Relevance: Useful for positioning this paper against dynamic regime-switching models. The present paper differs because regime is fixed within driver in the Monte Carlo design, and the key information source is an external classifier rather than a Markov transition law.

**Hamilton, James D. 1989. "A New Approach to the Economic Analysis of Nonstationary Time Series and the Business Cycle." Econometrica.**
- Summary: Develops the Markov-switching time-series model that became central in macroeconometrics.
- Relevance: Shows the breadth of likelihood-based regime-switching methods. The present setting is cross-sectional/panel switching with classifier-produced regime probabilities, not time-series regime transitions.

**Dempster, A. P., N. M. Laird, and D. B. Rubin. 1977. "Maximum Likelihood from Incomplete Data via the EM Algorithm." Journal of the Royal Statistical Society, Series B, 39(1), 1-38.**
- Summary: Formalizes the EM algorithm for maximum-likelihood estimation with latent or incomplete data.
- Relevance: The IRLS routine used here can be interpreted as an EM-style weighted least-squares algorithm: posterior regime weights are updated, then regime-specific regression parameters are updated conditionally on those weights.

**Aigner, Dennis J. 1973. "Regression with a Binary Independent Variable Subject to Errors of Observation." Journal of Econometrics.**
- Summary: A foundational econometric treatment of misclassification in a binary regressor.
- Relevance: Establishes the basic point that mismeasured discrete regressors can generate biased estimates and require explicit correction.

**Bollinger, Christopher R. 1996. "Bounding Mean Regressions When a Binary Regressor Is Mismeasured." Journal of Econometrics.**
- Summary: Develops partial-identification bounds for regression models with a misclassified binary regressor.
- Relevance: Provides a contrast to this paper's point-identification approach. Here, the confusion matrix and classifier probabilities supply enough structure to write a likelihood for the latent-regime model.

**Mahajan, Aprajit. 2006. "Identification and Estimation of Regression Models with Misclassification." Econometrica.**
- Summary: Studies identification and estimation in regression models with misclassified regressors.
- Relevance: Central reference for the econometrics of misclassification. The present paper differs by treating regime membership as latent switching-regression membership and by using soft machine-learning output plus an estimated or calibrated confusion matrix.

**Lewbel, Arthur. 2007. "Estimation of Average Treatment Effects with Misclassification." Econometrica.**
- Summary: Studies treatment-effect estimation with misclassified treatment status.
- Relevance: Helps position the paper in the broader literature on correcting misclassified discrete states.

**Athey, Susan, and Guido W. Imbens. 2019. "Machine Learning Methods That Economists Should Know About." Annual Review of Economics, 11, 685-725.**
- Summary: Reviews machine-learning methods and their uses in economics.
- Relevance: Provides context for using ML-generated objects inside empirical workflows. This paper focuses on a downstream inference problem that arises after an ML classifier is used to predict economic regimes.

**Chernozhukov, Victor, Denis Chetverikov, Mert Demirer, Esther Duflo, Christian Hansen, Whitney Newey, and James Robins. 2018. "Double/Debiased Machine Learning for Treatment and Structural Parameters." Econometrics Journal.**
- Summary: Develops orthogonal/debiased methods for using high-dimensional or machine-learning nuisance estimates in structural/econometric models.
- Relevance: The present paper is not a DML paper, but it belongs to the same broad agenda of making post-ML econometric inference valid. The key nuisance object here is a classifier-based regime probability rather than a conditional mean or propensity score.

**Imai, Kosuke, and Kabir Khanna. 2016. "Improving Ecological Inference by Predicting Individual Ethnicity from Voter Registration Records." Political Analysis, 24(2), 263-272.**
- Summary: Demonstrates how names and voter records can be used to predict individual ethnicity and improve aggregate inference.
- Relevance: A useful example of classifier-predicted latent demographics entering social-science analysis. It motivates why classifier error should be modeled rather than ignored.

**Michuda, Aleksandr. Uganda ride-hailing application paper, local file `literature/michuda-uganda-uber-jde-reduced.pdf`.**
- Summary: Provides the motivating application for heterogeneous responses to weather shocks in ride-hailing labor supply.
- Relevance: The present paper is framed as a methodological complement to this empirical setting. The simulation design mirrors features of that application: driver panels, region-specific shocks, and classifier-based region assignment.

**Aronsson. 2024. "A Maximum Likelihood Bunching Estimator of the Elasticity of Taxable Income." Journal of Applied Econometrics.**
- Summary: Uses maximum-likelihood structure to solve a measurement/inference problem in applied public finance.
- Relevance: Not a switching-regression paper, but useful as a recent JAE example of an applied econometric likelihood estimator motivated by a concrete empirical setting.

## Synthesis

### Core Findings

The switching-regression literature provides the likelihood framework for models in which coefficients vary across latent regimes. The misclassification literature shows that treating an error-prone discrete state as observed can induce bias and invalid inference. The EM literature provides a natural algorithmic interpretation for weighted-regression updates in latent-regime models. The ML-in-economics literature explains why economists increasingly use predicted labels or probabilities, but it also implies a downstream need to account for prediction error.

### Active Debates

The relevant debate is less about whether misclassification matters and more about how much structure is needed to correct it. Bounds approaches require fewer assumptions but deliver intervals. Likelihood approaches can deliver point estimates and standard errors, but they require the econometrician to specify how classifier outputs map into latent states. This paper takes the likelihood route and studies the finite-sample behavior of that choice.

### Methodological Landscape

The paper fits into a methods niche between classical switching regressions and modern post-ML empirical work. Instead of estimating regimes as entirely latent states, it treats a classifier as an informative but noisy measurement device. The core object is the conditional regime probability implied by soft classifier predictions and a confusion matrix. Given those probabilities, the model can be estimated by EM/IRLS for initialization and by full maximum likelihood for inference.

### Gaps and Open Questions

The main gap is finite-sample inference when classifier quality varies, especially with many regimes and correlated regime-specific shocks. Existing theory tells us that misclassification can bias regression estimates, but applied researchers need practical guidance about when a classifier-informed likelihood is well behaved, when coverage is reliable, and when classification is too weak to support high-dimensional regime heterogeneity. The Monte Carlo evidence in this project is designed to answer that practical question.

## Citation List for Drafting

- Aigner, Dennis J. 1973. "Regression with a Binary Independent Variable Subject to Errors of Observation." Journal of Econometrics.
- Athey, Susan, and Guido W. Imbens. 2019. "Machine Learning Methods That Economists Should Know About." Annual Review of Economics, 11, 685-725.
- Bollinger, Christopher R. 1996. "Bounding Mean Regressions When a Binary Regressor Is Mismeasured." Journal of Econometrics.
- Chernozhukov, Victor, Denis Chetverikov, Mert Demirer, Esther Duflo, Christian Hansen, Whitney Newey, and James Robins. 2018. "Double/Debiased Machine Learning for Treatment and Structural Parameters." Econometrics Journal.
- Dempster, A. P., N. M. Laird, and D. B. Rubin. 1977. "Maximum Likelihood from Incomplete Data via the EM Algorithm." Journal of the Royal Statistical Society, Series B, 39(1), 1-38.
- Goldfeld, Stephen M., and Richard E. Quandt. 1973. "A Markov Model for Switching Regressions." Journal of Econometrics, 1, 3-15.
- Hamilton, James D. 1989. "A New Approach to the Economic Analysis of Nonstationary Time Series and the Business Cycle." Econometrica.
- Imai, Kosuke, and Kabir Khanna. 2016. "Improving Ecological Inference by Predicting Individual Ethnicity from Voter Registration Records." Political Analysis, 24(2), 263-272.
- Lewbel, Arthur. 2007. "Estimation of Average Treatment Effects with Misclassification." Econometrica.
- Mahajan, Aprajit. 2006. "Identification and Estimation of Regression Models with Misclassification." Econometrica.
- Quandt, Richard E. 1972. "A New Approach to Estimating Switching Regressions." Journal of the American Statistical Association.

## Follow-Up Before Submission

The final paper should verify full bibliographic metadata, DOIs, and citation counts. Semantic Scholar access through the local helper timed out during this drafting pass, so this file should be treated as a structured drafting library rather than a final bibliography.
