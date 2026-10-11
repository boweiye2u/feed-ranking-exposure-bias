# Portfolio descriptions

These descriptions summarize the completed offline study without claiming deployment, online experimentation, or causal policy improvement.

## Two-sentence overview

Built a leakage-audited recommendation research pipeline on KuaiRand-Pure to test how popularity, LightGBM, shared-bottom, and MMoE rankers generalize from normal-policy logs to randomized exposure. Froze chronological features, candidates, and user-level inference before final evaluation, then quantified model-selection regret and compared three exposure-aware training strategies against a normal-only baseline.

## Resume bullet alternatives

### Applied Scientist

- Designed a frozen, user-level evaluation of 12 recommendation rankers under normal and randomized exposure; measured +0.027714 AUC cross-fitted selection regret across all candidates and showed that the narrower contender-only interval included zero.

### Machine Learning Engineer

- Implemented an end-to-end, leakage-safe ranking pipeline with point-in-time FS1–FS3 features, LightGBM and PyTorch multi-task models, strict artifact manifests, deterministic training, checksum validation, and 140 automated tests.

### Research Scientist

- Conducted a preregistered-style offline study of exposure bias and model selection on KuaiRand-Pure, using disjoint-user cross-fitting, paired user bootstraps, prespecified comparisons, and normal versus randomized held-out evaluation.

## 30-second interview explanation

Most recommender evaluations use data selected by the previous ranking policy, so the apparent best model can be the one that best matches historical exposure. I built and froze 12 ranking workflows on KuaiRand-Pure, selected models using normal-policy validation, and evaluated them on both normal and randomized impressions. Across all candidates, normal-policy selection incurred 0.027714 AUC of cross-fitted regret on randomized exposure, while the smaller prespecified contender set was inconclusive. Randomized training examples substantially improved randomized-test ranking, but a simple marginal-frequency weighting heuristic did not.

## Two-minute technical interview explanation

The project asks whether an offline ranking workflow selected on ordinary recommender logs still ranks well when exposure is less policy-driven. I used KuaiRand-Pure because it provides normal-policy and randomized-exposure impressions. The chronology was frozen before final evaluation: Apr 9–11 for feature burn-in, Apr 12–29 for normal training, Apr 22–29 for randomized RQ3 training, Apr 30–May 2 for normal validation, and May 3–8 for the two held-out test populations.

The implementation has three nested point-in-time feature sets. FS1 contains released static and context fields; FS2 adds user/video aggregates from strictly earlier normal-policy days; FS3 adds user–category affinity and recency. Neural models additionally use a masked-mean last-20 history containing only strictly earlier normal-policy events. The candidate set contains popularity, six LightGBM variants, shared-bottom, full MMoE, and three frozen MMoE ablations. Loaders enforce exact impression-ID alignment, training-fitted categorical vocabularies, temporal cutoffs, and outcome embargoes.

The primary metric is pair-weighted within-user long-view AUC. For model-selection regret, candidate selection and evaluation use disjoint user halves, with paired user-level bootstrap inference. The normal-validation winner, FS1-deep LightGBM, scored 0.576135 on randomized test impressions. Joint normal-plus-random and random-only LightGBM reached 0.667952 and 0.668881 respectively, while marginal-frequency reweighting reached 0.575991. Those differences show the value of randomized training examples in this offline setting, but they are not causal estimates or evidence of online uplift.

## Quantitative results and boundaries

- Twelve frozen ranking configurations were evaluated under both normal and randomized exposure.
- All-candidate cross-fitted selection regret: **+0.027714 AUC**, 95% percentile interval **[0.024180, 0.030318]**.
- Three-contender regret: **-0.000527**, interval **[-0.001580, 0]**; this is an important inconclusive result.
- T-random pair-weighted AUC: normal-only **0.576135**, marginal weighting **0.575991**, joint **0.667952**, random-only **0.668881**.
- The results are offline. Randomized insertion does not randomize every contextual variable, propensities are unavailable, and the RQ3 arms differ in sample size and source distribution.

## GitHub subtitle

Leakage-safe ranking research on KuaiRand-Pure: exposure shift, cross-fitted selection regret, LightGBM, and multi-task MMoE.
