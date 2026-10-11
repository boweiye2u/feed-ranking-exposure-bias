# M5.3 prospective statistical-methods amendment v1

Status: **approved before held-out outcome access**.

This document resolves implementation details left unspecified by the frozen v5 plan. It does not rewrite the plan or Week-0 amendment.

1. Confidence intervals are two-sided percentile intervals using the 2.5th and 97.5th percentiles.
2. User bootstraps resample unique user IDs with replacement and represent repeated selections as multiplicities. One unique user identity can belong to only one cross-fit half.
3. Random-pick regret is the cross-fitted reference AUC minus the arithmetic mean of candidate AUCs on the held-out half. Prediction probabilities are never averaged for this estimand.
4. Exact candidate-selection ties are resolved lexicographically by frozen candidate ID.
5. Contenders are frozen using 10,000 paired user bootstraps of authorized normal validation with seed 0. A candidate is included when its paired-difference 95% interval versus `fs1_deep` contains zero. The top three normal-validation point estimates are always included.
6. Predefined comparison 3 uses `fs3_deep` as LightGBM-FS3.
7. The six confirmatory p-values use standard step-down Holm adjustment.

RQ2 uses 50 deterministic user A/B splits. Seed 0 is primary; seeds 1 and 2 are stability runs. Regret inference uses 2,000 unique-user bootstrap replicates, and each replicate applies all 50 splits. Per-user concordance credits and eligible pair counts are precomputed; bootstrap multiplicities weight these counts identically across candidates. Reference selection and reference evaluation always use disjoint unique-user halves.

The approximate two-sided bootstrap p-value remains `2 × min(share(diff ≤ 0), share(diff ≥ 0))`, capped at one. This tail-share calculation is descriptive and is not a rigorously null-centered bootstrap test.

No May 3–8 outcome was accessed in adopting or implementing these decisions.
