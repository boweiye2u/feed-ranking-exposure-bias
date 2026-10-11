# M6.3 final cross-fitted model-selection regret

Status: **PASS.** RQ2 used the 12 frozen seed-0 candidates, unchanged T-random predictions, the fixed normal-validation workflow pick `fs1_deep`, and the frozen three-candidate contender set. RQ3 alternatives were excluded. No prediction was regenerated and no model was retrained, tuned, or reselected.

## Population and estimator

- T-random long-view impressions: 637,492.
- Unique users: 26,186.
- Users with positive-negative pairs: 14,805.
- Zero-pair users retained in population/bootstrap diagnostics: 11,381.
- Total positive-negative pairs: 2,840,873.
- Cross-fitting: 50 deterministic user A/B splits, evaluated in both directions (100 selection/evaluation halves).
- Bootstrap: 2,000 paired unique-user multinomial replicates, seed 0, selection recomputed in every replicate.
- Primary statistic: uncapped pair-weighted within-user AUC with 0.5 tie credit.
- Exact candidate ties: lexicographic candidate ID.
- Random-pick reference: arithmetic mean of candidate AUC values on the evaluation half, not an ensemble prediction.

## Regret estimates

| Reference population | Candidates | Cross-fitted reference AUC | FS1-deep evaluation AUC | Normal-selection regret (95% CI) | Random-pick regret (95% CI) |
|---|---:|---:|---:|---:|---:|
| All frozen candidates | 12 | 0.603866 | 0.576152 | +0.027714 [0.024180, 0.030318] | +0.011881 [0.009673, 0.013311] |
| Frozen contenders | 3 | 0.575625 | 0.576152 | -0.000527 [-0.001580, 0.000000] | +0.000815 [-0.000817, 0.002402] |

The all-candidate analysis indicates that normal-validation selection loses randomized-exposure ranking performance relative to the cross-fitted randomized reference. The contender analysis does not show the same loss: its interval includes zero and the point estimate slightly favors frozen FS1-deep. This contrast is expected because the contender set was frozen from normal-validation uncertainty and excludes several configurations that perform substantially better under randomized exposure.

Cross-fitted reference AUC is an average of evaluation-half AUCs after selection on the disjoint half. It is not the retrospectively highest full-sample T-random AUC, an oracle, or evidence of an optimal online recommendation policy. The cross-fitted average FS1-deep AUC can also differ slightly from its full-population AUC because it averages ratios over held-out halves.

## Reference-selection frequencies

### All 12 candidates

| Candidate | Selected halves |
|---|---:|
| Shared-bottom | 93 |
| MMoE dense-only | 7 |
| Every other candidate | 0 |

### Frozen contenders

| Candidate | Selected halves |
|---|---:|
| FS1-deep | 90 |
| MMoE no-aggregates | 10 |
| FS1-shallow | 0 |

## Benchmark and reproducibility

The exact production algorithm was first benchmarked on 20 deterministic bootstrap replicates. It took 0.212 seconds and projected 21.18 seconds for 2,000 replicates, well below the two-hour stop threshold. The complete bootstrap itself took 2.87 seconds; total execution, including frozen-input verification, label loading, per-user statistics, and artifact writing, took 20.43 seconds. Peak recorded RSS was 2,669.25 MiB.

Synthetic tests compare the vectorized implementation against the slower frozen estimator and cover paired multiplicities, zero-pair users, score ties, disjoint and uneven user halves, deterministic splits, lexicographic ties, and random-pick arithmetic. The implementation uses exact per-user concordance and pair arrays and does not approximate the primary metric or remove users.

## Limitations

- RQ2 measures exposure-dependent offline selection behavior, not a causal policy effect.
- The randomized reference is trained on normal-policy data except where the frozen candidate architecture itself dictates otherwise; it is a cross-fitted evaluation construct, not an online policy.
- The all-candidate and contender estimates answer different predeclared questions and should not be substituted for one another after seeing results.
- Heavy pair-count concentration means the primary estimand emphasizes high-activity users; capped and equal-user metrics from M6.2 remain relevant sensitivity analyses.
- Only one held-out calendar window is evaluated, and neural seed 0 is primary by prior decision.

Machine-readable results, per-user sufficient statistics, selection frequencies, and the reproducibility manifest are stored under `~/recom/evaluation/final/v1/`.
