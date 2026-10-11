# M6.2 final held-out RQ1 and RQ3 evaluation

Status: **PASS.** All frozen references and M6.1 prediction hashes were verified before May 3–8 outcomes were opened. No prediction, model, adapter, feature, candidate set, or selection decision changed. RQ2 was not executed.

## Evaluation populations

| Population/task | Impressions | Positives | Prevalence | Users | AUC-eligible users | Positive-negative pairs | Top-10% pair share |
|---|---:|---:|---:|---:|---:|---:|---:|
| T-normal long view | 93,201 | 28,945 | 31.0565% | 20,833 | 10,304 | 133,092 | 70.7390% |
| T-normal like | 93,201 | 1,539 | 1.6513% | 20,833 | 939 | 10,059 | 100.0000% |
| T-normal valid play | 71,812 | 34,831 | 48.5030% | 18,515 | 9,018 | 110,916 | 73.4907% |
| T-normal literal click | 7,898 | 712 | 9.0149% | 4,019 | 329 | 1,320 | 100.0000% |
| T-random long view | 637,492 | 56,197 | 8.8153% | 26,186 | 14,805 | 2,840,873 | 79.6813% |
| T-random like | 637,492 | 3,098 | 0.4860% | 26,186 | 1,546 | 155,773 | 100.0000% |
| T-random valid play | 634,585 | 115,952 | 18.2721% | 25,947 | 19,057 | 5,312,070 | 78.5032% |
| T-random literal click | 0 | — | — | — | — | — | — |

The top-10% definition is the ceiling of 10% of all eligible task-population users ranked by pair count, matching the frozen Week-0 definition. For the sparse like and literal-click tasks, this group includes every user with a nonzero pair count. T-random literal click is not estimable because tab 0 is absent.

## RQ1 long-view results

Neural rows use seed 0. `Pair AUC` is the primary uncapped pair-weighted within-user AUC; `Equal AUC` is the per-user mean; `Cap AUC` uses the frozen 952-pair cap.

### T-normal

| Candidate | Pair AUC | Equal AUC | Cap AUC | Pooled AUC | LogLoss | NDCG@5 |
|---|---:|---:|---:|---:|---:|---:|
| R0 | 0.631755 | 0.635356 | 0.631755 | 0.687040 | 0.571562 | 0.639709 |
| FS1 shallow | 0.652015 | 0.657807 | 0.652015 | 0.727003 | 0.548629 | 0.660297 |
| FS1 deep | 0.650272 | 0.654025 | 0.650272 | 0.732906 | 0.545235 | 0.659287 |
| FS2 shallow | 0.643495 | 0.648801 | 0.643495 | 0.757630 | 0.526317 | 0.649656 |
| FS2 deep | 0.644768 | 0.650645 | 0.644768 | 0.754669 | 0.529899 | 0.651077 |
| FS3 shallow | 0.643172 | 0.647424 | 0.643172 | 0.757379 | 0.526500 | 0.650384 |
| FS3 deep | 0.643810 | 0.648359 | 0.643810 | 0.754321 | 0.529977 | 0.649560 |
| Shared-bottom | 0.644336 | 0.651936 | 0.644336 | 0.753341 | 0.528979 | 0.650949 |
| Full MMoE | 0.643991 | 0.648805 | 0.643991 | 0.752930 | 0.530512 | 0.650434 |
| MMoE dense-only | 0.644682 | 0.646531 | 0.644682 | 0.756654 | 0.526117 | 0.648281 |
| MMoE no-history | 0.642289 | 0.648258 | 0.642289 | 0.752284 | 0.531150 | 0.648582 |
| MMoE no-aggregates | 0.642856 | 0.654164 | 0.642856 | 0.740590 | 0.540828 | 0.650959 |

### T-random

| Candidate | Pair AUC | Equal AUC | Cap AUC | Pooled AUC | LogLoss | NDCG@5 |
|---|---:|---:|---:|---:|---:|---:|
| R0 | 0.578477 | 0.566204 | 0.577191 | 0.569322 | 0.433295 | 0.297021 |
| FS1 shallow | 0.574344 | 0.563372 | 0.574196 | 0.612455 | 0.503111 | 0.281146 |
| FS1 deep | 0.576135 | 0.565366 | 0.575784 | 0.624028 | 0.507577 | 0.284442 |
| FS2 shallow | 0.598350 | 0.585825 | 0.597343 | 0.693521 | 0.483036 | 0.302353 |
| FS2 deep | 0.595867 | 0.583867 | 0.594525 | 0.689452 | 0.487830 | 0.300296 |
| FS3 shallow | 0.600146 | 0.587833 | 0.599410 | 0.694440 | 0.480544 | 0.305378 |
| FS3 deep | 0.596792 | 0.585377 | 0.595469 | 0.690132 | 0.484442 | 0.303153 |
| Shared-bottom | 0.604043 | 0.591534 | 0.603506 | 0.695490 | 0.448718 | 0.316335 |
| Full MMoE | 0.601330 | 0.590155 | 0.601212 | 0.693832 | 0.459250 | 0.312189 |
| MMoE dense-only | 0.602779 | 0.589039 | 0.601981 | 0.694181 | 0.467691 | 0.311800 |
| MMoE no-history | 0.601447 | 0.587416 | 0.600188 | 0.684524 | 0.460278 | 0.310077 |
| MMoE no-aggregates | 0.573903 | 0.566061 | 0.574257 | 0.669018 | 0.495815 | 0.279544 |

The normal-validation-selected workflow remains FS1-deep; final outcomes were not used to revise it. On held-out normal exposure, FS1-shallow has the highest point primary AUC. On randomized exposure, shared-bottom seed 0 has the highest point primary AUC among the 12 frozen candidates. These are descriptive comparisons, not a causal interpretation of the difference between exposure populations.

FS2/FS3 improve randomized-exposure primary AUC relative to FS1 but have lower T-normal primary AUC. Depth effects are small and inconsistent across feature sets. Shared-bottom exceeds full MMoE on T-random by 0.002713, while dense-only MMoE exceeds FS3-deep by 0.005987. Removing aggregates causes the clearest randomized-exposure degradation among MMoE ablations.

## Neural seed robustness

Values are pair-weighted long-view AUC for seeds 0, 1, and 2.

| Candidate | T-normal seeds 0/1/2 | T-random seeds 0/1/2 |
|---|---|---|
| Shared-bottom | 0.644336 / 0.643450 / 0.643915 | 0.604043 / 0.601602 / 0.601882 |
| Full MMoE | 0.643991 / 0.642368 / 0.643420 | 0.601330 / 0.602187 / 0.599552 |
| MMoE dense-only | 0.644682 / 0.639828 / 0.641045 | 0.602779 / 0.598431 / 0.598523 |
| MMoE no-history | 0.642289 / 0.642631 / 0.643495 | 0.601447 / 0.599347 / 0.596219 |
| MMoE no-aggregates | 0.642856 / 0.641365 / 0.639062 | 0.573903 / 0.577248 / 0.572622 |

The three-seed randomized AUC ranges are 0.00244 for shared-bottom, 0.00264 for full MMoE, 0.00435 for dense-only, 0.00523 for no-history, and 0.00463 for no-aggregates. Seed variation is reported as robustness, not used for selection.

## RQ3 on the identical T-random population

| Arm | Training rows | Pair AUC | Difference vs baseline | Equal AUC | Cap AUC | Pooled AUC | LogLoss | NDCG@5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Normal-only FS1-deep | 722,481 | 0.576135 | 0 | 0.565366 | 0.575784 | 0.624028 | 0.507577 | 0.284442 |
| Marginal reweighting | 722,481 | 0.575991 | -0.000145 | 0.565142 | 0.575724 | 0.596253 | 0.487366 | 0.271063 |
| Normal + random fusion | 1,060,414 | 0.667952 | +0.091817 | 0.665492 | 0.668471 | 0.702992 | 0.276344 | 0.396416 |
| Random-only | 337,933 | 0.668881 | +0.092746 | 0.664761 | 0.669228 | 0.694461 | 0.280162 | 0.393539 |

The marginal arm's precomputed ESS is 152,212.85 (21.07% of 722,481). The joint and random-only arms differ substantially in training population size and source composition. Their observed advantages must not be interpreted as causal IPS correction effects or attributed solely to exposure source.

## Six frozen paired comparisons

All comparisons use T-random long-view, 10,000 paired unique-user bootstrap resamples, identical multinomial multiplicities, and the uncapped primary statistic.

| Comparison (left minus right) | Observed difference | 95% percentile CI | Tail-share p | Holm p |
|---|---:|---:|---:|---:|
| Full MMoE − FS1-deep | +0.025195 | [0.022390, 0.027997] | 0 | 0 |
| Full MMoE − shared-bottom | -0.002713 | [-0.004158, -0.001263] | 0.0002 | 0.0004 |
| Dense-only MMoE − FS3-deep | +0.005987 | [0.003629, 0.008376] | 0 | 0 |
| Marginal reweighting − FS1-deep | -0.000145 | [-0.002236, 0.001917] | 0.8840 | 0.8840 |
| Joint fusion − FS1-deep | +0.091817 | [0.087470, 0.096109] | 0 | 0 |
| Random-only − FS1-deep | +0.092746 | [0.088016, 0.097389] | 0 | 0 |

A reported bootstrap p-value of zero means no opposite-sign replicate occurred among 10,000 draws; it is not an exact zero probability. The frozen tail-share calculation is descriptive and is not a rigorously null-centered bootstrap hypothesis test.

## Sensitivity, resources, and artifacts

The 952-pair cap has no visible effect on T-normal at six-decimal precision and small effects on T-random. The difference between pair-weighted and equal-user AUC confirms sensitivity to the heavy-tailed user contribution; T-random's top 10% of users supply 79.68% of long-view pairs.

- Runtime: 71.90 seconds.
- Peak RSS: 2,663.33 MiB.
- Result directory size: approximately 208 KiB.
- RQ2 cross-fitted regret: not executed.
- Model training, tuning, reselection, or prediction modification: none.

Machine-readable results are stored under `~/recom/evaluation/final/v1/`. The complete task-level metric records include long view, like, valid play, and literal click wherever estimable.
