# Frozen metric specification

Version: Week-0 v1.

## Primary: pair-weighted within-user AUC

For each user `u`, let `n+_u` and `n-_u` be positive and negative long-view counts and `P_u = n+_u × n-_u`. Users with `P_u = 0` are excluded from every within-user AUC aggregate and their count and share are reported.

For every positive-negative pair within a user, assign concordance credit 1 when the positive score is higher, 0 when lower, and 0.5 on a score tie. Let `C_u` be summed credit. The primary value is:

`sum_u C_u / sum_u P_u`.

This is an uncapped pair-weighted average of user AUCs. Report eligible users, excluded single-class users, and total pair weight.

## Frozen capped-weight sensitivity

The cap was selected once from the permitted Apr 22–29 randomized training slice as the higher-observation 95th percentile of nonzero `P_u`, yielding 952. It is never re-estimated on validation or test data.

Define `w_u = min(P_u, 952)` and report `sum_u(w_u × C_u/P_u) / sum_u w_u`. This deterministic statistic is sensitivity-only; no pairs are sampled and it does not replace the primary metric.

## Secondary metrics

- **Per-user-average AUC:** `mean_u(C_u/P_u)` over eligible users, with each eligible user weighted equally.
- **Pooled AUC:** ordinary impression-level ROC AUC. It includes cross-user comparisons and is not a substitute for the primary metric.
- **NDCG@5:** within each user, order impressions by descending prediction. Resolve score ties deterministically by ascending `time_ms`, then `video_id`, then stable input row index. Use binary relevance and gain `2^rel - 1`. Compute DCG over the first five positions, divide by that user's ideal DCG, and average over users with at least five impressions and at least one positive. Report excluded counts. Interface-specific secondary tasks retain their task mask.
- **LogLoss:** pooled binary cross-entropy over eligible impressions after clipping probabilities to `[1e-15, 1-1e-15]`. Report the number of impressions. Interface-specific tasks use only their declared tab mask.

Every metric report identifies target, exposure population, date window, interface mask, model seed, and eligible denominator. Metrics are not selected post hoc based on favorable direction.
