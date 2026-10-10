# Proposed metric specification after M1.2

Status: **proposed for review, not frozen by this milestone**. This document does not modify `study_plan_v5.md`.

## Evidence from the permitted diagnostic slice

The only engagement outcomes used were randomized impressions dated April 22–29, 2022. This slice has 337,933 impressions, 20,997 users, and a long-view positive rate of 8.0480%. There are 7,678 users with both long-view classes and 1,515,626 within-user positive-negative pairs. Long view is therefore inside the frozen plan's 2%–98% feasibility range and has enough within-user pairs to retain it as the proposed primary task.

## Proposed primary metric

Use **pair-weighted within-user long-view AUC** on the eventual held-out evaluation set:

1. For each user `u`, let `P_u = n_positive,u × n_negative,u`.
2. Exclude users with `P_u = 0` from AUC; report their count separately.
3. Compare every positive-negative impression pair belonging to the same user. A correctly ordered pair contributes 1, an incorrect pair 0, and a tied prediction 0.5.
4. Let `C_u` be that user's total concordance credit. Report `sum(C_u) / sum(P_u)`.

This equals a weighted mean of per-user AUCs with weight `P_u`. It evaluates within-user ordering and never compares an impression from one user with an impression from another.

## Secondary metrics

- **Per-user-average GAUC:** mean of `C_u / P_u` over eligible users, giving each eligible user equal weight. Always report the number and share of eligible users.
- **Pooled AUC:** ordinary AUC over all impressions. It includes cross-user pairs, so it may reward user-level base-rate separation rather than feed ordering. Treat disagreement with the within-user metrics as diagnostic rather than choosing the favorable result.
- Other secondary metrics remain as specified by the frozen plan; M1.2 does not revise them.

## Proposed pair-cap sensitivity metric

The permitted slice is highly concentrated: the top 2,100 users (ceil of 10% of all slice users, ranked by pair count) contribute 92.5674% of all pairs.

For a sensitivity analysis only, propose a deterministic cap:

- Compute the 95th percentile of nonzero per-user pair counts on the Apr 22–29 randomized slice using the nearest observed value at or above the quantile (`method="higher"`).
- The resulting proposed cap is **952 pairs per user**.
- For evaluation, define `w_u = min(P_u, 952)` and report `sum(w_u × AUC_u) / sum(w_u)`.
- Do not sample pairs. This weighting is deterministic for a given set of predictions and labels.
- Keep the uncapped pair-weighted metric primary; report the capped result only as sensitivity.

On the diagnostic slice, the cap retains 1,007,994 of 1,515,626 total pair weight (66.5068%). Because this is a material change in weighting, the cap requires explicit review before it is frozen and must not be revised after held-out outcomes are inspected.
