# Week-0 protocol amendment v1

Status: **frozen for implementation**. Approved in Milestone M1.3.

This document is a versioned amendment to `study_plan_v5.md`; it does not edit or replace that source file. Where the two differ, the decisions below govern subsequent implementation. Branch B1 is confirmed.

## Approved decisions

1. Long view remains the primary target. Uncapped pair-weighted within-user AUC remains the primary metric.
2. The deterministic 952-pair per-user cap is frozen for sensitivity analysis only. Per-user-average AUC, pooled AUC, NDCG@5, and LogLoss are secondary.
3. ESMM is excluded because 664 of 1,583 likes on the permitted randomized training slice occur with `is_click=0`.
4. `is_click` is interface-dependent. On tab 1 it is valid play in the autoplay single-column interface, not a literal click. Literal click analysis is restricted to inferred two-column tab 0 and must be reported separately. Labels with incompatible meanings must not be pooled as one click task.
5. Shipped aggregate video statistics are excluded because their temporal cutoff cannot be verified as preceding the study window.
6. Normal validation remains April 30–May 2. T-normal and T-random remain May 3–8. Test engagement outcomes are inaccessible to feature engineering, diagnostics, and selection and are opened only by the final evaluation workflow.

## Burn-in amendment

The release contains no April 8 normal rows. The amended dates are:

| Role | Exposure | Inclusive dates |
|---|---|---|
| Feature-only burn-in | Normal | Apr 9–11, 2022 |
| Normal training | Normal | Apr 12–29, 2022 |
| Random training arms and Week-0 diagnostics | Random | Apr 22–29, 2022 |
| Validation | Normal | Apr 30–May 2, 2022 |
| Embargoed random validation outcomes | Random | Apr 30–May 2, 2022 |
| Final tests | Normal and random | May 3–8, 2022 |

April 11 has 278,835 impressions, 21,110 users, 7,090 videos, and all 24 hourly buckets. On the structural key `(user_id, video_id, date, hourmin, time_ms, tab)`, 2,779 rows are duplicate excess (0.9966%), across 2,766 groups with maximum multiplicity four. April 12 has a higher structural duplicate-excess rate (2.1442%). April 11's user-level impression count has median 8, p90 31, p99 69, and maximum 237. The volume therefore reflects broad coverage and more impressions per user, not an isolated repeated block or hour spike. No blocking structural issue was found, so the preferred three-observed-day burn-in is adopted.

| Date | Normal impressions | Date | Normal impressions | Date | Normal impressions |
|---|---:|---|---:|---|---:|
| Apr 9 | 52,736 | Apr 16 | 60,904 | Apr 23 | 26,645 |
| Apr 10 | 227,808 | Apr 17 | 44,023 | Apr 24 | 18,240 |
| Apr 11 | 278,835 | Apr 18 | 24,560 | Apr 25 | 14,911 |
| Apr 12 | 166,076 | Apr 19 | 20,443 | Apr 26 | 14,530 |
| Apr 13 | 94,711 | Apr 20 | 20,851 | Apr 27 | 14,328 |
| Apr 14 | 71,252 | Apr 21 | 20,021 | Apr 28 | 13,972 |
| Apr 15 | 58,892 | Apr 22 | 22,283 | Apr 29 | 15,839 |

The reproducible, outcome-free evidence is written to `~/recom/processed/audits/m1_3_week0_structural_audit.json` by `scripts/week0_structural_audit.py`.

## Access boundary

No May 3–8 outcome may be loaded by a pre-final workflow. No final-test outcome may enter a feature, including history features. Randomized April 30–May 2 outcomes remain embargoed. Identifier and structural-field access is allowed for integrity and cold-start checks. `configs/week0_protocol.toml` and `kuairand_pure.protocol.authorize_outcome_access` are the executable specification and guard.
