# M5.3 pre-final readiness report

## Freeze result

All 12 RQ1/RQ2 configurations and all four RQ3 long-view strategies passed file-presence and checksum verification. FS1-deep LightGBM remains the selected workflow model. Neural seed 0 is primary and seeds 1–2 remain frozen robustness checkpoints.

The four RQ3 model hashes are:

| Arm | Model SHA-256 | Adapter SHA-256 |
|---|---|---|
| Normal-only baseline | `0d393cac1ea7293a93bb9e7703e0829da6d81da5cc3a1c43adfaf38c1da64414` | `5f1d2ef50e60785f056debac076994efc63253da6e1db71a148bf1db69283773` |
| Marginal reweighting | `3575b212a0b3b9f0cd7147ce4742412373f487cd1ddb522fe3d2c3be367bd4a8` | `5f1d2ef50e60785f056debac076994efc63253da6e1db71a148bf1db69283773` |
| Normal + random | `4818f17139b32666dc5a510da7c57ad169ffe48f9fc89687c6aafbba157d451c` | `75ebff11de9d4f569cf9f161c2ac367296231521a507580cf47858497a34afd4` |
| Random only | `d25b3771b9c5cb80659ad3decd6a3ebcd4e6d372bc95340d506c9333898f4496` | `813f9ad26e1937deeadafa41d53699b581596e50d23666da271aa42349462ecb` |

## Contender freeze

Using 10,000 paired unique-user normal-validation bootstraps with seed 0, the contender set is:

- `fs1_deep`
- `fs1_shallow`
- `mmoe_no_aggregates`

`fs1_shallow` has a paired-difference interval versus FS1-deep of `[-0.004125, 0.001940]` and therefore qualifies directly. `mmoe_no_aggregates` is included by the mandatory top-three rule; its interval is `[-0.012165, -0.000051]`. No held-out observation was involved.

## Frozen inference

- Primary: pair-weighted within-user long-view AUC with 0.5 tie credit.
- Secondary: equal-user AUC, 952-pair-capped AUC, pooled AUC, LogLoss, and NDCG@5.
- Main comparisons: 10,000 paired user bootstraps; two-sided percentile interval; approximate tail-share p-value.
- RQ2: 50 disjoint unique-user A/B splits per run; seed 0 primary and seeds 1–2 stability.
- RQ2 inference: 2,000 unique-user bootstrap replicates, each containing all 50 splits.
- Random-pick references average held-out-half AUC values, never probabilities.
- Exact selection ties use lexicographic candidate ID.
- Six p-values receive standard step-down Holm adjustment.

The six comparisons are full MMoE seed 0 minus FS1-deep LightGBM; full MMoE seed 0 minus shared-bottom seed 0; dense-only MMoE seed 0 minus FS3-deep LightGBM; and each of marginal reweighting, joint fusion, and random-only minus the unchanged FS1-deep baseline.

## Held-out safety

T-normal and T-random remain May 3–8, 2022. Final aggregate histories are normal-policy only and strictly prior day; sequence events are normal-policy only and strictly earlier in timestamp. The final interfaces reject prefinal outcome access, outcome-bearing feature frames, schema drift, incompatible source-indicator adapters, missing models, checksum changes, and prediction/label misalignment. No final feature or prediction artifact was materialized in M5.3.

Randomized Apr 30–May 2 outcomes and all May 3–8 outcomes remained inaccessible. No final RQ1, RQ2, or RQ3 result was calculated.

Full repository test suite: 123 passed in 6.81 seconds.
