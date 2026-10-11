# M6.1 held-out feature and prediction quality report

Status: **PASS.** The frozen M5 preflight, feature construction, model scoring, and independent integration audit completed without opening target-label tables or computing a final performance statistic.

## Preflight

HEAD remains `c936a93cf2c81efb95840a1e8a52e42437f8fb54`, the commit referenced by `m5-prefinal-v1`. The `week0-protocol-v1`, `m2-features-v1`, and `m4-candidates-v1` tags also resolve to their frozen commits. All protocol, feature-manifest, evaluation-code/config, candidate-manifest, candidate-bundle, and RQ3-bundle checksums match the M5 readiness manifest. No model, training adapter, vocabulary, preprocessing object, contender, or selection decision was changed.

## Held-out features

| Population | Rows | FS1 SHA-256 | FS2 SHA-256 | FS3 SHA-256 | Sequence SHA-256 |
|---|---:|---|---|---|---|
| T-normal | 93,201 | `c7f3f64726643a1c9e5fdf05ad8945b470c13aa27085e13edf24cde3bf75d539` | `a9d4cff02e720b47bb9225c00a3cffcd8d748173e6b283e72ce1d41fd61a3d1b` | `85663ca8e9a66088cbb1d32052d81c62d380a0c588addc8c7ea0d955157b755f` | `cecf5c6d62583d4498e9903e65817cbda55cab3d448e96f35dd5f91279c57142` |
| T-random | 637,492 | `15d121eeef28ccdbeae5efc471dc176ede8aebc4874b6ae5de5f0b60375f415a` | `e39b358585c818c3c6a4b6b7c038cbb92c531bc62f80251e1452de0dd7a3c009` | `5b875b86c1c08407757da613fad7d5e212ba2e276e495669712665584da3857c` | `c4a237138757ffbc3d73def71bb1abd5922afcba8acc637c98b6257e13091f79` |

All three feature stages have the frozen training schema, unique non-null ascending impression IDs, exact FS1-to-FS2-to-FS3 nesting, and no outcome columns. FS3 sequences have exactly 20 positions. Mask sums equal recorded lengths and every valid timestamp is strictly earlier than its target. T-normal has mean history length 18.8945, 207 empty histories, and 3,400 deterministic tie-boundary drops; T-random has mean 17.8002, 3,566 empty histories, and 24,880 tie-boundary drops.

Aggregate features use normal-policy outcomes from April 9 through May 7. Within the held-out interval, May 3–7 normal outcomes are accessed only as strictly prior-day feature history: day d cannot read day d or later. May 8 outcomes are not opened. Sequence construction reads only structural normal-policy fields through May 8 and applies a strictly earlier timestamp cutoff. No randomized outcome is used as history, and randomized-validation outcomes remain unopened.

## Frozen predictions

There are 146 physical prediction artifacts, 73 per exposure population, plus two checksum references that reuse the identical FS1-deep long-view prediction for the RQ3 normal-only baseline.

| Family per population | Artifacts |
|---|---:|
| R0 | 1 |
| LightGBM RQ1/RQ2 | 24 |
| Neural, five candidates × three seeds × three tasks | 45 |
| RQ3 alternatives | 3 |

| Population | Long-view eligible rows | Like eligible rows | Valid-play eligible rows | Literal-click eligible rows |
|---|---:|---:|---:|---:|
| T-normal | 93,201 | 93,201 | 71,812 | 7,898 |
| T-random | 637,492 | 637,492 | 634,585 | 0 |

T-random contains no tab-0 impressions, so each literal-click LightGBM artifact is an intentionally empty, schema-valid table. Neural models do not have a literal-click head. Every nonempty score is finite and within `[0,1]`; every artifact contains only `impression_id` and `score`, matches its structural task mask exactly, and has a verified SHA-256 checksum. The joint RQ3 adapter receives `context__is_rand=1` for T-random and 0 for T-normal. No categorical vocabulary or preprocessor was fitted on held-out rows.

## Resources and storage

- Feature construction: 33.39 seconds.
- Frozen scoring: 614.29 seconds.
- Independent integration audit: 5.34 seconds.
- Recorded process peak RSS: 4,493.39 MiB during the combined feature/scoring process.
- Feature artifacts: approximately 92 MiB under `~/recom/processed/final/v1/`.
- Prediction artifacts: approximately 287 MiB under `~/recom/predictions/final/v1/`.

## Verification and restrictions

The production gate verified frozen schemas, feature nesting, exact impression alignment, task subsets, sequence masks and chronology, checksums, model/adapter compatibility, finite probability ranges, and RQ3 baseline references. No target-label artifact was materialized, no engagement prevalence or pair count was calculated, no AUC/LogLoss/NDCG/bootstrap/regret routine ran, and no model was trained, tuned, or selected.

The complete repository suite passes with 128 tests. Machine-readable records are `feature_build_report.json`, `prediction_report.json`, `integration_audit.json`, and `m6_1_manifest.json` in the large-artifact roots documented above.

The only structural limitation is that literal-click final evaluation is infeasible on T-random because that population has no tab-0 impressions. This does not affect the frozen primary long-view analysis or tab-1 valid-play predictions.
