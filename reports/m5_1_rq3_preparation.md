# M5.1 RQ3 preparation audit

## Result

The three authorized FS1-deep long-view training strategies were prepared successfully without training a new model. All feature, label, weight, and adapter inputs passed exact-ID, checksum, schema, source, and date-window checks.

## Frozen baseline

- Bundle: `~/recom/models/lightgbm/v1/fs1/deep/long_view/`
- Training: 722,481 normal-policy rows, April 12–29, 2022
- Model: FS1-deep long-view LightGBM, 300 trees, seed 202211
- Model SHA-256: `0d393cac1ea7293a93bb9e7703e0829da6d81da5cc3a1c43adfaf38c1da64414`
- Adapter SHA-256: `5f1d2ef50e60785f056debac076994efc63253da6e1db71a148bf1db69283773`
- Feature-manifest SHA-256: `f613933fde572aefb5ada5e3c99d078c576da90103bbfe79a4e3c920631b408f`
- M4.5 selection metric: 0.6555485966471009 normal-validation long-view pair-weighted AUC

The model and its adapter were loaded with checksum verification and were neither modified nor retrained. The candidate manifest was not changed.

## Populations and class balance

| Population | Dates | Rows | Long-view positives | Prevalence |
|---|---|---:|---:|---:|
| Normal | Apr 12–29 | 722,481 | 239,795 | 0.331905 |
| Random | Apr 22–29 | 337,933 | 27,197 | 0.080480 |
| Joint total | Combined windows | 1,060,414 | 266,992 | 0.251781 |

Random rows are 31.8680% of the joint population. The released randomized slice is materially smaller than the study plan's explicitly rough estimate; the observed audited count, 337,933, governs implementation.

All normal rows have `context__is_rand=0`; all random rows have `context__is_rand=1`. Impression IDs are unique both within and across sources.

## Marginal-exposure weights

The approved rule was recomputed independently on 722,481 normal-training rows and 7,293 observed videos.

- Additive smoothing: 1
- Raw weight: `1 / (n_i + 1)`
- Raw 99th-percentile clip: 0.1111111111
- Frozen post-clipping normalization mean: 0.0085245150
- Normalized mean: exactly 1 within floating-point precision
- Normalized minimum: 0.02048345
- Normalized maximum and unseen-video fallback: 13.03430302
- ESS: 152,212.8492, or 21.0681% of rows
- Top 1% of rows by weight: 13.0346% of total weight
- Top 10% of rows by weight: 56.4436% of total weight

| Quantile | Normalized weight |
|---:|---:|
| 0% | 0.020483 |
| 25% | 0.156203 |
| 50% | 0.343008 |
| 75% | 0.909370 |
| 90% | 2.346175 |
| 95% | 4.344768 |
| 99% | 13.034303 |
| 100% | 13.034303 |

The expected maximum (about 13.0343) and ESS (about 152,212.85) were reproduced without a meaningful discrepancy. These weights are explicitly labeled marginal-exposure reweighting, not causal IPS.

## Adapter audit

| Strategy | Features | Categorical fields | Total fitted category tokens | Vocabulary population |
|---|---:|---:|---:|---|
| Marginal reweighting | 44 | 35 | 15,601 | Frozen normal-only baseline |
| Joint | 45 | 36 | 16,063 | Normal + random training |
| Random only | 44 | 35 | 15,833 | Random training only |

The joint model's only additional feature is the already permitted categorical `context__is_rand`, whose vocabulary contains values 0 and 1. It is excluded from the normal-only weighting adapter and random-only adapter. Random-only does not reuse the normal vocabulary. Reloaded adapters produce deterministic encoded schemas, and the common loader returned exactly 722,481, 1,060,414, and 337,933 aligned examples respectively. The weighting loader also returned one positive finite sample weight per normal-training row.

All three strategies are bound to the same FS1-deep configuration: learning rate 0.05, 300 rounds, 127 leaves, maximum depth 10, minimum leaf size 100, L2 1.0, seed 202211, deterministic single-thread execution. Equal boosting rounds do not remove sample-size or distribution differences.

## Artifacts and resources

- Root: `~/recom/processed/rq3/v1/`
- Machine audit: `~/recom/processed/rq3/v1/preparation_audit.json`
- Marginal features, labels, weights, and weighter: approximately 27 MiB
- Joint features, labels, and adapter: approximately 30 MiB
- Random-only features, labels, and adapter: approximately 9.3 MiB
- Total: approximately 66 MiB
- Preparation runtime: 50.17 seconds
- Peak RSS: 2,871.9 MiB
- Full repository test suite: 115 passed.

No random-validation or May 3–8 outcomes were accessed. No randomized outcomes were used as feature-history sources. No production model was trained, and no validation or target-test metric was calculated.
