# M5.2 RQ3 LightGBM training report

## Outcome

All three frozen RQ3 alternatives completed exactly 300 boosting rounds. Each bundle passed checksum validation and exact save/reload prediction equality. There were no LightGBM warnings, numerical errors, nonfinite predictions, or probabilities outside `[0,1]`.

| Arm | Rows | Positives / rate | Weights | Time | Peak RSS | Model size | Model SHA-256 | Reload |
|---|---:|---:|---|---:|---:|---:|---|---|
| Marginal reweighting | 722,481 | 239,795 / 0.331905 | Approved marginal | 32.30 s | 963.4 MiB | 6.94 MiB | `3575b212a0b3b9f0cd7147ce4742412373f487cd1ddb522fe3d2c3be367bd4a8` | Exact |
| Normal + random | 1,060,414 | 266,992 / 0.251781 | Uniform | 53.23 s | 1,181.1 MiB | 7.80 MiB | `4818f17139b32666dc5a510da7c57ad169ffe48f9fc89687c6aafbba157d451c` | Exact |
| Random only | 337,933 | 27,197 / 0.080480 | Uniform | 19.61 s | 614.3 MiB | 4.57 MiB | `d25b3771b9c5cb80659ad3decd6a3ebcd4e6d372bc95340d506c9333898f4496` | Exact |

Bundle sizes including adapters and manifests are approximately 7.32, 8.20, and 4.97 MiB respectively. The complete M5.2 model directory is approximately 21 MiB.

## Frozen configuration

All effective model manifests confirm:

- Binary objective
- Learning rate 0.05
- 300 trees
- 127 leaves
- Maximum depth 10
- Minimum data in leaf 100
- L1 0 and L2 1.0
- Feature and bagging fractions 1.0
- Seed, feature seed, bagging seed, and data seed 202211
- Deterministic, force-column-wise, single-thread execution

No arm used early stopping or a validation label. Matching rounds preserves the optimization-budget convention but does not equalize row counts, source distributions, compute time, or the number of examples considered during split construction.

## Input and adapter integrity

| Arm | Features | Adapter SHA-256 | Adapter policy |
|---|---:|---|---|
| Marginal reweighting | 44 | `5f1d2ef50e60785f056debac076994efc63253da6e1db71a148bf1db69283773` | Exact frozen normal baseline adapter |
| Joint | 45 | `75ebff11de9d4f569cf9f161c2ac367296231521a507580cf47858497a34afd4` | M5.1 combined-population adapter; source indicator included |
| Random only | 44 | `813f9ad26e1937deeadafa41d53699b581596e50d23666da271aa42349462ecb` | M5.1 random-only adapter; source indicator excluded |

All saved adapter hashes exactly match the M5.1 strategy manifests. The joint bundle records that later randomized-exposure scoring must use `context__is_rand=1`. The other two bundles reject the source indicator.

Vocabulary SHA-256 values are `d5e5042a13e2ba54df78b06c1629d6d06ffb2c5363f55a814d20e065e9c1bc24` (marginal), `484ac5f96e1a831c1f166758c48524ea47f896649cba8c29b723941ff7252bdf` (joint), and `df40de385993a2b3144507d80904c12bc35513aa8be17c5454b920a4b1684c4b` (random-only). Each model manifest also embeds the ordered feature names and complete dtype mapping.

The marginal arm used weight artifact SHA-256 `666016ef8884edc4e37751b0f970ef10a74c6e740a8484396998a8101b28abea`. Recomputed diagnostics remained ESS 152,212.8492 (21.0681%), maximum 13.034303, top-1% weight share 13.0346%, and top-10% share 56.4436%.

## Training diagnostics

| Arm | Prediction minimum | Prediction maximum | Warnings |
|---|---:|---:|---:|
| Marginal reweighting | 0.000238 | 0.990447 | 0 |
| Joint | 0.000501 | 0.976760 | 0 |
| Random only | 0.000325 | 0.963949 | 0 |

These are training-input sanity checks only and were not used to compare or select arms.

## Baseline and access preservation

The original FS1-deep normal-only checkpoint remained byte-identical before and after every fit:

- Model SHA-256: `0d393cac1ea7293a93bb9e7703e0829da6d81da5cc3a1c43adfaf38c1da64414`
- Adapter SHA-256: `5f1d2ef50e60785f056debac076994efc63253da6e1db71a148bf1db69283773`
- Model-manifest SHA-256: `8cff6926e62efaa225c2bc9dcdcba05902cd889106dede1fc19ae1ebb83698a3`

The M4.5 candidate manifest was not modified. No normal-validation outcome was used for tuning or early stopping. Randomized-validation and May 3–8 outcomes were not accessed. No neural RQ3 model was trained.

Machine-readable records are under `~/recom/models/lightgbm/rq3/v1/`, including `training_report.json` and one manifest/record per arm.

Full repository test suite: 117 passed.
