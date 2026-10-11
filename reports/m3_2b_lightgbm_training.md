# M3.2b LightGBM training quality report

Status: **PASS — 24/24 task predictors completed.** No validation/test outcomes were accessed, no model-selection metric was computed, and no configuration failed.

## Benchmark gate

The exact FS3/shallow/long-view production workload used 722,481 rows and 83 features. Vocabulary fitting took 8.41 s, encoding 10.69 s, and training 46.02 s. Peak process RSS after training was 1,652.8 MiB, below the prespecified 12,288 MiB gate; training was below the 900 s gate. The model completed 300 rounds, emitted no warnings, produced finite probabilities from 0.000878 to 0.949437, preserved exact impression alignment, and reproduced predictions exactly after serialization/reload.

## Full matrix

All entries completed 300 rounds, passed exact serialization/reload equivalence and alignment checks, emitted zero captured warnings/numerical errors, and produced probabilities in [0,1]. `Bundle MiB` covers the three serialized loading inputs (`model.txt`, `adapter.json`, and `model_manifest.json`); the small diagnostic record is reported separately in directory totals. Peak RSS is cumulative process high-water memory at that point, not incremental per-model allocation.

| FS | Depth | Task | Rows | Positives | Positive % | Train s | Bundle MiB | Peak RSS MiB | Features | Warnings |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| FS1 | shallow | long_view | 722,481 | 239,795 | 33.190 | 26.57 | 2.97 | 1,049.2 | 44 | 0 |
| FS1 | shallow | like | 722,481 | 13,706 | 1.897 | 26.30 | 2.87 | 1,071.6 | 44 | 0 |
| FS1 | shallow | valid_play | 535,569 | 279,124 | 52.117 | 19.45 | 2.81 | 1,114.4 | 44 | 0 |
| FS1 | shallow | literal_click | 78,575 | 6,568 | 8.359 | 3.29 | 2.21 | 1,114.4 | 44 | 0 |
| FS1 | deep | long_view | 722,481 | 239,795 | 33.190 | 33.78 | 7.70 | 1,114.4 | 44 | 0 |
| FS1 | deep | like | 722,481 | 13,706 | 1.897 | 31.95 | 6.42 | 1,114.4 | 44 | 0 |
| FS1 | deep | valid_play | 535,569 | 279,124 | 52.117 | 25.00 | 7.08 | 1,136.0 | 44 | 0 |
| FS1 | deep | literal_click | 78,575 | 6,568 | 8.359 | 4.57 | 4.75 | 1,136.0 | 44 | 0 |
| FS2 | shallow | long_view | 722,481 | 239,795 | 33.190 | 39.68 | 3.06 | 1,518.8 | 66 | 0 |
| FS2 | shallow | like | 722,481 | 13,706 | 1.897 | 36.65 | 3.04 | 1,518.8 | 66 | 0 |
| FS2 | shallow | valid_play | 535,569 | 279,124 | 52.117 | 29.00 | 2.95 | 1,519.8 | 66 | 0 |
| FS2 | shallow | literal_click | 78,575 | 6,568 | 8.359 | 4.76 | 2.35 | 1,519.8 | 66 | 0 |
| FS2 | deep | long_view | 722,481 | 239,795 | 33.190 | 48.99 | 8.42 | 1,519.8 | 66 | 0 |
| FS2 | deep | like | 722,481 | 13,706 | 1.897 | 44.67 | 7.38 | 1,519.8 | 66 | 0 |
| FS2 | deep | valid_play | 535,569 | 279,124 | 52.117 | 36.28 | 7.78 | 1,536.9 | 66 | 0 |
| FS2 | deep | literal_click | 78,575 | 6,568 | 8.359 | 6.82 | 5.34 | 1,536.9 | 66 | 0 |
| FS3 | shallow | long_view | 722,481 | 239,795 | 33.190 | 46.02 | 3.07 | 1,652.8 | 83 | 0 |
| FS3 | shallow | like | 722,481 | 13,706 | 1.897 | 42.28 | 3.05 | 1,818.5 | 83 | 0 |
| FS3 | shallow | valid_play | 535,569 | 279,124 | 52.117 | 33.64 | 2.98 | 1,856.7 | 83 | 0 |
| FS3 | shallow | literal_click | 78,575 | 6,568 | 8.359 | 5.54 | 2.35 | 1,856.7 | 83 | 0 |
| FS3 | deep | long_view | 722,481 | 239,795 | 33.190 | 56.44 | 8.47 | 1,856.7 | 83 | 0 |
| FS3 | deep | like | 722,481 | 13,706 | 1.897 | 51.34 | 7.49 | 1,856.7 | 83 | 0 |
| FS3 | deep | valid_play | 535,569 | 279,124 | 52.117 | 41.82 | 7.90 | 1,858.3 | 83 | 0 |
| FS3 | deep | literal_click | 78,575 | 6,568 | 8.359 | 7.86 | 5.36 | 1,858.3 | 83 | 0 |

Matrix wall time was 1,209.29 s (20.15 min), including three feature-set loads, vocabulary construction, encoding, predictions, serialization, reload, and exact comparison. Peak process RSS was 1,858.3 MiB. The 24 serialized loading bundles total 123,512,733 bytes (117.79 MiB); all files in the v1 artifact directory, including reports and training records, total 123,560,725 bytes (117.84 MiB).

## Guard results

- Normal-policy April 12–29 data only; validation, randomized-training, and test artifacts were not loaded.
- Feature/label impression IDs matched position by position before task selection.
- Training-only categorical vocabularies and the frozen M2 feature-manifest checksum are embedded in each bundle.
- `context__is_rand`, outcomes, and alignment metadata are excluded from model inputs.
- Valid play and literal click use disjoint interface-specific masks; pooled click is rejected.
- Every eligible task contained both classes.
- All 24 bundles contain 300 trees, valid checksums, and complete metadata.
- No training scores were used for selection; no AUC or LogLoss was calculated.

## Failures and unresolved issues

There were no failed or incomplete configurations. Process high-water RSS is cumulative and therefore cannot isolate each model's incremental peak; the report states this limitation explicitly. Actual held-out evaluation remains intentionally deferred to an authorized later milestone.

## Verification

- Full suite: `88 passed in 4.49s` (4.96 s wall time; 392.20 MiB peak RSS as reported by `/usr/bin/time -v`).
- Production bundle audit: 24/24 complete, checksums valid, expected feature set/task/complexity accepted, frozen training dates and protocol identifiers present, and exactly 300 trees each.
- Frozen study-plan SHA-256: `3db23cac54770d5ace5279443dc28e3f1ae0c44a5da78539085660ce1201bdaf` (unchanged).
