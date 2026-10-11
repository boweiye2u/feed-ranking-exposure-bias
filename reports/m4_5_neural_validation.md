# M4.5 neural normal-validation report

## Outcome

All 15 pretrained neural checkpoints loaded with verified checksums and produced aligned probabilities for 61,548 authorized normal-validation impressions. No model was retrained and no preprocessing was fitted on validation data. The frozen overall selection rule retained **LightGBM FS1 deep**, with long-view pair-weighted within-user AUC **0.655549**.

Task populations were unchanged from M3.3: long view 61,548 impressions / 19,757 positives / 7,167 qualifying AUC users / 68,657 pairs; like 61,548 / 1,186 / 698 / 5,743; valid play 47,341 / 24,215 / 6,244 / 56,129.

## Seed-0 neural results

Seed 0 is the prespecified primary neural run. Columns are pair-weighted AUC, equal-user AUC, capped AUC, pooled AUC, LogLoss, and NDCG@5.

| Candidate | Task | Pair AUC | User AUC | Capped AUC | Pooled AUC | LogLoss | NDCG@5 |
|---|---|---:|---:|---:|---:|---:|---:|
| Shared bottom | Long view | 0.647691 | 0.650226 | 0.647691 | 0.759549 | 0.529388 | 0.672820 |
| Shared bottom | Like | 0.640258 | 0.636461 | 0.640258 | 0.882766 | 0.067612 | 0.535926 |
| Shared bottom | Valid play | 0.588306 | 0.589395 | 0.588306 | 0.710979 | 0.623642 | 0.723892 |
| Full MMoE | Long view | 0.648543 | 0.648051 | 0.648543 | 0.759149 | 0.530715 | 0.674821 |
| Full MMoE | Like | 0.636079 | 0.633531 | 0.636079 | 0.879616 | 0.068198 | 0.525723 |
| Full MMoE | Valid play | 0.587539 | 0.587389 | 0.587539 | 0.709596 | 0.625540 | 0.725902 |
| Dense only | Long view | 0.645622 | 0.646526 | 0.645622 | 0.760886 | 0.528037 | 0.672707 |
| Dense only | Like | 0.639213 | 0.635960 | 0.639213 | 0.882303 | 0.067312 | 0.521487 |
| Dense only | Valid play | 0.583638 | 0.581963 | 0.583638 | 0.712603 | 0.622320 | 0.721840 |
| No history | Long view | 0.645215 | 0.645251 | 0.645215 | 0.757146 | 0.532833 | 0.671124 |
| No history | Like | 0.637994 | 0.635914 | 0.637994 | 0.874697 | 0.068486 | 0.523909 |
| No history | Valid play | 0.586934 | 0.583969 | 0.586934 | 0.708207 | 0.626747 | 0.723463 |
| No aggregates | Long view | 0.649409 | 0.650962 | 0.649409 | 0.747684 | 0.540414 | 0.677014 |
| No aggregates | Like | 0.638342 | 0.665579 | 0.638342 | 0.825740 | 0.078357 | 0.554360 |
| No aggregates | Valid play | 0.591316 | 0.593900 | 0.591316 | 0.696836 | 0.635980 | 0.729568 |

The 952-pair cap is nonbinding in this validation population, so capped and uncapped pair-weighted AUC are identical.

## Three-seed long-view robustness

Values are mean / range and are descriptive only.

| Candidate | Pair AUC | User AUC | Pooled AUC | LogLoss | NDCG@5 |
|---|---:|---:|---:|---:|---:|
| Shared bottom | 0.646778 / 0.001631 | 0.649626 / 0.002388 | 0.759164 / 0.000853 | 0.530719 / 0.003532 | 0.673306 / 0.000802 |
| Full MMoE | 0.647390 / 0.002542 | 0.649477 / 0.002318 | 0.759080 / 0.001355 | 0.531311 / 0.001734 | 0.674298 / 0.003628 |
| Dense only | 0.644564 / 0.002636 | 0.647441 / 0.003753 | 0.761677 / 0.001468 | 0.527645 / 0.001533 | 0.672308 / 0.001756 |
| No history | 0.644608 / 0.003831 | 0.647591 / 0.004281 | 0.757373 / 0.000348 | 0.532797 / 0.002716 | 0.672042 / 0.001511 |
| No aggregates | 0.645195 / 0.008346 | 0.650802 / 0.000630 | 0.746804 / 0.001662 | 0.541432 / 0.001737 | 0.673620 / 0.006332 |

Among neural seed-0 runs, no-aggregates has the highest primary long-view AUC (0.649409), 0.006139 below FS1-deep LightGBM. Full MMoE exceeds shared bottom by 0.000852. These are validation descriptions, not final-test claims.

## Frozen 12-candidate comparison

| Candidate | Pair-weighted AUC |
|---|---:|
| R0 | 0.632776 |
| FS1 shallow LightGBM | 0.654485 |
| FS1 deep LightGBM | **0.655549** |
| FS2 shallow LightGBM | 0.645440 |
| FS2 deep LightGBM | 0.644916 |
| FS3 shallow LightGBM | 0.644705 |
| FS3 deep LightGBM | 0.644778 |
| Shared bottom | 0.647691 |
| Full MMoE | 0.648543 |
| MMoE dense only | 0.645622 |
| MMoE no history | 0.645215 |
| MMoE no aggregates | 0.649409 |

FS1-deep LightGBM is frozen as the selected overall configuration. The remaining 11 configurations and all neural robustness seeds are preserved.

## Integrity and resources

- Feature-manifest SHA-256: `f613933fde572aefb5ada5e3c99d078c576da90103bbfe79a4e3c920631b408f`.
- Neural-preprocessor SHA-256: `8d598f9187238cd670913aedbe5fbb89e58a27d44565e60ac2b00e7d023d19b7`.
- Frozen study-plan SHA-256: `3db23cac54770d5ace5279443dc28e3f1ae0c44a5da78539085660ce1201bdaf`, unchanged from both frozen tags.
- Sequence: 61,548 rows, 20 positions, mean length 18.542, 192 empty histories, 2,389 tie-boundary drops, 3.9 MiB.
- Neural predictions: 45 parquet files, about 12.7 MiB; complete validation prediction directory about 23 MiB.
- Evaluation: 63.39 seconds on CPU, peak RSS 925.0 MiB. Sequence construction took 1.87 seconds.
- Every checkpoint reports 7,060 updates and 3,612,405 examples; checksum and row-alignment checks passed.
- Full repository test suite: 108 passed in 7.05 seconds.
- Validation preprocessing refits: none. Randomized-validation outcomes accessed: no. May 3–8 outcomes accessed: no.

Machine-readable results are at `~/recom/predictions/validation/v1/neural_validation_results.json`; the candidate freeze is `~/recom/predictions/validation/v1/candidate_manifest.json`.
