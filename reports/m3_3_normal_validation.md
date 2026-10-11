# M3.3 normal-validation report

Status: **PASS.** Features and labels cover 61,548 normal-policy impressions from April 30–May 2, 2022. All 24 unchanged LightGBM bundles and R0 were scored. No randomized-validation or final-test outcomes were opened.

Metric abbreviations: `Pair` is the primary pair-weighted within-user AUC; `Equal` is per-user-average AUC; `Cap952` is capped-pair sensitivity; `Pool` is pooled AUC; `LL` is binary LogLoss; `NDCG5` is mean within-user NDCG@5. Score ties receive 0.5 AUC credit. NDCG score ties use ascending `time_ms`, then `video_id`, then stable row order.

## Population and denominators

| Task | Impressions | Positives | Prevalence | AUC-eligible users | Single-class users | Positive-negative pairs | NDCG@5 users |
|---|---:|---:|---:|---:|---:|---:|---:|
| Long view | 61,548 | 19,757 | 32.100% | 7,167 | 10,082 | 68,657 | 3,498 |
| Like | 61,548 | 1,186 | 1.927% | 698 | 16,551 | 5,743 | 418 |
| Valid play (tab 1) | 47,341 | 24,215 | 51.150% | 6,244 | 8,631 | 56,129 | 2,852 |
| Literal click (tab 0) | 5,095 | 428 | 8.400% | 187 | 2,656 | 568 | 39 |

No eligible user exceeds the frozen 952-pair cap: maxima are 546, 120, 703, and 32 for long view, like, valid play, and literal click. Consequently `Cap952` equals `Pair` throughout this checkpoint.

## Long view

| Candidate | Pair | Equal | Cap952 | Pool | LL | NDCG5 |
|---|---:|---:|---:|---:|---:|---:|
| R0 | 0.632776 | 0.636531 | 0.632776 | 0.696210 | 0.573040 | 0.664616 |
| FS1 shallow | 0.654485 | 0.659310 | 0.654485 | 0.736241 | 0.547153 | 0.680834 |
| **FS1 deep** | **0.655549** | **0.660210** | **0.655549** | 0.741522 | 0.543802 | **0.681922** |
| FS2 shallow | 0.645440 | 0.651329 | 0.645440 | **0.761448** | **0.528350** | 0.671167 |
| FS2 deep | 0.644916 | 0.645900 | 0.644916 | 0.759023 | 0.531444 | 0.673617 |
| FS3 shallow | 0.644705 | 0.647367 | 0.644705 | 0.761367 | 0.528423 | 0.671827 |
| FS3 deep | 0.644778 | 0.649397 | 0.644778 | 0.759800 | 0.530689 | 0.673824 |

FS1 deep is selected by the prespecified primary rule. Its primary AUC exceeds R0 by 0.022773. The different behavior of within-user and pooled/calibration metrics is expected: FS2/FS3 have higher pooled AUC and lower LogLoss but lower pair-weighted within-user AUC. Selection follows the frozen primary metric rather than choosing post hoc among those summaries.

Primary-AUC feature increments are FS2−FS1: −0.009045 shallow and −0.010633 deep; FS3−FS2: −0.000736 shallow and −0.000138 deep. Deep−shallow is +0.001063 for FS1, −0.000524 for FS2, and +0.000073 for FS3. Equal-user AUC for the selected candidate is 0.660210, 0.004662 above its pair-weighted result.

## Like

| Candidate | Pair | Equal | Cap952 | Pool | LL | NDCG5 |
|---|---:|---:|---:|---:|---:|---:|
| FS1 shallow | 0.669511 | 0.674938 | 0.669511 | 0.840833 | 0.073454 | **0.574989** |
| **FS1 deep** | **0.671078** | **0.690952** | **0.671078** | 0.843294 | 0.073514 | 0.572128 |
| FS2 shallow | 0.641302 | 0.649583 | 0.641302 | 0.880498 | 0.068217 | 0.537416 |
| FS2 deep | 0.639213 | 0.641726 | 0.639213 | 0.871463 | 0.070823 | 0.520274 |
| FS3 shallow | 0.643566 | 0.655081 | 0.643566 | **0.881095** | **0.068055** | 0.533273 |
| FS3 deep | 0.634076 | 0.649512 | 0.634076 | 0.872934 | 0.070890 | 0.518510 |

FS1 deep has the highest task-specific pair-weighted AUC. Like is rare and only 698 users contribute within-user pairs, so small differences should not be over-interpreted without later uncertainty analysis.

## Valid play — tab 1 only

| Candidate | Pair | Equal | Cap952 | Pool | LL | NDCG5 |
|---|---:|---:|---:|---:|---:|---:|
| FS1 shallow | 0.593169 | 0.592161 | 0.593169 | 0.678687 | 0.644709 | 0.728479 |
| **FS1 deep** | **0.595521** | **0.594148** | **0.595521** | 0.684584 | 0.641871 | **0.728739** |
| FS2 shallow | 0.586622 | 0.579763 | 0.586622 | **0.713320** | **0.621640** | 0.724574 |
| FS2 deep | 0.583727 | 0.577242 | 0.583727 | 0.709545 | 0.626001 | 0.724001 |
| FS3 shallow | 0.586622 | 0.580133 | 0.586622 | 0.713225 | 0.621693 | 0.724952 |
| FS3 deep | 0.582702 | 0.578398 | 0.582702 | 0.709270 | 0.626221 | 0.722627 |

FS1 deep has the highest task-specific pair-weighted AUC. This target means valid play in tab 1 and is not pooled with literal click.

## Literal click — tab 0 only

| Candidate | Pair | Equal | Cap952 | Pool | LL | NDCG5 |
|---|---:|---:|---:|---:|---:|---:|
| FS1 shallow | 0.544894 | 0.560830 | 0.544894 | 0.651389 | 0.282984 | 0.504984 |
| FS1 deep | 0.551937 | **0.581272** | 0.551937 | 0.649376 | 0.291654 | 0.515877 |
| FS2 shallow | 0.535211 | 0.549763 | 0.535211 | 0.683936 | 0.277924 | 0.519302 |
| FS2 deep | 0.549296 | 0.540590 | 0.549296 | 0.673426 | 0.292073 | **0.548020** |
| FS3 shallow | 0.558099 | 0.579247 | 0.558099 | **0.685622** | **0.277914** | 0.547824 |
| **FS3 deep** | **0.563380** | 0.557599 | **0.563380** | 0.678777 | 0.291874 | 0.539027 |

FS3 deep has the highest task-specific pair-weighted AUC. This comparison is especially uncertain: only 187 users, 568 pairs, and 39 NDCG-eligible users contribute. It is interface-specific and cannot be combined with valid play.

## Integrity, resources, and limitations

- All 24 pretrained model and adapter checksums passed; every saved training vocabulary was reused unchanged.
- Every prediction is finite, lies in [0,1], and is positionally aligned to its eligible label rows.
- Validation feature schemas equal frozen training schemas; FS3 preserves FS2 and labels never enter feature or prediction tables.
- Feature construction took 18.34 s; scoring and metrics took 40.95 s; total command wall time was 60.71 s and peak RSS was 2,287,116 KiB (2,233.51 MiB).
- Results are descriptive on one normal-policy validation window. No bootstrap intervals or multiplicity-adjusted inference were requested at this selection checkpoint.
- The cap sensitivity cannot reveal weighting sensitivity here because no user reaches the frozen cap.
- R0 is available only for long view. No overall LightGBM-versus-neural conclusion is possible before neural models are trained.
- Complete project test suite: `94 passed in 4.27s` (4.71 s wall time; 400,672 KiB peak RSS reported by `/usr/bin/time -v`).
