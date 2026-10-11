# M4.1 neural infrastructure quality report

Status: **PASS.** The normal-training-fitted adapter, shared-bottom MLP, MMoE, predefined ablation inputs, masked three-task loss, deterministic training loop, and checkpoint loader are implemented. Only two optimizer updates per model were run on a deterministic 2,048-row normal-training sample.

## Training populations

| Task | Eligible rows | Positives | Prevalence | Neural head |
|---|---:|---:|---:|---|
| Long view | 722,481 | 239,795 | 33.1905% | Yes |
| Like | 722,481 | 13,706 | 1.8971% | Yes |
| Valid play, tab 1 | 535,569 | 279,124 | 52.1173% | Yes |
| Literal click, tab 0 | 78,575 | 6,568 | 8.3589% | No; supplementary only |

Each eligible task is normalized by its own mask denominator before equal task weights `[1, 1, 1]` are applied. Inapplicable valid-play rows have mask 0. Literal-click labels never enter the valid-play head.

## Input dimensions

| Input/configuration | Width | Detail |
|---|---:|---|
| Numerical | 94 | 47 imputed/scaled values + 47 explicit missing indicators |
| Dense categorical embeddings | 144 | 36 fields × 4 dimensions |
| User ID | 16 | Separate training-only vocabulary |
| Target video ID | 16 | Shared table with history videos |
| Masked-mean history | 16 | Fixed 20 positions, padding 0, unknown 1 |
| Full model input | 286 | Dense + IDs + history |
| Dense-only ablation | 238 | No IDs or history |
| No-history ablation | 270 | Dense + IDs |
| No-aggregates ablation | 220 | Removes `__hist_` dense values and indicators |

## Vocabulary sizes

Sizes include reserved missing/unknown positions where applicable.

| Field | Size | Field | Size |
|---|---:|---|---:|
| `user_id` | 26,523 | `video_id` | 7,295 |
| `context__tab` | 17 | `category__key` | 113 |
| `user__user_active_degree` | 11 | `user__is_lowactive_period` | 3 |
| `user__is_live_streamer` | 4 | `user__is_video_author` | 4 |
| `user__follow_user_num_range` | 10 | `user__fans_user_num_range` | 11 |
| `user__friend_user_num_range` | 9 | `user__register_days_range` | 10 |
| `user__onehot_feat0` | 4 | `user__onehot_feat1` | 9 |
| `user__onehot_feat2` | 52 | `user__onehot_feat3` | 1,447 |
| `user__onehot_feat4` | 17 | `user__onehot_feat5` | 36 |
| `user__onehot_feat6` | 5 | `user__onehot_feat7` | 120 |
| `user__onehot_feat8` | 456 | `user__onehot_feat9` | 9 |
| `user__onehot_feat10` | 7 | `user__onehot_feat11` | 7 |
| `user__onehot_feat12` | 4 | `user__onehot_feat13` | 4 |
| `user__onehot_feat14` | 4 | `user__onehot_feat15` | 4 |
| `user__onehot_feat16` | 4 | `user__onehot_feat17` | 4 |
| `video__author_id` | 6,310 | `video__music_id` | 6,941 |
| `video__video_type` | 5 | `video__upload_dt` | 5 |
| `video__upload_type` | 16 | `video__visible_status` | 3 |
| `video__music_type` | 7 | `video__tag` | 112 |

The machine-readable preprocessor contains every exact token-to-index mapping. Unknown validation or later authorized RQ3 values map to 1 without changing these training vocabularies.

## Smoke execution

| Architecture | Parameters | Examples | Updates | First loss | Last loss | Probability range | Reload exact |
|---|---:|---:|---:|---:|---:|---|---|
| Shared-bottom | 649,411 | 1,024 | 2 | 0.688937 | 0.671755 | 0.417462–0.541902 | Yes |
| MMoE | 687,479 | 1,024 | 2 | 0.662666 | 0.648916 | 0.358353–0.540129 | Yes |

The adapter fit took approximately 9.7 s and the final complete smoke script 11.16 s internally. The measured process peak was 1,428.42 MiB. Execution used PyTorch 2.14.1+cpu; CUDA was unavailable, so CPU was selected. Device routing supports CUDA and fails clearly if CUDA is explicitly requested but unavailable.

Artifacts:

- preprocessor: 1,121,180 bytes;
- shared-bottom smoke checkpoint: 7,850,065 bytes;
- MMoE smoke checkpoint: 8,325,329 bytes;
- each architecture also has a checkpoint manifest and two-update training log.

## Verification and limitations

- Synthetic tests cover numerical imputation/scaling/indicators, missing and unknown categories/IDs, 20-position padding, strict history time, masked pooling, task masks, normalized loss, forward/backward passes, parameter updates, deterministic initialization/batching, predefined ablation inputs, counters, checkpoint compatibility, and exact reload predictions.
- Feature and sequence IDs must match position by position; inputs are never silently realigned.
- Outcome fields are rejected from feature preprocessing.
- `context__is_rand` is excluded from normal-only adapters. The explicit joint-adapter switch is present but was not used to fit an RQ3 adapter.
- No randomized-validation or May 3–8 outcome was read.
- These two-update losses are functionality checks, not performance estimates or model-selection evidence.
- GPU execution could not be exercised because this environment exposes no CUDA device; the same tensor/device path is used when CUDA is available.
- Complete project suite: `99 passed in 6.07s` (6.97 s wall time; 667,364 KiB peak RSS reported by `/usr/bin/time -v`).
