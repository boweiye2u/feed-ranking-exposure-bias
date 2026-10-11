# M4.4 MMoE ablation training-quality report

Status: **PASS.** Nine of nine frozen ablation models completed the common budget and passed counter, gradient, checksum, metadata, alignment, probability, and exact-reload checks.

## Architectures

| Ablation | Input width | Parameters | Explicitly excluded |
|---|---:|---:|---|
| Dense only | 238 | 133,527 | User ID, target-video ID, pooled history |
| No history | 270 | 683,191 | Pooled history |
| No aggregates | 220 | 669,791 | Frozen dense `__hist_` channels |
| Full MMoE reference | 286 | 687,479 | None |

All variants retain four 64-unit experts, three four-way gates, three 32-unit towers, dropout 0.1 and the long-view/like/tab-1-valid-play heads.

## Nine-model completion table

Losses are full-run training means and are descriptive only.

| Ablation | Seed | Train seconds | Peak RSS MiB | Overall | Long view | Like | Valid play |
|---|---:|---:|---:|---:|---:|---:|---:|
| Dense only | 0 | 229.51 | 2,853.05 | 0.401873 | 0.524591 | 0.066273 | 0.614756 |
| Dense only | 1 | 221.78 | 2,852.98 | 0.402053 | 0.524905 | 0.066543 | 0.614711 |
| Dense only | 2 | 236.35 | 2,853.19 | 0.401564 | 0.524575 | 0.066059 | 0.614059 |
| No history | 0 | 264.08 | 2,852.60 | 0.399048 | 0.521148 | 0.065797 | 0.610198 |
| No history | 1 | 255.55 | 2,853.28 | 0.399171 | 0.521495 | 0.065650 | 0.610368 |
| No history | 2 | 270.95 | 2,853.67 | 0.399023 | 0.521524 | 0.065451 | 0.610093 |
| No aggregates | 0 | 289.16 | 2,853.28 | 0.409534 | 0.531105 | 0.077141 | 0.620357 |
| No aggregates | 1 | 288.34 | 2,852.64 | 0.409961 | 0.531365 | 0.077014 | 0.621506 |
| No aggregates | 2 | 303.20 | 2,853.13 | 0.409650 | 0.531509 | 0.077034 | 0.620406 |

Every row completed exactly 7,060 optimizer updates and 3,612,405 examples with batch size 512 and `drop_last=False`. AdamW learning rate 0.001, weight decay 1e-5, constant schedule and equal task weights were identical across models.

## Checkpoint integrity

| Ablation | Seed | Checkpoint bytes | SHA-256 |
|---|---:|---:|---|
| Dense only | 0 | 1,675,465 | `7d588129499db986dcbde8a193886be03936034cae2374ac8984974ee0d7b0c3` |
| Dense only | 1 | 1,675,465 | `07bc4517bb1ee16a68b94837bef8f2fab58447525a872a2659e60006ffec0e92` |
| Dense only | 2 | 1,675,465 | `cd01640a69f2e8d3130ce1a3e5c6def7cb4c6f753b7f2e28e49d787c141c2b9e` |
| No history | 0 | 8,273,873 | `23035c25cc51cc581ec7a8e8e4149e9e06562a06ebe4e373f4c414cc178afbcd` |
| No history | 1 | 8,273,873 | `d2a2bbc6677d2cd136861a240105ee4c4f0e85b24ea63d454a3b7067e18936e7` |
| No history | 2 | 8,273,873 | `f6dda5ff42ef3626ef335331f7fe25db81080ee1d0833c21f861f8e826c1bfc6` |
| No aggregates | 0 | 8,113,297 | `bb05e2e1cfa18506db4d2969e70cb2c8c2fb5247d32d984f59bbd1b59d030bdc` |
| No aggregates | 1 | 8,113,297 | `1bed71cd2761193b44ebcbf4e34831a513d1b51e2909dc1c4e87c5ad64bca654` |
| No aggregates | 2 | 8,113,297 | `5d77d4439c985803e5d8c345dd23dfffe650926ae7e405e66502b1c523898e64` |

Every model reproduced its probabilities bit-for-bit over the same 1,024 aligned training impressions; maximum absolute reload difference was 0.0. All nine hashes are unique. The aggregate manifest at `~/recom/models/neural/ablations/v1/manifest.json` revalidated every checkpoint from disk.

## Descriptive size, cost and loss summary

| Model | Parameters | Mean train seconds | Mean overall loss | Mean long-view loss | Mean like loss | Mean valid-play loss |
|---|---:|---:|---:|---:|---:|---:|
| Full MMoE | 687,479 | 283.78 | 0.398081 | 0.520481 | 0.065168 | 0.608593 |
| Dense only | 133,527 | 229.21 | 0.401830 | 0.524690 | 0.066292 | 0.614508 |
| No history | 683,191 | 263.53 | 0.399080 | 0.521389 | 0.065633 | 0.610219 |
| No aggregates | 669,791 | 293.57 | 0.409715 | 0.531326 | 0.077063 | 0.620756 |

These are optimization diagnostics, not evaluation results. They must not be used to select an architecture or interpreted as evidence about ranking performance.

## Safeguards and execution notes

- Input dimensions and excluded feature groups are asserted by code and tests.
- Exact feature/label/sequence alignment and three-head masks are inherited from the strict neural loader.
- Every active parameter received a finite gradient at the final update; no nonfinite loss, output or gradient occurred.
- Checkpoint architecture names prevent cross-ablation loading.
- The feature-manifest and preprocessing hashes match M4.2/M4.3.
- Only April 12–29 normal-training outcomes were loaded.
- No normal-validation, randomized-validation or May 3–8 outcome was accessed.
- A temporary concurrent seed-0 attempt was interrupted before any artifact was saved because thread oversubscription made it unacceptably slow. All reported models are clean sequential reruns.
- The frozen v5 study plan was not modified.

Complete repository suite: **105 passed in 6.79 seconds** (7.96 seconds wall time; 660,376 KiB peak RSS reported by `/usr/bin/time -v`). The final aggregate audit independently instantiated and loaded all nine architecture-specific checkpoints, re-hashed them, and verified their embedded selectors, dimensions, seeds, counters and preprocessing identities.
