# M4.3 MMoE training-quality report

Status: **PASS.** All three production MMoE seeds completed the frozen training budget and passed numerical, routing, counter, checkpoint, and reload checks.

## Configuration and population

| Setting | Value |
|---|---:|
| Input width | 286 |
| Experts | 4 × 64 units |
| Task gates | 3 × four-way softmax |
| Task towers | 3 × 32 units |
| Parameters | 687,479 |
| Training rows | 722,481 |
| Updates per seed | 7,060 |
| Examples per seed | 3,612,405 |
| Batch size / `drop_last` | 512 / False |
| Optimizer | AdamW, lr 0.001, weight decay 1e-5 |
| Schedule | Constant |
| Seeds | 0, 1, 2 |

Task populations were long view 722,481 eligible / 239,795 positive, like 722,481 / 13,706, and tab-1 valid play 535,569 / 279,124. Equal task weights `[1, 1, 1]` were retained.

## Seed execution and integrity

| Seed | Training time | Total runtime | Peak RSS | Checkpoint bytes | Reload difference |
|---:|---:|---:|---:|---:|---:|
| 0 | 316.33 s | 400.59 s | 2,854.10 MiB | 8,325,329 | 0.0 |
| 1 | 248.07 s | 326.22 s | 2,852.58 MiB | 8,325,329 | 0.0 |
| 2 | 286.95 s | 366.10 s | 2,854.45 MiB | 8,325,329 | 0.0 |

Every seed completed exactly 7,060 updates and 3,612,405 examples. No nonfinite loss, output, or gradient was encountered. The post-training gradient norms `(experts, gates, towers)` were `(0.15315, 0.08901, 0.13549)`, `(0.15720, 0.07950, 0.11123)`, and `(0.17030, 0.08794, 0.20399)` for seeds 0–2 respectively.

## Training-loss diagnostics

These are descriptive optimization losses, not ranking-performance measurements.

| Seed | Loss | Full-run mean | Last-100 mean | Last |
|---:|---|---:|---:|---:|
| 0 | Overall | 0.398157 | 0.390038 | 0.420226 |
| 0 | Long view | 0.520732 | 0.511273 | 0.539702 |
| 0 | Like | 0.065012 | 0.061640 | 0.168039 |
| 0 | Valid play | 0.608728 | 0.597200 | 0.552937 |
| 1 | Overall | 0.398125 | 0.390406 | 0.377679 |
| 1 | Long view | 0.520276 | 0.510819 | 0.468658 |
| 1 | Like | 0.065412 | 0.062555 | 0.007709 |
| 1 | Valid play | 0.608688 | 0.597843 | 0.656671 |
| 2 | Overall | 0.397960 | 0.391258 | 0.463205 |
| 2 | Long view | 0.520436 | 0.509313 | 0.481816 |
| 2 | Like | 0.065079 | 0.065215 | 0.148558 |
| 2 | Valid play | 0.608364 | 0.599248 | 0.759240 |

The volatile final values come from the retained 209-row final batch. Complete update-level logs are stored outside Git.

## Gate and expert-use diagnostics

Each vector lists experts 0–3. Mean routing weights sum to one; all individual routing rows were nonnegative and normalized.

| Seed | Task | Mean gate weights | Mean entropy |
|---:|---|---|---:|
| 0 | Long view | `[0.7052, 0.0365, 0.1237, 0.1346]` | 0.7186 |
| 0 | Like | `[0.0300, 0.4577, 0.0330, 0.4793]` | 0.6666 |
| 0 | Valid play | `[0.3294, 0.1282, 0.4632, 0.0791]` | 0.9253 |
| 1 | Long view | `[0.1000, 0.5118, 0.2648, 0.1234]` | 0.8084 |
| 1 | Like | `[0.1557, 0.0517, 0.3459, 0.4467]` | 0.8768 |
| 1 | Valid play | `[0.1586, 0.6563, 0.0775, 0.1076]` | 0.7413 |
| 2 | Long view | `[0.6480, 0.2262, 0.0790, 0.0467]` | 0.6307 |
| 2 | Like | `[0.0257, 0.0917, 0.1310, 0.7515]` | 0.5815 |
| 2 | Valid play | `[0.5412, 0.1945, 0.2055, 0.0588]` | 0.8853 |

All experts received nonzero mean routing mass for every task and seed. The seed-specific reports additionally contain routing standard deviations, argmax-expert fractions, and minimum/maximum weights over all 722,481 rows. Expert numbering is seed-local, so differing allocations across seeds are expected and are not aligned component identities.

## Checkpoints

| Seed | SHA-256 |
|---:|---|
| 0 | `b9b41b6e7f2dfccb351e6736fcbbd04dd7d6b72e059552d57285b02f041ec3ed` |
| 1 | `e93bd599d2cf562570c07310cb99e90c978fa38404fe94803ba9e102f7241448` |
| 2 | `fb5cbe8482732298d958e3d3459f7328519fdbc4468e2d876cca763ac04d8980` |

All checkpoints reloaded through the strict compatibility loader. Predictions over the same 1,024 aligned impression IDs were bit-exact before and after reload.

## Descriptive comparison with shared-bottom

| Architecture | Parameters | Updates/seed | Examples/seed | Mean training time | Mean overall training loss |
|---|---:|---:|---:|---:|---:|
| Shared-bottom | 649,411 | 7,060 | 3,612,405 | 202.13 s | 0.399035 |
| MMoE | 687,479 | 7,060 | 3,612,405 | 283.78 s | 0.398081 |

MMoE has 38,068 more parameters (5.86%) and took about 40.4% longer in the recorded optimization phase. Its seed-mean per-task training losses were 0.520481 long view, 0.065168 like, and 0.608593 valid play, versus 0.521369, 0.065937, and 0.609799 for shared-bottom. These small training-loss differences do **not** establish a ranking-performance difference; no validation metric was computed.

## Safeguards

- Feature/label/sequence IDs were checked position by position.
- The feature-manifest and preprocessing hashes match M4.2.
- `context__is_rand` and outcomes were excluded from model inputs.
- Only April 12–29 normal-training outcomes were loaded.
- No randomized-validation or May 3–8 outcome was accessed.
- No MMoE ablation, early stopping, class weighting, or extra objective was used.
- The frozen study-plan hash remains unchanged.

Complete repository suite: **102 passed in 6.95 seconds** (7.77 seconds wall time; 666,032 KiB peak RSS reported by `/usr/bin/time -v`). A separate final audit loaded all three production checkpoints through the strict compatibility path and matched every embedded checksum, architecture field, update count and example count to its seed report.
