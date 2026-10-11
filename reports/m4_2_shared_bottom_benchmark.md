# M4.2 shared-bottom completion report

Status: **PASS.** The resource gate passed and all three approved production shared-bottom seeds completed the frozen budget without warnings, numerical errors, early stopping, or held-out outcome access.

## Resource benchmark

The mandatory temporary benchmark used the complete production pipeline, 10 warm-up updates and 100 measured updates. It measured 44.215 updates/s, 22,638 examples/s and 2,853.53 MiB peak RSS on CPU. Full preprocessing took 59.41 seconds. System utilities reported eight NVIDIA L40S devices, but PyTorch 2.14.1 is a CPU-only build, so production training used CPU.

## Frozen configuration and population

| Setting | Value |
|---|---:|
| Normal-training dates | Apr 12–29, 2022 |
| Training rows | 722,481 |
| Parameters | 649,411 |
| Batch size | 512 |
| `drop_last` | False |
| Optimizer | AdamW |
| Learning rate | 0.001, constant |
| Weight decay | 1e-5 |
| Updates per seed | 7,060 |
| Complete passes | 5 |
| Actual examples per seed | 3,612,405 |
| Seeds | 0, 1, 2 |

Task populations were identical for all seeds: long view 722,481 eligible / 239,795 positive; like 722,481 / 13,706; tab-1 valid play 535,569 / 279,124. Equal mask-normalized task weights `[1, 1, 1]` were preserved.

## Seed-level execution and integrity

| Seed | Training time | Total runtime | Peak RSS | Updates | Examples | Reload max difference |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | 201.36 s | 263.30 s | 2,853.51 MiB | 7,060 | 3,612,405 | 0.0 |
| 1 | 200.71 s | 266.33 s | 2,852.62 MiB | 7,060 | 3,612,405 | 0.0 |
| 2 | 204.30 s | 264.57 s | 2,854.14 MiB | 7,060 | 3,612,405 | 0.0 |

All reload comparisons used the same 1,024 aligned impression IDs (ID-byte SHA-256 `8176bd09f5b6bc10c6c2e7e5a43f2747960788baf66fa9eccc0c5953f6a33bf8`) and were bit-exact.

## Training-loss diagnostics

These losses describe optimization only; they are not validation metrics or model-selection evidence.

| Seed | Loss | First | Mean | Last-100 mean | Last |
|---:|---|---:|---:|---:|---:|
| 0 | Overall | 0.687213 | 0.399053 | 0.390951 | 0.426395 |
| 0 | Long view | 0.708224 | 0.521348 | 0.511982 | 0.530140 |
| 0 | Like | 0.658291 | 0.065953 | 0.062095 | 0.172692 |
| 0 | Valid play | 0.695123 | 0.609859 | 0.598776 | 0.576353 |
| 1 | Overall | 0.702481 | 0.398948 | 0.392104 | 0.390740 |
| 1 | Long view | 0.697909 | 0.521170 | 0.512513 | 0.494469 |
| 1 | Like | 0.714946 | 0.065874 | 0.063612 | 0.006853 |
| 1 | Valid play | 0.694588 | 0.609801 | 0.600189 | 0.670898 |
| 2 | Overall | 0.708230 | 0.399103 | 0.392623 | 0.455918 |
| 2 | Long view | 0.682234 | 0.521590 | 0.510175 | 0.462528 |
| 2 | Like | 0.748019 | 0.065984 | 0.067102 | 0.173235 |
| 2 | Valid play | 0.694437 | 0.609736 | 0.600592 | 0.731991 |

The variable last-batch losses reflect the 209-row final batch and are retained rather than discarded. The complete 7,060-update logs are stored with each external bundle.

## Checkpoints

| Seed | Checkpoint bytes | SHA-256 |
|---:|---:|---|
| 0 | 7,850,129 | `17b1a38642f2472eac980d6f100aad428afe4e9828799bfb44b7f57e0243d795` |
| 1 | 7,850,129 | `e21e73477c1805d6da35432d2ce19a150308bbe0ac7832e9b23d2d1831502ffe` |
| 2 | 7,850,129 | `ada834c9be5fd0ff30095f8798257fe3bd9d7b415ddf758b0ca19e7dc0993b70` |

The frozen feature-manifest SHA-256 is `f613933fde572aefb5ada5e3c99d078c576da90103bbfe79a4e3c920631b408f`; the preprocessing SHA-256 is `8d598f9187238cd670913aedbe5fbb89e58a27d44565e60ac2b00e7d023d19b7`. Software versions were Python 3.12.7, PyTorch 2.14.1+cpu, NumPy 2.5.3 and Polars 2.0.0.

## Safeguards

- Exact feature/label/sequence impression-ID alignment was enforced before every run.
- Outcome fields were absent from model inputs; `context__is_rand` remained excluded.
- Only April 12–29 normal-training labels were loaded.
- No validation labels, randomized-validation labels, or May 3–8 outcomes were accessed.
- Checkpoint tests cover masked loss, exact counters, seed independence, save/reload equality, checksum tampering, stale metadata, and row-alignment rejection.
- The frozen v5 plan SHA-256 remains `3db23cac54770d5ace5279443dc28e3f1ae0c44a5da78539085660ce1201bdaf`.

Complete project suite: **101 passed in 6.33 seconds** (7.48 seconds wall time; 659,552 KiB peak RSS reported by `/usr/bin/time -v`). A separate final audit loaded all three production bundles through the strict checkpoint loader and verified their embedded hashes, update counts and example counts against their run reports.
