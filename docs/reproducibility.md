# Reproducibility guide

This guide maps the frozen scientific workflow to executable repository entry points. It supplements, rather than changes, the v5 study plan and approved amendments.

## Checkpoints and protocol

| Checkpoint | Meaning |
|---|---|
| `week0-protocol-v1` | Branch B1, dates, targets, metrics, click semantics, pair cap |
| `m2-features-v1` | Leakage-safe FS1–FS3 schemas and sequence histories |
| `m4-candidates-v1` | Twelve frozen RQ1/RQ2 configurations and neural seeds |
| `m5-prefinal-v1` | Contenders, RQ3 models, inference definitions, readiness gate |
| `m7-results-v1` | Verified M6 results and publication research package |

The immutable source documents are `docs/study_plan_v5.md`, `docs/week0_amendment_v1.md`, and `docs/m5_3_statistical_amendment_v1.md`. Configuration lives under `configs/`; seeds and budgets are in `configs/lightgbm.toml`, `configs/neural.toml`, and `configs/final_evaluation.toml`.

## Data and artifact root

Large artifacts default to `~/recom/`. Override this before invoking scripts:

```bash
export KUAIRAND_ARTIFACT_ROOT=/absolute/path/to/recom
python scripts/init_artifacts.py
```

The extracted dataset root is `$KUAIRAND_ARTIFACT_ROOT/raw/KuaiRand-Pure/data/`. Obtain only `KuaiRand-Pure.tar.gz` from the official Zenodo record linked in `docs/data_card.md`; expected MD5 is `0820331067a3784d9691136f772b35a7`.

Raw records, processed Parquet files, model weights, categorical vocabularies, and predictions are local-only. They are covered by `.gitignore` and are not part of the results tag.

## Chronology and leakage controls

- Burn-in: Apr 9–11 normal-policy rows, feature history only.
- Normal training: Apr 12–29.
- Randomized RQ3 training: Apr 22–29.
- Normal validation: Apr 30–May 2.
- Final T-normal and T-random: May 3–8.

FS2/FS3 aggregate history is normal-policy only and strictly prior calendar day. Sequence history is normal-policy only and strictly earlier in timestamp. Feature, label, sequence, and prediction tables carry unique ascending `impression_id`; loaders reject positional mismatches, duplicate IDs, stale build versions, and schema drift.

## Pipeline stages

1. `scripts/audit_dataset.py` verifies checksum, structure, schema, and dates.
2. Week-0 scripts reproduce only permitted diagnostics.
3. `scripts/build_fs1.py`, `build_fs2.py`, and `build_fs3.py` construct authorized features.
4. `scripts/review_m2_features.py` freezes feature schemas and alignment.
5. M3 scripts build R0, LightGBM models, and normal-validation results.
6. M4 scripts build neural preprocessing, train seeds 0–2, and freeze 12 candidates.
7. M5 scripts prepare/train RQ3 alternatives and freeze contender/inference manifests.
8. `scripts/build_final_predictions.py` builds immutable held-out predictions without metrics.
9. `scripts/evaluate_final_rq1_rq3.py` computes authorized RQ1/RQ3 final results.
10. `scripts/evaluate_rq2_regret.py` computes 50-split, 2,000-bootstrap RQ2 regret.
11. `scripts/build_results_package.py` verifies hashes and rebuilds only compact tables/figures.

Final-evaluation scripts intentionally fail closed when earlier manifests or authorization conditions are absent. They should not be run casually because they are separated from pre-final workflows to preserve the outcome embargo.

## Model configurations and inference

LightGBM uses fixed shallow/deep capacity, 300 rounds, learning rate 0.05, seed 202211, minimum leaf size 100, L2 regularization 1.0, and deterministic single-thread execution.

Neural models use AdamW, learning rate 0.001, weight decay `1e-5`, batch size 512, constant schedule, seeds 0–2, and 7,060 optimizer updates. Three equal-weight masked BCE losses train long view, like, and tab-1 valid play. Inputs use training-only preprocessing and masked-mean last-20 histories.

FS1-deep was selected on normal validation before final outcomes opened. Final T-random results do not revise that choice. Neural seed 0 is primary; seeds 1–2 are robustness checks.

The primary metric is uncapped pair-weighted within-user long-view AUC. The 952-pair cap is sensitivity-only. RQ2 uses 50 deterministic A/B user splits in both directions and 2,000 paired user bootstraps with seed 0. Six prespecified comparisons use 10,000 paired bootstraps, percentile intervals, a finite-resolution tail-share approximation, and step-down Holm adjustment.

## Verification

```bash
source .venv/bin/activate
python scripts/check_environment.py
python -m pytest -q
python scripts/check_public_repo.py
MPLCONFIGDIR=/tmp/kuairand-mpl python scripts/build_results_package.py
git status --short
```

The packaging command verifies live Git tag targets, protocol hashes, the M2 feature manifest, all 146 prediction hashes, M6.2/M6.3 output hashes, candidate/contender identities, RQ3 arms, and comparison definitions before reading result values.

The recorded environment is in `reports/tables/reproducibility_inventory.json`; compact results are under `reports/tables/`; figure hashes are in `reports/figures/manifest.json`.

## Scope

The complete workflow was executed incrementally under the recorded environment and frozen checkpoints. A fresh-machine end-to-end rerun has not been performed. Lightweight installation and synthetic/unit tests do not require the dataset; checksum-bound tests of the frozen M6 results skip when their external artifacts are absent. Full reproduction requires KuaiRand-Pure, roughly 1.4 GiB of local storage for the observed artifacts, and sufficient CPU time for model training.
