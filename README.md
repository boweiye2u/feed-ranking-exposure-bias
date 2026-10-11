# KuaiRand-Pure Recommendation Research

Study: **Feed Ranking under Exposure Bias, v5**.

This repository contains code, configuration, tests, documentation, and reports for the study. Large data and model artifacts live outside Git under `~/recom/` by default. Set `KUAIRAND_ARTIFACT_ROOT` to override that location.

## Preliminary research questions

- **RQ1 — Architecture:** How do candidate representation and feed-ranking architecture affect recommendation quality under the study's exposure setting? The codebase will keep data preparation, model construction, training, and evaluation as separable stages so architecture comparisons share the same inputs and metrics.
- **RQ2 — Selection regret:** How much outcome quality is lost when a selection policy chooses an item other than the best available candidate? Evaluation will report selection regret alongside ranking metrics, using only definitions fixed by the study protocol.
- **RQ3 — Exposure-aware training:** Can training objectives or weighting strategies that account for non-uniform exposure improve generalization? Exposure-aware methods will be implemented as explicit, configurable training variants and compared with protocol-defined baselines.

These statements describe the intended project structure; they do not add to or revise the v5 study protocol.

## Layout

```text
configs/                 Versioned experiment configuration
docs/                    Project and milestone documentation
reports/                 Versioned report sources (not large outputs)
scripts/                 Command-line utilities
src/kuairand_pure/       Python package
tests/                   Automated tests
~/recom/raw/             Immutable source data
~/recom/processed/       Derived datasets
~/recom/models/          Model artifacts
~/recom/predictions/     Prediction outputs
~/recom/cache/           Disposable caches
```

## M0 setup

Python 3.11 or newer is required. Create an isolated environment and install the project in development mode:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev]'
python scripts/init_artifacts.py
pytest
```

The artifact initialization command creates only the directory structure. It does not download data or train a model.

## Dataset structural audit

With the separately downloaded KuaiRand-Pure release at the default artifact root, run the outcome-safe structural audit with:

```bash
source .venv/bin/activate
python scripts/audit_dataset.py
```

The command writes its machine-readable result to `~/recom/processed/audits/m1_1_structural_audit.json`. It intentionally does not aggregate engagement values.

The protocol-bounded M1.2 diagnostics are reproduced with:

```bash
python scripts/engagement_diagnostics.py
```

This command permits outcomes only from the April 22–29 randomized training slice and uses identifiers alone for held-out overlap checks.

## Frozen Week-0 protocol

The approved Week-0 v1 amendment, executable configuration, and final metric definitions are in `docs/week0_amendment_v1.md`, `configs/week0_protocol.toml`, and `docs/metric_spec.md`. The local protocol tag is `week0-protocol-v1`.

Reproduce the outcome-free burn-in audit with:

```bash
python scripts/week0_structural_audit.py
```

It writes `~/recom/processed/audits/m1_3_week0_structural_audit.json`. Outcome-bearing loaders are subject to the protocol guard; final-test outcomes cannot be opened by pre-final workflows.

## M2.1 static features

Build the authorized normal and RQ3-randomized training FS1 tables with:

```bash
python scripts/build_fs1.py
```

Outputs are written under `~/recom/processed/fs1/v1/`. Features and labels are separate Parquet files joined by stable `impression_id`; only metadata schemas are checked for validation and final-test windows. See `docs/milestones/M2.1.md` and `reports/m2_1_fs1_quality.md`.

Build point-in-time FS2 user/video aggregates with:

```bash
python scripts/build_fs2.py
```

FS2 uses only strictly earlier-day normal-policy outcomes, freezes `alpha=20` and Apr 9–11 priors, and writes training-only artifacts under `~/recom/processed/fs2/v1/`. Click history remains separated into tab-1 valid play and tab-0 literal click. See `docs/milestones/M2.2.md` and `reports/m2_2_fs2_quality.md`.

Build FS3 category-affinity, recency, and separate last-20 neural-history inputs with:

```bash
python scripts/build_fs3.py
```

FS3 preserves FS2, uses prior-day normal-only aggregates, and writes fixed-length strictly earlier normal-policy histories under `~/recom/processed/fs3/v1/`. The canonical tag-set representation treats undocumented numeric tags as opaque IDs.

Run the M2 feature-freeze integration gate with:

```bash
python scripts/review_m2_features.py
```

All feature, label, and sequence artifacts use ascending `impression_id` order and versioned `m2-features-v1` manifests. Loaders fail on positional misalignment or incompatible builds rather than silently joining by row. The exhaustive feature manifest and quality report are written to `~/recom/processed/m2/v1/`; see `docs/milestones/M2.4a.md` and `reports/m2_4_integration_quality.md`.

## M3.1 R0 popularity baseline

Materialize deterministic training-split R0 scores with:

```bash
python scripts/score_r0.py
```

R0 directly uses the FS2 strictly prior-day normal-policy smoothed video long-view rate. It reads no outcome column, performs no fitting, and writes aligned normal/random training predictions under `~/recom/predictions/r0/v1/`. See `docs/milestones/M3.1.md` and `reports/m3_1_r0_quality.md`.

## M3.2a LightGBM infrastructure

Audit the frozen-manifest LightGBM adapters and fit normal-training-only categorical vocabularies with:

```bash
python scripts/audit_lightgbm_interface.py
```

The audit trains no model. It validates FS1–FS3 schemas, saves deterministic categorical adapters under `~/recom/models/lightgbm/infrastructure/v1/`, and reports interface-specific task masks. The planned matrix is FS1/FS2/FS3 × shallow/deep, with separate long-view, like, tab-1 valid-play, and tab-0 literal-click models. See `docs/milestones/M3.2a.md` and `reports/m3_2a_lightgbm_infrastructure.md`.

## M3.2b LightGBM training

Run the frozen production benchmark and, only after it passes its resource gate, the full normal-training matrix:

```bash
python scripts/train_lightgbm_matrix.py benchmark
python scripts/train_lightgbm_matrix.py full
```

The 24 task predictors are stored outside Git under `~/recom/models/lightgbm/v1/`. Each bundle binds its model to the exact feature schema, categorical vocabulary, task mask, training dates, fixed hyperparameters, software versions, and frozen protocol identifiers. M3.2b computes no validation/test metrics or model ranking. See `docs/milestones/M3.2b.md` and `reports/m3_2b_lightgbm_training.md`.

## M3.3 normal-validation evaluation

Build the authorized April 30–May 2 normal-validation features and score the unchanged R0/LightGBM candidates with:

```bash
python scripts/evaluate_normal_validation.py
```

The workflow reuses saved training categorical vocabularies, enforces strictly prior-day normal-only histories and exact ID alignment, and writes feature/prediction artifacts under `~/recom/`. By the frozen long-view pair-weighted-AUC rule, the selected LightGBM configuration is FS1 deep. This is a normal-validation selection, not a final randomized-test or cross-architecture conclusion. See `docs/milestones/M3.3.md` and `reports/m3_3_normal_validation.md`.

## M4.1 neural infrastructure

Fit the normal-training-only neural preprocessor and run the two-update shared-bottom/MMoE infrastructure smoke check with:

```bash
python scripts/smoke_neural_infrastructure.py
```

Neural inputs combine frozen FS3 dense fields, separate user/video ID embeddings, and masked-mean pooling of the aligned last-20 normal-policy video history. The three heads are long view, like, and tab-1 valid play; literal click remains supplementary. Preprocessing and smoke checkpoints are stored outside Git under `~/recom/models/neural/infrastructure/v1/`. No production neural model is trained in M4.1. See `docs/milestones/M4.1.md` and `reports/m4_1_neural_infrastructure.md`.

## M4.2 shared-bottom training

Train one approved production seed with:

```bash
python scripts/train_shared_bottom.py --seed 0
```

Seeds 0, 1 and 2 use the fixed common neural budget of 7,060 optimizer updates, batch size 512 and `drop_last=False`, corresponding to exactly 3,612,405 examples per seed. Bundles are stored under `~/recom/models/neural/shared_bottom/v1/`; see `docs/milestones/M4.2.md` and `reports/m4_2_shared_bottom_benchmark.md`.

## M4.3 MMoE training

Train one frozen production MMoE seed with:

```bash
python scripts/train_mmoe.py --seed 0
```

Seeds 0, 1 and 2 use the same data, masks, preprocessing, and 7,060-update budget as shared-bottom. Aggregate gate/expert-use diagnostics and checksum-bound bundles are stored under `~/recom/models/neural/mmoe/v1/`; see `docs/milestones/M4.3.md` and `reports/m4_3_mmoe_training.md`.

## M4.4 MMoE ablations

Train one frozen ablation seed with:

```bash
python scripts/train_mmoe_ablation.py --ablation dense_only --seed 0
```

The only allowed ablations are dense-only, no-history, and no-aggregates. All nine seed bundles use the common 7,060-update budget and are stored under `~/recom/models/neural/ablations/v1/`. Run `python scripts/summarize_mmoe_ablations.py` to rebuild and verify the aggregate manifest. See `docs/milestones/M4.4.md` and `reports/m4_4_mmoe_ablations.md`.

## M4.5 neural validation and candidate freeze

Run the immutable neural checkpoints on the authorized normal-validation data and freeze the 12-configuration candidate population with:

```bash
python scripts/evaluate_neural_validation.py
```

The workflow reuses the training-fitted neural adapter, builds strictly earlier normal-policy last-20 histories, evaluates all three seeds, and writes large predictions plus the candidate manifest under `~/recom/predictions/validation/v1/`. See `docs/milestones/M4.5.md` and `reports/m4_5_neural_validation.md`.

## M5.1 RQ3 training-data preparation

Prepare the frozen FS1-deep marginal-reweighting, normal-plus-random fusion, and random-only training inputs with:

```bash
python scripts/prepare_rq3_training.py
```

The workflow verifies the untouched M4.5-selected baseline, creates strategy-specific adapters, and writes large aligned artifacts under `~/recom/processed/rq3/v1/`. It does not train a model. See `docs/milestones/M5.1.md` and `reports/m5_1_rq3_preparation.md`.

## M5.2 RQ3 LightGBM training

Train one of the three frozen FS1-deep RQ3 alternatives with:

```bash
python scripts/train_rq3_lightgbm.py --arm marginal_reweighting
```

Valid arms are `marginal_reweighting`, `joint`, and `random_only`. Each uses the same 300-round deep-tree configuration and writes a checksum-bound bundle under `~/recom/models/lightgbm/rq3/v1/`. Run `python scripts/summarize_rq3_lightgbm.py` to verify the completed three-model population. See `docs/milestones/M5.2.md` and `reports/m5_2_rq3_lightgbm_training.md`.

## M5.3 pre-final freeze

Verify all frozen candidates and RQ3 bundles, then rebuild the outcome-free final-evaluation readiness manifest with:

```bash
python scripts/freeze_m5_prefinal.py
```

The contender set and readiness manifest are stored under `~/recom/processed/prefinal/v1/`. M5.3 does not materialize final features or open held-out outcomes. See `docs/m5_3_statistical_amendment_v1.md`, `docs/milestones/M5.3.md`, and `reports/m5_3_prefinal_readiness.md`.

## Final results

The completed RQ1/RQ2/RQ3 study, frozen statistical results, limitations, and publication figures are summarized in [`reports/final_research_report.md`](reports/final_research_report.md).

Key findings are exposure-dependent: the normal-validation-selected FS1-deep model has positive randomized-exposure selection regret across all 12 candidates, but the frozen contender-only interval includes zero. Joint and random-only RQ3 training substantially improve randomized-exposure ranking over the normal-only baseline, while marginal-frequency reweighting does not improve primary AUC. These are offline results and are not claims of causal policy effects or online business uplift.

Rebuild the compact result tables and figures from the checksum-verified external artifacts with:

```bash
MPLCONFIGDIR=/tmp/kuairand-mpl PYTHONPATH=src .venv/bin/python scripts/build_results_package.py
```

The public repository intentionally excludes KuaiRand-Pure data, processed tables, predictions, categorical vocabularies, and trained model weights.
