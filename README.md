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
