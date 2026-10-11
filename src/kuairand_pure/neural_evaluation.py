"""M4.5 normal-validation scoring for frozen neural candidates."""

from __future__ import annotations

import hashlib
import json
import resource
import time
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
import torch

from kuairand_pure.data.alignment import load_aligned_model_data, write_artifact_manifest
from kuairand_pure.evaluation import (
    HISTORY_START,
    VALIDATION_END,
    VALIDATION_START,
    _evaluate_scores,
    _write_predictions,
)
from kuairand_pure.features.fs3 import (
    build_sequence_frame,
    load_fs3_config,
    load_normal_structural_history,
    video_categories,
)
from kuairand_pure.models.neural import MMOE_ABLATION_SPECS, MMoE, SharedBottomMLP
from kuairand_pure.paths import get_artifact_paths
from kuairand_pure.training.neural import load_checkpoint, move_batch, resolve_device
from kuairand_pure.training.neural_data import (
    TASK_NAMES,
    NeuralDataset,
    NeuralPreprocessor,
    build_neural_targets,
    load_neural_config,
    make_dataloader,
)

EXPECTED_ROWS = 61_548
EXPECTED_UPDATES = 7_060
EXPECTED_EXAMPLES = 3_612_405
PRIMARY_SEED = 0
SEEDS = (0, 1, 2)

NEURAL_CANDIDATES: dict[str, dict[str, Any]] = {
    "shared_bottom": {
        "architecture": "shared_bottom",
        "kind": "shared_bottom",
        "relative_root": ("shared_bottom", "v1"),
        "kwargs": {},
    },
    "mmoe_full": {
        "architecture": "mmoe",
        "kind": "mmoe",
        "relative_root": ("mmoe", "v1"),
        "kwargs": {},
    },
    **{
        f"mmoe_{name}": {
            "architecture": f"mmoe_{name}",
            "kind": "mmoe",
            "relative_root": ("ablations", "v1", name),
            "kwargs": spec["kwargs"],
        }
        for name, spec in MMOE_ABLATION_SPECS.items()
    },
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def materialize_neural_validation_sequence(
    *,
    data_dir: Path | None = None,
    processed_root: Path | None = None,
) -> dict[str, Any]:
    """Build the aligned validation history without reading engagement outcomes."""
    started = time.perf_counter()
    paths = get_artifact_paths()
    data_dir = data_dir or paths.raw / "KuaiRand-Pure" / "data"
    processed_root = processed_root or paths.processed
    directory = processed_root / "validation" / "v1" / "fs3" / "normal_validation"
    feature_path = directory / "features.parquet"
    label_path = directory / "labels.parquet"
    sequence_path = directory / "sequence.parquet"
    features = pl.read_parquet(feature_path)
    if features.height != EXPECTED_ROWS:
        raise ValueError("Normal-validation FS3 row count drifted")

    settings = load_fs3_config()
    sequence_settings = settings["sequence"]
    categories = video_categories(data_dir, settings["category"]["missing_token"])
    # Structural history intentionally includes validation-day normal exposures.
    # Strict timestamp filtering in build_sequence_frame excludes each target,
    # simultaneous events, and all future events; there are no outcome columns.
    history = load_normal_structural_history(data_dir, HISTORY_START, VALIDATION_END)
    sequence = build_sequence_frame(
        features.select("impression_id", "user_id", "time_ms"),
        history,
        set(categories["video_id"].to_list()),
        length=int(sequence_settings["length"]),
        padding_id=int(sequence_settings["padding_video_id"]),
        unknown_id=int(sequence_settings["unknown_video_id"]),
        known_offset=int(sequence_settings["known_video_id_offset"]),
        padding_time_ms=int(sequence_settings["padding_time_ms"]),
    ).sort("impression_id")
    sequence.write_parquet(sequence_path)
    manifest = write_artifact_manifest(
        directory,
        stage="fs3",
        split="normal_validation",
        features_path=feature_path,
        labels_path=label_path,
        sequence_path=sequence_path,
    )
    masks = np.asarray(sequence["history_mask"].to_list(), dtype=np.uint8)
    times = np.asarray(sequence["history_time_ms"].to_list(), dtype=np.int64)
    target_times = sequence["target_time_ms"].to_numpy()[:, None]
    if np.any((masks == 1) & (times >= target_times)):
        raise ValueError("Validation sequence contains a non-prior event")
    report = {
        "rows": sequence.height,
        "history_length": int(sequence_settings["length"]),
        "history_source": "normal_policy_structural_impressions_only",
        "history_dates_loaded": [HISTORY_START, VALIDATION_END],
        "cutoff": "strictly_earlier_time_ms",
        "randomized_history_used": False,
        "outcome_columns_read_for_sequence": False,
        "mean_history_length": float(sequence["history_length"].mean()),
        "zero_history_rows": int((sequence["history_length"] == 0).sum()),
        "tie_boundary_rows": int(sequence["history_tie_boundary_dropped"].sum()),
        "sequence_sha256": _sha256(sequence_path),
        "sequence_bytes": sequence_path.stat().st_size,
        "artifact_manifest": manifest,
        "runtime_seconds": time.perf_counter() - started,
    }
    (directory / "neural_sequence_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n"
    )
    return report


def _model_for(
    candidate: str,
    preprocessor: NeuralPreprocessor,
    config: dict[str, Any],
) -> torch.nn.Module:
    specification = NEURAL_CANDIDATES[candidate]
    model_type = SharedBottomMLP if specification["kind"] == "shared_bottom" else MMoE
    return model_type(preprocessor, config, **specification["kwargs"])


def _predict(
    model: torch.nn.Module,
    loader,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    ids: list[np.ndarray] = []
    scores: list[np.ndarray] = []
    with torch.inference_mode():
        for raw_batch in loader:
            batch = move_batch(raw_batch, device)
            probabilities = model.probabilities(batch)
            if not torch.isfinite(probabilities).all():
                raise ValueError("Neural validation produced non-finite probabilities")
            ids.append(raw_batch["impression_id"].numpy())
            scores.append(probabilities.cpu().numpy())
    all_ids = np.concatenate(ids)
    all_scores = np.concatenate(scores)
    if all_scores.shape != (EXPECTED_ROWS, len(TASK_NAMES)):
        raise ValueError("Neural prediction dimensions drifted")
    if np.any((all_scores < 0) | (all_scores > 1)):
        raise ValueError("Neural probabilities are outside [0, 1]")
    return all_ids, all_scores


def summarize_seed_metrics(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Summarize three frozen seeds without using the summary for selection."""
    metric_names = (
        "pair_weighted_auc",
        "per_user_average_auc",
        "capped_pair_weighted_auc",
        "pooled_auc",
        "log_loss",
        "ndcg_at_k",
    )
    summaries: list[dict[str, Any]] = []
    for candidate in NEURAL_CANDIDATES:
        for task in TASK_NAMES:
            relevant = sorted(
                (record for record in records if record["candidate"] == candidate and record["task"] == task),
                key=lambda record: record["seed"],
            )
            if [record["seed"] for record in relevant] != list(SEEDS):
                raise ValueError(f"Incomplete seed set for {candidate}/{task}")
            summary: dict[str, Any] = {"candidate": candidate, "task": task, "seeds": list(SEEDS)}
            for metric in metric_names:
                values = [float(record["metrics"][metric]) for record in relevant]
                summary[metric] = {
                    "mean": float(np.mean(values)),
                    "minimum": min(values),
                    "maximum": max(values),
                    "range": max(values) - min(values),
                }
            summaries.append(summary)
    return summaries


def validate_candidate_manifest(manifest: dict[str, Any]) -> None:
    candidates = manifest.get("candidates", [])
    identifiers = [candidate.get("candidate") for candidate in candidates]
    if len(candidates) != 12 or len(set(identifiers)) != 12:
        raise ValueError("Candidate freeze must contain exactly 12 unique configurations")
    if "esmm" in " ".join(str(value).lower() for value in identifiers):
        raise ValueError("ESMM is excluded from the candidate population")
    selected = manifest.get("selected_overall", {}).get("candidate")
    if selected not in identifiers:
        raise ValueError("Selected configuration is absent from candidate freeze")
    for candidate in candidates:
        if candidate.get("model_family") == "neural":
            seeds = [item.get("seed") for item in candidate.get("checkpoints", [])]
            if seeds != list(SEEDS):
                raise ValueError("Neural candidate checkpoint seeds are incomplete or unordered")


def evaluate_neural_normal_validation(
    *,
    processed_root: Path | None = None,
    model_root: Path | None = None,
    prediction_root: Path | None = None,
    feature_manifest_path: Path | None = None,
    device_name: str | None = None,
) -> dict[str, Any]:
    """Score all 15 immutable neural checkpoints on authorized validation labels."""
    started = time.perf_counter()
    paths = get_artifact_paths()
    processed_root = processed_root or paths.processed
    model_root = model_root or paths.models / "neural"
    prediction_root = prediction_root or paths.predictions / "validation" / "v1"
    feature_manifest_path = feature_manifest_path or processed_root / "m2" / "v1" / "feature_manifest.json"
    feature_manifest_sha256 = _sha256(feature_manifest_path)
    preprocessor_path = model_root / "infrastructure" / "v1" / "preprocessor.json"
    preprocessing_sha256 = _sha256(preprocessor_path)
    preprocessor = NeuralPreprocessor.load(
        preprocessor_path, expected_manifest_sha256=feature_manifest_sha256
    )
    if preprocessor.include_is_rand:
        raise ValueError("context__is_rand entered normal-only neural validation")
    config = load_neural_config()
    device = resolve_device(device_name or config["neural"]["device"])
    source = processed_root / "validation" / "v1" / "fs3" / "normal_validation"
    features, labels, sequence = load_aligned_model_data(
        source / "features.parquet",
        source / "labels.parquet",
        expected_stage="fs3",
        expected_split="normal_validation",
        sequence_path=source / "sequence.parquet",
    )
    assert sequence is not None
    arrays = preprocessor.transform(features, sequence)
    targets = build_neural_targets(labels)
    dataset = NeuralDataset(arrays, targets)
    records: list[dict[str, Any]] = []
    checkpoint_records: dict[str, list[dict[str, Any]]] = {name: [] for name in NEURAL_CANDIDATES}

    for candidate, specification in NEURAL_CANDIDATES.items():
        checkpoint_root = model_root.joinpath(*specification["relative_root"])
        for seed in SEEDS:
            model = _model_for(candidate, preprocessor, config)
            bundle = checkpoint_root / f"seed_{seed}"
            checkpoint_manifest = load_checkpoint(
                bundle,
                model=model,
                expected_architecture=specification["architecture"],
                expected_preprocessing_sha256=preprocessing_sha256,
                expected_feature_manifest_sha256=feature_manifest_sha256,
                device=device,
            )
            if checkpoint_manifest.get("seed") != seed:
                raise ValueError("Checkpoint seed differs from its bundle location")
            counters = checkpoint_manifest.get("training_counters", {})
            if counters != {"training_examples": EXPECTED_EXAMPLES, "optimizer_updates": EXPECTED_UPDATES}:
                raise ValueError("Checkpoint training budget differs from the frozen budget")
            loader = make_dataloader(
                dataset,
                batch_size=4096,
                seed=seed,
                shuffle=False,
            )
            predicted_ids, scores = _predict(model, loader, device)
            if not np.array_equal(predicted_ids.astype(arrays.impression_ids.dtype), arrays.impression_ids):
                raise ValueError("Neural predictions differ from FS3 impression order")
            checkpoint_records[candidate].append({
                "seed": seed,
                "bundle": str(bundle),
                "checkpoint_sha256": checkpoint_manifest["checkpoint_sha256"],
                "architecture": checkpoint_manifest["architecture"],
                "optimizer_updates": counters["optimizer_updates"],
                "examples_processed": counters["training_examples"],
            })
            for task_index, task in enumerate(TASK_NAMES):
                predictions, metrics = _evaluate_scores(
                    features, labels, scores[:, task_index].astype(np.float64), task
                )
                artifact = _write_predictions(
                    predictions,
                    prediction_root / "neural" / candidate / f"seed_{seed}" / task,
                    candidate=f"{candidate}_seed_{seed}",
                    task=task,
                    model_manifest={"model_sha256": checkpoint_manifest["checkpoint_sha256"]},
                    version="m4.5-neural-normal-validation-v1",
                )
                records.append({
                    "candidate": candidate,
                    "model_family": "neural",
                    "architecture": checkpoint_manifest["architecture"],
                    "seed": seed,
                    "task": task,
                    "metrics": metrics,
                    "checkpoint_bundle": str(bundle),
                    "checkpoint_sha256": checkpoint_manifest["checkpoint_sha256"],
                    "prediction_artifact": artifact,
                })

    result = {
        "milestone": "M4.5",
        "split": "normal_validation",
        "dates": [VALIDATION_START, VALIDATION_END],
        "rows": features.height,
        "tasks": list(TASK_NAMES),
        "primary_seed": PRIMARY_SEED,
        "records": records,
        "seed_summaries": summarize_seed_metrics(records),
        "checkpoints": checkpoint_records,
        "feature_manifest_sha256": feature_manifest_sha256,
        "preprocessing_path": str(preprocessor_path),
        "preprocessing_sha256": preprocessing_sha256,
        "training_preprocessor_reused": True,
        "validation_preprocessor_fitted": False,
        "context_is_rand_included": False,
        "device": str(device),
        "randomized_validation_outcomes_accessed": False,
        "final_test_outcomes_accessed": False,
        "runtime_seconds": time.perf_counter() - started,
        "peak_rss_mib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
    }
    prediction_root.mkdir(parents=True, exist_ok=True)
    (prediction_root / "neural_validation_results.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n"
    )
    return result


def freeze_candidate_population(
    neural_result: dict[str, Any],
    *,
    prediction_root: Path | None = None,
) -> dict[str, Any]:
    """Freeze the 12 eligible configurations and the seed-0 selection result."""
    paths = get_artifact_paths()
    prediction_root = prediction_root or paths.predictions / "validation" / "v1"
    lightgbm_result_path = prediction_root / "validation_results.json"
    lightgbm_result = json.loads(lightgbm_result_path.read_text())
    lgbm_long = [
        record for record in lightgbm_result["records"]
        if record["model_family"] == "lightgbm" and record["task"] == "long_view"
    ]
    r0_record = next(
        record for record in lightgbm_result["records"]
        if record["model_family"] == "r0" and record["task"] == "long_view"
    )
    neural_primary = [
        record for record in neural_result["records"]
        if record["seed"] == PRIMARY_SEED and record["task"] == "long_view"
    ]
    selection_records = [r0_record, *lgbm_long, *neural_primary]
    selected = max(selection_records, key=lambda record: record["metrics"]["pair_weighted_auc"])

    candidates: list[dict[str, Any]] = [{
        "candidate": "r0",
        "model_family": "r0",
        "tasks": ["long_view"],
        "scoring_rule": "video__hist_long_view_rate",
        "training_settings": {"fit": False, "alpha": 20.0},
        "validation_primary_metric": r0_record["metrics"]["pair_weighted_auc"],
    }]
    for record in lgbm_long:
        matching_tasks = [
            item for item in lightgbm_result["records"]
            if item["candidate"] == record["candidate"] and item["model_family"] == "lightgbm"
        ]
        candidates.append({
            "candidate": record["candidate"],
            "model_family": "lightgbm",
            "feature_set": record["feature_set"],
            "complexity": record["complexity"],
            "tasks": ["long_view", "like", "valid_play", "literal_click"],
            "model_bundles": {
                item["task"]: {
                    "path": item["model_bundle"],
                    "model_sha256": item["model_sha256"],
                    "adapter_sha256": item["adapter_sha256"],
                }
                for item in matching_tasks
            },
            "training_settings": "configs/lightgbm.toml frozen M3.2b settings",
            "validation_primary_metric": record["metrics"]["pair_weighted_auc"],
        })
    for candidate in NEURAL_CANDIDATES:
        primary = next(
            record for record in neural_primary if record["candidate"] == candidate
        )
        candidates.append({
            "candidate": candidate,
            "model_family": "neural",
            "tasks": list(TASK_NAMES),
            "checkpoints": neural_result["checkpoints"][candidate],
            "preprocessing_path": neural_result["preprocessing_path"],
            "preprocessing_sha256": neural_result["preprocessing_sha256"],
            "training_settings": {
                "configuration": "configs/neural.toml",
                "optimizer_updates": EXPECTED_UPDATES,
                "examples_processed": EXPECTED_EXAMPLES,
                "seeds": list(SEEDS),
            },
            "primary_seed": PRIMARY_SEED,
            "validation_primary_metric": primary["metrics"]["pair_weighted_auc"],
        })
    manifest = {
        "manifest_version": "m4-candidates-v1",
        "candidate_count": 12,
        "population_rule": "R0 + six LightGBM configurations + five neural configurations; no ESMM",
        "metric_definition": {
            "primary": "pair-weighted within-user AUC",
            "tie_credit": 0.5,
            "neural_primary_seed": PRIMARY_SEED,
            "robustness_seeds": [1, 2],
            "secondary": [
                "per-user-average AUC", "pair-capped AUC (cap 952)",
                "pooled AUC", "binary LogLoss", "NDCG@5",
            ],
        },
        "selected_overall": {
            "selection_rule": "maximum normal-validation long-view pair-weighted within-user AUC; neural seed 0 is primary",
            "candidate": selected["candidate"],
            "model_family": selected["model_family"],
            "seed": selected.get("seed"),
            "metric": selected["metrics"]["pair_weighted_auc"],
            "checkpoint_or_bundle": selected.get("checkpoint_bundle", selected.get("model_bundle")),
            "checksum": selected.get("checkpoint_sha256", selected.get("model_sha256")),
        },
        "candidates": candidates,
        "lightgbm_results_sha256": _sha256(lightgbm_result_path),
        "neural_results_sha256": _sha256(prediction_root / "neural_validation_results.json"),
        "protocol_identifiers": {
            "week0": "week0-protocol-v1",
            "features": "m2-features-v1",
        },
        "randomized_validation_outcomes_accessed": False,
        "final_test_outcomes_accessed": False,
    }
    validate_candidate_manifest(manifest)
    (prediction_root / "candidate_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )
    return manifest
