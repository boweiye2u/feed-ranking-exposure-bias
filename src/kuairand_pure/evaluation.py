"""Frozen normal-validation feature construction and candidate evaluation."""

from __future__ import annotations

import hashlib
import json
import resource
import shutil
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl

from kuairand_pure.baselines.r0 import R0_INPUT_COLUMNS, score_r0
from kuairand_pure.data.alignment import (
    load_aligned_model_data,
    validate_row_alignment,
    write_artifact_manifest,
)
from kuairand_pure.data.fs1 import (
    LABEL_COLUMNS,
    RAW_FEATURE_COLUMNS,
    _bounded_label_scan,
    _impressions,
    _labels,
    _paths,
    _quality as fs1_quality,
    _static_tables,
    build_fs1_lazy,
)
from kuairand_pure.data.splits import SplitSpec, frozen_splits
from kuairand_pure.features.fs2 import (
    _quality as fs2_quality,
    build_fs2_lazy,
    compute_priors,
    daily_history_snapshots,
    load_fs2_config,
    load_normal_history,
    verify_frozen_priors,
)
from kuairand_pure.features.fs3 import (
    FS3_NEW_COLUMNS,
    _daily_affinity_snapshots,
    _daily_recency_snapshots,
    build_fs3_lazy,
    load_fs3_config,
    load_normal_structural_history,
    video_categories,
)
from kuairand_pure.metrics import evaluate_ranking_metrics
from kuairand_pure.models.lightgbm import (
    FEATURE_SETS,
    TASKS,
    load_model_bundle,
    predict_probabilities,
)
from kuairand_pure.paths import get_artifact_paths
from kuairand_pure.protocol import authorize_outcome_access, load_protocol

VALIDATION_START = 20220430
VALIDATION_END = 20220502
HISTORY_START = 20220409
HISTORY_END = 20220501
PAIR_CAP = 952
NDCG_K = 5


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_normal_validation_history(data_dir: Path) -> pl.DataFrame:
    """Load Apr 9-May 1 normal history under the existing split guards.

    Training history and the permitted earlier validation days are authorized
    separately, making it impossible for randomized or May 2 outcomes to enter.
    """
    training = load_normal_history(data_dir, HISTORY_START, 20220429)
    authorize_outcome_access(
        load_protocol(), source="normal", start=20220430, end=HISTORY_END,
        workflow="validation", purpose="selection",
    )
    dataset, files = _paths(data_dir)
    codes = dataset["source_codes"]
    validation_history = pl.concat(
        [
            _bounded_label_scan(
                files["normal_early"], codes["normal_early"],
                SplitSpec("normal_validation_history", "normal", 20220430, HISTORY_END, True),
            ),
            _bounded_label_scan(
                files["normal_late"], codes["normal_late"],
                SplitSpec("normal_validation_history", "normal", 20220430, HISTORY_END, True),
            ),
        ]
    ).select("user_id", "video_id", "date", "tab", *LABEL_COLUMNS).collect()
    history = pl.concat([training, validation_history])
    if history["date"].min() != HISTORY_START or history["date"].max() != HISTORY_END:
        raise ValueError("Normal-validation feature-history dates drifted")
    return history


def _assert_training_schema(validation_path: Path, training_path: Path, stage: str) -> None:
    observed = pl.scan_parquet(validation_path).collect_schema()
    expected = pl.scan_parquet(training_path).collect_schema()
    if observed != expected:
        raise ValueError(f"{stage} validation schema differs from frozen training schema")


def materialize_normal_validation_features(
    *,
    data_dir: Path | None = None,
    processed_root: Path | None = None,
    output_root: Path | None = None,
) -> dict[str, Any]:
    """Materialize FS1-FS3 and isolated labels for normal validation only."""
    started = time.perf_counter()
    paths = get_artifact_paths()
    data_dir = data_dir or paths.raw / "KuaiRand-Pure" / "data"
    processed_root = processed_root or paths.processed
    output_root = output_root or paths.processed / "validation" / "v1"
    spec = frozen_splits()["normal_validation"]
    if (spec.start, spec.end, spec.source) != (VALIDATION_START, VALIDATION_END, "normal"):
        raise ValueError("Frozen normal-validation split drifted")

    # FS1: static features and separately authorized/isolated labels.
    fs1_dir = output_root / "fs1" / "normal_validation"
    fs1_dir.mkdir(parents=True, exist_ok=True)
    fs1_path, label_path = fs1_dir / "features.parquet", fs1_dir / "labels.parquet"
    users, videos, user_features, video_features = _static_tables(data_dir)
    build_fs1_lazy(
        _impressions(data_dir, spec, RAW_FEATURE_COLUMNS), users, videos,
        user_features, video_features,
    ).sort("impression_id").sink_parquet(fs1_path)
    _labels(
        data_dir, spec, workflow="validation", purpose="selection"
    ).sort("impression_id").sink_parquet(label_path)
    if pl.scan_parquet(fs1_path).select(pl.len()).collect().item() != 61_548:
        raise ValueError("Normal-validation row count is not the audited 61,548")
    fs1_report = fs1_quality(fs1_path, label_path, spec)
    write_artifact_manifest(
        fs1_dir, stage="fs1", split="normal_validation",
        features_path=fs1_path, labels_path=label_path,
    )
    _assert_training_schema(
        fs1_path, processed_root / "fs1" / "v1" / "normal_train" / "features.parquet", "FS1"
    )

    # History stops at May 1. Every aggregate join independently enforces date < target date.
    history = load_normal_validation_history(data_dir)
    fs2_config = load_fs2_config()
    priors = compute_priors(
        history, fs2_config["prior_window"]["start"], fs2_config["prior_window"]["end"]
    )
    verify_frozen_priors(priors, fs2_config)
    user_snapshots = daily_history_snapshots(history, "user_id", "user")
    video_snapshots = daily_history_snapshots(history, "video_id", "video")

    fs2_dir = output_root / "fs2" / "normal_validation"
    fs2_dir.mkdir(parents=True, exist_ok=True)
    fs2_path, fs2_labels = fs2_dir / "features.parquet", fs2_dir / "labels.parquet"
    build_fs2_lazy(
        pl.scan_parquet(fs1_path), user_snapshots, video_snapshots,
        priors, float(fs2_config["fs2"]["alpha"]),
    ).sort("impression_id").sink_parquet(fs2_path)
    shutil.copy2(label_path, fs2_labels)
    fs2_report = fs2_quality(fs1_path, fs2_path, label_path, fs2_labels)
    write_artifact_manifest(
        fs2_dir, stage="fs2", split="normal_validation",
        features_path=fs2_path, labels_path=fs2_labels,
    )
    _assert_training_schema(
        fs2_path, processed_root / "fs2" / "v1" / "normal_train" / "features.parquet", "FS2"
    )

    fs3_config = load_fs3_config()
    missing_token = fs3_config["category"]["missing_token"]
    categories = video_categories(data_dir, missing_token)
    outcome_history = history.join(
        categories.select("video_id", "category__key"),
        on="video_id", how="left", validate="m:1",
    ).with_columns(pl.col("category__key").fill_null(missing_token))
    structural_history = load_normal_structural_history(
        data_dir, HISTORY_START, HISTORY_END
    ).join(
        categories.select("video_id", "category__key"),
        on="video_id", how="left", validate="m:1",
    ).with_columns(pl.col("category__key").fill_null(missing_token))
    fs3_dir = output_root / "fs3" / "normal_validation"
    fs3_dir.mkdir(parents=True, exist_ok=True)
    fs3_path, fs3_labels = fs3_dir / "features.parquet", fs3_dir / "labels.parquet"
    build_fs3_lazy(
        pl.scan_parquet(fs2_path), categories,
        _daily_affinity_snapshots(outcome_history),
        _daily_recency_snapshots(structural_history, ["user_id"], "user__recency_"),
        _daily_recency_snapshots(
            structural_history, ["user_id", "category__key"], "user_category__recency_"
        ),
        priors, float(fs3_config["fs3"]["alpha"]), missing_token,
    ).sort("impression_id").sink_parquet(fs3_path)
    shutil.copy2(label_path, fs3_labels)
    fs2_frame, fs3_frame, labels = (
        pl.read_parquet(fs2_path), pl.read_parquet(fs3_path), pl.read_parquet(fs3_labels)
    )
    validate_row_alignment(fs3_frame, labels)
    if not fs3_frame.select(fs2_frame.columns).equals(fs2_frame):
        raise ValueError("FS3 did not preserve FS2 validation values and order")
    if set(LABEL_COLUMNS).intersection(fs3_frame.columns):
        raise ValueError("Outcomes entered validation FS3")
    write_artifact_manifest(
        fs3_dir, stage="fs3", split="normal_validation",
        features_path=fs3_path, labels_path=fs3_labels,
    )
    _assert_training_schema(
        fs3_path, processed_root / "fs3" / "v1" / "normal_train" / "features.parquet", "FS3"
    )
    fs3_report = {
        "rows": fs3_frame.height,
        "fs2_values_and_order_preserved": True,
        "new_feature_schema": {name: str(fs3_frame.schema[name]) for name in FS3_NEW_COLUMNS},
    }
    report = {
        "milestone": "M3.3_validation_features",
        "split": "normal_validation",
        "dates": [VALIDATION_START, VALIDATION_END],
        "rows": 61_548,
        "history_source": "normal_only",
        "history_dates_loaded": [HISTORY_START, HISTORY_END],
        "aggregate_cutoff": "strictly_prior_calendar_day",
        "randomized_history_used": False,
        "same_day_outcomes_used": False,
        "final_test_outcomes_accessed": False,
        "schemas_equal_to_frozen_training": True,
        "fs1": fs1_report,
        "fs2": fs2_report,
        "fs3": fs3_report,
        "runtime_seconds": time.perf_counter() - started,
        "peak_rss_mib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
        "output_root": str(output_root),
    }
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "feature_build_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n"
    )
    return report


def _task_mask_and_label(labels: pl.DataFrame, task: str) -> tuple[np.ndarray, str]:
    if task == "long_view":
        return np.ones(labels.height, dtype=bool), "long_view"
    if task == "like":
        return np.ones(labels.height, dtype=bool), "is_like"
    if task == "valid_play":
        return (labels["click_semantics"] == "valid_play").to_numpy(), "is_click"
    if task == "literal_click":
        return (labels["click_semantics"] == "literal_click").to_numpy(), "is_click"
    raise ValueError(f"Unknown or pooled task: {task}")


def _evaluate_scores(
    features: pl.DataFrame,
    labels: pl.DataFrame,
    scores: np.ndarray,
    task: str,
) -> tuple[pl.DataFrame, dict[str, Any]]:
    validate_row_alignment(features, labels)
    if scores.shape != (features.height,):
        raise ValueError("Prediction row count differs from validation features")
    mask, label_name = _task_mask_and_label(labels, task)
    evaluation = labels.filter(pl.Series(mask))
    predictions = pl.DataFrame(
        {
            "impression_id": features["impression_id"].filter(pl.Series(mask)),
            "score": scores[mask],
        },
        schema_overrides={"impression_id": pl.UInt64, "score": pl.Float64},
    )
    if not predictions["impression_id"].equals(evaluation["impression_id"]):
        raise ValueError("Task prediction and label IDs differ by row")
    result = evaluate_ranking_metrics(
        evaluation["user_id"].to_numpy(), evaluation[label_name].to_numpy(),
        predictions["score"].to_numpy(), evaluation["time_ms"].to_numpy(),
        evaluation["video_id"].to_numpy(), pair_cap=PAIR_CAP, ndcg_k=NDCG_K,
    )
    return predictions, asdict(result)


def _write_predictions(
    predictions: pl.DataFrame,
    destination: Path,
    *,
    candidate: str,
    task: str,
    model_manifest: dict[str, Any] | None,
    version: str = "m3.3-normal-validation-v1",
) -> dict[str, Any]:
    destination.mkdir(parents=True, exist_ok=True)
    path = destination / "predictions.parquet"
    predictions.write_parquet(path)
    manifest = {
        "version": version,
        "candidate": candidate,
        "task": task,
        "split": "normal_validation",
        "dates": [VALIDATION_START, VALIDATION_END],
        "rows": predictions.height,
        "canonical_order": "impression_id_ascending",
        "prediction_sha256": _sha256(path),
        "prediction_schema": {name: str(dtype) for name, dtype in predictions.schema.items()},
        "pretrained_model_sha256": (
            model_manifest["model_sha256"] if model_manifest is not None else None
        ),
        "training_adapter_reused": model_manifest is not None,
        "labels_in_prediction_table": False,
    }
    (destination / "prediction_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )
    return {"path": str(path), "bytes": path.stat().st_size, "manifest": manifest}


def evaluate_normal_validation(
    *,
    processed_root: Path | None = None,
    model_root: Path | None = None,
    prediction_root: Path | None = None,
    feature_manifest_path: Path | None = None,
) -> dict[str, Any]:
    """Score unchanged M3.2b bundles and R0 on normal validation."""
    started = time.perf_counter()
    paths = get_artifact_paths()
    processed_root = processed_root or paths.processed
    model_root = model_root or paths.models / "lightgbm" / "v1"
    prediction_root = prediction_root or paths.predictions / "validation" / "v1"
    feature_manifest_path = feature_manifest_path or processed_root / "m2" / "v1" / "feature_manifest.json"
    manifest_sha256 = _sha256(feature_manifest_path)
    validation_root = processed_root / "validation" / "v1"
    records: list[dict[str, Any]] = []

    loaded: dict[str, tuple[pl.DataFrame, pl.DataFrame]] = {}
    for feature_set in FEATURE_SETS:
        source = validation_root / feature_set / "normal_validation"
        features, labels, _ = load_aligned_model_data(
            source / "features.parquet", source / "labels.parquet",
            expected_stage=feature_set, expected_split="normal_validation",
        )
        loaded[feature_set] = (features, labels)

    fs2_features, fs2_labels = loaded["fs2"]
    r0_inputs = fs2_features.select(*R0_INPUT_COLUMNS)
    r0_frame = score_r0(r0_inputs)
    r0_predictions, r0_metrics = _evaluate_scores(
        fs2_features, fs2_labels, r0_frame["r0_score"].to_numpy(), "long_view"
    )
    r0_artifact = _write_predictions(
        r0_predictions, prediction_root / "r0" / "long_view",
        candidate="r0", task="long_view", model_manifest=None,
    )
    records.append({
        "candidate": "r0", "model_family": "r0", "feature_set": "fs2",
        "complexity": None, "task": "long_view", "metrics": r0_metrics,
        "prediction_artifact": r0_artifact,
    })

    for feature_set in FEATURE_SETS:
        features, labels = loaded[feature_set]
        for complexity in ("shallow", "deep"):
            for task in TASKS:
                bundle = model_root / feature_set / complexity / task
                booster, adapter, model_manifest = load_model_bundle(
                    bundle, expected_manifest_sha256=manifest_sha256,
                    expected_feature_set=feature_set, expected_task=task,
                    expected_complexity=complexity,
                )
                scores = predict_probabilities(booster, adapter, features)
                predictions, metrics = _evaluate_scores(features, labels, scores, task)
                candidate = f"{feature_set}_{complexity}"
                artifact = _write_predictions(
                    predictions, prediction_root / candidate / task,
                    candidate=candidate, task=task, model_manifest=model_manifest,
                )
                records.append({
                    "candidate": candidate, "model_family": "lightgbm",
                    "feature_set": feature_set, "complexity": complexity,
                    "task": task, "metrics": metrics,
                    "prediction_artifact": artifact,
                    "model_bundle": str(bundle),
                    "model_sha256": model_manifest["model_sha256"],
                    "adapter_sha256": model_manifest["adapter_sha256"],
                })

    long_view = [
        record for record in records
        if record["model_family"] == "lightgbm" and record["task"] == "long_view"
    ]
    selected = max(long_view, key=lambda record: record["metrics"]["pair_weighted_auc"])
    by_key = {(r["feature_set"], r["complexity"]): r for r in long_view}
    increments = []
    for complexity in ("shallow", "deep"):
        for newer, older in (("fs2", "fs1"), ("fs3", "fs2")):
            increments.append({
                "complexity": complexity,
                "contrast": f"{newer}_minus_{older}",
                "pair_weighted_auc_difference": (
                    by_key[(newer, complexity)]["metrics"]["pair_weighted_auc"]
                    - by_key[(older, complexity)]["metrics"]["pair_weighted_auc"]
                ),
            })
    depth_differences = [
        {
            "feature_set": feature_set,
            "contrast": "deep_minus_shallow",
            "pair_weighted_auc_difference": (
                by_key[(feature_set, "deep")]["metrics"]["pair_weighted_auc"]
                - by_key[(feature_set, "shallow")]["metrics"]["pair_weighted_auc"]
            ),
        }
        for feature_set in FEATURE_SETS
    ]
    r0_record = records[0]
    report = {
        "milestone": "M3.3",
        "split": "normal_validation",
        "dates": [VALIDATION_START, VALIDATION_END],
        "records": records,
        "selected_lightgbm": {
            "selection_rule": "maximum normal-validation long-view pair-weighted within-user AUC",
            "candidate": selected["candidate"],
            "feature_set": selected["feature_set"],
            "complexity": selected["complexity"],
            "metric": selected["metrics"]["pair_weighted_auc"],
        },
        "long_view_feature_increments": increments,
        "long_view_depth_differences": depth_differences,
        "selected_minus_r0_pair_weighted_auc": (
            selected["metrics"]["pair_weighted_auc"]
            - r0_record["metrics"]["pair_weighted_auc"]
        ),
        "pretrained_models_modified_or_retrained": False,
        "training_vocabulary_reused": True,
        "validation_vocabulary_fitted": False,
        "randomized_validation_outcomes_accessed": False,
        "final_test_outcomes_accessed": False,
        "runtime_seconds": time.perf_counter() - started,
        "peak_rss_mib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
    }
    prediction_root.mkdir(parents=True, exist_ok=True)
    (prediction_root / "validation_results.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n"
    )
    return report
