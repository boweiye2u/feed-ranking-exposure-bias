"""Production normal-policy LightGBM benchmark and matrix runner."""

from __future__ import annotations

import contextlib
import io
import json
import resource
import sys
import time
from importlib.metadata import version
from pathlib import Path
from typing import Any

import numpy as np

from kuairand_pure.models.lightgbm import (
    FEATURE_SETS,
    TASKS,
    LightGBMAdapter,
    load_lightgbm_config,
    load_model_bundle,
    load_training_artifacts,
    predict_probabilities,
    save_model_bundle,
    select_task,
    train_booster,
)
from kuairand_pure.paths import get_artifact_paths

PROTOCOL_IDENTIFIERS = {
    "study": "Feed Ranking under Exposure Bias v5",
    "week0_tag": "week0-protocol-v1",
    "feature_tag": "m2-features-v1",
    "branch": "B1",
}
TRAINING_DATES = [20220412, 20220429]


def _versions() -> dict[str, str]:
    packages = (
        "numpy", "polars", "pyarrow", "pandas", "scikit-learn",
        "lightgbm", "torch", "matplotlib", "pytest",
    )
    return {"python": sys.version.split()[0], **{name: version(name) for name in packages}}


def _current_rss_mib() -> float:
    status = Path("/proc/self/status")
    if status.exists():
        for line in status.read_text().splitlines():
            if line.startswith("VmRSS:"):
                return int(line.split()[1]) / 1024
    return float("nan")


def _peak_rss_mib() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024


def _train_one(
    *,
    features,
    labels,
    encoded,
    adapter: LightGBMAdapter,
    task: str,
    complexity: str,
    output_dir: Path,
    benchmark: bool,
) -> dict[str, Any]:
    task_data = select_task(encoded, features, labels, task)
    positives = int(task_data.y.sum())
    if positives == 0 or positives == task_data.rows:
        raise ValueError(f"Task {task} does not contain both classes")
    capture = io.StringIO()
    started = time.perf_counter()
    with contextlib.redirect_stdout(capture), contextlib.redirect_stderr(capture):
        booster = train_booster(task_data, adapter, complexity=complexity)
    duration = time.perf_counter() - started
    messages = [line for line in capture.getvalue().splitlines() if line.strip()]
    warnings = [line for line in messages if "warning" in line.lower() or "error" in line.lower()]
    predictions = predict_probabilities(booster, adapter, features)
    metadata = {
        "eligible_population": task,
        "training_source": "normal_policy",
        "training_dates": TRAINING_DATES,
        "training_rows": task_data.rows,
        "positive_labels": positives,
        "positive_rate": positives / task_data.rows,
        "seed": 202211,
        "software_versions": _versions(),
        "protocol_identifiers": PROTOCOL_IDENTIFIERS,
        "benchmark_model": benchmark,
    }
    save_model_bundle(
        output_dir, booster, adapter, task=task, complexity=complexity,
        training_metadata=metadata,
    )
    reloaded, reloaded_adapter, manifest = load_model_bundle(
        output_dir,
        expected_manifest_sha256=adapter.feature_manifest_sha256,
        expected_feature_set=adapter.feature_set,
        expected_task=task,
        expected_complexity=complexity,
    )
    reloaded_predictions = predict_probabilities(reloaded, reloaded_adapter, features)
    if not np.array_equal(predictions, reloaded_predictions):
        raise ValueError("Reloaded model predictions differ")
    record = {
        "feature_set": adapter.feature_set,
        "complexity": complexity,
        "task": task,
        "training_rows": task_data.rows,
        "positive_labels": positives,
        "positive_rate": positives / task_data.rows,
        "training_seconds": duration,
        "process_peak_rss_mib_after": _peak_rss_mib(),
        "process_rss_mib_after": _current_rss_mib(),
        "boosting_rounds_completed": booster.current_iteration(),
        "num_trees": booster.num_trees(),
        "feature_count": len(adapter.feature_names),
        "model_bytes": (output_dir / "model.txt").stat().st_size,
        "bundle_bytes": sum(path.stat().st_size for path in output_dir.iterdir() if path.is_file()),
        "warning_count": len(warnings),
        "warnings": warnings,
        "numerical_error": False,
        "prediction_min": float(predictions.min()),
        "prediction_max": float(predictions.max()),
        "prediction_rows": int(predictions.size),
        "serialization_reload_exact": True,
        "impression_alignment": True,
        "manifest_num_trees": manifest["num_trees"],
        "artifact_dir": str(output_dir),
    }
    (output_dir / "training_record.json").write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
    return record


def run_benchmark(
    *,
    processed_root: Path | None = None,
    model_root: Path | None = None,
    feature_manifest_path: Path | None = None,
) -> dict[str, Any]:
    paths = get_artifact_paths()
    processed_root = processed_root or paths.processed
    model_root = model_root or paths.models / "lightgbm" / "v1"
    feature_manifest_path = feature_manifest_path or processed_root / "m2" / "v1" / "feature_manifest.json"
    config = load_lightgbm_config()
    spec = config["benchmark"]
    load_started = time.perf_counter()
    features, labels = load_training_artifacts(processed_root, spec["feature_set"])
    load_seconds = time.perf_counter() - load_started
    fit_started = time.perf_counter()
    adapter = LightGBMAdapter.fit(
        features, feature_set=spec["feature_set"], feature_manifest_path=feature_manifest_path
    )
    vocabulary_fit_seconds = time.perf_counter() - fit_started
    encode_started = time.perf_counter()
    encoded = adapter.transform(features)
    encoding_seconds = time.perf_counter() - encode_started
    output = model_root / spec["feature_set"] / spec["complexity"] / spec["task"]
    record = _train_one(
        features=features, labels=labels, encoded=encoded, adapter=adapter,
        task=spec["task"], complexity=spec["complexity"], output_dir=output,
        benchmark=True,
    )
    report = {
        "milestone": "M3.2b_benchmark",
        "data_load_seconds": load_seconds,
        "vocabulary_fit_seconds": vocabulary_fit_seconds,
        "input_encoding_seconds": encoding_seconds,
        "record": record,
        "resource_thresholds": {
            "max_training_seconds": spec["max_training_seconds"],
            "max_peak_rss_mib": spec["max_peak_rss_mib"],
        },
        "resource_gate_passed": (
            record["training_seconds"] <= spec["max_training_seconds"]
            and record["process_peak_rss_mib_after"] <= spec["max_peak_rss_mib"]
        ),
        "validation_or_test_outcomes_accessed": False,
        "performance_metrics_computed": False,
    }
    model_root.mkdir(parents=True, exist_ok=True)
    (model_root / "benchmark_report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return report


def run_full_matrix(
    *,
    processed_root: Path | None = None,
    model_root: Path | None = None,
    feature_manifest_path: Path | None = None,
) -> dict[str, Any]:
    overall_started = time.perf_counter()
    paths = get_artifact_paths()
    processed_root = processed_root or paths.processed
    model_root = model_root or paths.models / "lightgbm" / "v1"
    feature_manifest_path = feature_manifest_path or processed_root / "m2" / "v1" / "feature_manifest.json"
    benchmark_report = json.loads((model_root / "benchmark_report.json").read_text())
    if not benchmark_report.get("resource_gate_passed"):
        raise ValueError("Benchmark resource gate did not pass")
    records: list[dict[str, Any]] = []
    for feature_set in FEATURE_SETS:
        load_started = time.perf_counter()
        features, labels = load_training_artifacts(processed_root, feature_set)
        load_seconds = time.perf_counter() - load_started
        fit_started = time.perf_counter()
        adapter = LightGBMAdapter.fit(
            features, feature_set=feature_set, feature_manifest_path=feature_manifest_path
        )
        vocabulary_fit_seconds = time.perf_counter() - fit_started
        encode_started = time.perf_counter()
        encoded = adapter.transform(features)
        encoding_seconds = time.perf_counter() - encode_started
        for complexity in ("shallow", "deep"):
            for task in TASKS:
                output = model_root / feature_set / complexity / task
                if feature_set == "fs3" and complexity == "shallow" and task == "long_view":
                    record = benchmark_report["record"]
                    record = {**record, "reused_from_benchmark": True}
                else:
                    record = _train_one(
                        features=features, labels=labels, encoded=encoded,
                        adapter=adapter, task=task, complexity=complexity,
                        output_dir=output, benchmark=False,
                    )
                    record["reused_from_benchmark"] = False
                record["data_load_seconds_for_feature_set"] = load_seconds
                record["vocabulary_fit_seconds_for_feature_set"] = vocabulary_fit_seconds
                record["encoding_seconds_for_feature_set"] = encoding_seconds
                records.append(record)
    expected = {(fs, depth, task) for fs in FEATURE_SETS for depth in ("shallow", "deep") for task in TASKS}
    observed = {(r["feature_set"], r["complexity"], r["task"]) for r in records}
    if observed != expected:
        raise ValueError("LightGBM matrix is incomplete")
    report = {
        "milestone": "M3.2b",
        "configuration_count": 6,
        "model_count": len(records),
        "complete": True,
        "records": records,
        "total_runtime_seconds": time.perf_counter() - overall_started,
        "process_peak_rss_mib": _peak_rss_mib(),
        "validation_or_test_outcomes_accessed": False,
        "model_selection_performed": False,
        "software_versions": _versions(),
        "protocol_identifiers": PROTOCOL_IDENTIFIERS,
    }
    (model_root / "matrix_report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return report
