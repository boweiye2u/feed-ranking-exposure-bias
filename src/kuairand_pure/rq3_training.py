"""Production training and integrity checks for the three frozen RQ3 arms."""

from __future__ import annotations

import contextlib
import hashlib
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
    load_model_bundle,
    save_model_bundle,
    train_booster,
)
from kuairand_pure.paths import get_artifact_paths
from kuairand_pure.rq3 import (
    EXPECTED_NORMAL_ROWS,
    STRATEGIES,
    _sha256,
    load_prepared_rq3_task,
    load_rq3_config,
    weight_diagnostics,
)

EXPECTED_ROWS = {
    "marginal_reweighting": EXPECTED_NORMAL_ROWS,
    "joint": 1_060_414,
    "random_only": 337_933,
}
EXPECTED_POSITIVES = {
    "marginal_reweighting": 239_795,
    "joint": 266_992,
    "random_only": 27_197,
}


def _versions() -> dict[str, str]:
    packages = ("numpy", "polars", "pyarrow", "pandas", "scikit-learn", "lightgbm")
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


def _baseline_hashes(paths) -> dict[str, str]:
    directory = paths.models / "lightgbm" / "v1" / "fs1" / "deep" / "long_view"
    return {
        "model_sha256": _sha256(directory / "model.txt"),
        "adapter_sha256": _sha256(directory / "adapter.json"),
        "manifest_sha256": _sha256(directory / "model_manifest.json"),
    }


def validate_training_record(record: dict[str, Any], arm: str) -> None:
    if arm not in STRATEGIES or record.get("arm") != arm:
        raise ValueError("RQ3 training record arm is incompatible")
    if record.get("training_rows") != EXPECTED_ROWS[arm]:
        raise ValueError("RQ3 model training-row count drifted")
    if record.get("positive_labels") != EXPECTED_POSITIVES[arm]:
        raise ValueError("RQ3 model positive-label count drifted")
    if record.get("context_is_rand_included") != (arm == "joint"):
        raise ValueError("RQ3 model source-indicator policy drifted")
    expected_weighting = "approved_marginal_exposure" if arm == "marginal_reweighting" else "uniform"
    if record.get("sample_weight_policy") != expected_weighting:
        raise ValueError("RQ3 model sample-weight policy drifted")
    if record.get("boosting_rounds_completed") != 300:
        raise ValueError("RQ3 model did not complete exactly 300 rounds")
    if record.get("serialization_reload_exact") is not True:
        raise ValueError("RQ3 model reload predictions are not exact")
    if record.get("baseline_byte_identical") is not True:
        raise ValueError("Frozen normal-only baseline changed")
    if record.get("randomized_validation_outcomes_accessed") is not False:
        raise ValueError("Randomized-validation outcome embargo was violated")
    if record.get("final_test_outcomes_accessed") is not False:
        raise ValueError("Final-test outcome embargo was violated")


def train_rq3_arm(
    arm: str,
    *,
    processed_root: Path | None = None,
    model_output_root: Path | None = None,
    feature_manifest_path: Path | None = None,
) -> dict[str, Any]:
    """Train one frozen RQ3 alternative; never trains the baseline bundle."""
    if arm not in STRATEGIES:
        raise ValueError(f"Unknown RQ3 arm: {arm}")
    paths = get_artifact_paths()
    processed_root = processed_root or paths.processed
    model_output_root = model_output_root or paths.models / "lightgbm" / "rq3" / "v1"
    feature_manifest_path = feature_manifest_path or processed_root / "m2" / "v1" / "feature_manifest.json"
    config = load_rq3_config()
    baseline_before = _baseline_hashes(paths)
    if baseline_before["model_sha256"] != config["baseline"]["model_sha256"]:
        raise ValueError("Frozen normal-only baseline model checksum drifted before RQ3 training")
    if baseline_before["adapter_sha256"] != config["baseline"]["adapter_sha256"]:
        raise ValueError("Frozen normal-only baseline adapter checksum drifted before RQ3 training")

    task, adapter, strategy_manifest = load_prepared_rq3_task(
        arm, processed_root=processed_root, feature_manifest_path=feature_manifest_path
    )
    positives = int(task.y.sum())
    if task.rows != EXPECTED_ROWS[arm] or positives != EXPECTED_POSITIVES[arm]:
        raise ValueError("Prepared RQ3 task counts differ from the audited population")
    if positives == 0 or positives == task.rows:
        raise ValueError("RQ3 target must contain both classes")
    if (arm == "marginal_reweighting") != (task.sample_weight is not None):
        raise ValueError("RQ3 sample-weight policy differs from the frozen arm definition")

    messages = io.StringIO()
    started = time.perf_counter()
    with contextlib.redirect_stdout(messages), contextlib.redirect_stderr(messages):
        booster = train_booster(task, adapter, complexity="deep")
    training_seconds = time.perf_counter() - started
    output = [line for line in messages.getvalue().splitlines() if line.strip()]
    warnings = [line for line in output if "warning" in line.lower() or "error" in line.lower()]
    if booster.current_iteration() != 300 or booster.num_trees() != 300:
        raise ValueError("RQ3 LightGBM did not complete the frozen 300 rounds")
    predictions = np.asarray(booster.predict(task.X), dtype=np.float64)
    if predictions.shape != (task.rows,) or not np.isfinite(predictions).all():
        raise ValueError("RQ3 model produced invalid training predictions")
    if np.any((predictions < 0) | (predictions > 1)):
        raise ValueError("RQ3 prediction probabilities are outside [0,1]")

    output_dir = model_output_root / arm
    adapter_source_path = Path(strategy_manifest["adapter_path"])
    weight_details = None
    if task.sample_weight is not None:
        diagnostics = weight_diagnostics(task.sample_weight)
        if abs(diagnostics["effective_sample_size"] - 152_212.84920627272) > 1e-6:
            raise ValueError("RQ3 training weights differ from the M5.1 ESS")
        weight_details = {
            "sample_weight_sha256": strategy_manifest["sample_weights_sha256"],
            "effective_sample_size": diagnostics["effective_sample_size"],
            "effective_sample_fraction": diagnostics["effective_sample_fraction"],
            "maximum": diagnostics["maximum"],
            "top_1_percent_weight_share": diagnostics["top_1_percent_weight_share"],
            "top_10_percent_weight_share": diagnostics["top_10_percent_weight_share"],
        }
    metadata = {
        "milestone": "M5.2",
        "arm": arm,
        "training_source": {
            "marginal_reweighting": "normal_policy_weighted",
            "joint": "normal_plus_random_data_fusion",
            "random_only": "random_policy_only",
        }[arm],
        "training_dates": {
            "normal": [20220412, 20220429] if arm != "random_only" else None,
            "random": [20220422, 20220429] if arm != "marginal_reweighting" else None,
        },
        "training_rows": task.rows,
        "positive_labels": positives,
        "positive_rate": positives / task.rows,
        "sample_weight_policy": "approved_marginal_exposure" if task.sample_weight is not None else "uniform",
        "weight_details": weight_details,
        "prepared_strategy_manifest_sha256": _sha256(
            processed_root / "rq3" / "v1" / arm / "strategy_manifest.json"
        ),
        "seed": 202211,
        "protocol_identifiers": {
            "week0": "week0-protocol-v1",
            "features": "m2-features-v1",
            "candidates": "m4-candidates-v1",
            "study": "Feed Ranking under Exposure Bias v5",
        },
        "joint_random_evaluation_context_is_rand": 1 if arm == "joint" else None,
        "matched_rounds_limitation": "Matching 300 rounds does not equalize examples, distributions, or computation.",
        "software_versions": _versions(),
        "validation_selection_or_early_stopping": False,
        "randomized_validation_outcomes_accessed": False,
        "final_test_outcomes_accessed": False,
    }
    save_model_bundle(
        output_dir, booster, adapter,
        task="long_view", complexity="deep", training_metadata=metadata,
        adapter_source_path=adapter_source_path,
    )
    reloaded, reloaded_adapter, model_manifest = load_model_bundle(
        output_dir,
        expected_manifest_sha256=_sha256(feature_manifest_path),
        expected_feature_set="fs1", expected_task="long_view", expected_complexity="deep",
    )
    if reloaded_adapter.include_is_rand != (arm == "joint"):
        raise ValueError("Saved RQ3 bundle has an incompatible source-indicator adapter")
    reloaded_predictions = np.asarray(reloaded.predict(task.X), dtype=np.float64)
    if not np.array_equal(predictions, reloaded_predictions):
        raise ValueError("Reloaded RQ3 model predictions differ exactly")
    baseline_after = _baseline_hashes(paths)
    record = {
        "milestone": "M5.2",
        "arm": arm,
        "training_rows": task.rows,
        "positive_labels": positives,
        "positive_rate": positives / task.rows,
        "sample_weight_policy": metadata["sample_weight_policy"],
        "weight_details": weight_details,
        "feature_count": len(adapter.feature_names),
        "context_is_rand_included": adapter.include_is_rand,
        "training_seconds": training_seconds,
        "process_peak_rss_mib": _peak_rss_mib(),
        "process_rss_mib_after": _current_rss_mib(),
        "boosting_rounds_completed": booster.current_iteration(),
        "num_trees": booster.num_trees(),
        "prediction_min": float(predictions.min()),
        "prediction_max": float(predictions.max()),
        "predictions_finite": True,
        "warning_count": len(warnings),
        "warnings": warnings,
        "model_bytes": (output_dir / "model.txt").stat().st_size,
        "bundle_bytes": sum(path.stat().st_size for path in output_dir.iterdir() if path.is_file()),
        "model_sha256": model_manifest["model_sha256"],
        "adapter_sha256": model_manifest["adapter_sha256"],
        "vocabulary_sha256": model_manifest["vocabulary_sha256"],
        "adapter_matches_m5_1": model_manifest["adapter_sha256"] == strategy_manifest["adapter_sha256"],
        "serialization_reload_exact": True,
        "baseline_hashes_before": baseline_before,
        "baseline_hashes_after": baseline_after,
        "baseline_byte_identical": baseline_before == baseline_after,
        "hyperparameters": model_manifest["effective_parameters"],
        "artifact_dir": str(output_dir),
        "randomized_validation_outcomes_accessed": False,
        "final_test_outcomes_accessed": False,
        "production_neural_model_trained": False,
    }
    validate_training_record(record, arm)
    (output_dir / "training_record.json").write_text(
        json.dumps(record, indent=2, sort_keys=True) + "\n"
    )
    return record


def summarize_rq3_models(
    *,
    model_output_root: Path | None = None,
) -> dict[str, Any]:
    """Verify and summarize the exactly three completed RQ3 bundles."""
    paths = get_artifact_paths()
    model_output_root = model_output_root or paths.models / "lightgbm" / "rq3" / "v1"
    records = []
    for arm in STRATEGIES:
        path = model_output_root / arm / "training_record.json"
        if not path.is_file():
            raise ValueError(f"Missing RQ3 training record for {arm}")
        record = json.loads(path.read_text())
        validate_training_record(record, arm)
        records.append(record)
    if {record["arm"] for record in records} != set(STRATEGIES):
        raise ValueError("RQ3 model population is incomplete")
    baseline = _baseline_hashes(paths)
    config = load_rq3_config()
    if baseline["model_sha256"] != config["baseline"]["model_sha256"] or baseline["adapter_sha256"] != config["baseline"]["adapter_sha256"]:
        raise ValueError("Frozen normal-only baseline changed after RQ3 training")
    report = {
        "milestone": "M5.2",
        "model_count": 3,
        "complete": True,
        "records": records,
        "baseline_hashes": baseline,
        "baseline_preserved": True,
        "matched_boosting_rounds": 300,
        "matched_rounds_equalize_sample_size_or_compute": False,
        "normal_validation_used_for_tuning": False,
        "randomized_validation_outcomes_accessed": False,
        "final_test_outcomes_accessed": False,
    }
    model_output_root.mkdir(parents=True, exist_ok=True)
    (model_output_root / "training_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n"
    )
    return report
