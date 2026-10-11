"""R0 strictly prior-day historical video-popularity baseline."""

from __future__ import annotations

import hashlib
import json
import resource
import time
import tomllib
from pathlib import Path
from typing import Any

import polars as pl

from kuairand_pure.data.alignment import (
    M2_BUILD_VERSION,
    load_aligned_feature_columns,
    validate_impression_ids,
)
from kuairand_pure.metrics import WithinUserAUC, within_user_auc
from kuairand_pure.paths import get_artifact_paths

R0_INPUT_COLUMNS = ("impression_id", "video__hist_long_view_rate")
R0_SCORE_COLUMN = "r0_score"
R0_PREDICTION_COLUMNS = ("impression_id", R0_SCORE_COLUMN)
R0_EVALUATION_COLUMNS = ("impression_id", "user_id", "long_view")


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def load_r0_config(path: Path | None = None) -> dict[str, Any]:
    config_path = path or _repo_root() / "configs" / "r0.toml"
    with config_path.open("rb") as stream:
        return tomllib.load(stream)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_r0_contract(
    r0_config: dict[str, Any],
    fs2_config_path: Path,
    feature_manifest_path: Path,
) -> dict[str, Any]:
    """Verify R0 references the frozen FS2 field rather than redefining it."""
    with fs2_config_path.open("rb") as stream:
        fs2 = tomllib.load(stream)
    manifest = json.loads(feature_manifest_path.read_text())
    settings = r0_config["r0"]
    history = r0_config["history_contract"]
    if settings["input_build_version"] != M2_BUILD_VERSION:
        raise ValueError("R0 input build version differs from the frozen M2 build")
    if settings["input_stage"] != "fs2":
        raise ValueError("R0 must read FS2")
    if tuple(r0_config["input"]["scoring_columns"]) != R0_INPUT_COLUMNS:
        raise ValueError("R0 scoring whitelist drifted")
    if float(history["alpha"]) != float(fs2["fs2"]["alpha"]):
        raise ValueError("R0 alpha differs from frozen FS2 alpha")
    if float(history["long_view_prior"]) != float(fs2["frozen_priors"]["long_view"]):
        raise ValueError("R0 prior differs from frozen FS2 long-view prior")
    if history["source"] != fs2["fs2"]["history_source"]:
        raise ValueError("R0 history source differs from FS2")
    if history["cutoff"] != "strictly_prior_calendar_day" or not fs2["fs2"]["strictly_prior_calendar_days"]:
        raise ValueError("R0/FS2 chronology contract drifted")
    if manifest.get("manifest_version") != M2_BUILD_VERSION:
        raise ValueError("R0 feature manifest has a stale build version")
    fields = {entry["name"]: entry for entry in manifest["feature_fields"]}
    score_feature = settings["score_feature"]
    if score_feature not in fields:
        raise ValueError("R0 score feature is absent from the M2 manifest")
    entry = fields[score_feature]
    if (
        entry["data_type"] != "Float64"
        or entry["introduced_in"] != "fs2"
        or entry["source"] != "strictly_prior_day_normal_policy_outcomes"
    ):
        raise ValueError("R0 score feature has an incompatible M2 manifest contract")
    return {
        "alpha": float(history["alpha"]),
        "long_view_prior": float(history["long_view_prior"]),
        "history_source": history["source"],
        "cutoff": history["cutoff"],
        "feature_manifest_sha256": _sha256(feature_manifest_path),
    }


def score_r0(features: pl.DataFrame) -> pl.DataFrame:
    """Pass through the frozen FS2 video rate using a strict input whitelist."""
    if tuple(features.columns) != R0_INPUT_COLUMNS:
        raise ValueError(
            f"R0 accepts exactly {R0_INPUT_COLUMNS}; received {tuple(features.columns)}"
        )
    validate_impression_ids(features, "R0 inputs")
    rates = features["video__hist_long_view_rate"]
    if rates.dtype != pl.Float64:
        raise ValueError("R0 score feature must be Float64")
    if rates.null_count() or rates.is_nan().sum():
        raise ValueError("R0 score feature contains missing values")
    if not rates.is_between(0.0, 1.0).all():
        raise ValueError("R0 scores must lie in [0, 1]")
    return features.select(
        "impression_id",
        pl.col("video__hist_long_view_rate").alias(R0_SCORE_COLUMN),
    )


def evaluate_aligned_r0_predictions(
    predictions: pl.DataFrame,
    evaluation: pl.DataFrame,
    *,
    pair_cap: int = 952,
) -> WithinUserAUC:
    """Adapt aligned R0 predictions to the frozen metric implementation."""
    if tuple(predictions.columns) != R0_PREDICTION_COLUMNS:
        raise ValueError("Unexpected R0 prediction schema")
    if tuple(evaluation.columns) != R0_EVALUATION_COLUMNS:
        raise ValueError("Unexpected R0 evaluation schema")
    validate_impression_ids(predictions, "R0 predictions")
    validate_impression_ids(evaluation, "R0 evaluation rows")
    if not predictions["impression_id"].equals(evaluation["impression_id"]):
        raise ValueError("R0 predictions and evaluation rows differ by impression ID")
    return within_user_auc(
        evaluation["user_id"].to_numpy(),
        evaluation["long_view"].to_numpy(),
        predictions[R0_SCORE_COLUMN].to_numpy(),
        pair_cap=pair_cap,
    )


def _score_diagnostics(
    predictions: pl.DataFrame, diagnostic_features: pl.DataFrame
) -> dict[str, Any]:
    if not predictions["impression_id"].equals(diagnostic_features["impression_id"]):
        raise ValueError("R0 diagnostic features are not aligned with predictions")
    score = pl.col(R0_SCORE_COLUMN)
    summary = predictions.select(
        pl.len().alias("rows"),
        pl.col("impression_id").n_unique().alias("unique_impressions"),
        score.null_count().alias("missing_scores"),
        score.is_nan().sum().alias("nan_scores"),
        score.min().alias("min"),
        score.quantile(0.01).alias("p01"),
        score.quantile(0.05).alias("p05"),
        score.quantile(0.25).alias("p25"),
        score.median().alias("median"),
        score.mean().alias("mean"),
        score.std().alias("std"),
        score.quantile(0.75).alias("p75"),
        score.quantile(0.95).alias("p95"),
        score.quantile(0.99).alias("p99"),
        score.max().alias("max"),
        score.n_unique().alias("distinct_scores"),
    ).row(0, named=True)
    zero_history = diagnostic_features.select(
        (pl.col("video__hist_impression_count") == 0).sum()
    ).item()
    return {
        **summary,
        "zero_video_history_rows": zero_history,
        "zero_video_history_share": zero_history / predictions.height,
        "scores_in_unit_interval": True,
    }


def materialize_r0_predictions(
    fs2_dir: Path | None = None,
    m2_manifest_path: Path | None = None,
    output_dir: Path | None = None,
) -> dict[str, Any]:
    """Write authorized training predictions without projecting outcomes."""
    started = time.perf_counter()
    paths = get_artifact_paths()
    fs2_dir = fs2_dir or paths.processed / "fs2" / "v1"
    m2_manifest_path = m2_manifest_path or paths.processed / "m2" / "v1" / "feature_manifest.json"
    config = load_r0_config()
    settings = config["r0"]
    contract = verify_r0_contract(
        config,
        _repo_root() / "configs" / "fs2.toml",
        m2_manifest_path,
    )
    output_dir = output_dir or paths.predictions / settings["output_subdirectory"]
    split_reports: dict[str, Any] = {}
    for split in settings["authorized_splits"]:
        source = fs2_dir / split
        scoring_features = load_aligned_feature_columns(
            source / "features.parquet",
            source / "labels.parquet",
            required_columns=R0_INPUT_COLUMNS,
            expected_stage="fs2",
            expected_split=split,
        )
        diagnostic_columns = tuple(config["input"]["diagnostic_columns"])
        diagnostic_features = load_aligned_feature_columns(
            source / "features.parquet",
            source / "labels.parquet",
            required_columns=diagnostic_columns,
            expected_stage="fs2",
            expected_split=split,
        )
        predictions = score_r0(scoring_features)
        destination = output_dir / split
        destination.mkdir(parents=True, exist_ok=True)
        prediction_path = destination / "predictions.parquet"
        predictions.write_parquet(prediction_path)
        diagnostics = _score_diagnostics(predictions, diagnostic_features)
        input_manifest_path = source / "artifact_manifest.json"
        prediction_manifest = {
            "build_version": settings["version"],
            "baseline": settings["name"],
            "split": split,
            "canonical_order": "impression_id_ascending",
            "rows": predictions.height,
            "input_build_version": settings["input_build_version"],
            "input_stage": settings["input_stage"],
            "input_artifact_manifest_sha256": _sha256(input_manifest_path),
            "input_feature_whitelist": list(R0_INPUT_COLUMNS),
            "score_column": R0_SCORE_COLUMN,
            "prediction_schema": {
                name: str(dtype) for name, dtype in predictions.schema.items()
            },
            "outcome_columns_read": False,
        }
        (destination / "prediction_manifest.json").write_text(
            json.dumps(prediction_manifest, indent=2, sort_keys=True) + "\n"
        )
        split_reports[split] = {
            "prediction_path": str(prediction_path),
            "prediction_bytes": prediction_path.stat().st_size,
            "prediction_manifest": prediction_manifest,
            "diagnostics": diagnostics,
        }
    report = {
        "milestone": "M3.1",
        "baseline": "R0",
        "scoring_rule": "r0_score = video__hist_long_view_rate",
        "contract": contract,
        "authorized_splits_only": list(settings["authorized_splits"]),
        "outcome_columns_read": False,
        "random_outcomes_used": False,
        "held_out_outcomes_accessed": False,
        "evaluation_metrics_computed": False,
        "split_reports": split_reports,
        "runtime_seconds": time.perf_counter() - started,
        "peak_rss_mib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "data_quality.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n"
    )
    return report
