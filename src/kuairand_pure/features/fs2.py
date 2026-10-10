"""Point-in-time normal-history aggregates for FS2."""

from __future__ import annotations

import hashlib
import json
import resource
import shutil
import time
import tomllib
from pathlib import Path
from typing import Any

import polars as pl

from kuairand_pure.data.alignment import write_artifact_manifest
from kuairand_pure.data.fs1 import (
    LABEL_COLUMNS,
    METADATA_COLUMNS,
    _bounded_label_scan,
    _load_toml,
    _paths,
)
from kuairand_pure.data.splits import SplitSpec, frozen_splits
from kuairand_pure.paths import get_artifact_paths
from kuairand_pure.protocol import authorize_outcome_access, load_protocol

COUNT_SUFFIXES = (
    "impression_count",
    "long_view_positive_count",
    "like_positive_count",
    "valid_play_impression_count",
    "valid_play_positive_count",
    "literal_click_impression_count",
    "literal_click_positive_count",
)
RATE_SUFFIXES = (
    "long_view_rate",
    "like_rate",
    "valid_play_rate",
    "literal_click_rate",
)
SCOPES = ("user", "video")


def load_fs2_config(path: Path | None = None) -> dict[str, Any]:
    config_path = path or Path(__file__).resolve().parents[3] / "configs" / "fs2.toml"
    with config_path.open("rb") as stream:
        return tomllib.load(stream)


def load_normal_history(data_dir: Path, start: int, end: int) -> pl.DataFrame:
    """Load only authorized normal outcomes for burn-in/training history."""
    authorize_outcome_access(
        load_protocol(),
        source="normal",
        start=start,
        end=end,
        workflow="training",
        purpose="features",
    )
    dataset, files = _paths(data_dir)
    codes = dataset["source_codes"]
    spec = SplitSpec("normal_feature_history", "normal", start, end, True)
    return pl.concat(
        [
            _bounded_label_scan(files["normal_early"], codes["normal_early"], spec),
            _bounded_label_scan(files["normal_late"], codes["normal_late"], spec),
        ]
    ).select("user_id", "video_id", "date", "tab", *LABEL_COLUMNS).collect()


def compute_priors(history: pl.DataFrame, start: int, end: int) -> dict[str, float | int]:
    """Compute task-specific global priors from the frozen normal burn-in."""
    burn_in = history.filter(pl.col("date").is_between(start, end))
    if burn_in.is_empty():
        raise ValueError("Frozen prior window is empty")
    valid = pl.col("tab") == 1
    literal = pl.col("tab") == 0
    counts = burn_in.select(
        pl.len().alias("impressions"),
        pl.col("long_view").sum().alias("long_view_positives"),
        pl.col("is_like").sum().alias("like_positives"),
        valid.sum().alias("valid_play_impressions"),
        (valid & (pl.col("is_click") == 1)).sum().alias("valid_play_positives"),
        literal.sum().alias("literal_click_impressions"),
        (literal & (pl.col("is_click") == 1)).sum().alias("literal_click_positives"),
    ).row(0, named=True)
    for denominator in ("impressions", "valid_play_impressions", "literal_click_impressions"):
        if not counts[denominator]:
            raise ValueError(f"Zero denominator for frozen prior: {denominator}")
    return {
        **{name: int(value) for name, value in counts.items()},
        "long_view": counts["long_view_positives"] / counts["impressions"],
        "like": counts["like_positives"] / counts["impressions"],
        "valid_play": counts["valid_play_positives"] / counts["valid_play_impressions"],
        "literal_click": counts["literal_click_positives"] / counts["literal_click_impressions"],
    }


def verify_frozen_priors(observed: dict[str, float | int], config: dict[str, Any]) -> None:
    frozen = config.get("frozen_priors")
    if frozen is None:
        raise ValueError("FS2 frozen_priors are not configured")
    for name in ("long_view", "like", "valid_play", "literal_click"):
        if abs(float(observed[name]) - float(frozen[name])) > 1e-15:
            raise ValueError(f"Frozen {name} prior drifted")


def daily_history_snapshots(history: pl.DataFrame, entity: str, scope: str) -> pl.DataFrame:
    """Aggregate normal history once per entity/day and cumulatively sum it."""
    valid = pl.col("tab") == 1
    literal = pl.col("tab") == 0
    daily = (
        history.group_by(entity, "date")
        .agg(
            pl.len().cast(pl.Int64).alias("impression_count"),
            pl.col("long_view").sum().cast(pl.Int64).alias("long_view_positive_count"),
            pl.col("is_like").sum().cast(pl.Int64).alias("like_positive_count"),
            valid.sum().cast(pl.Int64).alias("valid_play_impression_count"),
            (valid & (pl.col("is_click") == 1)).sum().cast(pl.Int64).alias("valid_play_positive_count"),
            literal.sum().cast(pl.Int64).alias("literal_click_impression_count"),
            (literal & (pl.col("is_click") == 1)).sum().cast(pl.Int64).alias("literal_click_positive_count"),
        )
        .sort(entity, "date")
        .with_columns(
            *(pl.col(name).cum_sum().over(entity).alias(f"{scope}__hist_{name}") for name in COUNT_SUFFIXES)
        )
        .select(entity, pl.col("date").alias(f"{scope}__history_date"), *(f"{scope}__hist_{name}" for name in COUNT_SUFFIXES))
    )
    return daily


def _join_snapshots(
    frame: pl.LazyFrame, snapshots: pl.DataFrame, entity: str, scope: str
) -> pl.LazyFrame:
    history_date = f"{scope}__history_date"
    # Perform the ordered as-of operation on narrow unique keys instead of the
    # full wide FS1 table, then use a many-to-one equality join to attach it.
    target_keys = frame.select(entity, "date").unique().collect()
    lookup = (
        target_keys.sort(entity, "date")
        .join_asof(
            snapshots.sort(entity, history_date),
            left_on="date",
            right_on=history_date,
            by=entity,
            strategy="backward",
            allow_exact_matches=False,
            check_sortedness=False,
        )
    )
    return frame.join(
        lookup.lazy(),
        on=[entity, "date"],
        how="left",
        validate="m:1",
        maintain_order="left",
    )


def build_fs2_lazy(
    fs1: pl.LazyFrame,
    user_snapshots: pl.DataFrame,
    video_snapshots: pl.DataFrame,
    priors: dict[str, float | int],
    alpha: float,
) -> pl.LazyFrame:
    """Append strictly-prior-day aggregate counts and smoothed rates."""
    if alpha <= 0:
        raise ValueError("FS2 smoothing alpha must be positive")
    fs1_columns = fs1.collect_schema().names()
    leaked = set(LABEL_COLUMNS).intersection(fs1_columns)
    if leaked:
        raise ValueError(f"Outcome columns entered FS2 base: {sorted(leaked)}")
    output = fs1
    output = _join_snapshots(output, user_snapshots, "user_id", "user")
    output = _join_snapshots(output, video_snapshots, "video_id", "video")
    count_columns = [f"{scope}__hist_{name}" for scope in SCOPES for name in COUNT_SUFFIXES]
    output = output.with_columns(*(pl.col(name).fill_null(0).cast(pl.Int64) for name in count_columns))
    rate_expressions: list[pl.Expr] = []
    for scope in SCOPES:
        mappings = {
            "long_view": ("long_view_positive_count", "impression_count"),
            "like": ("like_positive_count", "impression_count"),
            "valid_play": ("valid_play_positive_count", "valid_play_impression_count"),
            "literal_click": ("literal_click_positive_count", "literal_click_impression_count"),
        }
        for task, (positive, denominator) in mappings.items():
            rate_expressions.append(
                (
                    (pl.col(f"{scope}__hist_{positive}") + alpha * float(priors[task]))
                    / (pl.col(f"{scope}__hist_{denominator}") + alpha)
                ).cast(pl.Float64).alias(f"{scope}__hist_{task}_rate")
            )
    aggregate_columns = [
        *(f"{scope}__hist_{name}" for scope in SCOPES for name in COUNT_SUFFIXES),
        *(f"{scope}__hist_{name}" for scope in SCOPES for name in RATE_SUFFIXES),
    ]
    return (
        output.with_columns(*rate_expressions)
        .select(*fs1_columns, *aggregate_columns)
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _quality(
    fs1_path: Path, fs2_path: Path, source_label_path: Path, label_path: Path
) -> dict[str, Any]:
    fs1 = pl.scan_parquet(fs1_path)
    fs2 = pl.scan_parquet(fs2_path)
    fs1_schema = fs1.collect_schema()
    fs2_schema = fs2.collect_schema()
    if fs2_schema.names()[: len(fs1_schema)] != fs1_schema.names():
        raise ValueError("FS1 column order/schema was not preserved in FS2")
    for name, dtype in fs1_schema.items():
        if fs2_schema[name] != dtype:
            raise ValueError(f"FS1 dtype changed in FS2: {name}")
    aggregate_columns = [name for name in fs2_schema.names() if "__hist_" in name]
    if set(LABEL_COLUMNS).intersection(fs2_schema.names()):
        raise ValueError("Labels entered FS2 feature table")
    count_violations = []
    rate_violations = []
    for scope in SCOPES:
        count_violations.extend(
            [
                pl.col(f"{scope}__hist_long_view_positive_count") > pl.col(f"{scope}__hist_impression_count"),
                pl.col(f"{scope}__hist_like_positive_count") > pl.col(f"{scope}__hist_impression_count"),
                pl.col(f"{scope}__hist_valid_play_positive_count") > pl.col(f"{scope}__hist_valid_play_impression_count"),
                pl.col(f"{scope}__hist_literal_click_positive_count") > pl.col(f"{scope}__hist_literal_click_impression_count"),
            ]
        )
        rate_violations.extend(
            [
                ~pl.col(f"{scope}__hist_{name}").is_between(0.0, 1.0)
                for name in RATE_SUFFIXES
            ]
        )
    counts = fs2.select(
        pl.len().alias("rows"),
        pl.col("impression_id").n_unique().alias("unique_impressions"),
        (pl.col("user__hist_impression_count") == 0).sum().alias("zero_user_history_rows"),
        (pl.col("video__hist_impression_count") == 0).sum().alias("zero_video_history_rows"),
        pl.any_horizontal(*(pl.col(name).is_null() for name in aggregate_columns)).sum().alias("rows_with_null_aggregates"),
        pl.any_horizontal(*count_violations).sum().alias("rows_with_invalid_counts"),
        pl.any_horizontal(*rate_violations).sum().alias("rows_with_invalid_rates"),
    ).collect().row(0, named=True)
    fs1_rows = fs1.select(pl.len()).collect().item()
    if counts["rows"] != fs1_rows or counts["unique_impressions"] != fs1_rows:
        raise ValueError("FS2 row count or impression identity differs from FS1")
    missing_ids = fs1.select("impression_id").join(
        fs2.select("impression_id"), on="impression_id", how="anti"
    ).select(pl.len()).collect().item()
    if missing_ids:
        raise ValueError("FS2 is missing FS1 impression IDs")
    fs1_hashes = fs1.select(pl.struct(fs1_schema.names()).hash(seed=0).alias("hash")).collect()
    fs2_hashes = fs2.select(pl.struct(fs1_schema.names()).hash(seed=0).alias("hash")).collect()
    if not fs1_hashes.equals(fs2_hashes):
        raise ValueError("FS1 row values or order changed in FS2")
    source_label_hash = _sha256(source_label_path)
    label_hash = _sha256(label_path)
    if source_label_hash != label_hash:
        raise ValueError("FS2 labels are not byte-identical to FS1 labels")
    if counts["rows_with_invalid_counts"] or counts["rows_with_invalid_rates"]:
        raise ValueError("FS2 contains invalid count/rate values")
    return {
        **counts,
        "zero_user_history_rate": counts["zero_user_history_rows"] / counts["rows"],
        "zero_video_history_rate": counts["zero_video_history_rows"] / counts["rows"],
        "fs1_columns_preserved": True,
        "fs1_row_values_and_order_preserved": True,
        "missing_fs1_impression_ids": missing_ids,
        "aggregate_columns": aggregate_columns,
        "aggregate_schema": {name: str(fs2_schema[name]) for name in aggregate_columns},
        "labels_byte_identical_to_fs1": True,
        "label_sha256": label_hash,
    }


def materialize_training_fs2(
    data_dir: Path | None = None,
    fs1_dir: Path | None = None,
    output_dir: Path | None = None,
) -> dict[str, Any]:
    """Build training-only FS2 tables from normal-policy history."""
    started = time.perf_counter()
    paths = get_artifact_paths()
    data_dir = data_dir or paths.raw / "KuaiRand-Pure" / "data"
    fs1_dir = fs1_dir or paths.processed / "fs1" / "v1"
    output_dir = output_dir or paths.processed / "fs2" / "v1"
    config = load_fs2_config()
    settings = config["fs2"]
    prior_window = config["prior_window"]
    history = load_normal_history(data_dir, settings["history_start"], settings["history_end"])
    priors = compute_priors(history, prior_window["start"], prior_window["end"])
    verify_frozen_priors(priors, config)
    user_snapshots = daily_history_snapshots(history, "user_id", "user")
    video_snapshots = daily_history_snapshots(history, "video_id", "video")
    split_reports: dict[str, Any] = {}
    output_schemas: dict[str, dict[str, str]] = {}
    for split_name in ("normal_train", "random_train"):
        source_dir = fs1_dir / split_name
        destination = output_dir / split_name
        destination.mkdir(parents=True, exist_ok=True)
        fs1_path = source_dir / "features.parquet"
        fs1_label_path = source_dir / "labels.parquet"
        fs2_path = destination / "features.parquet"
        fs2_label_path = destination / "labels.parquet"
        build_fs2_lazy(
            pl.scan_parquet(fs1_path), user_snapshots, video_snapshots,
            priors, float(settings["alpha"]),
        ).sort("impression_id").sink_parquet(fs2_path)
        shutil.copy2(fs1_label_path, fs2_label_path)
        split_reports[split_name] = _quality(
            fs1_path, fs2_path, fs1_label_path, fs2_label_path
        )
        write_artifact_manifest(
            destination,
            stage="fs2",
            split=split_name,
            features_path=fs2_path,
            labels_path=fs2_label_path,
        )
        output_schemas[split_name] = {
            name: str(dtype) for name, dtype in pl.scan_parquet(fs2_path).collect_schema().items()
        }
    if output_schemas["normal_train"] != output_schemas["random_train"]:
        raise ValueError("FS2 schemas differ across training sources")
    report = {
        "milestone": "M2.2",
        "history_source": "normal_only",
        "history_window": [settings["history_start"], settings["history_end"]],
        "prior_window": [prior_window["start"], prior_window["end"]],
        "alpha": settings["alpha"],
        "frozen_priors": priors,
        "strictly_prior_calendar_days": True,
        "random_outcomes_used_for_history": False,
        "held_out_outcomes_accessed": False,
        "split_reports": split_reports,
        "same_schema_across_training_sources": True,
        "runtime_seconds": time.perf_counter() - started,
        "peak_rss_mib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
        "output_dir": str(output_dir),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "data_quality.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n"
    )
    return report
