"""Leakage-safe FS3 affinity, recency, and neural history artifacts."""

from __future__ import annotations

import json
import resource
import shutil
import time
import tomllib
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
from kuairand_pure.data.alignment import write_artifact_manifest
from kuairand_pure.data.fs1 import LABEL_COLUMNS, RAW_FEATURE_COLUMNS, _impressions, _paths
from kuairand_pure.data.splits import SplitSpec
from kuairand_pure.features.fs2 import (
    COUNT_SUFFIXES,
    RATE_SUFFIXES,
    _sha256,
    compute_priors,
    load_fs2_config,
    load_normal_history,
    verify_frozen_priors,
)
from kuairand_pure.paths import get_artifact_paths

AFFINITY_PREFIX = "user_category__hist_"
FS3_NEW_COLUMNS = (
    "category__key",
    "category__atomic_count",
    *(AFFINITY_PREFIX + name for name in COUNT_SUFFIXES),
    *(AFFINITY_PREFIX + name for name in RATE_SUFFIXES),
    "user__recency_since_last_normal_ms",
    "user__recency_missing",
    "user_category__recency_since_last_normal_ms",
    "user_category__recency_missing",
)


def load_fs3_config(path: Path | None = None) -> dict[str, Any]:
    config_path = path or Path(__file__).resolve().parents[3] / "configs" / "fs3.toml"
    with config_path.open("rb") as stream:
        return tomllib.load(stream)


def canonicalize_tag(value: str | None, missing_token: str = "__MISSING__") -> str:
    """Canonicalize the documented comma-delimited list of opaque tag IDs."""
    if value is None or not value.strip():
        return missing_token
    tokens = {token.strip() for token in value.split(",") if token.strip()}
    if not tokens:
        return missing_token
    try:
        return "|".join(sorted(tokens, key=lambda token: int(token)))
    except ValueError as exc:
        raise ValueError(f"Non-numeric KuaiRand tag token in {value!r}") from exc


def atomic_tag_count(category_key: str, missing_token: str = "__MISSING__") -> int:
    return 0 if category_key == missing_token else len(category_key.split("|"))


def video_categories(data_dir: Path, missing_token: str) -> pl.DataFrame:
    _, files = _paths(data_dir)
    videos = pl.read_csv(files["videos_basic"], columns=["video_id", "tag"])
    return videos.with_columns(
        pl.col("tag")
        .map_elements(
            lambda value: canonicalize_tag(value, missing_token),
            return_dtype=pl.String,
            skip_nulls=False,
        )
        .alias("category__key")
    ).with_columns(
        pl.col("category__key")
        .map_elements(lambda value: atomic_tag_count(value, missing_token), return_dtype=pl.Int16)
        .alias("category__atomic_count")
    ).select("video_id", "category__key", "category__atomic_count")


def load_normal_structural_history(data_dir: Path, start: int, end: int) -> pl.DataFrame:
    """Load only non-outcome columns from normal-policy history."""
    spec = SplitSpec("normal_structural_history", "normal", start, end, False)
    return _impressions(data_dir, spec, RAW_FEATURE_COLUMNS).select(
        "impression_id", "user_id", "video_id", "date", "time_ms", "tab"
    ).collect()


def _daily_affinity_snapshots(history: pl.DataFrame) -> pl.DataFrame:
    valid = pl.col("tab") == 1
    literal = pl.col("tab") == 0
    group = ["user_id", "category__key"]
    return (
        history.group_by(*group, "date")
        .agg(
            pl.len().cast(pl.Int64).alias("impression_count"),
            pl.col("long_view").sum().cast(pl.Int64).alias("long_view_positive_count"),
            pl.col("is_like").sum().cast(pl.Int64).alias("like_positive_count"),
            valid.sum().cast(pl.Int64).alias("valid_play_impression_count"),
            (valid & (pl.col("is_click") == 1)).sum().cast(pl.Int64).alias("valid_play_positive_count"),
            literal.sum().cast(pl.Int64).alias("literal_click_impression_count"),
            (literal & (pl.col("is_click") == 1)).sum().cast(pl.Int64).alias("literal_click_positive_count"),
        )
        .sort(*group, "date")
        .with_columns(
            *(pl.col(name).cum_sum().over(group).alias(AFFINITY_PREFIX + name) for name in COUNT_SUFFIXES)
        )
        .select(*group, pl.col("date").alias("user_category__history_date"), *(AFFINITY_PREFIX + name for name in COUNT_SUFFIXES))
    )


def _daily_recency_snapshots(
    history: pl.DataFrame, by: list[str], prefix: str
) -> pl.DataFrame:
    return (
        history.group_by(*by, "date")
        .agg(pl.col("time_ms").max().alias("daily_last_time_ms"))
        .sort(*by, "date")
        .with_columns(
            pl.col("daily_last_time_ms").cum_max().over(by).alias(prefix + "last_time_ms")
        )
        .select(*by, pl.col("date").alias(prefix + "history_date"), prefix + "last_time_ms")
    )


def _asof_lookup(
    frame: pl.LazyFrame,
    snapshots: pl.DataFrame,
    by: list[str],
    history_date: str,
) -> pl.LazyFrame:
    keys = frame.select(*by, "date").unique().collect()
    lookup = keys.sort(*by, "date").join_asof(
        snapshots.sort(*by, history_date),
        left_on="date",
        right_on=history_date,
        by=by,
        strategy="backward",
        allow_exact_matches=False,
        check_sortedness=False,
    )
    return frame.join(
        lookup.lazy(), on=[*by, "date"], how="left", validate="m:1", maintain_order="left"
    )


def build_fs3_lazy(
    fs2: pl.LazyFrame,
    categories: pl.DataFrame,
    affinity_snapshots: pl.DataFrame,
    user_recency_snapshots: pl.DataFrame,
    category_recency_snapshots: pl.DataFrame,
    priors: dict[str, float | int],
    alpha: float,
    missing_token: str = "__MISSING__",
) -> pl.LazyFrame:
    """Append prior-day category affinity and recency while preserving FS2."""
    fs2_columns = fs2.collect_schema().names()
    if set(LABEL_COLUMNS).intersection(fs2_columns):
        raise ValueError("Outcome columns entered FS3 base")
    output = fs2.join(categories.lazy(), on="video_id", how="left", validate="m:1", maintain_order="left")
    output = output.with_columns(
        pl.col("category__key").fill_null(missing_token),
        pl.col("category__atomic_count").fill_null(0).cast(pl.Int16),
    )
    output = _asof_lookup(
        output, affinity_snapshots, ["user_id", "category__key"], "user_category__history_date"
    )
    output = _asof_lookup(
        output, user_recency_snapshots, ["user_id"], "user__recency_history_date"
    )
    output = _asof_lookup(
        output,
        category_recency_snapshots,
        ["user_id", "category__key"],
        "user_category__recency_history_date",
    )
    count_columns = [AFFINITY_PREFIX + name for name in COUNT_SUFFIXES]
    output = output.with_columns(*(pl.col(name).fill_null(0).cast(pl.Int64) for name in count_columns))
    mappings = {
        "long_view": ("long_view_positive_count", "impression_count"),
        "like": ("like_positive_count", "impression_count"),
        "valid_play": ("valid_play_positive_count", "valid_play_impression_count"),
        "literal_click": ("literal_click_positive_count", "literal_click_impression_count"),
    }
    output = output.with_columns(
        *(
            (
                (pl.col(AFFINITY_PREFIX + positive) + alpha * float(priors[task]))
                / (pl.col(AFFINITY_PREFIX + denominator) + alpha)
            ).cast(pl.Float64).alias(AFFINITY_PREFIX + task + "_rate")
            for task, (positive, denominator) in mappings.items()
        )
    )
    recencies = (
        ("user__recency_", "user__recency_last_time_ms"),
        ("user_category__recency_", "user_category__recency_last_time_ms"),
    )
    output = output.with_columns(
        *(
            ((pl.col(last).is_null()) | (pl.col(last) >= pl.col("time_ms"))).alias(prefix + "missing")
            for prefix, last in recencies
        )
    ).with_columns(
        *(
            pl.when(~pl.col(prefix + "missing"))
            .then(pl.col("time_ms") - pl.col(last))
            .otherwise(-1)
            .cast(pl.Int64)
            .alias(prefix + "since_last_normal_ms")
            for prefix, last in recencies
        )
    )
    return output.select(*fs2_columns, *FS3_NEW_COLUMNS)


def build_sequence_frame(
    targets: pl.DataFrame,
    normal_history: pl.DataFrame,
    known_video_ids: set[int],
    *,
    length: int,
    padding_id: int,
    unknown_id: int,
    known_offset: int,
    padding_time_ms: int,
) -> pl.DataFrame:
    """Create deterministic, fixed-length, strictly-earlier normal histories."""
    if length <= 0:
        raise ValueError("Sequence length must be positive")
    history = normal_history.with_columns(
        pl.when(pl.col("video_id").is_in(known_video_ids))
        .then(pl.col("video_id") + known_offset)
        .otherwise(unknown_id)
        .cast(pl.Int32)
        .alias("encoded_video_id")
    ).sort("user_id", "time_ms", "encoded_video_id", "impression_id")
    history_users = history["user_id"].to_numpy()
    history_times = history["time_ms"].to_numpy()
    history_videos = history["encoded_video_id"].to_numpy()
    unique_users, starts = np.unique(history_users, return_index=True)
    ends = np.append(starts[1:], len(history_users))
    user_bounds = {
        int(user): (int(start), int(end))
        for user, start, end in zip(unique_users, starts, ends)
    }
    row_count = targets.height
    videos = np.full((row_count, length), padding_id, dtype=np.int32)
    times = np.full((row_count, length), padding_time_ms, dtype=np.int64)
    masks = np.zeros((row_count, length), dtype=np.uint8)
    lengths = np.zeros(row_count, dtype=np.int16)
    boundary_dropped = np.zeros(row_count, dtype=np.bool_)
    for row_index, row in enumerate(
        targets.select("user_id", "time_ms").iter_rows(named=True)
    ):
        bounds = user_bounds.get(int(row["user_id"]))
        if bounds is None:
            continue
        user_start, user_end = bounds
        user_times = history_times[user_start:user_end]
        relative_end = int(np.searchsorted(user_times, int(row["time_ms"]), side="left"))
        relative_start = max(0, relative_end - length)
        if (
            relative_start > 0
            and relative_start < relative_end
            and user_times[relative_start] == user_times[relative_start - 1]
        ):
            tied_time = user_times[relative_start]
            while relative_start < relative_end and user_times[relative_start] == tied_time:
                relative_start += 1
            boundary_dropped[row_index] = True
        selected_length = relative_end - relative_start
        if selected_length:
            destination = length - selected_length
            absolute_start = user_start + relative_start
            absolute_end = user_start + relative_end
            videos[row_index, destination:] = history_videos[absolute_start:absolute_end]
            times[row_index, destination:] = history_times[absolute_start:absolute_end]
            masks[row_index, destination:] = 1
            lengths[row_index] = selected_length
    return pl.DataFrame(
        {
            "impression_id": targets["impression_id"],
            "target_time_ms": targets["time_ms"],
            "history_video_ids": videos.tolist(),
            "history_time_ms": times.tolist(),
            "history_mask": masks.tolist(),
            "history_length": lengths,
            "history_tie_boundary_dropped": boundary_dropped,
        },
        schema_overrides={
            "impression_id": pl.UInt64,
            "target_time_ms": pl.Int64,
            "history_video_ids": pl.Array(pl.Int32, length),
            "history_time_ms": pl.Array(pl.Int64, length),
            "history_mask": pl.Array(pl.UInt8, length),
            "history_length": pl.Int16,
            "history_tie_boundary_dropped": pl.Boolean,
        },
    )


def _quality(
    fs2_path: Path,
    fs3_path: Path,
    sequence_path: Path,
    source_label_path: Path,
    label_path: Path,
    expected_length: int,
) -> dict[str, Any]:
    fs2 = pl.scan_parquet(fs2_path)
    fs3 = pl.scan_parquet(fs3_path)
    sequence = pl.scan_parquet(sequence_path)
    fs2_schema = fs2.collect_schema()
    fs3_schema = fs3.collect_schema()
    if fs3_schema.names()[: len(fs2_schema)] != fs2_schema.names():
        raise ValueError("FS2 columns are not an exact prefix of FS3")
    fs2_hash = fs2.select(pl.struct(fs2_schema.names()).hash(seed=0)).collect()
    fs3_hash = fs3.select(pl.struct(fs2_schema.names()).hash(seed=0)).collect()
    if not fs2_hash.equals(fs3_hash):
        raise ValueError("FS2 values/order changed in FS3")
    if set(LABEL_COLUMNS).intersection(fs3_schema.names()):
        raise ValueError("Labels entered FS3 features")
    rows = fs3.select(
        pl.len().alias("rows"),
        pl.col("impression_id").n_unique().alias("unique_impressions"),
        (pl.col(AFFINITY_PREFIX + "impression_count") == 0).sum().alias("zero_affinity_rows"),
        pl.col("user__recency_missing").sum().alias("missing_user_recency_rows"),
        pl.col("user_category__recency_missing").sum().alias("missing_category_recency_rows"),
    ).collect().row(0, named=True)
    sequence_stats = sequence.select(
        pl.len().alias("rows"),
        pl.col("impression_id").n_unique().alias("unique_impressions"),
        (pl.col("history_length") == 0).sum().alias("empty_history_rows"),
        pl.col("history_length").mean().alias("mean_history_length"),
        (pl.col("history_length") == expected_length).sum().alias("full_history_rows"),
        pl.col("history_tie_boundary_dropped").sum().alias("tie_boundary_drop_rows"),
    ).collect()
    # Verify every unpadded event is strictly earlier than its target.
    temporal_violations = sequence.select(
        pl.any_horizontal(*(
            (pl.col("history_mask").arr.get(index) == 1)
            & (pl.col("history_time_ms").arr.get(index) >= pl.col("target_time_ms"))
            for index in range(expected_length)
        )).sum().alias("temporal_violations")
    ).collect().item()
    sequence_row = sequence_stats.row(0, named=True)
    if rows["rows"] != sequence_row["rows"] or temporal_violations:
        raise ValueError("FS3/sequence alignment or temporal validation failed")
    missing_sequence_ids = fs3.select("impression_id").join(
        sequence.select("impression_id"), on="impression_id", how="anti"
    ).select(pl.len()).collect().item()
    if missing_sequence_ids:
        raise ValueError("Sequence artifact lacks FS3 impression IDs")
    source_hash = _sha256(source_label_path)
    label_hash = _sha256(label_path)
    if source_hash != label_hash:
        raise ValueError("FS3 labels differ from FS2 labels")
    return {
        **rows,
        "zero_affinity_rate": rows["zero_affinity_rows"] / rows["rows"],
        "missing_user_recency_rate": rows["missing_user_recency_rows"] / rows["rows"],
        "missing_category_recency_rate": rows["missing_category_recency_rows"] / rows["rows"],
        **sequence_row,
        "empty_history_rate": sequence_row["empty_history_rows"] / sequence_row["rows"],
        "full_history_rate": sequence_row["full_history_rows"] / sequence_row["rows"],
        "temporal_violations": temporal_violations,
        "missing_sequence_ids": missing_sequence_ids,
        "fs2_values_and_order_preserved": True,
        "labels_byte_identical_to_fs2": True,
        "label_sha256": label_hash,
        "new_feature_schema": {name: str(fs3_schema[name]) for name in FS3_NEW_COLUMNS},
        "sequence_schema": {
            name: str(dtype) for name, dtype in sequence.collect_schema().items()
        },
    }


def materialize_training_fs3(
    data_dir: Path | None = None,
    fs2_dir: Path | None = None,
    output_dir: Path | None = None,
) -> dict[str, Any]:
    """Materialize authorized training FS3 and aligned neural histories."""
    started = time.perf_counter()
    paths = get_artifact_paths()
    data_dir = data_dir or paths.raw / "KuaiRand-Pure" / "data"
    fs2_dir = fs2_dir or paths.processed / "fs2" / "v1"
    output_dir = output_dir or paths.processed / "fs3" / "v1"
    config = load_fs3_config()
    settings, category_config, sequence_config = (
        config["fs3"], config["category"], config["sequence"]
    )
    fs2_config = load_fs2_config()
    normal_outcomes = load_normal_history(
        data_dir, settings["history_start"], settings["history_end"]
    )
    priors = compute_priors(
        normal_outcomes,
        fs2_config["prior_window"]["start"],
        fs2_config["prior_window"]["end"],
    )
    verify_frozen_priors(priors, fs2_config)
    categories = video_categories(data_dir, category_config["missing_token"])
    normal_outcomes = normal_outcomes.join(
        categories.select("video_id", "category__key"), on="video_id", how="left", validate="m:1"
    ).with_columns(pl.col("category__key").fill_null(category_config["missing_token"]))
    normal_structural = load_normal_structural_history(
        data_dir, settings["history_start"], settings["history_end"]
    ).join(categories.select("video_id", "category__key"), on="video_id", how="left", validate="m:1").with_columns(
        pl.col("category__key").fill_null(category_config["missing_token"])
    )
    affinity = _daily_affinity_snapshots(normal_outcomes)
    user_recency = _daily_recency_snapshots(normal_structural, ["user_id"], "user__recency_")
    category_recency = _daily_recency_snapshots(
        normal_structural, ["user_id", "category__key"], "user_category__recency_"
    )
    known_video_ids = set(categories["video_id"].to_list())
    split_reports: dict[str, Any] = {}
    schemas: dict[str, dict[str, str]] = {}
    for split_name in ("normal_train", "random_train"):
        source = fs2_dir / split_name
        destination = output_dir / split_name
        destination.mkdir(parents=True, exist_ok=True)
        fs2_path = source / "features.parquet"
        source_label_path = source / "labels.parquet"
        fs3_path = destination / "features.parquet"
        label_path = destination / "labels.parquet"
        sequence_path = destination / "sequence.parquet"
        build_fs3_lazy(
            pl.scan_parquet(fs2_path), categories, affinity, user_recency,
            category_recency, priors, float(settings["alpha"]), category_config["missing_token"],
        ).sort("impression_id").sink_parquet(fs3_path)
        shutil.copy2(source_label_path, label_path)
        targets = pl.read_parquet(fs2_path, columns=["impression_id", "user_id", "time_ms"])
        sequence = build_sequence_frame(
            targets,
            normal_structural.select("impression_id", "user_id", "video_id", "time_ms"),
            known_video_ids,
            length=sequence_config["length"],
            padding_id=sequence_config["padding_video_id"],
            unknown_id=sequence_config["unknown_video_id"],
            known_offset=sequence_config["known_video_id_offset"],
            padding_time_ms=sequence_config["padding_time_ms"],
        )
        sequence.sort("impression_id").write_parquet(sequence_path)
        split_reports[split_name] = _quality(
            fs2_path, fs3_path, sequence_path, source_label_path,
            label_path, sequence_config["length"],
        )
        write_artifact_manifest(
            destination,
            stage="fs3",
            split=split_name,
            features_path=fs3_path,
            labels_path=label_path,
            sequence_path=sequence_path,
        )
        schemas[split_name] = {
            name: str(dtype) for name, dtype in pl.scan_parquet(fs3_path).collect_schema().items()
        }
    if schemas["normal_train"] != schemas["random_train"]:
        raise ValueError("FS3 schemas differ across training sources")
    category_counts = categories.select(
        pl.len().alias("videos"),
        pl.col("category__key").n_unique().alias("canonical_category_cardinality"),
        (pl.col("category__key") == category_config["missing_token"]).sum().alias("missing_category_videos"),
    ).row(0, named=True)
    atomic_cardinality = (
        categories.filter(pl.col("category__atomic_count") > 0)
        .select(pl.col("category__key").str.split("|").explode().n_unique())
        .item()
    )
    report = {
        "milestone": "M2.3",
        "output_dir": str(output_dir),
        "history_source": "normal_only",
        "held_out_outcomes_accessed": False,
        "random_outcomes_used": False,
        "category_representation": category_config,
        "category_cardinality": {**category_counts, "atomic_tag_cardinality": atomic_cardinality},
        "alpha": settings["alpha"],
        "frozen_priors": priors,
        "sequence": sequence_config,
        "split_reports": split_reports,
        "same_schema_across_training_sources": True,
        "runtime_seconds": time.perf_counter() - started,
        "peak_rss_mib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "data_quality.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n"
    )
    return report
