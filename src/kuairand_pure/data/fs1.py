"""Build leakage-safe FS1 static feature and aligned label tables."""

from __future__ import annotations

import json
import resource
import tempfile
import time
import tomllib
from pathlib import Path
from typing import Any

import polars as pl

from kuairand_pure.data.alignment import write_artifact_manifest
from kuairand_pure.data.splits import (
    SplitSpec,
    assert_exact_date_coverage,
    assert_nonoverlapping_layout,
    frozen_splits,
    select_dates,
)
from kuairand_pure.paths import get_artifact_paths
from kuairand_pure.protocol import authorize_outcome_access, load_protocol

METADATA_COLUMNS = ("impression_id", "user_id", "video_id", "date", "time_ms")
LABEL_COLUMNS = ("long_view", "is_click", "is_like")
RAW_FEATURE_COLUMNS = ("user_id", "video_id", "date", "time_ms", "tab", "is_rand")
NON_MODEL_COLUMNS = frozenset(METADATA_COLUMNS)


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _load_toml(name: str) -> dict[str, Any]:
    with (_repo_root() / "configs" / name).open("rb") as stream:
        return tomllib.load(stream)


def _source_scan(path: Path, code: int, columns: tuple[str, ...]) -> pl.LazyFrame:
    multiplier = int(_load_toml("dataset.toml")["identity"]["source_id_multiplier"])
    return (
        pl.scan_csv(path, row_index_name="_source_row")
        .select("_source_row", *columns)
        .with_columns(
            (pl.lit(code, dtype=pl.UInt64) * multiplier + pl.col("_source_row").cast(pl.UInt64))
            .alias("impression_id")
        )
        .drop("_source_row")
    )


def _paths(data_dir: Path) -> tuple[dict[str, Any], dict[str, Path]]:
    config = _load_toml("dataset.toml")
    files = {name: data_dir / filename for name, filename in config["files"].items()}
    if files["videos_statistics_excluded"] in {
        files["users"], files["videos_basic"]
    }:
        raise ValueError("Excluded aggregate video statistics entered FS1 sources")
    return config, files


def _impressions(data_dir: Path, spec: SplitSpec, columns: tuple[str, ...]) -> pl.LazyFrame:
    config, files = _paths(data_dir)
    codes = config["source_codes"]
    if spec.source == "normal":
        scan = pl.concat(
            [
                _source_scan(files["normal_early"], codes["normal_early"], columns),
                _source_scan(files["normal_late"], codes["normal_late"], columns),
            ]
        )
    else:
        scan = _source_scan(files["random"], codes["random"], columns)
    return select_dates(scan, spec)


def _bounded_label_scan(path: Path, code: int, spec: SplitSpec) -> pl.LazyFrame:
    """Parse outcomes only after rows have been selected from their date field.

    Release rows are not date-contiguous. For non-authorized rows this routine
    parses only the third CSV field (``date``); engagement fields are never
    tokenized, typed, selected, or materialized.
    """
    selected_rows = 0
    with tempfile.NamedTemporaryFile(suffix=".csv") as bounded_file:
        with path.open("rb") as source:
            header = source.readline().rstrip(b"\r\n")
            bounded_file.write(header + b",_source_row\n")
            source_row = 0
            while True:
                user_field = _read_field(source)
                if user_field is None:
                    break
                video_field = _read_field(source)
                date_field = _read_field(source)
                if video_field is None or date_field is None:
                    raise ValueError(f"Malformed CSV row {source_row} in {path.name}")
                date = int(date_field)
                if spec.start <= date <= spec.end:
                    authorized_remainder = source.readline().rstrip(b"\r\n")
                    bounded_file.write(
                        user_field + b"," + video_field + b"," + date_field + b"," +
                        authorized_remainder + b"," + str(source_row).encode("ascii") + b"\n"
                    )
                    selected_rows += 1
                else:
                    # Advance to the next record without returning, storing,
                    # tokenizing, or interpreting the outcome-field bytes.
                    source.readline()
                source_row += 1
        bounded_file.flush()
        if selected_rows == 0:
            return pl.LazyFrame(schema={
                "impression_id": pl.UInt64,
                "user_id": pl.Int64,
                "video_id": pl.Int64,
                "date": pl.Int64,
                "time_ms": pl.Int64,
                "tab": pl.Int64,
                **{label: pl.Int64 for label in LABEL_COLUMNS},
            })
        bounded_frame = pl.read_csv(
            bounded_file.name,
            schema_overrides={
                "user_id": pl.Int64,
                "video_id": pl.Int64,
                "date": pl.Int64,
                "time_ms": pl.Int64,
                "tab": pl.Int64,
                **{label: pl.Int64 for label in LABEL_COLUMNS},
                "_source_row": pl.UInt64,
            },
        ).select(
            "_source_row", "user_id", "video_id", "date", "time_ms", "tab", *LABEL_COLUMNS
        )
    bounded = bounded_frame.lazy()
    multiplier = int(_load_toml("dataset.toml")["identity"]["source_id_multiplier"])
    return bounded.with_columns(
        (pl.lit(code, dtype=pl.UInt64) * multiplier + pl.col("_source_row").cast(pl.UInt64))
        .alias("impression_id")
    ).select("impression_id", "user_id", "video_id", "date", "time_ms", "tab", *LABEL_COLUMNS)


def _read_field(source: Any) -> bytes | None:
    """Read one unquoted CSV field; only used for numeric fields before outcomes."""
    value = bytearray()
    while True:
        byte = source.read(1)
        if byte == b"":
            return None if not value else bytes(value)
        if byte == b",":
            return bytes(value)
        if byte in (b"\n", b"\r"):
            raise ValueError("Unexpected row ending before date field")
        value.extend(byte)


def _labels(
    data_dir: Path,
    spec: SplitSpec,
    *,
    workflow: str = "training",
    purpose: str = "training",
) -> pl.LazyFrame:
    authorize_outcome_access(
        load_protocol(), source=spec.source, start=spec.start, end=spec.end,
        workflow=workflow, purpose=purpose,
    )
    config, files = _paths(data_dir)
    codes = config["source_codes"]
    if spec.source == "normal":
        frames = [
            _bounded_label_scan(files["normal_early"], codes["normal_early"], spec),
            _bounded_label_scan(files["normal_late"], codes["normal_late"], spec),
        ]
        labels = pl.concat(frames)
    else:
        labels = _bounded_label_scan(files["random"], codes["random"], spec)
    return labels.with_columns(
        pl.when(pl.col("tab") == 1)
        .then(pl.lit("valid_play"))
        .when(pl.col("tab") == 0)
        .then(pl.lit("literal_click"))
        .otherwise(pl.lit("unsupported_interface"))
        .alias("click_semantics")
    ).select(*METADATA_COLUMNS, "click_semantics", *LABEL_COLUMNS)


def _static_tables(data_dir: Path) -> tuple[pl.LazyFrame, pl.LazyFrame, list[str], list[str]]:
    fs1 = _load_toml("fs1.toml")["fs1"]
    _, files = _paths(data_dir)
    users = pl.read_csv(files["users"])
    videos = pl.read_csv(files["videos_basic"])
    for name, frame, key in (("user", users, "user_id"), ("video", videos, "video_id")):
        if frame[key].null_count() or frame[key].n_unique() != frame.height:
            raise ValueError(f"{name} static join key is null or non-unique")
    user_features = [column for column in users.columns if column != "user_id"]
    video_features = [column for column in videos.columns if column != "video_id"]
    users = users.rename({column: fs1["feature_prefix_user"] + column for column in user_features})
    videos = videos.rename({column: fs1["feature_prefix_video"] + column for column in video_features})
    return (
        users.with_columns(pl.lit(True).alias("_user_found")).lazy(),
        videos.with_columns(pl.lit(True).alias("_video_found")).lazy(),
        [fs1["feature_prefix_user"] + column for column in user_features],
        [fs1["feature_prefix_video"] + column for column in video_features],
    )


def build_fs1_lazy(
    impressions: pl.LazyFrame,
    users: pl.LazyFrame,
    videos: pl.LazyFrame,
    user_features: list[str],
    video_features: list[str],
) -> pl.LazyFrame:
    """Left-join static features without changing impression cardinality."""
    forbidden = set(_load_toml("fs1.toml")["forbidden"]["feature_columns"])
    output = (
        impressions.join(
            users, on="user_id", how="left", validate="m:1", maintain_order="left"
        )
        .join(
            videos, on="video_id", how="left", validate="m:1", maintain_order="left"
        )
        .with_columns(
            pl.col("_user_found").is_null().alias("user__static_missing"),
            pl.col("_video_found").is_null().alias("video__static_missing"),
        )
        .select(
            *METADATA_COLUMNS,
            pl.col("tab").alias("context__tab"),
            pl.col("is_rand").alias("context__is_rand"),
            "user__static_missing", "video__static_missing",
            *user_features, *video_features,
        )
    )
    leaked = forbidden.intersection(output.collect_schema().names())
    if leaked:
        raise ValueError(f"Outcome or engagement columns entered FS1: {sorted(leaked)}")
    return output


def model_feature_columns(schema_names: list[str], workflow: str) -> list[str]:
    """Return FS1 model columns while enforcing the exposure-indicator rule."""
    allowed = {"baseline", "rq3_random_only", "rq3_joint"}
    if workflow not in allowed:
        raise ValueError(f"Unknown FS1 workflow: {workflow}")
    columns = [name for name in schema_names if name not in NON_MODEL_COLUMNS]
    if workflow != "rq3_joint":
        columns.remove("context__is_rand")
    return columns


def _quality(features_path: Path, labels_path: Path, spec: SplitSpec) -> dict[str, Any]:
    features = pl.scan_parquet(features_path)
    labels = pl.scan_parquet(labels_path)
    feature_schema = features.collect_schema()
    forbidden = set(_load_toml("fs1.toml")["forbidden"]["feature_columns"])
    feature_stats = features.select(
        pl.len().alias("rows"),
        pl.col("impression_id").n_unique().alias("unique_impressions"),
        pl.col("user_id").is_null().sum().alias("null_user_keys"),
        pl.col("video_id").is_null().sum().alias("null_video_keys"),
        pl.col("user__static_missing").sum().alias("unknown_user_rows"),
        pl.col("video__static_missing").sum().alias("unknown_video_rows"),
        pl.col("date").min().alias("min_date"),
        pl.col("date").max().alias("max_date"),
    ).collect().row(0, named=True)
    label_stats = labels.select(
        pl.len().alias("rows"),
        pl.col("impression_id").n_unique().alias("unique_impressions"),
    ).collect().row(0, named=True)
    null_counts = features.null_count().collect().row(0, named=True)
    assert_exact_date_coverage(features.select("date").collect()["date"], spec)
    if feature_stats["rows"] != label_stats["rows"]:
        raise ValueError("Feature/label row counts differ")
    if feature_stats["null_user_keys"] or feature_stats["null_video_keys"]:
        raise ValueError("Null impression join keys are not allowed")
    if feature_stats["unique_impressions"] != feature_stats["rows"]:
        raise ValueError("FS1 has duplicate impression identifiers")
    if label_stats["unique_impressions"] != label_stats["rows"]:
        raise ValueError("Labels have duplicate impression identifiers")
    null_feature_ids = features.select(pl.col("impression_id").null_count()).collect().item()
    null_label_ids = labels.select(pl.col("impression_id").null_count()).collect().item()
    if null_feature_ids or null_label_ids:
        raise ValueError("Missing impression identifiers are not allowed")
    feature_ids = features.select("impression_id").collect()
    label_ids = labels.select("impression_id").collect()
    if not feature_ids["impression_id"].is_sorted() or not label_ids["impression_id"].is_sorted():
        raise ValueError("FS1 artifacts are not in canonical impression_id order")
    if not feature_ids.equals(label_ids):
        raise ValueError("FS1 feature and label impression IDs differ by row")
    unmatched = features.select("impression_id").join(
        labels.select("impression_id"), on="impression_id", how="anti"
    ).select(pl.len()).collect().item()
    if unmatched:
        raise ValueError(f"{unmatched} FS1 rows lack aligned labels")
    unmatched_labels = labels.select("impression_id").join(
        features.select("impression_id"), on="impression_id", how="anti"
    ).select(pl.len()).collect().item()
    if unmatched_labels:
        raise ValueError(f"{unmatched_labels} label rows lack aligned FS1 rows")
    expected_is_rand = 0 if spec.source == "normal" else 1
    observed_is_rand = features.select(pl.col("context__is_rand").unique()).collect().to_series().to_list()
    if observed_is_rand != [expected_is_rand]:
        raise ValueError(f"Unexpected is_rand values for {spec.name}: {observed_is_rand}")
    leaked = forbidden.intersection(feature_schema.names())
    if leaked:
        raise ValueError(f"Forbidden FS1 columns: {sorted(leaked)}")
    return {
        "split": spec.name,
        "source": spec.source,
        **feature_stats,
        "feature_label_rows_match": True,
        "feature_label_positional_alignment": True,
        "canonical_order": "impression_id_ascending",
        "unmatched_feature_ids": unmatched,
        "unmatched_label_ids": unmatched_labels,
        "null_counts": null_counts,
        "feature_columns": feature_schema.names(),
        "model_feature_columns": {
            workflow: model_feature_columns(feature_schema.names(), workflow)
            for workflow in ("baseline", "rq3_random_only", "rq3_joint")
        },
        "feature_schema": {name: str(dtype) for name, dtype in feature_schema.items()},
        "label_schema": {name: str(dtype) for name, dtype in labels.collect_schema().items()},
        "forbidden_feature_columns_present": sorted(leaked),
        "video_statistics_source_used": False,
    }


def _metadata_only_quality(data_dir: Path, spec: SplitSpec) -> dict[str, Any]:
    """Validate held-out structure while projecting no outcome columns."""
    frame = _impressions(data_dir, spec, RAW_FEATURE_COLUMNS)
    schema = frame.collect_schema()
    if set(LABEL_COLUMNS).intersection(schema.names()):
        raise ValueError("Outcome columns entered metadata-only held-out check")
    summary = frame.select(
        pl.len().alias("rows"),
        pl.col("impression_id").n_unique().alias("unique_impressions"),
        pl.col("date").min().alias("min_date"),
        pl.col("date").max().alias("max_date"),
        pl.col("user_id").is_null().sum().alias("null_user_keys"),
        pl.col("video_id").is_null().sum().alias("null_video_keys"),
    ).collect().row(0, named=True)
    dates = frame.select("date").collect()["date"]
    assert_exact_date_coverage(dates, spec)
    if summary["rows"] != summary["unique_impressions"]:
        raise ValueError(f"Duplicate impression IDs in metadata-only {spec.name}")
    return {
        "split": spec.name,
        "source": spec.source,
        **summary,
        "projected_columns": schema.names(),
        "outcome_columns_projected": False,
        "materialized": False,
    }


def materialize_training_fs1(
    data_dir: Path | None = None, output_dir: Path | None = None
) -> dict[str, Any]:
    """Materialize only normal and RQ3-random authorized training tables."""
    started = time.perf_counter()
    paths = get_artifact_paths()
    data_dir = data_dir or paths.raw / "KuaiRand-Pure" / "data"
    output_dir = output_dir or paths.processed / "fs1" / "v1"
    splits = frozen_splits()
    assert_nonoverlapping_layout(splits)
    users, videos, user_features, video_features = _static_tables(data_dir)
    reports: dict[str, Any] = {}
    schemas: dict[str, dict[str, str]] = {}
    for name in ("normal_train", "random_train"):
        spec = splits[name]
        split_dir = output_dir / name
        split_dir.mkdir(parents=True, exist_ok=True)
        feature_path = split_dir / "features.parquet"
        label_path = split_dir / "labels.parquet"
        impressions = _impressions(data_dir, spec, RAW_FEATURE_COLUMNS)
        build_fs1_lazy(
            impressions, users, videos, user_features, video_features
        ).sort("impression_id").sink_parquet(feature_path)
        _labels(data_dir, spec).sort("impression_id").sink_parquet(label_path)
        reports[name] = _quality(feature_path, label_path, spec)
        write_artifact_manifest(
            split_dir,
            stage="fs1",
            split=name,
            features_path=feature_path,
            labels_path=label_path,
        )
        schemas[name] = reports[name]["feature_schema"]
    if schemas["normal_train"] != schemas["random_train"]:
        raise ValueError("FS1 feature schema differs across training sources")
    metadata_checks = {
        name: _metadata_only_quality(data_dir, splits[name])
        for name in (
            "normal_validation",
            "random_validation_embargo",
            "normal_test",
            "random_test",
        )
    }
    elapsed = time.perf_counter() - started
    report = {
        "milestone": "M2.1",
        "output_dir": str(output_dir),
        "splits": reports,
        "same_feature_schema_across_sources": True,
        "held_out_metadata_only_checks": metadata_checks,
        "runtime_seconds": elapsed,
        "peak_rss_mib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
        "outcomes_accessed": {
            "normal_train": [20220412, 20220429],
            "random_train": [20220422, 20220429],
            "normal_validation": False,
            "random_validation": False,
            "final_test": False,
        },
    }
    report_path = output_dir / "data_quality.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return report
