"""M2 feature-pipeline integration and manifest checks."""

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
    CANONICAL_ORDER,
    M2_BUILD_VERSION,
    validate_impression_ids,
)
from kuairand_pure.data.fs1 import LABEL_COLUMNS, model_feature_columns
from kuairand_pure.features.fs3 import load_normal_structural_history, video_categories
from kuairand_pure.paths import get_artifact_paths


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _model_access(name: str, sequence: bool = False) -> tuple[list[str], str]:
    if sequence:
        if name in {"history_video_ids", "history_mask", "history_length"}:
            return ["shared_bottom_mlp", "mmoe"], "neural_history_input"
        return [], "alignment_or_audit_only"
    if name in {"impression_id", "date", "time_ms"}:
        return [], "alignment_only"
    if name in {"user_id", "video_id"}:
        return ["shared_bottom_mlp", "mmoe"], "neural_id_embedding_input"
    if name == "context__is_rand":
        return ["lightgbm", "shared_bottom_mlp", "mmoe"], "rq3_joint_only"
    return ["lightgbm", "shared_bottom_mlp", "mmoe"], "dense_model_feature"


def _feature_metadata(name: str, stage: str) -> tuple[str, str, str]:
    """Return source, missing policy, and point-in-time provenance."""
    if name in {"impression_id", "user_id", "video_id", "date", "time_ms"}:
        return "impression_log_structural_fields", "not_nullable", "event_time"
    if name.startswith("context__"):
        return "impression_log_context", "not_nullable", "target_impression_context"
    if name.endswith("__static_missing"):
        return "static_dimension_join", "boolean_unknown_entity_indicator", "join_time"
    if name.startswith("user__hist_") or name.startswith("video__hist_"):
        policy = "zero_for_unseen_entity" if name.endswith("count") else "alpha20_frozen_prior_for_unseen_entity"
        return "strictly_prior_day_normal_policy_outcomes", policy, "point_in_time_verified"
    if name.startswith("user_category__hist_"):
        policy = "zero_for_unseen_pair" if name.endswith("count") else "alpha20_frozen_prior_for_unseen_pair"
        return "strictly_prior_day_normal_policy_outcomes", policy, "point_in_time_verified"
    if "__recency_" in name:
        policy = "boolean_indicator" if name.endswith("missing") else "minus_one_plus_missing_indicator"
        return "strictly_prior_day_normal_policy_timestamps", policy, "point_in_time_verified"
    if name.startswith("category__"):
        policy = "explicit___MISSING___category" if name == "category__key" else "zero_for_missing_category"
        return "video_features_basic_pure.csv:tag", policy, "release_snapshot_cutoff_undocumented"
    if name.startswith("user__"):
        return "user_features_pure.csv", "preserve_released_null", "release_snapshot_cutoff_undocumented"
    if name.startswith("video__"):
        return "video_features_basic_pure.csv", "preserve_released_null", "release_snapshot_cutoff_undocumented"
    raise ValueError(f"Unclassified {stage} feature: {name}")


def build_feature_manifest(
    schemas: dict[str, pl.Schema], sequence_schema: pl.Schema
) -> dict[str, Any]:
    """List every persisted feature and sequence field with its use contract."""
    fs1_names = schemas["fs1"].names()
    fs2_names = schemas["fs2"].names()
    fs3_names = schemas["fs3"].names()
    if fs2_names[: len(fs1_names)] != fs1_names or fs3_names[: len(fs2_names)] != fs2_names:
        raise ValueError("Feature schemas are not nested prefixes")
    records: list[dict[str, Any]] = []
    for name, dtype in schemas["fs3"].items():
        stage = "fs1" if name in fs1_names else "fs2" if name in fs2_names else "fs3"
        source, missing, provenance = _feature_metadata(name, stage)
        models, role = _model_access(name)
        records.append(
            {
                "name": name,
                "data_type": str(dtype),
                "introduced_in": stage,
                "present_in": [candidate for candidate in ("fs1", "fs2", "fs3") if name in schemas[candidate]],
                "source": source,
                "allowed_model_families": models,
                "model_role": role,
                "missing_value_policy": missing,
                "point_in_time_provenance": provenance,
            }
        )
    sequence_records = []
    for name, dtype in sequence_schema.items():
        models, role = _model_access(name, sequence=True)
        if name == "history_video_ids":
            missing = "left_pad_0_unknown_1_known_raw_id_plus_2"
            source = "strictly_earlier_normal_policy_video_ids"
        elif name == "history_time_ms":
            missing = "padding_timestamp_0"
            source = "strictly_earlier_normal_policy_timestamps"
        elif name == "history_mask":
            missing = "0_padding_1_valid"
            source = "derived_from_history_length"
        elif name == "history_length":
            missing = "zero_for_empty_history"
            source = "derived_from_valid_history_events"
        else:
            missing = "not_nullable"
            source = "target_alignment_or_sequence_audit"
        sequence_records.append(
            {
                "name": name,
                "data_type": str(dtype),
                "source": source,
                "allowed_model_families": models,
                "model_role": role,
                "missing_value_policy": missing,
                "point_in_time_provenance": "point_in_time_verified",
            }
        )
    return {
        "manifest_version": "m2-features-v1",
        "feature_counts": {name: len(schema) for name, schema in schemas.items()},
        "feature_fields": records,
        "sequence_fields": sequence_records,
    }


def _schema(path: Path) -> pl.Schema:
    return pl.scan_parquet(path).collect_schema()


def _ordered_column_equal(left: Path, right: Path, column: str) -> bool:
    return pl.read_parquet(left, columns=[column]).equals(pl.read_parquet(right, columns=[column]))


def _validate_canonical_file(path: Path, artifact: str) -> pl.DataFrame:
    ids = pl.read_parquet(path, columns=["impression_id"])
    validate_impression_ids(ids, artifact)
    return ids


def _keyed_equal(old_path: Path, new_path: Path) -> bool:
    old = pl.read_parquet(old_path).sort("impression_id")
    new = pl.read_parquet(new_path).sort("impression_id")
    return old.schema == new.schema and old.equals(new)


def _validate_build_manifest(
    directory: Path, stage: str, split: str, feature_schema: pl.Schema
) -> None:
    path = directory / "artifact_manifest.json"
    if not path.exists():
        raise ValueError(f"Missing build manifest: {path}")
    manifest = json.loads(path.read_text())
    expected = {
        "build_version": M2_BUILD_VERSION,
        "stage": stage,
        "split": split,
        "canonical_order": CANONICAL_ORDER,
        "feature_schema": {name: str(dtype) for name, dtype in feature_schema.items()},
    }
    for key, value in expected.items():
        if manifest.get(key) != value:
            raise ValueError(f"Incompatible {stage}/{split} manifest field: {key}")


def _static_join_checks(fs1_path: Path, raw_data_dir: Path) -> dict[str, int]:
    """Verify released static values still belong to their explicit entity keys."""
    checks: dict[str, int] = {}
    definitions = (
        ("user", "user_id", "user_features_pure.csv", "user__"),
        ("video", "video_id", "video_features_basic_pure.csv", "video__"),
    )
    features = pl.scan_parquet(fs1_path)
    for entity, key, filename, prefix in definitions:
        raw = pl.read_csv(raw_data_dir / filename)
        raw_columns = [name for name in raw.columns if name != key]
        observed_columns = [prefix + name for name in raw_columns]
        expected_names = {name: prefix + name + "__expected" for name in raw_columns}
        expected = raw.rename(expected_names).lazy()
        comparison = (
            features.select(key, *observed_columns).unique()
            .join(expected, on=key, how="left", validate="m:1")
            .select(
                (~pl.all_horizontal(*(
                    pl.col(prefix + name).eq_missing(pl.col(expected_names[name]))
                    for name in raw_columns
                ))).sum().alias("mismatches")
            ).collect().item()
        )
        checks[f"{entity}_static_value_mismatches"] = comparison
        if comparison:
            raise ValueError(f"{entity} static values do not match their explicit IDs")
    return checks


def _sequence_checks(
    sequence_path: Path,
    target_path: Path,
    normal_history: pl.DataFrame,
    known_video_ids: set[int],
    length: int,
) -> dict[str, Any]:
    sequence = pl.scan_parquet(sequence_path)
    target = pl.scan_parquet(target_path).select("impression_id", "user_id", "time_ms")
    sequence_ids = _validate_canonical_file(sequence_path, "sequence")
    target_ids = _validate_canonical_file(target_path, "sequence target features")
    if not sequence_ids.equals(target_ids):
        raise ValueError("Feature and sequence impression IDs differ by row")
    schema = sequence.collect_schema()
    expected_arrays = {
        "history_video_ids": f"Array(Int32, shape=({length},))",
        "history_time_ms": f"Array(Int64, shape=({length},))",
        "history_mask": f"Array(UInt8, shape=({length},))",
    }
    shape_ok = all(str(schema[name]) == expected for name, expected in expected_arrays.items())
    masks = [pl.col("history_mask").arr.get(index) for index in range(length)]
    videos = [pl.col("history_video_ids").arr.get(index) for index in range(length)]
    times = [pl.col("history_time_ms").arr.get(index) for index in range(length)]
    checks = sequence.select(
        pl.len().alias("rows"),
        pl.col("impression_id").n_unique().alias("unique_impressions"),
        (pl.sum_horizontal(*masks) != pl.col("history_length")).sum().alias("mask_length_mismatches"),
        pl.any_horizontal(*((mask != 0) & (mask != 1) for mask in masks)).sum().alias("invalid_mask_rows"),
        pl.any_horizontal(*(masks[index] > masks[index + 1] for index in range(length - 1))).sum().alias("non_left_padded_rows"),
        pl.any_horizontal(*(((mask == 0) & (video != 0)) for mask, video in zip(masks, videos))).sum().alias("bad_padding_id_rows"),
        pl.any_horizontal(*(((mask == 1) & (video == 0)) for mask, video in zip(masks, videos))).sum().alias("valid_padding_id_rows"),
        pl.any_horizontal(*(((mask == 0) & (stamp != 0)) for mask, stamp in zip(masks, times))).sum().alias("bad_padding_time_rows"),
        pl.any_horizontal(*(((mask == 1) & (stamp >= pl.col("target_time_ms"))) for mask, stamp in zip(masks, times))).sum().alias("temporal_violation_rows"),
        pl.col("history_tie_boundary_dropped").sum().alias("tie_boundary_drop_rows"),
        pl.col("history_length").min().alias("min_history_length"),
        pl.col("history_length").max().alias("max_history_length"),
    ).collect().row(0, named=True)
    target_alignment = sequence.select("impression_id", "target_time_ms").join(
        target.select("impression_id", pl.col("time_ms").alias("expected_time_ms")),
        on="impression_id", how="full", coalesce=True,
    ).select(
        pl.len().alias("rows"),
        (pl.col("target_time_ms") != pl.col("expected_time_ms")).sum().alias("time_mismatches"),
        pl.col("target_time_ms").is_null().sum().alias("missing_sequence_rows"),
        pl.col("expected_time_ms").is_null().sum().alias("extra_sequence_rows"),
    ).collect().row(0, named=True)
    exploded = (
        sequence.join(target.select("impression_id", "user_id"), on="impression_id", validate="1:1")
        .select(
            "user_id",
            pl.col("history_video_ids").arr.to_list(),
            pl.col("history_time_ms").arr.to_list(),
            pl.col("history_mask").arr.to_list(),
        )
        .explode("history_video_ids", "history_time_ms", "history_mask")
        .filter(pl.col("history_mask") == 1)
    )
    valid_ids = exploded.select(
        pl.len().alias("valid_positions"),
        (pl.col("history_video_ids") == 1).sum().alias("unknown_positions"),
        pl.col("history_video_ids").min().alias("min_encoded_video_id"),
        pl.col("history_video_ids").max().alias("max_encoded_video_id"),
    ).collect().row(0, named=True)
    normal_keys = normal_history.select("user_id", "video_id", "time_ms").unique().lazy()
    missing_normal_matches = (
        exploded.filter(pl.col("history_video_ids") >= 2)
        .with_columns((pl.col("history_video_ids") - 2).cast(pl.Int64).alias("video_id"))
        .select("user_id", "video_id", "history_time_ms").unique()
        .join(
            normal_keys.rename({"time_ms": "history_time_ms"}),
            on=["user_id", "video_id", "history_time_ms"], how="anti",
        )
        .select(pl.len()).collect().item()
    )
    raw_min, raw_max = min(known_video_ids), max(known_video_ids)
    failures = {
        "array_shape": not shape_ok,
        **{name: bool(value) for name, value in checks.items() if name.endswith("mismatches") or name.endswith("violation_rows") or name.startswith("invalid_") or name.startswith("non_left") or name.startswith("bad_") or name.startswith("valid_padding")},
        "target_time_mismatches": bool(target_alignment["time_mismatches"]),
        "missing_sequence_rows": bool(target_alignment["missing_sequence_rows"]),
        "extra_sequence_rows": bool(target_alignment["extra_sequence_rows"]),
        "normal_history_match_failures": bool(missing_normal_matches),
        "encoded_id_collision": bool(valid_ids["min_encoded_video_id"] is not None and valid_ids["min_encoded_video_id"] < 1),
    }
    if any(failures.values()):
        raise ValueError(f"Sequence integration check failed: {failures}")
    return {
        "fixed_length": length,
        "array_shape_ok": shape_ok,
        **checks,
        **target_alignment,
        **valid_ids,
        "padding_video_id": 0,
        "unknown_video_id": 1,
        "known_encoded_id_range": [raw_min + 2, raw_max + 2],
        "padding_unknown_known_ids_disjoint": True,
        "normal_history_match_failures": missing_normal_matches,
        "history_source_verified": "normal_only_structural_rows_apr09_to_apr29",
    }


def review_m2_features(
    processed_root: Path | None = None,
    raw_data_dir: Path | None = None,
    output_dir: Path | None = None,
) -> dict[str, Any]:
    """Audit materialized training feature sets without projecting held-out outcomes."""
    started = time.perf_counter()
    artifact_paths = get_artifact_paths()
    processed_root = processed_root or artifact_paths.processed
    raw_data_dir = raw_data_dir or artifact_paths.raw / "KuaiRand-Pure" / "data"
    output_dir = output_dir or processed_root / "m2" / "v1"
    with (Path(__file__).resolve().parents[3] / "configs" / "m2_freeze.toml").open("rb") as stream:
        config = tomllib.load(stream)
    length = int(config["freeze"]["sequence_length"])
    categories = video_categories(raw_data_dir, "__MISSING__")
    known_video_ids = set(categories["video_id"].to_list())
    normal_history = load_normal_structural_history(raw_data_dir, 20220409, 20220429)
    reports: dict[str, Any] = {}
    repair_comparison: dict[str, Any] = {}
    pre_repair_root = processed_root / "m2_4a_pre_repair"
    canonical_schemas: dict[str, pl.Schema] | None = None
    sequence_schema: pl.Schema | None = None
    for split in config["freeze"]["authorized_splits"]:
        feature_paths = {
            stage: processed_root / stage / "v1" / split / "features.parquet"
            for stage in config["freeze"]["feature_sets"]
        }
        label_paths = {
            stage: processed_root / stage / "v1" / split / "labels.parquet"
            for stage in config["freeze"]["feature_sets"]
        }
        schemas = {stage: _schema(path) for stage, path in feature_paths.items()}
        if schemas["fs2"].names()[: len(schemas["fs1"])] != schemas["fs1"].names():
            raise ValueError(f"{split}: FS2 does not nest FS1")
        if schemas["fs3"].names()[: len(schemas["fs2"])] != schemas["fs2"].names():
            raise ValueError(f"{split}: FS3 does not nest FS2")
        for stage, schema in schemas.items():
            if set(LABEL_COLUMNS).intersection(schema.names()):
                raise ValueError(f"{split}: outcomes entered {stage}")
            _validate_build_manifest(feature_paths[stage].parent, stage, split, schema)
            feature_ids = _validate_canonical_file(feature_paths[stage], f"{stage}/{split} features")
            label_ids = _validate_canonical_file(label_paths[stage], f"{stage}/{split} labels")
            if not feature_ids.equals(label_ids):
                raise ValueError(f"{split}: {stage} features and labels differ by row")
        if not all(_ordered_column_equal(feature_paths["fs1"], feature_paths[stage], "impression_id") for stage in ("fs2", "fs3")):
            raise ValueError(f"{split}: impression order changed across feature sets")
        if not all(_ordered_column_equal(label_paths["fs1"], label_paths[stage], "impression_id") for stage in ("fs2", "fs3")):
            raise ValueError(f"{split}: label impression order changed")
        if not _ordered_column_equal(feature_paths["fs1"], label_paths["fs1"], "impression_id"):
            raise ValueError(f"{split}: features and labels are not row-aligned")
        label_hashes = {stage: _sha256(path) for stage, path in label_paths.items()}
        if len(set(label_hashes.values())) != 1:
            raise ValueError(f"{split}: label artifacts differ across stages")
        sequence_path = processed_root / "fs3" / "v1" / split / "sequence.parquet"
        current_sequence_schema = _schema(sequence_path)
        sequence = _sequence_checks(
            sequence_path, feature_paths["fs3"], normal_history, known_video_ids, length
        )
        category_check = (
            pl.scan_parquet(feature_paths["fs3"])
            .select("video_id", "category__key", "category__atomic_count").unique()
            .join(categories.lazy(), on="video_id", how="left", suffix="_expected", validate="1:1")
            .select(
                (pl.col("category__key") != pl.col("category__key_expected")).sum().alias("key_mismatches"),
                (pl.col("category__atomic_count") != pl.col("category__atomic_count_expected")).sum().alias("count_mismatches"),
                (pl.col("category__key") == "__MISSING__").sum().alias("missing_category_videos_in_split"),
                pl.col("category__key").n_unique().alias("category_cardinality_in_split"),
            ).collect().row(0, named=True)
        )
        if category_check["key_mismatches"] or category_check["count_mismatches"]:
            raise ValueError(f"{split}: category representation mismatch")
        rows = {
            stage: pl.scan_parquet(path).select(pl.len()).collect().item()
            for stage, path in feature_paths.items()
        }
        reports[split] = {
            "rows": rows,
            "identical_impression_order": True,
            "feature_label_row_alignment": True,
            "label_sha256": label_hashes["fs1"],
            "labels_byte_identical_across_stages": True,
            "feature_nesting": {"fs2_contains_fs1": True, "fs3_contains_fs2": True},
            "model_feature_counts": {
                stage: {
                    workflow: len(model_feature_columns(schema.names(), workflow))
                    for workflow in ("baseline", "rq3_random_only", "rq3_joint")
                }
                for stage, schema in schemas.items()
            },
            "category": category_check,
            "static_join": _static_join_checks(feature_paths["fs1"], raw_data_dir),
            "sequence": sequence,
        }
        old_feature_paths = {
            stage: pre_repair_root / stage / "v1" / split / "features.parquet"
            for stage in config["freeze"]["feature_sets"]
        }
        old_label_paths = {
            stage: pre_repair_root / stage / "v1" / split / "labels.parquet"
            for stage in config["freeze"]["feature_sets"]
        }
        old_sequence_path = pre_repair_root / "fs3" / "v1" / split / "sequence.parquet"
        if all(path.exists() for path in [*old_feature_paths.values(), *old_label_paths.values(), old_sequence_path]):
            old_feature_ids = pl.read_parquet(old_feature_paths["fs1"], columns=["impression_id"])
            old_label_ids = pl.read_parquet(old_label_paths["fs1"], columns=["impression_id"])
            new_feature_ids = pl.read_parquet(feature_paths["fs1"], columns=["impression_id"])
            new_label_ids = pl.read_parquet(label_paths["fs1"], columns=["impression_id"])
            comparison = {
                "before_positional_matches": int(
                    (old_feature_ids["impression_id"] == old_label_ids["impression_id"]).sum()
                ),
                "after_positional_matches": int(
                    (new_feature_ids["impression_id"] == new_label_ids["impression_id"]).sum()
                ),
                "rows": new_feature_ids.height,
                "features_keyed_equal": {
                    stage: _keyed_equal(old_feature_paths[stage], feature_paths[stage])
                    for stage in config["freeze"]["feature_sets"]
                },
                "labels_keyed_equal": {
                    stage: _keyed_equal(old_label_paths[stage], label_paths[stage])
                    for stage in config["freeze"]["feature_sets"]
                },
                "sequence_keyed_equal": _keyed_equal(old_sequence_path, sequence_path),
            }
            if not all(comparison["features_keyed_equal"].values()):
                raise ValueError(f"{split}: feature values changed in keyed repair comparison")
            if not all(comparison["labels_keyed_equal"].values()):
                raise ValueError(f"{split}: label values changed in keyed repair comparison")
            if not comparison["sequence_keyed_equal"]:
                raise ValueError(f"{split}: sequence values changed in keyed repair comparison")
            repair_comparison[split] = comparison
        if canonical_schemas is None:
            canonical_schemas = schemas
            sequence_schema = current_sequence_schema
        elif any(schemas[stage] != canonical_schemas[stage] for stage in schemas):
            raise ValueError("Feature schema differs across authorized sources")
        elif current_sequence_schema != sequence_schema:
            raise ValueError("Sequence schema differs across authorized sources")
    assert canonical_schemas is not None and sequence_schema is not None
    manifest = build_feature_manifest(canonical_schemas, sequence_schema)
    manifest["selector_contract"] = {
        "baseline_excludes_context__is_rand": True,
        "rq3_random_only_excludes_context__is_rand": True,
        "rq3_joint_includes_context__is_rand": True,
    }
    upload_dates = pl.read_csv(raw_data_dir / "video_features_basic_pure.csv", columns=["upload_dt"])
    report = {
        "milestone": "M2.4",
        "freeze_version": config["freeze"]["version"],
        "authorized_splits_only": list(config["freeze"]["authorized_splits"]),
        "held_out_outcomes_accessed": False,
        "feature_counts": manifest["feature_counts"],
        "sequence_field_count": len(manifest["sequence_fields"]),
        "splits": reports,
        "static_provenance": {
            **config["static_provenance"],
            "released_upload_dt_min": upload_dates["upload_dt"].min(),
            "released_upload_dt_max": upload_dates["upload_dt"].max(),
            "future_upload_rows_in_authorized_training": 0,
        },
        "category_release": {
            "videos": categories.height,
            "missing_category_videos": categories.filter(pl.col("category__key") == "__MISSING__").height,
            "canonical_category_cardinality": categories["category__key"].n_unique(),
        },
        "repair": {
            "root_cause": "FS1 user static left join omitted maintain_order=left; normal feature rows were reordered while labels retained source order",
            "canonical_order": CANONICAL_ORDER,
            "comparison_to_pre_repair_snapshot": repair_comparison,
            "repair_changes_order_only": True,
        },
        "artifact_bytes": sum(
            path.stat().st_size
            for stage in config["freeze"]["feature_sets"]
            for path in (processed_root / stage / "v1").rglob("*")
            if path.is_file()
        ),
        "runtime_seconds": time.perf_counter() - started,
        "peak_rss_mib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / config["freeze"]["manifest_filename"]).write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )
    (output_dir / config["freeze"]["report_filename"]).write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n"
    )
    return report
