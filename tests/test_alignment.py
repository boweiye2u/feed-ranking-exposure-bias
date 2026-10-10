import json
from pathlib import Path

import polars as pl
import pytest

from kuairand_pure.data.alignment import (
    M2_BUILD_VERSION,
    load_aligned_model_data,
    validate_impression_ids,
    validate_row_alignment,
    write_artifact_manifest,
)


def _frame(ids: list[int], value_name: str) -> pl.DataFrame:
    return pl.DataFrame(
        {"impression_id": pl.Series(ids, dtype=pl.UInt64), value_name: range(len(ids))}
    )


def test_equal_id_sets_in_different_order_fail_alignment() -> None:
    features = _frame([1, 2, 3], "x")
    labels = _frame([2, 1, 3], "y")

    with pytest.raises(ValueError, match="canonical|differ by row"):
        validate_row_alignment(features, labels)


def test_independently_reordered_sequence_fails_alignment() -> None:
    features = _frame([1, 2, 3], "x")
    labels = _frame([1, 2, 3], "y")
    sequence = _frame([1, 3, 2], "history_length")

    with pytest.raises(ValueError, match="canonical|sequence impression IDs"):
        validate_row_alignment(features, labels, sequence)


def test_canonical_sorting_passes_alignment() -> None:
    features = _frame([3, 1, 2], "x").sort("impression_id")
    labels = _frame([2, 3, 1], "y").sort("impression_id")
    sequence = _frame([1, 3, 2], "history_length").sort("impression_id")

    validate_row_alignment(features, labels, sequence)


@pytest.mark.parametrize(
    ("ids", "message"),
    [([1, 1], "duplicate"), ([1, None], "missing")],
)
def test_duplicate_or_missing_impression_ids_are_rejected(ids, message) -> None:
    frame = pl.DataFrame(
        {"impression_id": pl.Series(ids, dtype=pl.UInt64), "x": [0, 1]}
    )

    with pytest.raises(ValueError, match=message):
        validate_impression_ids(frame, "synthetic")


def test_model_loader_fails_instead_of_realigning(tmp_path: Path) -> None:
    features_path = tmp_path / "features.parquet"
    labels_path = tmp_path / "labels.parquet"
    _frame([1, 2], "x").write_parquet(features_path)
    _frame([2, 1], "y").write_parquet(labels_path)
    manifest = {
        "build_version": M2_BUILD_VERSION,
        "stage": "fs1",
        "split": "normal_train",
        "canonical_order": "impression_id_ascending",
        "rows": 2,
        "feature_schema": {"impression_id": "UInt64", "x": "Int64"},
        "label_schema": {"impression_id": "UInt64", "y": "Int64"},
        "sequence_schema": None,
    }
    (tmp_path / "artifact_manifest.json").write_text(json.dumps(manifest))

    with pytest.raises(ValueError, match="canonical|differ by row"):
        load_aligned_model_data(
            features_path,
            labels_path,
            expected_stage="fs1",
            expected_split="normal_train",
        )


def test_model_loader_rejects_stale_build_manifest(tmp_path: Path) -> None:
    features_path = tmp_path / "features.parquet"
    labels_path = tmp_path / "labels.parquet"
    _frame([1, 2], "x").write_parquet(features_path)
    _frame([1, 2], "y").write_parquet(labels_path)
    write_artifact_manifest(
        tmp_path,
        stage="fs1",
        split="normal_train",
        features_path=features_path,
        labels_path=labels_path,
    )
    manifest_path = tmp_path / "artifact_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["build_version"] = "stale-v0"
    manifest_path.write_text(json.dumps(manifest))

    with pytest.raises(ValueError, match="Incompatible artifact manifest build_version"):
        load_aligned_model_data(
            features_path,
            labels_path,
            expected_stage="fs1",
            expected_split="normal_train",
        )
