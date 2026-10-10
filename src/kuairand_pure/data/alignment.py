"""Canonical artifact alignment and fail-closed model input loading."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import polars as pl

M2_BUILD_VERSION = "m2-features-v1"
CANONICAL_ORDER = "impression_id_ascending"


def validate_impression_ids(frame: pl.DataFrame, artifact: str) -> None:
    """Require unique, non-null, ascending IDs; never repair inputs silently."""
    if "impression_id" not in frame.columns:
        raise ValueError(f"{artifact} has no impression_id")
    ids = frame["impression_id"]
    if ids.null_count():
        raise ValueError(f"{artifact} has missing impression IDs")
    if ids.n_unique() != frame.height:
        raise ValueError(f"{artifact} has duplicate impression IDs")
    if not ids.is_sorted():
        raise ValueError(f"{artifact} is not in canonical impression_id order")


def validate_row_alignment(
    features: pl.DataFrame,
    labels: pl.DataFrame,
    sequence: pl.DataFrame | None = None,
) -> None:
    """Require exact positional ID equality across already-canonical artifacts."""
    validate_impression_ids(features, "features")
    validate_impression_ids(labels, "labels")
    if not features["impression_id"].equals(labels["impression_id"]):
        raise ValueError("Feature and label impression IDs differ by row")
    if sequence is not None:
        validate_impression_ids(sequence, "sequence")
        if not features["impression_id"].equals(sequence["impression_id"]):
            raise ValueError("Feature and sequence impression IDs differ by row")


def write_artifact_manifest(
    directory: Path,
    *,
    stage: str,
    split: str,
    features_path: Path,
    labels_path: Path,
    sequence_path: Path | None = None,
) -> dict[str, Any]:
    """Validate an artifact bundle and write its build-compatibility sidecar."""
    features = pl.read_parquet(features_path)
    labels = pl.read_parquet(labels_path)
    sequence = pl.read_parquet(sequence_path) if sequence_path is not None else None
    validate_row_alignment(features, labels, sequence)
    manifest = {
        "build_version": M2_BUILD_VERSION,
        "stage": stage,
        "split": split,
        "canonical_order": CANONICAL_ORDER,
        "rows": features.height,
        "feature_schema": {name: str(dtype) for name, dtype in features.schema.items()},
        "label_schema": {name: str(dtype) for name, dtype in labels.schema.items()},
        "sequence_schema": (
            {name: str(dtype) for name, dtype in sequence.schema.items()}
            if sequence is not None else None
        ),
    }
    (directory / "artifact_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )
    return manifest


def load_aligned_model_data(
    features_path: Path,
    labels_path: Path,
    *,
    expected_stage: str,
    expected_split: str,
    sequence_path: Path | None = None,
    expected_build_version: str = M2_BUILD_VERSION,
) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame | None]:
    """Load X/y (and optional history) only after strict manifest and row checks."""
    manifest_path = features_path.parent / "artifact_manifest.json"
    if not manifest_path.exists():
        raise ValueError(f"Missing artifact manifest: {manifest_path}")
    manifest = json.loads(manifest_path.read_text())
    expected = {
        "build_version": expected_build_version,
        "stage": expected_stage,
        "split": expected_split,
        "canonical_order": CANONICAL_ORDER,
    }
    for field, value in expected.items():
        if manifest.get(field) != value:
            raise ValueError(
                f"Incompatible artifact manifest {field}: "
                f"expected {value!r}, found {manifest.get(field)!r}"
            )
    features = pl.read_parquet(features_path)
    labels = pl.read_parquet(labels_path)
    sequence = pl.read_parquet(sequence_path) if sequence_path is not None else None
    if manifest.get("rows") != features.height:
        raise ValueError("Artifact row count differs from its manifest")
    observed_feature_schema = {name: str(dtype) for name, dtype in features.schema.items()}
    if manifest.get("feature_schema") != observed_feature_schema:
        raise ValueError("Feature schema differs from its artifact manifest")
    if manifest.get("label_schema") != {name: str(dtype) for name, dtype in labels.schema.items()}:
        raise ValueError("Label schema differs from its artifact manifest")
    if sequence is not None and manifest.get("sequence_schema") != {
        name: str(dtype) for name, dtype in sequence.schema.items()
    }:
        raise ValueError("Sequence schema differs from its artifact manifest")
    validate_row_alignment(features, labels, sequence)
    return features, labels, sequence
