"""Structural-only audit utilities for the KuaiRand-Pure release.

This module deliberately never aggregates engagement values. In particular, it
does not inspect outcomes from the frozen May 3--8, 2022 test window.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import polars as pl

from kuairand_pure.paths import get_artifact_paths

EXPECTED_FILES = (
    "log_random_4_22_to_5_08_pure.csv",
    "log_standard_4_08_to_4_21_pure.csv",
    "log_standard_4_22_to_5_08_pure.csv",
    "user_features_pure.csv",
    "video_features_basic_pure.csv",
    "video_features_statistic_pure.csv",
)
EXPECTED_ARCHIVE_MD5 = "0820331067a3784d9691136f772b35a7"
LOG_PREFIXES = ("log_random_", "log_standard_")
OUTCOME_COLUMNS = frozenset(
    {
        "is_click",
        "is_like",
        "is_follow",
        "is_comment",
        "is_forward",
        "is_hate",
        "long_view",
        "play_time_ms",
        "profile_stay_time",
        "comment_stay_time",
        "is_profile_enter",
    }
)
CANDIDATE_COLUMNS = {
    "long_view": "long_view",
    "click": "is_click",
    "like": "is_like",
    "play_time": "play_time_ms",
    "video_duration": "duration_ms",
    "random_exposure_indicator": "is_rand",
    "interface_or_policy": "tab",
}


def file_digest(path: Path, algorithm: str = "sha256") -> str:
    """Hash a file without loading it into memory."""
    digest = hashlib.new(algorithm)
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _timestamp_iso(milliseconds: int | None) -> str | None:
    if milliseconds is None:
        return None
    return datetime.fromtimestamp(milliseconds / 1000, tz=UTC).isoformat()


def audit_csv(path: Path) -> dict[str, Any]:
    """Return schema and structural counts, never outcome distributions."""
    frame = pl.scan_csv(path, infer_schema_length=10_000, try_parse_dates=False)
    schema = frame.collect_schema()
    columns = schema.names()

    expressions: list[pl.Expr] = [pl.len().alias("row_count")]
    expressions.extend(pl.col(column).null_count().alias(column) for column in columns)
    summary = frame.select(expressions).collect().row(0, named=True)

    result: dict[str, Any] = {
        "bytes": path.stat().st_size,
        "sha256": file_digest(path),
        "row_count": summary.pop("row_count"),
        "schema": {name: str(dtype) for name, dtype in schema.items()},
        "missing_values": summary,
    }

    identifier_expressions = []
    for identifier in ("user_id", "video_id"):
        if identifier in columns:
            identifier_expressions.append(
                pl.col(identifier).n_unique().alias(f"unique_{identifier}s")
            )
    if identifier_expressions:
        result["identifier_counts"] = (
            frame.select(identifier_expressions).collect().row(0, named=True)
        )

    if path.name.startswith(LOG_PREFIXES):
        coverage = (
            frame.select(
                pl.col("date").min().alias("minimum_date"),
                pl.col("date").max().alias("maximum_date"),
                pl.col("time_ms").min().alias("minimum_time_ms"),
                pl.col("time_ms").max().alias("maximum_time_ms"),
            )
            .collect()
            .row(0, named=True)
        )
        coverage["minimum_time_utc"] = _timestamp_iso(coverage["minimum_time_ms"])
        coverage["maximum_time_utc"] = _timestamp_iso(coverage["maximum_time_ms"])
        result["timestamp_coverage"] = coverage
        result["rows_by_date"] = {
            str(row["date"]): row["row_count"]
            for row in (
                frame.group_by("date")
                .agg(pl.len().alias("row_count"))
                .sort("date")
                .collect()
                .iter_rows(named=True)
            )
        }
        result["candidate_columns_present"] = {
            role: column in columns for role, column in CANDIDATE_COLUMNS.items()
        }
        result["exposure_metadata_values"] = {
            "is_rand": sorted(
                frame.select(pl.col("is_rand").drop_nulls().unique())
                .collect()
                .get_column("is_rand")
                .to_list()
            ),
            "tab": sorted(
                frame.select(pl.col("tab").drop_nulls().unique())
                .collect()
                .get_column("tab")
                .to_list()
            ),
        }

    return result


def run_audit(raw_data_dir: Path, archive_path: Path | None = None) -> dict[str, Any]:
    """Audit the exact KuaiRand-Pure file set and determine protocol branch."""
    missing = [name for name in EXPECTED_FILES if not (raw_data_dir / name).is_file()]
    if missing:
        raise FileNotFoundError(f"Missing expected KuaiRand-Pure files: {missing}")

    files = {name: audit_csv(raw_data_dir / name) for name in EXPECTED_FILES}
    normal_names = (
        "log_standard_4_08_to_4_21_pure.csv",
        "log_standard_4_22_to_5_08_pure.csv",
    )
    combined_normal_ids = pl.concat(
        [
            pl.scan_csv(raw_data_dir / name).select("user_id", "video_id")
            for name in normal_names
        ]
    )
    combined_normal_counts = (
        combined_normal_ids.select(
            pl.col("user_id").n_unique().alias("unique_user_ids"),
            pl.col("video_id").n_unique().alias("unique_video_ids"),
        )
        .collect()
        .row(0, named=True)
    )
    combined_normal_counts["row_count"] = sum(files[name]["row_count"] for name in normal_names)
    later_normal = files["log_standard_4_22_to_5_08_pure.csv"]
    maximum_normal_date = later_normal["timestamp_coverage"]["maximum_date"]
    branch = "B1" if maximum_normal_date >= 20220508 else "B2"

    report: dict[str, Any] = {
        "audit_kind": "structural_only",
        "outcome_value_aggregation_performed": False,
        "test_outcome_embargo": "No May 3-8 engagement values or rates inspected.",
        "raw_data_directory": str(raw_data_dir),
        "candidate_columns": CANDIDATE_COLUMNS,
        "combined_normal_logs": combined_normal_counts,
        "files": files,
        "branch": {
            "determination": branch,
            "rule": "B1 iff observed standard-log coverage continues through 2022-05-08.",
            "observed_maximum_normal_date": maximum_normal_date,
        },
    }
    if archive_path is not None:
        observed_md5 = file_digest(archive_path, "md5")
        if observed_md5 != EXPECTED_ARCHIVE_MD5:
            raise ValueError(
                f"Archive MD5 mismatch: expected {EXPECTED_ARCHIVE_MD5}, got {observed_md5}"
            )
        report["archive"] = {
            "path": str(archive_path),
            "bytes": archive_path.stat().st_size,
            "md5": observed_md5,
            "md5_verified": True,
        }
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    default_root = get_artifact_paths()
    parser.add_argument(
        "--raw-data-dir",
        type=Path,
        default=default_root.raw / "KuaiRand-Pure" / "data",
    )
    parser.add_argument(
        "--archive",
        type=Path,
        default=default_root.raw / "KuaiRand-Pure.tar.gz",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=default_root.processed / "audits" / "m1_1_structural_audit.json",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    report = run_audit(args.raw_data_dir, args.archive)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(args.output)


if __name__ == "__main__":
    main()
