from pathlib import Path

import polars as pl

from kuairand_pure.audit import EXPECTED_FILES, OUTCOME_COLUMNS, run_audit


LOG_COLUMNS = {
    "user_id": [1, 1],
    "video_id": [10, 11],
    "date": [20220422, 20220508],
    "hourmin": [100, 200],
    "time_ms": [1650589200000, 1651975200000],
    "is_click": [0, 1],
    "is_like": [0, 1],
    "is_follow": [0, 0],
    "is_comment": [0, 0],
    "is_forward": [0, 0],
    "is_hate": [0, 0],
    "long_view": [0, 1],
    "play_time_ms": [100, 200],
    "duration_ms": [300, 300],
    "profile_stay_time": [0, 0],
    "comment_stay_time": [0, 0],
    "is_profile_enter": [0, 0],
    "is_rand": [1, 1],
    "tab": [1, 1],
}


def _write_fixture(directory: Path) -> None:
    for name in EXPECTED_FILES:
        path = directory / name
        if name.startswith("log_"):
            pl.DataFrame(LOG_COLUMNS).write_csv(path)
        elif name == "user_features_pure.csv":
            pl.DataFrame({"user_id": [1]}).write_csv(path)
        else:
            pl.DataFrame({"video_id": [10, 11]}).write_csv(path)


def test_structural_audit_selects_b1_without_outcome_statistics(tmp_path: Path) -> None:
    _write_fixture(tmp_path)

    report = run_audit(tmp_path)

    assert report["branch"]["determination"] == "B1"
    assert report["combined_normal_logs"] == {
        "unique_user_ids": 1,
        "unique_video_ids": 2,
        "row_count": 4,
    }
    assert report["files"][EXPECTED_FILES[0]]["exposure_metadata_values"] == {
        "is_rand": [1],
        "tab": [1],
    }
    assert report["outcome_value_aggregation_performed"] is False
    keys: set[str] = set()

    def collect_keys(value) -> None:
        if isinstance(value, dict):
            keys.update(value)
            for child in value.values():
                collect_keys(child)
        elif isinstance(value, list):
            for child in value:
                collect_keys(child)

    collect_keys(report)
    for forbidden_statistic in (
        "mean",
        "label_rate",
        "positive_count",
        "negative_count",
        "value_counts",
    ):
        assert forbidden_statistic not in keys
    assert OUTCOME_COLUMNS.isdisjoint(report.keys())


def test_structural_audit_reports_missing_expected_file(tmp_path: Path) -> None:
    _write_fixture(tmp_path)
    (tmp_path / EXPECTED_FILES[0]).unlink()

    try:
        run_audit(tmp_path)
    except FileNotFoundError as exc:
        assert EXPECTED_FILES[0] in str(exc)
    else:
        raise AssertionError("Expected a missing-file error")


def test_structural_audit_rejects_wrong_archive_checksum(tmp_path: Path) -> None:
    _write_fixture(tmp_path)
    archive = tmp_path / "KuaiRand-Pure.tar.gz"
    archive.write_bytes(b"not the official archive")

    try:
        run_audit(tmp_path, archive)
    except ValueError as exc:
        assert "MD5 mismatch" in str(exc)
    else:
        raise AssertionError("Expected an archive checksum error")
