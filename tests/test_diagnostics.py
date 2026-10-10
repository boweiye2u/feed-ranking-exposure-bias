from pathlib import Path

import polars as pl

from kuairand_pure.diagnostics import (
    engagement_diagnostics,
    load_id_window,
    load_permitted_random_slice,
    timestamp_consistency,
)


def _write_log(path: Path) -> None:
    pl.DataFrame(
        {
            "user_id": [1, 1, 1, 2],
            "video_id": [10, 11, 12, 13],
            "date": [20220422, 20220429, 20220430, 20220503],
            "hourmin": [1000, 1001, 1002, 1003],
            "time_ms": [1_650_000_000_000] * 4,
            "tab": [1, 1, 1, 1],
            "long_view": [1, 0, 99, 99],
            "is_click": [1, 0, 99, 99],
            "is_like": [1, 0, 99, 99],
            "play_time_ms": [20_000, 1_000, 99, 99],
            "duration_ms": [30_000, 30_000, 99, 99],
        }
    ).write_csv(path)


def test_outcomes_are_restricted_to_permitted_random_slice(tmp_path: Path) -> None:
    path = tmp_path / "random.csv"
    _write_log(path)

    frame = load_permitted_random_slice(path)
    report = engagement_diagnostics(frame)

    assert frame["date"].to_list() == [20220422, 20220429]
    assert report["label_frequencies"]["long_view"]["positive_count"] == 1
    assert report["long_view_pairs"]["users_with_both_classes"] == 1
    assert report["long_view_pairs"]["total_positive_negative_pairs"] == 1


def test_held_out_window_projects_identifiers_only(tmp_path: Path) -> None:
    path = tmp_path / "random.csv"
    _write_log(path)

    frame = load_id_window(path, 20220503, 20220508)

    assert frame.columns == ["user_id", "video_id", "date"]
    assert frame.to_dicts() == [{"user_id": 2, "video_id": 13, "date": 20220503}]


def test_timestamp_consistency_detects_utc_plus_8_hour_bucket(tmp_path: Path) -> None:
    path = tmp_path / "timestamps.csv"
    pl.DataFrame(
        {
            "date": [20220503],
            "hourmin": [800],
            "time_ms": [1651535940163],  # 2022-05-03 07:59:00.163 at UTC+8
        }
    ).write_csv(path)

    report = timestamp_consistency(path)

    assert report["utc_plus_8_date_match_rate"] == 1
    assert report["utc_plus_8_hour_within_one_bucket_rate"] == 1
