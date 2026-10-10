import polars as pl
import pytest

from kuairand_pure.features.fs3 import (
    _daily_affinity_snapshots,
    _daily_recency_snapshots,
    atomic_tag_count,
    build_fs3_lazy,
    build_sequence_frame,
    canonicalize_tag,
    video_categories,
)

PRIORS = {
    "long_view": 0.25,
    "like": 0.10,
    "valid_play": 0.40,
    "literal_click": 0.20,
}


def _fs2(rows: list[tuple[int, int, int, int, int]]) -> pl.DataFrame:
    return pl.DataFrame(
        {
            "impression_id": [row[0] for row in rows],
            "user_id": [row[1] for row in rows],
            "video_id": [row[2] for row in rows],
            "date": [row[3] for row in rows],
            "time_ms": [row[4] for row in rows],
            "context__tab": [1] * len(rows),
            "context__is_rand": [0] * len(rows),
            "fs2_value": [7] * len(rows),
        }
    )


def _categories() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "video_id": [10, 11, 12, 99],
            "category__key": ["1|2", "1|2", "3", "__MISSING__"],
            "category__atomic_count": pl.Series([2, 2, 1, 0], dtype=pl.Int16),
        }
    )


def _outcomes(rows: list[tuple[int, int, int, str, int, int, int, int]]) -> pl.DataFrame:
    return pl.DataFrame(
        rows,
        schema=(
            "user_id", "video_id", "date", "category__key", "tab",
            "long_view", "is_click", "is_like",
        ),
        orient="row",
    )


def _structural(rows: list[tuple[int, int, int, str, int]]) -> pl.DataFrame:
    return pl.DataFrame(
        rows,
        schema=("user_id", "video_id", "date", "category__key", "time_ms"),
        orient="row",
    )


def _build(fs2: pl.DataFrame, outcomes: pl.DataFrame, structural: pl.DataFrame) -> pl.DataFrame:
    return build_fs3_lazy(
        fs2.lazy(),
        _categories(),
        _daily_affinity_snapshots(outcomes),
        _daily_recency_snapshots(structural, ["user_id"], "user__recency_"),
        _daily_recency_snapshots(
            structural, ["user_id", "category__key"], "user_category__recency_"
        ),
        PRIORS,
        alpha=2.0,
    ).collect()


def test_multivalued_and_missing_tags_are_canonicalized_deterministically() -> None:
    assert canonicalize_tag("65, 12,65") == "12|65"
    assert canonicalize_tag("12,65") == "12|65"
    assert canonicalize_tag(None) == "__MISSING__"
    assert canonicalize_tag("  ") == "__MISSING__"
    assert atomic_tag_count("12|65") == 2
    assert atomic_tag_count("__MISSING__") == 0


def test_video_category_loader_maps_null_tags_to_missing(tmp_path, monkeypatch) -> None:
    video_path = tmp_path / "videos.csv"
    pl.DataFrame({"video_id": [1, 2], "tag": [None, "65,12"]}).write_csv(video_path)
    monkeypatch.setattr(
        "kuairand_pure.features.fs3._paths",
        lambda _: ({}, {"videos_basic": video_path}),
    )

    output = video_categories(tmp_path, "__MISSING__")

    assert output["category__key"].to_list() == ["__MISSING__", "12|65"]
    assert output["category__atomic_count"].to_list() == [0, 2]


def test_fs3_preserves_fs2_and_computes_affinity_and_recency() -> None:
    fs2 = _fs2([(100, 1, 10, 20220410, 300)])
    outcomes = _outcomes([(1, 11, 20220409, "1|2", 1, 1, 1, 0)])
    structural = _structural([(1, 11, 20220409, "1|2", 100)])

    output = _build(fs2, outcomes, structural)
    row = output.row(0, named=True)

    assert output.select(fs2.columns).equals(fs2)
    assert row["user_category__hist_impression_count"] == 1
    assert row["user_category__hist_long_view_positive_count"] == 1
    assert row["user_category__hist_long_view_rate"] == pytest.approx((1 + 0.5) / 3)
    assert row["user__recency_since_last_normal_ms"] == 200
    assert row["user_category__recency_since_last_normal_ms"] == 200
    assert row["user__recency_missing"] is False


def test_same_day_target_and_future_outcomes_do_not_change_fs3() -> None:
    target = _fs2([(100, 1, 10, 20220410, 300)])
    base = _outcomes(
        [
            (1, 11, 20220409, "1|2", 1, 0, 0, 0),
            (1, 10, 20220410, "1|2", 1, 0, 0, 0),
            (1, 10, 20220411, "1|2", 1, 0, 0, 0),
        ]
    )
    changed = base.with_columns(
        pl.when(pl.col("date") >= 20220410).then(1).otherwise(pl.col("long_view")).alias("long_view"),
        pl.when(pl.col("date") >= 20220410).then(1).otherwise(pl.col("is_like")).alias("is_like"),
    )
    structural = _structural([(1, 11, 20220409, "1|2", 100)])

    assert _build(target, base, structural).to_dicts() == _build(
        target, changed, structural
    ).to_dicts()


def test_cold_start_user_category_has_safe_defaults() -> None:
    target = _fs2([(100, 9, 99, 20220410, 300)])
    output = _build(
        target,
        _outcomes([(1, 11, 20220409, "1|2", 1, 1, 1, 1)]),
        _structural([(1, 11, 20220409, "1|2", 100)]),
    )
    row = output.row(0, named=True)

    assert row["category__key"] == "__MISSING__"
    assert row["user_category__hist_impression_count"] == 0
    assert row["user_category__hist_long_view_rate"] == pytest.approx(PRIORS["long_view"])
    assert row["user__recency_since_last_normal_ms"] == -1
    assert row["user__recency_missing"] is True
    assert row["user_category__recency_missing"] is True


def test_random_target_has_same_schema_and_uses_only_supplied_normal_history() -> None:
    normal = _fs2([(100, 1, 10, 20220410, 300)])
    random = normal.with_columns(pl.lit(1, dtype=pl.Int64).alias("context__is_rand"))
    outcomes = _outcomes([(1, 11, 20220409, "1|2", 1, 1, 1, 0)])
    structural = _structural([(1, 11, 20220409, "1|2", 100)])

    assert _build(normal, outcomes, structural).schema == _build(
        random, outcomes, structural
    ).schema


def test_sequence_is_strictly_earlier_and_cold_start_is_padded() -> None:
    targets = _fs2([(100, 1, 10, 20220410, 200), (101, 9, 99, 20220410, 200)])
    history = pl.DataFrame(
        {
            "impression_id": [1, 2, 3],
            "user_id": [1, 1, 1],
            "video_id": [10, 11, 12],
            "time_ms": [100, 200, 300],
        }
    )

    sequence = build_sequence_frame(
        targets.select("impression_id", "user_id", "time_ms"),
        history,
        {10, 11, 12},
        length=3,
        padding_id=0,
        unknown_id=1,
        known_offset=2,
        padding_time_ms=0,
    )

    assert sequence["history_length"].to_list() == [1, 0]
    assert sequence["history_video_ids"].to_list()[0] == [0, 0, 12]
    assert sequence["history_time_ms"].to_list()[0] == [0, 0, 100]
    assert sequence["history_mask"].to_list()[1] == [0, 0, 0]
    assert all(
        history_time < 200
        for history_time, mask in zip(
            sequence["history_time_ms"].to_list()[0], sequence["history_mask"].to_list()[0]
        )
        if mask
    )


def test_timestamp_ties_are_input_order_invariant_and_not_split_at_boundary() -> None:
    target = _fs2([(100, 1, 10, 20220410, 300)]).select(
        "impression_id", "user_id", "time_ms"
    )
    history = pl.DataFrame(
        {
            "impression_id": [1, 2, 3, 4],
            "user_id": [1, 1, 1, 1],
            "video_id": [10, 13, 12, 11],
            "time_ms": [100, 200, 200, 200],
        }
    )
    kwargs = dict(
        known_video_ids={10, 11, 12, 13}, length=2, padding_id=0,
        unknown_id=1, known_offset=2, padding_time_ms=0,
    )

    first = build_sequence_frame(target, history, **kwargs)
    shuffled = build_sequence_frame(target, history.reverse(), **kwargs)

    assert first.equals(shuffled)
    assert first["history_length"].item() == 0
    assert first["history_tie_boundary_dropped"].item() is True


def test_unknown_history_video_has_explicit_representation() -> None:
    target = _fs2([(100, 1, 10, 20220410, 200)]).select(
        "impression_id", "user_id", "time_ms"
    )
    history = pl.DataFrame(
        {"impression_id": [1], "user_id": [1], "video_id": [999], "time_ms": [100]}
    )
    output = build_sequence_frame(
        target, history, {10}, length=2, padding_id=0, unknown_id=1,
        known_offset=2, padding_time_ms=0,
    )

    assert output["history_video_ids"].to_list()[0] == [0, 1]
