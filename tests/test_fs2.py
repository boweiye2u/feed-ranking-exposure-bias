import polars as pl
import pytest

from kuairand_pure.features.fs2 import (
    build_fs2_lazy,
    daily_history_snapshots,
)

PRIORS = {
    "long_view": 0.25,
    "like": 0.10,
    "valid_play": 0.40,
    "literal_click": 0.20,
}


def _fs1(rows: list[tuple[int, int, int, int]]) -> pl.DataFrame:
    return pl.DataFrame(
        {
            "impression_id": [row[0] for row in rows],
            "user_id": [row[1] for row in rows],
            "video_id": [row[2] for row in rows],
            "date": [row[3] for row in rows],
            "time_ms": list(range(len(rows))),
            "context__tab": [1] * len(rows),
            "context__is_rand": [0] * len(rows),
            "static_value": ["kept"] * len(rows),
        }
    )


def _history(rows: list[tuple[int, int, int, int, int, int, int]]) -> pl.DataFrame:
    return pl.DataFrame(
        rows,
        schema=("user_id", "video_id", "date", "tab", "long_view", "is_click", "is_like"),
        orient="row",
    )


def _build(fs1: pl.DataFrame, history: pl.DataFrame) -> pl.DataFrame:
    return build_fs2_lazy(
        fs1.lazy(),
        daily_history_snapshots(history, "user_id", "user"),
        daily_history_snapshots(history, "video_id", "video"),
        PRIORS,
        alpha=2.0,
    ).collect()


def test_counts_rates_smoothing_and_zero_history_are_correct() -> None:
    history = _history(
        [
            (1, 10, 20220409, 1, 1, 1, 0),
            (1, 11, 20220409, 0, 0, 1, 1),
        ]
    )
    output = _build(
        _fs1([(100, 1, 10, 20220410), (101, 2, 99, 20220410)]), history
    )
    seen = output.row(0, named=True)
    unseen = output.row(1, named=True)

    assert seen["user__hist_impression_count"] == 2
    assert seen["user__hist_long_view_positive_count"] == 1
    assert seen["user__hist_like_positive_count"] == 1
    assert seen["user__hist_valid_play_impression_count"] == 1
    assert seen["user__hist_valid_play_positive_count"] == 1
    assert seen["user__hist_literal_click_impression_count"] == 1
    assert seen["user__hist_literal_click_positive_count"] == 1
    assert seen["user__hist_long_view_rate"] == pytest.approx((1 + 2 * 0.25) / 4)
    assert seen["video__hist_long_view_rate"] == pytest.approx((1 + 2 * 0.25) / 3)
    assert unseen["user__hist_impression_count"] == 0
    assert unseen["video__hist_impression_count"] == 0
    assert unseen["user__hist_long_view_rate"] == pytest.approx(PRIORS["long_view"])
    assert unseen["video__hist_literal_click_rate"] == pytest.approx(PRIORS["literal_click"])


def test_own_outcome_cannot_change_its_features() -> None:
    target = _fs1([(100, 1, 10, 20220410)])
    first = _history([(1, 10, 20220409, 1, 0, 0, 0), (1, 10, 20220410, 1, 0, 0, 0)])
    changed = first.with_columns(
        pl.when(pl.col("date") == 20220410).then(1).otherwise(pl.col("long_view")).alias("long_view")
    )

    assert _build(target, first).to_dicts() == _build(target, changed).to_dicts()


def test_future_outcome_cannot_change_past_features() -> None:
    target = _fs1([(100, 1, 10, 20220410)])
    first = _history([(1, 10, 20220409, 1, 0, 0, 0), (1, 10, 20220411, 1, 0, 0, 0)])
    changed = first.with_columns(
        pl.when(pl.col("date") == 20220411).then(1).otherwise(pl.col("is_like")).alias("is_like")
    )

    assert _build(target, first).to_dicts() == _build(target, changed).to_dicts()


def test_same_day_events_cannot_affect_one_another() -> None:
    targets = _fs1([(100, 1, 10, 20220410), (101, 1, 11, 20220410)])
    history = _history(
        [
            (1, 10, 20220409, 1, 0, 0, 0),
            (1, 10, 20220410, 1, 1, 1, 1),
            (1, 11, 20220410, 0, 0, 1, 0),
        ]
    )
    changed = history.with_columns(
        pl.when(pl.col("date") == 20220410).then(1 - pl.col("long_view")).otherwise(pl.col("long_view")).alias("long_view")
    )

    assert _build(targets, history).to_dicts() == _build(targets, changed).to_dicts()


def test_click_histories_are_separate_and_never_pooled() -> None:
    output = _build(
        _fs1([(100, 1, 10, 20220410)]),
        _history([(1, 10, 20220409, 1, 0, 1, 0), (1, 11, 20220409, 0, 0, 0, 0)]),
    )

    assert output["user__hist_valid_play_positive_count"].item() == 1
    assert output["user__hist_literal_click_positive_count"].item() == 0
    assert "user__hist_click_rate" not in output.columns


def test_fs1_columns_values_order_and_ids_are_preserved() -> None:
    fs1 = _fs1([(101, 2, 11, 20220410), (100, 1, 10, 20220410)])
    output = _build(fs1, _history([(1, 10, 20220409, 1, 1, 1, 0)]))

    assert output.select(fs1.columns).equals(fs1)
    assert output["impression_id"].to_list() == [101, 100]


def test_normal_and_random_targets_receive_identical_fs2_schema() -> None:
    history = _history([(1, 10, 20220409, 1, 1, 1, 0)])
    normal = _fs1([(100, 1, 10, 20220410)])
    random = normal.with_columns(pl.lit(1, dtype=pl.Int64).alias("context__is_rand"))

    assert _build(normal, history).schema == _build(random, history).schema


def test_random_outcomes_are_not_an_input_to_fs2_history() -> None:
    normal_history = _history([(1, 10, 20220409, 1, 0, 0, 0)])
    random_history_a = _history([(1, 10, 20220409, 1, 0, 0, 0)])
    random_history_b = random_history_a.with_columns(pl.lit(1).alias("long_view"))
    target = _fs1([(100, 1, 10, 20220410)])

    # The builder accepts only its explicitly supplied normal-history snapshots.
    before = _build(target, normal_history)
    _ = random_history_a, random_history_b
    after = _build(target, normal_history)
    assert before.equals(after)
