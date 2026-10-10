from pathlib import Path

import polars as pl
import pytest

from kuairand_pure.data.fs1 import (
    LABEL_COLUMNS,
    _bounded_label_scan,
    build_fs1_lazy,
    model_feature_columns,
)
from kuairand_pure.data.splits import SplitSpec


def _impressions(source: int = 0) -> pl.LazyFrame:
    return pl.DataFrame(
        {
            "impression_id": [1, 2],
            "user_id": [10, 999],
            "video_id": [20, 888],
            "date": [20220412, 20220412],
            "time_ms": [1, 2],
            "tab": [1, 0],
            "is_rand": [source, source],
        }
    ).lazy()


def _users() -> pl.LazyFrame:
    return pl.DataFrame(
        {"user_id": [10], "user__category": [None], "_user_found": [True]},
        schema={"user_id": pl.Int64, "user__category": pl.String, "_user_found": pl.Boolean},
    ).lazy()


def _videos() -> pl.LazyFrame:
    return pl.DataFrame(
        {"video_id": [20], "video__duration": [3.0], "_video_found": [True]}
    ).lazy()


def test_fs1_left_joins_without_expansion_and_preserves_missing_values() -> None:
    output = build_fs1_lazy(
        _impressions(), _users(), _videos(), ["user__category"], ["video__duration"]
    ).collect()

    assert output.height == 2
    assert output["impression_id"].n_unique() == 2
    assert output["user__category"].null_count() == 2
    assert output["user__static_missing"].to_list() == [False, True]
    assert output["video__static_missing"].to_list() == [False, True]
    assert not set(LABEL_COLUMNS).intersection(output.columns)


def test_outcome_column_on_input_cannot_enter_fs1() -> None:
    impressions = _impressions().with_columns(pl.lit(1).alias("long_view"))

    output = build_fs1_lazy(
        impressions, _users(), _videos(), ["user__category"], ["video__duration"]
    ).collect()

    assert "long_view" not in output.columns


def test_duplicate_static_keys_are_rejected_by_many_to_one_join() -> None:
    duplicate_users = pl.DataFrame(
        {"user_id": [10, 10], "user__category": ["a", "b"], "_user_found": [True, True]}
    ).lazy()

    with pytest.raises(Exception, match="did not fulfill|validation"):
        build_fs1_lazy(
            _impressions(), duplicate_users, _videos(), ["user__category"], ["video__duration"]
        ).collect()


def test_training_sources_have_identical_fs1_schema() -> None:
    normal = build_fs1_lazy(
        _impressions(0), _users(), _videos(), ["user__category"], ["video__duration"]
    ).collect_schema()
    random = build_fs1_lazy(
        _impressions(1), _users(), _videos(), ["user__category"], ["video__duration"]
    ).collect_schema()

    assert normal == random


def test_fs1_join_preserves_left_order_before_canonical_sort() -> None:
    impressions = pl.DataFrame(
        {
            "impression_id": [3, 1, 2],
            "user_id": [10, 999, 10],
            "video_id": [20, 888, 20],
            "date": [20220412] * 3,
            "time_ms": [3, 1, 2],
            "tab": [1, 0, 1],
            "is_rand": [0, 0, 0],
        }
    ).lazy()

    output = build_fs1_lazy(
        impressions, _users(), _videos(), ["user__category"], ["video__duration"]
    ).collect()

    assert output["impression_id"].to_list() == [3, 1, 2]


def test_exposure_indicator_is_available_only_to_rq3_joint() -> None:
    names = build_fs1_lazy(
        _impressions(), _users(), _videos(), ["user__category"], ["video__duration"]
    ).collect_schema().names()

    assert "context__is_rand" not in model_feature_columns(names, "baseline")
    assert "context__is_rand" not in model_feature_columns(names, "rq3_random_only")
    assert "context__is_rand" in model_feature_columns(names, "rq3_joint")
    assert "context__tab" in model_feature_columns(names, "baseline")


def test_unknown_fs1_workflow_is_rejected() -> None:
    with pytest.raises(ValueError, match="Unknown FS1 workflow"):
        model_feature_columns(["context__is_rand"], "ad_hoc")


def test_bounded_label_reader_does_not_parse_held_out_outcomes(tmp_path: Path) -> None:
    path = tmp_path / "log.csv"
    path.write_text(
        "user_id,video_id,date,time_ms,tab,long_view,is_click,is_like\n"
        "1,10,20220422,1,1,1,1,0\n"
        "2,11,20220503,2,1,DO_NOT_READ,DO_NOT_READ,DO_NOT_READ\n"
    )
    spec = SplitSpec("random_train", "random", 20220422, 20220422, True)

    labels = _bounded_label_scan(path, 3, spec).collect()

    assert labels.height == 1
    assert labels.select(*LABEL_COLUMNS).row(0) == (1, 1, 0)
