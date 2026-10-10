from copy import deepcopy

import polars as pl
import pytest

from kuairand_pure.data.splits import (
    assert_exact_date_coverage,
    assert_nonoverlapping_layout,
    frozen_splits,
)


def test_frozen_split_boundaries_and_nonoverlap() -> None:
    splits = frozen_splits()

    assert_nonoverlapping_layout(splits)
    assert (splits["burn_in"].start, splits["burn_in"].end) == (20220409, 20220411)
    assert (splits["normal_train"].start, splits["normal_train"].end) == (20220412, 20220429)
    assert (splits["random_train"].start, splits["random_train"].end) == (20220422, 20220429)
    assert (splits["normal_validation"].start, splits["normal_validation"].end) == (20220430, 20220502)
    assert (splits["normal_test"].start, splits["normal_test"].end) == (20220503, 20220508)


def test_split_drift_is_rejected() -> None:
    splits = deepcopy(frozen_splits())
    splits["normal_train"] = type(splits["normal_train"])(
        "normal_train", "normal", 20220411, 20220429, True
    )

    with pytest.raises(ValueError, match="date drift"):
        assert_nonoverlapping_layout(splits)


def test_missing_calendar_day_is_rejected() -> None:
    spec = frozen_splits()["random_train"]

    with pytest.raises(ValueError, match="coverage drift"):
        assert_exact_date_coverage(pl.Series("date", [20220422, 20220429]), spec)
