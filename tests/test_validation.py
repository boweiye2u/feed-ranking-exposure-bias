from pathlib import Path

import numpy as np
import polars as pl
import pytest

from kuairand_pure.evaluation import (
    _evaluate_scores,
    _task_mask_and_label,
    load_normal_validation_history,
)


def _features() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "impression_id": pl.Series(range(1, 11), dtype=pl.UInt64),
            "user_id": [1] * 5 + [2] * 5,
            "video_id": list(range(10, 20)),
            "date": [20220430] * 10,
            "time_ms": list(range(1, 11)),
            "context__tab": [1, 0, 2, 1, 0, 1, 0, 2, 1, 0],
        }
    )


def _labels() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "impression_id": pl.Series(range(1, 11), dtype=pl.UInt64),
            "user_id": [1] * 5 + [2] * 5,
            "video_id": list(range(10, 20)),
            "date": [20220430] * 10,
            "time_ms": list(range(1, 11)),
            "click_semantics": [
                "valid_play", "literal_click", "unsupported_interface",
                "valid_play", "literal_click", "valid_play", "literal_click",
                "unsupported_interface", "valid_play", "literal_click",
            ],
            "long_view": [1, 0, 0, 0, 0, 0, 1, 0, 0, 0],
            "is_click": [1, 0, 0, 0, 1, 0, 1, 0, 1, 0],
            "is_like": [1, 0, 0, 0, 0, 0, 1, 0, 0, 0],
        }
    )


def test_validation_task_masks_never_pool_click_semantics() -> None:
    labels = _labels()
    assert _task_mask_and_label(labels, "long_view")[0].sum() == 10
    assert _task_mask_and_label(labels, "like")[0].sum() == 10
    assert _task_mask_and_label(labels, "valid_play")[0].tolist() == [True, False, False, True, False, True, False, False, True, False]
    assert _task_mask_and_label(labels, "literal_click")[0].tolist() == [False, True, False, False, True, False, True, False, False, True]
    with pytest.raises(ValueError, match="pooled"):
        _task_mask_and_label(labels, "is_click")


def test_validation_scoring_requires_exact_row_alignment() -> None:
    with pytest.raises(ValueError, match="canonical|differ by row"):
        _evaluate_scores(_features(), _labels().reverse(), np.linspace(0.1, 0.9, 10), "long_view")


def test_validation_scoring_preserves_task_ids_and_reports_denominators() -> None:
    predictions, metrics = _evaluate_scores(
        _features(), _labels(),
        np.asarray([0.9, 0.1, 0.2, 0.3, 0.4, 0.1, 0.9, 0.2, 0.3, 0.4]),
        "long_view",
    )

    assert predictions["impression_id"].to_list() == list(range(1, 11))
    assert metrics["impressions"] == 10
    assert metrics["positives"] == 2
    assert metrics["total_pairs"] == 8


def test_validation_history_is_normal_only_and_stops_before_may2(monkeypatch) -> None:
    columns = {
        "user_id": [1, 1], "video_id": [10, 10],
        "date": [20220409, 20220429], "tab": [1, 1],
        "long_view": [0, 1], "is_click": [0, 1], "is_like": [0, 0],
    }
    monkeypatch.setattr(
        "kuairand_pure.evaluation.load_normal_history",
        lambda data_dir, start, end: pl.DataFrame(columns),
    )
    authorization_calls = []
    monkeypatch.setattr(
        "kuairand_pure.evaluation.authorize_outcome_access",
        lambda protocol, **kwargs: authorization_calls.append(kwargs),
    )
    monkeypatch.setattr(
        "kuairand_pure.evaluation._paths",
        lambda data_dir: (
            {"source_codes": {"normal_early": 1, "normal_late": 2}},
            {"normal_early": Path("early"), "normal_late": Path("late")},
        ),
    )
    seen_specs = []

    def bounded(path, code, spec):
        seen_specs.append(spec)
        if code == 1:
            return pl.DataFrame(
                {
                    "user_id": [1], "video_id": [10], "date": [20220430], "tab": [1],
                    "long_view": [0], "is_click": [0], "is_like": [0],
                }
            ).lazy()
        return pl.DataFrame(
            {
                "user_id": [2], "video_id": [11], "date": [20220501], "tab": [0],
                "long_view": [1], "is_click": [1], "is_like": [0],
            }
        ).lazy()

    monkeypatch.setattr("kuairand_pure.evaluation._bounded_label_scan", bounded)
    history = load_normal_validation_history(Path("unused"))

    assert history["date"].to_list() == [20220409, 20220429, 20220430, 20220501]
    assert all(spec.source == "normal" and spec.end == 20220501 for spec in seen_specs)
    assert authorization_calls == [{
        "source": "normal", "start": 20220430, "end": 20220501,
        "workflow": "validation", "purpose": "selection",
    }]
