import numpy as np
import polars as pl
import pytest

from kuairand_pure.final_predictions import (
    _eligible, _feature_manifest, _write_prediction, authorize_final_normal_feature_history,
)


def test_final_history_authorization_is_narrow() -> None:
    authorize_final_normal_feature_history(source="normal", start=20220503, end=20220507,
                                           purpose="strictly_prior_day_feature_history")
    for kwargs in (
        {"source": "random", "start": 20220503, "end": 20220507, "purpose": "strictly_prior_day_feature_history"},
        {"source": "normal", "start": 20220503, "end": 20220508, "purpose": "strictly_prior_day_feature_history"},
        {"source": "normal", "start": 20220503, "end": 20220507, "purpose": "evaluation"},
    ):
        with pytest.raises(PermissionError): authorize_final_normal_feature_history(**kwargs)


def test_task_eligibility_uses_context_without_labels() -> None:
    frame = pl.DataFrame({"context__tab": [0, 1, 2]})
    assert _eligible(frame, "long_view").tolist() == [True, True, True]
    assert _eligible(frame, "like").tolist() == [True, True, True]
    assert _eligible(frame, "valid_play").tolist() == [False, True, False]
    assert _eligible(frame, "literal_click").tolist() == [True, False, False]


def test_feature_only_manifest_rejects_outcomes(tmp_path) -> None:
    directory = tmp_path / "fs1"; directory.mkdir()
    path = directory / "features.parquet"
    pl.DataFrame({"impression_id": [1], "long_view": [0]}).write_parquet(path)
    with pytest.raises(ValueError, match="Outcome columns"):
        _feature_manifest(directory, stage="fs1", split="T-normal", feature_path=path)


def test_prediction_writer_rejects_invalid_scores(tmp_path) -> None:
    with pytest.raises(ValueError, match="probabilities"):
        _write_prediction(tmp_path, pl.Series("impression_id", [1], dtype=pl.UInt64),
                          np.array([1.1]), {"source": "normal"})


def test_prediction_writer_preserves_canonical_ids_and_has_no_labels(tmp_path) -> None:
    result = _write_prediction(
        tmp_path, pl.Series("impression_id", [2, 3], dtype=pl.UInt64),
        np.array([0.2, 0.8]), {"source": "normal", "task": "long_view"},
    )
    frame = pl.read_parquet(result["path"])
    assert frame.columns == ["impression_id", "score"]
    assert frame["impression_id"].to_list() == [2, 3]
