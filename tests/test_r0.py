import json
from pathlib import Path

import polars as pl
import pytest

from kuairand_pure.baselines.r0 import (
    R0_INPUT_COLUMNS,
    evaluate_aligned_r0_predictions,
    score_r0,
    verify_r0_contract,
)
from kuairand_pure.data.alignment import (
    load_aligned_feature_columns,
    write_artifact_manifest,
)
from kuairand_pure.features.fs2 import build_fs2_lazy, daily_history_snapshots

PRIOR = 0.3364927893253054
PRIORS = {
    "long_view": PRIOR,
    "like": 0.02,
    "valid_play": 0.5,
    "literal_click": 0.1,
}


def _r0_frame(ids=(1, 2), scores=(0.2, 0.8)) -> pl.DataFrame:
    return pl.DataFrame(
        {
            "impression_id": pl.Series(ids, dtype=pl.UInt64),
            "video__hist_long_view_rate": pl.Series(scores, dtype=pl.Float64),
        }
    )


def _history() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "user_id": [1, 1],
            "video_id": [10, 10],
            "date": [20220409, 20220410],
            "tab": [1, 1],
            "long_view": [0, 1],
            "is_click": [1, 1],
            "is_like": [0, 0],
        }
    )


def test_r0_score_is_deterministic_and_preserves_ids() -> None:
    inputs = _r0_frame()

    first = score_r0(inputs)
    second = score_r0(inputs)

    assert first.equals(second)
    assert first["impression_id"].equals(inputs["impression_id"])
    assert first["r0_score"].to_list() == [0.2, 0.8]


def test_r0_unseen_video_uses_frozen_prior() -> None:
    output = score_r0(_r0_frame(ids=(1,), scores=(PRIOR,)))

    assert output["r0_score"].item() == pytest.approx(PRIOR)


def test_r0_chronology_and_smoothing_are_inherited_from_fs2() -> None:
    history = _history()
    base = pl.DataFrame(
        {
            "impression_id": pl.Series([1, 2, 3], dtype=pl.UInt64),
            "user_id": [1, 1, 2],
            "video_id": [10, 10, 99],
            "date": [20220410, 20220411, 20220410],
        }
    ).lazy()
    fs2 = build_fs2_lazy(
        base,
        daily_history_snapshots(history, "user_id", "user"),
        daily_history_snapshots(history, "video_id", "video"),
        PRIORS,
        alpha=20.0,
    ).collect()
    output = score_r0(fs2.select(*R0_INPUT_COLUMNS))

    assert output["r0_score"].to_list() == pytest.approx(
        [
            (0 + 20 * PRIOR) / 21,
            (1 + 20 * PRIOR) / 22,
            PRIOR,
        ]
    )


def test_r0_rejects_missing_out_of_range_and_extra_inputs() -> None:
    with pytest.raises(ValueError, match="missing"):
        score_r0(_r0_frame(ids=(1,), scores=(None,)))
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        score_r0(_r0_frame(ids=(1,), scores=(1.1,)))
    with pytest.raises(ValueError, match="accepts exactly"):
        score_r0(_r0_frame().with_columns(pl.lit(1).alias("long_view")))
    with pytest.raises(ValueError, match="accepts exactly"):
        score_r0(_r0_frame().with_columns(pl.lit(4).alias("user_id")))


def test_r0_rejects_duplicate_or_missing_impression_ids() -> None:
    with pytest.raises(ValueError, match="duplicate"):
        score_r0(_r0_frame(ids=(1, 1)))
    with pytest.raises(ValueError, match="missing"):
        score_r0(_r0_frame(ids=(1, None)))


def test_feature_only_loader_checks_alignment_without_projecting_outcomes(tmp_path: Path) -> None:
    features_path = tmp_path / "features.parquet"
    labels_path = tmp_path / "labels.parquet"
    features = _r0_frame()
    labels = pl.DataFrame(
        {
            "impression_id": pl.Series([1, 2], dtype=pl.UInt64),
            "long_view": [1, 0],
        }
    )
    features.write_parquet(features_path)
    labels.write_parquet(labels_path)
    write_artifact_manifest(
        tmp_path,
        stage="fs2",
        split="normal_train",
        features_path=features_path,
        labels_path=labels_path,
    )

    loaded = load_aligned_feature_columns(
        features_path,
        labels_path,
        required_columns=R0_INPUT_COLUMNS,
        expected_stage="fs2",
        expected_split="normal_train",
    )

    assert loaded.columns == list(R0_INPUT_COLUMNS)
    assert "long_view" not in loaded.columns

    labels.reverse().write_parquet(labels_path)
    with pytest.raises(ValueError, match="canonical|differ by row"):
        load_aligned_feature_columns(
            features_path,
            labels_path,
            required_columns=R0_INPUT_COLUMNS,
            expected_stage="fs2",
            expected_split="normal_train",
        )


def test_r0_contract_rejects_stale_manifest(tmp_path: Path) -> None:
    fs2_path = tmp_path / "fs2.toml"
    fs2_path.write_text(
        "[fs2]\nalpha=20.0\nhistory_source='normal_only'\nstrictly_prior_calendar_days=true\n"
        "[frozen_priors]\nlong_view=0.3364927893253054\n"
    )
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "manifest_version": "m2-features-v1",
                "feature_fields": [
                    {
                        "name": "video__hist_long_view_rate",
                        "data_type": "Float64",
                        "introduced_in": "fs1",
                        "source": "static",
                    }
                ]
            }
        )
    )
    config = {
        "r0": {
            "input_build_version": "m2-features-v1",
            "input_stage": "fs2",
            "score_feature": "video__hist_long_view_rate",
        },
        "history_contract": {
            "alpha": 20.0,
            "long_view_prior": PRIOR,
            "source": "normal_only",
            "cutoff": "strictly_prior_calendar_day",
        },
        "input": {"scoring_columns": list(R0_INPUT_COLUMNS)},
    }

    with pytest.raises(ValueError, match="incompatible M2 manifest"):
        verify_r0_contract(config, fs2_path, manifest_path)

    stale = json.loads(manifest_path.read_text())
    stale["manifest_version"] = "m2-stale-v0"
    manifest_path.write_text(json.dumps(stale))
    with pytest.raises(ValueError, match="stale build version"):
        verify_r0_contract(config, fs2_path, manifest_path)


def test_synthetic_evaluation_adapter_uses_frozen_metric() -> None:
    predictions = pl.DataFrame(
        {
            "impression_id": pl.Series([1, 2, 3, 4], dtype=pl.UInt64),
            "r0_score": [0.9, 0.1, 0.4, 0.4],
        }
    )
    evaluation = pl.DataFrame(
        {
            "impression_id": pl.Series([1, 2, 3, 4], dtype=pl.UInt64),
            "user_id": [1, 1, 2, 2],
            "long_view": [1, 0, 1, 0],
        }
    )

    result = evaluate_aligned_r0_predictions(predictions, evaluation)

    assert result.pair_weighted_auc == pytest.approx(0.75)
    assert result.total_pairs == 2
