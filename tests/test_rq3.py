import json

import numpy as np
import polars as pl
import pytest

from kuairand_pure.data.alignment import validate_row_alignment
from kuairand_pure.models.lightgbm import LightGBMAdapter
from kuairand_pure.rq3 import (
    MarginalExposureWeighter,
    _validate_source,
    load_rq3_config,
    validate_rq3_config,
    weight_diagnostics,
)


def test_marginal_weights_follow_approved_rule_and_normalize() -> None:
    videos = np.array([10, 10, 10, 20, 20, 30], dtype=np.int64)
    weighter = MarginalExposureWeighter.fit(videos, alpha=1.0, clip_quantile=0.99)
    raw = weighter.raw_weights(videos)
    assert np.array_equal(raw, np.array([0.25, 0.25, 0.25, 1 / 3, 1 / 3, 0.5]))
    weights = weighter.transform(videos)
    assert weights.mean() == pytest.approx(1.0)
    diagnostics = weight_diagnostics(weights)
    assert diagnostics["effective_sample_size"] == pytest.approx(
        weights.sum() ** 2 / np.square(weights).sum()
    )


def test_unseen_video_uses_frozen_clipped_maximum() -> None:
    videos = np.array([1, 1, 2, 3, 4], dtype=np.int64)
    weighter = MarginalExposureWeighter.fit(videos)
    unseen = weighter.transform(np.array([999], dtype=np.int64))[0]
    assert unseen == pytest.approx(weighter.unseen_normalized_weight)
    assert unseen == pytest.approx(weighter.transform(np.array([3]))[0])


def _manifest(path):
    content = {
        "manifest_version": "m2-features-v1",
        "feature_fields": [
            {
                "name": "context__tab", "data_type": "Int64",
                "present_in": ["fs1"], "model_role": "dense_model_feature",
                "allowed_model_families": ["lightgbm"],
            },
            {
                "name": "context__is_rand", "data_type": "Int64",
                "present_in": ["fs1"], "model_role": "rq3_joint_only",
                "allowed_model_families": ["lightgbm"],
            },
        ],
    }
    path.write_text(json.dumps(content))
    return path


def test_source_indicator_is_joint_only_and_encoding_is_deterministic(tmp_path) -> None:
    features = pl.DataFrame({
        "impression_id": [1, 2, 3],
        "context__tab": [0, 1, 1],
        "context__is_rand": [0, 1, 1],
    }, schema_overrides={"impression_id": pl.UInt64})
    manifest = _manifest(tmp_path / "manifest.json")
    baseline = LightGBMAdapter.fit(features, feature_set="fs1", feature_manifest_path=manifest)
    joint = LightGBMAdapter.fit(
        features, feature_set="fs1", feature_manifest_path=manifest, include_is_rand=True
    )
    assert "context__is_rand" not in baseline.feature_names
    assert "context__is_rand" in joint.feature_names
    assert joint.transform(features).equals(joint.transform(features))
    joint_path = tmp_path / "joint.json"
    joint.save(joint_path)
    loaded = LightGBMAdapter.load(
        joint_path, expected_manifest_sha256=joint.feature_manifest_sha256
    )
    assert loaded.include_is_rand is True
    assert loaded.feature_names == joint.feature_names


def test_alignment_rejects_duplicate_and_reordered_ids() -> None:
    features = pl.DataFrame({"impression_id": [1, 2]}, schema_overrides={"impression_id": pl.UInt64})
    reordered = pl.DataFrame({"impression_id": [2, 1]}, schema_overrides={"impression_id": pl.UInt64})
    duplicate = pl.DataFrame({"impression_id": [1, 1]}, schema_overrides={"impression_id": pl.UInt64})
    with pytest.raises(ValueError, match="canonical"):
        validate_row_alignment(features, reordered)
    with pytest.raises(ValueError, match="duplicate"):
        validate_row_alignment(duplicate, duplicate)


def test_rq3_config_matches_frozen_baseline_hyperparameters() -> None:
    config = load_rq3_config()
    assert config["lightgbm"]["num_boost_round"] == 300
    assert config["lightgbm"]["num_leaves"] == 127
    drifted = {**config, "lightgbm": {**config["lightgbm"], "num_boost_round": 301}}
    with pytest.raises(ValueError, match="differ"):
        validate_rq3_config(drifted)


def test_task_adapter_rejects_bad_sample_weights(tmp_path) -> None:
    from kuairand_pure.models.lightgbm import select_task

    features = pl.DataFrame({
        "impression_id": [1, 2], "context__tab": [0, 1], "context__is_rand": [0, 0],
    }, schema_overrides={"impression_id": pl.UInt64})
    labels = pl.DataFrame({
        "impression_id": [1, 2], "long_view": [0, 1], "is_like": [0, 0],
        "is_click": [0, 1], "click_semantics": ["literal_click", "valid_play"],
    }, schema_overrides={"impression_id": pl.UInt64})
    adapter = LightGBMAdapter.fit(
        features, feature_set="fs1", feature_manifest_path=_manifest(tmp_path / "manifest.json")
    )
    encoded = adapter.transform(features)
    task = select_task(encoded, features, labels, "long_view", np.array([0.5, 1.5]))
    assert np.array_equal(task.sample_weight, np.array([0.5, 1.5]))
    with pytest.raises(ValueError, match="positive, finite"):
        select_task(encoded, features, labels, "long_view", np.array([1.0, 0.0]))


def test_source_validation_enforces_dates_indicator_and_alignment() -> None:
    features = pl.DataFrame({
        "impression_id": [1, 2], "date": [20220422, 20220429],
        "context__is_rand": [1, 1],
    }, schema_overrides={"impression_id": pl.UInt64})
    labels = pl.DataFrame({
        "impression_id": [1, 2], "date": [20220422, 20220429],
    }, schema_overrides={"impression_id": pl.UInt64})
    _validate_source(features, labels, dates=(20220422, 20220429), is_rand=1)
    with pytest.raises(ValueError, match="date boundary"):
        _validate_source(features, labels, dates=(20220421, 20220429), is_rand=1)
    with pytest.raises(ValueError, match="source indicator"):
        _validate_source(features, labels, dates=(20220422, 20220429), is_rand=0)
    with pytest.raises(ValueError, match="canonical"):
        _validate_source(features, labels.reverse(), dates=(20220422, 20220429), is_rand=1)
