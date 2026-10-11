import json
from pathlib import Path

import numpy as np
import polars as pl
import pytest

from kuairand_pure.models.lightgbm import (
    LightGBMAdapter,
    lightgbm_parameters,
    load_lightgbm_config,
    load_model_bundle,
    predict_probabilities,
    save_model_bundle,
    select_task,
    train_booster,
)


def _field(name: str, present_in: list[str], dtype: str = "Int64") -> dict:
    return {
        "name": name,
        "data_type": dtype,
        "present_in": present_in,
        "model_role": "dense_model_feature",
        "allowed_model_families": ["lightgbm"],
    }


def _manifest(path: Path) -> Path:
    fields = [
        _field("impression_id", ["fs1", "fs2", "fs3"], "UInt64") | {"model_role": "alignment_only"},
        _field("context__tab", ["fs1", "fs2", "fs3"]),
        _field("context__is_rand", ["fs1", "fs2", "fs3"]),
        _field("user__onehot_feat0", ["fs1", "fs2", "fs3"]),
        _field("user__user_active_degree", ["fs1", "fs2", "fs3"], "String"),
        _field("video__tag", ["fs1", "fs2", "fs3"], "String"),
        _field("video__video_duration", ["fs1", "fs2", "fs3"], "Float64"),
        _field("video__hist_long_view_rate", ["fs2", "fs3"], "Float64"),
        _field("category__key", ["fs3"], "String"),
    ]
    path.write_text(json.dumps({"manifest_version": "m2-features-v1", "feature_fields": fields}))
    return path


def _features() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "impression_id": pl.Series(range(1, 9), dtype=pl.UInt64),
            "context__tab": [1, 0, 2, 1, 0, 1, 2, 1],
            "context__is_rand": [0] * 8,
            "user__onehot_feat0": [3, 2, 3, 1, 2, 1, 4, 4],
            "user__user_active_degree": ["a", "b", None, "a", "c", "b", "a", "c"],
            "video__tag": ["12,65", "12", None, "12,65", "2", "12", "2", "12,65"],
            "video__video_duration": [1.0, 2.0, None, 4.0, 5.0, 6.0, 7.0, 8.0],
            "video__hist_long_view_rate": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8],
            "category__key": ["12|65", "12", "__MISSING__", "12|65", "2", "12", "2", "12|65"],
        }
    )


def _labels() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "impression_id": pl.Series(range(1, 9), dtype=pl.UInt64),
            "long_view": [0, 1, 0, 1, 0, 1, 0, 1],
            "is_like": [0, 0, 0, 1, 0, 0, 1, 0],
            "is_click": [1, 1, 0, 0, 1, 1, 0, 1],
            "click_semantics": [
                "valid_play", "literal_click", "unsupported_interface", "valid_play",
                "literal_click", "valid_play", "unsupported_interface", "valid_play",
            ],
        }
    )


def test_categorical_codes_strings_missing_and_unseen_are_deterministic(tmp_path: Path) -> None:
    adapter = LightGBMAdapter.fit(
        _features(), feature_set="fs3", feature_manifest_path=_manifest(tmp_path / "manifest.json")
    )
    transformed = adapter.transform(_features())

    assert "context__is_rand" not in adapter.feature_names
    assert "user__onehot_feat0" in adapter.categorical_features
    assert "video__video_duration" in adapter.numerical_features
    assert "category__key" in adapter.categorical_features
    assert transformed["user__onehot_feat0"].dtype.name == "category"
    assert transformed["video__video_duration"].isna().sum() == 1
    assert transformed["user__user_active_degree"].cat.codes.iloc[2] == 0

    changed = _features().with_columns(
        pl.when(pl.col("impression_id") == 1)
        .then(pl.lit("never-seen"))
        .otherwise(pl.col("category__key"))
        .alias("category__key")
    )
    assert adapter.transform(changed)["category__key"].cat.codes.iloc[0] == 1
    assert adapter.transform(changed).equals(adapter.transform(changed))


@pytest.mark.parametrize(("feature_set", "expected"), [("fs1", 5), ("fs2", 6), ("fs3", 7)])
def test_adapter_supports_nested_feature_sets(tmp_path: Path, feature_set: str, expected: int) -> None:
    adapter = LightGBMAdapter.fit(
        _features(), feature_set=feature_set, feature_manifest_path=_manifest(tmp_path / "manifest.json")
    )
    assert len(adapter.feature_names) == expected


def test_complete_tag_set_is_one_feature_and_not_expanded(tmp_path: Path) -> None:
    adapter = LightGBMAdapter.fit(
        _features(), feature_set="fs3", feature_manifest_path=_manifest(tmp_path / "manifest.json")
    )
    assert adapter.feature_names.count("category__key") == 1
    assert not any(name.startswith("category__key=") for name in adapter.feature_names)


def test_task_masks_are_interface_specific_and_pooled_click_is_rejected(tmp_path: Path) -> None:
    features, labels = _features(), _labels()
    adapter = LightGBMAdapter.fit(
        features, feature_set="fs1", feature_manifest_path=_manifest(tmp_path / "manifest.json")
    )
    encoded = adapter.transform(features)

    assert select_task(encoded, features, labels, "long_view").rows == 8
    assert select_task(encoded, features, labels, "like").rows == 8
    assert select_task(encoded, features, labels, "valid_play").rows == 4
    assert select_task(encoded, features, labels, "literal_click").rows == 2
    with pytest.raises(ValueError, match="Pooled is_click is forbidden"):
        select_task(encoded, features, labels, "is_click")


def test_task_selection_rejects_row_misalignment(tmp_path: Path) -> None:
    features, labels = _features(), _labels().reverse()
    adapter = LightGBMAdapter.fit(
        features, feature_set="fs1", feature_manifest_path=_manifest(tmp_path / "manifest.json")
    )
    with pytest.raises(ValueError, match="differ by row"):
        select_task(adapter.transform(features), features, labels, "long_view")


def test_fixed_complexities_only_change_tree_capacity() -> None:
    config = load_lightgbm_config()
    shallow, shallow_rounds = lightgbm_parameters(config, "shallow")
    deep, deep_rounds = lightgbm_parameters(config, "deep")
    differing = {key for key in shallow if shallow[key] != deep[key]}

    assert differing == {"num_leaves", "max_depth"}
    assert shallow_rounds == deep_rounds == 300
    assert shallow["seed"] == deep["seed"] == 202211


def test_training_prediction_serialization_and_reproducibility(tmp_path: Path) -> None:
    features, labels = _features(), _labels()
    manifest_path = _manifest(tmp_path / "manifest.json")
    adapter = LightGBMAdapter.fit(features, feature_set="fs1", feature_manifest_path=manifest_path)
    dataset = select_task(adapter.transform(features), features, labels, "long_view")

    first = train_booster(dataset, adapter, complexity="shallow", num_boost_round=5)
    second = train_booster(dataset, adapter, complexity="shallow", num_boost_round=5)
    first_predictions = predict_probabilities(first, adapter, features)
    second_predictions = predict_probabilities(second, adapter, features)

    assert np.array_equal(first_predictions, second_predictions)
    assert ((first_predictions >= 0) & (first_predictions <= 1)).all()

    bundle = tmp_path / "bundle"
    save_model_bundle(bundle, first, adapter, task="long_view", complexity="shallow")
    reloaded, reloaded_adapter, metadata = load_model_bundle(
        bundle, expected_manifest_sha256=adapter.feature_manifest_sha256
    )
    assert metadata["task"] == "long_view"
    assert np.array_equal(
        first_predictions, predict_probabilities(reloaded, reloaded_adapter, features)
    )
    with pytest.raises(ValueError, match="incompatible"):
        load_model_bundle(bundle, expected_manifest_sha256="wrong")


def test_transform_rejects_missing_model_feature(tmp_path: Path) -> None:
    adapter = LightGBMAdapter.fit(
        _features(), feature_set="fs1", feature_manifest_path=_manifest(tmp_path / "manifest.json")
    )
    with pytest.raises(ValueError, match="absent"):
        adapter.transform(_features().drop("video__tag"))
    with pytest.raises(ValueError, match="dtypes drifted"):
        adapter.transform(_features().with_columns(pl.col("video__video_duration").cast(pl.String)))


def test_model_loader_rejects_incomplete_and_stale_bundles(tmp_path: Path) -> None:
    features, labels = _features(), _labels()
    manifest_path = _manifest(tmp_path / "manifest.json")
    adapter = LightGBMAdapter.fit(features, feature_set="fs1", feature_manifest_path=manifest_path)
    dataset = select_task(adapter.transform(features), features, labels, "long_view")
    booster = train_booster(dataset, adapter, complexity="shallow", num_boost_round=2)

    incomplete = tmp_path / "incomplete"
    incomplete.mkdir()
    booster.save_model(incomplete / "model.txt")
    with pytest.raises(ValueError, match="Incomplete"):
        load_model_bundle(incomplete, expected_manifest_sha256=adapter.feature_manifest_sha256)

    bundle = tmp_path / "bundle"
    save_model_bundle(bundle, booster, adapter, task="long_view", complexity="shallow")
    with pytest.raises(ValueError, match="feature_set is incompatible"):
        load_model_bundle(
            bundle,
            expected_manifest_sha256=adapter.feature_manifest_sha256,
            expected_feature_set="fs2",
        )
    with pytest.raises(ValueError, match="task is incompatible"):
        load_model_bundle(
            bundle,
            expected_manifest_sha256=adapter.feature_manifest_sha256,
            expected_task="like",
        )
    with pytest.raises(ValueError, match="complexity is incompatible"):
        load_model_bundle(
            bundle,
            expected_manifest_sha256=adapter.feature_manifest_sha256,
            expected_complexity="deep",
        )
