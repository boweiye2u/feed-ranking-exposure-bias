import copy
import json

import numpy as np
import polars as pl
import pytest

from kuairand_pure.models.lightgbm import (
    LightGBMAdapter,
    load_model_bundle,
    save_model_bundle,
    select_task,
    train_booster,
)
from kuairand_pure.rq3_training import validate_training_record


def _feature_manifest(path):
    path.write_text(json.dumps({
        "manifest_version": "m2-features-v1",
        "feature_fields": [
            {
                "name": "context__tab", "data_type": "Int64",
                "present_in": ["fs1"], "model_role": "dense_model_feature",
                "allowed_model_families": ["lightgbm"],
            }
        ],
    }))
    return path


def _config():
    return {
        "lightgbm": {
            "objective": "binary", "learning_rate": 0.05,
            "num_boost_round": 3, "min_data_in_leaf": 1,
            "lambda_l1": 0.0, "lambda_l2": 1.0,
            "feature_fraction": 1.0, "bagging_fraction": 1.0,
            "bagging_freq": 0, "max_bin": 255, "num_threads": 1,
            "deterministic": True, "force_col_wise": True,
            "verbosity": -1, "seed": 202211,
        },
        "complexity": {"deep": {"num_leaves": 7, "max_depth": 3}},
    }


def test_weighted_lightgbm_training_and_audited_adapter_reload(tmp_path) -> None:
    rows = 200
    features = pl.DataFrame({
        "impression_id": np.arange(rows, dtype=np.uint64),
        "context__tab": np.arange(rows) % 2,
    })
    labels = pl.DataFrame({
        "impression_id": np.arange(rows, dtype=np.uint64),
        "long_view": (np.arange(rows) % 3 == 0).astype(np.int64),
        "is_like": np.zeros(rows, dtype=np.int64),
        "is_click": np.arange(rows) % 2,
        "click_semantics": ["valid_play"] * rows,
    })
    manifest_path = _feature_manifest(tmp_path / "features.json")
    adapter = LightGBMAdapter.fit(
        features, feature_set="fs1", feature_manifest_path=manifest_path
    )
    audited_adapter = tmp_path / "audited_adapter.json"
    adapter.save(audited_adapter)
    weights = np.linspace(0.5, 1.5, rows)
    task = select_task(
        adapter.transform(features), features, labels, "long_view", weights
    )
    booster = train_booster(task, adapter, complexity="deep", config=_config())
    assert booster.current_iteration() == 3
    bundle = tmp_path / "bundle"
    save_model_bundle(
        bundle, booster, adapter, task="long_view", complexity="deep",
        adapter_source_path=audited_adapter,
    )
    reloaded, loaded_adapter, _ = load_model_bundle(
        bundle, expected_manifest_sha256=adapter.feature_manifest_sha256,
        expected_feature_set="fs1", expected_task="long_view", expected_complexity="deep",
    )
    assert (bundle / "adapter.json").read_bytes() == audited_adapter.read_bytes()
    assert np.array_equal(booster.predict(task.X), reloaded.predict(task.X))
    assert loaded_adapter.include_is_rand is False


def _record(arm):
    rows = {"marginal_reweighting": 722481, "joint": 1060414, "random_only": 337933}[arm]
    positives = {"marginal_reweighting": 239795, "joint": 266992, "random_only": 27197}[arm]
    return {
        "arm": arm, "training_rows": rows, "positive_labels": positives,
        "context_is_rand_included": arm == "joint",
        "sample_weight_policy": "approved_marginal_exposure" if arm == "marginal_reweighting" else "uniform",
        "boosting_rounds_completed": 300, "serialization_reload_exact": True,
        "baseline_byte_identical": True,
        "randomized_validation_outcomes_accessed": False,
        "final_test_outcomes_accessed": False,
    }


def test_training_record_rejects_wrong_source_weight_and_embargo_state() -> None:
    for arm in ("marginal_reweighting", "joint", "random_only"):
        validate_training_record(_record(arm), arm)
    bad_source = copy.deepcopy(_record("random_only"))
    bad_source["context_is_rand_included"] = True
    with pytest.raises(ValueError, match="source-indicator"):
        validate_training_record(bad_source, "random_only")
    bad_weight = copy.deepcopy(_record("joint"))
    bad_weight["sample_weight_policy"] = "approved_marginal_exposure"
    with pytest.raises(ValueError, match="sample-weight"):
        validate_training_record(bad_weight, "joint")
    embargo = copy.deepcopy(_record("joint"))
    embargo["final_test_outcomes_accessed"] = True
    with pytest.raises(ValueError, match="embargo"):
        validate_training_record(embargo, "joint")
