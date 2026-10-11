"""Leakage-safe LightGBM adapters and deterministic training utilities."""

from __future__ import annotations

import hashlib
import json
import math
import shutil
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import lightgbm as lgb
import numpy as np
import pandas as pd
import polars as pl

from kuairand_pure.data.alignment import M2_BUILD_VERSION, load_aligned_model_data

FEATURE_SETS = ("fs1", "fs2", "fs3")
TASKS = ("long_view", "like", "valid_play", "literal_click")
MISSING_CODE = 0
UNKNOWN_CODE = 1
KNOWN_OFFSET = 2


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json_sha256(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def load_lightgbm_config(path: Path | None = None) -> dict[str, Any]:
    with (path or _repo_root() / "configs" / "lightgbm.toml").open("rb") as stream:
        return tomllib.load(stream)


def _token(value: Any) -> str | None:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return None
    if isinstance(value, (np.integer, int)):
        return "i:" + str(int(value))
    if isinstance(value, (np.floating, float)):
        return "f:" + repr(float(value))
    if isinstance(value, (bool, np.bool_)):
        return "b:" + ("1" if value else "0")
    return "s:" + str(value)


def _categorical_feature_names() -> set[str]:
    with (_repo_root() / "configs" / "fs1.toml").open("rb") as stream:
        config = tomllib.load(stream)["categorical"]
    names = {
        *("user__" + name for name in config["user"]),
        *("video__" + name for name in config["video"]),
        *config["context"],
        "category__key",
    }
    return names


@dataclass(frozen=True, slots=True)
class TaskDataset:
    X: pd.DataFrame
    y: np.ndarray
    impression_ids: np.ndarray
    task: str
    rows: int
    sample_weight: np.ndarray | None = None


@dataclass(slots=True)
class LightGBMAdapter:
    feature_set: str
    feature_manifest_sha256: str
    feature_names: list[str]
    feature_dtypes: dict[str, str]
    categorical_features: list[str]
    numerical_features: list[str]
    vocabularies: dict[str, dict[str, int]]
    include_is_rand: bool = False

    @classmethod
    def fit(
        cls,
        features: pl.DataFrame,
        *,
        feature_set: str,
        feature_manifest_path: Path,
        include_is_rand: bool = False,
    ) -> "LightGBMAdapter":
        if feature_set not in FEATURE_SETS:
            raise ValueError(f"Unknown feature set: {feature_set}")
        manifest = json.loads(feature_manifest_path.read_text())
        if manifest.get("manifest_version") != M2_BUILD_VERSION:
            raise ValueError("Feature manifest build version is stale")
        fields = manifest["feature_fields"]
        feature_names = [
            item["name"] for item in fields
            if feature_set in item["present_in"]
            and (
                item["model_role"] == "dense_model_feature"
                or (
                    include_is_rand
                    and item["name"] == "context__is_rand"
                    and item["model_role"] == "rq3_joint_only"
                )
            )
            and "lightgbm" in item["allowed_model_families"]
            and (include_is_rand or item["name"] != "context__is_rand")
        ]
        if include_is_rand and "context__is_rand" not in feature_names:
            raise ValueError("RQ3 joint adapter did not include context__is_rand")
        if not feature_names or any(name not in features.columns for name in feature_names):
            raise ValueError("Training features do not match the frozen manifest")
        expected_dtypes = {item["name"]: item["data_type"] for item in fields}
        mismatches = {
            name: (expected_dtypes[name], str(features.schema[name]))
            for name in feature_names
            if str(features.schema[name]) != expected_dtypes[name]
        }
        if mismatches:
            raise ValueError(f"Training feature dtypes differ from the manifest: {mismatches}")
        categorical_declared = _categorical_feature_names()
        categorical = [name for name in feature_names if name in categorical_declared]
        numerical = [name for name in feature_names if name not in categorical_declared]
        vocabularies: dict[str, dict[str, int]] = {}
        for name in categorical:
            tokens = {_token(value) for value in features[name].to_list()}
            known = sorted(token for token in tokens if token is not None)
            vocabularies[name] = {
                token: index + KNOWN_OFFSET for index, token in enumerate(known)
            }
        return cls(
            feature_set=feature_set,
            feature_manifest_sha256=_sha256(feature_manifest_path),
            feature_names=feature_names,
            feature_dtypes={name: expected_dtypes[name] for name in feature_names},
            categorical_features=categorical,
            numerical_features=numerical,
            vocabularies=vocabularies,
            include_is_rand=include_is_rand,
        )

    def transform(self, features: pl.DataFrame) -> pd.DataFrame:
        missing = set(self.feature_names).difference(features.columns)
        if missing:
            raise ValueError(f"Model features are absent: {sorted(missing)}")
        mismatches = {
            name: (expected, str(features.schema[name]))
            for name, expected in self.feature_dtypes.items()
            if str(features.schema[name]) != expected
        }
        if mismatches:
            raise ValueError(f"Model feature dtypes drifted: {mismatches}")
        output: dict[str, Any] = {}
        for name in self.feature_names:
            if name in self.vocabularies:
                vocabulary = self.vocabularies[name]
                codes = np.fromiter(
                    (
                        MISSING_CODE if (token := _token(value)) is None
                        else vocabulary.get(token, UNKNOWN_CODE)
                        for value in features[name].to_list()
                    ),
                    dtype=np.int32,
                    count=features.height,
                )
                output[name] = pd.Categorical(
                    codes,
                    categories=range(max(vocabulary.values(), default=UNKNOWN_CODE) + 1),
                )
            else:
                values = features[name].cast(pl.Float64).to_numpy()
                output[name] = values.astype(np.float32, copy=False)
        frame = pd.DataFrame(output, columns=self.feature_names)
        if list(frame.columns) != self.feature_names:
            raise ValueError("Encoded feature schema/order drifted")
        return frame

    def to_dict(self) -> dict[str, Any]:
        return {
            "adapter_version": "lgbm-infra-v1",
            "input_build_version": M2_BUILD_VERSION,
            "feature_set": self.feature_set,
            "feature_manifest_sha256": self.feature_manifest_sha256,
            "feature_names": self.feature_names,
            "feature_dtypes": self.feature_dtypes,
            "categorical_features": self.categorical_features,
            "numerical_features": self.numerical_features,
            "vocabularies": self.vocabularies,
            "include_is_rand": self.include_is_rand,
            "encoding": {
                "missing_code": MISSING_CODE,
                "unknown_code": UNKNOWN_CODE,
                "known_code_offset": KNOWN_OFFSET,
            },
        }

    def save(self, path: Path) -> None:
        path.write_text(json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n")

    @classmethod
    def load(cls, path: Path, *, expected_manifest_sha256: str) -> "LightGBMAdapter":
        data = json.loads(path.read_text())
        if data.get("input_build_version") != M2_BUILD_VERSION:
            raise ValueError("Adapter has a stale M2 build version")
        if data.get("feature_manifest_sha256") != expected_manifest_sha256:
            raise ValueError("Adapter is incompatible with the feature manifest")
        return cls(
            feature_set=data["feature_set"],
            feature_manifest_sha256=data["feature_manifest_sha256"],
            feature_names=data["feature_names"],
            feature_dtypes=data["feature_dtypes"],
            categorical_features=data["categorical_features"],
            numerical_features=data["numerical_features"],
            vocabularies=data["vocabularies"],
            include_is_rand=bool(data.get("include_is_rand", False)),
        )


def select_task(
    encoded: pd.DataFrame,
    features: pl.DataFrame,
    labels: pl.DataFrame,
    task: str,
    sample_weights: np.ndarray | None = None,
) -> TaskDataset:
    if task not in TASKS:
        if task == "is_click":
            raise ValueError("Pooled is_click is forbidden; choose an interface-specific task")
        raise ValueError(f"Unknown task: {task}")
    if not features["impression_id"].equals(labels["impression_id"]):
        raise ValueError("Feature and label impression IDs differ by row")
    if len(encoded) != features.height:
        raise ValueError("Encoded features differ in row count")
    if task in {"long_view", "like"}:
        mask = np.ones(features.height, dtype=bool)
        label_name = "long_view" if task == "long_view" else "is_like"
    else:
        semantic = "valid_play" if task == "valid_play" else "literal_click"
        mask = (labels["click_semantics"] == semantic).to_numpy()
        label_name = "is_click"
    y = labels[label_name].to_numpy()[mask].astype(np.int8, copy=False)
    if y.size == 0 or not np.isin(y, (0, 1)).all():
        raise ValueError(f"Task {task} has no valid binary training rows")
    ids = features["impression_id"].to_numpy()[mask]
    selected_weights = None
    if sample_weights is not None:
        weights = np.asarray(sample_weights, dtype=np.float64)
        if weights.shape != (features.height,) or not np.isfinite(weights).all() or np.any(weights <= 0):
            raise ValueError("Sample weights must be positive, finite, and row-aligned")
        selected_weights = weights[mask]
    return TaskDataset(
        encoded.loc[mask].reset_index(drop=True), y, ids, task,
        int(mask.sum()), selected_weights,
    )


def lightgbm_parameters(config: dict[str, Any], complexity: str) -> tuple[dict[str, Any], int]:
    if complexity not in {"shallow", "deep"}:
        raise ValueError(f"Unknown LightGBM complexity: {complexity}")
    common = config["lightgbm"]
    params = {
        key: common[key] for key in (
            "objective", "learning_rate", "min_data_in_leaf", "lambda_l1",
            "lambda_l2", "feature_fraction", "bagging_fraction", "bagging_freq",
            "max_bin", "num_threads", "deterministic", "force_col_wise", "verbosity",
        )
    }
    seed = int(common["seed"])
    params.update(config["complexity"][complexity])
    params.update({"seed": seed, "feature_fraction_seed": seed, "bagging_seed": seed, "data_random_seed": seed})
    return params, int(common["num_boost_round"])


def train_booster(
    dataset: TaskDataset,
    adapter: LightGBMAdapter,
    *,
    complexity: str,
    config: dict[str, Any] | None = None,
    num_boost_round: int | None = None,
) -> lgb.Booster:
    config = config or load_lightgbm_config()
    params, configured_rounds = lightgbm_parameters(config, complexity)
    if list(dataset.X.columns) != adapter.feature_names:
        raise ValueError("Training matrix schema/order differs from the adapter")
    if len(dataset.X) != dataset.y.size or dataset.rows != dataset.y.size:
        raise ValueError("Training matrix, labels, and task row count differ")
    training = lgb.Dataset(
        dataset.X,
        label=dataset.y,
        weight=dataset.sample_weight,
        categorical_feature=adapter.categorical_features,
        free_raw_data=False,
    )
    return lgb.train(params, training, num_boost_round=num_boost_round or configured_rounds)


def predict_probabilities(booster: lgb.Booster, adapter: LightGBMAdapter, features: pl.DataFrame) -> np.ndarray:
    predictions = np.asarray(booster.predict(adapter.transform(features)), dtype=np.float64)
    if predictions.shape != (features.height,) or not np.isfinite(predictions).all():
        raise ValueError("LightGBM returned invalid prediction shape/values")
    if ((predictions < 0.0) | (predictions > 1.0)).any():
        raise ValueError("LightGBM probabilities lie outside [0,1]")
    return predictions


def save_model_bundle(
    directory: Path,
    booster: lgb.Booster,
    adapter: LightGBMAdapter,
    *,
    task: str,
    complexity: str,
    training_metadata: dict[str, Any] | None = None,
    adapter_source_path: Path | None = None,
) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    model_path = directory / "model.txt"
    adapter_path = directory / "adapter.json"
    booster.save_model(str(model_path))
    if adapter_source_path is None:
        adapter.save(adapter_path)
    else:
        if not adapter_source_path.is_file():
            raise ValueError("Audited adapter source does not exist")
        shutil.copy2(adapter_source_path, adapter_path)
    manifest = {
        "model_version": "lgbm-infra-v1",
        "input_build_version": M2_BUILD_VERSION,
        "feature_set": adapter.feature_set,
        "feature_manifest_sha256": adapter.feature_manifest_sha256,
        "task": task,
        "complexity": complexity,
        "feature_names": adapter.feature_names,
        "feature_dtypes": adapter.feature_dtypes,
        "categorical_features": adapter.categorical_features,
        "vocabulary_sha256": _json_sha256(adapter.vocabularies),
        "training_config_version": "lgbm-infra-v1",
        "effective_parameters": booster.params,
        "num_trees": booster.num_trees(),
        "model_sha256": _sha256(model_path),
        "adapter_sha256": _sha256(adapter_path),
        "training_metadata": training_metadata or {},
    }
    (directory / "model_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")


def load_model_bundle(
    directory: Path,
    *,
    expected_manifest_sha256: str,
    expected_feature_set: str | None = None,
    expected_task: str | None = None,
    expected_complexity: str | None = None,
) -> tuple[lgb.Booster, LightGBMAdapter, dict[str, Any]]:
    required = (directory / "model.txt", directory / "adapter.json", directory / "model_manifest.json")
    missing = [path.name for path in required if not path.is_file()]
    if missing:
        raise ValueError(f"Incomplete model bundle; missing {missing}")
    manifest = json.loads((directory / "model_manifest.json").read_text())
    if (
        manifest.get("model_version") != "lgbm-infra-v1"
        or manifest.get("training_config_version") != "lgbm-infra-v1"
        or manifest.get("input_build_version") != M2_BUILD_VERSION
    ):
        raise ValueError("Model bundle has a stale version")
    if manifest.get("feature_manifest_sha256") != expected_manifest_sha256:
        raise ValueError("Model bundle is incompatible with the feature manifest")
    expected_fields = {
        "feature_set": expected_feature_set,
        "task": expected_task,
        "complexity": expected_complexity,
    }
    for field, expected in expected_fields.items():
        if expected is not None and manifest.get(field) != expected:
            raise ValueError(f"Model bundle {field} is incompatible")
    model_path, adapter_path = directory / "model.txt", directory / "adapter.json"
    if _sha256(model_path) != manifest["model_sha256"] or _sha256(adapter_path) != manifest["adapter_sha256"]:
        raise ValueError("Model bundle checksum mismatch")
    adapter = LightGBMAdapter.load(adapter_path, expected_manifest_sha256=expected_manifest_sha256)
    booster = lgb.Booster(model_file=str(model_path))
    if booster.feature_name() != adapter.feature_names:
        raise ValueError("Model and adapter feature schemas differ")
    return booster, adapter, manifest


def load_training_artifacts(
    processed_root: Path,
    feature_set: str,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    if feature_set not in FEATURE_SETS:
        raise ValueError(f"Unknown feature set: {feature_set}")
    source = processed_root / feature_set / "v1" / "normal_train"
    features, labels, _ = load_aligned_model_data(
        source / "features.parquet",
        source / "labels.parquet",
        expected_stage=feature_set,
        expected_split="normal_train",
    )
    return features, labels
