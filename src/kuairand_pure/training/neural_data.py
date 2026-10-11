"""Training-fitted, manifest-bound FS3 adapter and aligned PyTorch dataset."""

from __future__ import annotations

import hashlib
import json
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
import torch
from torch.utils.data import DataLoader, Dataset

from kuairand_pure.data.alignment import M2_BUILD_VERSION, validate_row_alignment
from kuairand_pure.data.fs1 import LABEL_COLUMNS
from kuairand_pure.models.lightgbm import _categorical_feature_names, _token

TASK_NAMES = ("long_view", "like", "valid_play")
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


def load_neural_config(path: Path | None = None) -> dict[str, Any]:
    with (path or _repo_root() / "configs" / "neural.toml").open("rb") as stream:
        return tomllib.load(stream)


@dataclass(slots=True)
class NeuralArrays:
    impression_ids: np.ndarray
    numerical: np.ndarray
    categorical: np.ndarray
    user_ids: np.ndarray
    video_ids: np.ndarray
    history_video_ids: np.ndarray
    history_mask: np.ndarray


@dataclass(slots=True)
class NeuralTargets:
    impression_ids: np.ndarray
    values: np.ndarray
    masks: np.ndarray


@dataclass(slots=True)
class NeuralPreprocessor:
    feature_manifest_sha256: str
    include_is_rand: bool
    numerical_features: list[str]
    categorical_features: list[str]
    numerical_dtypes: dict[str, str]
    categorical_dtypes: dict[str, str]
    medians: dict[str, float]
    means: dict[str, float]
    scales: dict[str, float]
    missing_indicator_features: list[str]
    vocabularies: dict[str, dict[str, int]]
    user_vocabulary: dict[str, int]
    video_vocabulary: dict[str, int]

    @classmethod
    def fit(
        cls,
        features: pl.DataFrame,
        *,
        feature_manifest_path: Path,
        include_is_rand: bool = False,
    ) -> "NeuralPreprocessor":
        if set(LABEL_COLUMNS).intersection(features.columns):
            raise ValueError("Outcome columns entered neural features")
        manifest = json.loads(feature_manifest_path.read_text())
        if manifest.get("manifest_version") != M2_BUILD_VERSION:
            raise ValueError("Neural feature manifest is stale")
        fields = manifest["feature_fields"]
        dense = [
            field for field in fields
            if "fs3" in field["present_in"]
            and field["model_role"] in {"dense_model_feature", "rq3_joint_only"}
            and ("shared_bottom_mlp" in field["allowed_model_families"] or "mmoe" in field["allowed_model_families"])
            and (include_is_rand or field["name"] != "context__is_rand")
        ]
        if include_is_rand is False and "context__is_rand" in {field["name"] for field in dense}:
            raise ValueError("context__is_rand entered a normal-only neural adapter")
        names = [field["name"] for field in dense]
        if any(name not in features.columns for name in names):
            raise ValueError("FS3 features differ from the frozen neural manifest")
        expected_dtypes = {field["name"]: field["data_type"] for field in fields}
        mismatches = {
            name: (expected_dtypes[name], str(features.schema[name]))
            for name in [*names, "user_id", "video_id"]
            if expected_dtypes[name] != str(features.schema[name])
        }
        if mismatches:
            raise ValueError(f"Neural input dtypes differ from manifest: {mismatches}")
        categorical_declared = _categorical_feature_names()
        categorical = [name for name in names if name in categorical_declared]
        numerical = [name for name in names if name not in categorical_declared]
        vocabularies: dict[str, dict[str, int]] = {}
        for name in categorical:
            observed_tokens = {
                token for value in features[name].to_list()
                for token in (_token(value),) if token is not None
            }
            vocabularies[name] = {
                token: index + KNOWN_OFFSET
                for index, token in enumerate(sorted(observed_tokens))
            }
        statistics: dict[str, tuple[float, float, float, bool]] = {}
        for name in numerical:
            values = features[name].cast(pl.Float64).to_numpy()
            missing = ~np.isfinite(values)
            finite = values[~missing]
            if not finite.size:
                median = mean = 0.0
                scale = 1.0
            else:
                median = float(np.median(finite))
                filled = np.where(missing, median, values)
                mean = 0.0 if features.schema[name] == pl.Boolean else float(filled.mean())
                observed_scale = float(filled.std())
                scale = 1.0 if features.schema[name] == pl.Boolean or observed_scale == 0 else observed_scale
            statistics[name] = (median, mean, scale, bool(missing.any()))
        def id_vocab(name: str) -> dict[str, int]:
            observed = {_token(value) for value in features[name].to_list()}
            if None in observed:
                raise ValueError(f"Training {name} contains a missing identifier")
            tokens = sorted(token for token in observed if token is not None)
            return {token: index + KNOWN_OFFSET for index, token in enumerate(tokens)}
        return cls(
            feature_manifest_sha256=_sha256(feature_manifest_path),
            include_is_rand=include_is_rand,
            numerical_features=numerical,
            categorical_features=categorical,
            numerical_dtypes={name: expected_dtypes[name] for name in numerical},
            categorical_dtypes={name: expected_dtypes[name] for name in categorical},
            medians={name: statistics[name][0] for name in numerical},
            means={name: statistics[name][1] for name in numerical},
            scales={name: statistics[name][2] for name in numerical},
            missing_indicator_features=list(numerical),
            vocabularies=vocabularies,
            user_vocabulary=id_vocab("user_id"),
            video_vocabulary=id_vocab("video_id"),
        )

    @property
    def numerical_output_names(self) -> list[str]:
        return [*self.numerical_features, *(f"{name}__missing" for name in self.missing_indicator_features)]

    @property
    def vocabulary_sizes(self) -> dict[str, int]:
        return {
            **{name: max(vocab.values(), default=UNKNOWN_CODE) + 1 for name, vocab in self.vocabularies.items()},
            "user_id": max(self.user_vocabulary.values(), default=UNKNOWN_CODE) + 1,
            "video_id": max(self.video_vocabulary.values(), default=UNKNOWN_CODE) + 1,
        }

    def _encode(self, values: list[Any], vocabulary: dict[str, int]) -> np.ndarray:
        return np.fromiter(
            (
                MISSING_CODE if (token := _token(value)) is None
                else vocabulary.get(token, UNKNOWN_CODE)
                for value in values
            ),
            dtype=np.int64,
            count=len(values),
        )

    def transform(self, features: pl.DataFrame, sequence: pl.DataFrame) -> NeuralArrays:
        validate_row_alignment(features, features.select("impression_id"), sequence)
        if set(LABEL_COLUMNS).intersection(features.columns):
            raise ValueError("Outcome columns entered neural transform")
        expected = {**self.numerical_dtypes, **self.categorical_dtypes}
        mismatches = {
            name: (dtype, str(features.schema.get(name)))
            for name, dtype in expected.items()
            if name not in features.columns or str(features.schema[name]) != dtype
        }
        if mismatches:
            raise ValueError(f"Neural feature schema drifted: {mismatches}")
        numeric_parts: list[np.ndarray] = []
        missing_parts: list[np.ndarray] = []
        for name in self.numerical_features:
            values = features[name].cast(pl.Float64).to_numpy()
            missing = ~np.isfinite(values)
            filled = np.where(missing, self.medians[name], values)
            numeric_parts.append(((filled - self.means[name]) / self.scales[name]).astype(np.float32))
            if name in self.missing_indicator_features:
                missing_parts.append(missing.astype(np.float32))
        numerical = np.column_stack([*numeric_parts, *missing_parts]).astype(np.float32)
        categorical = np.column_stack([
            self._encode(features[name].to_list(), self.vocabularies[name])
            for name in self.categorical_features
        ]).astype(np.int64)
        user_ids = self._encode(features["user_id"].to_list(), self.user_vocabulary)
        video_ids = self._encode(features["video_id"].to_list(), self.video_vocabulary)

        source_history = np.asarray(sequence["history_video_ids"].to_list(), dtype=np.int64)
        history_times = np.asarray(sequence["history_time_ms"].to_list(), dtype=np.int64)
        history_mask = np.asarray(sequence["history_mask"].to_list(), dtype=np.float32)
        lengths = np.asarray(sequence["history_length"].to_list(), dtype=np.int64)
        if source_history.shape != (features.height, 20) or history_mask.shape != source_history.shape:
            raise ValueError("Neural history must contain exactly 20 positions")
        if not np.array_equal(
            sequence["target_time_ms"].to_numpy(), features["time_ms"].to_numpy()
        ):
            raise ValueError("Sequence target timestamps differ from FS3")
        if not np.array_equal(history_mask.sum(axis=1).astype(np.int64), lengths):
            raise ValueError("History mask and length disagree")
        if np.any((history_mask == 0) & (source_history != 0)):
            raise ValueError("Nonzero video ID appears in a padded history position")
        if np.any((history_mask == 1) & (source_history == 0)):
            raise ValueError("Padding ID appears in a valid history position")
        targets = features["time_ms"].to_numpy()[:, None]
        if np.any((history_mask == 1) & (history_times >= targets)):
            raise ValueError("History timestamp is not strictly earlier than target")
        if np.any((history_mask == 0) & (history_times != 0)):
            raise ValueError("Padded history timestamp is not zero")
        history_video_ids = np.zeros_like(source_history)
        history_video_ids[(history_mask == 1) & (source_history == 1)] = UNKNOWN_CODE
        for encoded in np.unique(source_history[(history_mask == 1) & (source_history >= 2)]):
            raw_video_id = int(encoded) - 2
            history_video_ids[source_history == encoded] = self.video_vocabulary.get(
                _token(raw_video_id), UNKNOWN_CODE
            )
        return NeuralArrays(
            impression_ids=features["impression_id"].to_numpy(),
            numerical=numerical,
            categorical=categorical,
            user_ids=user_ids,
            video_ids=video_ids,
            history_video_ids=history_video_ids,
            history_mask=history_mask,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "adapter_version": "neural-infra-v1",
            "input_build_version": M2_BUILD_VERSION,
            "feature_set": "fs3",
            "feature_manifest_sha256": self.feature_manifest_sha256,
            "include_is_rand": self.include_is_rand,
            "numerical_features": self.numerical_features,
            "categorical_features": self.categorical_features,
            "numerical_dtypes": self.numerical_dtypes,
            "categorical_dtypes": self.categorical_dtypes,
            "medians": self.medians,
            "means": self.means,
            "scales": self.scales,
            "missing_indicator_features": self.missing_indicator_features,
            "vocabularies": self.vocabularies,
            "user_vocabulary": self.user_vocabulary,
            "video_vocabulary": self.video_vocabulary,
            "history": {"length": 20, "padding_id": 0, "unknown_id": 1, "pooling": "masked_mean"},
            "normal_only_context_is_rand_excluded": not self.include_is_rand,
        }

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n")

    @classmethod
    def load(cls, path: Path, *, expected_manifest_sha256: str) -> "NeuralPreprocessor":
        data = json.loads(path.read_text())
        if data.get("adapter_version") != "neural-infra-v1" or data.get("input_build_version") != M2_BUILD_VERSION:
            raise ValueError("Neural preprocessor has a stale version")
        if data.get("feature_manifest_sha256") != expected_manifest_sha256:
            raise ValueError("Neural preprocessor is incompatible with the feature manifest")
        fields = {key: data[key] for key in (
            "feature_manifest_sha256", "include_is_rand", "numerical_features",
            "categorical_features", "numerical_dtypes", "categorical_dtypes",
            "medians", "means", "scales", "missing_indicator_features",
            "vocabularies", "user_vocabulary", "video_vocabulary",
        )}
        return cls(**fields)


def build_neural_targets(labels: pl.DataFrame) -> NeuralTargets:
    if not set(("impression_id", "long_view", "is_like", "is_click", "click_semantics")).issubset(labels.columns):
        raise ValueError("Neural label table is incomplete")
    long_view = labels["long_view"].to_numpy().astype(np.float32)
    like = labels["is_like"].to_numpy().astype(np.float32)
    valid_mask = (labels["click_semantics"] == "valid_play").to_numpy()
    valid_play = np.where(valid_mask, labels["is_click"].to_numpy(), 0).astype(np.float32)
    values = np.column_stack((long_view, like, valid_play)).astype(np.float32)
    masks = np.column_stack((
        np.ones(labels.height, dtype=np.float32),
        np.ones(labels.height, dtype=np.float32),
        valid_mask.astype(np.float32),
    ))
    if not np.isin(values, (0, 1)).all():
        raise ValueError("Neural targets must be binary")
    return NeuralTargets(labels["impression_id"].to_numpy(), values, masks)


class NeuralDataset(Dataset):
    def __init__(self, arrays: NeuralArrays, targets: NeuralTargets) -> None:
        if not np.array_equal(arrays.impression_ids, targets.impression_ids):
            raise ValueError("Neural features, sequence, and targets differ by impression ID")
        if len(np.unique(arrays.impression_ids)) != len(arrays.impression_ids):
            raise ValueError("Neural dataset has duplicate impression IDs")
        self.arrays = arrays
        self.targets = targets

    def __len__(self) -> int:
        return self.arrays.impression_ids.size

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        return {
            "impression_id": torch.as_tensor(self.arrays.impression_ids[index].astype(np.int64)),
            "numerical": torch.from_numpy(self.arrays.numerical[index]),
            "categorical": torch.from_numpy(self.arrays.categorical[index]),
            "user_id": torch.as_tensor(self.arrays.user_ids[index]),
            "video_id": torch.as_tensor(self.arrays.video_ids[index]),
            "history_video_ids": torch.from_numpy(self.arrays.history_video_ids[index]),
            "history_mask": torch.from_numpy(self.arrays.history_mask[index]),
            "targets": torch.from_numpy(self.targets.values[index]),
            "task_masks": torch.from_numpy(self.targets.masks[index]),
        }


def make_dataloader(
    dataset: NeuralDataset,
    *,
    batch_size: int,
    seed: int,
    shuffle: bool = True,
) -> DataLoader:
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle, generator=generator,
        num_workers=0, drop_last=False,
    )
