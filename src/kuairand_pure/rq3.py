"""M5.1 preparation for frozen exposure-aware LightGBM training arms."""

from __future__ import annotations

import hashlib
import json
import resource
import time
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl

from kuairand_pure.data.alignment import (
    load_aligned_model_data,
    validate_row_alignment,
    write_artifact_manifest,
)
from kuairand_pure.models.lightgbm import (
    LightGBMAdapter,
    TaskDataset,
    load_lightgbm_config,
    load_model_bundle,
    select_task,
)
from kuairand_pure.paths import get_artifact_paths

NORMAL_DATES = (20220412, 20220429)
RANDOM_DATES = (20220422, 20220429)
EXPECTED_NORMAL_ROWS = 722_481
STRATEGIES = ("marginal_reweighting", "joint", "random_only")


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_rq3_config(path: Path | None = None) -> dict[str, Any]:
    with (path or _repo_root() / "configs" / "rq3.toml").open("rb") as stream:
        config = tomllib.load(stream)
    validate_rq3_config(config)
    return config


def validate_rq3_config(config: dict[str, Any]) -> None:
    rq3 = config["rq3"]
    if (rq3["feature_set"], rq3["complexity"], rq3["task"]) != ("fs1", "deep", "long_view"):
        raise ValueError("RQ3 architecture drifted from FS1-deep long-view LightGBM")
    weighting = config["weighting"]
    expected_weighting = {
        "additive_alpha": 1.0,
        "clip_quantile": 0.99,
        "quantile_method": "linear",
        "normalization": "arithmetic_mean_one_after_clipping",
    }
    for key, expected in expected_weighting.items():
        if weighting.get(key) != expected:
            raise ValueError(f"Frozen RQ3 weighting setting drifted: {key}")
    if weighting.get("causal_ips") is not False:
        raise ValueError("Marginal exposure weighting must not be labeled causal IPS")
    lightgbm = config["lightgbm"]
    frozen = load_lightgbm_config()
    params = frozen["lightgbm"]
    deep = frozen["complexity"]["deep"]
    expected = {
        "seed": params["seed"], "learning_rate": params["learning_rate"],
        "num_boost_round": params["num_boost_round"],
        "min_data_in_leaf": params["min_data_in_leaf"],
        "lambda_l1": params["lambda_l1"], "lambda_l2": params["lambda_l2"],
        "deterministic": params["deterministic"], "num_threads": params["num_threads"],
        "num_leaves": deep["num_leaves"], "max_depth": deep["max_depth"],
    }
    if any(lightgbm.get(key) != value for key, value in expected.items()):
        raise ValueError("RQ3 LightGBM settings differ from frozen FS1-deep baseline")


@dataclass(frozen=True, slots=True)
class MarginalExposureWeighter:
    counts: dict[int, int]
    alpha: float
    clip_quantile: float
    raw_clip: float
    normalization_mean: float

    @classmethod
    def fit(
        cls,
        video_ids: np.ndarray,
        *,
        alpha: float = 1.0,
        clip_quantile: float = 0.99,
    ) -> "MarginalExposureWeighter":
        values = np.asarray(video_ids)
        if values.ndim != 1 or values.size == 0:
            raise ValueError("Weighting population must be a nonempty vector")
        if alpha <= 0 or not 0 < clip_quantile < 1:
            raise ValueError("Invalid smoothing or clipping constant")
        unique, observed_counts = np.unique(values, return_counts=True)
        counts = {int(video): int(count) for video, count in zip(unique, observed_counts)}
        raw = np.fromiter(
            (1.0 / (counts[int(video)] + alpha) for video in values),
            dtype=np.float64,
            count=values.size,
        )
        raw_clip = float(np.quantile(raw, clip_quantile, method="linear"))
        normalization_mean = float(np.minimum(raw, raw_clip).mean())
        if raw_clip <= 0 or normalization_mean <= 0:
            raise ValueError("Invalid learned marginal-weight constants")
        return cls(counts, alpha, clip_quantile, raw_clip, normalization_mean)

    @property
    def unseen_normalized_weight(self) -> float:
        return self.raw_clip / self.normalization_mean

    def raw_weights(self, video_ids: np.ndarray) -> np.ndarray:
        return np.fromiter(
            (
                1.0 / (self.counts[int(video)] + self.alpha)
                if int(video) in self.counts else self.raw_clip
                for video in np.asarray(video_ids)
            ),
            dtype=np.float64,
            count=len(video_ids),
        )

    def transform(self, video_ids: np.ndarray) -> np.ndarray:
        return np.minimum(self.raw_weights(video_ids), self.raw_clip) / self.normalization_mean

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": "rq3-marginal-exposure-v1",
            "alpha": self.alpha,
            "clip_quantile": self.clip_quantile,
            "quantile_method": "linear",
            "raw_clip": self.raw_clip,
            "normalization_mean": self.normalization_mean,
            "unseen_normalized_weight": self.unseen_normalized_weight,
            "video_count": len(self.counts),
            "counts": {str(key): value for key, value in sorted(self.counts.items())},
        }


def weight_diagnostics(weights: np.ndarray) -> dict[str, Any]:
    values = np.asarray(weights, dtype=np.float64)
    if values.ndim != 1 or values.size == 0 or not np.isfinite(values).all() or np.any(values <= 0):
        raise ValueError("Sample weights must be finite, positive, and nonempty")
    total = float(values.sum())
    ordered = np.sort(values)[::-1]
    quantiles = (0.0, 0.01, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99, 0.995, 0.999, 1.0)
    return {
        "rows": int(values.size),
        "mean": float(values.mean()),
        "minimum": float(values.min()),
        "maximum": float(values.max()),
        "standard_deviation": float(values.std()),
        "quantiles": {str(q): float(np.quantile(values, q, method="linear")) for q in quantiles},
        "effective_sample_size": total * total / float(values @ values),
        "effective_sample_fraction": (total * total / float(values @ values)) / values.size,
        "top_1_percent_weight_share": float(ordered[: max(1, int(np.ceil(values.size * 0.01)))].sum() / total),
        "top_10_percent_weight_share": float(ordered[: max(1, int(np.ceil(values.size * 0.10)))].sum() / total),
    }


def _load_fs1(processed_root: Path, split: str) -> tuple[pl.DataFrame, pl.DataFrame]:
    directory = processed_root / "fs1" / "v1" / split
    features, labels, _ = load_aligned_model_data(
        directory / "features.parquet",
        directory / "labels.parquet",
        expected_stage="fs1",
        expected_split=split,
    )
    return features, labels


def _validate_source(
    features: pl.DataFrame,
    labels: pl.DataFrame,
    *,
    dates: tuple[int, int],
    is_rand: int,
) -> None:
    validate_row_alignment(features, labels)
    if (features["date"].min(), features["date"].max()) != dates:
        raise ValueError("RQ3 training date boundary drifted")
    if set(features["context__is_rand"].unique().to_list()) != {is_rand}:
        raise ValueError("RQ3 source indicator disagrees with training source")
    if not features["date"].equals(labels["date"]):
        raise ValueError("Feature and label dates differ by row")


def _write_strategy(
    directory: Path,
    *,
    split: str,
    features: pl.DataFrame,
    labels: pl.DataFrame,
) -> tuple[Path, Path, dict[str, Any]]:
    directory.mkdir(parents=True, exist_ok=True)
    feature_path = directory / "features.parquet"
    label_path = directory / "labels.parquet"
    features.sort("impression_id").write_parquet(feature_path)
    labels.sort("impression_id").write_parquet(label_path)
    artifact = write_artifact_manifest(
        directory, stage="fs1", split=split,
        features_path=feature_path, labels_path=label_path,
    )
    return feature_path, label_path, artifact


def _class_balance(labels: pl.DataFrame) -> dict[str, Any]:
    positives = int(labels["long_view"].sum())
    return {
        "rows": labels.height,
        "long_view_positives": positives,
        "long_view_prevalence": positives / labels.height,
    }


def prepare_rq3_training_data(
    *,
    processed_root: Path | None = None,
    model_root: Path | None = None,
    prediction_root: Path | None = None,
    output_root: Path | None = None,
    feature_manifest_path: Path | None = None,
) -> dict[str, Any]:
    """Materialize and audit the three authorized RQ3 training strategies."""
    started = time.perf_counter()
    paths = get_artifact_paths()
    processed_root = processed_root or paths.processed
    model_root = model_root or paths.models / "lightgbm" / "v1"
    prediction_root = prediction_root or paths.predictions / "validation" / "v1"
    output_root = output_root or processed_root / "rq3" / "v1"
    feature_manifest_path = feature_manifest_path or processed_root / "m2" / "v1" / "feature_manifest.json"
    config = load_rq3_config()
    manifest_sha256 = _sha256(feature_manifest_path)

    baseline_dir = model_root / "fs1" / "deep" / "long_view"
    _, baseline_adapter, baseline_manifest = load_model_bundle(
        baseline_dir,
        expected_manifest_sha256=manifest_sha256,
        expected_feature_set="fs1",
        expected_task="long_view",
        expected_complexity="deep",
    )
    candidate_manifest_path = prediction_root / "candidate_manifest.json"
    candidate_manifest = json.loads(candidate_manifest_path.read_text())
    selected = candidate_manifest["selected_overall"]
    if selected.get("candidate") != "fs1_deep" or selected.get("checksum") != config["baseline"]["model_sha256"]:
        raise ValueError("M4.5 selection metadata differs from the frozen RQ3 baseline")
    if baseline_manifest["model_sha256"] != config["baseline"]["model_sha256"]:
        raise ValueError("Frozen baseline model checksum drifted")
    if baseline_manifest["adapter_sha256"] != config["baseline"]["adapter_sha256"]:
        raise ValueError("Frozen baseline adapter checksum drifted")
    if baseline_adapter.include_is_rand or "context__is_rand" in baseline_adapter.feature_names:
        raise ValueError("Normal-only baseline unexpectedly uses source indicator")

    normal_features, normal_labels = _load_fs1(processed_root, "normal_train")
    random_features, random_labels = _load_fs1(processed_root, "random_train")
    _validate_source(normal_features, normal_labels, dates=NORMAL_DATES, is_rand=0)
    _validate_source(random_features, random_labels, dates=RANDOM_DATES, is_rand=1)
    if normal_features.height != EXPECTED_NORMAL_ROWS:
        raise ValueError("Normal weighting population is not the approved 722,481 rows")
    if set(normal_features["impression_id"].to_list()).intersection(random_features["impression_id"].to_list()):
        raise ValueError("Normal and randomized impression IDs overlap")

    output_root.mkdir(parents=True, exist_ok=True)
    marginal_dir = output_root / "marginal_reweighting"
    marginal_features_path, marginal_labels_path, marginal_artifact = _write_strategy(
        marginal_dir, split="rq3_marginal_reweighting",
        features=normal_features, labels=normal_labels,
    )
    weighting = config["weighting"]
    weighter = MarginalExposureWeighter.fit(
        normal_features["video_id"].to_numpy(),
        alpha=float(weighting["additive_alpha"]),
        clip_quantile=float(weighting["clip_quantile"]),
    )
    raw_weights = weighter.raw_weights(normal_features["video_id"].to_numpy())
    normalized_weights = weighter.transform(normal_features["video_id"].to_numpy())
    weight_path = marginal_dir / "sample_weights.parquet"
    pl.DataFrame({
        "impression_id": normal_features["impression_id"],
        "video_id": normal_features["video_id"],
        "raw_weight": raw_weights,
        "clipped_weight": np.minimum(raw_weights, weighter.raw_clip),
        "sample_weight": normalized_weights,
    }).write_parquet(weight_path)
    weighter_path = marginal_dir / "weighter.json"
    weighter_path.write_text(json.dumps(weighter.to_dict(), indent=2, sort_keys=True) + "\n")

    joint_features = pl.concat([normal_features, random_features]).sort("impression_id")
    joint_labels = pl.concat([normal_labels, random_labels]).sort("impression_id")
    joint_dir = output_root / "joint"
    joint_features_path, joint_labels_path, joint_artifact = _write_strategy(
        joint_dir, split="rq3_joint", features=joint_features, labels=joint_labels,
    )
    if joint_features["context__is_rand"].value_counts().sort("context__is_rand").to_dicts() != [
        {"context__is_rand": 0, "count": normal_features.height},
        {"context__is_rand": 1, "count": random_features.height},
    ]:
        raise ValueError("Joint source indicator counts are incorrect")
    joint_adapter = LightGBMAdapter.fit(
        joint_features, feature_set="fs1",
        feature_manifest_path=feature_manifest_path, include_is_rand=True,
    )
    joint_adapter_path = joint_dir / "adapter.json"
    joint_adapter.save(joint_adapter_path)

    random_dir = output_root / "random_only"
    random_features_path, random_labels_path, random_artifact = _write_strategy(
        random_dir, split="rq3_random_only",
        features=random_features, labels=random_labels,
    )
    random_adapter = LightGBMAdapter.fit(
        random_features, feature_set="fs1",
        feature_manifest_path=feature_manifest_path, include_is_rand=False,
    )
    random_adapter_path = random_dir / "adapter.json"
    random_adapter.save(random_adapter_path)

    # Exercise each future training interface without fitting a booster.
    interface_rows: dict[str, int] = {}
    for strategy, features, labels, adapter in (
        ("marginal_reweighting", normal_features, normal_labels, baseline_adapter),
        ("joint", joint_features, joint_labels, joint_adapter),
        ("random_only", random_features, random_labels, random_adapter),
    ):
        encoded = adapter.transform(features)
        task = select_task(encoded, features, labels, "long_view")
        if task.rows != features.height or not np.array_equal(task.impression_ids, features["impression_id"].to_numpy()):
            raise ValueError(f"{strategy} training interface lost row alignment")
        interface_rows[strategy] = task.rows

    joint_normal = joint_labels.filter(joint_features["context__is_rand"] == 0)
    joint_random = joint_labels.filter(joint_features["context__is_rand"] == 1)
    diagnostics = weight_diagnostics(normalized_weights)
    expected_ess = 152_212.85
    if abs(diagnostics["effective_sample_size"] - expected_ess) > 1.0:
        raise ValueError("Production ESS differs meaningfully from preliminary expectation")
    if abs(diagnostics["maximum"] - 13.0343) > 0.01:
        raise ValueError("Production maximum weight differs meaningfully from expectation")

    artifacts = {
        "marginal_reweighting": {
            "features": str(marginal_features_path), "labels": str(marginal_labels_path),
            "weights": str(weight_path), "weighter": str(weighter_path),
            "artifact_manifest": marginal_artifact,
            "adapter": str(baseline_dir / "adapter.json"),
            "adapter_policy": "exact_frozen_normal_only_baseline_adapter",
        },
        "joint": {
            "features": str(joint_features_path), "labels": str(joint_labels_path),
            "artifact_manifest": joint_artifact, "adapter": str(joint_adapter_path),
            "adapter_fit_population": "normal_plus_random_authorized_training_rows",
        },
        "random_only": {
            "features": str(random_features_path), "labels": str(random_labels_path),
            "artifact_manifest": random_artifact, "adapter": str(random_adapter_path),
            "adapter_fit_population": "random_training_rows_only",
        },
    }
    for strategy, details in artifacts.items():
        manifest = {
            "version": "rq3-preparation-v1",
            "strategy": strategy,
            "feature_set": "fs1",
            "complexity": "deep",
            "task": "long_view",
            "boosting_rounds": 300,
            "seed": 202211,
            "features_sha256": _sha256(Path(details["features"])),
            "labels_sha256": _sha256(Path(details["labels"])),
            "adapter_path": details["adapter"],
            "adapter_sha256": _sha256(Path(details["adapter"])),
            "feature_manifest_sha256": manifest_sha256,
            "context_is_rand_in_model": strategy == "joint",
            "random_evaluation_context_is_rand": 1 if strategy == "joint" else None,
            "training_rows": interface_rows[strategy],
            "production_model_trained": False,
        }
        if strategy == "marginal_reweighting":
            manifest.update({
                "sample_weights_sha256": _sha256(weight_path),
                "weighter_sha256": _sha256(weighter_path),
            })
        strategy_dir = output_root / strategy
        (strategy_dir / "strategy_manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n"
        )

    report = {
        "milestone": "M5.1",
        "status": "complete",
        "baseline": {
            "bundle": str(baseline_dir),
            "model_sha256": baseline_manifest["model_sha256"],
            "adapter_sha256": baseline_manifest["adapter_sha256"],
            "model_manifest_sha256": _sha256(baseline_dir / "model_manifest.json"),
            "feature_manifest_sha256": baseline_manifest["feature_manifest_sha256"],
            "training_dates": baseline_manifest["training_metadata"]["training_dates"],
            "training_rows": baseline_manifest["training_metadata"]["training_rows"],
            "num_trees": baseline_manifest["num_trees"],
            "selection_metadata": selected,
            "candidate_manifest_sha256": _sha256(candidate_manifest_path),
            "modified_or_retrained": False,
        },
        "populations": {
            "normal": {"dates": list(NORMAL_DATES), **_class_balance(normal_labels)},
            "random": {"dates": list(RANDOM_DATES), **_class_balance(random_labels)},
            "joint": {
                "rows": joint_features.height,
                "normal": _class_balance(joint_normal),
                "random": _class_balance(joint_random),
                "random_row_fraction": random_features.height / joint_features.height,
            },
        },
        "weighting": {
            "method": "marginal-exposure reweighting (SNIPS-style heuristic; not causal IPS)",
            "population": "normal training only, 2022-04-12 through 2022-04-29",
            "video_count": len(weighter.counts),
            "alpha": weighter.alpha,
            "raw_rule": "1 / (n_i + 1)",
            "clip_quantile": weighter.clip_quantile,
            "raw_clip": weighter.raw_clip,
            "normalization_mean": weighter.normalization_mean,
            "unseen_video_fallback_normalized": weighter.unseen_normalized_weight,
            "diagnostics": diagnostics,
        },
        "adapters": {
            "marginal_reweighting": {
                "vocabulary_source": "normal training (frozen baseline adapter)",
                "feature_count": len(baseline_adapter.feature_names),
                "context_is_rand_included": False,
            },
            "joint": {
                "vocabulary_source": "authorized normal plus random training",
                "feature_count": len(joint_adapter.feature_names),
                "context_is_rand_included": joint_adapter.include_is_rand,
            },
            "random_only": {
                "vocabulary_source": "authorized random training only",
                "feature_count": len(random_adapter.feature_names),
                "context_is_rand_included": random_adapter.include_is_rand,
            },
            "interface_rows": interface_rows,
        },
        "artifacts": artifacts,
        "matched_rounds_limitation": "All arms use 300 rounds, but training sample size and source distribution differ.",
        "randomized_outcomes_used_as_history": False,
        "randomized_validation_outcomes_accessed": False,
        "final_test_outcomes_accessed": False,
        "production_models_trained": False,
        "candidate_population_modified": False,
        "runtime_seconds": time.perf_counter() - started,
        "peak_rss_mib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
    }
    (output_root / "preparation_audit.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n"
    )
    return report


def load_prepared_rq3_task(
    strategy: str,
    *,
    processed_root: Path | None = None,
    feature_manifest_path: Path | None = None,
) -> tuple[TaskDataset, LightGBMAdapter, dict[str, Any]]:
    """Load one prepared arm through a single fail-closed training interface."""
    if strategy not in STRATEGIES:
        raise ValueError(f"Unknown RQ3 strategy: {strategy}")
    paths = get_artifact_paths()
    processed_root = processed_root or paths.processed
    feature_manifest_path = feature_manifest_path or processed_root / "m2" / "v1" / "feature_manifest.json"
    manifest_sha256 = _sha256(feature_manifest_path)
    directory = processed_root / "rq3" / "v1" / strategy
    strategy_manifest_path = directory / "strategy_manifest.json"
    if not strategy_manifest_path.is_file():
        raise ValueError("Missing RQ3 strategy manifest")
    strategy_manifest = json.loads(strategy_manifest_path.read_text())
    if strategy_manifest.get("version") != "rq3-preparation-v1" or strategy_manifest.get("strategy") != strategy:
        raise ValueError("Stale or incompatible RQ3 strategy manifest")
    expected_split = {
        "marginal_reweighting": "rq3_marginal_reweighting",
        "joint": "rq3_joint",
        "random_only": "rq3_random_only",
    }[strategy]
    features, labels, _ = load_aligned_model_data(
        directory / "features.parquet", directory / "labels.parquet",
        expected_stage="fs1", expected_split=expected_split,
    )
    for name, path in (("features", directory / "features.parquet"), ("labels", directory / "labels.parquet")):
        if _sha256(path) != strategy_manifest[f"{name}_sha256"]:
            raise ValueError(f"Prepared RQ3 {name} checksum mismatch")
    adapter_path = Path(strategy_manifest["adapter_path"])
    if _sha256(adapter_path) != strategy_manifest["adapter_sha256"]:
        raise ValueError("Prepared RQ3 adapter checksum mismatch")
    adapter = LightGBMAdapter.load(adapter_path, expected_manifest_sha256=manifest_sha256)
    expected_indicator = strategy == "joint"
    if adapter.include_is_rand != expected_indicator:
        raise ValueError("RQ3 adapter source-indicator policy is incompatible")
    weights = None
    if strategy == "marginal_reweighting":
        weight_path = directory / "sample_weights.parquet"
        if _sha256(weight_path) != strategy_manifest["sample_weights_sha256"]:
            raise ValueError("Prepared RQ3 sample-weight checksum mismatch")
        weight_frame = pl.read_parquet(weight_path)
        if not weight_frame["impression_id"].equals(features["impression_id"]):
            raise ValueError("RQ3 sample weights differ from feature impression order")
        weights = weight_frame["sample_weight"].to_numpy()
    encoded = adapter.transform(features)
    task = select_task(encoded, features, labels, "long_view", sample_weights=weights)
    if task.rows != strategy_manifest["training_rows"]:
        raise ValueError("RQ3 task rows differ from strategy manifest")
    return task, adapter, strategy_manifest
