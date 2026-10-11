"""M6.1 held-out feature construction and prediction-only scoring."""

from __future__ import annotations

import hashlib
import json
import resource
import subprocess
import time
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
import torch
from torch.utils.data import DataLoader, Dataset

from kuairand_pure.baselines.r0 import R0_INPUT_COLUMNS, score_r0
from kuairand_pure.data.alignment import M2_BUILD_VERSION, validate_impression_ids
from kuairand_pure.data.fs1 import (
    LABEL_COLUMNS, RAW_FEATURE_COLUMNS, _bounded_label_scan, _impressions, _paths,
    _static_tables, build_fs1_lazy,
)
from kuairand_pure.data.splits import SplitSpec
from kuairand_pure.features.fs2 import (
    build_fs2_lazy, compute_priors, daily_history_snapshots, load_fs2_config,
    load_normal_history, verify_frozen_priors,
)
from kuairand_pure.features.fs3 import (
    _daily_affinity_snapshots, _daily_recency_snapshots, build_fs3_lazy,
    build_sequence_frame, load_fs3_config, load_normal_structural_history,
    video_categories,
)
from kuairand_pure.final_evaluation import (
    FINAL_END, FINAL_START, apply_evaluation_source_context,
)
from kuairand_pure.models.lightgbm import FEATURE_SETS, TASKS, load_model_bundle, predict_probabilities
from kuairand_pure.neural_evaluation import NEURAL_CANDIDATES, SEEDS, _model_for
from kuairand_pure.paths import get_artifact_paths
from kuairand_pure.training.neural import load_checkpoint, move_batch, resolve_device
from kuairand_pure.training.neural_data import NeuralArrays, NeuralPreprocessor, TASK_NAMES, load_neural_config

EXPECTED_ROWS = {"normal": 93_201, "random": 637_492}
HISTORY_START = 20220409
BASE_HISTORY_END = 20220502
FINAL_VERSION = "m6.1-final-predictions-v1"
FINAL_HISTORY_END = 20220507


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def authorize_final_normal_feature_history(
    *, source: str, start: int, end: int, purpose: str
) -> None:
    """Authorize only May 3-7 normal outcomes for strictly prior-day features."""
    if source != "normal":
        raise PermissionError("Randomized outcomes cannot be feature history")
    if purpose != "strictly_prior_day_feature_history":
        raise PermissionError("Final outcomes are not authorized for this purpose")
    if not (FINAL_START <= start <= end <= FINAL_HISTORY_END):
        raise PermissionError("Rolling final history must stay within May 3-7")


def verify_m6_1_preflight() -> dict[str, Any]:
    """Verify frozen refs and every M5 readiness hash before final construction."""
    paths = get_artifact_paths()
    required_tags = ("week0-protocol-v1", "m2-features-v1", "m4-candidates-v1", "m5-prefinal-v1")
    tag_commits = {
        tag: subprocess.check_output(["git", "rev-list", "-n", "1", tag], cwd=_repo_root(), text=True).strip()
        for tag in required_tags
    }
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=_repo_root(), text=True).strip()
    if head != tag_commits["m5-prefinal-v1"]:
        raise ValueError("HEAD is not the frozen m5-prefinal-v1 checkpoint")
    readiness_path = paths.processed / "prefinal" / "v1" / "readiness_manifest.json"
    readiness = json.loads(readiness_path.read_text())
    if readiness.get("candidate_count") != 12 or readiness.get("held_out_outcomes_accessed") is not False:
        raise ValueError("M5 readiness manifest is incomplete or unsafe")
    code_paths = {
        "final_evaluation_config": _repo_root() / "configs" / "final_evaluation.toml",
        "rq3_config": _repo_root() / "configs" / "rq3.toml",
        "lightgbm_config": _repo_root() / "configs" / "lightgbm.toml",
        "metrics_code": _repo_root() / "src" / "kuairand_pure" / "metrics.py",
        "inference_code": _repo_root() / "src" / "kuairand_pure" / "inference.py",
        "final_evaluation_code": _repo_root() / "src" / "kuairand_pure" / "final_evaluation.py",
    }
    mismatches = {name: (_sha256(path), readiness["code_and_config_hashes"][name])
                  for name, path in code_paths.items()
                  if _sha256(path) != readiness["code_and_config_hashes"][name]}
    if mismatches:
        raise ValueError(f"Frozen readiness code/config hashes changed: {mismatches}")
    protocol_paths = {
        "study_plan_v5": _repo_root() / "docs" / "study_plan_v5.md",
        "week0_amendment_v1": _repo_root() / "docs" / "week0_amendment_v1.md",
        "feature_manifest": paths.processed / "m2" / "v1" / "feature_manifest.json",
    }
    if any(_sha256(path) != readiness["protocol_hashes"][name] for name, path in protocol_paths.items()):
        raise ValueError("Frozen protocol or feature-manifest hash changed")
    # Reuse the exhaustive bundle/checksum verifier that produced readiness.
    from kuairand_pure.prefinal import _verify_candidate_artifacts
    candidate_path = paths.predictions / "validation" / "v1" / "candidate_manifest.json"
    candidates = json.loads(candidate_path.read_text())
    if _sha256(candidate_path) != readiness["candidate_manifest_sha256"]:
        raise ValueError("Frozen candidate manifest checksum changed")
    _verify_candidate_artifacts(candidates)
    rq3_report = json.loads((paths.models / "lightgbm" / "rq3" / "v1" / "training_report.json").read_text())
    for record in rq3_report["records"]:
        directory = Path(record["artifact_dir"])
        if _sha256(directory / "model.txt") != record["model_sha256"] or _sha256(directory / "adapter.json") != record["adapter_sha256"]:
            raise ValueError("Frozen RQ3 bundle checksum changed")
    return {"head": head, "tag_commits": tag_commits, "readiness_sha256": _sha256(readiness_path),
            "candidate_manifest_sha256": _sha256(candidate_path), "candidate_count": 12,
            "rq3_model_count": len(rq3_report["records"]), "frozen_hashes_valid": True}


def _feature_manifest(directory: Path, *, stage: str, split: str, feature_path: Path,
                      sequence_path: Path | None = None) -> dict[str, Any]:
    features = pl.read_parquet(feature_path)
    validate_impression_ids(features, f"{split} {stage} features")
    sequence = pl.read_parquet(sequence_path) if sequence_path else None
    if sequence is not None:
        validate_impression_ids(sequence, f"{split} sequence")
        if not features["impression_id"].equals(sequence["impression_id"]):
            raise ValueError("Held-out features and sequence differ by row")
    if set(LABEL_COLUMNS).intersection(features.columns):
        raise ValueError("Outcome columns entered held-out features")
    manifest = {
        "build_version": M2_BUILD_VERSION, "final_build_version": FINAL_VERSION,
        "stage": stage, "split": split, "canonical_order": "impression_id_ascending",
        "rows": features.height, "labels_materialized": False,
        "feature_schema": {name: str(dtype) for name, dtype in features.schema.items()},
        "feature_sha256": _sha256(feature_path),
        "sequence_schema": ({name: str(dtype) for name, dtype in sequence.schema.items()} if sequence is not None else None),
        "sequence_sha256": _sha256(sequence_path) if sequence_path else None,
    }
    (directory / "artifact_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return manifest


def _assert_training_schema(path: Path, training_path: Path, stage: str) -> None:
    if pl.scan_parquet(path).collect_schema() != pl.scan_parquet(training_path).collect_schema():
        raise ValueError(f"{stage} held-out schema differs from frozen training schema")


def load_final_normal_outcome_history(data_dir: Path) -> pl.DataFrame:
    """Normal-only Apr 9-May 7 outcomes for strictly prior-day features."""
    training = load_normal_history(data_dir, HISTORY_START, 20220429)
    _, files = _paths(data_dir)
    codes = _paths(data_dir)[0]["source_codes"]
    validation = pl.concat([
        _bounded_label_scan(files["normal_late"], codes["normal_late"],
                            SplitSpec("normal_validation_history", "normal", 20220430, 20220502, True))
    ]).select("user_id", "video_id", "date", "tab", *LABEL_COLUMNS).collect()
    authorize_final_normal_feature_history(
        source="normal", start=FINAL_START, end=FINAL_HISTORY_END,
        purpose="strictly_prior_day_feature_history",
    )
    rolling = _bounded_label_scan(
        files["normal_late"], codes["normal_late"],
        SplitSpec("final_rolling_history", "normal", FINAL_START, FINAL_HISTORY_END, True),
    ).select("user_id", "video_id", "date", "tab", *LABEL_COLUMNS).collect()
    history = pl.concat([training, validation, rolling])
    if history["date"].min() != HISTORY_START or history["date"].max() != FINAL_HISTORY_END:
        raise ValueError("Final normal outcome-history dates drifted")
    return history


def materialize_final_features(*, data_dir: Path | None = None,
                               processed_root: Path | None = None) -> dict[str, Any]:
    started = time.perf_counter()
    paths = get_artifact_paths()
    data_dir = data_dir or paths.raw / "KuaiRand-Pure" / "data"
    processed_root = processed_root or paths.processed
    output_root = processed_root / "final" / "v1"
    users, videos, user_features, video_features = _static_tables(data_dir)
    history = load_final_normal_outcome_history(data_dir)
    fs2_config = load_fs2_config()
    priors = compute_priors(history, fs2_config["prior_window"]["start"], fs2_config["prior_window"]["end"])
    verify_frozen_priors(priors, fs2_config)
    user_snapshots = daily_history_snapshots(history, "user_id", "user")
    video_snapshots = daily_history_snapshots(history, "video_id", "video")
    fs3_config = load_fs3_config()
    missing_token = fs3_config["category"]["missing_token"]
    categories = video_categories(data_dir, missing_token)
    outcome_history = history.join(categories.select("video_id", "category__key"), on="video_id", how="left", validate="m:1").with_columns(pl.col("category__key").fill_null(missing_token))
    structural_history = load_normal_structural_history(data_dir, HISTORY_START, FINAL_END).join(
        categories.select("video_id", "category__key"), on="video_id", how="left", validate="m:1"
    ).with_columns(pl.col("category__key").fill_null(missing_token))
    reports: dict[str, Any] = {}
    for source in ("normal", "random"):
        split = f"T-{source}"
        spec = SplitSpec(split, source, FINAL_START, FINAL_END, False)
        source_root = output_root / source
        for stage in FEATURE_SETS:
            (source_root / stage).mkdir(parents=True, exist_ok=True)
        fs1_path = source_root / "fs1" / "features.parquet"
        build_fs1_lazy(_impressions(data_dir, spec, RAW_FEATURE_COLUMNS), users, videos,
                       user_features, video_features).sort("impression_id").sink_parquet(fs1_path)
        fs1 = pl.read_parquet(fs1_path)
        if fs1.height != EXPECTED_ROWS[source]:
            raise ValueError(f"Unexpected {split} row count")
        validate_impression_ids(fs1, f"{split} FS1")
        _assert_training_schema(fs1_path, processed_root / "fs1" / "v1" / "normal_train" / "features.parquet", "FS1")
        fs1_manifest = _feature_manifest(source_root / "fs1", stage="fs1", split=split, feature_path=fs1_path)

        fs2_path = source_root / "fs2" / "features.parquet"
        build_fs2_lazy(pl.scan_parquet(fs1_path), user_snapshots, video_snapshots, priors,
                       float(fs2_config["fs2"]["alpha"])).sort("impression_id").sink_parquet(fs2_path)
        _assert_training_schema(fs2_path, processed_root / "fs2" / "v1" / "normal_train" / "features.parquet", "FS2")
        fs2_manifest = _feature_manifest(source_root / "fs2", stage="fs2", split=split, feature_path=fs2_path)

        fs3_path = source_root / "fs3" / "features.parquet"
        build_fs3_lazy(
            pl.scan_parquet(fs2_path), categories, _daily_affinity_snapshots(outcome_history),
            _daily_recency_snapshots(structural_history, ["user_id"], "user__recency_"),
            _daily_recency_snapshots(structural_history, ["user_id", "category__key"], "user_category__recency_"),
            priors, float(fs3_config["fs3"]["alpha"]), missing_token,
        ).sort("impression_id").sink_parquet(fs3_path)
        _assert_training_schema(fs3_path, processed_root / "fs3" / "v1" / "normal_train" / "features.parquet", "FS3")
        fs2_frame, fs3 = pl.read_parquet(fs2_path), pl.read_parquet(fs3_path)
        if not fs3.select(fs2_frame.columns).equals(fs2_frame):
            raise ValueError("FS3 did not preserve held-out FS2")
        seq_cfg = fs3_config["sequence"]
        sequence_path = source_root / "fs3" / "sequence.parquet"
        sequence = build_sequence_frame(
            fs3.select("impression_id", "user_id", "time_ms"),
            structural_history.select("impression_id", "user_id", "video_id", "time_ms"),
            set(categories["video_id"].to_list()), length=int(seq_cfg["length"]),
            padding_id=int(seq_cfg["padding_video_id"]), unknown_id=int(seq_cfg["unknown_video_id"]),
            known_offset=int(seq_cfg["known_video_id_offset"]), padding_time_ms=int(seq_cfg["padding_time_ms"]),
        ).sort("impression_id")
        sequence.write_parquet(sequence_path)
        fs3_manifest = _feature_manifest(source_root / "fs3", stage="fs3", split=split,
                                         feature_path=fs3_path, sequence_path=sequence_path)
        masks = np.asarray(sequence["history_mask"].to_list(), dtype=np.uint8)
        times = np.asarray(sequence["history_time_ms"].to_list(), dtype=np.int64)
        target_times = sequence["target_time_ms"].to_numpy()[:, None]
        if np.any((masks == 1) & (times >= target_times)):
            raise ValueError("Held-out sequence contains a non-prior event")
        reports[source] = {
            "split": split, "rows": fs3.height, "unique_impression_ids": fs3["impression_id"].n_unique(),
            "dates": [int(fs3["date"].min()), int(fs3["date"].max())],
            "feature_hashes": {"fs1": fs1_manifest["feature_sha256"], "fs2": fs2_manifest["feature_sha256"], "fs3": fs3_manifest["feature_sha256"], "sequence": fs3_manifest["sequence_sha256"]},
            "sequence_mean_length": float(sequence["history_length"].mean()),
            "sequence_zero_history_rows": int((sequence["history_length"] == 0).sum()),
            "sequence_tie_boundary_rows": int(sequence["history_tie_boundary_dropped"].sum()),
        }
    report = {
        "version": FINAL_VERSION, "populations": reports,
        "history": {"source": "normal_only", "outcome_dates": [HISTORY_START, FINAL_HISTORY_END],
                    "final_window_outcome_dates_accessed_for_features_only": [FINAL_START, FINAL_HISTORY_END],
                    "aggregate_cutoff": "strictly_prior_calendar_day", "sequence_cutoff": "strictly_earlier_time_ms",
                    "randomized_outcomes_accessed": False, "may_8_outcomes_accessed": False},
        "labels_materialized": False, "metrics_computed": False,
        "runtime_seconds": time.perf_counter() - started,
        "peak_rss_mib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
    }
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "feature_build_report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return report


class _InferenceDataset(Dataset):
    def __init__(self, arrays: NeuralArrays): self.arrays = arrays
    def __len__(self) -> int: return self.arrays.impression_ids.size
    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        return {
            "impression_id": torch.as_tensor(self.arrays.impression_ids[index].astype(np.int64)),
            "numerical": torch.from_numpy(self.arrays.numerical[index]),
            "categorical": torch.from_numpy(self.arrays.categorical[index]),
            "user_id": torch.as_tensor(self.arrays.user_ids[index]),
            "video_id": torch.as_tensor(self.arrays.video_ids[index]),
            "history_video_ids": torch.from_numpy(self.arrays.history_video_ids[index]),
            "history_mask": torch.from_numpy(self.arrays.history_mask[index]),
        }


def _eligible(features: pl.DataFrame, task: str) -> np.ndarray:
    if task in {"long_view", "like"}: return np.ones(features.height, dtype=bool)
    if task == "valid_play": return (features["context__tab"] == 1).to_numpy()
    if task == "literal_click": return (features["context__tab"] == 0).to_numpy()
    raise ValueError(f"Unknown task: {task}")


def _write_prediction(root: Path, ids: pl.Series, scores: np.ndarray, metadata: dict[str, Any]) -> dict[str, Any]:
    root.mkdir(parents=True, exist_ok=True)
    frame = pl.DataFrame({"impression_id": ids, "score": scores}, schema_overrides={"impression_id": pl.UInt64, "score": pl.Float64})
    validate_impression_ids(frame, "held-out predictions")
    if not np.isfinite(scores).all() or np.any((scores < 0) | (scores > 1)):
        raise ValueError("Held-out probabilities are invalid")
    path = root / "predictions.parquet"
    frame.write_parquet(path)
    manifest = {"version": FINAL_VERSION, **metadata, "rows": frame.height,
                "canonical_order": "impression_id_ascending", "labels_in_artifact": False,
                "prediction_sha256": _sha256(path), "prediction_bytes": path.stat().st_size}
    (root / "prediction_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return {"path": str(path), **manifest}


def score_final_predictions(*, processed_root: Path | None = None, model_root: Path | None = None,
                            prediction_root: Path | None = None, device_name: str | None = None) -> dict[str, Any]:
    started = time.perf_counter()
    paths = get_artifact_paths()
    processed_root = processed_root or paths.processed
    model_root = model_root or paths.models
    prediction_root = prediction_root or paths.predictions / "final" / "v1"
    feature_manifest_path = processed_root / "m2" / "v1" / "feature_manifest.json"
    feature_manifest_sha = _sha256(feature_manifest_path)
    preprocessor_path = model_root / "neural" / "infrastructure" / "v1" / "preprocessor.json"
    preprocessing_sha = _sha256(preprocessor_path)
    preprocessor = NeuralPreprocessor.load(preprocessor_path, expected_manifest_sha256=feature_manifest_sha)
    config = load_neural_config()
    device = resolve_device(device_name or config["neural"]["device"])
    records: list[dict[str, Any]] = []
    references: list[dict[str, Any]] = []
    for source in ("normal", "random"):
        source_root = processed_root / "final" / "v1" / source
        features = {stage: pl.read_parquet(source_root / stage / "features.parquet") for stage in FEATURE_SETS}
        sequence = pl.read_parquet(source_root / "fs3" / "sequence.parquet")
        for frame in features.values(): validate_impression_ids(frame, f"T-{source} features")
        # R0
        r0 = score_r0(features["fs2"].select(*R0_INPUT_COLUMNS))["r0_score"].to_numpy()
        records.append(_write_prediction(prediction_root / source / "r0" / "long_view",
            features["fs2"]["impression_id"], r0, {"source": source, "candidate": "r0", "task": "long_view", "model_family": "r0"}))
        # Frozen LightGBM candidate matrix.
        for stage in FEATURE_SETS:
            for complexity in ("shallow", "deep"):
                candidate = f"{stage}_{complexity}"
                for task in TASKS:
                    bundle = model_root / "lightgbm" / "v1" / stage / complexity / task
                    booster, adapter, manifest = load_model_bundle(bundle, expected_manifest_sha256=feature_manifest_sha,
                        expected_feature_set=stage, expected_task=task, expected_complexity=complexity)
                    scores = predict_probabilities(booster, adapter, features[stage])
                    mask = _eligible(features[stage], task)
                    records.append(_write_prediction(prediction_root / source / candidate / task,
                        features[stage]["impression_id"].filter(pl.Series(mask)), scores[mask],
                        {"source": source, "candidate": candidate, "task": task, "model_family": "lightgbm",
                         "model_sha256": manifest["model_sha256"], "adapter_sha256": manifest["adapter_sha256"]}))
        # Frozen neural candidates, all seeds.
        arrays = preprocessor.transform(features["fs3"], sequence)
        dataset = _InferenceDataset(arrays)
        for candidate, specification in NEURAL_CANDIDATES.items():
            for seed in SEEDS:
                model = _model_for(candidate, preprocessor, config)
                checkpoint = model_root / "neural" / Path(*specification["relative_root"]) / f"seed_{seed}"
                manifest = load_checkpoint(checkpoint, model=model, expected_architecture=specification["architecture"],
                    expected_preprocessing_sha256=preprocessing_sha, expected_feature_manifest_sha256=feature_manifest_sha, device=device)
                loader = DataLoader(dataset, batch_size=4096, shuffle=False, num_workers=0, drop_last=False)
                ids, chunks = [], []
                model.eval()
                with torch.inference_mode():
                    for raw in loader:
                        batch = move_batch(raw, device)
                        probs = model.probabilities(batch)
                        if not torch.isfinite(probs).all(): raise ValueError("Neural held-out probabilities are nonfinite")
                        ids.append(raw["impression_id"].numpy()); chunks.append(probs.cpu().numpy())
                observed_ids, scores = np.concatenate(ids), np.concatenate(chunks)
                if not np.array_equal(observed_ids.astype(arrays.impression_ids.dtype), arrays.impression_ids):
                    raise ValueError("Neural held-out prediction order drifted")
                for index, task in enumerate(TASK_NAMES):
                    mask = _eligible(features["fs3"], task)
                    records.append(_write_prediction(prediction_root / source / "neural" / candidate / f"seed_{seed}" / task,
                        features["fs3"]["impression_id"].filter(pl.Series(mask)), scores[mask, index].astype(np.float64),
                        {"source": source, "candidate": candidate, "seed": seed, "task": task, "model_family": "neural",
                         "checkpoint_sha256": manifest["checkpoint_sha256"], "preprocessing_sha256": preprocessing_sha}))
        # RQ3 alternatives; normal-only baseline is the identical fs1_deep artifact.
        baseline_path = prediction_root / source / "fs1_deep" / "long_view" / "predictions.parquet"
        references.append({"source": source, "arm": "normal_only_baseline", "referenced_path": str(baseline_path), "prediction_sha256": _sha256(baseline_path)})
        for arm in ("marginal_reweighting", "joint", "random_only"):
            bundle = model_root / "lightgbm" / "rq3" / "v1" / arm
            booster, adapter, manifest = load_model_bundle(bundle, expected_manifest_sha256=feature_manifest_sha,
                expected_feature_set="fs1", expected_task="long_view", expected_complexity="deep")
            prepared = apply_evaluation_source_context(features["fs1"], model_arm=arm, evaluation_source=source)
            scores = predict_probabilities(booster, adapter, prepared)
            records.append(_write_prediction(prediction_root / source / "rq3" / arm / "long_view",
                features["fs1"]["impression_id"], scores,
                {"source": source, "candidate": f"rq3_{arm}", "task": "long_view", "model_family": "lightgbm_rq3",
                 "model_sha256": manifest["model_sha256"], "adapter_sha256": manifest["adapter_sha256"],
                 "context_is_rand_value": 1 if arm == "joint" and source == "random" else (0 if arm == "joint" else None)}))
    report = {
        "version": FINAL_VERSION, "records": records, "rq3_baseline_references": references,
        "record_count": len(records), "metrics_computed": False, "labels_read_for_scoring": False,
        "training_adapters_refit": False, "device": str(device),
        "runtime_seconds": time.perf_counter() - started,
        "peak_rss_mib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
    }
    prediction_root.mkdir(parents=True, exist_ok=True)
    (prediction_root / "prediction_report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return report


def audit_m6_1_outputs(*, processed_root: Path | None = None,
                       prediction_root: Path | None = None) -> dict[str, Any]:
    """Independently validate every production feature and prediction artifact."""
    started = time.perf_counter()
    paths = get_artifact_paths()
    processed_root = processed_root or paths.processed
    prediction_root = prediction_root or paths.predictions / "final" / "v1"
    report = json.loads((prediction_root / "prediction_report.json").read_text())
    if len(report["records"]) != 146 or len(report["rq3_baseline_references"]) != 2:
        raise ValueError("M6.1 prediction coverage is incomplete")
    checks: dict[str, Any] = {}
    for source in ("normal", "random"):
        source_root = processed_root / "final" / "v1" / source
        frames = {stage: pl.read_parquet(source_root / stage / "features.parquet") for stage in FEATURE_SETS}
        for stage, frame in frames.items():
            validate_impression_ids(frame, f"T-{source} {stage}")
            if frame.height != EXPECTED_ROWS[source] or set(LABEL_COLUMNS).intersection(frame.columns):
                raise ValueError(f"T-{source} {stage} coverage or label isolation failed")
            _assert_training_schema(source_root / stage / "features.parquet",
                processed_root / stage / "v1" / "normal_train" / "features.parquet", stage.upper())
        if not frames["fs2"].select(frames["fs1"].columns).equals(frames["fs1"]):
            raise ValueError("FS2 does not preserve FS1")
        if not frames["fs3"].select(frames["fs2"].columns).equals(frames["fs2"]):
            raise ValueError("FS3 does not preserve FS2")
        sequence = pl.read_parquet(source_root / "fs3" / "sequence.parquet")
        validate_impression_ids(sequence, f"T-{source} sequence")
        if not sequence["impression_id"].equals(frames["fs3"]["impression_id"]):
            raise ValueError("Sequence target alignment failed")
        masks = np.asarray(sequence["history_mask"].to_list(), dtype=np.uint8)
        times = np.asarray(sequence["history_time_ms"].to_list(), dtype=np.int64)
        lengths = sequence["history_length"].to_numpy()
        if masks.shape != (EXPECTED_ROWS[source], 20) or np.any(masks.sum(axis=1) != lengths):
            raise ValueError("Sequence mask/length integrity failed")
        if np.any((masks == 1) & (times >= sequence["target_time_ms"].to_numpy()[:, None])):
            raise ValueError("Sequence chronology failed")
        expected_ids = {
            "long_view": frames["fs3"]["impression_id"],
            "like": frames["fs3"]["impression_id"],
            "valid_play": frames["fs3"].filter(pl.col("context__tab") == 1)["impression_id"],
            "literal_click": frames["fs3"].filter(pl.col("context__tab") == 0)["impression_id"],
        }
        source_records = [item for item in report["records"] if item["source"] == source]
        if len(source_records) != 73:
            raise ValueError(f"T-{source} prediction count is not 73")
        for item in source_records:
            path = Path(item["path"])
            if _sha256(path) != item["prediction_sha256"]:
                raise ValueError("Prediction checksum mismatch")
            prediction = pl.read_parquet(path)
            validate_impression_ids(prediction, "prediction")
            if prediction.columns != ["impression_id", "score"] or not prediction["impression_id"].equals(expected_ids[item["task"]]):
                raise ValueError("Prediction coverage or schema mismatch")
            values = prediction["score"].to_numpy()
            if not np.isfinite(values).all() or np.any((values < 0) | (values > 1)):
                raise ValueError("Prediction probability integrity failed")
        checks[source] = {
            "rows": EXPECTED_ROWS[source], "feature_stages": 3, "prediction_artifacts": 73,
            "task_eligible_rows": {task: ids.len() for task, ids in expected_ids.items()},
            "feature_schema_matches_training": True, "feature_nesting": True,
            "sequence_positions": 20, "sequence_alignment": True, "sequence_chronology": True,
            "prediction_alignment": True, "probability_range": True,
        }
    for reference in report["rq3_baseline_references"]:
        path = Path(reference["referenced_path"])
        if _sha256(path) != reference["prediction_sha256"]:
            raise ValueError("RQ3 baseline reference checksum mismatch")
    audit = {
        "version": FINAL_VERSION, "status": "PASS", "populations": checks,
        "feature_artifacts": 8, "prediction_artifacts": 146, "rq3_checksum_references": 2,
        "outcome_columns_in_features_or_predictions": False,
        "randomized_history_used": False, "metrics_computed": False,
        "runtime_seconds": time.perf_counter() - started,
        "peak_rss_mib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
    }
    (prediction_root / "integration_audit.json").write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n")
    return audit


def build_m6_1_manifest(feature_report: dict[str, Any], prediction_report: dict[str, Any],
                        preflight: dict[str, Any] | None = None,
                        integration_audit: dict[str, Any] | None = None) -> dict[str, Any]:
    paths = get_artifact_paths()
    readiness_path = paths.processed / "prefinal" / "v1" / "readiness_manifest.json"
    readiness = json.loads(readiness_path.read_text())
    manifest = {
        "version": FINAL_VERSION, "required_tags": ["week0-protocol-v1", "m2-features-v1", "m4-candidates-v1", "m5-prefinal-v1"],
        "preflight": preflight or verify_m6_1_preflight(),
        "prefinal_readiness_sha256": _sha256(readiness_path), "candidate_count": readiness["candidate_count"],
        "features": feature_report, "predictions": prediction_report,
        "integration_audit": integration_audit,
        "outcomes": {"target_labels_accessed": False, "labels_materialized": False,
                     "randomized_validation_accessed": False, "randomized_history_accessed": False,
                     "rolling_normal_history_for_features": [FINAL_START, FINAL_HISTORY_END]},
        "metrics_computed": False, "selection_or_tuning_performed": False,
        "protocol_hashes": readiness["protocol_hashes"],
    }
    destination = paths.predictions / "final" / "v1" / "m6_1_manifest.json"
    destination.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return manifest
