"""M5.3 contender freeze and final-evaluation readiness audit."""

from __future__ import annotations

import hashlib
import json
import tomllib
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl

from kuairand_pure.data.alignment import validate_impression_ids
from kuairand_pure.final_evaluation import FinalEvaluationRequest, final_feature_contract
from kuairand_pure.inference import (
    candidate_stat_matrix,
    contender_set,
    paired_user_bootstrap,
)
from kuairand_pure.paths import get_artifact_paths


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _prediction_paths(root: Path) -> dict[str, Path]:
    paths = {
        "r0": root / "r0" / "long_view" / "predictions.parquet",
        **{
            f"fs{feature}_{depth}": root / f"fs{feature}_{depth}" / "long_view" / "predictions.parquet"
            for feature in (1, 2, 3) for depth in ("shallow", "deep")
        },
    }
    neural = {
        "shared_bottom": "shared_bottom",
        "mmoe_full": "mmoe_full",
        "mmoe_dense_only": "mmoe_dense_only",
        "mmoe_no_history": "mmoe_no_history",
        "mmoe_no_aggregates": "mmoe_no_aggregates",
    }
    paths.update({
        candidate: root / "neural" / directory / "seed_0" / "long_view" / "predictions.parquet"
        for candidate, directory in neural.items()
    })
    return paths


def freeze_normal_validation_contenders(
    *,
    processed_root: Path | None = None,
    prediction_root: Path | None = None,
    output_root: Path | None = None,
) -> dict[str, Any]:
    paths = get_artifact_paths()
    processed_root = processed_root or paths.processed
    prediction_root = prediction_root or paths.predictions / "validation" / "v1"
    output_root = output_root or processed_root / "prefinal" / "v1"
    labels = pl.read_parquet(
        processed_root / "validation" / "v1" / "fs3" / "normal_validation" / "labels.parquet"
    )
    validate_impression_ids(labels, "normal-validation labels")
    candidates: dict[str, np.ndarray] = {}
    prediction_hashes = {}
    for candidate, path in _prediction_paths(prediction_root).items():
        frame = pl.read_parquet(path)
        validate_impression_ids(frame, f"{candidate} predictions")
        if not frame["impression_id"].equals(labels["impression_id"]):
            raise ValueError(f"{candidate} validation predictions are misaligned")
        candidates[candidate] = frame["score"].to_numpy()
        prediction_hashes[candidate] = _sha256(path)
    names, users, credits, pairs = candidate_stat_matrix(
        labels["user_id"].to_numpy(), labels["long_view"].to_numpy(), candidates
    )
    eligible = pairs > 0
    point = credits[:, eligible].sum(axis=1) / pairs[eligible].sum()
    bootstrap = paired_user_bootstrap(
        credits[:, eligible], pairs[eligible], replicates=10_000, seed=0, batch_size=100
    )
    contenders, intervals = contender_set(
        names, point, bootstrap, validation_best="fs1_deep", minimum_top=3
    )
    result = {
        "version": "m5-prefinal-v1",
        "source": "authorized_normal_validation_only",
        "dates": [20220430, 20220502],
        "bootstrap_replicates": 10_000,
        "bootstrap_seed": 0,
        "confidence_interval": "two_sided_percentile_2.5_97.5",
        "validation_best": "fs1_deep",
        "minimum_top_three": True,
        "candidates": names,
        "point_pair_weighted_auc": {name: float(point[index]) for index, name in enumerate(names)},
        "paired_difference_intervals_vs_fs1_deep": {
            name: list(intervals[name]) for name in names
        },
        "contenders": contenders,
        "prediction_sha256": prediction_hashes,
        "held_out_outcomes_accessed": False,
    }
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "contender_manifest.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n"
    )
    return result


def _verify_candidate_artifacts(candidate_manifest: dict[str, Any]) -> list[dict[str, Any]]:
    verified = []
    for candidate in candidate_manifest["candidates"]:
        entry = {"candidate": candidate["candidate"], "model_family": candidate["model_family"]}
        if candidate["model_family"] == "lightgbm":
            bundles = {}
            for task, item in candidate["model_bundles"].items():
                directory = Path(item["path"])
                if _sha256(directory / "model.txt") != item["model_sha256"]:
                    raise ValueError("Frozen LightGBM model checksum mismatch")
                if _sha256(directory / "adapter.json") != item["adapter_sha256"]:
                    raise ValueError("Frozen LightGBM adapter checksum mismatch")
                bundles[task] = item
            entry["model_bundles"] = bundles
        elif candidate["model_family"] == "neural":
            preprocessing_path = Path(candidate["preprocessing_path"])
            if _sha256(preprocessing_path) != candidate["preprocessing_sha256"]:
                raise ValueError("Frozen neural preprocessing checksum mismatch")
            checkpoints = []
            for item in candidate["checkpoints"]:
                path = Path(item["bundle"]) / "checkpoint.pt"
                if _sha256(path) != item["checkpoint_sha256"]:
                    raise ValueError("Frozen neural checkpoint checksum mismatch")
                checkpoints.append(item)
            entry["checkpoints"] = checkpoints
            entry["preprocessing_path"] = candidate["preprocessing_path"]
            entry["preprocessing_sha256"] = candidate["preprocessing_sha256"]
            entry["primary_seed"] = 0
            entry["robustness_seeds"] = [1, 2]
        else:
            entry["scoring_rule"] = candidate["scoring_rule"]
        verified.append(entry)
    if len(verified) != 12:
        raise ValueError("Frozen candidate population is not exactly 12")
    return verified


def build_readiness_manifest(
    contenders: dict[str, Any],
    *,
    output_root: Path | None = None,
) -> dict[str, Any]:
    paths = get_artifact_paths()
    output_root = output_root or paths.processed / "prefinal" / "v1"
    candidate_path = paths.predictions / "validation" / "v1" / "candidate_manifest.json"
    candidate_manifest = json.loads(candidate_path.read_text())
    verified_candidates = _verify_candidate_artifacts(candidate_manifest)
    rq3_report_path = paths.models / "lightgbm" / "rq3" / "v1" / "training_report.json"
    rq3_report = json.loads(rq3_report_path.read_text())
    if rq3_report.get("model_count") != 3 or not rq3_report.get("baseline_preserved"):
        raise ValueError("RQ3 model freeze is incomplete")
    rq3_arms = [{
        "arm": "normal_only_baseline",
        "model_sha256": rq3_report["baseline_hashes"]["model_sha256"],
        "adapter_sha256": rq3_report["baseline_hashes"]["adapter_sha256"],
        "training_rows": 722_481,
        "training_dates": [20220412, 20220429],
        "source_indicator": "excluded",
        "boosting_rounds": 300,
        "feature_names": json.loads(
            (paths.models / "lightgbm" / "v1" / "fs1" / "deep" / "long_view" / "model_manifest.json").read_text()
        )["feature_names"],
        "feature_dtypes": json.loads(
            (paths.models / "lightgbm" / "v1" / "fs1" / "deep" / "long_view" / "adapter.json").read_text()
        )["feature_dtypes"],
    }]
    for record in rq3_report["records"]:
        directory = Path(record["artifact_dir"])
        if _sha256(directory / "model.txt") != record["model_sha256"]:
            raise ValueError("RQ3 model checksum mismatch")
        if _sha256(directory / "adapter.json") != record["adapter_sha256"]:
            raise ValueError("RQ3 adapter checksum mismatch")
        model_manifest = json.loads((directory / "model_manifest.json").read_text())
        rq3_arms.append({
            "arm": record["arm"], "model_sha256": record["model_sha256"],
            "adapter_sha256": record["adapter_sha256"],
            "vocabulary_sha256": record["vocabulary_sha256"],
            "training_rows": record["training_rows"],
            "positive_labels": record["positive_labels"],
            "boosting_rounds": record["boosting_rounds_completed"],
            "training_dates": model_manifest["training_metadata"]["training_dates"],
            "feature_names": model_manifest["feature_names"],
            "feature_dtypes": model_manifest["feature_dtypes"],
            "source_indicator": "included_random_equals_1" if record["arm"] == "joint" else "excluded",
            "weight_details": record["weight_details"],
        })
    with (_repo_root() / "configs" / "final_evaluation.toml").open("rb") as stream:
        evaluation_config = tomllib.load(stream)
    comparisons = [evaluation_config["comparisons"][f"comparison_{index}"] for index in range(1, 7)]
    manifest = {
        "version": "m5-prefinal-v1",
        "candidate_manifest_sha256": _sha256(candidate_path),
        "candidate_population": verified_candidates,
        "candidate_count": len(verified_candidates),
        "normal_validation_selected": candidate_manifest["selected_overall"],
        "contender_manifest": contenders,
        "rq3_arms": rq3_arms,
        "evaluation_populations": {
            "T-normal": final_feature_contract(FinalEvaluationRequest("normal")),
            "T-random": final_feature_contract(FinalEvaluationRequest("random")),
        },
        "metrics": evaluation_config["evaluation"],
        "rq2": evaluation_config["rq2"],
        "inference": evaluation_config["inference"],
        "six_predefined_comparisons": comparisons,
        "protected_outcome_columns": evaluation_config["access"]["protected_outcomes"],
        "access": evaluation_config["access"],
        "protocol_hashes": {
            "study_plan_v5": _sha256(_repo_root() / "docs" / "study_plan_v5.md"),
            "week0_amendment_v1": _sha256(_repo_root() / "docs" / "week0_amendment_v1.md"),
            "feature_manifest": _sha256(paths.processed / "m2" / "v1" / "feature_manifest.json"),
        },
        "code_and_config_hashes": {
            "final_evaluation_config": _sha256(_repo_root() / "configs" / "final_evaluation.toml"),
            "rq3_config": _sha256(_repo_root() / "configs" / "rq3.toml"),
            "lightgbm_config": _sha256(_repo_root() / "configs" / "lightgbm.toml"),
            "metrics_code": _sha256(_repo_root() / "src" / "kuairand_pure" / "metrics.py"),
            "inference_code": _sha256(_repo_root() / "src" / "kuairand_pure" / "inference.py"),
            "final_evaluation_code": _sha256(_repo_root() / "src" / "kuairand_pure" / "final_evaluation.py"),
        },
        "final_features_materialized": False,
        "held_out_outcomes_accessed": False,
        "final_results_computed": False,
    }
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "readiness_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )
    return manifest
