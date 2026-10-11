"""M6.3 frozen cross-fitted model-selection regret analysis."""

from __future__ import annotations

import hashlib
import json
import resource
import time
from collections import Counter
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import polars as pl

from kuairand_pure.final_predictions import audit_m6_1_outputs, verify_m6_1_preflight
from kuairand_pure.final_results import load_final_outcomes, per_user_credits, prepare_population
from kuairand_pure.inference import _split_masks, percentile_interval
from kuairand_pure.paths import get_artifact_paths

VERSION = "m6.3-rq2-v1"
REPLICATES = 2_000
SPLIT_COUNT = 50
SEED = 0
WORKFLOW_PICK = "fs1_deep"
CONTENDERS = ("fs1_deep", "fs1_shallow", "mmoe_no_aggregates")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def deterministic_half_masks(user_count: int, *, split_count: int = SPLIT_COUNT,
                             seed: int = SEED) -> np.ndarray:
    pairs = _split_masks(np.arange(user_count), split_count, seed)
    masks = np.stack([mask for pair in pairs for mask in pair])
    if masks.shape != (2 * split_count, user_count) or np.any(masks[0::2] & masks[1::2]):
        raise ValueError("Cross-fit user halves are not disjoint")
    if np.any(~(masks[0::2] | masks[1::2])):
        raise ValueError("Cross-fit user halves do not cover all users")
    return masks


def _population_indices(names: Sequence[str], population: Sequence[str]) -> np.ndarray:
    if sorted(set(names)) != list(names):
        raise ValueError("Candidate IDs must be unique and lexicographically ordered")
    if not set(population).issubset(names) or WORKFLOW_PICK not in population:
        raise ValueError("Frozen RQ2 population is incompatible")
    return np.array([names.index(name) for name in population], dtype=int)


def crossfit_estimates(
    names: Sequence[str], credits: np.ndarray, pairs: np.ndarray,
    multiplicities: np.ndarray, *, populations: dict[str, Sequence[str]],
    split_count: int = SPLIT_COUNT, seed: int = SEED,
) -> dict[str, dict[str, Any]]:
    """Vectorized exact frozen estimator for one or many bootstrap replicates."""
    candidate_ids = list(names)
    matrix = np.asarray(credits, dtype=np.float64)
    pair_counts = np.asarray(pairs, dtype=np.float64)
    weights = np.asarray(multiplicities, dtype=np.float64)
    if matrix.shape != (len(candidate_ids), pair_counts.size):
        raise ValueError("Candidate per-user statistics are incompatible")
    if weights.ndim == 1: weights = weights[None, :]
    if weights.shape[1] != pair_counts.size or np.any(weights < 0):
        raise ValueError("Bootstrap multiplicities are incompatible")
    masks = deterministic_half_masks(pair_counts.size, split_count=split_count, seed=seed)
    total_credit = weights @ matrix.T
    total_pairs = weights @ pair_counts
    if np.any(total_pairs <= 0): raise ValueError("Bootstrap replicate has no eligible pairs")
    accumulator = {
        key: {"reference_auc": np.zeros(weights.shape[0]), "workflow_auc": np.zeros(weights.shape[0]),
              "normal_selection_regret": np.zeros(weights.shape[0]), "random_pick_regret": np.zeros(weights.shape[0]),
              "selection_counts": Counter()}
        for key in populations
    }
    for mask in masks:
        selected_credit = weights @ (matrix * mask).T
        selected_pairs = weights @ (pair_counts * mask)
        evaluation_credit = total_credit - selected_credit
        evaluation_pairs = total_pairs - selected_pairs
        if np.any(selected_pairs <= 0) or np.any(evaluation_pairs <= 0):
            raise ValueError("Cross-fit half has no eligible pairs")
        selection_auc = selected_credit / selected_pairs[:, None]
        evaluation_auc = evaluation_credit / evaluation_pairs[:, None]
        for key, population in populations.items():
            indices = _population_indices(candidate_ids, population)
            local_choice = np.argmax(selection_auc[:, indices], axis=1)
            chosen = indices[local_choice]
            reference = evaluation_auc[np.arange(weights.shape[0]), chosen]
            workflow = evaluation_auc[:, candidate_ids.index(WORKFLOW_PICK)]
            random_pick = evaluation_auc[:, indices].mean(axis=1)
            values = accumulator[key]
            values["reference_auc"] += reference
            values["workflow_auc"] += workflow
            values["normal_selection_regret"] += reference - workflow
            values["random_pick_regret"] += reference - random_pick
            if weights.shape[0] == 1:
                values["selection_counts"].update(candidate_ids[index] for index in chosen)
    denominator = masks.shape[0]
    return {key: {
        **{metric: value / denominator for metric, value in values.items() if metric != "selection_counts"},
        "selection_counts": dict(values["selection_counts"]),
    } for key, values in accumulator.items()}


def bootstrap_crossfit(
    names: Sequence[str], credits: np.ndarray, pairs: np.ndarray,
    *, populations: dict[str, Sequence[str]], replicates: int = REPLICATES,
    split_count: int = SPLIT_COUNT, seed: int = SEED,
) -> dict[str, dict[str, np.ndarray]]:
    rng = np.random.default_rng(seed)
    user_count = len(pairs)
    multiplicities = rng.multinomial(user_count, np.full(user_count, 1 / user_count), size=replicates)
    result = crossfit_estimates(names, credits, pairs, multiplicities,
                                populations=populations, split_count=split_count, seed=seed)
    return {key: {metric: value for metric, value in values.items() if metric != "selection_counts"}
            for key, values in result.items()}


def _verify_m6_2(paths) -> dict[str, str]:
    root = paths.root / "evaluation" / "final" / "v1"
    manifest_path = root / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    for name, expected in manifest["outputs"].items():
        if _sha256(root / name) != expected:
            raise ValueError("M6.2 aggregate result changed before RQ2")
    if manifest.get("rq2_executed") is not False:
        raise ValueError("M6.2 manifest unexpectedly reports RQ2 execution")
    return {"manifest": _sha256(manifest_path), **manifest["outputs"]}


def run_m6_3(*, output_root: Path | None = None, benchmark_replicates: int = 20) -> dict[str, Any]:
    started = time.perf_counter()
    paths = get_artifact_paths()
    output_root = output_root or paths.root / "evaluation" / "final" / "v1"
    preflight = verify_m6_1_preflight()
    m6_1_audit = audit_m6_1_outputs()
    m6_2_hashes = _verify_m6_2(paths)
    contender_path = paths.processed / "prefinal" / "v1" / "contender_manifest.json"
    contender_manifest = json.loads(contender_path.read_text())
    if tuple(contender_manifest["contenders"]) != CONTENDERS:
        raise ValueError("Frozen contender set changed")

    # Authorized final labels are loaded only after every frozen input verifies.
    labels = load_final_outcomes()["random"]
    population = prepare_population(labels, "long_view")
    report = json.loads((paths.predictions / "final" / "v1" / "prediction_report.json").read_text())
    candidate_records = [item for item in report["records"] if item["source"] == "random"
                         and item["task"] == "long_view" and item["model_family"] in {"r0", "lightgbm", "neural"}
                         and item.get("seed") in (None, 0)]
    names = sorted(item["candidate"] for item in candidate_records)
    if len(names) != 12 or len(set(names)) != 12 or WORKFLOW_PICK not in names:
        raise ValueError("Frozen 12-candidate RQ2 population is incomplete")
    by_name = {item["candidate"]: item for item in candidate_records}
    credit_rows = []
    for name in names:
        prediction = pl.read_parquet(by_name[name]["path"])
        if not prediction["impression_id"].equals(labels["impression_id"]):
            raise ValueError(f"{name} is misaligned with T-random outcomes")
        credit_rows.append(per_user_credits(population, prediction["score"].to_numpy()))
    credits = np.vstack(credit_rows)
    sorted_labels = population.labels[population.order]
    ends = np.r_[population.starts[1:], len(population.order)]
    positives = np.add.reduceat(sorted_labels, population.starts).astype(np.int64)
    negatives = (ends - population.starts - positives).astype(np.int64)
    statistics = pl.DataFrame({"user_id": population.unique_users, "positive_count": positives,
        "negative_count": negatives, "pair_count": population.pairs,
        **{f"concordance__{name}": credits[index] for index, name in enumerate(names)}})
    statistics_path = output_root / "rq2_user_statistics.parquet"
    statistics.write_parquet(statistics_path)

    populations = {"all_candidates": tuple(names), "contenders": CONTENDERS}
    point = crossfit_estimates(names, credits, population.pairs, np.ones(len(population.pairs)), populations=populations)
    benchmark_start = time.perf_counter()
    benchmark = bootstrap_crossfit(names, credits, population.pairs, populations=populations,
                                   replicates=benchmark_replicates)
    benchmark_seconds = time.perf_counter() - benchmark_start
    estimated_seconds = benchmark_seconds * REPLICATES / benchmark_replicates
    if estimated_seconds > 7200:
        raise RuntimeError(f"Projected frozen RQ2 bootstrap runtime {estimated_seconds:.1f}s exceeds two hours")
    bootstrap_start = time.perf_counter()
    draws = bootstrap_crossfit(names, credits, population.pairs, populations=populations)
    bootstrap_seconds = time.perf_counter() - bootstrap_start
    results = {}
    for key in populations:
        results[key] = {
            "candidate_population": list(populations[key]), "candidate_count": len(populations[key]),
            "cross_fitted_reference_auc": float(point[key]["reference_auc"][0]),
            "frozen_fs1_deep_evaluation_auc": float(point[key]["workflow_auc"][0]),
            "normal_selection_regret": float(point[key]["normal_selection_regret"][0]),
            "normal_selection_regret_percentile_95_ci": list(percentile_interval(draws[key]["normal_selection_regret"])),
            "random_pick_regret": float(point[key]["random_pick_regret"][0]),
            "random_pick_regret_percentile_95_ci": list(percentile_interval(draws[key]["random_pick_regret"])),
            "reference_selection_frequency": point[key]["selection_counts"],
        }
    output = {
        "version": VERSION, "preflight": preflight, "m6_1_integration": m6_1_audit,
        "m6_2_hashes": m6_2_hashes, "contender_manifest_sha256": _sha256(contender_path),
        "population": {"impressions": len(population.labels), "users": len(population.unique_users),
                       "eligible_users": int((population.pairs > 0).sum()), "zero_pair_users": int((population.pairs == 0).sum()),
                       "pairs": int(population.pairs.sum())},
        "candidate_ids": names, "workflow_pick": WORKFLOW_PICK, "results": results,
        "bootstrap": {"replicates": REPLICATES, "split_count": SPLIT_COUNT, "directions_per_replicate": 100,
                      "seed": SEED, "unique_user_multinomial_multiplicities": True,
                      "selection_recomputed_each_replicate": True, "ci": "two_sided_percentile"},
        "benchmark": {"replicates": benchmark_replicates, "seconds": benchmark_seconds,
                      "estimated_full_seconds": estimated_seconds, "temporary_results_used": False},
        "bootstrap_runtime_seconds": bootstrap_seconds, "runtime_seconds": time.perf_counter() - started,
        "peak_rss_mib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
        "rq3_candidates_included": False, "predictions_regenerated": False,
        "models_retrained_or_reselected": False,
    }
    result_path = output_root / "m6_3_rq2_results.json"
    result_path.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n")
    frequency_rows = [{"population": key, "candidate": name, "selected_halves": results[key]["reference_selection_frequency"].get(name, 0)}
                      for key in populations for name in populations[key]]
    frequency_path = output_root / "rq2_selection_frequencies.parquet"
    pl.DataFrame(frequency_rows).write_parquet(frequency_path)
    manifest = {"version": VERSION, "input_hashes": m6_2_hashes,
                "outputs": {"results": _sha256(result_path), "user_statistics": _sha256(statistics_path),
                            "selection_frequencies": _sha256(frequency_path)},
                "replicates": REPLICATES, "split_count": SPLIT_COUNT, "seed": SEED}
    manifest_path = output_root / "m6_3_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return output
