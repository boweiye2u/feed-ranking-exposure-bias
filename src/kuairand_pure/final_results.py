"""M6.2 authorized final RQ1/RQ3 evaluation without model mutation."""

from __future__ import annotations

import hashlib
import json
import math
import resource
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl

from kuairand_pure.data.alignment import validate_impression_ids
from kuairand_pure.data.fs1 import _labels
from kuairand_pure.data.splits import SplitSpec
from kuairand_pure.final_predictions import audit_m6_1_outputs, verify_m6_1_preflight
from kuairand_pure.inference import holm_adjust, paired_user_bootstrap, percentile_interval, tail_share_p_value
from kuairand_pure.metrics import _concordance
from kuairand_pure.paths import get_artifact_paths

PAIR_CAP = 952
NDCG_K = 5
BOOTSTRAP_REPLICATES = 10_000
BOOTSTRAP_SEED = 0
VERSION = "m6.2-final-evaluation-v1"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


@dataclass(slots=True)
class PreparedPopulation:
    ids: np.ndarray
    users: np.ndarray
    labels: np.ndarray
    times: np.ndarray
    videos: np.ndarray
    order: np.ndarray
    starts: np.ndarray
    unique_users: np.ndarray
    pairs: np.ndarray


def prepare_population(frame: pl.DataFrame, label_name: str) -> PreparedPopulation:
    if frame.height == 0:
        raise ValueError("Cannot prepare an empty metric population")
    users = frame["user_id"].to_numpy()
    labels = frame[label_name].to_numpy().astype(int)
    if not np.isin(labels, (0, 1)).all():
        raise ValueError("Final outcomes must be binary")
    order = np.argsort(users, kind="stable")
    sorted_users = users[order]
    unique, starts = np.unique(sorted_users, return_index=True)
    ends = np.r_[starts[1:], len(order)]
    positives = np.add.reduceat(labels[order], starts).astype(np.int64)
    counts = ends - starts
    pairs = positives * (counts - positives)
    return PreparedPopulation(
        frame["impression_id"].to_numpy(), users, labels, frame["time_ms"].to_numpy(),
        frame["video_id"].to_numpy(), order, starts, unique, pairs,
    )


def per_user_credits(population: PreparedPopulation, scores: np.ndarray) -> np.ndarray:
    values = np.asarray(scores, dtype=float)
    if values.shape != population.labels.shape or not np.isfinite(values).all():
        raise ValueError("Final scores are not finite and aligned")
    sorted_y = population.labels[population.order]
    sorted_scores = values[population.order]
    ends = np.r_[population.starts[1:], len(values)]
    credits = np.zeros(population.unique_users.size, dtype=np.float64)
    for index, (start, end) in enumerate(zip(population.starts, ends)):
        credits[index], observed_pairs = _concordance(sorted_y[start:end], sorted_scores[start:end])
        if observed_pairs != population.pairs[index]:
            raise ValueError("Per-user pair accounting drifted")
    return credits


def evaluate_prepared(population: PreparedPopulation, scores: np.ndarray) -> dict[str, Any]:
    values = np.asarray(scores, dtype=float)
    credits = per_user_credits(population, values)
    eligible = population.pairs > 0
    if not eligible.any():
        raise ValueError("No user has both classes")
    aucs = credits[eligible] / population.pairs[eligible]
    capped = np.minimum(population.pairs[eligible], PAIR_CAP)
    pooled_credit, pooled_pairs = _concordance(population.labels, values)
    clipped = np.clip(values, 1e-15, 1 - 1e-15)
    loss = -np.mean(population.labels * np.log(clipped) + (1 - population.labels) * np.log(1 - clipped))
    stable = np.arange(values.size)
    sorted_labels = population.labels[population.order]
    sorted_scores = values[population.order]
    sorted_times = population.times[population.order]
    sorted_videos = population.videos[population.order]
    sorted_stable = stable[population.order]
    ends = np.r_[population.starts[1:], len(values)]
    discounts = 1 / np.log2(np.arange(2, NDCG_K + 2))
    ndcg: list[float] = []
    for start, end in zip(population.starts, ends):
        y = sorted_labels[start:end]
        if y.size < NDCG_K or y.sum() == 0:
            continue
        ranking = np.lexsort((sorted_stable[start:end], sorted_videos[start:end],
                              sorted_times[start:end], -sorted_scores[start:end]))
        dcg = float(np.sum(y[ranking[:NDCG_K]] * discounts))
        ndcg.append(dcg / float(np.sum(discounts[:min(int(y.sum()), NDCG_K)])))
    positives = int(population.labels.sum())
    return {
        "pair_weighted_auc": float(credits.sum() / population.pairs.sum()),
        "per_user_average_auc": float(aucs.mean()),
        "capped_pair_weighted_auc": float(np.sum(aucs * capped) / capped.sum()),
        "pooled_auc": float(pooled_credit / pooled_pairs), "log_loss": float(loss),
        "ndcg_at_5": float(np.mean(ndcg)), "impressions": int(values.size),
        "positives": positives, "prevalence": float(positives / values.size),
        "eligible_auc_users": int(eligible.sum()), "single_class_users": int((~eligible).sum()),
        "total_pairs": int(population.pairs.sum()), "capped_total_weight": int(capped.sum()),
        "ndcg_eligible_users": len(ndcg), "ndcg_excluded_users": int(len(population.unique_users) - len(ndcg)),
    }


def load_final_outcomes(*, data_dir: Path | None = None) -> dict[str, pl.DataFrame]:
    """The sole M6.2 loader authorized to materialize May 3-8 target outcomes."""
    paths = get_artifact_paths()
    data_dir = data_dir or paths.raw / "KuaiRand-Pure" / "data"
    output = {}
    for source in ("normal", "random"):
        spec = SplitSpec(f"T-{source}", source, 20220503, 20220508, True)
        frame = _labels(data_dir, spec, workflow="final_evaluation", purpose="evaluation").sort("impression_id").collect()
        validate_impression_ids(frame, f"T-{source} final outcomes")
        expected = 93_201 if source == "normal" else 637_492
        if frame.height != expected:
            raise ValueError(f"T-{source} outcome row count drifted")
        output[source] = frame
    return output


def _task_frame(labels: pl.DataFrame, task: str) -> tuple[pl.DataFrame, str]:
    if task == "long_view": return labels, "long_view"
    if task == "like": return labels, "is_like"
    if task == "valid_play": return labels.filter(pl.col("click_semantics") == "valid_play"), "is_click"
    if task == "literal_click": return labels.filter(pl.col("click_semantics") == "literal_click"), "is_click"
    raise ValueError(f"Unknown task {task}")


def run_m6_2(*, output_root: Path | None = None) -> dict[str, Any]:
    started = time.perf_counter()
    paths = get_artifact_paths()
    output_root = output_root or paths.root / "evaluation" / "final" / "v1"
    # Every frozen check is deliberately completed before outcome loading.
    preflight = verify_m6_1_preflight()
    integration = audit_m6_1_outputs()
    m6_manifest_path = paths.predictions / "final" / "v1" / "m6_1_manifest.json"
    m6_manifest_sha = _sha256(m6_manifest_path)
    prediction_report_path = paths.predictions / "final" / "v1" / "prediction_report.json"
    prediction_report = json.loads(prediction_report_path.read_text())
    for item in prediction_report["records"]:
        if _sha256(Path(item["path"])) != item["prediction_sha256"]:
            raise ValueError("M6.1 prediction changed before final outcome access")
    outcomes = load_final_outcomes()
    output_root.mkdir(parents=True, exist_ok=True)

    prepared: dict[tuple[str, str], PreparedPopulation] = {}
    population_diagnostics: dict[str, Any] = {}
    for source, labels in outcomes.items():
        diagnostics = {}
        for task in ("long_view", "like", "valid_play", "literal_click"):
            frame, label_name = _task_frame(labels, task)
            if frame.height == 0:
                diagnostics[task] = {"status": "not_estimable", "reason": "no_eligible_impressions", "impressions": 0}
                continue
            pop = prepare_population(frame, label_name)
            prepared[(source, task)] = pop
            top_n = math.ceil(0.10 * pop.unique_users.size)
            pair_order = np.sort(pop.pairs)[::-1]
            diagnostics[task] = {
                "status": "estimable", "impressions": frame.height, "positives": int(pop.labels.sum()),
                "prevalence": float(pop.labels.mean()), "users": int(pop.unique_users.size),
                "qualifying_users": int((pop.pairs > 0).sum()), "single_class_users": int((pop.pairs == 0).sum()),
                "positive_negative_pairs": int(pop.pairs.sum()), "top_10_percent_user_count": top_n,
                "top_10_percent_pair_share": float(pair_order[:top_n].sum() / pop.pairs.sum()) if pop.pairs.sum() else None,
            }
        population_diagnostics[source] = diagnostics

    metric_records = []
    score_cache: dict[tuple[str, str, str, int | None], np.ndarray] = {}
    for item in prediction_report["records"]:
        source, task = item["source"], item["task"]
        candidate = item["candidate"]
        seed = item.get("seed")
        prediction = pl.read_parquet(item["path"])
        task_labels, _ = _task_frame(outcomes[source], task)
        if not prediction["impression_id"].equals(task_labels["impression_id"]):
            raise ValueError("Final prediction and outcome IDs differ by row")
        if prediction.height == 0:
            metric_records.append({"source": source, "candidate": candidate, "task": task,
                                   "seed": seed, "status": "not_estimable", "reason": "no_tab_0_impressions"})
            continue
        scores = prediction["score"].to_numpy()
        score_cache[(source, candidate, task, seed)] = scores
        metric_records.append({"source": source, "candidate": candidate, "task": task,
                               "seed": seed, "status": "estimated",
                               "metrics": evaluate_prepared(prepared[(source, task)], scores),
                               "prediction_sha256": item["prediction_sha256"]})

    # RQ1 primary population: seed 0 for neural candidates, every frozen non-neural candidate.
    rq1 = [r for r in metric_records if r["task"] == "long_view" and r["status"] == "estimated"
           and not r["candidate"].startswith("rq3_") and (r["seed"] in (None, 0))]
    if len(rq1) != 24:  # 12 candidates x two exposure populations
        raise ValueError("RQ1 frozen candidate coverage is not 12 per population")
    neural_robustness = [r for r in metric_records if r["task"] == "long_view" and r["seed"] is not None]
    neural_summary = []
    for source in ("normal", "random"):
        for candidate in sorted({r["candidate"] for r in neural_robustness}):
            rows = sorted((r for r in neural_robustness if r["source"] == source and r["candidate"] == candidate), key=lambda r: r["seed"])
            values = [r["metrics"]["pair_weighted_auc"] for r in rows]
            neural_summary.append({"source": source, "candidate": candidate, "seeds": [0, 1, 2],
                                   "pair_weighted_auc_mean": float(np.mean(values)), "minimum": min(values),
                                   "maximum": max(values), "range": max(values) - min(values)})

    # RQ3 is evaluated only on the identical T-random long-view population.
    baseline_scores = score_cache[("random", "fs1_deep", "long_view", None)]
    rq3_ids = ("normal_only_baseline", "marginal_reweighting", "joint", "random_only")
    rq3_scores = {
        "normal_only_baseline": baseline_scores,
        **{arm: score_cache[("random", f"rq3_{arm}", "long_view", None)]
           for arm in rq3_ids[1:]},
    }
    random_pop = prepared[("random", "long_view")]
    rq3 = []
    baseline_metric = evaluate_prepared(random_pop, baseline_scores)
    training_rows = {"normal_only_baseline": 722481, "marginal_reweighting": 722481,
                     "joint": 1060414, "random_only": 337933}
    for arm in rq3_ids:
        metrics = baseline_metric if arm == "normal_only_baseline" else evaluate_prepared(random_pop, rq3_scores[arm])
        rq3.append({"arm": arm, "training_rows": training_rows[arm], "metrics": metrics,
                    "pair_weighted_auc_difference_vs_normal_only": metrics["pair_weighted_auc"] - baseline_metric["pair_weighted_auc"],
                    "weighting_ess": 152212.849206 if arm == "marginal_reweighting" else None})

    comparison_specs = [
        ("mmoe_full_seed0_minus_fs1_deep", ("mmoe_full", 0), ("fs1_deep", None)),
        ("mmoe_full_seed0_minus_shared_bottom_seed0", ("mmoe_full", 0), ("shared_bottom", 0)),
        ("mmoe_dense_only_seed0_minus_fs3_deep", ("mmoe_dense_only", 0), ("fs3_deep", None)),
        ("rq3_marginal_reweighting_minus_fs1_deep", ("rq3_marginal_reweighting", None), ("fs1_deep", None)),
        ("rq3_joint_minus_fs1_deep", ("rq3_joint", None), ("fs1_deep", None)),
        ("rq3_random_only_minus_fs1_deep", ("rq3_random_only", None), ("fs1_deep", None)),
    ]
    unique_keys = sorted({key for _, left, right in comparison_specs for key in (left, right)})
    credits = np.vstack([per_user_credits(random_pop, score_cache[("random", key[0], "long_view", key[1])]) for key in unique_keys])
    draws = paired_user_bootstrap(credits, random_pop.pairs, replicates=BOOTSTRAP_REPLICATES, seed=BOOTSTRAP_SEED, batch_size=100)
    point = credits.sum(axis=1) / random_pop.pairs.sum()
    comparisons = []
    p_values = []
    for name, left, right in comparison_specs:
        li, ri = unique_keys.index(left), unique_keys.index(right)
        differences = draws[li] - draws[ri]
        p = tail_share_p_value(differences); p_values.append(p)
        comparisons.append({"comparison": name, "left": left[0], "right": right[0],
                            "observed_pair_weighted_auc_difference": float(point[li] - point[ri]),
                            "percentile_95_ci": list(percentile_interval(differences)),
                            "tail_share_approximate_p_value": p})
    adjusted = holm_adjust(p_values)
    for item, value in zip(comparisons, adjusted): item["holm_adjusted_p_value"] = float(value)

    result = {
        "version": VERSION, "preflight": preflight, "m6_1_integration": integration,
        "m6_1_manifest_sha256": m6_manifest_sha, "prediction_report_sha256": _sha256(prediction_report_path),
        "population_diagnostics": population_diagnostics, "metric_records": metric_records,
        "rq1_long_view_primary": rq1, "neural_seed_robustness": neural_robustness,
        "neural_three_seed_summary": neural_summary, "rq3": rq3, "paired_comparisons": comparisons,
        "inference": {"replicates": BOOTSTRAP_REPLICATES, "seed": BOOTSTRAP_SEED,
                      "unit": "unique_user_with_multinomial_multiplicity", "paired": True,
                      "ci": "two_sided_percentile", "p_value": "two_sided_tail_share_approximation",
                      "holm": "standard_step_down_six_comparisons"},
        "rq2_executed": False, "models_retrained": False, "predictions_modified": False,
        "runtime_seconds": time.perf_counter() - started,
        "peak_rss_mib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
    }
    result_path = output_root / "m6_2_results.json"
    result_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    # Compact machine-readable tables.
    pl.DataFrame([{"source": r["source"], "candidate": r["candidate"], **r["metrics"]} for r in rq1]).write_parquet(output_root / "rq1_long_view.parquet")
    pl.DataFrame([{"arm": r["arm"], "training_rows": r["training_rows"], "difference_vs_baseline": r["pair_weighted_auc_difference_vs_normal_only"], **r["metrics"]} for r in rq3]).write_parquet(output_root / "rq3.parquet")
    pl.DataFrame(comparisons).write_json(output_root / "six_comparisons.json")
    manifest = {"version": VERSION, "inputs": {"m6_1_manifest_sha256": m6_manifest_sha,
                "prediction_report_sha256": _sha256(prediction_report_path)},
                "outputs": {name: _sha256(output_root / name) for name in
                            ("m6_2_results.json", "rq1_long_view.parquet", "rq3.parquet", "six_comparisons.json")},
                "outcomes_authorized": [20220503, 20220508], "rq2_executed": False}
    (output_root / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return result
