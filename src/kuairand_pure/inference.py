"""Frozen user-level inference and RQ2 cross-fitted regret routines."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np

from kuairand_pure.metrics import _concordance


@dataclass(frozen=True, slots=True)
class UserPairStatistics:
    user_ids: np.ndarray
    concordance: np.ndarray
    pairs: np.ndarray


def per_user_pair_statistics(user_ids, labels, scores) -> UserPairStatistics:
    users = np.asarray(user_ids)
    y = np.asarray(labels, dtype=int)
    predictions = np.asarray(scores, dtype=float)
    if not (users.ndim == y.ndim == predictions.ndim == 1 and users.size == y.size == predictions.size):
        raise ValueError("Aligned user, label, and score vectors are required")
    if not np.isin(y, (0, 1)).all():
        raise ValueError("Labels must be binary")
    if not np.isfinite(predictions).all():
        raise ValueError("Scores must be finite")
    unique = np.unique(users)
    credits = np.zeros(unique.size, dtype=np.float64)
    pairs = np.zeros(unique.size, dtype=np.int64)
    for index, user in enumerate(unique):
        mask = users == user
        credits[index], pairs[index] = _concordance(y[mask], predictions[mask])
    return UserPairStatistics(unique, credits, pairs)


def candidate_stat_matrix(
    user_ids,
    labels,
    candidate_scores: Mapping[str, np.ndarray],
) -> tuple[list[str], np.ndarray, np.ndarray, np.ndarray]:
    names = sorted(candidate_scores)
    if not names:
        raise ValueError("At least one candidate is required")
    statistics = [per_user_pair_statistics(user_ids, labels, candidate_scores[name]) for name in names]
    reference = statistics[0]
    for item in statistics[1:]:
        if not np.array_equal(item.user_ids, reference.user_ids) or not np.array_equal(item.pairs, reference.pairs):
            raise ValueError("Candidate per-user populations are not aligned")
    return names, reference.user_ids, np.vstack([item.concordance for item in statistics]), reference.pairs


def _auc(credits: np.ndarray, pairs: np.ndarray, weights: np.ndarray) -> np.ndarray:
    denominator = pairs @ weights
    if denominator <= 0:
        raise ValueError("Selected users contain no eligible positive-negative pairs")
    return (credits @ weights) / denominator


def percentile_interval(values: np.ndarray) -> tuple[float, float]:
    array = np.asarray(values, dtype=float)
    if array.size == 0 or not np.isfinite(array).all():
        raise ValueError("Bootstrap values must be finite and nonempty")
    low, high = np.quantile(array, [0.025, 0.975], method="linear")
    return float(low), float(high)


def tail_share_p_value(values: np.ndarray) -> float:
    array = np.asarray(values, dtype=float)
    if array.size == 0 or not np.isfinite(array).all():
        raise ValueError("Bootstrap differences must be finite and nonempty")
    return float(min(1.0, 2 * min(np.mean(array <= 0), np.mean(array >= 0))))


def paired_user_bootstrap(
    credits: np.ndarray,
    pairs: np.ndarray,
    *,
    replicates: int = 10_000,
    seed: int = 0,
    batch_size: int = 100,
) -> np.ndarray:
    """Return candidate AUCs under paired unique-user multinomial resampling."""
    matrix = np.asarray(credits, dtype=float)
    pair_counts = np.asarray(pairs, dtype=float)
    if matrix.ndim != 2 or pair_counts.shape != (matrix.shape[1],):
        raise ValueError("Candidate credits and pair counts are incompatible")
    if replicates <= 0 or batch_size <= 0:
        raise ValueError("Bootstrap sizes must be positive")
    rng = np.random.default_rng(seed)
    output = np.empty((matrix.shape[0], replicates), dtype=np.float64)
    probability = np.full(matrix.shape[1], 1 / matrix.shape[1])
    for start in range(0, replicates, batch_size):
        stop = min(start + batch_size, replicates)
        multiplicities = rng.multinomial(matrix.shape[1], probability, size=stop - start).T
        denominator = pair_counts @ multiplicities
        if np.any(denominator <= 0):
            raise ValueError("Bootstrap replicate contains no eligible pairs")
        output[:, start:stop] = (matrix @ multiplicities) / denominator
    return output


def contender_set(
    candidate_ids: Sequence[str],
    point_aucs: np.ndarray,
    bootstrap_aucs: np.ndarray,
    *,
    validation_best: str = "fs1_deep",
    minimum_top: int = 3,
) -> tuple[list[str], dict[str, tuple[float, float]]]:
    names = list(candidate_ids)
    if (
        validation_best not in names
        or bootstrap_aucs.ndim != 2
        or bootstrap_aucs.shape[0] != len(names)
        or np.asarray(point_aucs).shape != (len(names),)
    ):
        raise ValueError("Contender inputs are incompatible")
    best_index = names.index(validation_best)
    intervals: dict[str, tuple[float, float]] = {}
    selected = set()
    for index, name in enumerate(names):
        interval = percentile_interval(bootstrap_aucs[index] - bootstrap_aucs[best_index])
        intervals[name] = interval
        if interval[0] <= 0 <= interval[1]:
            selected.add(name)
    ranked = sorted(names, key=lambda name: (-float(point_aucs[names.index(name)]), name))
    selected.update(ranked[:minimum_top])
    return sorted(selected), intervals


def holm_adjust(p_values: Sequence[float]) -> np.ndarray:
    values = np.asarray(p_values, dtype=float)
    if values.ndim != 1 or values.size == 0 or np.any((values < 0) | (values > 1)):
        raise ValueError("Holm inputs must be probabilities")
    order = np.argsort(values, kind="stable")
    adjusted_sorted = np.maximum.accumulate((values.size - np.arange(values.size)) * values[order])
    adjusted = np.empty_like(values)
    adjusted[order] = np.minimum(adjusted_sorted, 1.0)
    return adjusted


def _split_masks(user_ids: np.ndarray, split_count: int, seed: int) -> list[tuple[np.ndarray, np.ndarray]]:
    rng = np.random.default_rng(seed)
    masks = []
    for _ in range(split_count):
        order = rng.permutation(user_ids.size)
        midpoint = user_ids.size // 2
        a = np.zeros(user_ids.size, dtype=bool)
        a[order[:midpoint]] = True
        masks.append((a, ~a))
    return masks


def cross_fitted_regret(
    candidate_ids: Sequence[str],
    credits: np.ndarray,
    pairs: np.ndarray,
    *,
    workflow_pick: str,
    contenders: Sequence[str],
    split_count: int = 50,
    seed: int = 0,
    multiplicities: np.ndarray | None = None,
) -> dict[str, float]:
    """Compute disjoint-half oracle and uniform-pick regret estimands."""
    names = list(candidate_ids)
    matrix = np.asarray(credits, dtype=float)
    pair_counts = np.asarray(pairs, dtype=float)
    if matrix.shape != (len(names), pair_counts.size):
        raise ValueError("Cross-fit candidate statistics are incompatible")
    if workflow_pick not in names or not set(contenders).issubset(names):
        raise ValueError("Cross-fit candidate identity is absent")
    weights = np.ones(pair_counts.size) if multiplicities is None else np.asarray(multiplicities, dtype=float)
    if weights.shape != pair_counts.shape or np.any(weights < 0):
        raise ValueError("Bootstrap user multiplicities are invalid")
    workflow_index = names.index(workflow_pick)
    contender_indices = [names.index(name) for name in contenders]
    values = {"workflow_regret": [], "random_pick_regret": [], "contender_pick_regret": []}
    for a, b in _split_masks(np.arange(pair_counts.size), split_count, seed):
        for selection, evaluation in ((a, b), (b, a)):
            selection_scores = _auc(matrix, pair_counts, weights * selection)
            best_value = np.max(selection_scores)
            tied = [names[i] for i, value in enumerate(selection_scores) if value == best_value]
            chosen = names.index(min(tied))
            evaluation_scores = _auc(matrix, pair_counts, weights * evaluation)
            reference = float(evaluation_scores[chosen])
            values["workflow_regret"].append(reference - evaluation_scores[workflow_index])
            values["random_pick_regret"].append(reference - float(evaluation_scores.mean()))
            values["contender_pick_regret"].append(
                reference - float(evaluation_scores[contender_indices].mean())
            )
    return {name: float(np.mean(items)) for name, items in values.items()}


def bootstrap_cross_fitted_regret(
    candidate_ids: Sequence[str],
    credits: np.ndarray,
    pairs: np.ndarray,
    *,
    workflow_pick: str,
    contenders: Sequence[str],
    replicates: int = 2_000,
    split_count: int = 50,
    seed: int = 0,
) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    user_count = np.asarray(pairs).size
    probability = np.full(user_count, 1 / user_count)
    output = {name: np.empty(replicates) for name in (
        "workflow_regret", "random_pick_regret", "contender_pick_regret"
    )}
    for replicate in range(replicates):
        for attempt in range(100):
            multiplicities = rng.multinomial(user_count, probability)
            try:
                result = cross_fitted_regret(
                    candidate_ids, credits, pairs, workflow_pick=workflow_pick,
                    contenders=contenders, split_count=split_count, seed=seed,
                    multiplicities=multiplicities,
                )
                break
            except ValueError as exc:
                if "no eligible" not in str(exc) or attempt == 99:
                    raise
        for name, value in result.items():
            output[name][replicate] = value
    return output
