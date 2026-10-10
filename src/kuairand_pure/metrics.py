"""Frozen Week-0 ranking metrics."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt


@dataclass(frozen=True, slots=True)
class WithinUserAUC:
    pair_weighted_auc: float
    per_user_average_auc: float
    capped_pair_weighted_auc: float
    eligible_users: int
    single_class_users: int
    total_pairs: int
    capped_total_weight: int


def _concordance(
    labels: npt.NDArray[np.int_], scores: npt.NDArray[np.float64]
) -> tuple[float, int]:
    positives = scores[labels == 1]
    negatives = np.sort(scores[labels == 0])
    pair_count = int(positives.size * negatives.size)
    if pair_count == 0:
        return 0.0, 0
    lower = np.searchsorted(negatives, positives, side="left")
    upper = np.searchsorted(negatives, positives, side="right")
    concordance = float(np.sum(lower + 0.5 * (upper - lower)))
    return concordance, pair_count


def within_user_auc(
    user_ids: npt.ArrayLike,
    labels: npt.ArrayLike,
    scores: npt.ArrayLike,
    *,
    pair_cap: int = 952,
) -> WithinUserAUC:
    """Compute uncapped primary, equal-user, and capped sensitivity AUCs."""
    users = np.asarray(user_ids)
    y = np.asarray(labels, dtype=int)
    predictions = np.asarray(scores, dtype=float)
    if not (users.size == y.size == predictions.size):
        raise ValueError("user_ids, labels, and scores must have equal lengths")
    if users.size == 0:
        raise ValueError("metric inputs must not be empty")
    if not np.isin(y, (0, 1)).all():
        raise ValueError("labels must be binary")
    if pair_cap <= 0:
        raise ValueError("pair_cap must be positive")

    concordance_sum = 0.0
    pair_sum = 0
    aucs: list[float] = []
    capped_numerator = 0.0
    capped_weight = 0
    single_class_users = 0
    for user in np.unique(users):
        mask = users == user
        concordance, pairs = _concordance(y[mask], predictions[mask])
        if pairs == 0:
            single_class_users += 1
            continue
        auc = concordance / pairs
        weight = min(pairs, pair_cap)
        concordance_sum += concordance
        pair_sum += pairs
        aucs.append(auc)
        capped_numerator += auc * weight
        capped_weight += weight
    if not aucs:
        raise ValueError("no user has both outcome classes")
    return WithinUserAUC(
        pair_weighted_auc=concordance_sum / pair_sum,
        per_user_average_auc=float(np.mean(aucs)),
        capped_pair_weighted_auc=capped_numerator / capped_weight,
        eligible_users=len(aucs),
        single_class_users=single_class_users,
        total_pairs=pair_sum,
        capped_total_weight=capped_weight,
    )
