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


@dataclass(frozen=True, slots=True)
class RankingMetrics:
    pair_weighted_auc: float
    per_user_average_auc: float
    capped_pair_weighted_auc: float
    pooled_auc: float
    log_loss: float
    ndcg_at_k: float
    ndcg_k: int
    impressions: int
    positives: int
    prevalence: float
    eligible_auc_users: int
    single_class_users: int
    total_pairs: int
    capped_total_weight: int
    ndcg_eligible_users: int
    ndcg_excluded_users: int


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


def evaluate_ranking_metrics(
    user_ids: npt.ArrayLike,
    labels: npt.ArrayLike,
    scores: npt.ArrayLike,
    time_ms: npt.ArrayLike,
    video_ids: npt.ArrayLike,
    *,
    pair_cap: int = 952,
    ndcg_k: int = 5,
    log_loss_epsilon: float = 1e-15,
) -> RankingMetrics:
    """Evaluate the complete frozen metric family for one aligned task."""
    users = np.asarray(user_ids)
    y = np.asarray(labels, dtype=int)
    predictions = np.asarray(scores, dtype=float)
    timestamps = np.asarray(time_ms)
    videos = np.asarray(video_ids)
    lengths = {users.size, y.size, predictions.size, timestamps.size, videos.size}
    if len(lengths) != 1 or not users.size:
        raise ValueError("metric inputs must be nonempty and equally aligned")
    if not np.isin(y, (0, 1)).all():
        raise ValueError("labels must be binary")
    if not np.isfinite(predictions).all() or ((predictions < 0) | (predictions > 1)).any():
        raise ValueError("scores must be finite probabilities in [0,1]")
    if ndcg_k <= 0:
        raise ValueError("ndcg_k must be positive")
    if not 0 < log_loss_epsilon < 0.5:
        raise ValueError("log_loss_epsilon must lie in (0,0.5)")

    within = within_user_auc(users, y, predictions, pair_cap=pair_cap)
    pooled_concordance, pooled_pairs = _concordance(y, predictions)
    if pooled_pairs == 0:
        raise ValueError("pooled AUC requires both outcome classes")
    clipped = np.clip(predictions, log_loss_epsilon, 1 - log_loss_epsilon)
    loss = float(-np.mean(y * np.log(clipped) + (1 - y) * np.log(1 - clipped)))

    stable_index = np.arange(users.size)
    ndcg_values: list[float] = []
    unique_users = np.unique(users)
    discounts = 1.0 / np.log2(np.arange(2, ndcg_k + 2))
    for user in unique_users:
        mask = users == user
        user_y = y[mask]
        if user_y.size < ndcg_k or user_y.sum() == 0:
            continue
        order = np.lexsort(
            (
                stable_index[mask],
                videos[mask],
                timestamps[mask],
                -predictions[mask],
            )
        )
        dcg = float(np.sum(user_y[order[:ndcg_k]] * discounts))
        ideal_positive_count = min(int(user_y.sum()), ndcg_k)
        idcg = float(np.sum(discounts[:ideal_positive_count]))
        ndcg_values.append(dcg / idcg)
    if not ndcg_values:
        raise ValueError("no user qualifies for NDCG")

    positives = int(y.sum())
    return RankingMetrics(
        pair_weighted_auc=within.pair_weighted_auc,
        per_user_average_auc=within.per_user_average_auc,
        capped_pair_weighted_auc=within.capped_pair_weighted_auc,
        pooled_auc=pooled_concordance / pooled_pairs,
        log_loss=loss,
        ndcg_at_k=float(np.mean(ndcg_values)),
        ndcg_k=ndcg_k,
        impressions=int(y.size),
        positives=positives,
        prevalence=positives / y.size,
        eligible_auc_users=within.eligible_users,
        single_class_users=within.single_class_users,
        total_pairs=within.total_pairs,
        capped_total_weight=within.capped_total_weight,
        ndcg_eligible_users=len(ndcg_values),
        ndcg_excluded_users=int(unique_users.size - len(ndcg_values)),
    )
