import pytest

import numpy as np

from kuairand_pure.metrics import evaluate_ranking_metrics, within_user_auc


def test_pair_weighted_and_per_user_average_auc_differ() -> None:
    result = within_user_auc(
        user_ids=[1, 1, 2, 2, 2, 2],
        labels=[1, 0, 1, 1, 0, 0],
        scores=[1.0, 0.0, 0.0, 0.1, 0.9, 1.0],
    )

    assert result.pair_weighted_auc == pytest.approx(1 / 5)
    assert result.per_user_average_auc == pytest.approx(1 / 2)
    assert result.total_pairs == 5


def test_prediction_ties_receive_half_credit() -> None:
    result = within_user_auc(
        user_ids=[1, 1], labels=[1, 0], scores=[0.5, 0.5]
    )

    assert result.pair_weighted_auc == pytest.approx(0.5)


def test_single_class_users_are_excluded_and_reported() -> None:
    result = within_user_auc(
        user_ids=[1, 1, 2, 2],
        labels=[1, 1, 1, 0],
        scores=[0.8, 0.7, 0.9, 0.1],
    )

    assert result.eligible_users == 1
    assert result.single_class_users == 1
    assert result.pair_weighted_auc == pytest.approx(1.0)


def test_capped_sensitivity_uses_capped_pair_weights() -> None:
    result = within_user_auc(
        user_ids=[1, 1, 2, 2, 2, 2],
        labels=[1, 0, 1, 1, 0, 0],
        scores=[1.0, 0.0, 0.0, 0.1, 0.9, 1.0],
        pair_cap=1,
    )

    assert result.pair_weighted_auc == pytest.approx(0.2)
    assert result.capped_pair_weighted_auc == pytest.approx(0.5)
    assert result.capped_total_weight == 2


def test_complete_metric_family_has_frozen_auc_logloss_and_ndcg_behavior() -> None:
    result = evaluate_ranking_metrics(
        user_ids=[1, 1, 1, 1, 1, 2, 2, 2, 2, 2],
        labels=[1, 0, 1, 0, 0, 0, 1, 0, 0, 0],
        scores=[0.9, 0.8, 0.7, 0.6, 0.5, 0.9, 0.8, 0.7, 0.6, 0.5],
        time_ms=list(range(10)),
        video_ids=list(range(10, 20)),
        ndcg_k=5,
    )

    assert result.impressions == 10
    assert result.positives == 3
    assert result.prevalence == pytest.approx(0.3)
    assert result.eligible_auc_users == 2
    assert result.ndcg_eligible_users == 2
    assert result.ndcg_excluded_users == 0
    assert result.pooled_auc == pytest.approx(16.5 / 21)
    expected_loss = -np.mean(
        np.asarray([1, 0, 1, 0, 0, 0, 1, 0, 0, 0]) * np.log(
            np.asarray([0.9, 0.8, 0.7, 0.6, 0.5, 0.9, 0.8, 0.7, 0.6, 0.5])
        )
        + (1 - np.asarray([1, 0, 1, 0, 0, 0, 1, 0, 0, 0])) * np.log(
            1 - np.asarray([0.9, 0.8, 0.7, 0.6, 0.5, 0.9, 0.8, 0.7, 0.6, 0.5])
        )
    )
    assert result.log_loss == pytest.approx(expected_loss)


def test_ndcg_score_ties_use_time_video_then_stable_order() -> None:
    first = evaluate_ranking_metrics(
        user_ids=[1] * 6,
        labels=[0, 1, 0, 1, 0, 0],
        scores=[0.5] * 6,
        time_ms=[1, 1, 2, 3, 4, 5],
        video_ids=[20, 10, 30, 40, 50, 60],
        ndcg_k=5,
    )
    second = evaluate_ranking_metrics(
        user_ids=[1] * 6,
        labels=[1, 0, 0, 1, 0, 0],
        scores=[0.5] * 6,
        time_ms=[1, 1, 2, 3, 4, 5],
        video_ids=[10, 20, 30, 40, 50, 60],
        ndcg_k=5,
    )

    assert first.ndcg_at_k == pytest.approx(second.ndcg_at_k)
    assert first.pooled_auc == pytest.approx(0.5)


def test_complete_metrics_reject_missing_scores() -> None:
    with pytest.raises(ValueError, match="finite probabilities"):
        evaluate_ranking_metrics(
            user_ids=[1, 1], labels=[1, 0], scores=[0.8, np.nan],
            time_ms=[1, 2], video_ids=[10, 11],
        )
