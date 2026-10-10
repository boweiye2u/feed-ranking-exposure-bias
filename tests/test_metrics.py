import pytest

from kuairand_pure.metrics import within_user_auc


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
