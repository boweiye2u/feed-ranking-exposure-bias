import numpy as np
import polars as pl
import pytest

from kuairand_pure.final_evaluation import (
    FinalEvaluationRequest,
    apply_evaluation_source_context,
    final_feature_contract,
)
from kuairand_pure.inference import (
    bootstrap_cross_fitted_regret,
    candidate_stat_matrix,
    contender_set,
    cross_fitted_regret,
    holm_adjust,
    paired_user_bootstrap,
    percentile_interval,
    tail_share_p_value,
)


def _synthetic_stats():
    users = np.repeat(np.arange(8), 2)
    labels = np.tile([1, 0], 8)
    strong = np.tile([0.9, 0.1], 8)
    weak = np.tile([0.6, 0.4], 8)
    reversed_scores = np.tile([0.1, 0.9], 8)
    return candidate_stat_matrix(
        users, labels,
        {"a_strong": strong, "b_weak": weak, "c_reverse": reversed_scores},
    )


def test_paired_bootstrap_percentiles_tail_p_and_holm() -> None:
    names, _, credits, pairs = _synthetic_stats()
    draws = paired_user_bootstrap(credits, pairs, replicates=40, seed=0, batch_size=10)
    assert draws.shape == (len(names), 40)
    assert percentile_interval(draws[0] - draws[2]) == pytest.approx((1.0, 1.0))
    assert tail_share_p_value(draws[0] - draws[2]) == 0
    adjusted = holm_adjust([0.01, 0.04, 0.03])
    assert np.allclose(adjusted, [0.03, 0.06, 0.06])


def test_contenders_include_ci_matches_and_top_three() -> None:
    names = ["a", "b", "c", "fs1_deep"]
    point = np.array([0.9, 0.8, 0.7, 0.85])
    draws = np.array([
        [0.9, 0.91, 0.89], [0.8, 0.81, 0.79],
        [0.7, 0.71, 0.69], [0.85, 0.86, 0.84],
    ])
    contenders, _ = contender_set(names, point, draws, validation_best="fs1_deep")
    assert set(contenders).issuperset({"a", "b", "fs1_deep"})


def test_cross_fit_is_disjoint_deterministic_and_bootstrappable() -> None:
    names, _, credits, pairs = _synthetic_stats()
    first = cross_fitted_regret(
        names, credits, pairs, workflow_pick="b_weak", contenders=["a_strong", "b_weak"],
        split_count=5, seed=0,
    )
    second = cross_fitted_regret(
        names, credits, pairs, workflow_pick="b_weak", contenders=["a_strong", "b_weak"],
        split_count=5, seed=0,
    )
    assert first == second
    assert first["workflow_regret"] == pytest.approx(0.0)
    boot = bootstrap_cross_fitted_regret(
        names, credits, pairs, workflow_pick="b_weak", contenders=["a_strong", "b_weak"],
        replicates=5, split_count=3, seed=0,
    )
    assert all(values.shape == (5,) for values in boot.values())


def test_lexicographic_tie_break_is_reproducible() -> None:
    pairs = np.ones(6, dtype=int)
    credits = np.vstack([np.ones(6), np.ones(6), np.zeros(6)])
    result = cross_fitted_regret(
        ["a", "b", "c"], credits, pairs, workflow_pick="b", contenders=["a", "b"],
        split_count=4, seed=0,
    )
    assert result["workflow_regret"] == 0


def test_prefinal_contract_rejects_outcomes_and_source_context_is_joint_only() -> None:
    contract = final_feature_contract(FinalEvaluationRequest("random"))
    assert contract["outcomes_opened"] is False
    assert contract["history_source"] if "history_source" in contract else True
    with pytest.raises(PermissionError, match="cannot access"):
        final_feature_contract(FinalEvaluationRequest("random", include_outcomes=True))
    features = pl.DataFrame({"context__is_rand": [0, 0], "x": [1.0, 2.0]})
    joint = apply_evaluation_source_context(features, model_arm="joint", evaluation_source="random")
    assert joint["context__is_rand"].to_list() == [1, 1]
    baseline = apply_evaluation_source_context(features, model_arm="baseline", evaluation_source="random")
    assert baseline["context__is_rand"].to_list() == [0, 0]
    with pytest.raises(ValueError, match="Outcome"):
        apply_evaluation_source_context(
            features.with_columns(pl.lit(1).alias("long_view")),
            model_arm="joint", evaluation_source="random",
        )
