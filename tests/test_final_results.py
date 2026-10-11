import numpy as np
import polars as pl

from kuairand_pure.final_results import evaluate_prepared, per_user_credits, prepare_population
from kuairand_pure.metrics import evaluate_ranking_metrics


def test_grouped_final_metrics_equal_frozen_reference_with_ties_and_zero_pair_user() -> None:
    frame = pl.DataFrame({
        "impression_id": list(range(1, 12)), "user_id": [1] * 5 + [2] * 5 + [3],
        "video_id": [5, 4, 3, 2, 1] * 2 + [1], "time_ms": [1, 2, 3, 4, 5] * 2 + [1],
        "long_view": [1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1],
    })
    scores = np.array([0.5, 0.5, 0.9, 0.1, 0.8, 0.2, 0.7, 0.4, 0.6, 0.3, 0.2])
    pop = prepare_population(frame, "long_view")
    observed = evaluate_prepared(pop, scores)
    expected = evaluate_ranking_metrics(frame["user_id"].to_numpy(), frame["long_view"].to_numpy(),
        scores, frame["time_ms"].to_numpy(), frame["video_id"].to_numpy(), pair_cap=952, ndcg_k=5)
    assert observed["pair_weighted_auc"] == expected.pair_weighted_auc
    assert observed["per_user_average_auc"] == expected.per_user_average_auc
    assert per_user_credits(pop, scores).shape == (3,)


def test_per_user_credits_preserve_score_ties() -> None:
    frame = pl.DataFrame({"impression_id": [1, 2], "user_id": [1, 1], "video_id": [1, 2],
                          "time_ms": [1, 2], "long_view": [1, 0]})
    pop = prepare_population(frame, "long_view")
    assert per_user_credits(pop, np.array([0.5, 0.5])).tolist() == [0.5]
