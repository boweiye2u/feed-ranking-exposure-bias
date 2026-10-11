import numpy as np

from kuairand_pure.inference import cross_fitted_regret
from kuairand_pure.rq2 import crossfit_estimates, deterministic_half_masks


def test_crossfit_halves_are_disjoint_complete_and_reproducible() -> None:
    first = deterministic_half_masks(9, split_count=3, seed=0)
    second = deterministic_half_masks(9, split_count=3, seed=0)
    assert np.array_equal(first, second)
    assert not np.any(first[0::2] & first[1::2])
    assert np.all(first[0::2] | first[1::2])


def test_vectorized_regret_equals_frozen_slow_estimator_with_multiplicities_and_ties() -> None:
    names = ["a", "b", "fs1_deep"]
    pairs = np.array([1, 2, 0, 3, 2, 1, 4, 2], dtype=float)
    credits = np.array([[1, 1, 0, 2, 1, .5, 3, 1], [1, 1, 0, 2, 1, .5, 3, 1],
                        [0, 1, 0, 1, 1, .5, 2, 1]], dtype=float)
    multiplicities = np.array([2, 0, 1, 1, 3, 0, 1, 2], dtype=float)
    observed = crossfit_estimates(names, credits, pairs, multiplicities,
                                  populations={"all": names}, split_count=3, seed=2)["all"]
    expected = cross_fitted_regret(names, credits, pairs, workflow_pick="fs1_deep",
                                   contenders=names, split_count=3, seed=2,
                                   multiplicities=multiplicities)
    assert np.isclose(observed["normal_selection_regret"][0], expected["workflow_regret"])
    assert np.isclose(observed["random_pick_regret"][0], expected["random_pick_regret"])
    assert sum(observed["selection_counts"].values()) == 6


def test_lexicographic_tie_breaking_selects_first_candidate() -> None:
    names = ["a", "fs1_deep"]
    pairs = np.ones(8)
    credits = np.ones((2, 8)) * 0.5
    result = crossfit_estimates(names, credits, pairs, np.ones(8),
                                populations={"all": names}, split_count=2)["all"]
    assert result["selection_counts"] == {"a": 4}
