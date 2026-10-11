import copy

import pytest

from kuairand_pure.neural_evaluation import (
    NEURAL_CANDIDATES,
    SEEDS,
    summarize_seed_metrics,
    validate_candidate_manifest,
)


METRICS = {
    "pair_weighted_auc": 0.6,
    "per_user_average_auc": 0.61,
    "capped_pair_weighted_auc": 0.59,
    "pooled_auc": 0.7,
    "log_loss": 0.5,
    "ndcg_at_k": 0.65,
}


def test_seed_summary_requires_all_three_frozen_seeds():
    records = [
        {"candidate": candidate, "task": task, "seed": seed, "metrics": METRICS}
        for candidate in NEURAL_CANDIDATES
        for task in ("long_view", "like", "valid_play")
        for seed in SEEDS
    ]
    summaries = summarize_seed_metrics(records)
    assert len(summaries) == 15
    assert summaries[0]["pair_weighted_auc"] == {
        "mean": 0.6, "minimum": 0.6, "maximum": 0.6, "range": 0.0
    }
    with pytest.raises(ValueError, match="Incomplete seed set"):
        summarize_seed_metrics(records[:-1])


def _manifest():
    neural = [
        {
            "candidate": candidate,
            "model_family": "neural",
            "checkpoints": [{"seed": seed} for seed in SEEDS],
        }
        for candidate in NEURAL_CANDIDATES
    ]
    other = [
        {"candidate": "r0", "model_family": "r0"},
        *(
            {"candidate": f"fs{feature}_{depth}", "model_family": "lightgbm"}
            for feature in (1, 2, 3)
            for depth in ("shallow", "deep")
        ),
    ]
    return {"candidates": [*other, *neural], "selected_overall": {"candidate": "r0"}}


def test_candidate_manifest_freezes_exact_population_and_seeds():
    validate_candidate_manifest(_manifest())
    duplicate = _manifest()
    duplicate["candidates"][-1]["candidate"] = "r0"
    with pytest.raises(ValueError, match="12 unique"):
        validate_candidate_manifest(duplicate)
    bad_seeds = copy.deepcopy(_manifest())
    bad_seeds["candidates"][-1]["checkpoints"] = [{"seed": 0}, {"seed": 1}]
    with pytest.raises(ValueError, match="seeds"):
        validate_candidate_manifest(bad_seeds)


def test_candidate_manifest_rejects_esmm_and_unknown_selection():
    esmm = _manifest()
    esmm["candidates"][-1]["candidate"] = "esmm"
    with pytest.raises(ValueError, match="ESMM"):
        validate_candidate_manifest(esmm)
    missing = _manifest()
    missing["selected_overall"]["candidate"] = "not_present"
    with pytest.raises(ValueError, match="absent"):
        validate_candidate_manifest(missing)
