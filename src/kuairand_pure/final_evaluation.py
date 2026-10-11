"""Held-out-safe final feature/scoring contracts; M5.3 never opens outcomes."""

from __future__ import annotations

from dataclasses import dataclass

import polars as pl

from kuairand_pure.data.fs1 import LABEL_COLUMNS
from kuairand_pure.models.lightgbm import LightGBMAdapter, predict_probabilities

FINAL_START = 20220503
FINAL_END = 20220508


@dataclass(frozen=True, slots=True)
class FinalEvaluationRequest:
    source: str
    start: int = FINAL_START
    end: int = FINAL_END
    phase: str = "prefinal"
    include_outcomes: bool = False

    def validate(self) -> None:
        if self.source not in {"normal", "random"}:
            raise ValueError("Final evaluation source must be normal or random")
        if (self.start, self.end) != (FINAL_START, FINAL_END):
            raise ValueError("Final evaluation dates drifted")
        if self.phase == "prefinal" and self.include_outcomes:
            raise PermissionError("M5.3 prefinal workflows cannot access held-out outcomes")
        if self.phase not in {"prefinal", "final_evaluation"}:
            raise ValueError("Unknown final-evaluation phase")


def final_feature_contract(request: FinalEvaluationRequest) -> dict[str, object]:
    request.validate()
    return {
        "source": request.source,
        "dates": [request.start, request.end],
        "aggregate_history": "strictly_prior_calendar_day_normal_policy_only",
        "sequence_history": "strictly_earlier_time_ms_normal_policy_only",
        "randomized_history_used": False,
        "outcomes_opened": request.include_outcomes,
        "adapter_policy": "reuse_training_fitted_without_refit",
    }


def apply_evaluation_source_context(
    features: pl.DataFrame,
    *,
    model_arm: str,
    evaluation_source: str,
) -> pl.DataFrame:
    """Apply the only permitted RQ3 joint-domain override before scoring."""
    if set(LABEL_COLUMNS).intersection(features.columns):
        raise ValueError("Outcome columns entered final model features")
    if evaluation_source not in {"normal", "random"}:
        raise ValueError("Unknown evaluation exposure source")
    if "context__is_rand" not in features.columns:
        raise ValueError("Final FS1/FS2/FS3 input lacks source context")
    if model_arm == "joint":
        return features.with_columns(
            pl.lit(1 if evaluation_source == "random" else 0, dtype=pl.Int64)
            .alias("context__is_rand")
        )
    return features


def score_frozen_lightgbm(
    booster,
    adapter: LightGBMAdapter,
    features: pl.DataFrame,
    *,
    model_arm: str,
    evaluation_source: str,
):
    """Score with a loaded training-fitted adapter; vocabulary fitting is impossible here."""
    if adapter.include_is_rand != (model_arm == "joint"):
        raise ValueError("Frozen adapter is incompatible with evaluation arm")
    prepared = apply_evaluation_source_context(
        features, model_arm=model_arm, evaluation_source=evaluation_source
    )
    return predict_probabilities(booster, adapter, prepared)
