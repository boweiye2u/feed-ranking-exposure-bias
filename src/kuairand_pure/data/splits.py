"""Frozen date split definitions and assertions."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import polars as pl

from kuairand_pure.protocol import load_protocol

Exposure = Literal["normal", "random"]


@dataclass(frozen=True, slots=True)
class SplitSpec:
    name: str
    source: Exposure
    start: int
    end: int
    outcomes_allowed: bool


def frozen_splits() -> dict[str, SplitSpec]:
    config = load_protocol()
    raw = config["splits"]
    return {
        "burn_in": SplitSpec("burn_in", "normal", raw["burn_in"]["start"], raw["burn_in"]["end"], False),
        "normal_train": SplitSpec("normal_train", "normal", raw["normal_train"]["start"], raw["normal_train"]["end"], True),
        "random_train": SplitSpec("random_train", "random", raw["random_train"]["start"], raw["random_train"]["end"], True),
        "normal_validation": SplitSpec("normal_validation", "normal", raw["normal_validation"]["start"], raw["normal_validation"]["end"], True),
        "random_validation_embargo": SplitSpec("random_validation_embargo", "random", raw["random_validation_embargo"]["start"], raw["random_validation_embargo"]["end"], False),
        "normal_test": SplitSpec("normal_test", "normal", raw["final_test"]["start"], raw["final_test"]["end"], False),
        "random_test": SplitSpec("random_test", "random", raw["final_test"]["start"], raw["final_test"]["end"], False),
    }


def assert_nonoverlapping_layout(splits: dict[str, SplitSpec]) -> None:
    expected = {
        "burn_in": ("normal", 20220409, 20220411),
        "normal_train": ("normal", 20220412, 20220429),
        "random_train": ("random", 20220422, 20220429),
        "normal_validation": ("normal", 20220430, 20220502),
        "random_validation_embargo": ("random", 20220430, 20220502),
        "normal_test": ("normal", 20220503, 20220508),
        "random_test": ("random", 20220503, 20220508),
    }
    observed = {name: (spec.source, spec.start, spec.end) for name, spec in splits.items()}
    if observed != expected:
        raise ValueError(f"Frozen split date drift: {observed}")
    for source in ("normal", "random"):
        ordered = sorted(
            (spec for spec in splits.values() if spec.source == source),
            key=lambda spec: spec.start,
        )
        for left, right in zip(ordered, ordered[1:]):
            if left.end >= right.start:
                raise ValueError(f"Overlapping {source} splits: {left.name}, {right.name}")


def select_dates(frame: pl.LazyFrame, spec: SplitSpec) -> pl.LazyFrame:
    return frame.filter(pl.col("date").is_between(spec.start, spec.end))


def assert_exact_date_coverage(dates: pl.Series, spec: SplitSpec) -> None:
    observed = sorted(int(value) for value in dates.unique().to_list())
    expected = (
        pl.date_range(
            pl.date(spec.start // 10000, spec.start // 100 % 100, spec.start % 100),
            pl.date(spec.end // 10000, spec.end // 100 % 100, spec.end % 100),
            interval="1d",
            eager=True,
        )
        .dt.strftime("%Y%m%d")
        .cast(pl.Int64)
        .to_list()
    )
    if observed != expected:
        raise ValueError(f"{spec.name} date coverage drift: {observed}; expected {expected}")
