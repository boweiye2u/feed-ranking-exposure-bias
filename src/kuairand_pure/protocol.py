"""Week-0 protocol loading, validation, and outcome-access safeguards."""

from __future__ import annotations

import tomllib
from pathlib import Path
from typing import Any, Literal

Source = Literal["normal", "random"]
Workflow = Literal["week0", "training", "validation", "final_evaluation"]
Purpose = Literal["diagnostics", "training", "selection", "evaluation", "features"]


class ProtocolError(ValueError):
    """Raised when the frozen protocol configuration is invalid."""


class OutcomeAccessError(PermissionError):
    """Raised when a workflow requests embargoed outcomes."""


def default_protocol_path() -> Path:
    return Path(__file__).resolve().parents[2] / "configs" / "week0_protocol.toml"


def load_protocol(path: Path | None = None) -> dict[str, Any]:
    config_path = path or default_protocol_path()
    with config_path.open("rb") as stream:
        protocol = tomllib.load(stream)
    validate_protocol(protocol)
    return protocol


def _window(config: dict[str, Any], name: str) -> tuple[int, int]:
    split = config["splits"][name]
    return int(split["start"]), int(split["end"])


def _overlaps(start: int, end: int, window: tuple[int, int]) -> bool:
    return start <= window[1] and end >= window[0]


def _inside(start: int, end: int, window: tuple[int, int]) -> bool:
    return window[0] <= start <= end <= window[1]


def validate_protocol(config: dict[str, Any]) -> None:
    """Reject accidental drift in the approved Week-0 decisions."""
    if config["protocol"]["branch"] != "B1":
        raise ProtocolError("Week-0 protocol must use confirmed Branch B1")
    expected_windows = {
        "burn_in": (20220409, 20220411),
        "normal_train": (20220412, 20220429),
        "random_train": (20220422, 20220429),
        "normal_validation": (20220430, 20220502),
        "random_validation_embargo": (20220430, 20220502),
        "final_test": (20220503, 20220508),
    }
    for name, expected in expected_windows.items():
        if _window(config, name) != expected:
            raise ProtocolError(f"Unexpected {name} dates: {_window(config, name)}")
    if config["metrics"]["primary"] != "pair_weighted_within_user_auc":
        raise ProtocolError("Primary metric drifted from pair-weighted within-user AUC")
    if config["metrics"]["primary_target"] != "long_view":
        raise ProtocolError("Primary target drifted from long view")
    if config["metrics"]["pair_cap_sensitivity"] != 952:
        raise ProtocolError("Sensitivity pair cap must remain frozen at 952")
    if config["metrics"]["pair_cap_primary"] is not False:
        raise ProtocolError("Capped weighting is sensitivity-only")
    if config["configuration_population"]["esmm_included"] is not False:
        raise ProtocolError("ESMM must remain excluded")
    if config["features"]["shipped_video_statistics"] != "excluded_unverified_temporal_cutoff":
        raise ProtocolError("Shipped aggregate video statistics must remain excluded")
    if config["features"]["final_test_outcomes_in_features"] is not False:
        raise ProtocolError("Final-test outcomes may not enter features")
    if config["targets"]["valid_play"]["tabs"] != [1]:
        raise ProtocolError("Tab-1 is_click must be isolated as valid play")
    if config["targets"]["literal_click"]["tabs"] != [0]:
        raise ProtocolError("Tab-0 is_click must be isolated as literal click")


def authorize_outcome_access(
    config: dict[str, Any],
    *,
    source: Source,
    start: int,
    end: int,
    workflow: Workflow,
    purpose: Purpose,
) -> None:
    """Authorize a narrowly defined outcome request or raise.

    Structural ID/date access does not use this function because it does not
    request outcomes. Every outcome-bearing loader must call this guard.
    """
    if start > end:
        raise OutcomeAccessError("Outcome access start must not exceed end")

    final_test = _window(config, "final_test")
    random_embargo = _window(config, "random_validation_embargo")
    if _overlaps(start, end, final_test):
        if workflow != "final_evaluation" or purpose != "evaluation":
            raise OutcomeAccessError(
                "Final-test outcomes are available only to the final-evaluation metric workflow"
            )
        if not _inside(start, end, final_test):
            raise OutcomeAccessError("Final-evaluation request must stay inside final-test dates")
        return

    if source == "random" and _overlaps(start, end, random_embargo):
        raise OutcomeAccessError("Randomized Apr 30-May 2 outcomes are embargoed")

    if source == "normal" and workflow == "training" and purpose == "features":
        normal_history = (
            _window(config, "burn_in")[0],
            _window(config, "normal_train")[1],
        )
        if _inside(start, end, normal_history):
            return
        raise OutcomeAccessError(
            f"Normal feature history must stay inside burn-in/training dates: {start}-{end}"
        )

    allowed: dict[tuple[Workflow, Purpose, Source], tuple[int, int]] = {
        ("week0", "diagnostics", "random"): _window(config, "random_train"),
        ("training", "training", "normal"): _window(config, "normal_train"),
        ("training", "training", "random"): _window(config, "random_train"),
        ("validation", "selection", "normal"): _window(config, "normal_validation"),
    }
    window = allowed.get((workflow, purpose, source))
    if window is None or not _inside(start, end, window):
        raise OutcomeAccessError(
            f"Unauthorized {source} outcome access for {workflow}/{purpose}: {start}-{end}"
        )
