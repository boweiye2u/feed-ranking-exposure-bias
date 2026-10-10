from copy import deepcopy

import pytest

from kuairand_pure.protocol import (
    OutcomeAccessError,
    ProtocolError,
    authorize_outcome_access,
    load_protocol,
    validate_protocol,
)


def test_frozen_protocol_configuration_is_valid() -> None:
    protocol = load_protocol()

    assert protocol["protocol"]["branch"] == "B1"
    assert protocol["splits"]["burn_in"]["start"] == 20220409
    assert protocol["splits"]["normal_train"]["start"] == 20220412
    assert protocol["metrics"]["pair_cap_sensitivity"] == 952


def test_configuration_drift_is_rejected() -> None:
    protocol = deepcopy(load_protocol())
    protocol["splits"]["normal_train"]["start"] = 20220411

    with pytest.raises(ProtocolError, match="normal_train"):
        validate_protocol(protocol)


@pytest.mark.parametrize("source", ["normal", "random"])
@pytest.mark.parametrize("purpose", ["diagnostics", "training", "selection", "features"])
def test_pre_final_workflows_cannot_access_final_outcomes(source, purpose) -> None:
    with pytest.raises(OutcomeAccessError, match="Final-test outcomes"):
        authorize_outcome_access(
            load_protocol(),
            source=source,
            start=20220503,
            end=20220508,
            workflow="training",
            purpose=purpose,
        )


def test_final_outcomes_are_evaluation_only() -> None:
    protocol = load_protocol()
    authorize_outcome_access(
        protocol,
        source="random",
        start=20220503,
        end=20220508,
        workflow="final_evaluation",
        purpose="evaluation",
    )
    with pytest.raises(OutcomeAccessError, match="only to the final-evaluation"):
        authorize_outcome_access(
            protocol,
            source="normal",
            start=20220503,
            end=20220508,
            workflow="final_evaluation",
            purpose="features",
        )


def test_random_validation_outcomes_are_always_embargoed() -> None:
    with pytest.raises(OutcomeAccessError, match="Apr 30-May 2"):
        authorize_outcome_access(
            load_protocol(),
            source="random",
            start=20220430,
            end=20220502,
            workflow="validation",
            purpose="selection",
        )


def test_permitted_week0_slice_is_authorized() -> None:
    authorize_outcome_access(
        load_protocol(),
        source="random",
        start=20220422,
        end=20220429,
        workflow="week0",
        purpose="diagnostics",
    )
