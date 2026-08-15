"""Tests for the scientific/progress contract independent of an AiiDA profile."""

import pytest

from aiida_itensor_spin.contracts import (
    MOMENT_CONVENTION,
    ContractError,
    normalize_ground_state_parameters,
    normalize_layout,
    normalize_parameters,
    validate_segment_status,
)

PARAMETERS = {
    "expected_total_tasks": 33,
    "chebyshev_convention": MOMENT_CONVENTION,
    "backend": "streaming",
    "threading_mode": "blocksparse",
}


def status(*, tasks=0, units=0, state="checkpointed"):
    return {
        "format_version": 1,
        "state": state,
        "total_tasks": 33,
        "completed_tasks": tasks,
        "progress_units_completed": units,
        "progress_units_total": 3300,
        "chebyshev_convention": MOMENT_CONVENTION,
        "backend": "streaming",
        "threading_mode": "blocksparse",
    }


def test_l11_repacking_layouts_are_valid():
    first = normalize_layout(
        {
            "nodes": 2,
            "processes_per_node": 4,
            "threads_per_process": 2,
            "wallclock_seconds": 600,
            "internal_stop_seconds": 480,
            "stop_after_n": 12,
        }
    )
    second = normalize_layout(
        {
            "nodes": 2,
            "processes_per_node": 6,
            "threads_per_process": 2,
            "wallclock_seconds": 600,
            "internal_stop_seconds": 480,
        }
    )
    assert first["processes_per_node"] == 4
    assert second["processes_per_node"] == 6


def test_layout_rejects_node_oversubscription():
    with pytest.raises(ContractError, match="288"):
        normalize_layout(
            {
                "nodes": 1,
                "processes_per_node": 100,
                "threads_per_process": 3,
                "wallclock_seconds": 600,
            }
        )


def test_parameters_require_corrected_chebyshev_convention():
    with pytest.raises(ContractError, match="chebyshev_convention"):
        normalize_parameters({"expected_total_tasks": 33})
    with pytest.raises(ContractError, match="legacy"):
        normalize_parameters(
            {
                "expected_total_tasks": 33,
                "chebyshev_convention": "trusted-reference-2026-08",
            }
        )


def test_progress_can_continue_with_a_different_layout():
    previous = validate_segment_status(status(tasks=4, units=400), PARAMETERS)
    current = validate_segment_status(status(tasks=8, units=900), PARAMETERS, previous)
    assert current["made_progress"] is True


def test_progress_rejects_regression():
    previous = validate_segment_status(status(tasks=4, units=400), PARAMETERS)
    with pytest.raises(ContractError, match="backwards"):
        validate_segment_status(status(tasks=3, units=401), PARAMETERS, previous)


def test_complete_requires_all_manifest_tasks():
    with pytest.raises(ContractError, match="inconsistent"):
        validate_segment_status(status(tasks=32, units=3300, state="complete"), PARAMETERS)
    complete = validate_segment_status(status(tasks=33, units=3300, state="complete"), PARAMETERS)
    assert complete["state"] == "complete"


def test_ground_state_parameters_are_explicit_and_valid():
    assert normalize_ground_state_parameters(
        {"length": 11, "impurity_site": 6, "seed": 20260724, "threading_mode": "blocksparse"}
    )["length"] == 11
    with pytest.raises(ContractError, match="cannot exceed"):
        normalize_ground_state_parameters(
            {"length": 11, "impurity_site": 12, "seed": 20260724}
        )


def test_convention_mismatch_is_rejected():
    value = status(tasks=1, units=10)
    value["chebyshev_convention"] = "revised-final-moment"
    with pytest.raises(ContractError, match="convention"):
        validate_segment_status(value, PARAMETERS)

