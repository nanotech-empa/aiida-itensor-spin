"""Tests for the scientific/progress contract independent of an AiiDA profile."""

import pytest

from aiida_itensor_spin.contracts import (
    MOMENT_CONVENTION,
    ContractError,
    add_discovered_task_count,
    normalize_ground_state_parameters,
    normalize_layout,
    normalize_parameters,
    select_largest_feasible_layout,
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


def test_task_count_is_discovered_from_ground_state_metadata():
    parameters = dict(PARAMETERS)
    parameters.pop("expected_total_tasks")
    discovered = add_discovered_task_count(
        parameters,
        {"length": 11, "ground_states": 1, "total_tasks": 33},
    )
    assert discovered["expected_total_tasks"] == 33


def test_discovered_task_count_rejects_user_guess_and_bad_metadata():
    with pytest.raises(ContractError, match="differs"):
        add_discovered_task_count(
            PARAMETERS,
            {"length": 101, "ground_states": 3, "total_tasks": 909},
        )
    without_guess = dict(PARAMETERS)
    without_guess.pop("expected_total_tasks")
    with pytest.raises(ContractError, match="inconsistent"):
        add_discovered_task_count(
            without_guess,
            {"length": 101, "ground_states": 3, "total_tasks": 303},
        )


def test_largest_feasible_layout_respects_ceilings():
    base = {
        "max_nodes": 2,
        "max_processes_per_node": 6,
        "max_threads_per_process": 2,
        "wallclock_seconds": 900,
        "internal_stop_seconds": 780,
        "partition": "debug",
    }
    l11 = select_largest_feasible_layout(33, base)
    assert (l11["nodes"], l11["processes_per_node"], l11["workers"]) == (2, 6, 12)

    assert l11["threads_per_process"] == 2
    one_thread = select_largest_feasible_layout(33, {**base, "max_threads_per_process": 1})
    assert one_thread["threads_per_process"] == 1

    l101 = select_largest_feasible_layout(
        909,
        {
            **base,
            "max_nodes": 10,
            "max_processes_per_node": 12,
        },
    )
    assert (l101["nodes"], l101["processes_per_node"], l101["workers"]) == (10, 12, 120)


def test_layout_packs_partial_final_node_instead_of_factoring_worker_count():
    limits = {
        "max_nodes": 70,
        "max_processes_per_node": 8,
        "max_threads_per_process": 4,
        "wallclock_seconds": 43200,
        "internal_stop_seconds": 41400,
        "partition": "normal",
    }
    layout = select_largest_feasible_layout(62, limits)
    assert (layout["nodes"], layout["processes_per_node"], layout["workers"]) == (8, 8, 62)

    layout = select_largest_feasible_layout(123, limits)
    assert (layout["nodes"], layout["processes_per_node"], layout["workers"]) == (16, 8, 123)


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
    normalized = normalize_ground_state_parameters(
        {"length": 23, "impurity_site": 12, "seed": 20260724, "threading_mode": "blocksparse"}
    )
    assert normalized["length"] == 23
    assert normalized["model"] == "spin1_chain_spinhalf_impurity_v1"
    assert normalized["J_meV"] == 18.4
    assert normalized["JL_meV"] == 16.7
    assert normalized["JR_meV"] == 51.1
    assert normalized["dmrg_cutoff"] == 1e-8
    assert normalized["spectral_cutoff"] == 1e-6
    assert normalized["omega_smearing_meV"] == 1.5
    with pytest.raises(ContractError, match="cannot exceed"):
        normalize_ground_state_parameters({"length": 11, "impurity_site": 12, "seed": 20260724})


def test_convention_mismatch_is_rejected():
    value = status(tasks=1, units=10)
    value["chebyshev_convention"] = "revised-final-moment"
    with pytest.raises(ContractError, match="convention"):
        validate_segment_status(value, PARAMETERS)


def test_ground_state_model_and_numerics_are_validated():
    base = {"length": 11, "impurity_site": 6, "seed": 1}
    with pytest.raises(ContractError, match="registered model"):
        normalize_ground_state_parameters({**base, "model": "unknown"})
    with pytest.raises(ContractError, match="spectral_cutoff"):
        normalize_ground_state_parameters({**base, "spectral_cutoff": 0})
    with pytest.raises(ContractError, match="temperature_K"):
        normalize_ground_state_parameters({**base, "temperature_K": -1})
