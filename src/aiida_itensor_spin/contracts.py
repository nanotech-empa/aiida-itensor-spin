"""Pure validation helpers shared by the CalcJob, parser, and WorkChain."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import PurePosixPath
from typing import Any

STATUS_FORMAT_VERSION = 1
MAX_CORES_PER_DAINT_NODE = 288
RECOMMENDED_THREADS_PER_PROCESS = 2
ALLOWED_BACKENDS = {"streaming"}
ALLOWED_THREADING_MODES = {"outer", "blocksparse"}
MOMENT_CONVENTION = "chebyshev-2n-minus-1-v2"


class ContractError(ValueError):
    """Raised when a submitted or reported segment violates the workflow contract."""


def _integer(mapping: Mapping[str, Any], key: str, *, minimum: int = 0) -> int:
    value = mapping.get(key)
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ContractError(f"{key} must be an integer >= {minimum}")
    return value


def safe_relative_path(value: str, *, label: str) -> str:
    """Validate a relative POSIX path that cannot escape its remote folder."""
    path = PurePosixPath(value)
    if path.is_absolute() or not path.parts or ".." in path.parts:
        raise ContractError(f"{label} must be a non-empty relative path without '..'")
    return str(path)


def normalize_layout(value: Mapping[str, Any]) -> dict[str, Any]:
    """Validate one Daint segment layout and return a JSON-compatible copy."""
    nodes = _integer(value, "nodes", minimum=1)
    processes_per_node = _integer(value, "processes_per_node", minimum=1)
    threads_per_process = _integer(value, "threads_per_process", minimum=1)
    wallclock_seconds = _integer(value, "wallclock_seconds", minimum=60)
    if processes_per_node * threads_per_process > MAX_CORES_PER_DAINT_NODE:
        raise ContractError(
            "processes_per_node * threads_per_process exceeds the 288 physical cores "
            "available on a Daint GH200 node"
        )

    internal_stop_seconds = (
        _integer(value, "internal_stop_seconds", minimum=1)
        if "internal_stop_seconds" in value
        else max(1, wallclock_seconds - 120)
    )
    if internal_stop_seconds >= wallclock_seconds:
        raise ContractError("internal_stop_seconds must be smaller than wallclock_seconds")

    result: dict[str, Any] = {
        "nodes": nodes,
        "processes_per_node": processes_per_node,
        "threads_per_process": threads_per_process,
        "wallclock_seconds": wallclock_seconds,
        "internal_stop_seconds": internal_stop_seconds,
        "partition": str(value.get("partition", "debug")),
    }
    if not result["partition"]:
        raise ContractError("partition cannot be empty")
    if "stop_after_n" in value:
        result["stop_after_n"] = _integer(value, "stop_after_n", minimum=2)
    return result


def normalize_resource_limits(value: Mapping[str, Any]) -> dict[str, Any]:
    """Validate user ceilings for automatic spectral resource selection."""
    max_nodes = _integer(value, "max_nodes", minimum=1)
    max_processes = _integer(value, "max_processes_per_node", minimum=1)
    max_threads = _integer(value, "max_threads_per_process", minimum=1)
    if max_processes > MAX_CORES_PER_DAINT_NODE:
        raise ContractError(f"max_processes_per_node cannot exceed {MAX_CORES_PER_DAINT_NODE}")
    if max_threads > MAX_CORES_PER_DAINT_NODE:
        raise ContractError(f"max_threads_per_process cannot exceed {MAX_CORES_PER_DAINT_NODE}")

    wallclock_seconds = _integer(value, "wallclock_seconds", minimum=60)
    internal_stop_seconds = _integer(value, "internal_stop_seconds", minimum=1)
    if internal_stop_seconds >= wallclock_seconds:
        raise ContractError("internal_stop_seconds must be smaller than wallclock_seconds")
    partition = str(value.get("partition", "normal"))
    if not partition:
        raise ContractError("partition cannot be empty")
    return {
        "max_nodes": max_nodes,
        "max_processes_per_node": max_processes,
        "max_threads_per_process": max_threads,
        "wallclock_seconds": wallclock_seconds,
        "internal_stop_seconds": internal_stop_seconds,
        "partition": partition,
    }


def select_largest_feasible_layout(
    total_tasks: int, resource_limits: Mapping[str, Any]
) -> dict[str, Any]:
    """Select the largest densely packed worker layout within user ceilings.

    ``processes_per_node`` is the allocation capacity, not a requirement that
    every node contain exactly that many active workers.  The final node may
    therefore be partially occupied.
    """
    tasks = _integer({"total_tasks": total_tasks}, "total_tasks", minimum=1)
    limits = normalize_resource_limits(resource_limits)
    workers = min(
        limits["max_nodes"] * limits["max_processes_per_node"],
        tasks,
    )
    processes_per_node = min(limits["max_processes_per_node"], workers)
    nodes = (workers + processes_per_node - 1) // processes_per_node
    threads_per_process = min(
        RECOMMENDED_THREADS_PER_PROCESS,
        limits["max_threads_per_process"],
        MAX_CORES_PER_DAINT_NODE // processes_per_node,
    )
    return {
        "nodes": nodes,
        "processes_per_node": processes_per_node,
        "threads_per_process": threads_per_process,
        "wallclock_seconds": limits["wallclock_seconds"],
        "internal_stop_seconds": limits["internal_stop_seconds"],
        "partition": limits["partition"],
        "workers": workers,
    }


def normalize_parameters(value: Mapping[str, Any]) -> dict[str, Any]:
    """Validate invariant scientific/algorithm inputs for every segment."""
    convention = str(value.get("chebyshev_convention", "")).strip()
    if convention != MOMENT_CONVENTION:
        raise ContractError(
            f"chebyshev_convention must be {MOMENT_CONVENTION!r}; "
            "legacy 2N-slot results cannot be mixed with corrected calculations"
        )

    backend = str(value.get("backend", "streaming")).lower()
    if backend not in ALLOWED_BACKENDS:
        raise ContractError(f"backend must be one of {sorted(ALLOWED_BACKENDS)}")
    threading_mode = str(value.get("threading_mode", "blocksparse")).lower()
    if threading_mode not in ALLOWED_THREADING_MODES:
        raise ContractError(f"threading_mode must be one of {sorted(ALLOWED_THREADING_MODES)}")

    checkpoint_every = (
        _integer(value, "checkpoint_every", minimum=1) if "checkpoint_every" in value else 1
    )
    heartbeat_every = (
        _integer(value, "heartbeat_every", minimum=1) if "heartbeat_every" in value else 1
    )
    checkpoint_name = safe_relative_path(
        str(value.get("ground_state_filename", "gs_checkpoint.jld2")),
        label="ground_state_filename",
    )
    result = dict(value)
    result.update(
        {
            "chebyshev_convention": convention,
            "backend": backend,
            "threading_mode": threading_mode,
            "ground_state_filename": checkpoint_name,
            "checkpoint_every": checkpoint_every,
            "heartbeat_every": heartbeat_every,
        }
    )
    if "expected_total_tasks" in value:
        result["expected_total_tasks"] = _integer(value, "expected_total_tasks", minimum=1)
    return result


def add_discovered_task_count(
    parameters: Mapping[str, Any], ground_state_metadata: Mapping[str, Any]
) -> dict[str, Any]:
    """Add the task count discovered from a completed ground-state checkpoint."""
    normalized = normalize_parameters(parameters)
    length = _integer(ground_state_metadata, "length", minimum=3)
    ground_states = _integer(ground_state_metadata, "ground_states", minimum=1)
    total_tasks = _integer(ground_state_metadata, "total_tasks", minimum=1)
    derived_total = 3 * length * ground_states
    if total_tasks != derived_total:
        raise ContractError(
            "ground-state metadata task count is inconsistent with 3 * length * ground_states"
        )
    supplied_total = normalized.get("expected_total_tasks")
    if supplied_total is not None and supplied_total != total_tasks:
        raise ContractError(
            f"supplied expected_total_tasks={supplied_total} differs from the "
            f"ground-state checkpoint value {total_tasks}"
        )
    normalized["expected_total_tasks"] = total_tasks
    return normalized


def normalize_ground_state_parameters(value: Mapping[str, Any]) -> dict[str, Any]:
    """Validate inputs of the trusted parameterized ground-state calculation."""
    length = _integer(value, "length", minimum=3)
    impurity_site = _integer(value, "impurity_site", minimum=1)
    if impurity_site > length:
        raise ContractError("impurity_site cannot exceed length")
    seed = _integer(value, "seed", minimum=0)
    threading_mode = str(value.get("threading_mode", "blocksparse")).lower()
    if threading_mode not in ALLOWED_THREADING_MODES:
        raise ContractError(f"threading_mode must be one of {sorted(ALLOWED_THREADING_MODES)}")
    return {
        "length": length,
        "impurity_site": impurity_site,
        "seed": seed,
        "threading_mode": threading_mode,
    }


def validate_segment_status(
    raw: Mapping[str, Any],
    parameters: Mapping[str, Any],
    previous: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Validate a segment report and check progress is compatible and monotonic."""
    params = normalize_parameters(parameters)
    status = dict(raw)
    if status.get("format_version") != STATUS_FORMAT_VERSION:
        raise ContractError(f"unsupported segment status format: {status.get('format_version')!r}")
    if status.get("state") not in {"checkpointed", "complete"}:
        raise ContractError("segment state must be 'checkpointed' or 'complete'")
    if status.get("chebyshev_convention") != params["chebyshev_convention"]:
        raise ContractError("Chebyshev convention differs from the workflow input")
    if status.get("backend") != params["backend"]:
        raise ContractError("backend differs from the workflow input")
    if status.get("threading_mode") != params["threading_mode"]:
        raise ContractError("threading mode differs from the workflow input")

    total_tasks = _integer(status, "total_tasks", minimum=1)
    completed_tasks = _integer(status, "completed_tasks", minimum=0)
    progress_units = _integer(status, "progress_units_completed", minimum=0)
    progress_units_total = _integer(status, "progress_units_total", minimum=1)
    expected_total_tasks = params.get("expected_total_tasks")
    if expected_total_tasks is not None and total_tasks != expected_total_tasks:
        raise ContractError(f"segment reports {total_tasks} tasks; expected {expected_total_tasks}")
    if completed_tasks > total_tasks:
        raise ContractError("completed_tasks exceeds total_tasks")
    if progress_units > progress_units_total:
        raise ContractError("progress_units_completed exceeds progress_units_total")

    is_complete = status["state"] == "complete"
    if is_complete != (completed_tasks == total_tasks):
        raise ContractError("complete state and completed task count are inconsistent")

    made_progress = completed_tasks > 0 or progress_units > 0
    if previous is not None:
        previous_tasks = _integer(previous, "completed_tasks", minimum=0)
        previous_units = _integer(previous, "progress_units_completed", minimum=0)
        if total_tasks != _integer(previous, "total_tasks", minimum=1):
            raise ContractError("total task count changed between segments")
        if completed_tasks < previous_tasks or progress_units < previous_units:
            raise ContractError("segment progress went backwards")
        if progress_units_total != _integer(previous, "progress_units_total", minimum=1):
            raise ContractError("total progress units changed between segments")
        made_progress = completed_tasks > previous_tasks or progress_units > previous_units

    status["made_progress"] = made_progress
    return status
