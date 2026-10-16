"""Pure validation helpers shared by the CalcJob, parser, and WorkChain."""

from __future__ import annotations

import math
from collections.abc import Mapping
from pathlib import PurePosixPath
from typing import Any

from .graphs import GRAPH_MODEL_IDENTIFIER, normalize_spin_graph

STATUS_FORMAT_VERSION = 1
MAX_CORES_PER_DAINT_NODE = 288
RECOMMENDED_THREADS_PER_PROCESS = 2
ALLOWED_BACKENDS = {"streaming"}
ALLOWED_THREADING_MODES = {"outer", "blocksparse"}
MOMENT_CONVENTION = "chebyshev-2n-minus-1-v2"
MODEL_IDENTIFIER = "spin1_chain_spinhalf_impurity_v1"
MIXED_MODEL_IDENTIFIER = "mixed_spin_heisenberg_chain_v1"
REGISTERED_MODELS = {MODEL_IDENTIFIER, MIXED_MODEL_IDENTIFIER, GRAPH_MODEL_IDENTIFIER}

GROUND_STATE_DEFAULTS = {
    "model": MODEL_IDENTIFIER,
    "J_meV": 18.4,
    "JL_meV": 16.7,
    "JR_meV": 51.1,
    "dmrg_cutoff": 1e-8,
    "spectral_cutoff": 1e-6,
    "energy_precision_meV": 1e-5,
    "highest_energy_precision_meV": 1e-3,
    "initial_linkdim": 100,
    "excited_states": 2,
    "excited_state_penalty_meV": 1000.0,
    "temperature_K": 4.0,
    "symmetry_tolerance": 1e-2,
    "omega_max_meV": 100.0,
    "omega_smearing_meV": 1.5,
    "try_doubleN": True,
}


class ContractError(ValueError):
    """Raised when a submitted or reported segment violates the workflow contract."""


def _integer(mapping: Mapping[str, Any], key: str, *, minimum: int = 0) -> int:
    value = mapping.get(key)
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ContractError(f"{key} must be an integer >= {minimum}")
    return value


def _positive_float(mapping: Mapping[str, Any], key: str) -> float:
    value = mapping.get(key)
    if isinstance(value, bool) or not isinstance(value, (int, float)) or value <= 0:
        raise ContractError(f"{key} must be a number > 0")
    return float(value)


def _real_float(mapping: Mapping[str, Any], key: str) -> float:
    value = mapping.get(key)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ContractError(f"{key} must be a real number")
    return float(value)


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
    explicit_threads = None
    if "threads_per_process" in value:
        explicit_threads = _integer(value, "threads_per_process", minimum=1)
        if explicit_threads > max_threads:
            raise ContractError("threads_per_process exceeds max_threads_per_process")
        if explicit_threads * max_processes > MAX_CORES_PER_DAINT_NODE:
            raise ContractError("explicit thread layout exceeds 288 physical cores per node")
    return {
        **({"threads_per_process": explicit_threads} if explicit_threads is not None else {}),
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
        limits.get("threads_per_process", RECOMMENDED_THREADS_PER_PROCESS),
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

    kernel_mode = value.get("kernel_threading_mode", "legacy")
    if kernel_mode not in {"legacy", "blas"}:
        raise ContractError("kernel_threading_mode must be legacy or blas")
    if kernel_mode == "blas":
        _integer(value, "blas_threads", minimum=1)
        if threading_mode != "blocksparse":
            raise ContractError("BLAS continuation requires blocksparse checkpoint compatibility")
    elif "blas_threads" in value:
        raise ContractError("blas_threads requires kernel_threading_mode=blas")

    checkpoint_every = (
        _integer(value, "checkpoint_every", minimum=1) if "checkpoint_every" in value else 1
    )
    heartbeat_every = (
        _integer(value, "heartbeat_every", minimum=1) if "heartbeat_every" in value else 1
    )
    spectral_cutoff = (
        _positive_float(value, "spectral_cutoff")
        if "spectral_cutoff" in value
        else None
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
    if "spectral_cutoff" in value:
        result["spectral_cutoff"] = spectral_cutoff
    if "max_bond_dimension" in value:
        result["max_bond_dimension"] = _integer(value, "max_bond_dimension", minimum=0)
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


def validate_ground_state_energy_window(
    ground_state_metadata: Mapping[str, Any],
) -> dict[str, float] | None:
    """Validate a post-GS spectral window when energy metadata is available.

    Old checkpoints without these fields remain usable for continuation.
    Fresh ground-state calculations publish all four values.
    """
    keys = {
        "ground_energy_meV",
        "highest_energy_meV",
        "spectral_bandwidth_meV",
        "omega_max_meV",
    }
    present = keys.intersection(ground_state_metadata)
    if not present:
        return None
    if present != keys:
        missing = sorted(keys - present)
        raise ContractError(f"ground-state energy metadata is incomplete; missing {missing}")

    values = {}
    for key in sorted(keys):
        value = ground_state_metadata[key]
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ContractError(f"{key} must be a finite real number")
        value = float(value)
        if not math.isfinite(value):
            raise ContractError(f"{key} must be a finite real number")
        values[key] = value

    bandwidth = values["spectral_bandwidth_meV"]
    if bandwidth <= 0:
        raise ContractError("spectral_bandwidth_meV must be > 0")
    derived = values["highest_energy_meV"] - values["ground_energy_meV"]
    if not math.isclose(bandwidth, derived, rel_tol=1e-10, abs_tol=1e-10):
        raise ContractError("spectral bandwidth differs from highest minus ground energy")
    omega_max = values["omega_max_meV"]
    if omega_max <= 0:
        raise ContractError("omega_max_meV must be > 0")
    if omega_max > bandwidth and not math.isclose(
        omega_max, bandwidth, rel_tol=1e-12, abs_tol=1e-12
    ):
        raise ContractError(
            f"omega_max_meV={omega_max:g} exceeds the GS spectral bandwidth {bandwidth:g} meV"
        )
    return values


def _common_ground_state_parameters(
    supplied: Mapping[str, Any], value: Mapping[str, Any]
) -> dict[str, Any]:
    threading_mode = str(supplied.get("threading_mode", "blocksparse")).lower()
    if threading_mode not in ALLOWED_THREADING_MODES:
        raise ContractError(f"threading_mode must be one of {sorted(ALLOWED_THREADING_MODES)}")
    temperature = _real_float(supplied, "temperature_K")
    if temperature < 0:
        raise ContractError("temperature_K must be >= 0")
    try_double = supplied.get("try_doubleN")
    if not isinstance(try_double, bool):
        raise ContractError("try_doubleN must be a boolean")
    return {
        "seed": _integer(value, "seed", minimum=0),
        "threading_mode": threading_mode,
        "dmrg_cutoff": _positive_float(supplied, "dmrg_cutoff"),
        "spectral_cutoff": _positive_float(supplied, "spectral_cutoff"),
        "energy_precision_meV": _positive_float(supplied, "energy_precision_meV"),
        "highest_energy_precision_meV": _positive_float(supplied, "highest_energy_precision_meV"),
        "initial_linkdim": _integer(supplied, "initial_linkdim", minimum=1),
        "excited_states": _integer(supplied, "excited_states", minimum=0),
        "excited_state_penalty_meV": _positive_float(supplied, "excited_state_penalty_meV"),
        "temperature_K": temperature,
        "symmetry_tolerance": _positive_float(supplied, "symmetry_tolerance"),
        "omega_max_meV": _positive_float(supplied, "omega_max_meV"),
        "omega_smearing_meV": _positive_float(supplied, "omega_smearing_meV"),
        "try_doubleN": try_double,
    }


def _normalize_site_spins(value: Any) -> list[float]:
    if not isinstance(value, list) or len(value) < 3:
        raise ContractError("site_spins must be a list containing at least three spins")
    result = []
    for index, spin in enumerate(value):
        if isinstance(spin, bool) or not isinstance(spin, (int, float)) or spin <= 0:
            raise ContractError(f"site_spins[{index}] must be a positive number")
        twice_spin = round(2 * float(spin))
        if abs(2 * float(spin) - twice_spin) > 1e-12:
            raise ContractError(f"site_spins[{index}] must be integer or half-integer")
        result.append(twice_spin / 2)
    return result


def _normalize_couplings(value: Any, length: int) -> list[float]:
    if not isinstance(value, list) or len(value) != length - 1:
        raise ContractError(
            "couplings_meV must contain exactly one nearest-neighbour coupling per bond"
        )
    result = []
    for index, coupling in enumerate(value):
        if isinstance(coupling, bool) or not isinstance(coupling, (int, float)):
            raise ContractError(f"couplings_meV[{index}] must be a real number")
        result.append(float(coupling))
    return result


def normalize_ground_state_parameters(value: Mapping[str, Any]) -> dict[str, Any]:
    """Validate one registered Hamiltonian model and its numerical controls."""
    model = str(value.get("model", MODEL_IDENTIFIER)).strip()
    if model not in REGISTERED_MODELS:
        raise ContractError(
            f"model must be one of {sorted(REGISTERED_MODELS)!r}; "
            "other Hamiltonians need a registered model"
        )
    supplied = {**GROUND_STATE_DEFAULTS, **value, "model": model}
    common = _common_ground_state_parameters(supplied, value)

    if model == MODEL_IDENTIFIER:
        length = _integer(value, "length", minimum=3)
        impurity_site = _integer(value, "impurity_site", minimum=1)
        if impurity_site > length:
            raise ContractError("impurity_site cannot exceed length")
        return {
            "model": model,
            "length": length,
            "impurity_site": impurity_site,
            "J_meV": _real_float(supplied, "J_meV"),
            "JL_meV": _real_float(supplied, "JL_meV"),
            "JR_meV": _real_float(supplied, "JR_meV"),
            **common,
        }

    graph = None
    if model == GRAPH_MODEL_IDENTIFIER:
        try:
            graph = normalize_spin_graph(value)
        except (TypeError, ValueError) as exception:
            raise ContractError(str(exception)) from exception
    site_spins = graph["site_spins"] if graph is not None else _normalize_site_spins(value.get("site_spins"))
    length = len(site_spins)
    if "length" in value and _integer(value, "length", minimum=2 if graph is not None else 3) != length:
        raise ContractError("length must equal len(site_spins)")
    topology = graph if graph is not None else {"couplings_meV": _normalize_couplings(value.get("couplings_meV"), length)}
    twice_spins = [round(2 * spin) for spin in site_spins]
    target = value.get("target_twice_sz", sum(twice_spins) % 2)
    if isinstance(target, bool) or not isinstance(target, int):
        raise ContractError("target_twice_sz must be an integer")
    if abs(target) > sum(twice_spins) or (target - sum(twice_spins)) % 2:
        raise ContractError("target_twice_sz is incompatible with the supplied spins")
    return {
        "model": model,
        "length": length,
        "site_spins": site_spins,
        **topology,
        "target_twice_sz": target,
        **common,
    }


def build_sector_ensemble_plan(
    ground_state_parameters: Mapping[str, Any],
    requested_sectors: list[Any],
    spectral_resource_limits: Mapping[str, Any],
    ensemble_resources: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate sectors and derive a conservative globally bounded execution plan.

    Sectors are represented by ``2 * Sz`` so every label remains integral. The
    current ensemble implementation is deliberately limited to Hamiltonian
    adapters that expose an explicit conserved-Sz contract.
    """
    if "target_twice_sz" in ground_state_parameters:
        raise ContractError(
            "target_twice_sz must be supplied through requested_sectors for an ensemble"
        )
    normalized = normalize_ground_state_parameters(ground_state_parameters)
    if normalized["model"] not in {MIXED_MODEL_IDENTIFIER, GRAPH_MODEL_IDENTIFIER}:
        raise ContractError(f"sector ensembles currently require model {MIXED_MODEL_IDENTIFIER!r}")
    if not isinstance(requested_sectors, list) or not requested_sectors:
        raise ContractError("requested_sectors must be a non-empty list")

    total_twice_spin = sum(round(2 * spin) for spin in normalized["site_spins"])
    allowed_sectors = list(range(-total_twice_spin, total_twice_spin + 1, 2))
    sectors: list[int] = []
    for value in requested_sectors:
        if isinstance(value, bool) or not isinstance(value, int):
            raise ContractError("every requested sector must be an integer value of 2Sz")
        if value not in allowed_sectors:
            raise ContractError(f"requested 2Sz={value} is incompatible with the supplied spins")
        if value in sectors:
            raise ContractError(f"requested sector 2Sz={value} is duplicated")
        sectors.append(value)
    sectors.sort()

    spectral_limits = normalize_resource_limits(spectral_resource_limits)
    max_total_nodes = _integer(ensemble_resources, "max_total_nodes", minimum=1)
    max_concurrent = _integer(ensemble_resources, "max_concurrent_sectors", minimum=1)
    per_sector_max_nodes = min(spectral_limits["max_nodes"], max_total_nodes)
    effective_limits = {**spectral_limits, "max_nodes": per_sector_max_nodes}
    concurrency = min(
        len(sectors),
        max_concurrent,
        max_total_nodes // per_sector_max_nodes,
    )
    scan_concurrency = min(len(sectors), max_concurrent, max_total_nodes)
    if concurrency < 1:
        raise ContractError("global node ceiling cannot accommodate one sector")

    labels = [f"twice_sz_{'m' if sector < 0 else 'p'}{abs(sector)}" for sector in sectors]
    return {
        "state": "selected",
        "model": normalized["model"],
        "site_spins": normalized["site_spins"],
        "allowed_sectors": allowed_sectors,
        "requested_sectors": sectors,
        "sector_labels": labels,
        "max_total_nodes": max_total_nodes,
        "max_concurrent_sectors": max_concurrent,
        "scan_concurrent_sectors": scan_concurrency,
        "concurrent_sectors": concurrency,
        "effective_spectral_resources": effective_limits,
        "reason": (
            "one-node sector scans and selected spectral WorkChains are independent; "
            "each stage is separately capped so its concurrent node ceilings cannot "
            "exceed max_total_nodes"
        ),
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
