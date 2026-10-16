"""BLAS resource ceilings, checkpoint compatibility, and worker launch regression checks."""

import json
import os
import subprocess

import pytest

from aiida_itensor_spin.contracts import (
    ContractError,
    normalize_parameters,
    normalize_resource_limits,
    select_largest_feasible_layout,
)

LIMITS = {
    "max_nodes": 30,
    "max_processes_per_node": 3,
    "max_threads_per_process": 8,
    "wallclock_seconds": 50400,
    "internal_stop_seconds": 43200,
    "partition": "normal",
}
PARAMETERS = {
    "chebyshev_convention": "chebyshev-2n-minus-1-v2",
    "threading_mode": "blocksparse",
    "backend": "streaming",
    "spectral_cutoff": 1e-8,
    "max_bond_dimension": 8000,
    "kernel_threading_mode": "blas",
    "blas_threads": 8,
}


def test_explicit_eight_cores_survive_replanning_and_memory_backoff():
    limits = normalize_resource_limits({**LIMITS, "threads_per_process": 8})
    for pending in (12, 11, 7, 1):
        for density in (3, 2, 1):
            plan = select_largest_feasible_layout(
                pending, {**limits, "max_processes_per_node": density}
            )
            assert plan["threads_per_process"] == 8
            assert plan["processes_per_node"] <= density
    assert select_largest_feasible_layout(12, limits)["nodes"] == 4
    assert select_largest_feasible_layout(12, LIMITS)["threads_per_process"] == 2


@pytest.mark.parametrize(
    "override",
    [
        {"threads_per_process": 9},
        {"threads_per_process": True},
        {"threads_per_process": 0},
        {"threads_per_process": 8, "max_processes_per_node": 40},
    ],
)
def test_explicit_core_limits_are_enforced(override):
    with pytest.raises(ContractError):
        normalize_resource_limits({**LIMITS, **override})


def test_blas_policy_keeps_original_restart_identity_and_scientific_inputs():
    normalized = normalize_parameters(PARAMETERS)
    assert all(normalized[key] == value for key, value in PARAMETERS.items())
    for override in (
        {"blas_threads": True},
        {"blas_threads": 0},
        {"threading_mode": "outer"},
        {"kernel_threading_mode": "unknown"},
    ):
        with pytest.raises(ContractError):
            normalize_parameters({**PARAMETERS, **override})


def test_actual_rank_wrapper_launches_one_julia_thread_and_allocates_eight_cores(tmp_path):
    manifests = tmp_path / "manifests"
    manifests.mkdir()
    (manifests / "worker_0.txt").write_text("73\n")
    executable = tmp_path / "fake-julia"
    executable.write_text(
        "#!/usr/bin/env python3\nimport os,json\nprint(json.dumps({k:os.environ[k] for k in ['JULIA_NUM_THREADS','SLURM_CPUS_PER_TASK','ITENSOR_SPIN_BLAS_THREADS']}))\n"
    )
    executable.chmod(0o755)
    env = {
        **os.environ,
        "SLURM_PROCID": "0",
        "SLURM_NTASKS": "12",
        "SLURM_CPUS_PER_TASK": "8",
        "ITENSOR_SPIN_DEADLINE_EPOCH": "9999999999",
        "ITENSOR_SPIN_MANIFEST_ROOT": str(manifests),
        "ITENSOR_SPIN_JULIA": str(executable),
        "ITENSOR_SPIN_KERNEL_THREADING_MODE": "blas",
        "ITENSOR_SPIN_BLAS_THREADS": "8",
    }
    cmd = [
        "bash",
        "remote/run_segment_rank.sh",
        "checkpoint",
        str(tmp_path / "workers"),
        "engine",
        "streaming",
        "blocksparse",
    ]
    result = subprocess.run(cmd, env=env, text=True, capture_output=True, check=True)
    assert json.loads(result.stdout.splitlines()[-1]) == {
        "JULIA_NUM_THREADS": "1,0",
        "SLURM_CPUS_PER_TASK": "8",
        "ITENSOR_SPIN_BLAS_THREADS": "8",
    }
    env["SLURM_CPUS_PER_TASK"] = "2"
    assert subprocess.run(cmd, env=env, capture_output=True, check=False).returncode != 0
