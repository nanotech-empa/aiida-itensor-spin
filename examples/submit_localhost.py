"""Submit the versioned spin-1 impurity model to the localhost computer."""

from __future__ import annotations

import argparse
import json
import os

from aiida import orm
from aiida.engine import submit
from aiida.plugins import WorkflowFactory

CODE_VERSION = "0.4.0"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--length", type=int, default=23)
    parser.add_argument("--impurity-site", type=int)
    parser.add_argument("--seed", type=int, default=20260724)
    parser.add_argument("--J-meV", type=float, default=18.4)
    parser.add_argument("--JL-meV", type=float, default=16.7)
    parser.add_argument("--JR-meV", type=float, default=51.1)
    parser.add_argument("--temperature-K", type=float, default=4.0)
    parser.add_argument("--omega-max-meV", type=float, default=100.0)
    parser.add_argument("--omega-smearing-meV", type=float, default=1.5)
    parser.add_argument("--dmrg-cutoff", type=float, default=1e-8)
    parser.add_argument("--spectral-cutoff", type=float, default=1e-6)
    parser.add_argument("--max-processes", type=int, default=4)
    parser.add_argument("--max-threads", type=int, default=2)
    parser.add_argument("--gs-threads", type=int)
    parser.add_argument("--wallclock-seconds", type=int, default=10800)
    parser.add_argument("--label")
    args = parser.parse_args()

    available_cpus = len(os.sched_getaffinity(0))
    gs_threads = args.gs_threads or available_cpus
    if args.max_processes * args.max_threads > available_cpus:
        raise RuntimeError(
            f"requested {args.max_processes * args.max_threads} spectral CPU threads; "
            f"container affinity provides {available_cpus}"
        )
    if gs_threads > available_cpus:
        raise RuntimeError(
            f"requested {gs_threads} GS threads; container affinity provides {available_cpus}"
        )

    impurity_site = args.impurity_site or (args.length + 1) // 2
    internal_stop = max(60, args.wallclock_seconds - 300)
    workflow = WorkflowFactory("itensor_spin.full")
    builder = workflow.get_builder()
    builder.metadata.label = args.label or f"ITensor localhost L{args.length}"
    builder.metadata.description = (
        "Local ARM64 validation of the versioned spin-1 chain with spin-1/2 impurity. "
        "Uses the same corrected checkpointed AiiDA workflow as the Daint calculation."
    )
    builder.ground_state_code = orm.load_code(
        f"itensor-spin-gs-{CODE_VERSION}-local@localhost"
    )
    builder.spectral_code = orm.load_code(
        f"itensor-spin-segment-{CODE_VERSION}-local@localhost"
    )
    builder.ground_state_parameters = orm.Dict(
        dict={
            "model": "spin1_chain_spinhalf_impurity_v1",
            "length": args.length,
            "impurity_site": impurity_site,
            "seed": args.seed,
            "threading_mode": "blocksparse",
            "J_meV": args.J_meV,
            "JL_meV": args.JL_meV,
            "JR_meV": args.JR_meV,
            "dmrg_cutoff": args.dmrg_cutoff,
            "spectral_cutoff": args.spectral_cutoff,
            "energy_precision_meV": 1e-5,
            "highest_energy_precision_meV": 1e-3,
            "initial_linkdim": 100,
            "excited_states": 2,
            "excited_state_penalty_meV": 1000.0,
            "temperature_K": args.temperature_K,
            "symmetry_tolerance": 1e-2,
            "omega_max_meV": args.omega_max_meV,
            "omega_smearing_meV": args.omega_smearing_meV,
            "try_doubleN": True,
        }
    )
    builder.ground_state_resources = orm.Dict(
        dict={
            "nodes": 1,
            "processes_per_node": 1,
            "threads_per_process": gs_threads,
            "wallclock_seconds": args.wallclock_seconds,
            "internal_stop_seconds": internal_stop,
            "partition": "local",
        }
    )
    builder.spectral_parameters = orm.Dict(
        dict={
            "chebyshev_convention": "chebyshev-2n-minus-1-v2",
            "backend": "streaming",
            "threading_mode": "blocksparse",
            "ground_state_filename": "gs_checkpoint.jld2",
            "checkpoint_every": 10,
            "heartbeat_every": 1,
        }
    )
    builder.spectral_resources = orm.Dict(
        dict={
            "max_nodes": 1,
            "max_processes_per_node": args.max_processes,
            "max_threads_per_process": args.max_threads,
            "wallclock_seconds": args.wallclock_seconds,
            "internal_stop_seconds": internal_stop,
            "partition": "local",
        }
    )
    builder.max_segments = orm.Int(6)
    builder.max_node_hours = orm.Float(24.0)
    builder.max_stalled_segments = orm.Int(1)
    builder.max_failed_segments = orm.Int(1)

    node = submit(builder)
    print(
        json.dumps(
            {
                "pk": node.pk,
                "uuid": str(node.uuid),
                "label": node.label,
                "length": args.length,
                "impurity_site": impurity_site,
                "available_cpus": available_cpus,
                "gs_threads": gs_threads,
                "max_processes": args.max_processes,
                "max_threads": args.max_threads,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
