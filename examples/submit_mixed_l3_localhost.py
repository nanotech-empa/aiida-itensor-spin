#!/usr/bin/env python3
"""Submit a nearest-neighbour mixed-spin chain to the localhost WorkChain."""

from __future__ import annotations

import argparse
import json
import os

from aiida import orm
from aiida.engine import submit
from aiida.plugins import WorkflowFactory

CODE_VERSION = "0.4.0"


def comma_floats(value: str) -> list[float]:
    try:
        return [float(item) for item in value.split(",")]
    except ValueError as exception:
        raise argparse.ArgumentTypeError("expected comma-separated numbers") from exception


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--site-spins", type=comma_floats, default=[0.5, 0.5, 0.5])
    parser.add_argument("--couplings-meV", type=comma_floats, default=[1.0, 1.0])
    parser.add_argument("--target-twice-sz", type=int)
    parser.add_argument("--seed", type=int, default=20260724)
    parser.add_argument("--omega-max-meV", type=float, default=1.5)
    parser.add_argument("--omega-smearing-meV", type=float, default=0.05)
    parser.add_argument("--excited-states", type=int, default=2)
    parser.add_argument("--max-processes", type=int, default=3)
    parser.add_argument("--max-threads", type=int, default=2)
    parser.add_argument("--gs-threads", type=int)
    parser.add_argument("--wallclock-seconds", type=int, default=900)
    parser.add_argument("--label")
    args = parser.parse_args()

    if len(args.site_spins) < 3:
        raise RuntimeError("at least three site spins are required")
    if len(args.couplings_meV) != len(args.site_spins) - 1:
        raise RuntimeError("provide exactly one coupling for each nearest-neighbour bond")

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

    internal_stop = max(60, args.wallclock_seconds - 120)
    parameters = {
        "model": "mixed_spin_heisenberg_chain_v1",
        "length": len(args.site_spins),
        "site_spins": args.site_spins,
        "couplings_meV": args.couplings_meV,
        "seed": args.seed,
        "threading_mode": "blocksparse",
        "dmrg_cutoff": 1e-12,
        "spectral_cutoff": 1e-10,
        "energy_precision_meV": 1e-10,
        "highest_energy_precision_meV": 1e-10,
        "initial_linkdim": 32,
        "excited_states": args.excited_states,
        "excited_state_penalty_meV": 1000.0,
        "temperature_K": 0.0,
        "symmetry_tolerance": 1e-8,
        "omega_max_meV": args.omega_max_meV,
        "omega_smearing_meV": args.omega_smearing_meV,
        "try_doubleN": True,
    }
    if args.target_twice_sz is not None:
        parameters["target_twice_sz"] = args.target_twice_sz

    workflow = WorkflowFactory("itensor_spin.full")
    builder = workflow.get_builder()
    builder.metadata.label = args.label or f"Mixed-spin localhost L{len(args.site_spins)}"
    builder.metadata.description = (
        "Nearest-neighbour mixed-spin Heisenberg validation using the restart-safe "
        "AiiDA/ITensor spectral workflow."
    )
    builder.ground_state_code = orm.load_code(
        f"itensor-spin-gs-{CODE_VERSION}-local@localhost"
    )
    builder.spectral_code = orm.load_code(
        f"itensor-spin-segment-{CODE_VERSION}-local@localhost"
    )
    builder.ground_state_parameters = orm.Dict(dict=parameters)
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
                "model": parameters["model"],
                "site_spins": args.site_spins,
                "couplings_meV": args.couplings_meV,
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
