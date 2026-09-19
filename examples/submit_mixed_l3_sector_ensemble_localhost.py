#!/usr/bin/env python3
"""Scan all L=3 sectors and calculate spectra only for global GS sectors."""

from __future__ import annotations

import json

from aiida import orm
from aiida.engine import submit
from aiida.plugins import WorkflowFactory

CODE_VERSION = "0.4.0"


def main():
    workflow = WorkflowFactory("itensor_spin.sector_ensemble")
    builder = workflow.get_builder()
    builder.metadata.label = "Mixed-spin L3 all-sector scan with selected spectra"
    builder.metadata.description = (
        "DMRG-scan all four fixed-Sz sectors, then run spectra only for the "
        "global/near-degenerate ground-state sectors."
    )
    builder.ground_state_code = orm.load_code(f"itensor-spin-gs-{CODE_VERSION}-local@localhost")
    builder.spectral_code = orm.load_code(f"itensor-spin-segment-{CODE_VERSION}-local@localhost")
    builder.ground_state_parameters = orm.Dict(
        dict={
            "model": "mixed_spin_heisenberg_chain_v1",
            "length": 3,
            "site_spins": [0.5, 0.5, 0.5],
            "couplings_meV": [1.0, 1.0],
            "seed": 20260724,
            "threading_mode": "blocksparse",
            "dmrg_cutoff": 1e-12,
            "spectral_cutoff": 1e-10,
            "energy_precision_meV": 1e-10,
            "highest_energy_precision_meV": 1e-10,
            "initial_linkdim": 32,
            "excited_states": 2,
            "excited_state_penalty_meV": 1000.0,
            "temperature_K": 0.0,
            "symmetry_tolerance": 1e-8,
            "omega_max_meV": 1.4,
            "omega_smearing_meV": 0.05,
            "try_doubleN": True,
        }
    )
    builder.requested_sectors = orm.List(list=[-3, -1, 1, 3])
    builder.ground_state_resources = orm.Dict(
        dict={
            "nodes": 1,
            "processes_per_node": 1,
            "threads_per_process": 1,
            "wallclock_seconds": 900,
            "internal_stop_seconds": 780,
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
            "max_processes_per_node": 1,
            "max_threads_per_process": 1,
            "wallclock_seconds": 900,
            "internal_stop_seconds": 780,
            "partition": "local",
        }
    )
    builder.ensemble_resources = orm.Dict(dict={"max_total_nodes": 4, "max_concurrent_sectors": 4})
    builder.max_segments = orm.Int(6)
    builder.max_node_hours = orm.Float(24.0)
    builder.max_stalled_segments = orm.Int(1)
    builder.max_failed_segments = orm.Int(1)

    node = submit(builder)
    print(json.dumps({"pk": node.pk, "uuid": str(node.uuid)}, indent=2))


if __name__ == "__main__":
    main()
