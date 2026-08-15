"""Submit the intended L=11 restart/repacking smoke test after code setup."""

from aiida import orm
from aiida.engine import submit
from aiida.plugins import WorkflowFactory

SpinChainWorkChain = WorkflowFactory("itensor_spin.full")

ground_state_code = orm.load_code("itensor-spin-gs-0.2.0-exp@daint.alps_lp83")
spectral_code = orm.load_code("itensor-spin-segment-0.2.0-exp@daint.alps_lp83")

builder = SpinChainWorkChain.get_builder()
builder.metadata.label = "ITensor L11 corrected 2N-1 fresh validation"
builder.ground_state_code = ground_state_code
builder.spectral_code = spectral_code
builder.ground_state_parameters = orm.Dict(
    dict={
        "length": 11,
        "impurity_site": 6,
        "seed": 20260724,
        "threading_mode": "blocksparse",
    }
)
builder.ground_state_layout = orm.Dict(
    dict={
        "nodes": 1,
        "processes_per_node": 1,
        "threads_per_process": 8,
        "wallclock_seconds": 900,
        "internal_stop_seconds": 780,
        "partition": "debug",
    }
)
builder.spectra.parameters = orm.Dict(
    dict={
        "expected_total_tasks": 33,
        "chebyshev_convention": "chebyshev-2n-minus-1-v2",
        "backend": "streaming",
        "threading_mode": "blocksparse",
        "ground_state_filename": "gs_checkpoint.jld2",
    }
)
builder.spectra.layouts = orm.List(
    list=[
        {
            "nodes": 2,
            "processes_per_node": 4,
            "threads_per_process": 2,
            "wallclock_seconds": 600,
            "internal_stop_seconds": 480,
            "stop_after_n": 12,
            "partition": "debug",
        },
        {
            "nodes": 2,
            "processes_per_node": 6,
            "threads_per_process": 2,
            "wallclock_seconds": 900,
            "internal_stop_seconds": 780,
            "partition": "debug",
        },
    ]
)
builder.spectra.max_segments = orm.Int(4)
builder.spectra.max_node_hours = orm.Float(2.0)
builder.spectra.max_stalled_segments = orm.Int(1)

node = submit(builder)
print(f"Submitted SpinChainWorkChain<{node.pk}> UUID={node.uuid}")

