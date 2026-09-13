"""Submit the v6 L=11 checkpoint/repacking acceptance workflow."""

from aiida import orm
from aiida.engine import submit
from aiida.plugins import WorkflowFactory

SpinChainWorkChain = WorkflowFactory("itensor_spin.full")

builder = SpinChainWorkChain.get_builder()
builder.metadata.label = "ITensor L11 v6 scalable recovery acceptance"
builder.ground_state_code = orm.load_code("itensor-spin-gs-0.3.0-v4@daint.alps_lp83")
builder.spectral_code = orm.load_code("itensor-spin-segment-0.3.0-v6@daint.alps_lp83")
builder.ground_state_parameters = orm.Dict(
    dict={
        "length": 11,
        "impurity_site": 6,
        "seed": 20260724,
        "threading_mode": "blocksparse",
    }
)
builder.ground_state_resources = orm.Dict(
    dict={
        "nodes": 1,
        "processes_per_node": 1,
        "threads_per_process": 8,
        "wallclock_seconds": 900,
        "internal_stop_seconds": 780,
        "partition": "debug",
    }
)
builder.spectral_parameters = orm.Dict(
    dict={
        "chebyshev_convention": "chebyshev-2n-minus-1-v2",
        "backend": "streaming",
        "threading_mode": "blocksparse",
        "ground_state_filename": "gs_checkpoint.jld2",
        "checkpoint_every": 1,
        "heartbeat_every": 1,
    }
)
builder.spectral_resources = orm.Dict(
    dict={
        "max_nodes": 2,
        "max_processes_per_node": 6,
        "max_threads_per_process": 2,
        "wallclock_seconds": 360,
        "internal_stop_seconds": 240,
        "partition": "debug",
    }
)
builder.max_segments = orm.Int(6)
builder.max_node_hours = orm.Float(2.0)
builder.max_stalled_segments = orm.Int(2)
builder.max_failed_segments = orm.Int(1)

node = submit(builder)
print(f"Submitted SpinChainWorkChain<{node.pk}> UUID={node.uuid}")
