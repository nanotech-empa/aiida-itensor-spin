"""Submit the corrected L=51 CPU qualification workflow."""

from aiida import orm
from aiida.engine import submit
from aiida.plugins import WorkflowFactory

SpinChainWorkChain = WorkflowFactory("itensor_spin.full")

ground_state_code = orm.load_code("itensor-spin-gs-0.3.0-v4@daint.alps_lp83")
spectral_code = orm.load_code("itensor-spin-segment-0.3.0-v4@daint.alps_lp83")

builder = SpinChainWorkChain.get_builder()
builder.metadata.label = "ITensor L51 corrected CPU qualification"
builder.ground_state_code = ground_state_code
builder.spectral_code = spectral_code
builder.ground_state_parameters = orm.Dict(
    dict={
        "length": 51,
        "impurity_site": 26,
        "seed": 20260724,
        "threading_mode": "blocksparse",
    }
)
builder.ground_state_resources = orm.Dict(
    dict={
        "nodes": 1,
        "processes_per_node": 1,
        "threads_per_process": 8,
        "wallclock_seconds": 43200,
        "internal_stop_seconds": 42300,
        "partition": "normal",
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
        "max_nodes": 20,
        "max_processes_per_node": 12,
        "max_threads_per_process": 4,
        "wallclock_seconds": 43200,
        "internal_stop_seconds": 42300,
        "partition": "normal",
    }
)
builder.max_segments = orm.Int(6)
builder.max_node_hours = orm.Float(480.0)
builder.max_stalled_segments = orm.Int(2)

node = submit(builder)
print(f"Submitted SpinChainWorkChain<{node.pk}> UUID={node.uuid}")
