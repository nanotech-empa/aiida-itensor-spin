"""Continue the corrected L=101 spectrum from the latest validated v5 checkpoint using explicit v6 task manifests."""

from aiida import orm
from aiida.engine import submit
from aiida.plugins import WorkflowFactory

from aiida_itensor_spin.workflows import derive_resource_plan


SpinSpectraWorkChain = WorkflowFactory("itensor_spin.spectra")

ground_state = orm.load_node("1d875abe-827a-4d80-a226-b698788e07a2")
ground_state_metadata = orm.load_node("90820699-6464-41b3-964b-3c4e74333cc5")
restart_folder = orm.load_node("d07e1c0b-4f00-4f4b-a8ed-e62a3e092c80")
restart_status = orm.load_node("759a398b-df43-4821-9f96-af85ff22e6b2")
spectral_code = orm.load_code("itensor-spin-segment-0.3.0-v6@daint.alps_lp83")

spectral_parameters = orm.Dict(
    dict={
        "chebyshev_convention": "chebyshev-2n-minus-1-v2",
        "backend": "streaming",
        "threading_mode": "blocksparse",
        "ground_state_filename": "gs_checkpoint.jld2",
        "checkpoint_every": 1,
        "heartbeat_every": 1,
    }
)
resource_limits = orm.Dict(
    dict={
        "max_nodes": 70,
        "max_processes_per_node": 8,
        "max_threads_per_process": 4,
        "wallclock_seconds": 43200,
        "internal_stop_seconds": 39600,
        "partition": "normal",
    }
)
initial_resource_plan = derive_resource_plan(
    ground_state_metadata,
    spectral_parameters,
    resource_limits,
    restart_status,
)
plan = initial_resource_plan.get_dict()
if plan.get("pending_tasks") != 40:
    raise RuntimeError(
        "Refusing continuation: expected 40 unfinished tasks, found "
        f"{plan.get('pending_tasks')}."
    )

layout = plan["selected_layout"]
if (layout["nodes"], layout["processes_per_node"], plan["workers"]) != (5, 8, 40):
    raise RuntimeError(f"Refusing unexpected dense layout: {layout}, workers={plan['workers']}")
print(
    "Validated checkpoint: "
    f"{plan['completed_tasks']}/{plan['total_tasks']} tasks complete; "
    f"continuing {plan['pending_tasks']} tasks with "
    f"{layout['nodes']} nodes x {layout['processes_per_node']} processes/node x "
    f"{layout['threads_per_process']} threads/process."
)

builder = SpinSpectraWorkChain.get_builder()
builder.metadata.label = "ITensor L101 v6 explicit unfinished-task continuation"
builder.code = spectral_code
builder.ground_state = ground_state
builder.ground_state_metadata = ground_state_metadata
builder.spectral_parameters = spectral_parameters
builder.resource_limits = resource_limits
builder.initial_resource_plan = initial_resource_plan
builder.restart_folder = restart_folder
builder.max_segments = orm.Int(20)
builder.max_node_hours = orm.Float(2000.0)
builder.max_stalled_segments = orm.Int(2)
builder.max_failed_segments = orm.Int(1)

node = submit(builder)
print(f"Submitted SpinSpectraWorkChain<{node.pk}> UUID={node.uuid}")
