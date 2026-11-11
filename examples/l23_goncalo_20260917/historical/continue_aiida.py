"""One bounded standard AiiDA continuation of the September 17 L23 test."""

from datetime import datetime, timezone
import json
from pathlib import Path

from aiida import orm
from aiida.engine import submit
from aiida.plugins import WorkflowFactory

from aiida_itensor_spin.workflows import derive_resource_plan


LABEL = "L23 Goncalo September17 checkpoint continuation"
DEADLINE = datetime(2026, 9, 17, 12, 0, tzinfo=timezone.utc)
ROOT = Path("/home/jovyan/updatespin_l23_validation_20260917")


def main():
    original = orm.load_node("b77905ce-9474-446a-a7f7-bd5ceb4df292")
    segment = orm.load_node("7b8e354b-f9b6-4ea7-b0bb-8d512a5ebb01")
    if not original.is_terminated or not segment.is_finished_ok:
        raise RuntimeError("Wait for the original workflow and validated segment to finish.")
    status = segment.outputs.segment_status
    progress = status.get_dict()
    if progress["state"] != "checkpointed" or progress["total_tasks"] != 69:
        raise RuntimeError(f"Not an incomplete validated L23 checkpoint: {progress}")
    pending = progress["total_tasks"] - progress["completed_tasks"]
    if not 1 <= pending <= 6:
        raise RuntimeError(f"Unexpected remaining task count: {pending}")
    receipt = ROOT / "aiida_continuation.json"
    existing = orm.QueryBuilder().append(
        orm.WorkChainNode, filters={"label": LABEL}, project=["uuid"]
    ).all()
    if receipt.exists() or existing:
        raise RuntimeError("Refusing duplicate continuation.")
    now = datetime.now(timezone.utc)
    available = int((DEADLINE - now).total_seconds()) - 180
    walltime = min(1200, (available // 60) * 60)
    if walltime < 600:
        raise RuntimeError("Too little time before maintenance; checkpoints remain preserved.")
    limits = original.inputs.spectral_resources.get_dict()
    limits.update(wallclock_seconds=walltime, internal_stop_seconds=walltime - 180)
    limits_node = orm.Dict(dict=limits)
    parameters = original.inputs.spectral_parameters
    metadata = original.outputs.ground_state_metadata
    plan = derive_resource_plan(metadata, parameters, limits_node, status)
    layout = plan.get_dict()["selected_layout"]
    if layout["nodes"] != 1 or layout["threads_per_process"] > 2:
        raise RuntimeError(f"Unexpected continuation layout: {layout}")
    builder = WorkflowFactory("itensor_spin.spectra").get_builder()
    builder.metadata.label = LABEL
    builder.metadata.description = (
        "Standard immutable-checkpoint continuation of the L23 gold-reference test; "
        "same GS, physics, executable and 14:00 CEST scheduler deadline."
    )
    builder.code = original.inputs.spectral_code
    builder.ground_state = original.outputs.ground_state
    builder.ground_state_metadata = metadata
    builder.spectral_parameters = parameters
    builder.resource_limits = limits_node
    builder.initial_resource_plan = plan
    builder.restart_folder = segment.outputs.remote_folder
    builder.max_segments = orm.Int(1)
    builder.max_node_hours = orm.Float(0.5)
    builder.max_stalled_segments = orm.Int(1)
    builder.max_failed_segments = orm.Int(0)
    node = submit(builder)
    payload = {
        "workchain_uuid": node.uuid,
        "original_workchain_uuid": original.uuid,
        "parent_segment_uuid": segment.uuid,
        "pending_tasks": pending,
        "resource_plan": plan.get_dict(),
        "submitted_utc": now.isoformat(),
    }
    with receipt.open("x") as handle:
        json.dump(payload, handle, indent=2)
    print(f"Submitted WorkChain {node.pk} ({node.uuid})")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
