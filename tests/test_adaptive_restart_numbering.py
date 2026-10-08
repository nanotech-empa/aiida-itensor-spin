"""Segment identity must survive AiiDA's resettable approval-window counter."""

import io

import pytest
from aiida import orm
from aiida.engine.utils import instantiate_process
from aiida.manage import get_manager

from aiida_itensor_spin.workflows import AdaptiveSpinSpectraWorkChain

pytest_plugins = ["aiida.tools.pytest_fixtures"]


@pytest.mark.parametrize("completed_segments", [0, 1, 4, 8])
def test_segment_identity_uses_cumulative_children(aiida_profile, aiida_localhost, completed_segments):
    layout = {
        "nodes": 1, "processes_per_node": 1, "threads_per_process": 2,
        "wallclock_seconds": 1800, "internal_stop_seconds": 1500,
        "partition": "local",
    }
    plan = {"state": "selected", "selected_layout": layout, "parameters": {}}
    process = instantiate_process(
        get_manager().get_runner(),
        AdaptiveSpinSpectraWorkChain,
        code=orm.InstalledCode(computer=aiida_localhost, filepath_executable="/bin/true"),
        ground_state=orm.SinglefileData(file=io.BytesIO(b"not evaluated in this control-flow test")),
        ground_state_metadata=orm.Dict(dict={}),
        spectral_parameters=orm.Dict(dict={}),
        resource_limits=orm.Dict(dict={}),
        initial_resource_plan=orm.Dict(dict=plan),
    )
    try:
        process.ctx.current_plan = plan
        process.ctx.current_plan_node = process.inputs.initial_resource_plan
        previous_folder = (
            orm.RemoteData(computer=aiida_localhost, remote_path="/tmp/preserved-checkpoint")
            if completed_segments else None
        )
        process.ctx.previous_folder = previous_folder
        process.ctx.iteration = 0  # Exactly the state after the max-iteration pause.
        process.ctx.children = [orm.CalcJobNode() for _ in range(completed_segments)]
        process._prepare_process_inputs()
        assert process.ctx.inputs["segment_index"].value == completed_segments
        assert process.ctx.inputs["metadata"]["label"] == (
            f"ITensor adaptive spectral segment {completed_segments + 1}"
        )
        assert process.ctx.inputs["metadata"]["options"]["max_wallclock_seconds"] == 1800
        assert process.ctx.inputs["metadata"]["options"]["resources"] == {
            "num_machines": 1, "num_mpiprocs_per_machine": 1, "num_cores_per_mpiproc": 2,
        }
        assert process.ctx.iteration == 0
        assert len(process.ctx.children) == completed_segments
        if previous_folder is not None:
            assert process.ctx.inputs["parent_folder"] is previous_folder
        else:
            assert "parent_folder" not in process.ctx.inputs
    finally:
        process.close()
