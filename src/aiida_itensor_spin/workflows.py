"""Restart and repacking orchestration for spin-chain spectral tasks."""

from __future__ import annotations

from aiida import orm
from aiida.engine import ToContext, WorkChain, calcfunction, while_

from .calculations import GroundStateCalculation, SpectralSegmentCalculation
from .contracts import (
    ContractError,
    add_discovered_task_count,
    normalize_layout,
    normalize_parameters,
    normalize_resource_limits,
    select_largest_feasible_layout,
    validate_segment_status,
)


def _calculation_layout(selected: dict) -> dict:
    """Remove resource-policy annotations from a concrete CalcJob layout."""
    return {key: value for key, value in selected.items() if key not in {"workers"}}


def _task_distribution(total_tasks: int, workers: int) -> tuple[int, int, int]:
    """Return minimum/maximum tasks per worker and number receiving one extra."""
    minimum, extra_workers = divmod(total_tasks, workers)
    maximum = minimum + (1 if extra_workers else 0)
    return minimum, maximum, extra_workers


@calcfunction
def derive_resource_plan(
    ground_state_metadata: orm.Dict,
    spectral_parameters: orm.Dict,
    resource_limits: orm.Dict,
    previous_status: orm.Dict | None = None,
) -> orm.Dict:
    """Derive one provenance-recorded spectral allocation from GS and progress data."""
    try:
        metadata = ground_state_metadata.get_dict()
        parameters = add_discovered_task_count(spectral_parameters.get_dict(), metadata)
        limits = normalize_resource_limits(resource_limits.get_dict())
        completed_tasks = 0
        if previous_status is not None:
            status = validate_segment_status(previous_status.get_dict(), parameters)
            completed_tasks = status["completed_tasks"]
        total_tasks = parameters["expected_total_tasks"]
        pending_tasks = total_tasks - completed_tasks
        if pending_tasks < 1:
            raise ContractError("resource planning requires at least one unfinished task")
        selected = select_largest_feasible_layout(pending_tasks, limits)
    except (KeyError, ContractError, TypeError, ValueError) as exception:
        return orm.Dict(dict={"state": "rejected", "reason": str(exception)})

    workers = selected["workers"]
    minimum, maximum, extra_workers = _task_distribution(pending_tasks, workers)
    return orm.Dict(
        dict={
            "state": "selected",
            "reason": (
                "largest uniform worker layout within the supplied ceilings "
                "while minimizing nodes when worker counts tie; threads per process are capped at 2 based on the validated CPU scaling benchmark"
            ),
            "length": metadata["length"],
            "ground_states": metadata["ground_states"],
            "total_tasks": total_tasks,
            "completed_tasks": completed_tasks,
            "pending_tasks": pending_tasks,
            "parameters": parameters,
            "resource_limits": limits,
            "selected_layout": _calculation_layout(selected),
            "workers": workers,
            "minimum_tasks_per_process": minimum,
            "maximum_tasks_per_process": maximum,
            "processes_with_extra_task": extra_workers,
        }
    )


class SpinSpectraWorkChain(WorkChain):
    """Run sequential, bounded, restartable spectral CalcJobs until completion."""

    @classmethod
    def define(cls, spec):
        super().define(spec)
        spec.input("code", valid_type=orm.AbstractCode)
        spec.input("ground_state", valid_type=orm.SinglefileData)
        spec.input("ground_state_metadata", valid_type=orm.Dict)
        spec.input("spectral_parameters", valid_type=orm.Dict)
        spec.input("resource_limits", valid_type=orm.Dict)
        spec.input("initial_resource_plan", valid_type=orm.Dict)
        spec.input(
            "restart_folder",
            valid_type=orm.RemoteData,
            required=False,
            help="Immutable state folder from a previous spectral segment.",
        )
        spec.input("max_segments", valid_type=orm.Int, default=lambda: orm.Int(6))
        spec.input("max_node_hours", valid_type=orm.Float, default=lambda: orm.Float(100.0))
        spec.input("max_stalled_segments", valid_type=orm.Int, default=lambda: orm.Int(1))
        spec.input("max_failed_segments", valid_type=orm.Int, default=lambda: orm.Int(1))
        spec.outline(
            cls.setup,
            while_(cls.should_run)(cls.submit_segment, cls.inspect_segment),
            cls.finalize,
        )
        spec.output("final_status", valid_type=orm.Dict, required=False)
        spec.output("final_remote_folder", valid_type=orm.RemoteData, required=False)
        spec.output("final_segment_summary", valid_type=orm.Dict, required=False)
        spec.output("final_resource_plan", valid_type=orm.Dict, required=False)
        spec.output("spectrum", valid_type=orm.SinglefileData, required=False)
        spec.output("collection_report", valid_type=orm.SinglefileData, required=False)
        spec.output("task_timings", valid_type=orm.SinglefileData, required=False)
        spec.exit_code(400, "ERROR_INVALID_INPUT", message="Invalid spectral workflow input.")
        spec.exit_code(401, "ERROR_SEGMENT_FAILED", message="A spectral CalcJob failed.")
        spec.exit_code(402, "ERROR_INVALID_PROGRESS", message="Invalid segment progress report.")
        spec.exit_code(403, "ERROR_STALLED", message="No progress in consecutive segments.")
        spec.exit_code(404, "ERROR_MAX_SEGMENTS", message="Maximum segment count reached.")
        spec.exit_code(405, "ERROR_NODE_HOURS", message="Node-hour safety limit reached.")
        spec.exit_code(
            406,
            "ERROR_FAILED_SEGMENTS",
            message="Consecutive failed-segment recovery limit reached.",
        )

    def setup(self):
        try:
            initial_plan = self.inputs.initial_resource_plan.get_dict()
            if initial_plan.get("state") != "selected":
                raise ContractError(
                    f"initial resource plan was rejected: {initial_plan.get('reason')}"
                )
            normalize_parameters(initial_plan["parameters"])
            normalize_layout(initial_plan["selected_layout"])
            normalize_resource_limits(self.inputs.resource_limits.get_dict())
            if self.inputs.max_segments.value < 1:
                raise ContractError("max_segments must be >= 1")
            if self.inputs.max_node_hours.value <= 0:
                raise ContractError("max_node_hours must be > 0")
            if self.inputs.max_stalled_segments.value < 1:
                raise ContractError("max_stalled_segments must be >= 1")
            if self.inputs.max_failed_segments.value < 0:
                raise ContractError("max_failed_segments must be >= 0")
        except (KeyError, ContractError, TypeError, ValueError) as exception:
            self.report(f"invalid spectral workflow input: {exception}")
            return self.exit_codes.ERROR_INVALID_INPUT

        self.ctx.segment_count = 0
        self.ctx.node_hours = 0.0
        self.ctx.stalled_segments = 0
        self.ctx.failed_segments = 0
        self.ctx.previous_status = None
        self.ctx.previous_status_node = None
        self.ctx.previous_summary_node = None
        self.ctx.previous_folder = (
            self.inputs.restart_folder if "restart_folder" in self.inputs else None
        )
        self.ctx.result_nodes = {}
        self.ctx.complete = False
        self.ctx.limit = None
        self.ctx.plan_error = None

    def should_run(self):
        if self.ctx.complete:
            return False
        if self.ctx.segment_count >= self.inputs.max_segments.value:
            self.ctx.limit = "max_segments"
            return False

        if self.ctx.segment_count == 0:
            plan_node = self.inputs.initial_resource_plan
        else:
            plan_node = derive_resource_plan(
                self.inputs.ground_state_metadata,
                self.inputs.spectral_parameters,
                self.inputs.resource_limits,
                self.ctx.previous_status_node,
            )
        plan = plan_node.get_dict()
        if plan.get("state") != "selected":
            self.ctx.plan_error = plan.get("reason", "unknown resource-planning error")
            return False

        layout = plan["selected_layout"]
        projected_node_hours = (
            self.ctx.node_hours + layout["nodes"] * layout["wallclock_seconds"] / 3600.0
        )
        if projected_node_hours > self.inputs.max_node_hours.value:
            self.ctx.limit = "node_hours"
            return False

        self.ctx.current_plan_node = plan_node
        self.ctx.current_plan = plan
        return True

    def submit_segment(self):
        plan = self.ctx.current_plan
        layout = plan["selected_layout"]
        self.report(
            f"segment {self.ctx.segment_count + 1}: {plan['pending_tasks']} unfinished "
            f"task(s); selected {layout['nodes']} node(s) x "
            f"{layout['processes_per_node']} processes/node x "
            f"{layout['threads_per_process']} threads/process = {plan['workers']} "
            f"workers. Each worker receives {plan['minimum_tasks_per_process']} to "
            f"{plan['maximum_tasks_per_process']} task(s). Reason: {plan['reason']}."
        )
        inputs = {
            "code": self.inputs.code,
            "ground_state": self.inputs.ground_state,
            "resource_plan": self.ctx.current_plan_node,
            "segment_index": orm.Int(self.ctx.segment_count),
            "metadata": {
                "label": f"ITensor spectral segment {self.ctx.segment_count + 1}",
                "options": {
                    "resources": {
                        "num_machines": layout["nodes"],
                        "num_mpiprocs_per_machine": layout["processes_per_node"],
                        "num_cores_per_mpiproc": layout["threads_per_process"],
                    },
                    "max_wallclock_seconds": layout["wallclock_seconds"],
                    "queue_name": layout["partition"],
                    "withmpi": False,
                },
            },
        }
        if self.ctx.previous_folder is not None:
            inputs["parent_folder"] = self.ctx.previous_folder

        future = self.submit(SpectralSegmentCalculation, **inputs)
        self.ctx.segment_count += 1
        self.ctx.node_hours += layout["nodes"] * layout["wallclock_seconds"] / 3600.0
        self.report(f"submitted spectral CalcJob<{future.pk}>")
        return ToContext(segment=future)

    def inspect_segment(self):
        child = self.ctx.segment
        if "segment_status" not in child.outputs:
            if "remote_folder" in child.outputs:
                self.ctx.previous_folder = child.outputs.remote_folder
            if (
                "remote_folder" in child.outputs
                and self.ctx.failed_segments < self.inputs.max_failed_segments.value
            ):
                self.ctx.failed_segments += 1
                self.report(
                    f"spectral CalcJob<{child.pk}> produced no status (exit status "
                    f"{child.exit_status}); retrying from its preserved remote folder "
                    f"({self.ctx.failed_segments}/{self.inputs.max_failed_segments.value})"
                )
                return None
            self.report(
                f"spectral CalcJob<{child.pk}> failed without a usable status "
                f"(exit status {child.exit_status})"
            )
            self._emit_outputs()
            return self.exit_codes.ERROR_FAILED_SEGMENTS

        parameters = self.ctx.current_plan["parameters"]
        try:
            status = validate_segment_status(
                child.outputs.segment_status.get_dict(),
                parameters,
                self.ctx.previous_status,
            )
        except ContractError as exception:
            self.ctx.previous_folder = child.outputs.remote_folder
            self.report(f"invalid segment progress: {exception}")
            self._emit_outputs()
            return self.exit_codes.ERROR_INVALID_PROGRESS

        self.ctx.stalled_segments = 0 if status["made_progress"] else self.ctx.stalled_segments + 1
        self.ctx.failed_segments = 0
        self.ctx.previous_status = status
        self.ctx.previous_status_node = child.outputs.segment_status
        self.ctx.previous_summary_node = child.outputs.segment_summary
        self.ctx.previous_folder = child.outputs.remote_folder
        self.ctx.complete = status["state"] == "complete"
        if not child.is_finished_ok:
            self.report(
                f"spectral CalcJob<{child.pk}> exited with status {child.exit_status}, "
                "but supplied a valid checkpoint status; continuing safely"
            )
        for label in ("spectrum", "collection_report", "task_timings"):
            if label in child.outputs:
                self.ctx.result_nodes[label] = child.outputs[label]

        self.report(
            f"segment {self.ctx.segment_count}: {status['completed_tasks']}/"
            f"{status['total_tasks']} tasks complete; recurrence progress "
            f"{status['progress_units_completed']}/{status['progress_units_total']}"
        )
        if self.ctx.stalled_segments >= self.inputs.max_stalled_segments.value:
            self.report("segment made no measurable task or recurrence progress")
            self._emit_outputs()
            return self.exit_codes.ERROR_STALLED

    def _emit_outputs(self):
        if "current_plan_node" in self.ctx:
            self.out("final_resource_plan", self.ctx.current_plan_node)
        if self.ctx.previous_status_node is not None:
            self.out("final_status", self.ctx.previous_status_node)
        if self.ctx.previous_folder is not None:
            self.out("final_remote_folder", self.ctx.previous_folder)
        if self.ctx.previous_summary_node is not None:
            self.out("final_segment_summary", self.ctx.previous_summary_node)
        for label, node in self.ctx.result_nodes.items():
            self.out(label, node)

    def finalize(self):
        self._emit_outputs()
        if self.ctx.plan_error is not None:
            self.report(f"resource planning rejected: {self.ctx.plan_error}")
            return self.exit_codes.ERROR_INVALID_INPUT
        if self.ctx.complete:
            missing = {"spectrum", "collection_report", "task_timings"} - set(self.ctx.result_nodes)
            if missing:
                self.report(
                    "completed calculation is missing parsed result files: "
                    + ", ".join(sorted(missing))
                )
                return self.exit_codes.ERROR_SEGMENT_FAILED
            return None
        if self.ctx.limit == "node_hours":
            return self.exit_codes.ERROR_NODE_HOURS
        return self.exit_codes.ERROR_MAX_SEGMENTS


class SpinChainWorkChain(WorkChain):
    """Complete GS-to-spectrum workflow with adaptive, restartable execution."""

    @classmethod
    def define(cls, spec):
        super().define(spec)
        spec.input("ground_state_code", valid_type=orm.AbstractCode)
        spec.input("spectral_code", valid_type=orm.AbstractCode)
        spec.input("ground_state_parameters", valid_type=orm.Dict)
        spec.input(
            "ground_state_resources",
            valid_type=orm.Dict,
            help="Single-node Slurm resources for the ground-state CalcJob.",
        )
        spec.input("spectral_parameters", valid_type=orm.Dict)
        spec.input(
            "spectral_resources",
            valid_type=orm.Dict,
            help=(
                "Ceilings for adaptive spectral execution: max nodes, max processes "
                "per node, maximum threads per process, wallclock, and partition."
            ),
        )
        spec.input("max_segments", valid_type=orm.Int, default=lambda: orm.Int(6))
        spec.input("max_node_hours", valid_type=orm.Float, default=lambda: orm.Float(100.0))
        spec.input("max_stalled_segments", valid_type=orm.Int, default=lambda: orm.Int(1))
        spec.input("max_failed_segments", valid_type=orm.Int, default=lambda: orm.Int(1))
        spec.outline(
            cls.setup,
            cls.submit_ground_state,
            cls.inspect_ground_state,
            cls.prepare_spectra,
            cls.submit_spectra,
            cls.inspect_spectra,
        )
        spec.output("ground_state", valid_type=orm.SinglefileData)
        spec.output("ground_state_metadata", valid_type=orm.Dict)
        spec.output("ground_state_summary", valid_type=orm.SinglefileData)
        spec.output("final_status", valid_type=orm.Dict, required=False)
        spec.output("final_remote_folder", valid_type=orm.RemoteData, required=False)
        spec.output("resource_plan", valid_type=orm.Dict, required=False)
        spec.output("final_resource_plan", valid_type=orm.Dict, required=False)
        spec.output("final_segment_summary", valid_type=orm.Dict, required=False)
        spec.output("spectrum", valid_type=orm.SinglefileData, required=False)
        spec.output("collection_report", valid_type=orm.SinglefileData, required=False)
        spec.output("task_timings", valid_type=orm.SinglefileData, required=False)
        spec.exit_code(500, "ERROR_INVALID_INPUT", message="Invalid full-workflow input.")
        spec.exit_code(501, "ERROR_GROUND_STATE_FAILED", message="Ground-state CalcJob failed.")
        spec.exit_code(
            502,
            "ERROR_GROUND_STATE_METADATA",
            message="Ground-state metadata is missing or inconsistent.",
        )
        spec.exit_code(503, "ERROR_SPECTRA_FAILED", message="Spectral WorkChain failed.")

    def setup(self):
        try:
            self.ctx.ground_state_layout = normalize_layout(
                self.inputs.ground_state_resources.get_dict()
            )
            if (
                self.ctx.ground_state_layout["nodes"] != 1
                or self.ctx.ground_state_layout["processes_per_node"] != 1
            ):
                raise ContractError(
                    "ground-state resources require exactly one node and one process"
                )
            self.ctx.spectral_parameters = normalize_parameters(
                self.inputs.spectral_parameters.get_dict()
            )
            self.ctx.spectral_resource_limits = normalize_resource_limits(
                self.inputs.spectral_resources.get_dict()
            )
            if self.inputs.max_segments.value < 1:
                raise ContractError("max_segments must be >= 1")
            if self.inputs.max_node_hours.value <= 0:
                raise ContractError("max_node_hours must be > 0")
            if self.inputs.max_stalled_segments.value < 1:
                raise ContractError("max_stalled_segments must be >= 1")
            if self.inputs.max_failed_segments.value < 0:
                raise ContractError("max_failed_segments must be >= 0")
            if (
                self.inputs.ground_state_code.computer.uuid
                != self.inputs.spectral_code.computer.uuid
            ):
                raise ContractError("ground-state and spectral codes must use one computer")
        except (AttributeError, ContractError, TypeError, ValueError) as exception:
            self.report(f"invalid full-workflow input: {exception}")
            return self.exit_codes.ERROR_INVALID_INPUT

    def submit_ground_state(self):
        layout = self.ctx.ground_state_layout
        inputs = {
            "code": self.inputs.ground_state_code,
            "parameters": self.inputs.ground_state_parameters,
            "metadata": {
                "label": "ITensor ground state",
                "options": {
                    "resources": {
                        "num_machines": 1,
                        "num_mpiprocs_per_machine": 1,
                        "num_cores_per_mpiproc": layout["threads_per_process"],
                    },
                    "max_wallclock_seconds": layout["wallclock_seconds"],
                    "queue_name": layout["partition"],
                    "withmpi": False,
                },
            },
        }
        future = self.submit(GroundStateCalculation, **inputs)
        self.report(f"submitted ground-state CalcJob<{future.pk}>")
        return ToContext(ground_state_calc=future)

    def inspect_ground_state(self):
        child = self.ctx.ground_state_calc
        required = {"ground_state", "ground_state_metadata", "ground_state_summary"}
        if not child.is_finished_ok or not required.issubset(child.outputs):
            self.report(
                f"ground-state CalcJob<{child.pk}> failed with exit status {child.exit_status}"
            )
            return self.exit_codes.ERROR_GROUND_STATE_FAILED
        self.out("ground_state", child.outputs.ground_state)
        self.out("ground_state_metadata", child.outputs.ground_state_metadata)
        self.out("ground_state_summary", child.outputs.ground_state_summary)

    def prepare_spectra(self):
        plan_node = derive_resource_plan(
            self.ctx.ground_state_calc.outputs.ground_state_metadata,
            self.inputs.spectral_parameters,
            self.inputs.spectral_resources,
        )
        self.out("resource_plan", plan_node)
        plan = plan_node.get_dict()
        if plan.get("state") != "selected":
            self.report(
                "cannot prepare spectral calculation after GS: "
                + plan.get("reason", "unknown resource-planning error")
            )
            return self.exit_codes.ERROR_GROUND_STATE_METADATA

        self.ctx.initial_resource_plan = plan_node
        layout = plan["selected_layout"]
        self.report(
            f"GS reports {plan['ground_states']} state(s) for L={plan['length']}, "
            f"hence {plan['total_tasks']} spectral tasks. The first segment will use "
            f"{layout['nodes']} node(s) x {layout['processes_per_node']} "
            f"processes/node x {layout['threads_per_process']} threads/process = "
            f"{plan['workers']} workers; each worker receives "
            f"{plan['minimum_tasks_per_process']} to "
            f"{plan['maximum_tasks_per_process']} task(s). Reason: {plan['reason']}. "
            f"Resource-plan CalcFunction<{plan_node.creator.pk}> created Dict<{plan_node.pk}>."
        )

    def submit_spectra(self):
        inputs = {
            "code": self.inputs.spectral_code,
            "ground_state": self.ctx.ground_state_calc.outputs.ground_state,
            "ground_state_metadata": self.ctx.ground_state_calc.outputs.ground_state_metadata,
            "spectral_parameters": self.inputs.spectral_parameters,
            "resource_limits": self.inputs.spectral_resources,
            "initial_resource_plan": self.ctx.initial_resource_plan,
            "max_segments": self.inputs.max_segments,
            "max_node_hours": self.inputs.max_node_hours,
            "max_stalled_segments": self.inputs.max_stalled_segments,
            "max_failed_segments": self.inputs.max_failed_segments,
            "metadata": {"label": "ITensor adaptive spectral workflow"},
        }
        future = self.submit(SpinSpectraWorkChain, **inputs)
        self.report(f"submitted spectral WorkChain<{future.pk}>")
        return ToContext(spectra=future)

    def inspect_spectra(self):
        child = self.ctx.spectra
        for label in (
            "final_status",
            "final_remote_folder",
            "final_resource_plan",
            "final_segment_summary",
            "spectrum",
            "collection_report",
            "task_timings",
        ):
            if label in child.outputs:
                self.out(label, child.outputs[label])
        if not child.is_finished_ok:
            self.report(
                f"spectral WorkChain<{child.pk}> stopped with exit status "
                f"{child.exit_status}; available restart state was preserved"
            )
            return self.exit_codes.ERROR_SPECTRA_FAILED
