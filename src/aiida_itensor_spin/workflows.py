"""Restart and repacking orchestration for spin-chain spectral tasks."""

from __future__ import annotations

import math

from aiida import orm
from aiida.engine import (
    BaseRestartWorkChain,
    ProcessHandlerReport,
    ToContext,
    WorkChain,
    calcfunction,
    process_handler,
    while_,
)

from .calculations import (
    GroundStateCalculation,
    SpectralInventoryCalculation,
    SpectralSegmentCalculation,
)
from .contracts import (
    ContractError,
    add_discovered_task_count,
    build_sector_ensemble_plan,
    normalize_ground_state_parameters,
    normalize_layout,
    normalize_parameters,
    normalize_resource_limits,
    select_largest_feasible_layout,
    validate_ground_state_energy_window,
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
        validate_ground_state_energy_window(metadata)
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
                "maximum useful worker count within the supplied ceilings, packed at "
                "the allowed processes-per-node capacity with a partially occupied "
                "final node when needed; threads per process are capped at 2 based "
                "on the validated CPU scaling benchmark"
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


def _oom_processes_per_node_backoff(previous: int) -> int:
    """Return a deterministic, monotonic memory backoff for worker packing."""
    if isinstance(previous, bool) or not isinstance(previous, int) or previous < 1:
        raise ValueError("previous processes_per_node must be a positive integer")
    return max(1, previous - max(1, previous // 3))


@calcfunction
def adapt_resource_limits_after_oom(
    resource_limits: orm.Dict,
    previous_plan: orm.Dict,
    diagnostics: orm.Dict,
) -> orm.Dict:
    """Reduce worker packing after a provenance-recorded out-of-memory failure."""
    values = diagnostics.get_dict()
    if values.get("failure_kind") != "out_of_memory":
        raise ValueError("OOM resource adaptation requires an out_of_memory diagnostic")
    limits = normalize_resource_limits(resource_limits.get_dict())
    plan = previous_plan.get_dict()
    layout = normalize_layout(plan["selected_layout"])
    previous = layout["processes_per_node"]
    reduced = _oom_processes_per_node_backoff(previous)
    limits["max_processes_per_node"] = min(
        limits["max_processes_per_node"], reduced
    )
    return orm.Dict(dict=limits)


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


class AdaptiveSpinSpectraWorkChain(BaseRestartWorkChain):
    """Restart spectral segments with provenance-tracked OOM backoff and approval."""

    _process_class = SpectralSegmentCalculation

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
        spec.input("max_node_hours", valid_type=orm.Float, default=lambda: orm.Float(100.0))
        spec.input("max_stalled_segments", valid_type=orm.Int, default=lambda: orm.Int(1))
        spec.input("max_failed_segments", valid_type=orm.Int, default=lambda: orm.Int(1))
        spec.inputs["on_unhandled_failure"].default = lambda: orm.Str("pause")
        spec.inputs["pause_on_max_iterations"].default = lambda: orm.Bool(True)
        spec.outline(
            cls.setup,
            while_(cls.should_run_process)(cls.run_process, cls.inspect_process),
            cls.results,
        )
        spec.output("final_status", valid_type=orm.Dict, required=False)
        spec.output("final_remote_folder", valid_type=orm.RemoteData, required=False)
        spec.output("final_segment_summary", valid_type=orm.Dict, required=False)
        spec.output("final_resource_plan", valid_type=orm.Dict, required=False)
        spec.output("final_resource_limits", valid_type=orm.Dict, required=False)
        spec.output("last_segment_diagnostics", valid_type=orm.Dict, required=False)
        spec.output("spectrum", valid_type=orm.SinglefileData, required=False)
        spec.output("collection_report", valid_type=orm.SinglefileData, required=False)
        spec.output("task_timings", valid_type=orm.SinglefileData, required=False)
        spec.exit_code(430, "ERROR_INVALID_INPUT", message="Invalid adaptive spectral input.")
        spec.exit_code(431, "ERROR_INVALID_PROGRESS", message="Invalid segment progress report.")
        spec.exit_code(432, "ERROR_PLAN_REJECTED", message="Adaptive resource plan was rejected.")
        spec.exit_code(433, "ERROR_NODE_HOURS", message="Node-hour safety limit reached.")
        spec.exit_code(434, "ERROR_MISSING_RESULT", message="Complete segment lacks results.")
        spec.exit_code(
            435,
            "ERROR_MISSING_DIAGNOSTICS",
            message="Segment failure diagnostics are missing.",
        )

    def setup(self):
        super().setup()
        try:
            initial_plan = self.inputs.initial_resource_plan.get_dict()
            if initial_plan.get("state") != "selected":
                raise ContractError(
                    f"initial resource plan was rejected: {initial_plan.get('reason')}"
                )
            normalize_parameters(initial_plan["parameters"])
            normalize_layout(initial_plan["selected_layout"])
            normalize_resource_limits(self.inputs.resource_limits.get_dict())
            if self.inputs.max_node_hours.value <= 0:
                raise ContractError("max_node_hours must be > 0")
            if self.inputs.max_stalled_segments.value < 1:
                raise ContractError("max_stalled_segments must be >= 1")
            if self.inputs.max_failed_segments.value < 0:
                raise ContractError("max_failed_segments must be >= 0")
        except (KeyError, ContractError, TypeError, ValueError) as exception:
            self.report(f"invalid adaptive spectral input: {exception}")
            return self.exit_codes.ERROR_INVALID_INPUT

        self.ctx.current_plan_node = self.inputs.initial_resource_plan
        self.ctx.current_plan = initial_plan
        self.ctx.current_limits_node = self.inputs.resource_limits
        self.ctx.previous_status = None
        self.ctx.previous_status_node = None
        self.ctx.previous_summary_node = None
        self.ctx.previous_folder = (
            self.inputs.restart_folder if "restart_folder" in self.inputs else None
        )
        self.ctx.last_diagnostics_node = None
        self.ctx.result_nodes = {}
        self.ctx.complete = False
        self.ctx.limit = None
        self.ctx.node_hours = 0.0
        self.ctx.stalled_segments = 0
        self._prepare_process_inputs()

    def _prepare_process_inputs(self):
        plan = self.ctx.current_plan
        layout = plan["selected_layout"]
        inputs = {
            "code": self.inputs.code,
            "ground_state": self.inputs.ground_state,
            "resource_plan": self.ctx.current_plan_node,
            "segment_index": orm.Int(self.ctx.iteration),
            "metadata": {
                "label": f"ITensor adaptive spectral segment {self.ctx.iteration + 1}",
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
        self.ctx.inputs = inputs

    def should_run_process(self):
        if self.ctx.complete:
            return False
        layout = self.ctx.current_plan["selected_layout"]
        projected = self.ctx.node_hours + layout["nodes"] * layout["wallclock_seconds"] / 3600.0
        if projected > self.inputs.max_node_hours.value:
            self.ctx.limit = "node_hours"
            self.ctx.is_finished = True
            self.report(
                f"next segment would raise the allocation ceiling to {projected:.1f} "
                f"node-hours, above max_node_hours={self.inputs.max_node_hours.value:.1f}"
            )
            return False
        return super().should_run_process()

    def run_process(self):
        plan = self.ctx.current_plan
        layout = plan["selected_layout"]
        self.report(
            f"adaptive segment {self.ctx.iteration + 1}: {plan['pending_tasks']} unfinished "
            f"task(s); proposing {layout['nodes']} node(s) x "
            f"{layout['processes_per_node']} processes/node x "
            f"{layout['threads_per_process']} threads/process = {plan['workers']} workers"
        )
        self.ctx.node_hours += layout["nodes"] * layout["wallclock_seconds"] / 3600.0
        return super().run_process()

    def _apply_oom_backoff(self, diagnostics_node):
        previous_ppn = self.ctx.current_plan["selected_layout"]["processes_per_node"]
        self.ctx.current_limits_node = adapt_resource_limits_after_oom(
            self.ctx.current_limits_node,
            self.ctx.current_plan_node,
            diagnostics_node,
        )
        new_ppn = self.ctx.current_limits_node.get_dict()["max_processes_per_node"]
        self.report(
            f"OOM detected. Proposed memory-safe backoff: {previous_ppn} -> {new_ppn} "
            "processes/node. No new CalcJob will be submitted until user approval."
        )

    def _prepare_next_plan(self):
        inputs = [
            self.inputs.ground_state_metadata,
            self.inputs.spectral_parameters,
            self.ctx.current_limits_node,
        ]
        if self.ctx.previous_status_node is not None:
            inputs.append(self.ctx.previous_status_node)
        next_plan = derive_resource_plan(*inputs)
        if next_plan.get_dict().get("state") != "selected":
            self.report(f"adaptive resource plan rejected: {next_plan.get_dict().get('reason')}")
            return False
        self.ctx.current_plan_node = next_plan
        self.ctx.current_plan = next_plan.get_dict()
        self._prepare_process_inputs()
        return True

    def _pause_for_decision(self, reason):
        self.report(
            f"{reason}. Inspect with `verdi process report {self.node.pk}`; "
            f"continue with `verdi process play {self.node.pk}` or abort with "
            f"`verdi process kill {self.node.pk}`."
        )
        self.pause(f"Paused for user inspection: {reason}")

    @process_handler(priority=100)
    def handle_segment(self, node):
        diagnostics_node = (
            node.outputs.segment_diagnostics
            if "segment_diagnostics" in node.outputs
            else None
        )
        diagnostics = (
            diagnostics_node.get_dict()
            if diagnostics_node is not None
            else {"failure_kind": "missing"}
        )
        if diagnostics_node is not None:
            self.ctx.last_diagnostics_node = diagnostics_node

        if "segment_status" not in node.outputs:
            if diagnostics.get("failure_kind") != "out_of_memory":
                return None
            self._apply_oom_backoff(diagnostics_node)
            if not self._prepare_next_plan():
                return ProcessHandlerReport(
                    do_break=True, exit_code=self.exit_codes.ERROR_PLAN_REJECTED
                )
            self.report(
                "OOM segment supplied no valid checkpoint status; its new remote state "
                "will not be trusted. Retaining the last validated restart folder."
            )
            self._pause_for_decision("OOM backoff requires user approval")
            return ProcessHandlerReport(do_break=True)

        try:
            status = validate_segment_status(
                node.outputs.segment_status.get_dict(),
                self.ctx.current_plan["parameters"],
                self.ctx.previous_status,
            )
        except ContractError as exception:
            self.report(f"invalid adaptive segment progress: {exception}")
            return ProcessHandlerReport(
                do_break=True, exit_code=self.exit_codes.ERROR_INVALID_PROGRESS
            )

        if diagnostics_node is None:
            self.report("segment did not expose mandatory failure diagnostics")
            return ProcessHandlerReport(
                do_break=True, exit_code=self.exit_codes.ERROR_MISSING_DIAGNOSTICS
            )
        self.ctx.stalled_segments = (
            0 if status["made_progress"] else self.ctx.stalled_segments + 1
        )
        self.ctx.previous_status = status
        self.ctx.previous_status_node = node.outputs.segment_status
        self.ctx.previous_summary_node = node.outputs.segment_summary
        self.ctx.previous_folder = node.outputs.remote_folder
        self.ctx.complete = status["state"] == "complete"
        for label in ("spectrum", "collection_report", "task_timings"):
            if label in node.outputs:
                self.ctx.result_nodes[label] = node.outputs[label]
        self.report(
            f"adaptive segment {self.ctx.iteration}: {status['completed_tasks']}/"
            f"{status['total_tasks']} tasks; recurrence "
            f"{status['progress_units_completed']}/{status['progress_units_total']}; "
            f"diagnostic={diagnostics['failure_kind']}"
        )
        if self.ctx.complete:
            return None

        if diagnostics.get("failure_kind") == "out_of_memory":
            self._apply_oom_backoff(diagnostics_node)

        if not self._prepare_next_plan():
            return ProcessHandlerReport(
                do_break=True, exit_code=self.exit_codes.ERROR_PLAN_REJECTED
            )

        pause_reason = None
        if diagnostics.get("failure_kind") == "out_of_memory":
            pause_reason = "OOM backoff requires user approval"
        elif diagnostics.get("failure_kind") != "none":
            pause_reason = (
                f"{diagnostics.get('failure_kind')} requires user inspection before retry"
            )
        elif self.ctx.stalled_segments >= self.inputs.max_stalled_segments.value:
            pause_reason = "segment made no measurable progress"
        if pause_reason:
            self._pause_for_decision(pause_reason)
        return ProcessHandlerReport(do_break=True)

    def _emit_outputs(self):
        self.out("final_resource_plan", self.ctx.current_plan_node)
        self.out("final_resource_limits", self.ctx.current_limits_node)
        if self.ctx.previous_status_node is not None:
            self.out("final_status", self.ctx.previous_status_node)
        if self.ctx.previous_folder is not None:
            self.out("final_remote_folder", self.ctx.previous_folder)
        if self.ctx.previous_summary_node is not None:
            self.out("final_segment_summary", self.ctx.previous_summary_node)
        if self.ctx.last_diagnostics_node is not None:
            self.out("last_segment_diagnostics", self.ctx.last_diagnostics_node)
        for label, output in self.ctx.result_nodes.items():
            self.out(label, output)

    def results(self):
        self._emit_outputs()
        if self.ctx.limit == "node_hours":
            return self.exit_codes.ERROR_NODE_HOURS
        if self.ctx.complete:
            missing = {"spectrum", "collection_report", "task_timings"} - set(
                self.ctx.result_nodes
            )
            if missing:
                self.report(
                    "completed adaptive calculation lacks parsed results: "
                    + ", ".join(sorted(missing))
                )
                return self.exit_codes.ERROR_MISSING_RESULT
        return None


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
        spec.output("final_resource_limits", valid_type=orm.Dict, required=False)
        spec.output("last_segment_diagnostics", valid_type=orm.Dict, required=False)
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
            "max_iterations": self.inputs.max_segments,
            "max_node_hours": self.inputs.max_node_hours,
            "max_stalled_segments": self.inputs.max_stalled_segments,
            "max_failed_segments": self.inputs.max_failed_segments,
            "on_unhandled_failure": orm.Str("pause"),
            "pause_on_max_iterations": orm.Bool(True),
            "metadata": {"label": "ITensor adaptive spectral workflow"},
        }
        future = self.submit(AdaptiveSpinSpectraWorkChain, **inputs)
        self.report(f"submitted spectral WorkChain<{future.pk}>")
        return ToContext(spectra=future)

    def inspect_spectra(self):
        child = self.ctx.spectra
        for label in (
            "final_status",
            "final_remote_folder",
            "final_resource_plan",
            "final_resource_limits",
            "last_segment_diagnostics",
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


@calcfunction
def derive_sector_ensemble_plan(
    ground_state_parameters: orm.Dict,
    requested_sectors: orm.List,
    spectral_resources: orm.Dict,
    ensemble_resources: orm.Dict,
) -> orm.Dict:
    """Create a provenance-recorded sector and global-concurrency plan."""
    try:
        plan = build_sector_ensemble_plan(
            ground_state_parameters.get_dict(),
            requested_sectors.get_list(),
            spectral_resources.get_dict(),
            ensemble_resources.get_dict(),
        )
    except (ContractError, TypeError, ValueError) as exception:
        plan = {"state": "rejected", "reason": str(exception)}
    return orm.Dict(dict=plan)


@calcfunction
def prepare_sector_parameters(
    ground_state_parameters: orm.Dict,
    sector_plan: orm.Dict,
    sector_index: orm.Int,
) -> orm.Dict:
    """Materialize one explicit sector input from the ensemble plan."""
    plan = sector_plan.get_dict()
    if plan.get("state") != "selected":
        raise ValueError("cannot prepare parameters from a rejected sector plan")
    index = sector_index.value
    sectors = plan["requested_sectors"]
    if index < 0 or index >= len(sectors):
        raise ValueError("sector index is outside the ensemble plan")
    parameters = ground_state_parameters.get_dict()
    if "target_twice_sz" in parameters:
        raise ValueError("ensemble base parameters already contain target_twice_sz")
    parameters["target_twice_sz"] = sectors[index]
    return orm.Dict(dict=normalize_ground_state_parameters(parameters))


@calcfunction
def prepare_sector_spectral_resources(sector_plan: orm.Dict) -> orm.Dict:
    """Expose the per-sector ceiling selected under the global node limit."""
    plan = sector_plan.get_dict()
    if plan.get("state") != "selected":
        raise ValueError("cannot prepare resources from a rejected sector plan")
    return orm.Dict(dict=plan["effective_spectral_resources"])


@calcfunction
def select_spectral_sectors(
    sector_plan: orm.Dict,
    ground_state_parameters: orm.Dict,
    **sector_scan_metadata: orm.Dict,
) -> orm.Dict:
    """Select the global/near-degenerate ground-state sectors for spectra."""
    plan = sector_plan.get_dict()
    parameters = normalize_ground_state_parameters(ground_state_parameters.get_dict())
    scans = []
    missing = []
    for sector, label in zip(
        plan.get("requested_sectors", []), plan.get("sector_labels", []), strict=True
    ):
        node = sector_scan_metadata.get(label)
        if node is None:
            missing.append(sector)
            continue
        metadata = node.get_dict()
        energy = metadata.get("ground_energy_meV")
        if (
            metadata.get("scan_only") is not True
            or metadata.get("target_twice_sz") != sector
            or isinstance(energy, bool)
            or not isinstance(energy, (int, float))
            or not math.isfinite(energy)
        ):
            raise ValueError(f"invalid sector-scan metadata for {label}")
        scans.append(
            {
                "target_twice_sz": sector,
                "label": label,
                "ground_energy_meV": float(energy),
            }
        )

    if missing:
        return orm.Dict(
            dict={
                "state": "rejected",
                "reason": f"sector scans are missing for 2Sz={missing}",
                "requested_sectors": plan.get("requested_sectors", []),
                "selected_sectors": [],
                "selected_labels": [],
                "scans": scans,
            }
        )

    global_energy = min(item["ground_energy_meV"] for item in scans)
    thermal_window = 8.617333262e-2 * parameters["temperature_K"] / 5.0
    numerical_tolerance = max(
        parameters["energy_precision_meV"],
        1.0e-12 * max(1.0, abs(global_energy)),
    )
    selection_window = max(thermal_window, numerical_tolerance)
    selected = [
        item
        for item in scans
        if item["ground_energy_meV"] - global_energy <= selection_window
    ]
    return orm.Dict(
        dict={
            "state": "selected",
            "reason": (
                "spectra are restricted to sectors whose lowest DMRG energy lies "
                "within max(k_B*T/5, energy_precision) of the global minimum; "
                "higher-energy sectors remain energy-scan results only"
            ),
            "requested_sectors": plan["requested_sectors"],
            "global_ground_energy_meV": global_energy,
            "temperature_K": parameters["temperature_K"],
            "thermal_window_meV": thermal_window,
            "numerical_tolerance_meV": numerical_tolerance,
            "selection_window_meV": selection_window,
            "selected_sectors": [item["target_twice_sz"] for item in selected],
            "selected_labels": [item["label"] for item in selected],
            "scans": scans,
        }
    )


@calcfunction
def collect_sector_summary(
    sector_plan: orm.Dict,
    spectral_sector_plan: orm.Dict,
    **sector_metadata: orm.Dict,
) -> orm.Dict:
    """Collect all scan energies and the selected spectral outcomes."""
    plan = sector_plan.get_dict()
    selection = spectral_sector_plan.get_dict()
    selected_labels = set(selection.get("selected_labels", []))
    results = []
    for sector, label in zip(
        plan.get("requested_sectors", []), plan.get("sector_labels", []), strict=True
    ):
        scan_node = sector_metadata.get(f"scan_{label}")
        spectrum_node = sector_metadata.get(f"spectrum_{label}")
        result = {
            "target_twice_sz": sector,
            "label": label,
            "spectrum_selected": label in selected_labels,
        }
        if scan_node is None:
            result.update({"scan_state": "missing", "spectrum_state": "not_run"})
        else:
            scan = scan_node.get_dict()
            if scan.get("target_twice_sz") != sector:
                raise ValueError(f"scan metadata for {label} reports a different sector")
            result.update(
                {
                    "scan_state": "complete",
                    "ground_energy_meV": scan["ground_energy_meV"],
                }
            )
            if label not in selected_labels:
                result["spectrum_state"] = "not_selected"
            elif spectrum_node is None:
                result["spectrum_state"] = "missing"
            else:
                spectrum = spectrum_node.get_dict()
                if spectrum.get("target_twice_sz") != sector:
                    raise ValueError(
                        f"spectral metadata for {label} reports a different sector"
                    )
                result.update(
                    {
                        "spectrum_state": "complete",
                        "highest_energy_meV": spectrum["highest_energy_meV"],
                        "spectral_bandwidth_meV": spectrum["spectral_bandwidth_meV"],
                        "ground_states": spectrum["ground_states"],
                        "total_tasks": spectrum["total_tasks"],
                    }
                )
        results.append(result)

    complete = (
        selection.get("state") == "selected"
        and all(result["scan_state"] == "complete" for result in results)
        and all(
            result["spectrum_state"] == "complete"
            for result in results
            if result["spectrum_selected"]
        )
    )
    return orm.Dict(
        dict={
            "state": "complete" if complete else "partial",
            "model": plan.get("model"),
            "requested_sectors": plan.get("requested_sectors", []),
            "selected_spectral_sectors": selection.get("selected_sectors", []),
            "global_ground_energy_meV": selection.get("global_ground_energy_meV"),
            "selection_window_meV": selection.get("selection_window_meV"),
            "results": results,
        }
    )


class SpinSectorEnsembleWorkChain(WorkChain):
    """Scan fixed-Sz energies, then run spectra only for ground-state sectors."""

    @classmethod
    def define(cls, spec):
        super().define(spec)
        spec.input("ground_state_code", valid_type=orm.AbstractCode)
        spec.input("spectral_code", valid_type=orm.AbstractCode)
        spec.input("ground_state_parameters", valid_type=orm.Dict)
        spec.input("ground_state_resources", valid_type=orm.Dict)
        spec.input("spectral_parameters", valid_type=orm.Dict)
        spec.input("spectral_resources", valid_type=orm.Dict)
        spec.input(
            "requested_sectors",
            valid_type=orm.List,
            help="Explicit integer values of 2Sz to scan.",
        )
        spec.input(
            "ensemble_resources",
            valid_type=orm.Dict,
            help="Global max_total_nodes and max_concurrent_sectors ceilings.",
        )
        spec.input("max_segments", valid_type=orm.Int, default=lambda: orm.Int(6))
        spec.input("max_node_hours", valid_type=orm.Float, default=lambda: orm.Float(100.0))
        spec.input("max_stalled_segments", valid_type=orm.Int, default=lambda: orm.Int(1))
        spec.input("max_failed_segments", valid_type=orm.Int, default=lambda: orm.Int(1))
        spec.outline(
            cls.setup,
            while_(cls.has_pending_scans)(cls.submit_scan_batch, cls.inspect_scan_batch),
            cls.select_spectral_sectors,
            while_(cls.has_pending_spectra)(
                cls.submit_spectral_batch, cls.inspect_spectral_batch
            ),
            cls.finalize,
        )
        spec.output("sector_plan", valid_type=orm.Dict)
        spec.output("spectral_sector_plan", valid_type=orm.Dict, required=False)
        spec.output("sector_summary", valid_type=orm.Dict, required=False)
        spec.output_namespace(
            "sector_scan_summaries",
            dynamic=True,
            valid_type=orm.SinglefileData,
            required=False,
        )
        spec.output_namespace(
            "sector_scan_metadata", dynamic=True, valid_type=orm.Dict, required=False
        )
        spec.output_namespace(
            "sector_ground_states", dynamic=True, valid_type=orm.SinglefileData, required=False
        )
        spec.output_namespace(
            "sector_ground_state_summaries",
            dynamic=True,
            valid_type=orm.SinglefileData,
            required=False,
        )
        spec.output_namespace("sector_metadata", dynamic=True, valid_type=orm.Dict, required=False)
        spec.output_namespace(
            "sector_spectra", dynamic=True, valid_type=orm.SinglefileData, required=False
        )
        spec.output_namespace(
            "sector_collection_reports",
            dynamic=True,
            valid_type=orm.SinglefileData,
            required=False,
        )
        spec.output_namespace(
            "sector_task_timings", dynamic=True, valid_type=orm.SinglefileData, required=False
        )
        spec.exit_code(600, "ERROR_INVALID_INPUT", message="Invalid sector ensemble input.")
        spec.exit_code(601, "ERROR_SECTOR_SCAN_FAILED", message="One or more sector scans failed.")
        spec.exit_code(
            602,
            "ERROR_SECTOR_SPECTRUM_FAILED",
            message="One or more selected sector spectra failed.",
        )

    def setup(self):
        plan_node = derive_sector_ensemble_plan(
            self.inputs.ground_state_parameters,
            self.inputs.requested_sectors,
            self.inputs.spectral_resources,
            self.inputs.ensemble_resources,
        )
        self.out("sector_plan", plan_node)
        plan = plan_node.get_dict()
        if plan.get("state") != "selected":
            self.report(f"invalid sector ensemble input: {plan.get('reason', 'unknown error')}")
            return self.exit_codes.ERROR_INVALID_INPUT

        try:
            scan_layout = normalize_layout(self.inputs.ground_state_resources.get_dict())
            if scan_layout["nodes"] != 1 or scan_layout["processes_per_node"] != 1:
                raise ContractError("sector scans require one node and one process each")
            if (
                self.inputs.ground_state_code.computer.uuid
                != self.inputs.spectral_code.computer.uuid
            ):
                raise ContractError("ground-state and spectral codes must use one computer")
        except (AttributeError, ContractError, TypeError, ValueError) as exception:
            self.report(f"invalid sector ensemble input: {exception}")
            return self.exit_codes.ERROR_INVALID_INPUT

        self.ctx.sector_plan_node = plan_node
        self.ctx.sector_plan = plan
        self.ctx.scan_layout = scan_layout
        self.ctx.sector_spectral_resources = prepare_sector_spectral_resources(plan_node)
        self.ctx.pending_scan_indices = list(range(len(plan["requested_sectors"])))
        self.ctx.current_scan_batch = []
        self.ctx.failed_scans = []
        self.ctx.scan_metadata_nodes = {}
        self.ctx.pending_spectral_indices = []
        self.ctx.current_spectral_batch = []
        self.ctx.failed_spectra = []
        self.ctx.spectral_metadata_nodes = {}
        self.report(
            f"selected 2Sz sectors {plan['requested_sectors']}. Stage 1 will run at most "
            f"{plan['scan_concurrent_sectors']} one-node DMRG energy scan(s) concurrently "
            f"under the {plan['max_total_nodes']}-node global ceiling."
        )

    def has_pending_scans(self):
        return bool(self.ctx.pending_scan_indices)

    def submit_scan_batch(self):
        width = self.ctx.sector_plan["scan_concurrent_sectors"]
        indices = self.ctx.pending_scan_indices[:width]
        self.ctx.pending_scan_indices = self.ctx.pending_scan_indices[width:]
        self.ctx.current_scan_batch = []
        futures = {}
        layout = self.ctx.scan_layout
        for index in indices:
            sector = self.ctx.sector_plan["requested_sectors"][index]
            label = self.ctx.sector_plan["sector_labels"][index]
            parameters = prepare_sector_parameters(
                self.inputs.ground_state_parameters,
                self.ctx.sector_plan_node,
                orm.Int(index),
            )
            future = self.submit(
                GroundStateCalculation,
                code=self.inputs.ground_state_code,
                parameters=parameters,
                scan_only=orm.Bool(True),
                metadata={
                    "label": f"ITensor sector scan 2Sz={sector}",
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
            )
            context_key = f"scan_child_{index}"
            futures[context_key] = future
            self.ctx.current_scan_batch.append((index, sector, label, context_key))
            self.report(f"submitted energy scan for 2Sz={sector} CalcJob<{future.pk}>")
        return ToContext(**futures)

    def inspect_scan_batch(self):
        for _index, sector, label, context_key in self.ctx.current_scan_batch:
            child = self.ctx[context_key]
            if "ground_state_summary" in child.outputs:
                self.out(
                    f"sector_scan_summaries.{label}",
                    child.outputs.ground_state_summary,
                )
            if "ground_state_metadata" in child.outputs:
                self.out(
                    f"sector_scan_metadata.{label}",
                    child.outputs.ground_state_metadata,
                )
            if child.is_finished_ok and "ground_state_metadata" in child.outputs:
                self.ctx.scan_metadata_nodes[label] = child.outputs.ground_state_metadata
                self.report(f"energy scan for 2Sz={sector} completed")
            else:
                self.ctx.failed_scans.append(sector)
                self.report(
                    f"energy scan for 2Sz={sector} CalcJob<{child.pk}> failed with "
                    f"exit status {child.exit_status}"
                )

    def select_spectral_sectors(self):
        selection = select_spectral_sectors(
            self.ctx.sector_plan_node,
            self.inputs.ground_state_parameters,
            **self.ctx.scan_metadata_nodes,
        )
        self.out("spectral_sector_plan", selection)
        self.ctx.spectral_sector_plan = selection
        selected = selection.get_dict()
        if selected.get("state") != "selected":
            summary_inputs = {
                f"scan_{label}": node
                for label, node in self.ctx.scan_metadata_nodes.items()
            }
            self.out(
                "sector_summary",
                collect_sector_summary(
                    self.ctx.sector_plan_node,
                    selection,
                    **summary_inputs,
                ),
            )
            self.report(selected.get("reason", "sector selection failed"))
            return self.exit_codes.ERROR_SECTOR_SCAN_FAILED

        label_to_index = {
            label: index
            for index, label in enumerate(self.ctx.sector_plan["sector_labels"])
        }
        self.ctx.pending_spectral_indices = [
            label_to_index[label] for label in selected["selected_labels"]
        ]
        self.report(
            f"sector scans found global E0={selected['global_ground_energy_meV']:.12g} meV. "
            f"Using a {selected['selection_window_meV']:.6g} meV ground/thermal "
            f"window, full spectra will run only for 2Sz={selected['selected_sectors']}. "
            f"Reason: {selected['reason']}."
        )

    def has_pending_spectra(self):
        return bool(self.ctx.pending_spectral_indices)

    def submit_spectral_batch(self):
        width = self.ctx.sector_plan["concurrent_sectors"]
        indices = self.ctx.pending_spectral_indices[:width]
        self.ctx.pending_spectral_indices = self.ctx.pending_spectral_indices[width:]
        self.ctx.current_spectral_batch = []
        futures = {}
        for index in indices:
            sector = self.ctx.sector_plan["requested_sectors"][index]
            label = self.ctx.sector_plan["sector_labels"][index]
            parameters = prepare_sector_parameters(
                self.inputs.ground_state_parameters,
                self.ctx.sector_plan_node,
                orm.Int(index),
            )
            future = self.submit(
                SpinChainWorkChain,
                ground_state_code=self.inputs.ground_state_code,
                spectral_code=self.inputs.spectral_code,
                ground_state_parameters=parameters,
                ground_state_resources=self.inputs.ground_state_resources,
                spectral_parameters=self.inputs.spectral_parameters,
                spectral_resources=self.ctx.sector_spectral_resources,
                max_segments=self.inputs.max_segments,
                max_node_hours=self.inputs.max_node_hours,
                max_stalled_segments=self.inputs.max_stalled_segments,
                max_failed_segments=self.inputs.max_failed_segments,
                metadata={
                    "label": f"ITensor selected sector 2Sz={sector}",
                    "description": (
                        "Full spectrum for a sector selected by the ensemble energy scan."
                    ),
                },
            )
            context_key = f"spectrum_child_{index}"
            futures[context_key] = future
            self.ctx.current_spectral_batch.append((index, sector, label, context_key))
            self.report(f"submitted selected 2Sz={sector} WorkChain<{future.pk}>")
        return ToContext(**futures)

    def inspect_spectral_batch(self):
        for _index, sector, label, context_key in self.ctx.current_spectral_batch:
            child = self.ctx[context_key]
            output_map = (
                ("ground_state", "sector_ground_states"),
                ("ground_state_summary", "sector_ground_state_summaries"),
                ("ground_state_metadata", "sector_metadata"),
                ("spectrum", "sector_spectra"),
                ("collection_report", "sector_collection_reports"),
                ("task_timings", "sector_task_timings"),
            )
            for child_label, namespace in output_map:
                if child_label in child.outputs:
                    self.out(f"{namespace}.{label}", child.outputs[child_label])
            if child.is_finished_ok and "ground_state_metadata" in child.outputs:
                self.ctx.spectral_metadata_nodes[label] = child.outputs.ground_state_metadata
                self.report(f"selected spectrum for 2Sz={sector} completed successfully")
            else:
                self.ctx.failed_spectra.append(sector)
                self.report(
                    f"selected spectrum for 2Sz={sector} WorkChain<{child.pk}> failed "
                    f"with exit status {child.exit_status}"
                )

    def finalize(self):
        summary_inputs = {
            f"scan_{label}": node for label, node in self.ctx.scan_metadata_nodes.items()
        }
        summary_inputs.update(
            {
                f"spectrum_{label}": node
                for label, node in self.ctx.spectral_metadata_nodes.items()
            }
        )
        self.out(
            "sector_summary",
            collect_sector_summary(
                self.ctx.sector_plan_node,
                self.ctx.spectral_sector_plan,
                **summary_inputs,
            ),
        )
        if self.ctx.failed_spectra:
            self.report(f"failed selected spectra for 2Sz={self.ctx.failed_spectra}")
            return self.exit_codes.ERROR_SECTOR_SPECTRUM_FAILED
        return None


class SpinSpectraRestartWorkChain(WorkChain):
    """Inventory an immutable remote state and continue spectra with a fresh plan."""

    @classmethod
    def define(cls, spec):
        super().define(spec)
        spec.input("code", valid_type=orm.AbstractCode)
        spec.input("ground_state", valid_type=orm.SinglefileData)
        spec.input("ground_state_metadata", valid_type=orm.Dict)
        spec.input("spectral_parameters", valid_type=orm.Dict)
        spec.input("resource_limits", valid_type=orm.Dict)
        spec.input("restart_folder", valid_type=orm.RemoteData)
        spec.input("max_segments", valid_type=orm.Int, default=lambda: orm.Int(12))
        spec.input("max_node_hours", valid_type=orm.Float, default=lambda: orm.Float(10000.0))
        spec.input("max_stalled_segments", valid_type=orm.Int, default=lambda: orm.Int(2))
        spec.input("max_failed_segments", valid_type=orm.Int, default=lambda: orm.Int(2))
        spec.outline(
            cls.setup,
            cls.submit_inventory,
            cls.inspect_inventory,
            cls.submit_continuation,
            cls.inspect_continuation,
        )
        spec.output("inventory_status", valid_type=orm.Dict, required=False)
        spec.output("inventory_summary", valid_type=orm.Dict, required=False)
        spec.output("resource_plan", valid_type=orm.Dict, required=False)
        for label, node_type in (
            ("final_status", orm.Dict),
            ("final_remote_folder", orm.RemoteData),
            ("final_segment_summary", orm.Dict),
            ("final_resource_plan", orm.Dict),
            ("final_resource_limits", orm.Dict),
            ("last_segment_diagnostics", orm.Dict),
            ("spectrum", orm.SinglefileData),
            ("collection_report", orm.SinglefileData),
            ("task_timings", orm.SinglefileData),
        ):
            spec.output(label, valid_type=node_type, required=False)
        spec.exit_code(700, "ERROR_INVALID_INPUT", message="Invalid restart input.")
        spec.exit_code(701, "ERROR_INVENTORY_FAILED", message="Restart inventory failed.")
        spec.exit_code(702, "ERROR_ALREADY_COMPLETE", message="Restart state is already complete.")
        spec.exit_code(703, "ERROR_PLAN_REJECTED", message="Restart resource plan was rejected.")
        spec.exit_code(704, "ERROR_CONTINUATION_FAILED", message="Spectral continuation failed.")

    def setup(self):
        try:
            normalize_parameters(self.inputs.spectral_parameters.get_dict())
            normalize_resource_limits(self.inputs.resource_limits.get_dict())
            validate_ground_state_energy_window(self.inputs.ground_state_metadata.get_dict())
            if self.inputs.restart_folder.computer.uuid != self.inputs.code.computer.uuid:
                raise ContractError("restart_folder and code must use the same computer")
            if self.inputs.max_node_hours.value <= 0:
                raise ContractError("max_node_hours must be > 0")
        except (KeyError, ContractError, TypeError, ValueError) as exception:
            self.report(f"invalid restart input: {exception}")
            return self.exit_codes.ERROR_INVALID_INPUT

    def submit_inventory(self):
        limits = normalize_resource_limits(self.inputs.resource_limits.get_dict())
        future = self.submit(
            SpectralInventoryCalculation,
            code=self.inputs.code,
            ground_state=self.inputs.ground_state,
            parent_folder=self.inputs.restart_folder,
            parameters=self.inputs.spectral_parameters,
            segment_index=orm.Int(0),
            metadata={
                "label": "ITensor spectral restart inventory",
                "options": {
                    "resources": {
                        "num_machines": 1,
                        "num_mpiprocs_per_machine": 1,
                        "num_cores_per_mpiproc": 1,
                    },
                    "max_wallclock_seconds": min(1800, limits["wallclock_seconds"]),
                    "queue_name": limits["partition"],
                    "withmpi": False,
                },
            },
        )
        self.report(f"submitted restart inventory CalcJob<{future.pk}>")
        return ToContext(inventory=future)

    def inspect_inventory(self):
        child = self.ctx.inventory
        if (
            not child.is_finished_ok
            or "segment_status" not in child.outputs
            or "inventory_summary" not in child.outputs
        ):
            self.report(
                f"restart inventory CalcJob<{child.pk}> failed with exit status "
                f"{child.exit_status}"
            )
            return self.exit_codes.ERROR_INVENTORY_FAILED
        status = child.outputs.segment_status
        self.out("inventory_status", status)
        self.out("inventory_summary", child.outputs.inventory_summary)
        values = status.get_dict()
        self.report(
            f"inventory found {values['completed_tasks']}/{values['total_tasks']} "
            f"complete tasks and {values['progress_units_completed']}/"
            f"{values['progress_units_total']} recurrence units"
        )
        if values["completed_tasks"] == values["total_tasks"]:
            return self.exit_codes.ERROR_ALREADY_COMPLETE
        plan = derive_resource_plan(
            self.inputs.ground_state_metadata,
            self.inputs.spectral_parameters,
            self.inputs.resource_limits,
            status,
        )
        if plan.get_dict().get("state") != "selected":
            self.report(f"restart resource plan rejected: {plan.get_dict().get('reason')}")
            return self.exit_codes.ERROR_PLAN_REJECTED
        self.ctx.resource_plan = plan
        self.out("resource_plan", plan)

    def submit_continuation(self):
        if "resource_plan" not in self.ctx:
            return None
        plan = self.ctx.resource_plan.get_dict()
        layout = plan["selected_layout"]
        self.report(
            f"continuing {plan['pending_tasks']} unfinished task(s) on "
            f"{layout['nodes']} node(s) x {layout['processes_per_node']} "
            f"processes/node x {layout['threads_per_process']} threads/process"
        )
        future = self.submit(
            AdaptiveSpinSpectraWorkChain,
            code=self.inputs.code,
            ground_state=self.inputs.ground_state,
            ground_state_metadata=self.inputs.ground_state_metadata,
            spectral_parameters=self.inputs.spectral_parameters,
            resource_limits=self.inputs.resource_limits,
            initial_resource_plan=self.ctx.resource_plan,
            restart_folder=self.inputs.restart_folder,
            max_iterations=self.inputs.max_segments,
            max_node_hours=self.inputs.max_node_hours,
            max_stalled_segments=self.inputs.max_stalled_segments,
            max_failed_segments=self.inputs.max_failed_segments,
            on_unhandled_failure=orm.Str("pause"),
            pause_on_max_iterations=orm.Bool(True),
            metadata={
                "label": "ITensor spectral continuation",
                "description": "Automatic continuation from inventoried immutable RemoteData.",
            },
        )
        return ToContext(continuation=future)

    def inspect_continuation(self):
        if "continuation" not in self.ctx:
            return None
        child = self.ctx.continuation
        for label in (
            "final_status",
            "final_remote_folder",
            "final_segment_summary",
            "final_resource_plan",
            "final_resource_limits",
            "last_segment_diagnostics",
            "spectrum",
            "collection_report",
            "task_timings",
        ):
            if label in child.outputs:
                self.out(label, child.outputs[label])
        if not child.is_finished_ok:
            self.report(
                f"spectral continuation WorkChain<{child.pk}> failed with exit status "
                f"{child.exit_status}"
            )
            return self.exit_codes.ERROR_CONTINUATION_FAILED
        return None
