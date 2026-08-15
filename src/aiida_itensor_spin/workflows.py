"""Restart and repacking orchestration for spin-chain spectral tasks."""

from __future__ import annotations

from aiida import orm
from aiida.engine import ToContext, WorkChain, while_

from .calculations import GroundStateCalculation, SpectralSegmentCalculation
from .contracts import ContractError, normalize_layout, normalize_parameters, validate_segment_status


class SpinSpectraWorkChain(WorkChain):
    """Run bounded spectral segments until every manifest task is complete."""

    @classmethod
    def define(cls, spec):
        super().define(spec)
        spec.input("code", valid_type=orm.AbstractCode)
        spec.input("ground_state", valid_type=orm.SinglefileData)
        spec.input(
            "restart_folder",
            valid_type=orm.RemoteData,
            required=False,
            help="Immutable state folder from a previous or externally recovered segment.",
        )
        spec.input("parameters", valid_type=orm.Dict)
        spec.input(
            "layouts",
            valid_type=orm.List,
            help="Ordered segment layouts; reused cyclically if more segments are needed.",
        )
        spec.input("max_segments", valid_type=orm.Int, default=lambda: orm.Int(6))
        spec.input("max_node_hours", valid_type=orm.Float, default=lambda: orm.Float(100.0))
        spec.input(
            "max_stalled_segments", valid_type=orm.Int, default=lambda: orm.Int(1)
        )
        spec.outline(
            cls.setup,
            while_(cls.should_run)(cls.submit_segment, cls.inspect_segment),
            cls.finalize,
        )
        spec.output("final_status", valid_type=orm.Dict, required=False)
        spec.output("final_remote_folder", valid_type=orm.RemoteData, required=False)
        spec.output("workflow_summary", valid_type=orm.Dict)
        spec.exit_code(400, "ERROR_INVALID_INPUT", message="Invalid workflow input contract.")
        spec.exit_code(401, "ERROR_SEGMENT_FAILED", message="A spectral segment failed.")
        spec.exit_code(402, "ERROR_INVALID_PROGRESS", message="Invalid segment progress report.")
        spec.exit_code(403, "ERROR_STALLED", message="No progress in consecutive segments.")
        spec.exit_code(404, "ERROR_MAX_SEGMENTS", message="Maximum segment count reached.")
        spec.exit_code(405, "ERROR_NODE_HOURS", message="Node-hour safety limit reached.")

    def setup(self):
        try:
            self.ctx.parameters = normalize_parameters(self.inputs.parameters.get_dict())
            layouts = [normalize_layout(item) for item in self.inputs.layouts.get_list()]
            if not layouts:
                raise ContractError("at least one layout is required")
            if self.inputs.max_segments.value < 1:
                raise ContractError("max_segments must be >= 1")
            if self.inputs.max_node_hours.value <= 0:
                raise ContractError("max_node_hours must be > 0")
            if self.inputs.max_stalled_segments.value < 1:
                raise ContractError("max_stalled_segments must be >= 1")
        except (ContractError, TypeError, ValueError):
            return self.exit_codes.ERROR_INVALID_INPUT

        self.ctx.layouts = layouts
        self.ctx.segment_count = 0
        self.ctx.node_hours = 0.0
        self.ctx.stalled_segments = 0
        self.ctx.previous_status = None
        self.ctx.previous_status_node = None
        self.ctx.previous_summary_node = None
        self.ctx.previous_folder = (
            self.inputs.restart_folder if "restart_folder" in self.inputs else None
        )
        self.ctx.complete = False
        self.ctx.child_uuids = []
        self.ctx.limit = None

    def should_run(self):
        if self.ctx.complete:
            return False
        if self.ctx.segment_count >= self.inputs.max_segments.value:
            self.ctx.limit = "max_segments"
            return False
        layout = self.ctx.layouts[self.ctx.segment_count % len(self.ctx.layouts)]
        projected = self.ctx.node_hours + (
            layout["nodes"] * layout["wallclock_seconds"] / 3600.0
        )
        if projected > self.inputs.max_node_hours.value:
            self.ctx.limit = "node_hours"
            return False
        return True

    def submit_segment(self):
        index = self.ctx.segment_count
        layout = self.ctx.layouts[index % len(self.ctx.layouts)]
        inputs = {
            "code": self.inputs.code,
            "ground_state": self.inputs.ground_state,
            "parameters": orm.Dict(dict=self.ctx.parameters),
            "layout": orm.Dict(dict=layout),
            "segment_index": orm.Int(index),
            "metadata": {
                "label": f"ITensor spectral segment {index + 1}",
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
        self.report(
            f"submitted segment {index + 1}<{future.pk}>: "
            f"{layout['nodes']} nodes, {layout['processes_per_node']} processes/node, "
            f"{layout['threads_per_process']} threads/process"
        )
        self.ctx.segment_count += 1
        self.ctx.node_hours += layout["nodes"] * layout["wallclock_seconds"] / 3600.0
        self.ctx.child_uuids.append(str(future.uuid))
        return ToContext(segment=future)

    def inspect_segment(self):
        child = self.ctx.segment
        if not child.is_finished_ok or "segment_status" not in child.outputs:
            self.report(f"segment {self.ctx.segment_count} failed with exit status {child.exit_status}")
            return self.exit_codes.ERROR_SEGMENT_FAILED
        try:
            status = validate_segment_status(
                child.outputs.segment_status.get_dict(),
                self.ctx.parameters,
                self.ctx.previous_status,
            )
        except ContractError as exception:
            self.report(f"invalid segment progress: {exception}")
            return self.exit_codes.ERROR_INVALID_PROGRESS

        if status["made_progress"]:
            self.ctx.stalled_segments = 0
        else:
            self.ctx.stalled_segments += 1
        if self.ctx.stalled_segments >= self.inputs.max_stalled_segments.value:
            self.report("segment made no measurable task or recurrence progress")
            return self.exit_codes.ERROR_STALLED

        self.ctx.previous_status = status
        self.ctx.previous_status_node = child.outputs.segment_status
        self.ctx.previous_summary_node = child.outputs.segment_summary
        self.ctx.previous_folder = child.outputs.remote_folder
        self.ctx.complete = status["state"] == "complete"
        self.report(
            f"segment {self.ctx.segment_count}: {status['completed_tasks']}/"
            f"{status['total_tasks']} tasks, recurrence progress "
            f"{status['progress_units_completed']}/{status['progress_units_total']}"
        )

    def finalize(self):
        if self.ctx.previous_summary_node is not None:
            self.out("workflow_summary", self.ctx.previous_summary_node)
        if self.ctx.previous_status_node is not None:
            self.out("final_status", self.ctx.previous_status_node)
        if self.ctx.previous_folder is not None:
            self.out("final_remote_folder", self.ctx.previous_folder)
        if self.ctx.complete:
            return None
        if self.ctx.limit == "node_hours":
            return self.exit_codes.ERROR_NODE_HOURS
        return self.exit_codes.ERROR_MAX_SEGMENTS



class SpinChainWorkChain(WorkChain):
    """Compute one ground state and run its restartable spectral workflow."""

    @classmethod
    def define(cls, spec):
        super().define(spec)
        spec.input("ground_state_code", valid_type=orm.AbstractCode)
        spec.input("spectral_code", valid_type=orm.AbstractCode)
        spec.input("ground_state_parameters", valid_type=orm.Dict)
        spec.input(
            "ground_state_layout",
            valid_type=orm.Dict,
            help="Slurm nodes, processes, threads, wallclock, and partition for the GS CalcJob.",
        )
        spec.expose_inputs(
            SpinSpectraWorkChain,
            namespace="spectra",
            exclude=("code", "ground_state", "restart_folder"),
        )
        spec.outline(
            cls.submit_ground_state,
            cls.inspect_ground_state,
            cls.submit_spectra,
            cls.inspect_spectra,
        )
        spec.output("ground_state", valid_type=orm.SinglefileData)
        spec.output("ground_state_metadata", valid_type=orm.Dict, required=False)
        spec.output("ground_state_summary", valid_type=orm.SinglefileData, required=False)
        spec.output("final_status", valid_type=orm.Dict, required=False)
        spec.output("final_remote_folder", valid_type=orm.RemoteData, required=False)
        spec.output("workflow_summary", valid_type=orm.Dict, required=False)
        spec.exit_code(500, "ERROR_GROUND_STATE_FAILED", message="Ground-state CalcJob failed.")
        spec.exit_code(501, "ERROR_SPECTRA_FAILED", message="Spectral WorkChain failed.")

    def submit_ground_state(self):
        try:
            layout = normalize_layout(self.inputs.ground_state_layout.get_dict())
        except (ContractError, TypeError, ValueError):
            return self.exit_codes.ERROR_GROUND_STATE_FAILED
        inputs = {
            "code": self.inputs.ground_state_code,
            "parameters": self.inputs.ground_state_parameters,
            "metadata": {
                "label": "ITensor ground state",
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
        future = self.submit(GroundStateCalculation, **inputs)
        self.report(f"submitted ground-state CalcJob<{future.pk}>")
        return ToContext(ground_state_calc=future)

    def inspect_ground_state(self):
        child = self.ctx.ground_state_calc
        if not child.is_finished_ok or "ground_state" not in child.outputs:
            self.report(f"ground-state CalcJob failed with exit status {child.exit_status}")
            return self.exit_codes.ERROR_GROUND_STATE_FAILED
        self.out("ground_state", child.outputs.ground_state)
        if "ground_state_metadata" in child.outputs:
            self.out("ground_state_metadata", child.outputs.ground_state_metadata)
        if "ground_state_summary" in child.outputs:
            self.out("ground_state_summary", child.outputs.ground_state_summary)

    def submit_spectra(self):
        inputs = self.exposed_inputs(SpinSpectraWorkChain, namespace="spectra")
        inputs["code"] = self.inputs.spectral_code
        inputs["ground_state"] = self.ctx.ground_state_calc.outputs.ground_state
        future = self.submit(SpinSpectraWorkChain, **inputs)
        self.report(f"submitted spectral WorkChain<{future.pk}>")
        return ToContext(spectra=future)

    def inspect_spectra(self):
        child = self.ctx.spectra
        if not child.is_finished_ok:
            self.report(f"spectral WorkChain failed with exit status {child.exit_status}")
            return self.exit_codes.ERROR_SPECTRA_FAILED
        for label in ("final_status", "final_remote_folder", "workflow_summary"):
            if label in child.outputs:
                self.out(label, child.outputs[label])
