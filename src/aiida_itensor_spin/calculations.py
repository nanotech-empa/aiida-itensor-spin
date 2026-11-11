"""CalcJobs for bounded, restartable ITensor spectral segments."""

from __future__ import annotations

import json

from aiida import orm
from aiida.common import CalcInfo, CodeInfo
from aiida.engine import CalcJob

from .contracts import normalize_ground_state_parameters, normalize_layout, normalize_parameters
from .graphs import GRAPH_MODEL_IDENTIFIER


class GroundStateCalculation(CalcJob):
    """Compute and retrieve one durable ITensor ground-state checkpoint."""

    @classmethod
    def define(cls, spec):
        super().define(spec)
        spec.input("parameters", valid_type=orm.Dict)
        spec.input(
            "scan_only",
            valid_type=orm.Bool,
            default=lambda: orm.Bool(False),
            help="Only determine the lowest energy in one conserved-Sz sector.",
        )
        spec.output("ground_state", valid_type=orm.SinglefileData, required=False)
        spec.output("ground_state_metadata", valid_type=orm.Dict, required=False)
        spec.output("ground_state_summary", valid_type=orm.SinglefileData, required=False)
        spec.exit_code(310, "ERROR_MISSING_RETRIEVED", message="No retrieved folder found.")
        spec.exit_code(311, "ERROR_MISSING_CHECKPOINT", message="GS checkpoint is missing.")
        spec.exit_code(312, "ERROR_MISSING_STATUS", message="GS status is missing.")
        spec.exit_code(313, "ERROR_INVALID_STATUS", message="GS status is invalid.")
        spec.inputs["metadata"]["options"]["parser_name"].default = "itensor_spin.ground_state"
        spec.inputs["metadata"]["options"]["withmpi"].default = False

    def prepare_for_submission(self, folder):
        parameters = normalize_ground_state_parameters(self.inputs.parameters.get_dict())
        if parameters["model"] == GRAPH_MODEL_IDENTIFIER and not self.inputs.code.base.extras.get(
            "supports_spin_graph", False
        ):
            raise ValueError("Choose an isolated spin-graph-capable ground-state Code")
        resources = self.inputs.metadata.options.resources
        threads = int(resources.get("num_cores_per_mpiproc", 1))
        with folder.open("ground_state.json", "w", encoding="utf8") as handle:
            json.dump(
                {
                    "format_version": 1,
                    "parameters": parameters,
                    "execution": {
                        "threads_per_process": threads,
                        "scan_only": self.inputs.scan_only.value,
                    },
                },
                handle,
                indent=2,
                sort_keys=True,
            )
            handle.write("\n")

        code_info = CodeInfo()
        code_info.code_uuid = self.inputs.code.uuid
        code_info.cmdline_params = ["ground_state.json"]
        code_info.stdout_name = "scheduler-stdout.txt"
        code_info.stderr_name = "scheduler-stderr.txt"
        code_info.withmpi = False

        calc_info = CalcInfo()
        calc_info.uuid = self.uuid
        calc_info.codes_info = [code_info]
        calc_info.retrieve_list = ["gs_summary.txt", "gs_status.json"]
        if not self.inputs.scan_only.value:
            calc_info.retrieve_list.insert(0, "gs_checkpoint.jld2")
        return calc_info


class SpectralSegmentCalculation(CalcJob):
    """Run one bounded spectral segment without mutating prior AiiDA outputs."""

    @classmethod
    def define(cls, spec):
        super().define(spec)
        spec.input("ground_state", valid_type=orm.SinglefileData)
        spec.input("parent_folder", valid_type=orm.RemoteData, required=False)
        spec.input(
            "resource_plan",
            valid_type=orm.Dict,
            help="Provenance-recorded output of the resource-planning CalcFunction.",
        )
        spec.input("segment_index", valid_type=orm.Int)
        spec.output("segment_status", valid_type=orm.Dict, required=False)
        spec.output("segment_summary", valid_type=orm.Dict, required=False)
        spec.output("segment_diagnostics", valid_type=orm.Dict, required=False)
        spec.output("spectrum", valid_type=orm.SinglefileData, required=False)
        spec.output("collection_report", valid_type=orm.SinglefileData, required=False)
        spec.output("task_timings", valid_type=orm.SinglefileData, required=False)
        spec.exit_code(300, "ERROR_MISSING_RETRIEVED", message="No retrieved folder found.")
        spec.exit_code(301, "ERROR_MISSING_STATUS", message="segment_status.json is missing.")
        spec.exit_code(302, "ERROR_INVALID_STATUS", message="Invalid segment status report.")
        spec.exit_code(303, "ERROR_MISSING_SUMMARY", message="segment_summary.json is missing.")
        spec.exit_code(304, "ERROR_REMOTE_SEGMENT", message="The remote segment reported failure.")
        spec.exit_code(
            305,
            "ERROR_MISSING_RESULT",
            message="A complete segment is missing compact result files.",
        )
        spec.inputs["metadata"]["options"]["parser_name"].default = "itensor_spin.spectral_segment"
        spec.inputs["metadata"]["options"]["withmpi"].default = False

    def prepare_for_submission(self, folder):
        plan = self.inputs.resource_plan.get_dict()
        if plan.get("state") != "selected":
            raise ValueError("resource_plan must have state='selected'")
        parameters = normalize_parameters(plan["parameters"])
        if parameters.get("max_bond_dimension", 0) and not self.inputs.code.base.extras.get(
            "supports_spectral_bond_cap", False
        ):
            raise ValueError("Selected Code does not declare spectral bond-cap support; use the updated engine")
        layout = normalize_layout(plan["selected_layout"])
        if parameters.get("kernel_threading_mode") == "blas":
            if not self.inputs.code.base.extras.get("supports_blas_execution_policy", False):
                raise ValueError("Selected Code does not support the BLAS execution policy")
            if parameters["blas_threads"] != layout["threads_per_process"]:
                raise ValueError("BLAS threads must match allocated cores per process")
        workers = plan.get("workers")
        capacity = layout["nodes"] * layout["processes_per_node"]
        if (
            plan.get("pending_tasks", 0) < 1
            or isinstance(workers, bool)
            or not isinstance(workers, int)
            or workers < layout["nodes"]
            or workers > capacity
            or workers > plan["pending_tasks"]
        ):
            raise ValueError("resource_plan task/worker metadata is inconsistent")
        if "parent_folder" in self.inputs and (
            self.inputs.parent_folder.computer.uuid != self.inputs.code.computer.uuid
        ):
            raise ValueError("parent_folder and code must use the same computer")

        config = {
            "format_version": 1,
            "segment_index": self.inputs.segment_index.value,
            "parameters": parameters,
            "layout": layout,
            "workers": workers,
            "ground_state": "gs_checkpoint.jld2",
            "resume_state": "resume_state" if "parent_folder" in self.inputs else None,
            "output_state": "state",
            "status_file": "segment_status.json",
            "summary_file": "segment_summary.json",
        }
        with folder.open("segment.json", "w", encoding="utf8") as handle:
            json.dump(config, handle, indent=2, sort_keys=True)
            handle.write("\n")

        remote_symlink_list = []
        if "parent_folder" in self.inputs:
            remote_symlink_list.append(
                (
                    self.inputs.code.computer.uuid,
                    f"{self.inputs.parent_folder.get_remote_path()}/state",
                    "resume_state",
                )
            )

        code_info = CodeInfo()
        code_info.code_uuid = self.inputs.code.uuid
        code_info.cmdline_params = ["segment.json"]
        code_info.stdout_name = "scheduler-stdout.txt"
        code_info.stderr_name = "scheduler-stderr.txt"
        code_info.withmpi = False

        calc_info = CalcInfo()
        calc_info.uuid = self.uuid
        calc_info.codes_info = [code_info]
        calc_info.local_copy_list = [
            (
                self.inputs.ground_state.uuid,
                self.inputs.ground_state.filename,
                "gs_checkpoint.jld2",
            )
        ]
        calc_info.remote_symlink_list = remote_symlink_list
        calc_info.retrieve_list = [
            "segment_status.json",
            "segment_summary.json",
            "scheduler-stdout.txt",
            "scheduler-stderr.txt",
            "compact_result",
        ]
        return calc_info


class SpectralInventoryCalculation(CalcJob):
    """Inventory checkpoint progress in an immutable spectral RemoteData folder."""

    @classmethod
    def define(cls, spec):
        super().define(spec)
        spec.input("ground_state", valid_type=orm.SinglefileData)
        spec.input("parent_folder", valid_type=orm.RemoteData)
        spec.input("parameters", valid_type=orm.Dict)
        spec.input("segment_index", valid_type=orm.Int, default=lambda: orm.Int(0))
        spec.output("segment_status", valid_type=orm.Dict)
        spec.output("inventory_summary", valid_type=orm.Dict)
        spec.exit_code(320, "ERROR_MISSING_RETRIEVED", message="No retrieved folder found.")
        spec.exit_code(321, "ERROR_MISSING_STATUS", message="Inventory status is missing.")
        spec.exit_code(322, "ERROR_INVALID_STATUS", message="Inventory status is invalid.")
        spec.inputs["metadata"]["options"]["parser_name"].default = "itensor_spin.spectral_inventory"
        spec.inputs["metadata"]["options"]["withmpi"].default = False

    def prepare_for_submission(self, folder):
        parameters = normalize_parameters(self.inputs.parameters.get_dict())
        if self.inputs.parent_folder.computer.uuid != self.inputs.code.computer.uuid:
            raise ValueError("parent_folder and code must use the same computer")
        config = {
            "format_version": 1,
            "segment_index": self.inputs.segment_index.value,
            "parameters": parameters,
            "ground_state": "gs_checkpoint.jld2",
            "resume_state": "resume_state",
            "status_file": "inventory_status.json",
            "summary_file": "inventory_summary.json",
        }
        with folder.open("inventory.json", "w", encoding="utf8") as handle:
            json.dump(config, handle, indent=2, sort_keys=True)
            handle.write("\n")

        code_info = CodeInfo()
        code_info.code_uuid = self.inputs.code.uuid
        code_info.cmdline_params = ["--inspect", "inventory.json"]
        code_info.stdout_name = "scheduler-stdout.txt"
        code_info.stderr_name = "scheduler-stderr.txt"
        code_info.withmpi = False

        calc_info = CalcInfo()
        calc_info.uuid = self.uuid
        calc_info.codes_info = [code_info]
        calc_info.local_copy_list = [
            (
                self.inputs.ground_state.uuid,
                self.inputs.ground_state.filename,
                "gs_checkpoint.jld2",
            )
        ]
        calc_info.remote_symlink_list = [
            (
                self.inputs.code.computer.uuid,
                f"{self.inputs.parent_folder.get_remote_path()}/state",
                "resume_state",
            )
        ]
        calc_info.retrieve_list = [
            "inventory_status.json",
            "inventory_summary.json",
            "scheduler-stdout.txt",
            "scheduler-stderr.txt",
        ]
        return calc_info
