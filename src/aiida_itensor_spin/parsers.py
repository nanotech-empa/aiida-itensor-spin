"""Parser for one restartable ITensor spectral segment."""

from __future__ import annotations

import json
import math

from aiida import orm
from aiida.parsers import Parser

from .contracts import (
    ContractError,
    validate_ground_state_energy_window,
    validate_segment_status,
)


def _read_json(retrieved, filename: str):
    with retrieved.base.repository.open(filename, mode="r") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise TypeError(f"{filename} must contain a JSON object")
    return value


def _segment_diagnostics(retrieved, names: set[str]) -> dict:
    """Classify scheduler/driver failures without making stderr part of workflow logic."""
    chunks = []
    for filename in ("scheduler-stderr.txt", "_scheduler-stderr.txt"):
        if filename not in names:
            continue
        try:
            with retrieved.base.repository.open(filename, mode="r") as handle:
                chunks.append(handle.read())
        except (OSError, UnicodeError):
            continue
    text = "\n".join(chunks)
    lowered = text.lower()
    oom_markers = ("out of memory", "oom-kill", "oom_kill", "killed process")
    oom_detected = any(marker in lowered for marker in oom_markers)
    worker_failure = "returned non-zero exit status" in lowered or "force terminated" in lowered
    if oom_detected:
        failure_kind = "out_of_memory"
    elif worker_failure:
        failure_kind = "worker_step_failure"
    else:
        failure_kind = "none"
    return {
        "format_version": 1,
        "failure_kind": failure_kind,
        "oom_detected": oom_detected,
        "worker_step_failure_detected": worker_failure,
    }


def _validate_spectrum(retrieved) -> None:
    """Check that the compact spectrum is non-empty, tabular, and finite."""
    rows = 0
    with retrieved.base.repository.open("compact_result/spectrum.tsv", mode="r") as handle:
        if handle.readline().rstrip("\n") != "state\tsite\tomega_meV\tchi":
            raise ValueError("invalid spectrum.tsv header")
        for line in handle:
            fields = line.rstrip("\n").split("\t")
            if len(fields) != 4:
                raise ValueError("invalid spectrum.tsv row")
            state, site = int(fields[0]), int(fields[1])
            omega, chi = float(fields[2]), float(fields[3])
            if state < 1 or site < 1 or not math.isfinite(omega) or not math.isfinite(chi):
                raise ValueError("invalid spectrum.tsv value")
            rows += 1
    if rows == 0:
        raise ValueError("spectrum.tsv contains no data rows")


class GroundStateParser(Parser):
    """Store a sector scan or a durable GS checkpoint with compact metadata."""

    def parse(self, **kwargs):
        try:
            retrieved = self.retrieved
        except (AttributeError, KeyError):
            return self.exit_codes.ERROR_MISSING_RETRIEVED
        names = set(retrieved.base.repository.list_object_names())
        scan_only = self.node.inputs.scan_only.value
        if not scan_only and "gs_checkpoint.jld2" not in names:
            return self.exit_codes.ERROR_MISSING_CHECKPOINT
        if "gs_status.json" not in names or "gs_summary.txt" not in names:
            return self.exit_codes.ERROR_MISSING_STATUS
        try:
            status = _read_json(retrieved, "gs_status.json")
            if status.get("format_version") != 1 or status.get("state") != "complete":
                raise ValueError("invalid GS status")
            if status.get("scan_only", False) is not scan_only:
                raise ValueError("GS scan mode differs from input")
            expected = self.node.inputs.parameters.get_dict()
            if status.get("length") != expected.get("length"):
                raise ValueError("GS length differs from input")
            if "model" in expected and status.get("model") != expected["model"]:
                raise ValueError("GS model differs from input")
            ground_energy = status.get("ground_energy_meV")
            if (
                isinstance(ground_energy, bool)
                or not isinstance(ground_energy, (int, float))
                or not math.isfinite(ground_energy)
            ):
                raise ValueError("invalid GS energy")
            if scan_only:
                if status.get("ground_states") != 1 or status.get("total_tasks") != 0:
                    raise ValueError("invalid sector-scan dimensions")
            else:
                validate_ground_state_energy_window(status)
                ground_states = status.get("ground_states")
                total_tasks = status.get("total_tasks")
                if (
                    isinstance(ground_states, bool)
                    or not isinstance(ground_states, int)
                    or ground_states < 1
                ):
                    raise ValueError("invalid GS ground-state count")
                if total_tasks != 3 * status["length"] * ground_states:
                    raise ValueError("invalid GS spectral task count")
        except (
            OSError,
            UnicodeError,
            json.JSONDecodeError,
            ValueError,
            ContractError,
        ):
            return self.exit_codes.ERROR_INVALID_STATUS

        if not scan_only:
            with retrieved.base.repository.open("gs_checkpoint.jld2", mode="rb") as handle:
                self.out(
                    "ground_state",
                    orm.SinglefileData(file=handle, filename="gs_checkpoint.jld2"),
                )
        with retrieved.base.repository.open("gs_summary.txt", mode="rb") as handle:
            self.out(
                "ground_state_summary",
                orm.SinglefileData(file=handle, filename="gs_summary.txt"),
            )
        self.out("ground_state_metadata", orm.Dict(dict=status))
        return None


class SpectralSegmentParser(Parser):
    """Parse the compact status while large checkpoints remain remote."""

    def parse(self, **kwargs):
        try:
            retrieved = self.retrieved
        except (AttributeError, KeyError):
            return self.exit_codes.ERROR_MISSING_RETRIEVED

        names = set(retrieved.base.repository.list_object_names())
        self.out("segment_diagnostics", orm.Dict(dict=_segment_diagnostics(retrieved, names)))
        if "segment_status.json" not in names:
            return self.exit_codes.ERROR_MISSING_STATUS
        if "segment_summary.json" not in names:
            return self.exit_codes.ERROR_MISSING_SUMMARY
        try:
            raw_status = _read_json(retrieved, "segment_status.json")
            status = validate_segment_status(
                raw_status, self.node.inputs.resource_plan.get_dict()["parameters"]
            )
            summary = _read_json(retrieved, "segment_summary.json")
        except (OSError, UnicodeError, json.JSONDecodeError, TypeError, ValueError, ContractError):
            return self.exit_codes.ERROR_INVALID_STATUS

        if status["state"] == "complete":
            try:
                result_names = set(retrieved.base.repository.list_object_names("compact_result"))
            except FileNotFoundError:
                return self.exit_codes.ERROR_MISSING_RESULT
            required = {"spectrum.tsv", "collection_report.txt", "task_timings.tsv"}
            if not required.issubset(result_names):
                return self.exit_codes.ERROR_MISSING_RESULT
            try:
                _validate_spectrum(retrieved)
            except (OSError, UnicodeError, ValueError):
                return self.exit_codes.ERROR_MISSING_RESULT

        self.out("segment_status", orm.Dict(dict=status))
        self.out("segment_summary", orm.Dict(dict=summary))
        if status["state"] == "complete":
            for label, filename in (
                ("spectrum", "spectrum.tsv"),
                ("collection_report", "collection_report.txt"),
                ("task_timings", "task_timings.tsv"),
            ):
                with retrieved.base.repository.open(
                    f"compact_result/{filename}", mode="rb"
                ) as handle:
                    self.out(label, orm.SinglefileData(file=handle, filename=filename))
        return None


class SpectralInventoryParser(Parser):
    """Parse a provenance-recorded inventory of an immutable restart folder."""

    def parse(self, **kwargs):
        try:
            retrieved = self.retrieved
        except (AttributeError, KeyError):
            return self.exit_codes.ERROR_MISSING_RETRIEVED
        names = set(retrieved.base.repository.list_object_names())
        if not {"inventory_status.json", "inventory_summary.json"}.issubset(names):
            return self.exit_codes.ERROR_MISSING_STATUS
        try:
            status = validate_segment_status(
                _read_json(retrieved, "inventory_status.json"),
                self.node.inputs.parameters.get_dict(),
            )
            summary = _read_json(retrieved, "inventory_summary.json")
            if summary.get("format_version") != 1 or summary.get("phase") != "inventory":
                raise ValueError("invalid inventory summary")
        except (OSError, UnicodeError, json.JSONDecodeError, TypeError, ValueError, ContractError):
            return self.exit_codes.ERROR_INVALID_STATUS
        self.out("segment_status", orm.Dict(dict=status))
        self.out("inventory_summary", orm.Dict(dict=summary))
        return None
