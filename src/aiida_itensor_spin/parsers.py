"""Parser for one restartable ITensor spectral segment."""

from __future__ import annotations

import json

from aiida import orm
from aiida.parsers import Parser

from .contracts import ContractError, validate_segment_status


def _read_json(retrieved, filename: str):
    with retrieved.base.repository.open(filename, mode="r") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"{filename} must contain a JSON object")
    return value


class GroundStateParser(Parser):
    """Store compact GS metadata and its durable checkpoint in AiiDA."""

    def parse(self, **kwargs):  # noqa: ARG002
        try:
            retrieved = self.retrieved
        except (AttributeError, KeyError):
            return self.exit_codes.ERROR_MISSING_RETRIEVED
        names = set(retrieved.base.repository.list_object_names())
        if "gs_checkpoint.jld2" not in names:
            return self.exit_codes.ERROR_MISSING_CHECKPOINT
        if "gs_status.json" not in names or "gs_summary.txt" not in names:
            return self.exit_codes.ERROR_MISSING_STATUS
        try:
            status = _read_json(retrieved, "gs_status.json")
            if status.get("format_version") != 1 or status.get("state") != "complete":
                raise ValueError("invalid GS status")
            expected = self.node.inputs.parameters.get_dict()
            if status.get("length") != expected.get("length"):
                raise ValueError("GS length differs from input")
        except (OSError, UnicodeError, json.JSONDecodeError, ValueError):
            return self.exit_codes.ERROR_INVALID_STATUS

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

    def parse(self, **kwargs):  # noqa: ARG002
        try:
            retrieved = self.retrieved
        except (AttributeError, KeyError):
            return self.exit_codes.ERROR_MISSING_RETRIEVED

        names = set(retrieved.base.repository.list_object_names())
        if "segment_status.json" not in names:
            return self.exit_codes.ERROR_MISSING_STATUS
        if "segment_summary.json" not in names:
            return self.exit_codes.ERROR_MISSING_SUMMARY
        try:
            raw_status = _read_json(retrieved, "segment_status.json")
            status = validate_segment_status(
                raw_status, self.node.inputs.parameters.get_dict()
            )
            summary = _read_json(retrieved, "segment_summary.json")
        except (OSError, UnicodeError, json.JSONDecodeError, ValueError, ContractError):
            return self.exit_codes.ERROR_INVALID_STATUS

        self.out("segment_status", orm.Dict(dict=status))
        self.out("segment_summary", orm.Dict(dict=summary))
        if raw_status.get("state") == "failed":
            return self.exit_codes.ERROR_REMOTE_SEGMENT
        return None

