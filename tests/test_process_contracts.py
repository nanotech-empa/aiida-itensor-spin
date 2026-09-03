"""Regression tests for the AiiDA process and parser interfaces."""

import inspect

from aiida_itensor_spin.calculations import (
    GroundStateCalculation,
    SpectralSegmentCalculation,
)
from aiida_itensor_spin.parsers import GroundStateParser, SpectralSegmentParser
from aiida_itensor_spin.workflows import SpinChainWorkChain, SpinSpectraWorkChain


def test_calculation_inputs_are_stage_specific():
    gs_inputs = GroundStateCalculation.spec().inputs
    segment_inputs = SpectralSegmentCalculation.spec().inputs

    assert "parameters" in gs_inputs
    assert "resource_plan" not in gs_inputs
    assert "resource_plan" in segment_inputs
    assert "parameters" not in segment_inputs
    assert "layout" not in segment_inputs


def test_parsers_read_the_matching_calculation_inputs():
    gs_source = inspect.getsource(GroundStateParser.parse)
    segment_source = inspect.getsource(SpectralSegmentParser.parse)

    assert "self.node.inputs.parameters" in gs_source
    assert "self.node.inputs.resource_plan" not in gs_source
    assert "self.node.inputs.resource_plan" in segment_source


def test_workflow_contract_is_complete_and_provenance_safe():
    full_inputs = SpinChainWorkChain.spec().inputs
    full_outputs = SpinChainWorkChain.spec().outputs
    spectra_inputs = SpinSpectraWorkChain.spec().inputs

    assert {
        "ground_state_parameters",
        "ground_state_resources",
        "spectral_parameters",
        "spectral_resources",
    }.issubset(full_inputs)
    assert {
        "ground_state",
        "ground_state_metadata",
        "resource_plan",
        "final_resource_plan",
        "spectrum",
        "collection_report",
        "task_timings",
    }.issubset(full_outputs)
    assert "initial_resource_plan" in spectra_inputs
    assert "max_failed_segments" in spectra_inputs
    assert "max_failed_segments" in full_inputs


def test_segment_driver_publishes_status_before_srun():
    with open("remote/itensor-spin-segment", encoding="utf8") as handle:
        driver = handle.read()
    initial_status = driver.index(
        'summary_payload(segment_index, layout, imported, started, "initialized")'
    )
    worker_srun = driver.index('"srun"', initial_status)
    assert initial_status < worker_srun


def test_progress_scan_never_deserializes_task_checkpoints():
    with open("remote/segment_progress.jl", encoding="utf8") as handle:
        source = handle.read()
    assert "data = load(joinpath(root, filename))" not in source
    assert 'filename * ".status"' in source
