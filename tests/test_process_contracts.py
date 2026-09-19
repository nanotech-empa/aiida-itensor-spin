"""Regression tests for the AiiDA process and parser interfaces."""

import inspect
from importlib.machinery import SourceFileLoader
from importlib.util import module_from_spec, spec_from_loader
from pathlib import Path

from aiida_itensor_spin.calculations import (
    GroundStateCalculation,
    SpectralSegmentCalculation,
)
from aiida_itensor_spin.parsers import GroundStateParser, SpectralSegmentParser
from aiida_itensor_spin.workflows import (
    SpinChainWorkChain,
    SpinSectorEnsembleWorkChain,
    SpinSpectraWorkChain,
)


def load_segment_driver():
    path = Path("remote/itensor-spin-segment")
    loader = SourceFileLoader("itensor_spin_segment_driver", str(path))
    spec = spec_from_loader(loader.name, loader)
    module = module_from_spec(spec)
    loader.exec_module(module)
    return module


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


def test_sector_ensemble_contract_is_explicit_and_namespaced():
    inputs = SpinSectorEnsembleWorkChain.spec().inputs
    outputs = SpinSectorEnsembleWorkChain.spec().outputs

    assert {
        "ground_state_parameters",
        "requested_sectors",
        "ensemble_resources",
        "spectral_resources",
    }.issubset(inputs)
    assert {"sector_plan", "spectral_sector_plan", "sector_summary"}.issubset(outputs)
    assert outputs["sector_scan_summaries"].dynamic
    assert outputs["sector_scan_metadata"].dynamic
    assert outputs["sector_ground_state_summaries"].dynamic
    assert outputs["sector_metadata"].dynamic
    assert outputs["sector_spectra"].dynamic
    assert SpinSectorEnsembleWorkChain.exit_codes.ERROR_INVALID_INPUT.status == 600
    assert SpinSectorEnsembleWorkChain.exit_codes.ERROR_SECTOR_SCAN_FAILED.status == 601
    assert SpinSectorEnsembleWorkChain.exit_codes.ERROR_SECTOR_SPECTRUM_FAILED.status == 602


def test_segment_driver_publishes_status_before_srun():
    with open("remote/itensor-spin-segment", encoding="utf8") as handle:
        driver = handle.read()
    initial_status = driver.index('started, "initialized"')
    worker_srun = driver.index('"srun"', initial_status)
    assert initial_status < worker_srun


def test_progress_scan_never_deserializes_task_checkpoints():
    with open("remote/segment_progress.jl", encoding="utf8") as handle:
        source = handle.read()
    assert "data = load(joinpath(root, filename))" not in source
    assert 'filename * ".status"' in source


def test_unfinished_task_manifest_uses_every_worker_without_collisions():
    driver = load_segment_driver()
    pending = {
        454,
        455,
        463,
        465,
        467,
        473,
        475,
        481,
        715,
        721,
        739,
        774,
        775,
        779,
        783,
        785,
        787,
        793,
    }
    completed = set(range(1, 910)) - pending
    progress = {task_id: (500 + task_id % 100, 1094) for task_id in pending}
    assignments, task_to_rank, summary = driver.build_task_assignments(
        909, completed, progress, len(pending)
    )

    assert set(task_to_rank) == pending
    assert len(assignments) == len(pending)
    assert all(len(task_ids) == 1 for task_ids in assignments)
    assert summary["active_workers"] == len(pending)
    assert summary["idle_workers"] == 0


def test_manifest_balances_more_pending_tasks_than_workers():
    driver = load_segment_driver()
    completed = set(range(1, 13)) - {1, 2, 3, 4, 5, 6}
    progress = {
        1: (10, 100),
        2: (20, 100),
        3: (30, 100),
        4: (40, 100),
        5: (50, 100),
        6: (60, 100),
    }
    assignments, task_to_rank, summary = driver.build_task_assignments(12, completed, progress, 2)

    assert set(task_to_rank) == {1, 2, 3, 4, 5, 6}
    assert all(assignments)
    assert summary["maximum_tasks_per_worker"] == 3
    assert summary["maximum_estimated_recurrence_units"] == 200
    assert summary["minimum_estimated_recurrence_units"] == 190
