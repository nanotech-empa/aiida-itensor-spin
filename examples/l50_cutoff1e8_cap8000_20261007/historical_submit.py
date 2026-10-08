"""Submit the L50 controlled test: fresh GS, cutoff 1e-8, bond cap 8000."""

import argparse
import json
from pathlib import Path

from aiida import load_profile, orm
from aiida.engine import submit
from aiida.plugins import WorkflowFactory
from aiida_itensor_spin.contracts import (
    normalize_ground_state_parameters,
    normalize_layout,
    normalize_parameters,
    normalize_resource_limits,
    select_largest_feasible_layout,
)


REFERENCE_UUID = "02cf7e9a-aeda-40db-8ad8-d09a89b05086"
CODE_UUID = "204b0f6a-f3cf-4b7a-be2f-01ec4ddf7671"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    load_profile("default")
    reference = orm.load_node(REFERENCE_UUID)
    assert reference.is_finished_ok

    physics = reference.inputs.ground_state_parameters.get_dict()
    original = dict(physics)
    physics["spectral_cutoff"] = 1e-8
    physics = normalize_ground_state_parameters(physics)
    assert {key for key in physics if physics[key] != original[key]} == {"spectral_cutoff"}
    assert physics["length"] == 50 and physics["target_twice_sz"] == 0
    assert physics["site_spins"][24:26] == [0.5, 0.5]
    assert physics["couplings_meV"][23:26] == [16.7, 96.8, 16.7]

    spectral = reference.inputs.spectral_parameters.get_dict()
    spectral.update(spectral_cutoff=1e-8, max_bond_dimension=8000)
    spectral = normalize_parameters(spectral)
    limits = normalize_resource_limits({
        "max_nodes": 30,
        "max_processes_per_node": 10,
        "max_threads_per_process": 2,
        "wallclock_seconds": 50400,
        "internal_stop_seconds": 43200,
        "partition": "normal",
    })
    expected_tasks = reference.outputs.ground_state_metadata.get_dict()["total_tasks"]
    assert expected_tasks == 300
    preview = select_largest_feasible_layout(expected_tasks, limits)
    assert (preview["nodes"], preview["processes_per_node"], preview["workers"]) == (30, 10, 300)

    builder = WorkflowFactory("itensor_spin.full").get_builder()
    builder.ground_state_code = orm.load_code(
        "itensor-spin-gs-0.4.3-bondcap@daint.alps_lp83_ritom"
    )
    builder.spectral_code = orm.load_code(CODE_UUID)
    assert builder.spectral_code.base.extras.get("supports_spectral_bond_cap", False)
    assert builder.spectral_code.base.extras.get("engine_manifest_sha256") == (
        "739f114a37287091acfcc68c7b97c5b35aa84741075fa9579a2b1eee84a41191"
    )
    assert builder.ground_state_code.computer.uuid == builder.spectral_code.computer.uuid
    assert builder.spectral_code.computer.hostname == "daint.alps.cscs.ch"
    assert builder.spectral_code.computer.get_workdir().startswith("/ritom/")
    builder.ground_state_parameters = orm.Dict(dict=physics)
    builder.spectral_parameters = orm.Dict(dict=spectral)
    builder.ground_state_resources = orm.Dict(dict=normalize_layout({
        "nodes": 1,
        "processes_per_node": 1,
        "threads_per_process": 8,
        "wallclock_seconds": 1800,
        "internal_stop_seconds": 1500,
        "partition": "normal",
    }))
    builder.spectral_resources = orm.Dict(dict=limits)
    # Retain the original L50 node-hour budget. Each spectral segment charges
    # at least 14 node-hours, so <=142 fit in this ceiling. A counter limit of
    # 200 cannot trigger the installed AiiDA 2.8 pause/reset defect before the
    # node-hour safety stop. This avoids the defect; it does not patch core.
    builder.max_segments = orm.Int(200)
    builder.max_node_hours = orm.Float(2000.0)
    builder.max_stalled_segments = orm.Int(1)
    builder.max_failed_segments = orm.Int(2)
    builder.metadata.label = "L50 double impurity: cutoff 1e-8, bond cap 8000"
    builder.metadata.description = (
        "Fresh standard full AiiDA workflow, same S/J/Sz/seed/DMRG controls as "
        "the L50 cutoff-1e-6 reference. GS is regenerated once on one node with "
        "8 threads and 30-minute walltime. No loose-cutoff spectral tasks are "
        "reused. Spectral cutoff 1e-8, max bond dimension 8000; cap hits and "
        "discarded-weight excesses are reported. Initial ceiling: 30 nodes x "
        "10 processes/node x 2 threads/process, expected 300 concurrent tasks. "
        "14-hour spectral walltime, internal stop at 12 hours with the existing "
        "per-iteration time forecast; max_node_hours=2000, inherited from the "
        "reference. Adaptive checkpoint continuations; OOM/error/stall pauses "
        "require user inspection. max_segments=200 keeps the known installed "
        "core counter-reset path unreachable within this node-hour budget. "
        "L101 and all existing codes/checkpoints are unchanged."
    )
    error = builder._process_class.spec().inputs.validate(builder._inputs(prune=True))
    if error:
        raise ValueError(error)

    # Check labels and scientific input metadata to catch an equivalent run
    # even if another client gave it a different display label.
    candidates = orm.QueryBuilder().append(
        orm.WorkChainNode,
        filters={"attributes.process_label": {"in": [
            "SpinChainWorkChain", "SpinSpectraWorkChain", "SpinSpectraRestartWorkChain",
            "AdaptiveSpinSpectraWorkChain",
        ]}},
        project=["*"],
    ).all(flat=True)
    duplicates = []
    for candidate in candidates:
        if "spectral_parameters" not in candidate.inputs:
            continue
        params = candidate.inputs.spectral_parameters.get_dict()
        if params.get("spectral_cutoff") != 1e-8:
            continue
        if "ground_state_parameters" in candidate.inputs:
            candidate_physics = candidate.inputs.ground_state_parameters.get_dict()
        elif "ground_state_metadata" in candidate.inputs:
            candidate_physics = candidate.inputs.ground_state_metadata.get_dict()
        else:
            continue
        if (
            candidate_physics.get("length") == 50
            and candidate_physics.get("site_spins") == physics["site_spins"]
            and candidate_physics.get("couplings_meV") == physics["couplings_meV"]
            and candidate_physics.get("target_twice_sz") == 0
        ):
            duplicates.append({"pk": candidate.pk, "uuid": str(candidate.uuid),
                               "state": str(candidate.process_state), "parameters": params})
    if duplicates:
        raise RuntimeError(f"Refusing duplicate L50 strict-spectrum run: {duplicates}")
    record = Path(__file__).with_name("submission.json")
    if record.exists():
        raise RuntimeError("Refusing duplicate submission: submission.json exists")

    details = {
        "reference_uuid": REFERENCE_UUID,
        "physics": physics,
        "spectral_parameters": spectral,
        "ground_state_resources": builder.ground_state_resources.get_dict(),
        "spectral_resources": limits,
        "preview_based_on_reference_gs": preview,
        "preview_is_not_the_new_post_gs_plan": True,
        "max_segments": builder.max_segments.value,
        "max_node_hours": builder.max_node_hours.value,
        "ground_state_reused": False,
        "previous_spectral_tasks_reused": False,
    }
    if args.dry_run:
        print(json.dumps({"validated": True, "submitted": False, **details}, indent=2))
        return
    node = submit(builder)
    node.base.extras.set_many({
        "aiidalab_app": "aiidalab-itensor-spectra",
        "hamiltonian_model": physics["model"],
        "requested_twice_sz_sectors": [0],
        "comparison_reference_uuid": REFERENCE_UUID,
    })
    result = {"pk": node.pk, "uuid": str(node.uuid), "label": node.label, **details}
    record.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
