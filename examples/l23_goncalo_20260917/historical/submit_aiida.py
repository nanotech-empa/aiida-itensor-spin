"""Register isolated fixed-reference Codes and submit the standard WorkChain."""
import datetime
import hashlib
import json
from pathlib import Path
import sys
import tomllib

from aiida import orm
from aiida.common.exceptions import NotExistent
from aiida.engine import submit
from aiida.plugins import WorkflowFactory

LOCAL = Path("/home/jovyan/updatespin_l23_validation_20260917")
ROOT = "/ritom/scratch/cscs/cpi/goncalo/updatespin_l23_validation_20260917"
RUNTIME = "/ritom/scratch/cscs/cpi/goncalo/l101_maintenance_20260915/runtime"
LABEL = "L23 Goncalo September17 gold-standard full workflow"
COMPUTER = "daint.alps_lp83_ritom"
DEADLINE = datetime.datetime(2026, 9, 17, 12, 0, tzinfo=datetime.timezone.utc)


def register():
    computer = orm.load_computer(COMPUTER)
    assert computer.get_workdir() == "/ritom/scratch/cscs/{username}/aiida/"
    assert computer.hostname == "daint.alps.cscs.ch"
    physics = tomllib.loads((LOCAL / "physics.toml").read_text())
    manifest_digest = hashlib.sha256((LOCAL / "SHA256SUMS").read_bytes()).hexdigest()
    prepend = "\n".join([
        "#SBATCH --uenv=julia/26.3:v1", "#SBATCH --view=juliaup",
        f"#SBATCH --repo={RUNTIME}/uenv-repo",
        "#SBATCH --deadline=2026-09-17T14:00:00",
        f"export JULIA_DEPOT_PATH={ROOT}/depot:{RUNTIME}/depot",
        f"export ITENSOR_SPIN_CASE_ROOT={ROOT}/engine",
        f"export ITENSOR_SPIN_JULIA={RUNTIME}/julia-1.12.6/bin/julia",
        "export GKSwstype=100", "ulimit -s unlimited",
        f"(cd {ROOT} && printf '%s  SHA256SUMS\\n' {manifest_digest} | sha256sum -c - && sha256sum -c SHA256SUMS) || exit 90",
    ])
    codes = {}
    for kind, plugin in (("gs", "itensor_spin.ground_state"), ("segment", "itensor_spin.spectral_segment")):
        code_label = f"itensor-spin-{kind}-l23-gold-20260917"
        append = ""
        if kind == "segment":
            append = "\n".join([
                "if test -f state/merged/chi_L23.jld2; then",
                f'  env JULIA_NUM_THREADS=1,0 "$ITENSOR_SPIN_JULIA" --startup-file=no --project={ROOT}/engine/ITensor_env {ROOT}/compare_reference.jl {ROOT} "$PWD" > compact_result/golden-comparison.txt 2> compact_result/golden-comparison.err || exit 91',
                "  cp comparison.toml comparison_by_site.tsv compact_result/",
                "fi",
            ])
        try:
            code = orm.load_code(f"{code_label}@{COMPUTER}")
            assert code.prepend_text == prepend and code.append_text == append
        except NotExistent:
            code = orm.InstalledCode(
                computer=computer, label=code_label,
                filepath_executable=f"{ROOT}/engine/itensor-spin-{kind}",
                default_calc_job_plugin=plugin, with_mpi=False,
                prepend_text=prepend, append_text=append,
                description="Isolated L23 gold-standard inputs: J=18.4, JL=16.7, JR=51.1 meV; DMRG cutoff=1e-8; spectral cutoff=1e-6; broadening=1.5 meV. No high-energy projection. Fixed inputs pinned by source checksum manifest.",
            ).store()
            code.base.extras.set_many({
                "reference_physics": physics,
                "source_commit": "210f0830c3e360a09c84acd8a25c8715062c9775",
                "source_manifest_sha256": manifest_digest,
                "validation_reference": str(LOCAL / "reference"),
            })
        codes[kind] = code
    return codes


def main():
    codes = register()
    if sys.argv[1:] == ["register"]:
        print(json.dumps({kind: {"pk": code.pk, "uuid": code.uuid, "label": code.full_label} for kind, code in codes.items()}, indent=2))
        return
    if sys.argv[1:] != ["submit"]:
        raise SystemExit("Usage: verdi run submit_aiida.py register|submit")
    now = datetime.datetime.now(datetime.timezone.utc)
    if now + datetime.timedelta(minutes=40) >= DEADLINE:
        raise RuntimeError("Insufficient time for the 10-minute GS and 30-minute spectral requests before 14:00 CEST.")
    receipt = LOCAL / "aiida_submission.json"
    assert not receipt.exists(), "Refusing duplicate submission"
    existing = orm.QueryBuilder().append(orm.WorkChainNode, filters={"label": LABEL}, project=["uuid"]).all()
    assert not existing, f"Workflow already exists: {existing}"
    builder = WorkflowFactory("itensor_spin.full").get_builder()
    builder.metadata.label = LABEL
    builder.metadata.description = "Fresh L23 GS and full 69-task gold-standard validation. Fixed-reference test Codes use Goncalo's approved asymmetric couplings, 1e-6 spectral cutoff and 1.5 meV broadening. Deadline 14:00 CEST; one spectral segment only; L101 untouched."
    builder.ground_state_code = codes["gs"]
    builder.spectral_code = codes["segment"]
    builder.ground_state_parameters = orm.Dict(dict={"length": 23, "impurity_site": 12, "seed": 20260724, "threading_mode": "blocksparse"})
    builder.ground_state_resources = orm.Dict(dict={"nodes": 1, "processes_per_node": 1, "threads_per_process": 8, "wallclock_seconds": 600, "internal_stop_seconds": 540, "partition": "debug"})
    builder.spectral_parameters = orm.Dict(dict={"chebyshev_convention": "chebyshev-2n-minus-1-v2", "backend": "streaming", "threading_mode": "blocksparse", "ground_state_filename": "gs_checkpoint.jld2", "checkpoint_every": 10, "heartbeat_every": 1})
    builder.spectral_resources = orm.Dict(dict={"max_nodes": 1, "max_processes_per_node": 23, "max_threads_per_process": 2, "wallclock_seconds": 1800, "internal_stop_seconds": 1560, "partition": "debug"})
    builder.max_segments = orm.Int(1)
    builder.max_node_hours = orm.Float(1.0)
    builder.max_stalled_segments = orm.Int(1)
    builder.max_failed_segments = orm.Int(0)
    node = submit(builder)
    payload = {"workchain_pk": node.pk, "workchain_uuid": node.uuid, "codes": {kind: code.uuid for kind, code in codes.items()}, "physics": tomllib.loads((LOCAL / "physics.toml").read_text()), "submitted_utc": now.isoformat(), "deadline_utc": DEADLINE.isoformat()}
    with receipt.open("x") as handle:
        json.dump(payload, handle, indent=2)
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
