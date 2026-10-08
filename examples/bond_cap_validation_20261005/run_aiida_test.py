"""Validate a pinned capped engine through the standard adaptive WorkChain."""
import csv
import hashlib
import io
import json
from pathlib import Path
from aiida import load_profile, orm
from aiida.common.exceptions import NotExistent
from aiida.engine import run_get_node
from aiida.plugins import WorkflowFactory
from aiida_itensor_spin.workflows import derive_resource_plan

load_profile("default")
root = Path(__file__).resolve().parent / "runtime"
digest = hashlib.sha256()
for path in sorted(p for p in root.rglob("*") if p.is_file() and "__pycache__" not in p.parts):
    digest.update(str(path.relative_to(root)).encode())
    digest.update(path.read_bytes())
engine_hash = digest.hexdigest()
computer = orm.load_computer("localhost")
codes = {}
prepend = "\n".join([
    f"export ITENSOR_SPIN_CASE_ROOT={root / 'engine'}",
    "export ITENSOR_SPIN_JULIA=/home/jovyan/opt/julia-1.12.6/bin/julia",
    "export JULIA_DEPOT_PATH=/home/jovyan/.julia-itensor-spin-localhost",
    "export GKSwstype=100",
    "export OMP_NUM_THREADS=1",
    "export OPENBLAS_NUM_THREADS=1",
])
for kind, plugin in [("gs", "ground_state"), ("segment", "spectral_segment")]:
    label = f"itensor-spin-{kind}-0.4.3-bondcap-local"
    try:
        code = orm.load_code(f"{label}@localhost")
        assert code.base.extras.get("engine_hash") == engine_hash
    except NotExistent:
        code = orm.InstalledCode(computer=computer, label=label,
            filepath_executable=str(root / f"itensor-spin-{kind}-local"),
            default_calc_job_plugin=f"itensor_spin.{plugin}", with_mpi=False,
            prepend_text=prepend, description="Pinned optional spectral bond-cap validation engine").store()
        code.base.extras.set_many({"engine_hash":engine_hash,
            "supports_spectral_bond_cap":kind == "segment"})
    codes[kind] = code

reference = orm.load_node("ccd512fe-7639-4222-9075-51919c745855")
parameters = reference.inputs.spectral_parameters.get_dict()
parameters.update(max_bond_dimension=2, spectral_cutoff=1e-10)
parameters = orm.Dict(dict=parameters)
limits = orm.Dict(dict={"max_nodes":1,"max_processes_per_node":1,
    "max_threads_per_process":2,"wallclock_seconds":900,
    "internal_stop_seconds":780,"partition":"local"})
plan = derive_resource_plan(reference.outputs.ground_state_metadata, parameters, limits)
builder = WorkflowFactory("itensor_spin.spectra_adaptive").get_builder()
builder.code = codes["segment"]
builder.ground_state = reference.outputs.ground_state
builder.ground_state_metadata = reference.outputs.ground_state_metadata
builder.spectral_parameters = parameters
builder.resource_limits = limits
builder.initial_resource_plan = plan
builder.max_iterations = orm.Int(1)
builder.max_node_hours = orm.Float(1.)
builder.on_unhandled_failure = orm.Str("abort")
builder.pause_on_max_iterations = orm.Bool(False)
builder.metadata.label = "Bond-cap software validation L3 (cap=2)"
outputs, node = run_get_node(builder)
print("WORKCHAIN", node.pk, node.uuid, node.process_state, node.exit_status, flush=True)
assert node.is_finished_ok
rows = list(csv.DictReader(io.StringIO(outputs["task_timings"].get_content()), delimiter="\t"))
assert len(rows) == 9
assert all(int(row["max_bond_dimension"]) == 2 for row in rows)
assert any(int(row["cap_reached_bonds"]) > 0 for row in rows)
assert "compression_diagnostics=" in outputs["collection_report"].get_content()
print(json.dumps({"passed":True,"workchain_uuid":str(node.uuid),
    "task_count":len(rows),"max_bond_dimension":2,
    "maximum_discarded_weight":max(float(row["max_discarded_weight"]) for row in rows),
    "code_labels":{k:v.full_label for k,v in codes.items()}},indent=2),flush=True)
