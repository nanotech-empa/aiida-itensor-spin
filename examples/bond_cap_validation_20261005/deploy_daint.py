"""Create a separate immutable Daint engine; never overwrite an existing Code."""
import hashlib
import json
from pathlib import Path
import re
import shlex
import shutil
import subprocess
import tempfile
from aiida import load_profile, orm
from aiida.common.exceptions import NotExistent

load_profile("default")
source = Path("/home/jovyan/opt/aiida-itensor-spin-localhost/remote")
destination = "/ritom/scratch/cscs/cpi/goncalo/aiida_itensor_spin_0.4.3_bondcap_20261005"
assert destination.startswith("/ritom/scratch/cscs/cpi/goncalo/aiida_itensor_spin_")
host = "daint.alps.cscs.ch"
stage = Path(tempfile.mkdtemp(prefix="itensor-bond-cap-deploy-"))
shutil.copytree(source / "engine", stage / "engine", ignore=shutil.ignore_patterns("__pycache__"))
for filename in ("itensor-spin-gs", "itensor-spin-segment", "run_segment_rank.sh", "segment_progress.jl"):
    shutil.copy2(source / filename, stage / "engine" / filename)
entries = []
for file in sorted(p for p in stage.rglob("*") if p.is_file()):
    entries.append(f"{hashlib.sha256(file.read_bytes()).hexdigest()}  {file.relative_to(stage)}")
manifest = stage / "SHA256SUMS"
manifest.write_text("\n".join(entries) + "\n")
manifest_hash = hashlib.sha256(manifest.read_bytes()).hexdigest()
command = f"test ! -e {shlex.quote(destination)} && mkdir -p {shlex.quote(destination)}"
subprocess.run(["ssh", "-o", "BatchMode=yes", host, command], check=True)
subprocess.run(["rsync", "-a", str(stage) + "/", host + ":" + destination + "/"], check=True)
verified = subprocess.run(["ssh", "-o", "BatchMode=yes", host,
    f"cd {shlex.quote(destination)} && sha256sum -c SHA256SUMS"],
    check=True, capture_output=True, text=True)
assert all(line.endswith(": OK") for line in verified.stdout.splitlines())
report = {"destination":destination,"manifest_sha256":manifest_hash,"codes":{}}
for kind in ("gs", "segment"):
    old = orm.load_code(f"itensor-spin-{kind}-0.4.3@daint.alps_lp83_ritom")
    label = f"itensor-spin-{kind}-0.4.3-bondcap"
    try:
        orm.load_code(f"{label}@{old.computer.label}")
    except NotExistent:
        pass
    else:
        raise RuntimeError("Refusing to overwrite an existing versioned Code")
    old_root = str(Path(old.filepath_executable).parent.parent)
    hashes = re.findall(r"\b[a-f0-9]{64}\b", old.prepend_text)
    assert len(hashes) == 1
    prepend = old.prepend_text.replace(old_root, destination).replace(hashes[0], manifest_hash)
    code = orm.InstalledCode(computer=old.computer, label=label,
        filepath_executable=f"{destination}/engine/itensor-spin-{kind}",
        default_calc_job_plugin=old.default_calc_job_plugin,
        prepend_text=prepend, with_mpi=False,
        description="Pinned optional spectral MPS bond-cap engine; old production engine unchanged").store()
    code.base.extras.set_many({"supports_spectral_bond_cap":kind == "segment",
        "engine_manifest_sha256":manifest_hash,"hidden":False})
    report["codes"][kind] = {"pk":code.pk,"uuid":str(code.uuid),"label":code.full_label}
print(json.dumps(report,indent=2))
