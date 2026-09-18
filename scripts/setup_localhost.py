"""Idempotently register the localhost ITensor Codes for this checkout."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import shlex
import subprocess
from pathlib import Path

from aiida import orm
from aiida.common.exceptions import NotExistent
from aiida.manage.configuration import load_profile

CODE_VERSION = "0.4.0"
COMPUTER_LABEL = "localhost"


def digest_files(paths: list[Path]) -> str:
    digest = hashlib.sha256()
    for path in sorted(paths):
        digest.update(path.name.encode())
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def validate_runtime(root: Path, julia: Path, depot: Path) -> dict[str, str]:
    if not julia.is_file():
        raise RuntimeError(f"Julia executable does not exist: {julia}")
    if not os.access(julia, os.X_OK):
        raise RuntimeError(f"Julia executable is not executable: {julia}")
    project = root / "remote" / "engine" / "ITensor_env"
    environment = os.environ.copy()
    environment.update(
        {
            "JULIA_DEPOT_PATH": str(depot),
            "GKSwstype": "100",
        }
    )
    command = [
        str(julia),
        "--startup-file=no",
        f"--project={project}",
        "-e",
        (
            "using ITensors, ITensorMPS, JLD2; "
            'print("Julia=", VERSION, " ITensors=", pkgversion(ITensors), '
            '" ITensorMPS=", pkgversion(ITensorMPS), " JLD2=", pkgversion(JLD2))'
        ),
    ]
    result = subprocess.run(
        command,
        check=True,
        capture_output=True,
        text=True,
        env=environment,
    )
    return {
        "architecture": platform.machine(),
        "runtime": result.stdout.strip(),
        "julia": str(julia),
        "depot": str(depot),
        "project": str(project),
    }


def expected_code(root: Path, julia: Path, depot: Path, kind: str) -> dict[str, object]:
    if kind == "gs":
        executable = root / "remote" / "itensor-spin-gs-local"
        plugin = "itensor_spin.ground_state"
    elif kind == "segment":
        executable = root / "remote" / "itensor-spin-segment-local"
        plugin = "itensor_spin.spectral_segment"
    else:
        raise ValueError(kind)
    prepend = "\n".join(
        [
            f"export ITENSOR_SPIN_CASE_ROOT={shlex.quote(str(root / 'remote' / 'engine'))}",
            f"export ITENSOR_SPIN_JULIA={shlex.quote(str(julia))}",
            f"export JULIA_DEPOT_PATH={shlex.quote(str(depot))}",
            "export GKSwstype=100",
            "export OMP_NUM_THREADS=1",
            "export OPENBLAS_NUM_THREADS=1",
            "export MKL_NUM_THREADS=1",
        ]
    )
    return {
        "label": f"itensor-spin-{kind}-{CODE_VERSION}-local",
        "executable": executable,
        "plugin": plugin,
        "prepend": prepend,
    }


def register_code(computer, expected: dict[str, object], provenance: dict[str, object]):
    full_label = f"{expected['label']}@{computer.label}"
    try:
        code = orm.load_code(full_label)
    except NotExistent:
        code = orm.InstalledCode(
            computer=computer,
            label=str(expected["label"]),
            filepath_executable=str(expected["executable"]),
            default_calc_job_plugin=str(expected["plugin"]),
            with_mpi=False,
            prepend_text=str(expected["prepend"]),
            description=(
                "Local direct-scheduler Julia/ITensor spin-chain engine. "
                "Versioned model parameters and numerical controls are stored in AiiDA."
            ),
        ).store()
        code.base.extras.set_many(provenance)
        return code, "created"

    mismatches = []
    if Path(code.filepath_executable) != expected["executable"]:
        mismatches.append("executable")
    if code.default_calc_job_plugin != expected["plugin"]:
        mismatches.append("default_calc_job_plugin")
    if code.prepend_text != expected["prepend"]:
        mismatches.append("prepend_text")
    if code.computer.uuid != computer.uuid:
        mismatches.append("computer")
    if code.base.extras.get("engine_sha256", None) != provenance["engine_sha256"]:
        mismatches.append("engine_sha256")
    if mismatches:
        raise RuntimeError(
            f"{full_label} exists with incompatible {', '.join(mismatches)}; "
            "use a new versioned label rather than modifying a provenance-linked Code"
        )
    return code, "verified"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--repository-root",
        type=Path,
        default=Path.cwd(),
    )
    parser.add_argument(
        "--julia",
        type=Path,
        default=Path("/home/jovyan/opt/julia-1.12.6/bin/julia"),
    )
    parser.add_argument(
        "--depot",
        type=Path,
        default=Path("/home/jovyan/.julia-itensor-spin-localhost"),
    )
    args = parser.parse_args()

    root = args.repository_root.resolve()
    if not (root / "pyproject.toml").is_file() or not (root / "remote" / "engine").is_dir():
        raise RuntimeError(
            f"not an aiida-itensor-spin repository root: {root}; "
            "run from the checkout root or pass --repository-root"
        )
    runtime = validate_runtime(root, args.julia.resolve(), args.depot.resolve())
    load_profile()
    computer = orm.load_computer(COMPUTER_LABEL)
    if computer.hostname != "localhost":
        raise RuntimeError("localhost computer has an unexpected hostname")
    if computer.transport_type != "core.local" or computer.scheduler_type != "core.direct":
        raise RuntimeError("localhost must use core.local transport and core.direct scheduler")

    source_paths = [
        root / "remote" / "itensor-spin-gs-local",
        root / "remote" / "itensor-spin-segment-local",
        *sorted((root / "remote" / "engine").glob("*.jl")),
        *sorted((root / "remote" / "engine" / "ITensor_libs").glob("*.jl")),
        root / "remote" / "engine" / "ITensor_env" / "Project.toml",
        root / "remote" / "engine" / "ITensor_env" / "Manifest.toml",
    ]
    provenance = {
        "engine_sha256": digest_files(source_paths),
        "engine_version": CODE_VERSION,
        "model": "spin1_chain_spinhalf_impurity_v1",
        **runtime,
    }

    report = {"computer": computer.uuid, "runtime": runtime, "codes": {}}
    for kind in ("gs", "segment"):
        expected = expected_code(root, args.julia.resolve(), args.depot.resolve(), kind)
        code, state = register_code(computer, expected, provenance)
        report["codes"][kind] = {
            "state": state,
            "label": code.full_label,
            "uuid": str(code.uuid),
            "executable": str(code.filepath_executable),
        }
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
