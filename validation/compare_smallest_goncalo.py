#!/usr/bin/env python3
"""Compare the smallest existing impurity model against dense ED."""

from __future__ import annotations

import argparse
import ast
import json
import os
import re
import subprocess
import tempfile
from pathlib import Path

import numpy as np
from exact_diagonalization import diagonalize

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_WRAPPER = ROOT / "remote" / "itensor-spin-gs-local"
DEFAULT_ENGINE = ROOT / "remote" / "engine"
DEFAULT_JULIA = Path("/home/jovyan/opt/julia-1.12.6/bin/julia")
DEFAULT_DEPOT = Path("/home/jovyan/.julia-itensor-spin-localhost")


def _list_after(text: str, heading: str) -> np.ndarray:
    match = re.search(re.escape(heading) + r"\s*\n(\[[^\n]*\])", text)
    if not match:
        raise RuntimeError(f"missing summary field: {heading}")
    return np.asarray(ast.literal_eval(match.group(1)), dtype=float)


def _float_after(text: str, heading: str) -> float:
    match = re.search(re.escape(heading) + r"\s*\n([-+0-9.eE]+)", text)
    if not match:
        raise RuntimeError(f"missing summary field: {heading}")
    return float(match.group(1))


def parameters() -> dict:
    return {
        "model": "spin1_chain_spinhalf_impurity_v1",
        "length": 3,
        "impurity_site": 2,
        "seed": 20260724,
        "threading_mode": "blocksparse",
        "J_meV": 18.4,
        "JL_meV": 16.7,
        "JR_meV": 51.1,
        "dmrg_cutoff": 1e-12,
        "spectral_cutoff": 1e-10,
        "energy_precision_meV": 1e-10,
        "highest_energy_precision_meV": 1e-10,
        "initial_linkdim": 16,
        "excited_states": 4,
        "excited_state_penalty_meV": 1000.0,
        "temperature_K": 0.0,
        "symmetry_tolerance": 1e-8,
        "omega_max_meV": 100.0,
        "omega_smearing_meV": 1.5,
        "try_doubleN": True,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path)
    parser.add_argument("--keep-run-directory", type=Path)
    args = parser.parse_args()

    config = {
        "format_version": 1,
        "parameters": parameters(),
        "execution": {"threads_per_process": 2},
    }
    temporary = (
        tempfile.TemporaryDirectory(prefix="itensor-l3-goncalo-")
        if args.keep_run_directory is None
        else None
    )
    run_directory = (
        Path(temporary.name) if temporary is not None else args.keep_run_directory.resolve()
    )
    run_directory.mkdir(parents=True, exist_ok=True)
    config_path = run_directory / "ground_state.json"
    config_path.write_text(json.dumps(config, indent=2, sort_keys=True) + "\n")

    environment = os.environ.copy()
    environment.update(
        {
            "ITENSOR_SPIN_CASE_ROOT": str(DEFAULT_ENGINE),
            "ITENSOR_SPIN_JULIA": str(DEFAULT_JULIA),
            "JULIA_DEPOT_PATH": str(DEFAULT_DEPOT),
        }
    )
    subprocess.run(
        [str(DEFAULT_WRAPPER), str(config_path)],
        cwd=run_directory,
        env=environment,
        check=True,
    )

    summary_text = (run_directory / "gs_summary.txt").read_text()
    tensor_energies = _list_after(summary_text, "# List of energies:")
    tensor_emax = _float_after(summary_text, "# Energy of highest-energy state:")

    exact = diagonalize(
        [1.0, 0.5, 1.0],
        [(1, 2, 16.7), (2, 3, 51.1)],
        total_twice_sz=1,
    )
    reference_energies = exact.energies_meV[: len(tensor_energies)]
    energy_errors = tensor_energies - reference_energies
    emax_error = tensor_emax - exact.energies_meV[-1]

    np.testing.assert_allclose(tensor_energies, reference_energies, atol=2e-8, rtol=0)
    np.testing.assert_allclose(tensor_emax, exact.energies_meV[-1], atol=2e-8, rtol=0)

    report = {
        "case": "smallest_goncalo_impurity_L3",
        "model": config["parameters"]["model"],
        "site_spins": [1.0, 0.5, 1.0],
        "bonds": [
            {"i": 1, "j": 2, "J_meV": 16.7},
            {"i": 2, "j": 3, "J_meV": 51.1},
        ],
        "total_twice_sz": 1,
        "tensor_energies_meV": tensor_energies.tolist(),
        "exact_energies_meV": reference_energies.tolist(),
        "energy_errors_meV": energy_errors.tolist(),
        "tensor_highest_energy_meV": tensor_emax,
        "exact_highest_energy_meV": float(exact.energies_meV[-1]),
        "highest_energy_error_meV": float(emax_error),
        "maximum_absolute_energy_error_meV": float(np.max(np.abs(energy_errors))),
        "tolerance_meV": 2e-8,
        "passed": True,
    }
    rendered = json.dumps(report, indent=2, sort_keys=True) + "\n"
    print(rendered, end="")
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
