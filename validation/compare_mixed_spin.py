#!/usr/bin/env python3
"""Validate the mixed-spin tensor backend against independent dense ED."""

from __future__ import annotations

import argparse
import ast
import json
import os
import re
import subprocess
import tempfile
import time
from pathlib import Path

import numpy as np
from exact_diagonalization import diagonalize

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_WRAPPER = ROOT / "remote" / "itensor-spin-gs-local"
DEFAULT_ENGINE = ROOT / "remote" / "engine"
DEFAULT_JULIA = Path("/home/jovyan/opt/julia-1.12.6/bin/julia")
DEFAULT_DEPOT = Path("/home/jovyan/.julia-itensor-spin-localhost")

CASES = (
    {
        "name": "three_spin_half_equal_antiferromagnetic",
        "site_spins": [0.5, 0.5, 0.5],
        "couplings_meV": [1.0, 1.0],
        "target_twice_sz": 1,
    },
    {
        "name": "smallest_goncalo_equivalent",
        "site_spins": [1.0, 0.5, 1.0],
        "couplings_meV": [16.7, 51.1],
        "target_twice_sz": 1,
    },
    {
        "name": "mixed_half_one_two_competing_couplings",
        "site_spins": [0.5, 1.0, 2.0],
        "couplings_meV": [0.35, -0.6],
        "target_twice_sz": 1,
    },
)


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


def _parameters(case: dict, excited_states: int) -> dict:
    return {
        "model": "mixed_spin_heisenberg_chain_v1",
        "length": len(case["site_spins"]),
        "site_spins": case["site_spins"],
        "couplings_meV": case["couplings_meV"],
        "target_twice_sz": case["target_twice_sz"],
        "seed": 20260724,
        "threading_mode": "blocksparse",
        "dmrg_cutoff": 1e-12,
        "spectral_cutoff": 1e-10,
        "energy_precision_meV": 1e-10,
        "highest_energy_precision_meV": 1e-10,
        "initial_linkdim": 32,
        "excited_states": excited_states,
        "excited_state_penalty_meV": 1000.0,
        "temperature_K": 0.0,
        "symmetry_tolerance": 1e-8,
        "omega_max_meV": 100.0,
        "omega_smearing_meV": 1.5,
        "try_doubleN": True,
    }


def _validate_case(case: dict, run_directory: Path) -> dict:
    bonds = [
        (index, index + 1, coupling)
        for index, coupling in enumerate(case["couplings_meV"], start=1)
    ]
    exact = diagonalize(
        case["site_spins"],
        bonds,
        total_twice_sz=case["target_twice_sz"],
    )
    parameters = _parameters(case, len(exact.energies_meV) - 1)
    config = {
        "format_version": 1,
        "parameters": parameters,
        "execution": {"threads_per_process": 2},
    }
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
    started = time.time()
    subprocess.run(
        [str(DEFAULT_WRAPPER), str(config_path)],
        cwd=run_directory,
        env=environment,
        check=True,
    )
    elapsed = time.time() - started

    summary_text = (run_directory / "gs_summary.txt").read_text()
    tensor_energies = _list_after(summary_text, "# List of energies:")
    tensor_emax = _float_after(summary_text, "# Energy of highest-energy state:")
    energy_errors = tensor_energies - exact.energies_meV
    emax_error = tensor_emax - exact.energies_meV[-1]

    np.testing.assert_allclose(
        tensor_energies,
        exact.energies_meV,
        atol=2e-8,
        rtol=0,
    )
    np.testing.assert_allclose(
        tensor_emax,
        exact.energies_meV[-1],
        atol=2e-8,
        rtol=0,
    )
    return {
        **case,
        "sector_dimension": len(exact.energies_meV),
        "tensor_energies_meV": tensor_energies.tolist(),
        "exact_energies_meV": exact.energies_meV.tolist(),
        "energy_errors_meV": energy_errors.tolist(),
        "tensor_highest_energy_meV": tensor_emax,
        "exact_highest_energy_meV": float(exact.energies_meV[-1]),
        "highest_energy_error_meV": float(emax_error),
        "maximum_absolute_energy_error_meV": float(np.max(np.abs(energy_errors))),
        "elapsed_seconds": elapsed,
        "tolerance_meV": 2e-8,
        "passed": True,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path)
    parser.add_argument("--keep-run-directory", type=Path)
    args = parser.parse_args()

    temporary = (
        tempfile.TemporaryDirectory(prefix="itensor-mixed-ed-")
        if args.keep_run_directory is None
        else None
    )
    root = Path(temporary.name) if temporary is not None else args.keep_run_directory.resolve()
    root.mkdir(parents=True, exist_ok=True)

    results = [
        _validate_case(case, root / case["name"])
        for case in CASES
    ]
    goncalo = next(
        result for result in results if result["name"] == "smallest_goncalo_equivalent"
    )
    report = {
        "suite": "mixed_spin_heisenberg_three_site_exact_diagonalization",
        "model": "mixed_spin_heisenberg_chain_v1",
        "independent_oracle": "NumPy dense exact diagonalization",
        "cases": results,
        "goncalo_legacy_reference_energies_meV": [
            -55.92447882910047,
            -46.59763902910314,
            12.697639029103145,
            22.024478829100463,
            33.9,
        ],
        "goncalo_generic_matches_legacy": bool(
            np.allclose(
                goncalo["tensor_energies_meV"],
                [
                    -55.92447882910047,
                    -46.59763902910314,
                    12.697639029103145,
                    22.024478829100463,
                    33.9,
                ],
                atol=2e-8,
                rtol=0,
            )
        ),
        "passed": all(result["passed"] for result in results),
    }
    if not report["goncalo_generic_matches_legacy"]:
        raise AssertionError("generic Gonçalo-equivalent case differs from the legacy tensor result")

    rendered = json.dumps(report, indent=2, sort_keys=True) + "\n"
    print(rendered, end="")
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
