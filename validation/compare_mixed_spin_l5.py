#!/usr/bin/env python3
"""Validate two five-site mixed-spin chains against dense ED."""

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

SITE_SPINS = [0.5, 1.0, 2.0, 1.0, 0.5]
TARGET_TWICE_SZ = 0
NOISE_SEED = 20260919
NUMBER_OF_LOW_ENERGIES = 10


def cases() -> tuple[dict, dict]:
    rng = np.random.default_rng(NOISE_SEED)
    noisy = (10.0 + rng.uniform(-1.0, 1.0, 4)).tolist()
    return (
        {
            "name": "L5_uniform_J10",
            "couplings_meV": [10.0, 10.0, 10.0, 10.0],
            "noise": None,
        },
        {
            "name": "L5_J10_uniform_noise_10pct",
            "couplings_meV": noisy,
            "noise": {
                "distribution": "independent uniform",
                "half_width_meV": 1.0,
                "relative_half_width": 0.1,
                "seed": NOISE_SEED,
            },
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


def _validate(case: dict, run_directory: Path) -> dict:
    bonds = [
        (site, site + 1, coupling)
        for site, coupling in enumerate(case["couplings_meV"], start=1)
    ]
    exact = diagonalize(SITE_SPINS, bonds, total_twice_sz=TARGET_TWICE_SZ)
    exact_bandwidth = float(exact.energies_meV[-1] - exact.energies_meV[0])
    compared = min(NUMBER_OF_LOW_ENERGIES, len(exact.energies_meV))
    parameters = {
        "model": "mixed_spin_heisenberg_chain_v1",
        "length": len(SITE_SPINS),
        "site_spins": SITE_SPINS,
        "couplings_meV": case["couplings_meV"],
        "target_twice_sz": TARGET_TWICE_SZ,
        "seed": 20260724,
        "threading_mode": "blocksparse",
        "dmrg_cutoff": 1e-12,
        "spectral_cutoff": 1e-10,
        "energy_precision_meV": 1e-10,
        "highest_energy_precision_meV": 1e-10,
        "initial_linkdim": 64,
        "excited_states": compared - 1,
        "excited_state_penalty_meV": 1000.0,
        "temperature_K": 0.0,
        "symmetry_tolerance": 1e-8,
        "omega_max_meV": exact_bandwidth,
        "omega_smearing_meV": 0.5,
        "try_doubleN": True,
    }
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

    summary = (run_directory / "gs_summary.txt").read_text()
    tensor_energies = _list_after(summary, "# List of energies:")
    tensor_emax = _float_after(summary, "# Energy of highest-energy state:")
    exact_low = exact.energies_meV[:compared]
    errors = tensor_energies - exact_low
    emax_error = tensor_emax - exact.energies_meV[-1]
    tolerance = 2e-7

    np.testing.assert_allclose(tensor_energies, exact_low, atol=tolerance, rtol=0)
    np.testing.assert_allclose(
        tensor_emax,
        exact.energies_meV[-1],
        atol=tolerance,
        rtol=0,
    )
    return {
        **case,
        "site_spins": SITE_SPINS,
        "target_twice_sz": TARGET_TWICE_SZ,
        "full_hilbert_dimension": 180,
        "sector_dimension": len(exact.energies_meV),
        "compared_low_energy_states": compared,
        "tensor_low_energies_meV": tensor_energies.tolist(),
        "exact_low_energies_meV": exact_low.tolist(),
        "low_energy_errors_meV": errors.tolist(),
        "maximum_absolute_low_energy_error_meV": float(np.max(np.abs(errors))),
        "tensor_highest_energy_meV": tensor_emax,
        "exact_highest_energy_meV": float(exact.energies_meV[-1]),
        "highest_energy_error_meV": float(emax_error),
        "exact_bandwidth_meV": exact_bandwidth,
        "elapsed_seconds": elapsed,
        "tolerance_meV": tolerance,
        "passed": True,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path)
    parser.add_argument("--keep-run-directory", type=Path)
    args = parser.parse_args()

    temporary = (
        tempfile.TemporaryDirectory(prefix="itensor-mixed-l5-ed-")
        if args.keep_run_directory is None
        else None
    )
    root = Path(temporary.name) if temporary is not None else args.keep_run_directory.resolve()
    root.mkdir(parents=True, exist_ok=True)

    results = [_validate(case, root / case["name"]) for case in cases()]
    report = {
        "suite": "mixed_spin_heisenberg_L5_exact_diagonalization",
        "model": "mixed_spin_heisenberg_chain_v1",
        "validation_scope": (
            "lowest ten energies and the highest energy in the conserved 2Sz=0 sector"
        ),
        "cases": results,
        "passed": all(result["passed"] for result in results),
    }
    rendered = json.dumps(report, indent=2, sort_keys=True) + "\n"
    print(rendered, end="")
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
