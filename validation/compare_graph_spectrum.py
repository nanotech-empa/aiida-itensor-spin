"""Isolated GS-to-collected-spectrum graph check; never uses AiiDA/daemon/Daint."""

import csv
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
from exact_diagonalization import heisenberg_hamiltonian, local_transition_weights

from aiida_itensor_spin.contracts import normalize_ground_state_parameters

root = Path(__file__).resolve().parents[1]
out = Path(sys.argv[1]).resolve()
out.mkdir(parents=True, exist_ok=True)
spins = [0.5] * 4
bonds = [(1, 2, 1.0), (2, 3, 0.7), (2, 4, 0.8), (1, 3, 1.2)]
h, _ = heisenberg_hamiltonian(spins, bonds)
energies = np.linalg.eigvalsh(h)
bandwidth = float(np.ptp(energies))
assert np.count_nonzero(np.abs(energies - energies[0]) < 1e-8) == 1
parameters = normalize_ground_state_parameters(
    {
        "model": "mixed_spin_heisenberg_graph_v1",
        "seed": 20261021,
        "site_spins": spins,
        "exchange_edges": [{"i": i, "j": j, "J_meV": J} for i, j, J in bonds],
        "target_twice_sz": 0,
        "dmrg_cutoff": 1e-12,
        "spectral_cutoff": 1e-12,
        "energy_precision_meV": 1e-10,
        "highest_energy_precision_meV": 1e-10,
        "initial_linkdim": 4,
        "excited_states": 0,
        "temperature_K": 0,
        "omega_max_meV": bandwidth * 0.95,
        "omega_smearing_meV": bandwidth / 12,
        "try_doubleN": True,
    }
)
(out / "ground_state.json").write_text(
    json.dumps(
        {"format_version": 1, "parameters": parameters, "execution": {"threads_per_process": 1}},
        indent=2,
    )
)
env = {
    **os.environ,
    "ITENSOR_SPIN_CASE_ROOT": str(root / "remote/engine"),
    "ITENSOR_SPIN_JULIA": "/home/jovyan/opt/julia-1.12.6/bin/julia",
    "JULIA_DEPOT_PATH": str(out.parent / "julia-depot")
    + ":/home/jovyan/.julia-itensor-spin-localhost",
    "OPENBLAS_NUM_THREADS": "1",
    "OMP_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "GKSwstype": "100",
}
reuse = Path(sys.argv[2]).resolve() if len(sys.argv) > 2 else None
if reuse is not None:
    assert json.loads((reuse / "ground_state.json").read_text())["parameters"] == parameters
    for name in ["gs_checkpoint.jld2", "gs_status.json", "gs_summary.txt"]:
        shutil.copy2(reuse / name, out / name)
else:
    subprocess.run(
        [str(root / "remote/itensor-spin-gs-local"), str(out / "ground_state.json")],
        env=env,
        cwd=out,
        check=True,
    )

status = json.loads((out / "gs_status.json").read_text())
np.testing.assert_allclose(status["ground_energy_meV"], energies[0], atol=1e-8, rtol=0)
np.testing.assert_allclose(status["highest_energy_meV"], energies[-1], atol=1e-8, rtol=0)
config = {
    "format_version": 1,
    "segment_index": 0,
    "parameters": {
        "chebyshev_convention": "chebyshev-2n-minus-1-v2",
        "backend": "streaming",
        "threading_mode": "blocksparse",
        "ground_state_filename": "gs_checkpoint.jld2",
        "checkpoint_every": 1,
        "heartbeat_every": 1,
        "spectral_cutoff": 1e-12,
        "max_bond_dimension": 0,
        "expected_total_tasks": 12,
    },
    "layout": {
        "nodes": 1,
        "processes_per_node": 1,
        "threads_per_process": 1,
        "wallclock_seconds": 1200,
        "internal_stop_seconds": 1140,
        "partition": "local",
    },
    "workers": 1,
    "ground_state": "gs_checkpoint.jld2",
    "resume_state": str(reuse / "state") if reuse is not None else None,
    "output_state": "state",
    "status_file": "segment_status.json",
    "summary_file": "segment_summary.json",
}
(out / "segment.json").write_text(json.dumps(config, indent=2))
with (out / "segment.stdout").open("w") as stdout, (out / "segment.stderr").open("w") as stderr:
    subprocess.run(
        [str(root / "remote/itensor-spin-segment-local"), str(out / "segment.json")],
        env=env,
        cwd=out,
        stdout=stdout,
        stderr=stderr,
        check=True,
    )
segment = json.loads((out / "segment_status.json").read_text())
assert segment["state"] == "complete" and segment["completed_tasks"] == 12
poles = local_transition_weights(spins, bonds, total_twice_sz=0)
# Compare with the same explicitly documented inherited Chebyshev/Jackson
# convention; this tests graph integration, not a new damping convention.
N = round(np.ceil(status["spectral_bandwidth_meV"] / parameters["omega_smearing_meV"]) / 2)
neff = 2 * N - 1
n = np.arange(1, neff + 1)
damping = (
    (neff - n + 1) * np.cos(np.pi * n / (neff + 1))
    + np.sin(np.pi * n / (neff + 1)) / np.tan(np.pi / (neff + 1))
) / (neff + 1)
a = status["spectral_bandwidth_meV"] / (2 * (1 - 0.025 / 2))
wp = 1 - 0.025 / 2
rows = list(csv.DictReader((out / "compact_result/spectrum.tsv").open(), delimiter="\t"))
errors = []
for site in range(1, 5):
    data = [r for r in rows if int(r["site"]) == site]
    omega = np.array([float(r["omega_meV"]) for r in data])
    observed = np.array([float(r["chi"]) for r in data])
    expected = np.zeros_like(omega)
    for channel, factor in [("zz", 1), ("pm", 0.5), ("mp", 0.5)]:
        energy, weight = poles[(site, channel)]
        moments = np.polynomial.chebyshev.chebvander(energy / a - wp, neff - 1).T @ weight
        coefficients = damping * moments
        coefficients[1:] *= 2
        x = omega / a - wp
        expected += (
            factor
            * np.polynomial.chebyshev.chebval(x, coefficients)
            / (a * np.pi * np.sqrt(1 - x * x))
        )
    np.testing.assert_allclose(observed, expected, atol=2e-8, rtol=2e-8)
    errors.append(float(np.max(np.abs(observed - expected))))
report = {
    "ground_energy_abs_error_meV": abs(status["ground_energy_meV"] - energies[0]),
    "highest_energy_abs_error_meV": abs(status["highest_energy_meV"] - energies[-1]),
    "maximum_spectrum_abs_error": max(errors),
    "completed_tasks": segment["completed_tasks"],
    "comparison": "independent ED poles reconstructed with inherited Chebyshev/Jackson convention",
    "production_touched": False,
}
(out / "comparison.json").write_text(json.dumps(report, indent=2) + "\n")
print(json.dumps(report, indent=2))
