"""Provisional whole-chain spectra from immutable saved moments.

Use fixed angular Gaussian damping, sigma_E(100 meV)=3 meV, consistent
across all 909 tasks. This is a reconstruction, not a new recurrence run.
Initial states remain separate; no thermal average is implied.
"""
from pathlib import Path
import json

import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import numpy as np
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "l101_3mev_provisional"
OUT.mkdir(exist_ok=True)
with h5py.File(ROOT / "all_moments.jld2", "r") as f:
    moments = np.asarray(f["moments"], dtype=float)
    completed = np.asarray(f["completed_vectors"], dtype=int)
    states = np.asarray(f["state_indices"], dtype=int)
    sites = np.asarray(f["site_indices"], dtype=int)
    channels = np.asarray(f["channel_indices"], dtype=int)
    final_exists = np.asarray(f["final_task_exists"], dtype=int)
    state_energies = np.asarray(f["state_energies_meV"], dtype=float)
    emax = float(f["highest_energy_meV"][()])
    temperature = float(f["temperature_K"][()])
assert moments.shape == (909, 2947)  # JLD2/HDF5 dimension order
assert np.array_equal(states, np.repeat(np.arange(1, 4), 303))
assert np.array_equal(sites, np.tile(np.repeat(np.arange(1, 102), 3), 3))
assert np.array_equal(channels, np.tile(np.arange(1, 4), 303))
assert np.sum(completed) == 1322972
assert np.sum(final_exists) == 891
assert np.sum(completed < 1474) == 18
assert np.all(np.isfinite(moments))
valid_counts = 2*completed - 1
for task in range(909):
    assert np.all(moments[task, valid_counts[task]:] == 0)

bandwidth = emax - np.min(state_energies)
wp = 1 - 0.025 / 2
a = bandwidth / (2*wp)
energy = np.linspace(0, 100, 1001)
x = energy/a - wp
order = np.arange(2947)
polynomials = np.cos(np.outer(np.arccos(x), order))
prefactor = 1 / (a*np.pi*np.sqrt(1-x*x))

with h5py.File(ROOT / "reference_checks.jld2", "r") as f:
    cutoffs = np.asarray(f["cutoffs"], dtype=float)
    reference_ids = np.asarray(f["reference_task_ids"], dtype=int)
    reference_energy = np.asarray(f["reference_omega_meV"], dtype=float)
    reference_spectra = np.asarray(f["reference_spectra"], dtype=float)
rx = reference_energy/a-wp
rp = np.cos(np.outer(np.arccos(rx), order))
n = order+1  # Preserve deployed Jackson indexing exactly for this check.
angle = np.pi/2948
jackson = ((2947-n+1)*np.cos(angle*n)+np.sin(angle*n)/np.tan(angle))/2948
coeff = moments[reference_ids-1]*jackson
coeff[:, 1:] *= 2
with threadpool_limits(limits=2):
    reconstructed_refs = (coeff @ rp.T)/(a*np.pi*np.sqrt(1-rx*rx))
np.testing.assert_allclose(reconstructed_refs, reference_spectra, atol=1e-9, rtol=1e-8)
reference_error = float(np.max(np.abs(reconstructed_refs-reference_spectra)))

def reconstruct(sigma100, fraction=1.0):
    sigma_angle = sigma100/(a*np.sqrt(1-x[-1]**2))
    damping = np.exp(-0.5*(sigma_angle*order)**2)
    coeff = moments*damping
    coeff[:, 1:] *= 2
    if fraction < 1:
        counts = 2*np.floor(completed*fraction).astype(int)-1
        coeff[order[None, :] >= counts[:, None]] = 0
    with threadpool_limits(limits=2):
        return (coeff @ polynomials.T)*prefactor

def combine(task_spectra):
    raw = task_spectra.reshape(3, 101, 3, len(energy))
    return raw[:, :, 0]+0.5*raw[:, :, 1]+0.5*raw[:, :, 2]

task3 = reconstruct(3)
task3_80 = reconstruct(3, 0.8)
task15 = reconstruct(1.5)
chi3, chi3_80, chi15 = map(combine, (task3, task3_80, task15))
assert np.all(np.isfinite(chi3))
task_changes = np.linalg.norm(task3-task3_80, axis=1)/np.maximum(np.linalg.norm(task3, axis=1), 1e-300)
site_changes = np.linalg.norm(chi3-chi3_80, axis=2)/np.maximum(np.linalg.norm(chi3, axis=2), 1e-300)
finer_changes = np.linalg.norm(chi3-chi15, axis=2)/np.maximum(np.linalg.norm(chi15, axis=2), 1e-300)

np.savez_compressed(OUT / "spectra.npz", energy_meV=energy, initial_state_energies_meV=state_energies,
    chi_3mev=chi3, chi_3mev_80pct=chi3_80, chi_1p5mev_diagnostic=chi15,
    task_spectra_3mev=task3, completed_vectors=completed, valid_moment_counts=valid_counts,
    state_indices=states, site_indices=sites, channel_indices=channels,
    sigma_energy_meV=3*np.sqrt(1-x*x)/np.sqrt(1-x[-1]**2))
with (OUT / "spectrum.tsv").open("w") as handle:
    handle.write("state\tsite\tomega_meV\tchi\n")
    for state in range(3):
        for site in range(101):
            np.savetxt(handle, np.column_stack((
                np.full(len(energy), state+1), np.full(len(energy), site+1), energy, chi3[state, site]
            )), fmt=["%d", "%d", "%.8g", "%.12g"], delimiter="\t")

positive = chi3[chi3 > 0]
vmax = float(np.max(positive))
vmin = max(float(np.quantile(positive, 0.05)), vmax*1e-5)
fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=True, constrained_layout=True)
for state, ax in enumerate(axes):
    im = ax.pcolormesh(np.arange(1, 102), energy, np.ma.masked_less_equal(chi3[state].T, 0),
        shading="auto", cmap="magma", norm=LogNorm(vmin=vmin, vmax=vmax))
    ax.axvline(51, color="cyan", lw=0.8, ls="--")
    ax.set_title(f"Initial state {state+1}")
    ax.set_xlabel("Site")
axes[0].set_ylabel("Energy (meV)")
fig.colorbar(im, ax=axes, label="Local spectrum (log colour scale)")
fig.suptitle("L101 provisional spectrum from saved moments\nGaussian width ≈3 meV at 100 meV; initial states shown separately")
fig.savefig(OUT / "whole_chain.png", dpi=170)
fig.savefig(OUT / "whole_chain.pdf")
plt.close(fig)

fig, axes = plt.subplots(3, 2, figsize=(12, 10), sharex=True, constrained_layout=True)
for state in range(3):
    for col, site in enumerate((51, 52)):
        ax = axes[state, col]
        ax.plot(energy, chi15[state, site-1], color="0.6", lw=1, label="1.5 meV diagnostic")
        ax.plot(energy, chi3_80[state, site-1], ls="--", lw=1, label="3 meV, 80% saved order")
        ax.plot(energy, chi3[state, site-1], lw=1.4, label="3 meV, all saved moments")
        ax.set_title(f"State {state+1}, site {site}")
        ax.set_ylabel("Local spectrum")
        ax.grid(alpha=0.2)
for ax in axes[-1]:
    ax.set_xlabel("Energy (meV)")
axes[0, 0].legend(fontsize=8)
fig.suptitle("L101 central sites: provisional reconstruction\nWidths are approximate Gaussian sigma at 100 meV, smaller at lower energies")
fig.savefig(OUT / "central_sites_comparison.png", dpi=170)
fig.savefig(OUT / "central_sites_comparison.pdf")
plt.close(fig)

report = {
    "status": "provisional saved-moment reconstruction; production L101 remains paused",
    "source_segment_uuid": "4b4117de-7c29-4a5b-a1fe-78f438c854eb",
    "source_ground_state_uuid": "9d941100-9de6-4492-8da5-5d6cd168359c",
    "source_remote_data_uuid": "9058e3b7-9381-4cb4-979d-1b9f6aefc819",
    "task_count": 909, "completed_recurrence_tasks": 891, "partial_recurrence_tasks": 18,
    "states": 3, "sites": 101, "channel_sum": "zz + 0.5*pm + 0.5*mp",
    "thermal_average": False, "saved_temperature_K": temperature,
    "state_energies_meV": state_energies.tolist(), "bandwidth_meV": float(bandwidth),
    "spectral_cutoffs_in_preserved_checkpoints": sorted(np.unique(cutoffs).tolist()),
    "kernel": "Gaussian damping in Chebyshev angle, exp(-0.5*(sigma_angle*n)^2)",
    "sigma_energy_at_100_meV": 3.0,
    "sigma_energy_at_0_meV": float(3*np.sqrt(1-x[0]**2)/np.sqrt(1-x[-1]**2)),
    "not_uniform_energy_broadening": True,
    "gaussian_sigma_is_not_FWHM": True,
    "new_recurrence_iterations": 0,
    "stored_completed_spectra_validation_max_abs_difference": reference_error,
    "stored_spectra_validation_tasks": reference_ids.tolist(),
    "all_909_checkpoint_ground_state_hashes_verified": True,
    "max_task_relative_L2_change_80pct_to_all_saved": float(np.max(task_changes)),
    "max_combined_site_relative_L2_change_80pct_to_all_saved": float(np.max(site_changes)),
    "median_combined_site_relative_L2_change_80pct_to_all_saved": float(np.median(site_changes)),
    "max_combined_site_relative_L2_difference_from_finer_diagnostic": float(np.max(finer_changes)),
    "minimum_combined_spectrum": float(np.min(chi3)),
    "negative_combined_sample_count": int(np.sum(chi3 < 0)),
    "caveats": [
        "Order sensitivity is not a final error bound or proof of scientific convergence.",
        "891 recurrence tasks finished; 18 tasks remain incomplete under their original 1474-vector target.",
        "The 3 meV Gaussian reconstruction differs from rerunning the original Jackson algorithm with omega_smearing_meV=3.",
        "Widths vary with energy and are Gaussian standard deviations, not FWHM.",
        "Three initial states are preserved separately; this is not a thermal average.",
        "Negative values are retained in exported data and masked only in the logarithmic heatmap.",
    ],
}
(OUT / "report.json").write_text(json.dumps(report, indent=2)+"\n")
print(json.dumps({key: report[key] for key in (
    "stored_completed_spectra_validation_max_abs_difference",
    "spectral_cutoffs_in_preserved_checkpoints", "sigma_energy_at_0_meV",
    "max_task_relative_L2_change_80pct_to_all_saved",
    "max_combined_site_relative_L2_change_80pct_to_all_saved",
    "minimum_combined_spectrum", "negative_combined_sample_count")}, indent=2))
print("Saved", OUT)
