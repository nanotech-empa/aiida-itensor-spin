"""Diagnostics only: reconstruct saved moments without changing AiiDA nodes.

Native Jackson curves preserve the deployed engine's exact damping indexing.
Fixed angular Gaussian curves compare orders at the same damping, separately
from Jackson order comparisons, which also change broadening. The stated
Gaussian sigma is approximate in energy and applies at 100 meV; it is smaller
at lower energies. These are not completed production results.
"""
from pathlib import Path
import csv
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent
BANDWIDTH = 4422.5335168197
WP = 1 - 0.025 / 2
A = BANDWIDTH / (2 * WP)
ENERGY = np.linspace(0, 100, 1001)
X = ENERGY / A - WP
MAX_MOMENTS = 1277
POLYNOMIALS = np.cos(np.outer(np.arccos(X), np.arange(MAX_MOMENTS)))
PREFACTOR = 1 / (A * np.pi * np.sqrt(1 - X * X))

groups = {}
with (ROOT / "partial_moments.tsv").open() as handle:
    for row in csv.DictReader(handle, delimiter="\t"):
        task = int(row["task"])
        entry = groups.setdefault(task, {
            "task": task, "state": int(row["state"]), "site": int(row["site"]),
            "channel": row["channel"], "completed_vectors": int(row["completed_vectors"]),
            "target_vectors": int(row["target_vectors"]), "moments": [],
        })
        assert int(row["order"]) == len(entry["moments"])
        entry["moments"].append(float(row["moment"]))
assert len(groups) == 18

def reconstruct(mu, count, mode="jackson", sigma_at_100=3.0):
    order = np.arange(count)
    if mode == "jackson":
        n = order + 1  # Exact deployed indexing, intentionally preserved.
        angle = np.pi / (count + 1)
        damping = ((count - n + 1) * np.cos(angle*n)
                   + np.sin(angle*n) / np.tan(angle)) / (count + 1)
    else:
        sigma_theta = sigma_at_100 / (A * np.sqrt(1-X[-1]**2))
        damping = np.exp(-0.5 * (sigma_theta * order)**2)
    coeff = mu[:count] * damping
    coeff[1:] *= 2
    return (POLYNOMIALS[:, :count] @ coeff) * PREFACTOR

def relative_change(old, new):
    return float(np.linalg.norm(old-new) / max(np.linalg.norm(new), 1e-300))

fig, axes = plt.subplots(6, 3, figsize=(13, 17), sharex=True)
metrics = []
curves = {}
for task, entry in sorted(groups.items()):
    mu = np.asarray(entry["moments"])
    count = len(mu)
    assert count == 2*entry["completed_vectors"] - 1
    counts = [2*int(entry["completed_vectors"]*f)-1 for f in (0.6, 0.8, 1.0)]
    jackson = [reconstruct(mu, n) for n in counts]
    row = 2*(entry["state"]-1)+(entry["site"]-51)
    col = ("zz", "pm", "mp").index(entry["channel"])
    ax = axes[row, col]
    for level, y in zip(("60%", "80%", "100% saved"), jackson):
        ax.plot(ENERGY, y, label=level, linewidth=1)
    ax.set_title(f"state {entry['state']}, site {entry['site']}, {entry['channel']} — task {task}", fontsize=9)
    ax.grid(alpha=0.2)
    curves[f"task_{task}_jackson_latest"] = jackson[-1]
    measure = {k: v for k, v in entry.items() if k != "moments"}
    measure.update({
        "valid_moments": count,
        "jackson_80pct_to_latest_relative_L2": relative_change(jackson[1], jackson[2]),
        "jackson_latest_peak_energy_meV": float(ENERGY[np.argmax(jackson[2])]),
        "jackson_latest_approx_sigma_at_100_meV": float(np.pi*A*np.sqrt(1-X[-1]**2)/count),
        "jackson_latest_approx_sigma_at_0_meV": float(np.pi*A*np.sqrt(1-X[0]**2)/count),
    })
    for sigma in (1.5, 3.0, 5.0):
        series = [reconstruct(mu, n, "gaussian", sigma) for n in counts]
        sigma_theta = sigma / (A*np.sqrt(1-X[-1]**2))
        label = f"angular_gaussian_sigma100_{sigma:g}"
        measure[label] = {
            "80pct_to_latest_relative_L2": relative_change(series[1], series[2]),
            "60pct_to_latest_relative_L2": relative_change(series[0], series[2]),
            "last_available_order_damping": float(np.exp(-0.5*(sigma_theta*(count-1))**2)),
            "min_spectral_value": float(np.min(series[-1])),
            "peak_energy_meV": float(ENERGY[np.argmax(series[-1])]),
        }
        curves[f"task_{task}_{label}_latest"] = series[-1]
    metrics.append(measure)

axes[0, 0].legend(fontsize=8)
for ax in axes[-1, :]:
    ax.set_xlabel("Energy (meV)")
for ax in axes[:, 0]:
    ax.set_ylabel("Partial task spectrum")
fig.suptitle("L101 unfinished tasks: saved-moment diagnostics\nJackson curves change broadening with expansion order; these are not final production spectra")
fig.tight_layout(rect=(0, 0, 1, 0.96))
fig.savefig(ROOT / "partial_spectra_order_sensitivity.png", dpi=150)
plt.close(fig)
np.savez_compressed(ROOT / "partial_spectra_diagnostics.npz", energy_meV=ENERGY, **curves)
report = {
    "checkpoint_segment_uuid": "4b4117de-7c29-4a5b-a1fe-78f438c854eb",
    "scope": "Only the 18 unfinished tasks; incomplete diagnostic spectra, not a finished L101 result.",
    "bandwidth_meV": BANDWIDTH,
    "caveats": [
        "Order stability alone does not prove truncation accuracy or convergence to the final spectrum.",
        "Native Jackson order comparisons include changes in kernel broadening.",
        "Gaussian diagnostics use fixed damping in Chebyshev angle, not uniform energy broadening.",
        "Gaussian sigma labels apply at 100 meV and are smaller at lower energies.",
        "No post-hoc scientific acceptance tolerance is assigned.",
    ],
    "tasks": metrics,
}
(ROOT / "partial_spectra_report.json").write_text(json.dumps(report, indent=2)+"\n")
for sigma in (1.5, 3.0, 5.0):
    name = f"angular_gaussian_sigma100_{sigma:g}"
    values = [m[name]["80pct_to_latest_relative_L2"] for m in metrics]
    print(f"Fixed angular Gaussian sigma(100 meV)={sigma:g}: median order sensitivity={np.median(values):.6g}, max={max(values):.6g}")
print("Jackson sigma at 100 meV range:", min(m["jackson_latest_approx_sigma_at_100_meV"] for m in metrics), max(m["jackson_latest_approx_sigma_at_100_meV"] for m in metrics))
print("Report:", ROOT / "partial_spectra_report.json")
