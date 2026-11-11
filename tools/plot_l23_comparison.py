"""Verify and plot the retained L23 reference comparison; submits no calculations."""

from __future__ import annotations

import csv
from pathlib import Path
import tomllib

import h5py
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main():
    root = Path(__file__).resolve().parents[1]
    result_dir = root / "results" / "L23"
    reference_file = (
        root / "examples" / "l23_goncalo_20260917" / "reference" / "chi_L23.jld2"
    )
    with h5py.File(reference_file, "r") as handle:
        references = handle[handle["χ_n_i"][0]][()]
        reference = np.array([handle[ref][()] for ref in references])
    with (result_dir / "streaming" / "spectrum.tsv").open(newline="") as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    if len(rows) != 23 * 333 or {int(row["state"]) for row in rows} != {1}:
        raise ValueError("Expected one state, 23 sites and 333 frequencies.")
    omega, spectra = [], []
    for site in range(1, 24):
        site_rows = [row for row in rows if int(row["site"]) == site]
        omega.append([float(row["omega_meV"]) for row in site_rows])
        spectra.append([float(row["chi"]) for row in site_rows])
    omega, spectra = np.array(omega), np.array(spectra)
    assert reference.shape == spectra.shape == omega.shape == (23, 333)
    assert np.isfinite(reference).all() and np.isfinite(spectra).all()
    np.testing.assert_allclose(omega, np.tile(np.linspace(0, 100, 333), (23, 1)))
    delta = spectra - reference
    relative_l2 = np.linalg.norm(delta) / np.linalg.norm(reference)
    max_abs = np.max(np.abs(delta))
    with (result_dir / "streaming" / "comparison.toml").open("rb") as handle:
        recorded = tomllib.load(handle)
    # Verify reproducibility of the published metrics, not a physics tolerance.
    np.testing.assert_allclose(relative_l2, recorded["relative_L2_difference"], rtol=1e-12)
    np.testing.assert_allclose(max_abs, recorded["max_abs_difference"], rtol=1e-12)
    matching_peaks = np.count_nonzero(
        np.argmax(reference, axis=1) == np.argmax(spectra, axis=1)
    )
    assert matching_peaks == 23

    fig, axes = plt.subplots(2, 3, figsize=(12, 6.4), sharex="col", constrained_layout=True)
    for column, site in enumerate((1, 12, 23)):
        index = site - 1
        axes[0, column].plot(omega[index], reference[index], color="#263b61", label="Gonçalo reference")
        axes[0, column].plot(omega[index], spectra[index], "--", color="#df7138", label="Daint streaming")
        axes[0, column].set_title(f"Site {site}" + (" — impurity" if site == 12 else ""))
        axes[1, column].plot(omega[index], delta[index], color="#9e3a59", linewidth=1.2)
        axes[1, column].axhline(0, color="0.5", linewidth=0.6)
        axes[1, column].set_xlabel("Energy ω (meV)")
        axes[1, column].ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
        for row in range(2):
            axes[row, column].set_xlim(0, 100)
            axes[row, column].grid(alpha=0.18)
    axes[0, 0].set_ylabel("χᵢ(ω)")
    axes[1, 0].set_ylabel("Daint − reference")
    axes[0, 1].legend(frameon=False, fontsize=8)
    fig.suptitle(
        f"L23: close agreement, not identical — global relative L2 difference {100 * relative_l2:.4f}%\n"
        "Independent GS; reference seed unknown; different hardware. Localhost result reported separately.",
        fontsize=11,
    )
    output = result_dir / "l23_comparison.png"
    fig.savefig(output, dpi=170)
    plt.close(fig)
    print(f"relative_L2={relative_l2:.17g}; max_abs={max_abs:.17g}; matching_peak_indices={matching_peaks}/23")
    print(output)


if __name__ == "__main__":
    main()
