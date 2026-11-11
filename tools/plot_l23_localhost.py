"""Verify and plot the retained L23 localhost validation; submits no calculations."""

from __future__ import annotations

import csv
import tomllib
from pathlib import Path

import h5py
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

REFERENCE_COLOR = "#263b61"
LOCAL_COLOR = "#e67e3f"
DELTA_COLOR = "#9e3a59"
CHANNEL_COLORS = {"zz": "#3b6ea8", "pm": "#e09f3e", "mp": "#6a994e"}


def read_spectrum(path: Path) -> tuple[np.ndarray, np.ndarray]:
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    if len(rows) != 23 * 333 or {int(row["state"]) for row in rows} != {1}:
        raise ValueError("Expected one state, 23 sites and 333 frequencies")
    omega, spectra = [], []
    for site in range(1, 24):
        site_rows = [row for row in rows if int(row["site"]) == site]
        omega.append([float(row["omega_meV"]) for row in site_rows])
        spectra.append([float(row["chi"]) for row in site_rows])
    return np.asarray(omega), np.asarray(spectra)


def read_reference(path: Path) -> np.ndarray:
    with h5py.File(path, "r") as handle:
        references = handle[handle["χ_n_i"][0]][()]
        return np.asarray([handle[reference][()] for reference in references])


def verify_metrics(reference, spectrum, omega, recorded):
    expected_omega = np.tile(np.linspace(0, 100, 333), (23, 1))
    if (
        reference.shape != spectrum.shape
        or reference.shape != omega.shape
        or reference.shape != (23, 333)
    ):
        raise ValueError("Reference and result must both contain 23 x 333 values")
    if not np.isfinite(reference).all() or not np.isfinite(spectrum).all():
        raise ValueError("Reference or result contains non-finite values")
    np.testing.assert_allclose(omega, expected_omega)
    delta = spectrum - reference
    relative_l2 = np.linalg.norm(delta) / np.linalg.norm(reference)
    max_abs = np.max(np.abs(delta))
    np.testing.assert_allclose(relative_l2, recorded["relative_L2_difference"], rtol=1e-12)
    np.testing.assert_allclose(max_abs, recorded["max_abs_difference"], rtol=1e-12)
    matching_peaks = np.count_nonzero(np.argmax(reference, axis=1) == np.argmax(spectrum, axis=1))
    if matching_peaks != 23:
        raise ValueError(f"Only {matching_peaks}/23 sampled peak positions match")
    return delta, relative_l2, max_abs, matching_peaks


def plot_spectra(output, reference, spectrum, omega, delta, relative_l2):
    fig, axes = plt.subplots(2, 3, figsize=(12.2, 6.5), sharex="col", constrained_layout=True)
    for column, site in enumerate((1, 12, 23)):
        index = site - 1
        axes[0, column].plot(
            omega[index],
            reference[index],
            color=REFERENCE_COLOR,
            linewidth=1.8,
            label="Gonçalo gold standard",
        )
        axes[0, column].plot(
            omega[index],
            spectrum[index],
            "--",
            color=LOCAL_COLOR,
            linewidth=1.45,
            label="AiiDA localhost",
        )
        axes[0, column].set_title(f"Site {site}" + (" — spin-½ impurity" if site == 12 else ""))
        axes[1, column].plot(omega[index], delta[index], color=DELTA_COLOR, linewidth=1.25)
        axes[1, column].axhline(0, color="0.45", linewidth=0.65)
        axes[1, column].set_xlabel("Energy ω (meV)")
        axes[1, column].ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
        for row in range(2):
            axes[row, column].set_xlim(0, 100)
            axes[row, column].grid(alpha=0.18)
    axes[0, 0].set_ylabel("χᵢ(ω)")
    axes[1, 0].set_ylabel("Localhost − gold")
    axes[0, 1].legend(frameon=False, fontsize=8.5)
    fig.suptitle(
        f"L=23 localhost validation — relative L2 difference {100 * relative_l2:.4f}%\n"
        "Independent ground states; reference seed unknown; sampled peaks match at all 23 sites",
        fontsize=11.5,
    )
    fig.savefig(output, dpi=180)
    plt.close(fig)


def plot_diagnostics(output, comparison_path, timing_path, summary_path):
    with comparison_path.open(newline="") as handle:
        comparison = list(csv.DictReader(handle, delimiter="\t"))
    with timing_path.open(newline="") as handle:
        timings = list(csv.DictReader(handle, delimiter="\t"))
    with summary_path.open(newline="") as handle:
        summaries = list(csv.DictReader(handle, delimiter="\t"))

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8), constrained_layout=True)

    sites = np.asarray([int(row["site"]) for row in comparison])
    site_l2 = 100 * np.asarray([float(row["relative_L2"]) for row in comparison])
    axes[0].bar(sites, site_l2, color="#4f8a8b", width=0.78)
    axes[0].axvspan(11.5, 12.5, color="#f4a261", alpha=0.18, label="impurity")
    axes[0].set(xlabel="Site", ylabel="Relative L2 difference (%)", title="Agreement by site")
    axes[0].set_xticks([1, 4, 8, 12, 16, 20, 23])
    axes[0].grid(axis="y", alpha=0.2)
    axes[0].legend(frameon=False, fontsize=8)

    offsets = {"zz": -0.24, "pm": 0.0, "mp": 0.24}
    for channel in ("zz", "pm", "mp"):
        rows = [row for row in timings if row["channel"] == channel]
        x = np.asarray([int(row["site"]) for row in rows], dtype=float) + offsets[channel]
        minutes = np.asarray([float(row["seconds"]) / 60 for row in rows])
        axes[1].bar(x, minutes, width=0.22, color=CHANNEL_COLORS[channel], label=channel)
    axes[1].axvspan(11.5, 12.5, color="#d62828", alpha=0.08)
    axes[1].set_yscale("log")
    axes[1].set(
        xlabel="Site", ylabel="Task time (minutes, log scale)", title="Measured spectral-task cost"
    )
    axes[1].set_xticks([1, 4, 8, 12, 16, 20, 23])
    axes[1].grid(axis="y", which="both", alpha=0.2)
    axes[1].legend(frameon=False, ncol=3, fontsize=8)

    labels = [row["label"] for row in summaries]
    minutes = [float(row["minutes"]) for row in summaries]
    colors = ["#7a8fa6", "#506784", LOCAL_COLOR, "#8b6f9e"]
    positions = np.arange(len(labels))
    axes[2].barh(positions, minutes, color=colors)
    axes[2].set_yticks(positions, labels)
    axes[2].invert_yaxis()
    axes[2].set(xlabel="Elapsed time (minutes)", title="Reported elapsed times")
    axes[2].grid(axis="x", alpha=0.2)
    for position, value in zip(positions, minutes):
        axes[2].text(value + 0.45, position, f"{value:.1f}", va="center", fontsize=8.5)
    axes[2].set_xlim(0, max(minutes) * 1.18)

    fig.suptitle(
        "L=23 numerical agreement and performance diagnostics\n"
        "Time bars use different machines/boundaries and are not a controlled speedup benchmark",
        fontsize=11.5,
    )
    fig.savefig(output, dpi=180)
    plt.close(fig)


def main():
    root = Path(__file__).resolve().parents[1]
    result_dir = root / "results" / "L23"
    local_dir = result_dir / "localhost"
    reference_path = root / "examples" / "l23_goncalo_20260917" / "reference" / "chi_L23.jld2"

    reference = read_reference(reference_path)
    omega, spectrum = read_spectrum(local_dir / "spectrum.tsv")
    with (local_dir / "comparison.toml").open("rb") as handle:
        recorded = tomllib.load(handle)
    delta, relative_l2, max_abs, matching_peaks = verify_metrics(
        reference, spectrum, omega, recorded
    )

    comparison_output = result_dir / "l23_localhost_comparison.png"
    diagnostics_output = result_dir / "l23_localhost_diagnostics.png"
    plot_spectra(comparison_output, reference, spectrum, omega, delta, relative_l2)
    plot_diagnostics(
        diagnostics_output,
        local_dir / "comparison_by_site.tsv",
        local_dir / "task_timings.tsv",
        result_dir / "timing_summary.tsv",
    )
    print(
        f"relative_L2={relative_l2:.17g}; max_abs={max_abs:.17g}; "
        f"matching_peak_indices={matching_peaks}/23"
    )
    print(comparison_output)
    print(diagnostics_output)


if __name__ == "__main__":
    main()
