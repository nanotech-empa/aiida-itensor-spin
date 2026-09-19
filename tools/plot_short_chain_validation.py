#!/usr/bin/env python3
"""Plot the short-chain tensor/ED and Gonçalo validation results."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RESULTS = ROOT / "validation" / "results"

ED_COLOR = "#263b61"
TENSOR_COLOR = "#df7138"
LEGACY_COLOR = "#3f8f68"
ERROR_COLOR = "#9e3a59"
GRID_COLOR = "#d5d9e2"

CASE_TITLES = {
    "three_spin_half_equal_antiferromagnetic": r"$[1/2,1/2,1/2]$, $J=[1,1]$ meV",
    "smallest_goncalo_equivalent": r"$[1,1/2,1]$, $J=[16.7,51.1]$ meV",
    "mixed_half_one_two_competing_couplings": r"$[1/2,1,2]$, $J=[0.35,-0.6]$ meV",
}


def load_json(path: Path) -> dict:
    return json.loads(path.read_text())


def save_both(figure, output_base: Path) -> None:
    figure.savefig(output_base.with_suffix(".png"), dpi=220, bbox_inches="tight")
    figure.savefig(
        output_base.with_suffix(".pdf"),
        bbox_inches="tight",
        metadata={
            "Creator": "aiida-itensor-spin validation",
            "CreationDate": None,
            "ModDate": None,
        },
    )
    plt.close(figure)


def plot_energy_suite(results: Path) -> None:
    report = load_json(results / "mixed_spin_L3_ed.json")
    figure, axes = plt.subplots(
        2,
        3,
        figsize=(13.2, 6.6),
        gridspec_kw={"height_ratios": [2.2, 1.0]},
        sharex="col",
    )
    for column, case in enumerate(report["cases"]):
        exact = np.asarray(case["exact_energies_meV"], dtype=float)
        tensor = np.asarray(case["tensor_energies_meV"], dtype=float)
        index = np.arange(len(exact))

        axis = axes[0, column]
        axis.plot(
            index,
            exact,
            "o-",
            color=ED_COLOR,
            linewidth=1.5,
            markersize=5,
            label="Exact diagonalization",
        )
        axis.plot(
            index,
            tensor,
            "x",
            color=TENSOR_COLOR,
            markersize=8,
            markeredgewidth=1.8,
            label="Mixed-spin ITensor",
        )
        axis.set_title(CASE_TITLES[case["name"]], fontsize=10.5)
        axis.set_ylabel("Sector energy (meV)" if column == 0 else "")
        axis.grid(color=GRID_COLOR, linewidth=0.6, alpha=0.65)
        if column == 0:
            axis.legend(frameon=False, fontsize=9)

        error_axis = axes[1, column]
        errors = tensor - exact
        error_axis.axhline(0.0, color="#777777", linewidth=0.8)
        error_axis.plot(index, errors, "o-", color=ERROR_COLOR, markersize=4)
        error_axis.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
        error_axis.grid(color=GRID_COLOR, linewidth=0.6, alpha=0.65)
        error_axis.set_xlabel("Eigenstate index")
        error_axis.set_ylabel(r"$E_{ITensor}-E_{ED}$ (meV)" if column == 0 else "")
        error_axis.text(
            0.03,
            0.9,
            f"max |ΔE| = {case['maximum_absolute_energy_error_meV']:.2e} meV",
            transform=error_axis.transAxes,
            fontsize=8.5,
            va="top",
        )

    figure.suptitle(
        "Three-site mixed-spin Heisenberg validation against exact diagonalization",
        fontsize=14,
    )
    figure.tight_layout()
    save_both(figure, results / "mixed_spin_L3_energy_comparison")


def plot_spectrum(results: Path) -> None:
    comparison = load_json(results / "mixed_spin_L3_spectrum_ed.json")
    spectrum = np.genfromtxt(
        results / "mixed_spin_L3_aiida_spectrum.tsv",
        names=True,
        delimiter="\t",
    )
    figure, axes = plt.subplots(1, 3, figsize=(13.4, 4.5), sharex=True)
    legend_handles = None
    legend_labels = None

    for axis, site_report in zip(axes, comparison["sites"], strict=True):
        site = site_report["site"]
        subset = spectrum[spectrum["site"] == site]
        order = np.argsort(subset["omega_meV"])
        omega = subset["omega_meV"][order]
        chi = subset["chi"][order]

        tensor_line = axis.plot(
            omega,
            chi,
            color=ED_COLOR,
            linewidth=1.8,
            label="AiiDA/ITensor spectrum",
        )[0]
        axis.set_title(f"Site {site}, S=1/2")
        axis.set_xlabel(r"$\omega$ (meV)")
        axis.set_ylabel(r"$\chi(\omega)$" if site == 1 else "")
        axis.grid(color=GRID_COLOR, linewidth=0.6, alpha=0.65)

        weight_axis = axis.twinx()
        frequencies = np.asarray(
            [pole["exact_frequency_meV"] for pole in site_report["exact_poles"]]
        )
        weights = np.asarray(
            [pole["combined_exact_weight"] for pole in site_report["exact_poles"]]
        )
        stems = weight_axis.vlines(
            frequencies,
            0.0,
            weights,
            color=TENSOR_COLOR,
            linewidth=2.2,
            alpha=0.9,
            label="Exact Lehmann weights",
        )
        weight_axis.scatter(
            frequencies,
            weights,
            color=TENSOR_COLOR,
            edgecolor="white",
            linewidth=0.6,
            s=34,
            zorder=4,
        )
        weight_axis.set_ylim(0, max(0.75, 1.15 * float(np.max(weights))))
        weight_axis.set_ylabel("Exact pole weight" if site == 3 else "")
        weight_axis.tick_params(axis="y", colors=TENSOR_COLOR)
        if site != 3:
            weight_axis.set_yticklabels([])

        area_error = site_report["area_error"]
        axis.text(
            0.04,
            0.94,
            f"area error = {area_error:+.3f}",
            transform=axis.transAxes,
            fontsize=9,
            va="top",
        )
        legend_handles = [tensor_line, stems]
        legend_labels = ["AiiDA/ITensor spectrum", "Exact Lehmann weights"]

    figure.legend(
        legend_handles,
        legend_labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.965),
        ncol=2,
        frameon=False,
    )
    figure.suptitle(
        "Three-spin-1/2 spectrum: broadened tensor result versus exact poles",
        fontsize=14,
        y=1.04,
    )
    figure.text(
        0.5,
        -0.015,
        "AiiDA WorkChain ccd512fe-7639-4222-9075-51919c745855; "
        "9/9 tasks, 135/135 recurrence units",
        ha="center",
        fontsize=9,
    )
    figure.tight_layout(rect=(0, 0.03, 1, 0.91))
    save_both(figure, results / "mixed_spin_L3_spectrum_comparison")


def plot_goncalo_comparison(results: Path) -> None:
    legacy = load_json(results / "smallest_goncalo_L3.json")
    suite = load_json(results / "mixed_spin_L3_ed.json")
    generic = next(
        case for case in suite["cases"] if case["name"] == "smallest_goncalo_equivalent"
    )

    exact = np.asarray(legacy["exact_energies_meV"], dtype=float)
    legacy_energies = np.asarray(legacy["tensor_energies_meV"], dtype=float)
    generic_energies = np.asarray(generic["tensor_energies_meV"], dtype=float)
    index = np.arange(len(exact))

    figure, axes = plt.subplots(
        2,
        1,
        figsize=(8.2, 7.0),
        gridspec_kw={"height_ratios": [2.2, 1.0]},
        sharex=True,
    )
    axes[0].plot(index, exact, "o-", color=ED_COLOR, label="Exact diagonalization")
    axes[0].plot(
        index,
        legacy_energies,
        "s",
        color=LEGACY_COLOR,
        markerfacecolor="none",
        markersize=8,
        markeredgewidth=1.7,
        label="Original Gonçalo model",
    )
    axes[0].plot(
        index,
        generic_energies,
        "x",
        color=TENSOR_COLOR,
        markersize=9,
        markeredgewidth=1.8,
        label="Generic mixed-spin model",
    )
    axes[0].set_ylabel("Sector energy (meV)")
    axes[0].set_title(
        r"Smallest Gonçalo-type chain: $[1,1/2,1]$, "
        r"$J_L=16.7$ meV, $J_R=51.1$ meV"
    )
    axes[0].grid(color=GRID_COLOR, linewidth=0.6, alpha=0.65)
    axes[0].legend(frameon=False)

    axes[1].axhline(0.0, color="#777777", linewidth=0.8)
    axes[1].plot(
        index,
        legacy_energies - exact,
        "s-",
        color=LEGACY_COLOR,
        label="Gonçalo − ED",
    )
    axes[1].plot(
        index,
        generic_energies - exact,
        "x-",
        color=TENSOR_COLOR,
        label="Generic − ED",
    )
    axes[1].set_xlabel("Eigenstate index")
    axes[1].set_ylabel("Energy residual (meV)")
    axes[1].ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
    axes[1].grid(color=GRID_COLOR, linewidth=0.6, alpha=0.65)
    axes[1].legend(frameon=False, ncol=2, fontsize=9)

    figure.suptitle(
        "Generic backend reproduces both ED and the original Gonçalo implementation",
        fontsize=13.5,
    )
    figure.tight_layout()
    save_both(figure, results / "smallest_goncalo_L3_comparison")



def plot_l5_energy_suite(results: Path) -> None:
    report = load_json(results / "mixed_spin_L5_ed.json")
    figure, axes = plt.subplots(
        2,
        2,
        figsize=(10.4, 7.0),
        gridspec_kw={"height_ratios": [2.2, 1.0]},
        sharex="col",
    )
    for column, case in enumerate(report["cases"]):
        exact = np.asarray(case["exact_low_energies_meV"], dtype=float)
        tensor = np.asarray(case["tensor_low_energies_meV"], dtype=float)
        index = np.arange(len(exact))
        couplings = ", ".join(f"{value:.3f}" for value in case["couplings_meV"])
        title = (
            r"Uniform antiferromagnetic bonds"
            if case["noise"] is None
            else r"Independent uniform bond disorder ($\pm 10\%$)"
        )

        axis = axes[0, column]
        axis.plot(
            index,
            exact,
            "o-",
            color=ED_COLOR,
            linewidth=1.5,
            markersize=5,
            label="Exact diagonalization",
        )
        axis.plot(
            index,
            tensor,
            "x",
            color=TENSOR_COLOR,
            markersize=8,
            markeredgewidth=1.8,
            label="Mixed-spin ITensor",
        )
        axis.set_title(f"{title}\nJ = [{couplings}] meV", fontsize=10.5)
        axis.set_ylabel("Low-energy spectrum (meV)" if column == 0 else "")
        axis.grid(color=GRID_COLOR, linewidth=0.6, alpha=0.65)
        axis.text(
            0.04,
            0.06,
            f"highest-energy error = {case['highest_energy_error_meV']:+.2e} meV",
            transform=axis.transAxes,
            fontsize=8.5,
        )
        if column == 0:
            axis.legend(frameon=False, fontsize=9)

        error_axis = axes[1, column]
        errors = tensor - exact
        error_axis.axhline(0.0, color="#777777", linewidth=0.8)
        error_axis.plot(index, errors, "o-", color=ERROR_COLOR, markersize=4)
        error_axis.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
        error_axis.grid(color=GRID_COLOR, linewidth=0.6, alpha=0.65)
        error_axis.set_xlabel("Low-energy eigenstate index")
        error_axis.set_ylabel(r"$E_{ITensor}-E_{ED}$ (meV)" if column == 0 else "")
        error_axis.text(
            0.04,
            0.9,
            f"max |ΔE| = {case['maximum_absolute_low_energy_error_meV']:.2e} meV",
            transform=error_axis.transAxes,
            fontsize=8.5,
            va="top",
        )

    figure.suptitle(
        r"Five-site chain $[1/2,1,2,1,1/2]$: "
        r"lowest 10 states in the $2S_z=0$ sector",
        fontsize=14,
    )
    figure.tight_layout()
    save_both(figure, results / "mixed_spin_L5_energy_comparison")

def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=Path, default=DEFAULT_RESULTS)
    args = parser.parse_args()
    plot_energy_suite(args.results)
    plot_spectrum(args.results)
    plot_goncalo_comparison(args.results)
    plot_l5_energy_suite(args.results)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
