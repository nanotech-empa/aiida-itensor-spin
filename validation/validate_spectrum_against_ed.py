#!/usr/bin/env python3
"""Compare a completed L=3 mixed-spin spectrum with exact Lehmann data."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from exact_diagonalization import local_transition_weights

CHANNEL_WEIGHTS = {"zz": 1.0, "pm": 0.5, "mp": 0.5}


def combined_exact_poles(site: int) -> list[tuple[float, float]]:
    transitions = local_transition_weights(
        [0.5, 0.5, 0.5],
        [(1, 2, 1.0), (2, 3, 1.0)],
        total_twice_sz=1,
    )
    combined: dict[float, float] = {}
    for channel, channel_weight in CHANNEL_WEIGHTS.items():
        frequencies, weights = transitions[(site, channel)]
        for frequency, weight in zip(frequencies, weights, strict=True):
            key = round(float(frequency), 12)
            combined[key] = combined.get(key, 0.0) + channel_weight * float(weight)
    return sorted(combined.items())


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("spectrum", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--smearing-meV", type=float, default=0.05)
    parser.add_argument("--area-tolerance", type=float, default=0.03)
    args = parser.parse_args()

    data = np.genfromtxt(args.spectrum, names=True, delimiter="\t")
    omega_max = float(np.max(data["omega_meV"]))
    reports = []
    for site in (1, 2, 3):
        subset = data[data["site"] == site]
        order = np.argsort(subset["omega_meV"])
        omega = subset["omega_meV"][order]
        chi = subset["chi"][order]
        poles = combined_exact_poles(site)

        peak_reports = []
        for exact_frequency, exact_weight in poles:
            lower = max(0.0, exact_frequency - 2 * args.smearing_meV)
            upper = min(omega_max, exact_frequency + 2 * args.smearing_meV)
            indices = np.flatnonzero((omega >= lower) & (omega <= upper))
            if not len(indices):
                raise AssertionError(f"no frequency samples around pole {exact_frequency}")
            peak_index = indices[np.argmax(chi[indices])]
            observed_frequency = float(omega[peak_index])
            frequency_error = observed_frequency - exact_frequency
            if abs(frequency_error) > 0.021:
                raise AssertionError(
                    f"site {site} pole {exact_frequency}: peak at {observed_frequency}"
                )
            peak_reports.append(
                {
                    "exact_frequency_meV": exact_frequency,
                    "combined_exact_weight": exact_weight,
                    "observed_peak_frequency_meV": observed_frequency,
                    "frequency_error_meV": frequency_error,
                }
            )

        exact_sum_rule = sum(weight for _, weight in poles)
        expected_spin_sum_rule = 0.5 * (0.5 + 1)
        np.testing.assert_allclose(exact_sum_rule, expected_spin_sum_rule, atol=1e-12, rtol=0)

        boundary_corrected_area = sum(
            weight
            * (
                0.5
                if np.isclose(frequency, 0.0, atol=1e-12)
                or np.isclose(frequency, omega_max, atol=1e-12)
                else 1.0
            )
            for frequency, weight in poles
        )
        numerical_area = float(np.trapezoid(chi, omega))
        area_error = numerical_area - boundary_corrected_area
        if abs(area_error) > args.area_tolerance:
            raise AssertionError(
                f"site {site}: spectral area error {area_error} exceeds "
                f"{args.area_tolerance}"
            )
        reports.append(
            {
                "site": site,
                "exact_poles": peak_reports,
                "exact_full_sum_rule": exact_sum_rule,
                "expected_S_times_S_plus_1": expected_spin_sum_rule,
                "exact_area_in_closed_frequency_window": boundary_corrected_area,
                "tensor_numerical_area": numerical_area,
                "area_error": area_error,
            }
        )

    report = {
        "case": "three_spin_half_equal_antiferromagnetic",
        "model": "mixed_spin_heisenberg_chain_v1",
        "site_spins": [0.5, 0.5, 0.5],
        "couplings_meV": [1.0, 1.0],
        "omega_max_meV": omega_max,
        "omega_smearing_meV": args.smearing_meV,
        "comparison": (
            "Exact Lehmann pole positions, local S(S+1) sum rule, and finite-window "
            "broadened spectral area"
        ),
        "sites": reports,
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
