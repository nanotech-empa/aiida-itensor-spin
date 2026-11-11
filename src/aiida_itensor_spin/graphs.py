"""Explicit finite, undirected isotropic Heisenberg spin graphs."""

from __future__ import annotations

import math
from collections.abc import Mapping

GRAPH_MODEL_IDENTIFIER = "mixed_spin_heisenberg_graph_v1"


def finite_real(value, label):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"{label} must be a finite real number")
    return float(value)


def normalize_spin_graph(value):
    """Validate topology without inferring bonds/J from geometry or renumbering sites."""
    if not isinstance(value, Mapping):
        raise TypeError("Spin graph must be an object")
    raw_spins = value.get("site_spins")
    if not isinstance(raw_spins, list) or len(raw_spins) < 2:
        raise ValueError("Provide at least two site spins")
    spins = []
    for i, raw in enumerate(raw_spins, 1):
        s = finite_real(raw, f"Spin at site {i}")
        if s <= 0 or abs(2 * s - round(2 * s)) > 1e-12:
            raise ValueError(f"Spin at site {i} must be a positive integer or half-integer")
        spins.append(round(2 * s) / 2)
    raw_edges = value.get("exchange_edges")
    if not isinstance(raw_edges, list) or not raw_edges:
        raise ValueError("Provide at least one exchange connection")
    edges, seen = [], set()
    for edge in raw_edges:
        if not isinstance(edge, Mapping) or set(edge) != {"i", "j", "J_meV"}:
            raise ValueError("Each connection requires exactly i, j and J_meV")
        i, j = edge["i"], edge["j"]
        if any(
            isinstance(k, bool) or not isinstance(k, int) or not 1 <= k <= len(spins)
            for k in (i, j)
        ):
            raise ValueError("Connection endpoints must be existing integer site numbers")
        if i == j:
            raise ValueError("Self connections are not supported")
        i, j = sorted((i, j))
        if (i, j) in seen:
            raise ValueError(f"Duplicate connection {i}–{j}; specify its total exchange once")
        seen.add((i, j))
        edges.append({"i": i, "j": j, "J_meV": finite_real(edge["J_meV"], f"J for {i}–{j}")})
    edges.sort(key=lambda e: (e["i"], e["j"]))
    result = {"site_spins": spins, "exchange_edges": edges}
    positions = value.get("site_positions")
    if positions is not None:
        if not isinstance(positions, list) or len(positions) != len(spins):
            raise ValueError("Provide one drawing position per site")
        result["site_positions"] = []
        for point in positions:
            if not isinstance(point, (list, tuple)) or len(point) != 2:
                raise ValueError("Drawing positions require x and y")
            result["site_positions"].append([finite_real(p, "Drawing coordinate") for p in point])
    return result
