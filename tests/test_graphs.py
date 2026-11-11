"""Scientific input and dispatch checks for arbitrary explicit exchange topology."""

import math
import runpy
from pathlib import Path

import pytest

from aiida_itensor_spin.contracts import normalize_ground_state_parameters
from aiida_itensor_spin.graphs import GRAPH_MODEL_IDENTIFIER, normalize_spin_graph

GRAPH = {
    "site_spins": [0.5, 1, 0.5, 1],
    "exchange_edges": [
        {"i": 1, "j": 2, "J_meV": 1},
        {"i": 2, "j": 3, "J_meV": -0.2},
        {"i": 2, "j": 4, "J_meV": 0.7},
        {"i": 1, "j": 3, "J_meV": 0.4},
    ],
}


def test_branch_and_non_neighbor_survive_normalization():
    graph = normalize_spin_graph(GRAPH)
    p = normalize_ground_state_parameters(dict(model=GRAPH_MODEL_IDENTIFIER, seed=1, **graph))
    assert p["site_spins"] == GRAPH["site_spins"]
    assert p["length"] == 4
    assert {tuple(e.values()) for e in p["exchange_edges"]} == {
        tuple(e.values()) for e in GRAPH["exchange_edges"]
    }
    assert "couplings_meV" not in p
    assert (
        normalize_spin_graph({**GRAPH, "site_positions": [[1, 1]] * 4})["exchange_edges"]
        == graph["exchange_edges"]
    )


@pytest.mark.parametrize(
    "edges",
    [
        [{"i": 1, "j": 1, "J_meV": 1}],
        [{"i": 1, "j": 8, "J_meV": 1}],
        [{"i": 1.0, "j": 2, "J_meV": 1}],
        [{"i": 1, "j": 2, "J_meV": math.nan}],
        [{"i": 1, "j": 2, "J_meV": 1}, {"i": 2, "j": 1, "J_meV": 1}],
        [],
        [{"i": 1, "j": 2, "J_meV": None}],
    ],
)
def test_invalid_or_unreviewed_exchange_rejected(edges):
    with pytest.raises(ValueError):
        normalize_spin_graph({**GRAPH, "exchange_edges": edges})


@pytest.mark.parametrize("spin", [True, 0, 0.7, math.inf, None])
def test_invalid_or_unreviewed_spin_rejected(spin):
    with pytest.raises(ValueError):
        normalize_spin_graph({**GRAPH, "site_spins": [spin, 0.5]})


@pytest.mark.parametrize("wrapper", ["itensor-spin-gs", "itensor-spin-gs-local"])
def test_graph_dispatch_records_exact_edges(wrapper):
    root = Path(__file__).resolve().parents[1]
    f = runpy.run_path(root / "remote" / wrapper)["model_command"]
    p = normalize_ground_state_parameters(dict(model=GRAPH_MODEL_IDENTIFIER, seed=1, **GRAPH))
    script, args = f(p, Path("/isolated/engine"))
    assert script == Path("/isolated/engine/prepare_mixed_spin_gs.jl")
    assert "--exchange-edges" in args
    assert "1,3,0.4" in args[args.index("--exchange-edges") + 1]
