"""Model dispatch tests for local and Daint ground-state wrappers."""

from __future__ import annotations

import runpy
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
MIXED = {
    "model": "mixed_spin_heisenberg_chain_v1",
    "site_spins": [0.5, 1.0, 2.0],
    "couplings_meV": [0.35, -0.6],
    "target_twice_sz": 1,
}


@pytest.mark.parametrize("wrapper", ["itensor-spin-gs-local", "itensor-spin-gs"])
def test_registered_models_dispatch_to_separate_gs_adapters(wrapper):
    namespace = runpy.run_path(ROOT / "remote" / wrapper)
    command = namespace["model_command"]
    engine = Path("/engine")

    legacy_script, legacy_arguments = command(
        {
            "model": "spin1_chain_spinhalf_impurity_v1",
            "length": 3,
            "impurity_site": 2,
            "J_meV": 18.4,
            "JL_meV": 16.7,
            "JR_meV": 51.1,
        },
        engine,
    )
    assert legacy_script == engine / "prepare_gs.jl"
    assert legacy_arguments[:4] == ["--L", "3", "--impurity-site", "2"]

    mixed_script, mixed_arguments = command(MIXED, engine)
    assert mixed_script == engine / "prepare_mixed_spin_gs.jl"
    assert mixed_arguments == [
        "--two-spins",
        "1,2,4",
        "--couplings-meV",
        "0.35,-0.6",
        "--target-twice-sz",
        "1",
    ]


def test_wrapper_rejects_an_unregistered_model():
    command = runpy.run_path(ROOT / "remote" / "itensor-spin-gs-local")["model_command"]
    with pytest.raises(RuntimeError, match="unsupported model"):
        command({"model": "unregistered"}, Path("/engine"))
