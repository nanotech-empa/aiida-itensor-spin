"""Fast mathematical tests for the independent ED oracle."""

import sys
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

import numpy as np

PATH = Path("validation/exact_diagonalization.py")
SPEC = spec_from_file_location("exact_diagonalization", PATH)
MODULE = module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def test_three_spin_half_analytic_energies():
    exact = MODULE.diagonalize(
        [0.5, 0.5, 0.5],
        [(1, 2, 1.0), (2, 3, 1.0)],
        total_twice_sz=1,
    )
    np.testing.assert_allclose(exact.energies_meV, [-1.0, 0.0, 0.5], atol=1e-13, rtol=0)


def test_spin_two_operators_obey_commutator():
    operators = MODULE.spin_operators(4)
    np.testing.assert_allclose(
        operators["S+"] @ operators["S-"] - operators["S-"] @ operators["S+"],
        2 * operators["Sz"],
        atol=1e-13,
        rtol=0,
    )


def test_mixed_spin_hamiltonian_is_hermitian():
    hamiltonian, _ = MODULE.heisenberg_hamiltonian(
        [0.5, 1.0, 2.0],
        [(1, 2, 0.1), (2, 3, -0.25)],
    )
    assert hamiltonian.shape == (30, 30)
    np.testing.assert_allclose(hamiltonian, hamiltonian.conj().T, atol=1e-13, rtol=0)
