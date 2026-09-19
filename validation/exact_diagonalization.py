"""Independent dense exact diagonalization for finite isotropic spin systems.

This module intentionally does not import ITensor or aiida_itensor_spin. It is
an external numerical oracle for validating the tensor-network implementation.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ExactSpectrum:
    """Eigensystem in one conserved twice-Sz sector."""

    energies_meV: np.ndarray
    vectors: np.ndarray
    full_basis_indices: np.ndarray
    total_twice_sz: int


def normalize_twice_spins(spins: list[float]) -> tuple[int, ...]:
    """Convert integer/half-integer spins to exact twice-spin integers."""
    result = []
    for index, spin in enumerate(spins, start=1):
        if isinstance(spin, bool) or not np.isfinite(spin) or spin <= 0:
            raise ValueError(f"spin at site {index} must be finite and positive")
        twice = round(2 * float(spin))
        if not np.isclose(2 * float(spin), twice, atol=1e-12, rtol=0):
            raise ValueError(f"spin at site {index} must be integer or half-integer")
        result.append(int(twice))
    return tuple(result)


def spin_operators(twice_spin: int) -> dict[str, np.ndarray]:
    """Return Sz, S+, and S- in the descending-m basis."""
    if isinstance(twice_spin, bool) or not isinstance(twice_spin, int) or twice_spin < 1:
        raise ValueError("twice_spin must be a positive integer")
    spin = twice_spin / 2
    dimension = twice_spin + 1
    m_values = spin - np.arange(dimension, dtype=float)
    sz = np.diag(m_values).astype(complex)
    raising = np.zeros((dimension, dimension), dtype=complex)
    for column, m_value in enumerate(m_values):
        if column == 0:
            continue
        raising[column - 1, column] = np.sqrt(
            spin * (spin + 1) - m_value * (m_value + 1)
        )
    lowering = raising.conj().T
    return {"Id": np.eye(dimension, dtype=complex), "Sz": sz, "S+": raising, "S-": lowering}


def _embedded_operator(
    local_operator: np.ndarray,
    site: int,
    twice_spins: tuple[int, ...],
) -> np.ndarray:
    factors = []
    for index, twice_spin in enumerate(twice_spins, start=1):
        factors.append(
            local_operator
            if index == site
            else np.eye(twice_spin + 1, dtype=complex)
        )
    result = factors[0]
    for factor in factors[1:]:
        result = np.kron(result, factor)
    return result


def local_operators(spins: list[float]) -> dict[str, list[np.ndarray]]:
    """Build every local spin operator in the full tensor-product space."""
    twice_spins = normalize_twice_spins(spins)
    result = {"Sz": [], "S+": [], "S-": []}
    for site, twice_spin in enumerate(twice_spins, start=1):
        operators = spin_operators(twice_spin)
        for name, embedded in result.items():
            embedded.append(_embedded_operator(operators[name], site, twice_spins))
    return result


def heisenberg_hamiltonian(
    spins: list[float],
    bonds: list[tuple[int, int, float]],
) -> tuple[np.ndarray, dict[str, list[np.ndarray]]]:
    """Construct H=sum_ij J_ij S_i.S_j in meV."""
    twice_spins = normalize_twice_spins(spins)
    operators = local_operators(spins)
    dimension = int(np.prod([value + 1 for value in twice_spins]))
    hamiltonian = np.zeros((dimension, dimension), dtype=complex)
    seen = set()
    for left, right, coupling in bonds:
        if (
            isinstance(left, bool)
            or isinstance(right, bool)
            or not isinstance(left, int)
            or not isinstance(right, int)
            or not 1 <= left <= len(spins)
            or not 1 <= right <= len(spins)
            or left == right
        ):
            raise ValueError(f"invalid bond ({left}, {right})")
        key = tuple(sorted((left, right)))
        if key in seen:
            raise ValueError(f"duplicate bond {key}")
        if isinstance(coupling, bool) or not np.isfinite(coupling):
            raise ValueError(f"coupling for bond {key} must be finite")
        seen.add(key)
        i, j = left - 1, right - 1
        hamiltonian += float(coupling) * (
            operators["Sz"][i] @ operators["Sz"][j]
            + 0.5 * operators["S+"][i] @ operators["S-"][j]
            + 0.5 * operators["S-"][i] @ operators["S+"][j]
        )
    if not np.allclose(hamiltonian, hamiltonian.conj().T, atol=1e-13, rtol=0):
        raise AssertionError("constructed Hamiltonian is not Hermitian")
    return hamiltonian, operators


def product_basis_twice_sz(spins: list[float]) -> np.ndarray:
    """Return twice-Sz for each tensor-product basis vector."""
    twice_spins = normalize_twice_spins(spins)
    local_values = [range(value, -value - 1, -2) for value in twice_spins]
    return np.asarray(
        [sum(configuration) for configuration in itertools.product(*local_values)],
        dtype=int,
    )


def diagonalize(
    spins: list[float],
    bonds: list[tuple[int, int, float]],
    *,
    total_twice_sz: int,
) -> ExactSpectrum:
    """Diagonalize the Hamiltonian in one exactly selected Sz sector."""
    hamiltonian, _ = heisenberg_hamiltonian(spins, bonds)
    basis_twice_sz = product_basis_twice_sz(spins)
    indices = np.flatnonzero(basis_twice_sz == total_twice_sz)
    if not len(indices):
        raise ValueError(f"no product states exist for twice-Sz={total_twice_sz}")
    sector = hamiltonian[np.ix_(indices, indices)]
    energies, vectors = np.linalg.eigh(sector)
    return ExactSpectrum(
        energies_meV=energies,
        vectors=vectors,
        full_basis_indices=indices,
        total_twice_sz=total_twice_sz,
    )


def local_transition_weights(
    spins: list[float],
    bonds: list[tuple[int, int, float]],
    *,
    total_twice_sz: int,
    ground_index: int = 0,
) -> dict[tuple[int, str], tuple[np.ndarray, np.ndarray]]:
    """Exact zero-temperature Lehmann poles and weights for all local channels."""
    hamiltonian, operators = heisenberg_hamiltonian(spins, bonds)
    full_energies, full_vectors = np.linalg.eigh(hamiltonian)
    sector = diagonalize(spins, bonds, total_twice_sz=total_twice_sz)
    ground_sector_vector = sector.vectors[:, ground_index]
    ground_vector = np.zeros(hamiltonian.shape[0], dtype=complex)
    ground_vector[sector.full_basis_indices] = ground_sector_vector
    ground_energy = sector.energies_meV[ground_index]

    result = {}
    for site in range(len(spins)):
        for channel, operator_name in (("zz", "Sz"), ("pm", "S-"), ("mp", "S+")):
            amplitudes = full_vectors.conj().T @ (operators[operator_name][site] @ ground_vector)
            weights = np.abs(amplitudes) ** 2
            omega = full_energies - ground_energy
            mask = weights > 1e-13
            result[(site + 1, channel)] = (omega[mask], weights[mask])
    return result
