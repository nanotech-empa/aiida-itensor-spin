"""Write independent dense eigenspectra for the Julia graph validation cases."""

import sys
from pathlib import Path

import numpy as np
from exact_diagonalization import heisenberg_hamiltonian

cases = {
    "triangle": ([0.5] * 3, [(1, 2, 1.0), (2, 3, 1.0), (1, 3, 1.0)]),
    "branch": ([0.5, 1.0, 0.5, 0.5], [(1, 2, 0.8), (2, 3, -0.3), (2, 4, 1.1)]),
    "long_range": ([0.5] * 4, [(1, 2, 1.0), (2, 3, 0.6), (3, 4, 0.9), (1, 3, 0.25), (1, 4, -0.2)]),
    "dimer": ([0.5] * 2, [(1, 2, 1.0)]),
}
root = Path(sys.argv[1])
root.mkdir(parents=True, exist_ok=True)
for name, (spins, bonds) in cases.items():
    h, _ = heisenberg_hamiltonian(spins, bonds)
    np.savetxt(root / f"{name}.txt", np.linalg.eigvalsh(h))
print("Wrote four independent graph eigenspectra.")
