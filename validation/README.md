# Short-chain exact-diagonalization validation

The dense reference in exact_diagonalization.py is intentionally independent of ITensor and
AiiDA. It constructs spin matrices and the full Heisenberg Hamiltonian directly with NumPy.

The first regression case is the smallest version of the existing Gonçalo impurity model:

- L = 3;
- site spins [1, 1/2, 1];
- impurity at site 2;
- J_L = 16.7 meV and J_R = 51.1 meV;
- conserved twice-Sz = 1.

Run:

    python validation/compare_smallest_goncalo.py \
      --output validation/results/smallest_goncalo_L3.json

The command runs the current tensor-network ground-state executable, diagonalizes the same
Hamiltonian independently, and fails unless the computed low energies and highest energy agree
within the recorded tolerance. Spectral Lehmann poles and weights are also available from
local_transition_weights for the mixed-spin spectral regression added in the next stage.
