# Corrected L=11 validation

This directory records the fresh validation performed on Daint/ALPS on
2026-08-15. Both calculations read the **same newly computed ground-state
checkpoint** (SHA-256
`1a6887aa863e987c19c7691208bf90a6b124f6245a12fb300c1b5698b813d105`) and
use the corrected convention `chebyshev-2n-minus-1-v2`:

> `N` stored Chebyshev vectors determine `2N-1` moments, `mu_0` through
> `mu_(2N-2)`.

## Comparison

| Calculation | Backend | Execution | Result |
|---|---|---|---|
| AiiDA restart test | streaming/checkpointed | segment 1 intentionally stopped at recurrence `n=12`; segment 2 resumed with a different worker layout and completed all 33 tasks | `streaming/chi_L11.jld2` |
| Independent reference | original full-vector algorithm with the `2N-1` correction | 1 node, 1 process, 11 Julia threads | `full_reference/chi_L11.jld2` |

The maximum absolute difference between the two collected spectra is
`9.048317650695026e-14`, comfortably below the validation tolerance of
`1e-10`. The reference calculation's computational pipeline took 168 seconds;
the enclosing Slurm job took 247 seconds including Julia/environment startup.

The tab-separated spectra, per-task timings, collection reports, and compact
JLD2 results are retained. The large ground-state and recurrence checkpoint
files remain in AiiDA/Daint provenance and are intentionally not copied into
Git.

## Provenance

- Full AiiDA WorkChain UUID:
  `e1201945-c28f-4128-835a-0fc8999964dd`
- Fresh ground-state CalcJob UUID:
  `027f0380-f4ae-425a-a990-e497bb868e73`
- Restartable spectral WorkChain UUID:
  `d67ad4b0-8a29-48cd-bb80-34ec97e0ed3b`
- Ground-state AiiDA code UUID:
  `4d9a8c63-7384-4b80-b215-654423a7b69c`
- Spectral AiiDA code UUID:
  `e67e5248-6bbf-4f3c-9264-e45e482d4690`
- Independent reference Slurm job: `4489512`

The AiiDA restart test deliberately used:

1. segment 1: 2 nodes, 4 processes/node, 2 threads/process, forced stop at
   recurrence `n=12`;
2. segment 2: 2 nodes, 6 processes/node, 2 threads/process, resuming the
   immutable checkpoints produced by segment 1.

This validates both checkpoint continuation and repartitioning between Slurm
allocations.
