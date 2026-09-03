# aiida-itensor-spin

Adaptation of the spin-chain ITensor code developed by
[Gonçalo Catarina (@GCatarina)](https://github.com/GCatarina).

Experimental, restart-safe AiiDA orchestration for the Julia/ITensor spin-chain
spectral calculation on Daint/ALPS.

The public entry point is one complete `SpinChainWorkChain`: it validates
inputs, computes and parses the ground state, calls a provenance-recorded
resource-planning CalcFunction, runs bounded spectral segments, collects the
spectrum, and returns compact result files. Each segment reads an immutable
ground-state checkpoint and, when necessary, the immutable state of the
preceding segment. A valid `checkpointed` result triggers the next submission
automatically. The workflow can also make a bounded continuation from preserved
remote state when an execution ends unexpectedly.

## Scientific correction and validation

The corrected convention is `chebyshev-2n-minus-1-v2`: `N` stored Chebyshev
vectors determine exactly `2N-1` moments, `mu_0` through `mu_(2N-2)`. Legacy
files are rejected by explicit format and convention checks.

The repository contains two independently exercised implementations:

- `remote/engine/ITensor_libs/SpectralFunction.jl` is Gonçalo's original
  full-vector implementation with only the `2N` to `2N-1` correction;
- `remote/engine/SpectralFunctionCheckpointed.jl` is the restartable,
  memory-saving implementation used by AiiDA.

A fresh corrected L=11 ground state and all 33 spectral tasks completed on
2026-08-15. A deliberately interrupted streaming calculation resumed from
`n=12` with a different worker layout. Its collected spectrum agrees with the
independent corrected full-vector calculation to a maximum absolute difference
of `9.048317650695026e-14` (acceptance tolerance `1e-10`). See
[`results/L11/README.md`](results/L11/README.md) and the compact retained result
files.

## Repository layout

- `src/aiida_itensor_spin/`: CalcJobs, parsers, input contracts, and WorkChains;
- `remote/engine/`: Julia environment, ITensor libraries, GS preparation,
  checkpointed spectral workers, progress validation, and collection;
- `remote/itensor-spin-gs` and `remote/itensor-spin-segment`: Daint-side AiiDA
  executables;
- `examples/submit_l11.py`: reproducible fresh GS plus forced-stop/resume test;
- `results/L11/`: corrected streaming and full-vector results and provenance.

Ground-state checkpoints are retrieved once as AiiDA `SinglefileData`.
Recurrence checkpoints and final task files stay on Daint and pass between
segments as immutable AiiDA `RemoteData`. Repartitioning a continuation does
not modify or discard the preceding segment.

## L=11 result for Gonçalo to review

| Item | Value |
|---|---|
| Chain and impurity | `L=11`, impurity at site `6` |
| Couplings | `J=18.4 meV`, `Jp=16.7 meV` |
| Spectral grid | `0–100 meV`, smearing `2 meV` |
| Work completed | 1 ground state, 11 sites × 3 channels = 33 tasks |
| Corrected moments | `N` Chebyshev vectors produce `2N-1` moments |
| Restart test | stopped at recurrence `n=12`, then resumed and repartitioned |
| Independent check | streaming versus corrected full-vector result |
| Maximum absolute difference | `9.048317650695026e-14` (tolerance `1e-10`) |

![Corrected L=11 site-resolved spin spectrum](results/L11/l11_spectrum.png)

Please check whether this L=11 spectrum and the corrected moment convention are
physically as intended. The directly readable spectrum is
[`results/L11/streaming/spectrum.tsv`](results/L11/streaming/spectrum.tsv); the
complete compact JLD2 output and independent reference result are documented in
[`results/L11/README.md`](results/L11/README.md).

## Current installation and L=11 example

The validated Daint engines are isolated at:

```text
/capstor/scratch/cscs/cpi/goncalo/parallel/aiida_spin_engine_v4
/capstor/scratch/cscs/cpi/goncalo/parallel/aiida_spin_engine_v5
```

The GS remains on v4; restartable spectral calculations use v5:

```text
itensor-spin-gs-0.3.0-v4@daint.alps_lp83
itensor-spin-segment-0.3.0-v5@daint.alps_lp83
```

From this repository, submit the complete fresh validation with:

```console
verdi run examples/submit_l11.py
```

The example provides only resource ceilings. After parsing the L=11 GS, the
CalcFunction derives 33 tasks and selects 2 nodes x 6 processes/node x 2
threads/process, giving 2--3 tasks/process. For continuations it recomputes the
layout from the remaining task count and may automatically use fewer nodes.
The current CPU policy caps the selected threads/process at 2, based on the
validated L=23 scaling benchmark, while respecting the user-provided
`max_threads_per_process`.

Local checks:

```console
pip install -e .
pytest -q tests
pip check
```


### Validated end-to-end AiiDA run

On 2026-09-02, `SpinChainWorkChain<9767>` completed with `Finished [0]`.
Its provenance contains GS CalcJob `9768`, resource CalcFunction `9774`
(plan Dict `9775`), spectral WorkChain `9776`, and spectral CalcJob `9778`.
The final status reports 33/33 tasks and 3300/3300 recurrence units. AiiDA
stored `spectrum.tsv` (2,750 finite rows), `collection_report.txt`, and
`task_timings.tsv`; the spectrum is identical to the retained corrected L=11
streaming reference (maximum absolute chi difference `0.0`).

A controlled automatic-continuation test also completed successfully as
`SpinChainWorkChain<9840>` (`Finished [0]`). With a 6-minute Slurm walltime
and a 4-minute graceful-stop limit, segment CalcJob `9851` completed 29/33
tasks (3100/3300 recurrence units). The WorkChain then called resource-planning
CalcFunction `9856`, reduced the allocation from 2 nodes x 6 processes/node to
1 node x 4 processes, and submitted segment CalcJob `9859`, which completed the
remaining four tasks. The final spectrum `SinglefileData<9864>` is numerically
identical to the retained corrected L=11 reference.

### Scalable checkpoint accounting

The v5 segment engine publishes a valid status before starting its expensive
worker step and records each recurrence iteration in a tiny atomic sidecar.
Segment accounting therefore never deserializes the large MPS checkpoints.
Legacy v4 checkpoints without sidecars are accepted with a conservative
progress lower bound and are fully validated by a worker when resumed.

This change was motivated by an L=101 segment whose 560 physics workers stopped
cleanly but whose old accounting pass tried to load 379 GiB of checkpoints and
hit the outer Slurm walltime before writing its status. The v5 scanner processed
the same preserved state in 43.6 seconds on one debug node. A fresh two-segment
L=11 v5 acceptance WorkChain
`b5a66148-7a0f-488b-b5ac-4583dc4fcacb` completed 30/33 tasks, automatically
repacked the remaining three tasks, and returned a spectrum byte-for-byte
identical to the retained corrected reference.

The user-facing resource ceilings are stored in the `spectral_resources`
Dict: `max_nodes`, `max_processes_per_node`, and
`max_threads_per_process`, plus the segment wallclock, internal graceful-stop
time, and partition. Inspect any linked Dict with, for example:

```console
verdi data core.dict show PK
```

## Physics inputs to review with Gonçalo

The WorkChain intentionally exposes only chain `length`, `impurity_site`, and
random `seed` until the physical interface is agreed. The fresh L=11 test kept
the working-case values: `J=18.4 meV`, `Jp=16.7 meV`, `T=4 K`, frequency range
`omega_max=100 meV`, and smearing `2 meV`.

Before widening the interface, please classify the following as user physics
inputs or expert numerical controls:

- model/physics: `J`, `Jp`, temperature, frequency range, frequency smearing;
- GS numerics: bond dimension, DMRG cutoff, energy tolerances, number of excited
  states, and orthogonality penalty;
- algorithm controls: double-moment construction, backend, and threading mode.

After Gonçalo's review, we plan to expose the agreed physical quantities as validated WorkChain inputs while keeping numerical controls in an expert section. Larger production cases should wait for that review. The corrected spectral
code and the restart/repartition machinery are now independently validated.

## Legacy provenance

Earlier WorkChains and Daint files using the old `2N`-slot convention are
retained for provenance and debugging only. Their last moment was an untouched
zero, so they are not scientific references for v2.
