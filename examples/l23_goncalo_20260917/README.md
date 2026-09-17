# Exact L23 experiment source — 17 September 2026

Adaptation of code by [Gonçalo Catarina (@GCatarina)](https://github.com/GCatarina).
See the [completed result and comparison](../../results/L23/README.md).

This directory preserves the **actual isolated engine used for the Daint
experiment**, not a newly generalized physics-input API. The `engine/` source
snapshot is intentional: it allows review and future same-Mac testing without
silently following changes to the production engine. No production defaults
or running L101 calculations were modified by publishing this example.

## What changed in the tested code

Relative to the recovery engine at
`210f0830c3e360a09c84acd8a25c8715062c9775`:

1. `engine/prepare_gs.jl` reads the fixed reference case from `physics.toml`:
   independent left/right impurity couplings (`JL=16.7`, `JR=51.1` meV),
   spectral broadening `1.5` meV, and a separate spectral cutoff. It stores
   these inputs with the GS checkpoint. GS DMRG still uses cutoff `1e-8`.
2. `engine/chi_single_task_checkpoint.jl` reads `spectral_cutoff=1e-6` from
   that checkpoint, retaining the old cutoff lookup as a fallback.
3. The streaming recurrence, corrected `2N-1` moment convention, task
   assignment, checkpointing and collection algorithms are unchanged.

The `physics.toml` also records the other fixed numerical settings; not all of
them are a generalized configurable API in this snapshot. The prepare script
keeps the matching DMRG constants explicitly in its source. Gonçalo's optional
high-energy projection changes were not enabled or incorporated here; this
test uses the full spectral bandwidth from the computed Emax.

## Files and how the completed test ran

- `engine/`: exact deployed Julia sources, libraries, environment manifest,
  and Daint AiiDA wrappers.
- `physics.toml`: recorded L23 settings, including our seed **20260724**.
- `reference/`: Gonçalo's supplied code, compact spectrum and timing summary,
  unchanged. His script does not fix a seed; his GS checkpoint and environment
  manifest were not supplied. **The runs did not share a controlled seed/GS.**
- `compare_reference.jl`: comparison used after collection; reads reference
  key `χ_n_i` and streaming key `χGSi`, with all 23 sites and 333 frequencies.
- `preflight.jl`: three-site comparison of all three channels; the preflight
  agreed exactly. This small test does not replace the full L23 comparison.
- `historical/submit_aiida.py`: the actual full WorkChain submission, including
  immutable test Code definitions and the 14:00 maintenance deadline.
- `historical/continue_aiida.py`: the actual standard spectral WorkChain
  continuation from immutable AiiDA outputs after the one-segment limit.

The run used `WorkflowFactory("itensor_spin.full")` and `aiida.engine.submit`.
After GS, its resource calcfunction discovered 69 tasks and selected one node,
23 processes and two threads each. Following the graceful stop, the standard
`itensor_spin.spectra` continuation used four processes on one node, resumed
the partial recurrences and returned the complete spectrum.

**The historical scripts are records, not portable launchers.** They contain
the original absolute deployment paths, AiiDA UUIDs, test Code labels and an
expired September 17 deadline. Do not run them blindly or edit existing
scientific nodes to reuse them. A new test needs a fresh isolated deployment
and supported AiiDA submission with fresh labels and appropriate limits.
The historical source-checksum manifest mentioned by the original submitter
belongs to that deployment; it is not the repository's checksum manifest.

## Planned controlled Mac test

The next step is the same Mac architecture as Gonçalo's reference, recording
the exact CPU and software versions, then comparing both implementations on
one Mac with the **same saved GS**, parameters and explicit thread budget.
This has **not been run yet**. No claim about Mac performance is made here.

The numerical Julia routines are separate from the Slurm submission wrappers;
the future Mac launcher is not implemented by this publication. Generalized
physics inputs for the WorkChain remain a separate review with Gonçalo.

For now, reproduce the local data comparison and plot (no Daint job) from the
repository root, in an environment with NumPy, h5py and Matplotlib:

```console
python tools/plot_l23_comparison.py
```

This checks the retained reference/result arrays against the published error
metrics and produces `results/L23/l23_comparison.png`.
