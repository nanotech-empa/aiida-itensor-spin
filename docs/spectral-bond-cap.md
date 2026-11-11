# Optional spectral bond-dimension cap

The standard workflow accepts `spectral_parameters.max_bond_dimension`:

```python
builder.spectral_parameters = orm.Dict(dict={
    "chebyshev_convention": "chebyshev-2n-minus-1-v2",
    "backend": "streaming",
    "threading_mode": "blocksparse",
    "spectral_cutoff": 1e-8,
    "max_bond_dimension": 9500,
})
```

Omission or `0` preserves the previous unlimited spectral calculation. A positive
limit requires an updated, versioned Code declaring
`supports_spectral_bond_cap=True`. Both the GUI and CalcJob reject older Codes
instead of silently ignoring the cap.

A positive
integer limits the **MPS link dimension**, not the chain length or physical spin.
The ground-state DMRG is unchanged. This does not impose a hard RAM limit:
contraction environments, MPO bonds and temporary tensors also consume memory.
The experimental reduced-bandwidth/high-energy-projection mode rejects a cap
until its projection routine has an equivalent verified implementation.

The requested truncation cutoff stays active. When it conflicts with the cap,
the cap wins and the calculation continues. Each density-matrix compression
records its discarded weight, cap hits, cutoff violations, and the worst
operation/bond/recurrence iteration. Reaching the cap does **not** necessarily
mean the cutoff was exceeded. Conversely, a truncation cutoff is not a final
spectral convergence criterion or an error bound on chi.

The measured discarded weights are persisted with recurrence checkpoints and
final task files. Compact `task_timings.tsv` adds per-task diagnostic columns;
`collection_report.txt` contains the detailed diagnostics and policy history.
The result viewer warns when any task reached the cap.

On the tested instance, choose the `0.4.3-bondcap` Daint execution target
(or `0.4.3-bondcap-local` on localhost). The normal submission form exposes
the limit in Advanced; the inventory-first continuation form has a
`Max bond dim.` field. Existing running jobs are not altered by installing
these separate, pinned Codes.

## Continuation without losing existing work

Use a new, supported `SpinSpectraRestartWorkChain` with the existing ground-state
data and spectral `RemoteData`, and a **new** spectral-parameters Dict. Never edit
the inputs or executable of a running process. Completed old tasks are reused
under their original policy, not relabelled as capped calculations.

When an unfinished old checkpoint adopts a cap, its existing moments are
preserved. If necessary its two recurrence vectors are compressed once before
continuing. The checkpoint records the old/new limits and the exact transition
iteration. This produces an explicitly documented mixed-policy continuation;
it is not equivalent to recomputing the entire task with a cap from iteration 1.
Legacy checkpoints have no historical discarded-weight measurements: absence
of telemetry must not be interpreted as zero error.

## Validation

`validation/test_bond_dimension_control.jl` compares capped operations with the
native ITensorMPS density-matrix operations, for dense and QN-conserving MPS,
checks cutoff violations under a deliberately small cap, and checks both capped
restart equivalence and introduction of a cap into an old uncapped checkpoint.
The uncapped path still calls the original ITensor routines directly.

The instrumented capped density-matrix operations follow ITensorMPS 0.4.1;
upstream Apache-2.0 licensing is preserved alongside the adapted source.

An extreme cap of 1 on the three-spin QN validation case exposes a native
ITensorMPS empty-QN compression error in one transverse channel. The same
operation fails with native `apply`; this must not be replaced by an invented
zero spectrum. A cap of 2 is used for the workflow-level L3 check, while the
six-spin tests deliberately exercise cutoff violations with a cap of 2.
