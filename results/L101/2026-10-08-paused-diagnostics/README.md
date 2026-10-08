# L101 paused-run diagnostics — 2026-10-08

This snapshot preserves the saved-moment analysis and provisional broader
spectra discussed before review with Gonçalo. It is not a successful completion
of the production WorkChain or a final scientific accuracy claim.

The restart WorkChain is `442684f2-65b4-486b-8fbd-f9fdd31669a0`, with adaptive
child `bfa71143-b8c7-4638-8d50-35dded5d24de`. They were initially paused and
their active allocation cancelled. Later on October 8 the user explicitly
requested permanent termination: both WorkChains and their latest CalcJob
are now **Killed**. See `stop_record.json`. Any future continuation would
require a new explicit request and a new WorkChain using supported checkpoint
inputs; these terminal processes cannot continue. L50 remains active.

Last successful segment: `4b4117de-7c29-4a5b-a1fe-78f438c854eb`.
RemoteData: `9058e3b7-9381-4cb4-979d-1b9f6aefc819`.
Ground state: `9d941100-9de6-4492-8da5-5d6cd168359c`.
The complete ground state and large tensor checkpoints remain on Ritom;
this Git snapshot contains their compact moments and scalar metadata.

## Contents

- `all_moments.jld2`: moments for all 909 tasks. There are 891 completed
  recurrence tasks and 18 unfinished tasks at sites 51–52. Uncomputed tails
  are zero-filled in this export and excluded using `completed_vectors`.
- `reference_checks.jld2`: six stored completed-task spectra and the audit of
  all checkpoint ground-state identities and cutoffs.
- `l101_3mev_provisional/`: whole-chain plots, central-site comparisons,
  TSV/NPZ spectra and the quantitative report, with three initial states kept
  separate and channel sum `zz + 0.5*pm + 0.5*mp`.
- `export_*.jl`: read-only exporters for the remote checkpoint directory.
  They read moment/scalar datasets without loading the large saved MPS vectors.
- `analyze_partial_moments.py` and `reconstruct_l101_3mev.py`: local diagnostic
  reconstructions. Paths are relative to this snapshot for local reconstruction.
- `validate_filter.jl`, its log/checkpoints and `production_engine/`: the
  isolated six-site exact-diagonalization comparison of the production library's
  full-bandwidth and energy-filtered modes.
- `pause_record.json`, `stop_record.json`, `publication_validation.json`, and `git-audit/`:
  historical operational and source/deployment records. Scheduler IDs and
  local paths describe this instance and are not portable launch configuration.

## What “3 meV” means here

The reconstruction uses fixed Gaussian damping in Chebyshev angle. Its
approximate Gaussian **standard deviation** is 3 meV at 100 meV and 1.419 meV
at zero. It is not uniform energy broadening, not FWHM, and not a new run of
the original Jackson algorithm with `omega_smearing_meV=3`.

The maximum combined-site relative L2 difference between 80% and all saved
orders is 0.0984%. This checks order sensitivity; it does not bound truncation
error or prove scientific convergence. The original 1474-vector target remains
unfinished in 18 tasks, and no new recurrence iterations were computed.

The provisional data retain 2146 negative combined samples, with minimum
-4.04924e-4, all outside the unfinished central sites. The log-colour heatmap
masks non-positive values white; these are not missing data. Their cause is
unresolved and requires review of the saved moments and reconstruction.

## Reproduction and validation

With Python, NumPy, h5py, Matplotlib and threadpoolctl installed:

```bash
python reconstruct_l101_3mev.py
python analyze_partial_moments.py
sha256sum -c SHA256SUMS
```

Checksums describe this archived snapshot; regenerating plots can change their
binary metadata without changing the numerical result. Six original Jackson
reconstructions matched stored completed-task outputs to maximum absolute
difference 5.13e-14. All 909 exported checkpoints used the same GS hash and
spectral cutoff 1e-8. The separate small-chain filtering test matched its own
exact reference, but its cost and changed broadening do not establish an L101
speedup or an equivalent production replacement.

Prepared with Codex assistance. The physical interpretation and acceptance
of these provisional results remain for discussion with Gonçalo.
