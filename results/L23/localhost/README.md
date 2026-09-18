# L23 localhost ARM64 result — 18 September 2026

This directory contains the compact, reproducible outputs of the complete
AiiDA WorkChain documented in the [parent report](../README.md). The run used
one Linux ARM64 localhost node, four spectral processes, two threads/process,
and the versioned `spin1_chain_spinhalf_impurity_v1` model.

- `ground_state.json`, `gs_status.json`, `gs_summary.txt`: resolved GS inputs
  and summary; the large GS checkpoint is intentionally not stored in Git.
- `segment_status.json`, `segment_summary.json`: final progress and actual
  resource layout.
- `chi_L23.jld2`, `spectrum.tsv`: compact Julia and readable spectral results.
- `collection_report.txt`, `task_timings.tsv`: collection validation and
  per-task performance data.
- `comparison.toml`, `comparison_by_site.tsv`: quantitative comparison with
  Gonçalo Catarina's supplied reference.

The reference and this run did not share a GS checkpoint, and the reference
seed is unknown. Treat the comparison as an independent physics reproduction,
not a bitwise-equivalence test.
