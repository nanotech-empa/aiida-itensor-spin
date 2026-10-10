# Completed L50 large-bond kernel comparison on Daint

The isolated benchmark job 5015465 completed successfully (Slurm exit 0:0) on October 10, 2026 at 02:08:33 Europe/Rome, after 3:59:44 total allocation time. All five configurations and their numerical comparisons completed; stderr was empty. This final snapshot supersedes the October 9 partial snapshot, which is retained separately.

All modes share the same fixture and script hashes. Physical reference: L50 central task 226, state 2, site 26, zz; saved vector 325/720; max current bond 7737; cutoff 1e-8; cap 8000. Source segment UUID `44ed04b9-1589-4781-ad8b-c81f83baef5c`. Julia 1.12.6, ITensors 0.9.30, ITensorMPS 0.4.1, Linux aarch64, OpenBLAS ILP64. Code deployment commit `1ad9c3a6e2f06d8a897ec989ef8a78a6e04d46ee`.

One isolated node, one process bound to eight cores, sequential fresh processes. Strided is fixed at one thread. Block modes use BLAS=1; BLAS mode uses Julia=1 and disables threaded blocks. Timed operations and stored compression controls are the existing production ones.

| Mode | Julia / BLAS threads | Apply (min) | Add (min) | Moments (min) | Kernel (min) | Speedup vs block 2 | Relative vector residual |
|---|---:|---:|---:|---:|---:|---:|---:|
| blocks | 2 / 1 | 23.180 | 12.081 | 0.825 | 36.087 | 1.000 | 0 |
| blocks | 8 / 1 | 17.661 | 11.044 | 0.510 | 29.215 | 1.235 | 0 |
| blas | 1 / 8 | 7.192 | 2.793 | 0.207 | 10.192 | 3.541 | 5.8e-15 |
| blocks | 4 / 1 | 20.990 | 11.703 | 0.656 | 33.349 | 1.082 | 0 |
| serial | 1 / 1 | 37.834 | 14.674 | 1.345 | 53.853 | 0.670 | 8.45e-15 |

Eight BLAS threads were fastest among these tested modes: 611.495 seconds versus 2165.198 seconds in the existing two-block-thread mode (3.541x, 71.76% less kernel time). Both apply and addition improve substantially. All outputs have max bond 7659. BLAS 8 differs from the reference by 5.804e-15 in relative vector norm and 1.998e-15 in absolute moments; maximum candidate vector residual across the entire grid is 8.449e-15. The residual is measured with MPS subtraction at cutoff 1e-14. These are implementation comparisons, not final-spectrum error guarantees or proofs of bitwise identity.

Representative checkpoint writes take 2.4–2.7 seconds. Loading and approximately 44 seconds of small-fixture warmup are separately recorded. The warning about bond cap 32 in each log belongs to that synthetic warmup; the measured physical step uses cap 8000 from the real checkpoint. Slurm reported MaxRSS 38957.50M for the driver step, not a complete memory budget for concurrent production workers.

Each mode measures one real step after a small QN warmup of the same capped implementation. This lacks repeat variability and may retain size-dependent first-use effects. Repeat the promising BLAS mode with full physical warmup and test concurrent-worker memory/throughput before adopting a production policy. No claim of optimal BLAS thread count, L101 speedup, or finished L50 spectrum follows from this test.

The existing production L50 still uses two block threads. At approximately 08:09 Europe/Rome on October 10 its current job 5015469 was RUNNING on two nodes, with 288/300 tasks complete. Latest worker logs showed the remaining 12 central tasks at 337–358/720, with recent steps taking 43–55 minutes. Several reached cap 8000; an observed discarded weight 1.4856580910430146e-8 exceeded the requested cutoff 1e-8, so the cap can bind and relax the requested truncation target. This is a recorded numerical limitation, not a final spectrum error estimate.

The compact results and logs are stored here; large fixtures/vector outputs remain under `/ritom/scratch/cscs/cpi/goncalo/kernel_benchmark_20261009_1ad9c3a/results-5015465/` on Daint. Production inputs, deployed engines and immutable AiiDA provenance were not changed.
