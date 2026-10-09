# Partial large-bond L50 benchmark on Daint

Job 5015465 was RUNNING when these results were collected at approximately 23:40 Europe/Rome on October 9, 2026. Block modes with 2 and 8 threads completed; BLAS 8 was running, with block 4 and serial still scheduled. This is a partial snapshot, not the final grid.

Input: central L50 task 226 (state 2, site 26, zz), saved at vector 325/720. Source segment UUID `44ed04b9-1589-4781-ad8b-c81f83baef5c`. Current max bond 7737, cutoff 1e-8, cap 8000. Both outputs have max bond 7659 and max discarded weight 9.99896807434797e-9. This discarded weight is not a final-spectrum error bound.

Daint Linux aarch64, Julia 1.12.6, ITensors 0.9.30, ITensorMPS 0.4.1, OpenBLAS ILP64. Both modes use BLAS and Strided at one thread. Same isolated node and eight-core CPU binding; fresh processes execute sequentially. Deployment code commit `1ad9c3a6e2f06d8a897ec989ef8a78a6e04d46ee`.

| Julia block threads | Apply H (min) | MPS add (min) | Moments (min) | Kernel (min) |
|---:|---:|---:|---:|---:|
| 2 | 23.180 | 12.081 | 0.825 | 36.087 |
| 8 | 17.661 | 11.044 | 0.510 | 29.215 |

Eight block threads give 1.235223x speedup versus two, or 19.04% less kernel time, using four times the threads. Most improvement is in apply; addition remains a substantial cost. Representative write time is about 2.5 seconds, excluded from kernel time.

The comparison reported relative vector norm difference 0.0 and maximum absolute moment difference 0.0. The vector metric uses explicit MPS subtraction with cutoff 1e-14; zero means no difference detected by this metric, not an independently demonstrated bitwise identity or a final observable error guarantee.

Each mode measures only one real step after a small QN fixture warms the same capped code path. Size-dependent first-use effects and variability are not excluded. Compare modes on this input; do not infer a general scaling law, L101 runtime, or optimal whole-node throughput. Different L23/Mac hardware and inputs prevent isolating bond size as the sole cause of differing speedups.

Raw compact TOML files preserve source/fixture hashes, timings and moments. Large fixture/vector outputs remain in the isolated Daint directory `/ritom/scratch/cscs/cpi/goncalo/kernel_benchmark_20261009_1ad9c3a/results-5015465/`; these are experimental outputs, not a production continuation. Retrieve the remaining modes before choosing a production policy.
