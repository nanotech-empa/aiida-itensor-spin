# L23 checkpoint kernel timings on Docker Linux / Apple M2

Docker allocation: 8 visible CPU cores, 32 GB RAM; host Mac M2 has 64 GB. Julia 1.12.6, ITensors 0.9.30, ITensorMPS 0.4.1. Configurations ran sequentially in fresh processes. One full physical warmup precedes three timed repetitions, each starting from identical saved vectors.

Physical input: L23 central zz task 34, vector 306/326, maximum current bond 915, cutoff 1e-6, no cap. Source segment UUID `7b8e354b-f9b6-4ea7-b0bb-8d512a5ebb01`. Input source and fixture hashes are recorded in each timing file.

| Mode | Julia / BLAS threads | Apply (s) | Add (s) | Kernel (s) | Speedup vs serial | Relative vector error |
|---|---:|---:|---:|---:|---:|---:|
| serial | 1 / 1 | 4.786 | 1.453 | 6.476 | 1.000 | 0 |
| blocks | 2 / 1 | 3.268 | 1.357 | 4.694 | 1.380 | 3.7e-15 |
| blocks | 4 / 1 | 2.680 | 1.238 | 4.009 | 1.615 | 3.7e-15 |
| blocks | 8 / 1 | 2.140 | 1.191 | 3.429 | 1.888 | 3.7e-15 |
| blas | 1 / 2 | 3.715 | 1.205 | 5.053 | 1.281 | 3.83e-15 |
| blas | 1 / 4 | 3.477 | 1.210 | 4.555 | 1.422 | 2.79e-15 |
| blas | 1 / 8 | 3.226 | 1.229 | 4.492 | 1.442 | 3.19e-15 |

Kernel includes apply, add and both moments; write, load and compilation are excluded. Each table entry is the median of that timer, so medians of components need not sum exactly. Relative vector error is `norm(reference-candidate)/norm(reference)`, evaluated with explicit MPS subtraction at cutoff 1e-14; it is not a pointwise spectral error. The maximum absolute moment difference across candidates is 1.67e-15.

Eight block threads improved this case by 1.369x relative to the existing two-thread mode. The largest gain is in apply; addition improves less. This supports testing large-bond production checkpoints, but does not establish scaling for L50/L101 or optimal node throughput across multiple workers.

The local controller was interrupted when the user reported battery use, then resumed after the charger was found. Completed measurements were reused; the interrupted four-thread numerical comparison was rerun. Serial and two-thread measurements predate script-hash metadata. Four-thread, eight-thread and BLAS 2/4 used script hash `c3f3be5d089bea68ff4437b59e8f42fdb5abf940bf8d780a8685b4ee96a0d9a0` (source in commit `81b4b8e`). BLAS 8 used `40042dc77f7adf2e9ddb2451fd7724412991afc9ce217b0344bd8a8491364f3d` (commit `1ad9c3a`); the added alternate warmup option was not used and the measured kernel remained unchanged. Preserve this provenance qualification.

Compact timings, logs, moment values and comparison residuals are archived here. Large input/output JLD2 files remain outside Git at `/home/jovyan/work/kernel_benchmark_20261009/`; the source checkpoint remains on Daint. These benchmark outputs are not production restart files or a completed spectrum.

Large-bond follow-up: isolated Daint deployment `/ritom/scratch/cscs/cpi/goncalo/kernel_benchmark_20261009_1ad9c3a`, code commit `1ad9c3a6e2f06d8a897ec989ef8a78a6e04d46ee`, source L50 segment UUID `44ed04b9-1589-4781-ad8b-c81f83baef5c`, central task 226. Scheduler job 5015465 was submitted; results were not yet available when this archive was created.
