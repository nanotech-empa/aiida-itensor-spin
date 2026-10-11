# L50 full-warmup and concurrent kernel confirmation

Retrieved 2026-10-11. Slurm job 5019606 completed, exit 0:0, elapsed 04:08:32.
Source commit: 558e1df497605d1a8fe8d15695818ef18dfc5955 (perf/single-task-kernels).
Deployment: /ritom/scratch/cscs/cpi/goncalo/kernel_confirm_20261010_558e1df.
Fixture SHA-256: 88ecbb4bd1aa116c593641ac6e536c7faa01320cfb7725845af3d536804808b1.
Large JLD2 files and transient barriers are excluded; TOML results, logs and
original deployment manifests are retained. SOURCE_INPUT_SHA256SUMS describes
remote fixtures; the deployment SHA256SUMS describes source files there.
ARCHIVE_SHA256SUMS verifies files actually included here.

## Measured result

After full physical warmup, three repetitions of the same saved recurrence
step gave median kernel time 2166.436 s for block-sparse Julia 2 / BLAS 1,
and 599.768 s for Julia 1 / BLAS 8: 3.612x speedup. Full vector relative
residual is 5.742e-15 and maximum absolute moment residual is 1.998e-15.
These repetitions reuse the identical checkpoint, rather than advancing the
recurrence three steps. See results-5019606/summary.toml for all samples.

Six simultaneous BLAS-8 workers, with disjoint eight-core affinities and two
repetitions each, took median 774.295–802.467 s per worker. Their kernel
slowdown versus isolated execution was 1.291–1.338x. Measurement-window
throughput, including GC and representative checkpoint writes, was 26.805
steps/hour, 74.99% of ideal independent-worker throughput. Moments matched
exactly for all six replicas; workers 1 and 6 also matched the isolated full
vector exactly. This is a contention proxy using identical inputs, not the
heterogeneous production workload or a final spectrum validation.

Measured per-worker VmHWM was about 41.8–42.1 GiB before final result export.
The sum of individual peaks is not a simultaneous node peak. The benchmark
allocation requested 384 GiB; it does not validate six workers within the
256 GiB production allocation. Production threading remains unchanged.

## Independently observed L50 production OOM

Adaptive WorkChain UUID db9ae06e-6246-4b65-a611-d58d7a42e264.
Segment UUID 038f41d4-c064-429d-8901-bbb739769684, Slurm job 5024541:
step .0 OUT_OF_MEMORY after 00:31:53; allocation FAILED after 00:32:58.
Step MaxRSS 47447808K is the maximum task RSS, not total node RSS.
Scheduler stderr explicitly identifies rank 2 as Out Of Memory.
Preserved status: 288/300 tasks, 211707/216000 aggregate vectors.

Backoff produced a new allocation ceiling of 7 processes/node, from 10,
but 12 workers remained on 2 nodes. Live host headers confirmed six active
workers per node in both allocations. The memory request stayed 262144 MiB
per node (256 GiB), so this first backoff did not reduce actual worker density.
The next segment UUID 28e6e0e8-f0cc-4f51-ac4f-e7bead6f3e13, job 5028063,
was RUNNING when inspected on 2026-10-11. This is an observation, not an
assurance that its memory problem is solved. A 4-process/node ceiling would
place the 12 remaining workers on 3 nodes and reduce density; it has not been
applied. No stored nodes or production process state were modified.
