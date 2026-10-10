# Single-step kernel benchmarks

These isolated experiments measure the existing production Chebyshev step.
They do not change deployed engines, AiiDA processes, numerical cutoffs,
energy windows, or bond caps. Each repeat starts from the same two vectors:
`2 * apply(H, current) - previous`. Timers separate MPO application, MPS
addition, moments, and a representative JLD2 write. One full warmup excludes
compilation from three measured repeats. Loading, warmup and write timing
remain available separately.

## Reference input

The physical reference is the central `zz` task 34 from L23, at vector 306 of
326, current maximum bond 915, cutoff 1e-6 and no bond cap. The source segment
UUID is `7b8e354b-f9b6-4ea7-b0bb-8d512a5ebb01`. It shares the exact saved GS,
recurrence vectors and moment convention across configurations. Fixture
preparation checks the GS hash recorded in the checkpoint. This is an
implementation equivalence and timing test, not a new spectrum or an
independent physical reproduction.

```bash
export JULIA_DEPOT_PATH=/path/to/existing/depot
julia --startup-file=no --threads=1,0 --project=remote/engine/ITensor_env \
  validation/benchmark_kernels.jl --action prepare \
  --ground-state /path/to/gs_checkpoint.jld2 \
  --progress /path/to/progress_task_000034_s0001_i00012_czz.jld2 \
  --source-uuid 7b8e354b-f9b6-4ea7-b0bb-8d512a5ebb01 \
  --fixture /path/to/l23_task34_fixture.jld2
```

An optional synthetic mixed-spin QN fixture (`--length 12 --bond 32`, omitting
GS/progress arguments) exercises the runner but supplies no physical or
large-bond performance evidence.

## Thread comparison

The serial reference uses one Julia, BLAS and Strided thread. Block threading
uses 2/4/8 Julia threads with BLAS and Strided held at one. BLAS threading uses
one Julia thread and 2/4/8 BLAS threads, with block threading disabled. All
configurations run sequentially in fresh processes, avoiding nested threading
and competition between measured jobs.

```bash
python tools/run_kernel_benchmarks.py --julia /path/to/julia \
  --depot /path/to/existing/depot --fixture /path/to/l23_task34_fixture.jld2 \
  --output /path/to/results --threads 1,2,4,8 --repeats 3
```

Python >=3.11 is required by this driver. `--resume` reuses completed
measurements with matching fixture hash, thread settings and repeat count,
then completes missing comparisons. It rejects a differing recorded script
hash and reports legacy results without that metadata explicitly. The first
local serial/two-thread measurements predate addition of script-hash metadata;
the measured kernel was unchanged. Preserve that qualification with results.

The Julia-only driver runs the same grid on clusters with older system Python:

```bash
julia --startup-file=no --threads=1,0 --project=remote/engine/ITensor_env \
  tools/run_kernel_benchmarks.jl --fixture /path/to/l23_task34_fixture.jld2 \
  --output /path/to/results --threads 1,2,4,8 --repeats 3
```

Each configuration preserves its next vector and moments. Comparisons report
the relative vector norm `norm(reference-candidate)/norm(reference)` and
maximum absolute difference of the two moments. They report residuals rather
than silently asserting an unspecified scientific acceptance tolerance.
The checkpoint-write output carries `benchmark_only=true` and is not a
supported production restart checkpoint.

## Daint allocation

Copy the scripts, included engine modules, pinned project/manifest and fixture
to an isolated scratch directory; include `SHA256SUMS` for all these inputs.
Submit `validation/kernel_benchmark_daint.sh` **from that directory**. It uses
one debug node, eight CPU cores, 32 GB requested memory, a 30-minute limit and
the existing Julia 1.12.6/uenv/depot. All comparisons run sequentially within
the same allocation and CPU binding. Record the source commit and hashes.
No packages are installed and no production runtime files are replaced.

Results from Docker Linux on Apple M2 and Daint must be reported separately.
L23 scaling is not a speedup prediction for the capped L50 or uncapped L101
production tasks. A representative large-bond checkpoint is needed for that.

`kernel_benchmark_daint_large.sh` provides that separate test: the central
L50 task 226 (`state 2, site 26, zz`) from completed segment UUID
`44ed04b9-1589-4781-ad8b-c81f83baef5c`, with cutoff/cap taken from the saved
checkpoint. It prepares the fixture on Daint, retaining source hashes, rather
than downloading the roughly 4.7 GB progress file to the Mac. It requests one
normal node, eight cores, 128 GB memory and at most 12 hours. The first
configuration is the existing two-thread block mode; eight-thread blocks and
eight-thread BLAS follow, then four-thread blocks and serial. Here speedups
use `reference_label` and `speedup_vs_reference`, relative to two threads.

One measured real step per configuration keeps the initial diagnostic cost
bounded. A separate small QN fixture warms the same capped code path;
`warmup_fixture_sha256` and `warmup_uses_measured_fixture=false` identify this
method explicitly. This can leave size-dependent first-use effects and lacks
the repeat variability estimate of the three-repeat L23 test. Strong
performance conclusions need repeated representative measurements. Numerical
comparisons are untimed and may themselves be expensive at these bond sizes.

## Repeated BLAS confirmation and concurrent workers

The completed initial grid found BLAS 8 fastest on the saved L50 step:
10.192 minutes versus 36.087 for block 2, with relative vector residual
5.804e-15. Evidence is in `results/kernel-benchmarks/2026-10-10-daint-l50-complete/`.
`kernel_benchmark_daint_confirm.sh` tests whether that result survives a full
physical warmup, repetition, and competition between six workers.

It requests one normal node, six tasks with eight CPU cores each, 384 GB
memory and an eight-hour maximum. Each step requests 64 GB. The conservative
memory request leaves headroom for measurement; it is not a production
memory recommendation. [Slurm's step resource controls](https://slurm.schedmd.com/srun.html)
provide disjoint resources through `--exact`/`--exclusive`; each worker's
logical CPU affinity is also recorded and checked for overlap.

Stages are: a small concurrent smoke test; isolated BLAS 8 with full physical
warmup and three repetitions; six simultaneous BLAS 8 workers with two
repetitions each; then the existing block-2 reference with full physical
warmup and three repetitions. All repetitions start from the same saved
vectors rather than advancing the recurrence. The existing large fixture,
small fixture and previous reference remain read-only in the original
deployment and are verified against `SOURCE_INPUT_SHA256SUMS`.

Concurrent workers publish readiness only after completing full warmup.
A shared release marker starts the measurement phase together. Load/warmup
and barrier waits are excluded. Kernel times remain separate from the
measurement wall window, which includes GC and representative writes.
Moment residuals are checked across all six workers; full vector comparisons
are performed for workers 1 and 6. The comparison itself uses BLAS 8 and
records that choice, with the same subtraction formula/cutoff as before.

`summarize_concurrent_kernels.jl` reports each worker's slowdown, CPU affinity,
peak process RSS and aggregate step rate. Peak RSS is recorded before the
final result export, and summing individual peaks does not measure
simultaneous node usage; use Slurm accounting as additional evidence.
Six identical replicas provide a contention diagnostic. They do not represent
the mixture of different channel/state checkpoints in production or establish
an accepted whole-node memory limit. `summarize_kernel_confirmation.jl` combines
the repeated comparison and the concurrent diagnostics. No deployed engine,
production WorkChain, cutoff, bond cap, or energy window is changed.
