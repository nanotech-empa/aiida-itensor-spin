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
