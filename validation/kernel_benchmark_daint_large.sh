#!/bin/bash -l
# Actual large-bond L50 checkpoint. One node; configurations run sequentially.
#SBATCH --job-name=itensor-large-bond
#SBATCH --account=lp83
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --hint=nomultithread
#SBATCH --time=12:00:00
#SBATCH --mem=128G
#SBATCH --no-requeue
#SBATCH --exclusive
#SBATCH --output=benchmark-large-%j.out
#SBATCH --error=benchmark-large-%j.err
#SBATCH --uenv=julia/26.3:v1
#SBATCH --view=juliaup
#SBATCH --repo=/ritom/scratch/cscs/cpi/goncalo/l101_maintenance_20260915/runtime/uenv-repo
set -euo pipefail
BENCH_ROOT=${SLURM_SUBMIT_DIR:?submit from the isolated benchmark directory}
BENCH_JULIA=/ritom/scratch/cscs/cpi/goncalo/l101_maintenance_20260915/runtime/julia-1.12.6/bin/julia
SOURCE_ROOT=/ritom/scratch/cscs/cpi/aiida/44/ed/04b9-1589-4781-ad8b-c81f83baef5c
export JULIA_DEPOT_PATH=/ritom/scratch/cscs/cpi/goncalo/updatespin_l23_validation_20260917/depot:/ritom/scratch/cscs/cpi/goncalo/l101_maintenance_20260915/runtime/depot
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
ulimit -s unlimited
sha256sum -c SHA256SUMS
BENCH_RESULTS="$BENCH_ROOT/results-$SLURM_JOB_ID"
mkdir "$BENCH_RESULTS"
# This snapshot comes from a completed segment and is only read.
srun --ntasks=1 --cpus-per-task=8 --cpu-bind=cores \
    "$BENCH_JULIA" --startup-file=no --threads=1,0 \
    --project="$BENCH_ROOT/remote/engine/ITensor_env" \
    "$BENCH_ROOT/validation/benchmark_kernels.jl" --action prepare \
    --ground-state "$SOURCE_ROOT/gs_checkpoint.jld2" \
    --progress "$SOURCE_ROOT/state/workers/worker_0/progress_task_000226_s0002_i00026_czz.jld2" \
    --source-uuid 44ed04b9-1589-4781-ad8b-c81f83baef5c \
    --fixture "$BENCH_RESULTS/l50_task226_fixture.jld2"
# Compile the capped code path on a small QN fixture before costly real steps.
srun --ntasks=1 --cpus-per-task=8 --cpu-bind=cores \
    "$BENCH_JULIA" --startup-file=no --threads=1,0 \
    --project="$BENCH_ROOT/remote/engine/ITensor_env" \
    "$BENCH_ROOT/validation/benchmark_kernels.jl" --action prepare \
    --length 12 --bond 32 --fixture "$BENCH_RESULTS/smoke_fixture.jld2"
# Smoke-test the Julia-only driver, including numerical comparisons.
srun --ntasks=1 --cpus-per-task=8 --cpu-bind=cores \
    "$BENCH_JULIA" --startup-file=no --threads=1,0 \
    --project="$BENCH_ROOT/remote/engine/ITensor_env" \
    "$BENCH_ROOT/tools/run_kernel_benchmarks.jl" --julia "$BENCH_JULIA" \
    --fixture "$BENCH_RESULTS/smoke_fixture.jld2" \
    --output "$BENCH_RESULTS/smoke" --threads 1,2 --repeats 1
srun --ntasks=1 --cpus-per-task=8 --cpu-bind=cores \
    "$BENCH_JULIA" --startup-file=no --threads=1,0 \
    --project="$BENCH_ROOT/remote/engine/ITensor_env" \
    "$BENCH_ROOT/tools/run_kernel_benchmarks.jl" --julia "$BENCH_JULIA" \
    --fixture "$BENCH_RESULTS/l50_task226_fixture.jld2" \
    --warmup-fixture "$BENCH_RESULTS/smoke_fixture.jld2" \
    --output "$BENCH_RESULTS/physical" \
    --configurations blocks:2:1,blocks:8:1,blas:1:8,blocks:4:1,serial:1:1 --repeats 1
