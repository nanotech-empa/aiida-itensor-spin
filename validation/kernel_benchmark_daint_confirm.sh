#!/bin/bash -l
# Repeated full-size steps and six simultaneous BLAS workers, one allocation.
#SBATCH --job-name=itensor-blas-confirm
#SBATCH --account=lp83
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=6
#SBATCH --cpus-per-task=8
#SBATCH --hint=nomultithread
#SBATCH --time=08:00:00
#SBATCH --mem=384G
#SBATCH --no-requeue
#SBATCH --exclusive
#SBATCH --output=benchmark-confirm-%j.out
#SBATCH --error=benchmark-confirm-%j.err
#SBATCH --uenv=julia/26.3:v1
#SBATCH --view=juliaup
#SBATCH --repo=/ritom/scratch/cscs/cpi/goncalo/l101_maintenance_20260915/runtime/uenv-repo
set -euo pipefail
export BENCH_ROOT=${SLURM_SUBMIT_DIR:?submit from the isolated benchmark directory}
export BENCH_JULIA=/ritom/scratch/cscs/cpi/goncalo/l101_maintenance_20260915/runtime/julia-1.12.6/bin/julia
BENCH_PREVIOUS=/ritom/scratch/cscs/cpi/goncalo/kernel_benchmark_20261009_1ad9c3a/results-5015465
BENCH_FIXTURE="$BENCH_PREVIOUS/l50_task226_fixture.jld2"
BENCH_SMOKE="$BENCH_PREVIOUS/smoke_fixture.jld2"
export JULIA_DEPOT_PATH=/ritom/scratch/cscs/cpi/goncalo/updatespin_l23_validation_20260917/depot:/ritom/scratch/cscs/cpi/goncalo/l101_maintenance_20260915/runtime/depot
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
ulimit -s unlimited
sha256sum -c SHA256SUMS
sha256sum -c SOURCE_INPUT_SHA256SUMS
BENCH_RESULTS="$BENCH_ROOT/results-$SLURM_JOB_ID"
mkdir "$BENCH_RESULTS"
mkdir "$BENCH_RESULTS/isolated"
mkdir "$BENCH_RESULTS/smoke"

run_isolated() {
    local fixture=$1 mode=$2 julia_threads=$3 blas_threads=$4 repeats=$5 target=$6
    srun --exclusive --exact --ntasks=1 --cpus-per-task=8 --mem=64G --cpu-bind=cores \
        "$BENCH_JULIA" --startup-file=no --threads="$julia_threads,0" \
        --project="$BENCH_ROOT/remote/engine/ITensor_env" \
        "$BENCH_ROOT/validation/benchmark_kernels.jl" --action run --fixture "$fixture" \
        --mode "$mode" --blas-threads "$blas_threads" --repeats "$repeats" --output "$target" \
        > "$target.log" 2>&1
}
compare_vectors() {
    local reference=$1 candidate=$2 target=$3
    srun --exclusive --exact --ntasks=1 --cpus-per-task=8 --mem=64G --cpu-bind=cores \
        "$BENCH_JULIA" --startup-file=no --threads=1,0 \
        --project="$BENCH_ROOT/remote/engine/ITensor_env" \
        "$BENCH_ROOT/validation/benchmark_kernels.jl" --action compare \
        --reference "$reference" --candidate "$candidate" --mode blas --blas-threads 8 \
        --output "$target.comparison.toml" > "$target.comparison.log" 2>&1
}

echo "STAGE smoke: barrier, CPU binding, concurrent driver and comparisons"
run_isolated "$BENCH_SMOKE" blas 1 8 1 "$BENCH_RESULTS/smoke/isolated"
bash "$BENCH_ROOT/validation/run_concurrent_kernel_workers.sh" \
    "$BENCH_SMOKE" "$BENCH_RESULTS/smoke/concurrent" "$BENCH_RESULTS/smoke/isolated.toml" 2 6
compare_vectors "$BENCH_RESULTS/smoke/isolated.jld2" \
    "$BENCH_RESULTS/smoke/concurrent/worker_1.jld2" "$BENCH_RESULTS/smoke/vector"

echo "STAGE isolated BLAS8: full physical warmup and three identical-input repeats"
run_isolated "$BENCH_FIXTURE" blas 1 8 3 "$BENCH_RESULTS/isolated/blas8"
compare_vectors "$BENCH_PREVIOUS/physical/blocks_j2_b1.jld2" \
    "$BENCH_RESULTS/isolated/blas8.jld2" "$BENCH_RESULTS/isolated/blas8.vs_previous"

echo "STAGE concurrent BLAS8: six replicas, full warmup, barrier, two repeats each"
bash "$BENCH_ROOT/validation/run_concurrent_kernel_workers.sh" \
    "$BENCH_FIXTURE" "$BENCH_RESULTS/concurrent" "$BENCH_RESULTS/isolated/blas8.toml" 2 6
for BENCH_ID in 1 6; do
    compare_vectors "$BENCH_RESULTS/isolated/blas8.jld2" \
        "$BENCH_RESULTS/concurrent/worker_$BENCH_ID.jld2" "$BENCH_RESULTS/concurrent/worker_$BENCH_ID"
done

echo "STAGE repeated baseline: full physical warmup and three identical-input repeats"
run_isolated "$BENCH_FIXTURE" blocks 2 1 3 "$BENCH_RESULTS/isolated/blocks2"
compare_vectors "$BENCH_RESULTS/isolated/blas8.jld2" \
    "$BENCH_RESULTS/isolated/blocks2.jld2" "$BENCH_RESULTS/isolated/blocks2.vs_blas8"
srun --exclusive --exact --ntasks=1 --cpus-per-task=8 --mem=64G --cpu-bind=cores \
    "$BENCH_JULIA" --startup-file=no --threads=1,0 \
    "$BENCH_ROOT/tools/summarize_kernel_confirmation.jl" --results "$BENCH_RESULTS"
echo "COMPLETE: $BENCH_RESULTS/summary.toml"
