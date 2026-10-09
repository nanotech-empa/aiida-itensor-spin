#!/bin/bash -l
# One isolated allocation. Tests run sequentially on the same eight-core set.
#SBATCH --job-name=itensor-kernels
#SBATCH --account=lp83
#SBATCH --partition=debug
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --hint=nomultithread
#SBATCH --time=00:30:00
#SBATCH --mem=32G
#SBATCH --no-requeue
#SBATCH --exclusive
#SBATCH --output=benchmark-%j.out
#SBATCH --error=benchmark-%j.err
#SBATCH --uenv=julia/26.3:v1
#SBATCH --view=juliaup
#SBATCH --repo=/ritom/scratch/cscs/cpi/goncalo/l101_maintenance_20260915/runtime/uenv-repo
set -euo pipefail
BENCH_ROOT=${SLURM_SUBMIT_DIR:?submit from the isolated benchmark directory}
BENCH_JULIA=/ritom/scratch/cscs/cpi/goncalo/l101_maintenance_20260915/runtime/julia-1.12.6/bin/julia
export JULIA_DEPOT_PATH=/ritom/scratch/cscs/cpi/goncalo/updatespin_l23_validation_20260917/depot:/ritom/scratch/cscs/cpi/goncalo/l101_maintenance_20260915/runtime/depot
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
ulimit -s unlimited
sha256sum -c SHA256SUMS
srun --ntasks=1 --cpus-per-task=8 --cpu-bind=cores \
    "$BENCH_JULIA" --startup-file=no --threads=1,0 \
    --project="$BENCH_ROOT/remote/engine/ITensor_env" \
    "$BENCH_ROOT/tools/run_kernel_benchmarks.jl" \
    --julia "$BENCH_JULIA" --fixture "$BENCH_ROOT/l23_task34_fixture.jld2" \
    --output "$BENCH_ROOT/results-$SLURM_JOB_ID" --threads 1,2,4,8 --repeats 3
