#!/bin/bash
# Used inside the benchmark allocation; starts only its own Slurm steps.
set -euo pipefail
BENCH_FIXTURE=${1:?fixture}
BENCH_OUTPUT=${2:?output directory}
BENCH_REFERENCE=${3:?isolated timing reference}
BENCH_REPEATS=${4:-2}
BENCH_WORKERS=${5:-6}
BENCH_ROOT=${BENCH_ROOT:?}
BENCH_JULIA=${BENCH_JULIA:?}
mkdir "$BENCH_OUTPUT"
BENCH_BARRIER="$BENCH_OUTPUT/barrier"
mkdir "$BENCH_BARRIER"
BENCH_PIDS=()
cleanup() {
    local pid
    for pid in $(jobs -pr); do kill "$pid" 2>/dev/null || true; done
}
trap cleanup EXIT
trap 'cleanup; exit 130' INT
trap 'cleanup; exit 143' TERM
for ((BENCH_ID=1; BENCH_ID<=BENCH_WORKERS; BENCH_ID++)); do
    srun --exclusive --exact --ntasks=1 --cpus-per-task=8 --mem=64G --cpu-bind=cores \
        "$BENCH_JULIA" --startup-file=no --threads=1,0 \
        --project="$BENCH_ROOT/remote/engine/ITensor_env" \
        "$BENCH_ROOT/validation/benchmark_kernels.jl" --action run \
        --fixture "$BENCH_FIXTURE" --mode blas --blas-threads 8 --repeats "$BENCH_REPEATS" \
        --output "$BENCH_OUTPUT/worker_$BENCH_ID" \
        --barrier-directory "$BENCH_BARRIER" --barrier-id "$BENCH_ID" --barrier-workers "$BENCH_WORKERS" \
        > "$BENCH_OUTPUT/worker_$BENCH_ID.log" 2>&1 &
    BENCH_PIDS+=("$!")
done
shopt -s nullglob
BENCH_WAIT_STARTED=$SECONDS
while true; do
    BENCH_READY=("$BENCH_BARRIER"/ready-*)
    [[ ${#BENCH_READY[@]} -eq $BENCH_WORKERS ]] && break
    for BENCH_PID in "${BENCH_PIDS[@]}"; do
        kill -0 "$BENCH_PID" 2>/dev/null || { echo "A worker exited before barrier release" >&2; exit 1; }
    done
    ((SECONDS - BENCH_WAIT_STARTED < 7200)) || { echo "Worker readiness timed out" >&2; exit 1; }
    sleep 2
done
date -Is > "$BENCH_BARRIER/release"
for BENCH_PID in "${BENCH_PIDS[@]}"; do wait "$BENCH_PID"; done
srun --exclusive --exact --ntasks=1 --cpus-per-task=8 --mem=64G --cpu-bind=cores \
    "$BENCH_JULIA" --startup-file=no --threads=1,0 \
    "$BENCH_ROOT/tools/summarize_concurrent_kernels.jl" \
    --directory "$BENCH_OUTPUT" --workers "$BENCH_WORKERS" \
    --reference "$BENCH_REFERENCE" --output "$BENCH_OUTPUT/summary.toml"
