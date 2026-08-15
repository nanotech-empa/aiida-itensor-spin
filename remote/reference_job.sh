#!/bin/bash -l
#SBATCH --account=lp83
#SBATCH --hint=nomultithread
#SBATCH --ntasks-per-core=1
#SBATCH --no-requeue
#SBATCH --job-name=chi-l11-reference
#SBATCH --output=generic-%j.out
#SBATCH --error=generic-%j.err
#SBATCH --uenv=julia/26.3:v1
#SBATCH --view=juliaup
#SBATCH --repo=/capstor/scratch/cscs/cpi/goncalo/.uenv-images

set -euo pipefail
: "${NODES:?NODES is required}"
: "${PPN:?PPN (processes per node) is required}"
: "${T:?T (threads per process) is required}"
: "${PRECOMPUTED_CHECKPOINT:?Set PRECOMPUTED_CHECKPOINT to a GS checkpoint}"
: "${CASE_ROOT:?CASE_ROOT is required}"

P=$((NODES * PPN))
RUN_LABEL=${RUN_LABEL:-n${NODES}_ppn${PPN}_t${T}}
BACKEND=${BACKEND:-full}
THREADING_MODE=${THREADING_MODE:-outer}
REFERENCE_RESULT=${REFERENCE_RESULT:-}
REFERENCE_TOLERANCE=${REFERENCE_TOLERANCE:-1e-10}
ENV_ROOT=${CASE_ROOT}/ITensor_env
JULIA_BIN=/capstor/scratch/cscs/cpi/goncalo/.julia/gh200/juliaup/bin/julia
CHECKPOINT=$(readlink -f "${PRECOMPUTED_CHECKPOINT}")
RESULT_ROOT=${CASE_ROOT}/results/${RUN_LABEL}/job_${SLURM_JOB_ID}

[[ ${BACKEND} == full ]] || {
    echo "Reference job requires BACKEND=full, got: ${BACKEND}" >&2
    exit 2
}
[[ ${THREADING_MODE} == outer || ${THREADING_MODE} == blocksparse ]] || {
    echo "THREADING_MODE must be outer or blocksparse, got: ${THREADING_MODE}" >&2
    exit 2
}
((NODES >= 1 && PPN >= 1 && T >= 1)) || {
    echo "Require NODES, PPN and T >= 1" >&2
    exit 2
}
((PPN * T <= 288)) || {
    echo "Per-node layout uses $((PPN * T)) cores; maximum is 288" >&2
    exit 2
}
((SLURM_NNODES == NODES && SLURM_NTASKS == P && SLURM_CPUS_PER_TASK == T)) || {
    echo "Allocation does not match NODES=${NODES}, P=${P}, T=${T}" >&2
    exit 2
}
[[ -r "${CHECKPOINT}" ]] || {
    echo "Cannot read GS checkpoint: ${CHECKPOINT}" >&2
    exit 2
}

export JULIA_DEPOT_PATH=/capstor/scratch/cscs/cpi/goncalo/.julia/depot
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export MPICH_GPU_SUPPORT_ENABLED=0
ulimit -s unlimited

checkpoint_info=$(
    "${JULIA_BIN}" --startup-file=no --project="${ENV_ROOT}" \
        "${CASE_ROOT}/checkpoint_info.jl" "${CHECKPOINT}"
)
read -r L NSTATES TOTAL_TASKS <<<"${checkpoint_info}"
[[ ${L} =~ ^[0-9]+$ && ${NSTATES} =~ ^[0-9]+$ && ${TOTAL_TASKS} =~ ^[0-9]+$ ]] || {
    echo "Could not parse checkpoint information: ${checkpoint_info}" >&2
    exit 2
}
((P <= TOTAL_TASKS)) || {
    echo "Requested ${P} workers but checkpoint has only ${TOTAL_TASKS} tasks" >&2
    exit 2
}

mkdir -p "${RESULT_ROOT}/workers" "${RESULT_ROOT}/merged"
for ((rank=0; rank<P; rank++)); do
    mkdir -p "${RESULT_ROOT}/workers/worker_${rank}"
done
pipeline_start=$SECONDS

echo "Checkpoint: ${CHECKPOINT}"
echo "Problem: L=${L}, states=${NSTATES}, tasks=${TOTAL_TASKS}"
echo "Layout: ${NODES} nodes, ${PPN} workers/node, ${T} threads/worker"
echo "Backend: ${BACKEND}"
echo "Moment convention: chebyshev-2n-minus-1-v2"
echo "Threading mode: ${THREADING_MODE}"
srun --exclusive --exact \
    --nodes="${NODES}" --ntasks="${P}" --ntasks-per-node="${PPN}" \
    --cpus-per-task="${T}" --cpu-bind=cores --distribution=block:block \
    --output="${RESULT_ROOT}/workers/worker_%t/stdout.txt" \
    --error="${RESULT_ROOT}/workers/worker_%t/stderr.txt" \
    "${CASE_ROOT}/run_reference_rank.sh" \
    "${CHECKPOINT}" "${RESULT_ROOT}/workers" "${CASE_ROOT}" "${BACKEND}" \
    "${THREADING_MODE}"

collect_arguments=(
    --input-root "${RESULT_ROOT}/workers"
    --output "${RESULT_ROOT}/merged"
    --checkpoint "${CHECKPOINT}"
    --backend "${BACKEND}"
    --threading-mode "${THREADING_MODE}"
    --reference-tolerance "${REFERENCE_TOLERANCE}"
)
if [[ -n "${REFERENCE_RESULT}" ]]; then
    collect_arguments+=(--reference "${REFERENCE_RESULT}")
fi

echo "Collecting and validating task results"
srun --exclusive --exact --nodes=1 --ntasks=1 --cpus-per-task=1 \
    --cpu-bind=cores env JULIA_NUM_THREADS=1,0 \
    "${JULIA_BIN}" --startup-file=no --project="${ENV_ROOT}" \
    "${CASE_ROOT}/collect_tasks.jl" "${collect_arguments[@]}" \
    >"${RESULT_ROOT}/merged/stdout.txt" \
    2>"${RESULT_ROOT}/merged/stderr.txt"

pipeline_seconds=$((SECONDS - pipeline_start))
{
    echo "slurm_job_id=${SLURM_JOB_ID}"
    echo "nodes=${NODES}"
    echo "processes_per_node=${PPN}"
    echo "processes=${P}"
    echo "threads_per_process=${T}"
    echo "allocated_worker_cores=$((P * T))"
    echo "L=${L}"
    echo "states=${NSTATES}"
    echo "channels=3"
    echo "tasks=${TOTAL_TASKS}"
    echo "backend=${BACKEND}"
    echo "moment_convention=chebyshev-2n-minus-1-v2"
    echo "threading_mode=${THREADING_MODE}"
    echo "checkpoint=${CHECKPOINT}"
    echo "pipeline_seconds=${pipeline_seconds}"
    echo "result=${RESULT_ROOT}/merged/chi_L${L}.jld2"
} >"${RESULT_ROOT}/layout.txt"

echo "Complete result: ${RESULT_ROOT}/merged/chi_L${L}.jld2"
echo "Pipeline seconds: ${pipeline_seconds}"
