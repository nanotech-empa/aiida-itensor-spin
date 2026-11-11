#!/bin/bash
set -euo pipefail

if (($# != 5)); then
    echo "Usage: $0 CHECKPOINT WORKER_ROOT CODE_ROOT BACKEND THREADING_MODE" >&2
    exit 2
fi

CHECKPOINT=$1
WORKER_ROOT=$2
CODE_ROOT=$3
BACKEND=$4
THREADING_MODE=$5
RANK=${SLURM_PROCID:?SLURM_PROCID is required}
WORKERS=${SLURM_NTASKS:?SLURM_NTASKS is required}
THREADS=${SLURM_CPUS_PER_TASK:?SLURM_CPUS_PER_TASK is required}
DEADLINE_EPOCH=${ITENSOR_SPIN_DEADLINE_EPOCH:?deadline is required}
STOP_AFTER_N=${ITENSOR_SPIN_STOP_AFTER_N:-0}
CHECKPOINT_EVERY=${ITENSOR_SPIN_CHECKPOINT_EVERY:-1}
HEARTBEAT_EVERY=${ITENSOR_SPIN_HEARTBEAT_EVERY:-1}
MANIFEST_ROOT=${ITENSOR_SPIN_MANIFEST_ROOT:?task manifest root is required}
ENV_ROOT=${CODE_ROOT}/ITensor_env
JULIA_BIN=${ITENSOR_SPIN_JULIA:-/capstor/scratch/cscs/cpi/goncalo/.julia/gh200/juliaup/bin/julia}
worker_dir=${WORKER_ROOT}/worker_${RANK}
task_ids_file=${MANIFEST_ROOT}/worker_${RANK}.txt

mkdir -p "${worker_dir}"
[[ -s "${task_ids_file}" ]] || {
    echo "Missing task manifest: ${task_ids_file}" >&2
    exit 2
}
echo "rank=${RANK}/${WORKERS}, host=$(hostname), threads=${THREADS}, deadline=${DEADLINE_EPOCH}"

export JULIA_NUM_THREADS=${THREADS},0
exec "${JULIA_BIN}" --startup-file=no --project="${ENV_ROOT}" \
    "${CODE_ROOT}/chi_single_task_checkpoint.jl" \
    --checkpoint "${CHECKPOINT}" \
    --output "${worker_dir}" \
    --worker-rank "${RANK}" \
    --worker-count "${WORKERS}" \
    --task-ids-file "${task_ids_file}" \
    --backend "${BACKEND}" \
    --threading-mode "${THREADING_MODE}" \
    --resume true \
    --checkpoint-every "${CHECKPOINT_EVERY}" \
    --heartbeat-every "${HEARTBEAT_EVERY}" \
    --stop-after-n "${STOP_AFTER_N}" \
    --stop-before-epoch "${DEADLINE_EPOCH}"

