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
ENV_ROOT=${CODE_ROOT}/ITensor_env
JULIA_BIN=/capstor/scratch/cscs/cpi/goncalo/.julia/gh200/juliaup/bin/julia
worker_dir=${WORKER_ROOT}/worker_${RANK}

mkdir -p "${worker_dir}"
echo "rank=${RANK}/${WORKERS}, host=$(hostname), threads=${THREADS}"

export JULIA_NUM_THREADS=${THREADS},0
exec "${JULIA_BIN}" --startup-file=no --project="${ENV_ROOT}" \
    "${CODE_ROOT}/chi_reference_task.jl" \
    --checkpoint "${CHECKPOINT}" \
    --output "${worker_dir}" \
    --worker-rank "${RANK}" \
    --worker-count "${WORKERS}" \
    --backend "${BACKEND}" \
    --threading-mode "${THREADING_MODE}"
