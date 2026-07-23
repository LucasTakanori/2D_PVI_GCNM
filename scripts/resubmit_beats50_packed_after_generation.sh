#!/usr/bin/env bash
# Replace only the held training/export tail after an existing generation job.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"
source "${REPO_ROOT}/env/cluster.env"

DATA_JOB="${DATA_JOB:?set DATA_JOB to the active 15-ring generation array}"
CHECKPOINT_ROOT="${REPO_ROOT}/models/mesh_representations/beats1000x50_v1"
PARQUET_ROOT="${GCNM_PARQUET_ROOT:-${REPO_ROOT}/gcnm_parquet}"
COORDINATE_PARQUET="${PARQUET_ROOT}/coordinate_v2"
GLOBAL_PARQUET="${PARQUET_ROOT}/global_voltage_slots_v2"
MANIFEST="${REPO_ROOT}/data/manifests/mesh_training_v3.json"

for path in "${CHECKPOINT_ROOT}" "${COORDINATE_PARQUET}" "${GLOBAL_PARQUET}"; do
  if [[ -e "${path}" ]]; then
    echo "ERROR: corrected training/export output already exists: ${path}" >&2
    exit 1
  fi
done

train_job="$(sbatch --parsable --dependency="afterok:${DATA_JOB}" \
  --export="ALL,REPO_ROOT=${REPO_ROOT},MESH_TASK_MANIFEST=${MANIFEST}" \
  slurm/launch_train_beats50_packed.sh)"
coordinate_export="$(sbatch --parsable --dependency="afterok:${train_job}" \
  --export="ALL,REPO_ROOT=${REPO_ROOT},FAMILY=coordinate,CHECKPOINT_ROOT=${CHECKPOINT_ROOT}/coordinate,OUTPUT_ROOT=${COORDINATE_PARQUET}" \
  slurm/launch_export_pvi_parquet.sh)"
global_export="$(sbatch --parsable --dependency="afterok:${train_job}" \
  --export="ALL,REPO_ROOT=${REPO_ROOT},FAMILY=global_voltage_slots,CHECKPOINT_ROOT=${CHECKPOINT_ROOT}/global_voltage_slots,OUTPUT_ROOT=${GLOBAL_PARQUET}" \
  slurm/launch_export_pvi_parquet.sh)"

echo "packed 16-process/4-GPU training: ${train_job}"
echo "coordinate export: ${coordinate_export}"
echo "global-voltage export: ${global_export}"
