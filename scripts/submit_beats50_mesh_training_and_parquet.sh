#!/usr/bin/env bash
# Generate 15 shared beat packs, train 30 ring models, then export two full
# target-neutral s1/s2 Parquet representation datasets. BP training is separate.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"
source "${REPO_ROOT}/env/cluster.env"
mkdir -p logs

"${GCNM_PYTHON}" -m gcnm_pvi.ring_configs
"${GCNM_PYTHON}" -m gcnm_pvi.mesh_registry --data-root "${PVI_DATA_ROOT}"
"${GCNM_PYTHON}" -m gcnm_pvi.mesh_run_manifest

DATASET_ROOT="${REPO_ROOT}/data/mesh_training/beats1000x50_v1"
CHECKPOINT_ROOT="${REPO_ROOT}/models/mesh_representations/beats1000x50_v1"
RESULT_ROOT="${REPO_ROOT}/data/mesh_training_results/beats1000x50_v1"
PARQUET_ROOT="${GCNM_PARQUET_ROOT:-${REPO_ROOT}/gcnm_parquet}"
COORDINATE_PARQUET="${PARQUET_ROOT}/coordinate_v2"
GLOBAL_PARQUET="${PARQUET_ROOT}/global_voltage_slots_v2"

for path in \
  "${DATASET_ROOT}" "${CHECKPOINT_ROOT}" "${RESULT_ROOT}" \
  "${COORDINATE_PARQUET}" "${GLOBAL_PARQUET}"; do
  if [[ -e "${path}" ]]; then
    echo "ERROR: immutable corrected-pipeline output already exists: ${path}" >&2
    exit 1
  fi
done

# Four CPU tasks exactly fill the agreed 64-CPU/1-TB ceiling.
data_job="$(sbatch --parsable --array=0-14%4 \
  --export="ALL,REPO_ROOT=${REPO_ROOT},DATASET_ROOT=${DATASET_ROOT}" \
  slurm/launch_generate_mesh_beats50.sh)"

# One four-GPU allocation runs sixteen model processes (four per NVL GPU). Each
# model receives four explicit CPU physics lanes; the packed job stays within the
# 4-GPU/64-CPU/1-TB ceiling.
train_job="$(sbatch --parsable \
  --dependency="afterok:${data_job}" \
  --export="ALL,REPO_ROOT=${REPO_ROOT},MESH_TASK_MANIFEST=${REPO_ROOT}/data/manifests/mesh_training_v3.json" \
  slurm/launch_train_beats50_packed.sh)"

# These two independent exports can run together and use all four GPUs.
coordinate_export="$(sbatch --parsable --dependency="afterok:${train_job}" \
  --export="ALL,REPO_ROOT=${REPO_ROOT},FAMILY=coordinate,CHECKPOINT_ROOT=${CHECKPOINT_ROOT}/coordinate,OUTPUT_ROOT=${COORDINATE_PARQUET}" \
  slurm/launch_export_pvi_parquet.sh)"
global_export="$(sbatch --parsable --dependency="afterok:${train_job}" \
  --export="ALL,REPO_ROOT=${REPO_ROOT},FAMILY=global_voltage_slots,CHECKPOINT_ROOT=${CHECKPOINT_ROOT}/global_voltage_slots,OUTPUT_ROOT=${GLOBAL_PARQUET}" \
  slurm/launch_export_pvi_parquet.sh)"

echo "15 ring beat datasets (array 0-14%4, CPU only): ${data_job}"
echo "30 coordinate/global GCNM trainings (16 processes on 4 GPUs): ${train_job}"
echo "full coordinate s1/s2 Parquet export (2 GPUs): ${coordinate_export}"
echo "full global-voltage s1/s2 Parquet export (2 GPUs): ${global_export}"
echo "BP CRT/CRS training was not submitted."
