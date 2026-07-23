#!/usr/bin/env bash
# Submit the corrected subject006/subject010 pilot dependency chain.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"
source "${REPO_ROOT}/env/cluster.env"
mkdir -p logs

"${GCNM_PYTHON}" -m gcnm_pvi.ring_configs
"${GCNM_PYTHON}" -m gcnm_pvi.mesh_registry --data-root "${PVI_DATA_ROOT}"

RING="US120"
CONFIG="${REPO_ROOT}/configs/rings_b045/${RING}.yaml"
DATASET_ROOT="${REPO_ROOT}/data/mesh_training/beats1000x50_v1"
DATASET_DIR="${DATASET_ROOT}/${RING}"
CHECKPOINT_ROOT="${REPO_ROOT}/models/mesh_representations/beats1000x50_v1"
RESULT_ROOT="${REPO_ROOT}/data/mesh_training_results/beats1000x50_v1"
PARQUET_ROOT="${GCNM_PARQUET_ROOT:-${REPO_ROOT}/gcnm_parquet}"
SUBJECTS="subject006 subject010"

coordinate_model="${CHECKPOINT_ROOT}/coordinate/${RING}"
global_model="${CHECKPOINT_ROOT}/global_voltage_slots/${RING}"
coordinate_result="${RESULT_ROOT}/coordinate/${RING}"
global_result="${RESULT_ROOT}/global_voltage_slots/${RING}"
coordinate_output="${PARQUET_ROOT}/coordinate_beats1000x50_us120_pilot_v2"
global_output="${PARQUET_ROOT}/global_voltage_slots_beats1000x50_us120_pilot_v2"

for path in \
  "${DATASET_DIR}" \
  "${coordinate_model}" "${global_model}" \
  "${coordinate_result}" "${global_result}" \
  "${coordinate_output}" "${global_output}"; do
  if [[ -e "${path}" ]]; then
    echo "ERROR: immutable pilot output already exists: ${path}" >&2
    exit 1
  fi
done

data_job="$(sbatch --parsable \
  --export="ALL,REPO_ROOT=${REPO_ROOT},CONFIG=${CONFIG},DATASET_DIR=${DATASET_DIR}" \
  slurm/launch_generate_us120_beats50.sh)"

coordinate_train="$(sbatch --parsable --dependency="afterok:${data_job}" \
  --export="ALL,REPO_ROOT=${REPO_ROOT},FAMILY=coordinate,RING=${RING},CONFIG=${CONFIG},DATASET_ROOT=${DATASET_ROOT},CHECKPOINT_ROOT=${CHECKPOINT_ROOT},RESULT_ROOT=${RESULT_ROOT}" \
  slurm/launch_train_mesh_beats50_model.sh)"
global_train="$(sbatch --parsable --dependency="afterok:${data_job}" \
  --export="ALL,REPO_ROOT=${REPO_ROOT},FAMILY=global_voltage_slots,RING=${RING},CONFIG=${CONFIG},DATASET_ROOT=${DATASET_ROOT},CHECKPOINT_ROOT=${CHECKPOINT_ROOT},RESULT_ROOT=${RESULT_ROOT}" \
  slurm/launch_train_mesh_beats50_model.sh)"

coordinate_export="$(sbatch --parsable --dependency="afterok:${coordinate_train}" \
  --export="ALL,REPO_ROOT=${REPO_ROOT},FAMILY=coordinate,CHECKPOINT_ROOT=${CHECKPOINT_ROOT}/coordinate,OUTPUT_ROOT=${coordinate_output},SUBJECTS=${SUBJECTS}" \
  slurm/launch_export_pvi_parquet.sh)"
global_export="$(sbatch --parsable --dependency="afterok:${global_train}" \
  --export="ALL,REPO_ROOT=${REPO_ROOT},FAMILY=global_voltage_slots,CHECKPOINT_ROOT=${CHECKPOINT_ROOT}/global_voltage_slots,OUTPUT_ROOT=${global_output},SUBJECTS=${SUBJECTS}" \
  slurm/launch_export_pvi_parquet.sh)"

echo "shared 1000-beat x 50-sample dataset (CPU): ${data_job}"
echo "coordinate training (1 GPU): ${coordinate_train}"
echo "signed global-voltage vessel-slot training (1 GPU): ${global_train}"
echo "coordinate subject006/subject010 export (2 GPUs): ${coordinate_export}"
echo "global subject006/subject010 export (2 GPUs): ${global_export}"
