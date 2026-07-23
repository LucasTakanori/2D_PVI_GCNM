#!/usr/bin/env bash
# One shared dataset allocation plus two comparable end-to-end model allocations.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"
mkdir -p logs

CONFIG="${CONFIG:-${REPO_ROOT}/configs/subject006_anatomical_gcnm.yaml}"
SIMULATOR_ROOT="${SIMULATOR_ROOT:-$(dirname "${REPO_ROOT}")/Finger-Conductivity-Simulator}"
DATASET_DIR="${DATASET_DIR:-${REPO_ROOT}/data/finger_generalized_beats_US120_v2}"
VOLTAGE_EXPERIMENT="${VOLTAGE_EXPERIMENT:-finger_v2_beat_voltage_slots_hom_seed0}"
DIFFUSION_EXPERIMENT="${DIFFUSION_EXPERIMENT:-finger_v2_beat_diffusion_slots_hom_seed0}"

for path in \
  "${DATASET_DIR}" \
  "${REPO_ROOT}/models/faithful/${VOLTAGE_EXPERIMENT}" \
  "${REPO_ROOT}/models/faithful/${DIFFUSION_EXPERIMENT}" \
  "${REPO_ROOT}/data/faithful_results/${VOLTAGE_EXPERIMENT}" \
  "${REPO_ROOT}/data/faithful_results/${DIFFUSION_EXPERIMENT}"; do
  if [[ -e "${path}" ]]; then
    echo "ERROR: output already exists: ${path}" >&2
    exit 1
  fi
done

DATA_JOB=$(sbatch --parsable \
  --time=02:00:00 \
  --job-name="finger-v2-beat-data" \
  --export="ALL,REPO_ROOT=${REPO_ROOT},CONFIG=${CONFIG},SIMULATOR_ROOT=${SIMULATOR_ROOT},DATASET_DIR=${DATASET_DIR},TRAIN_BEATS=1000,VALIDATION_BEATS=200,TEST_BEATS=8,TRAIN_PHASES_PER_BEAT=4,VALIDATION_PHASES_PER_BEAT=4,JACOBIAN_BANK_SIZE=32,DATA_SEED=20260718" \
  slurm/launch_generate_generalized_finger_beats.sh)

VOLTAGE_JOB=$(sbatch --parsable \
  --time=02:00:00 \
  --dependency="afterok:${DATA_JOB}" \
  --job-name="finger-v2-voltage-slots" \
  --export="ALL,REPO_ROOT=${REPO_ROOT},CONFIG=${CONFIG},DATASET_DIR=${DATASET_DIR},EXPERIMENT=${VOLTAGE_EXPERIMENT},ARCHITECTURE=beat_voltage_slots" \
  slurm/launch_generalized_vessel_pipeline.sh)

DIFFUSION_JOB=$(sbatch --parsable \
  --time=02:00:00 \
  --dependency="afterok:${DATA_JOB}" \
  --job-name="finger-v2-diffusion-slots" \
  --export="ALL,REPO_ROOT=${REPO_ROOT},CONFIG=${CONFIG},DATASET_DIR=${DATASET_DIR},EXPERIMENT=${DIFFUSION_EXPERIMENT},ARCHITECTURE=beat_diffusion_slots" \
  slurm/launch_generalized_vessel_pipeline.sh)

echo "dataset=${DATA_JOB}"
echo "voltage_slots=${VOLTAGE_JOB}"
echo "diffusion_slots=${DIFFUSION_JOB}"
