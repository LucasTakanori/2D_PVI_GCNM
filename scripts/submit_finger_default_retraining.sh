#!/usr/bin/env bash
# Generate exact default-finger PVI data, then retrain the three requested models.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"
mkdir -p logs

CONFIG="${CONFIG:-${REPO_ROOT}/configs/subject006_anatomical_gcnm.yaml}"
SIMULATOR_ROOT="${SIMULATOR_ROOT:-$(dirname "${REPO_ROOT}")/Finger-Conductivity-Simulator}"
DATASET_DIR="${DATASET_DIR:-${REPO_ROOT}/data/finger_default_anatomical_exact}"
REAL_TEST_FILE="${DATASET_DIR}/subject006_test_default_finger_baseline.npz"

COORD_EXPERIMENT="${COORD_EXPERIMENT:-finger_default_coords_mlp_a1_b025_seed0}"
SLOT_EXPERIMENT="${SLOT_EXPERIMENT:-finger_default_voltage_vessel_slots_mlp_seed0}"
SPATIAL_EXPERIMENT="${SPATIAL_EXPERIMENT:-finger_default_spatial_slots_corr010_mlp_seed0}"

DATA_JOB=$(sbatch --parsable \
  --time=02:00:00 \
  --job-name="finger-default-data" \
  --export="ALL,REPO_ROOT=${REPO_ROOT},CONFIG=${CONFIG},SIMULATOR_ROOT=${SIMULATOR_ROOT},DATASET_DIR=${DATASET_DIR},SYNTH_TRAIN=512,SYNTH_VALIDATION=128,SYNTH_TEST=32,SEED=20260716" \
  slurm/launch_generate_finger_simulator_dataset.sh)

COORD_TRAIN=$(sbatch --parsable \
  --time=02:00:00 \
  --dependency="afterok:${DATA_JOB}" \
  --job-name="${COORD_EXPERIMENT}-train" \
  --export="ALL,REPO_ROOT=${REPO_ROOT},CONFIG=${CONFIG},DATASET_DIR=${DATASET_DIR},EXPERIMENT=${COORD_EXPERIMENT},OUTPUT_MODE=direct,USE_COORDINATES=1,USE_VOLTAGE_MLP=1,POSITIVE_WEIGHT=1.0,BACKGROUND_WEIGHT=0.25,DICE_WEIGHT=0.0,CHECKPOINT_MODE=composite,BASELINE_MODE=saved,SEED=0" \
  slurm/launch_train_faithful.sh)

SLOT_TRAIN=$(sbatch --parsable \
  --time=02:00:00 \
  --dependency="afterok:${DATA_JOB}" \
  --job-name="${SLOT_EXPERIMENT}-train" \
  --export="ALL,REPO_ROOT=${REPO_ROOT},CONFIG=${CONFIG},DATASET_DIR=${DATASET_DIR},EXPERIMENT=${SLOT_EXPERIMENT},ARCHITECTURE=mean_pool_dense_refiner,BASELINE_MODE=saved,CONDUCTIVITY_SCALE_MODE=positive_p995,POSITIVE_WEIGHT=1.0,BACKGROUND_WEIGHT=0.25,DICE_WEIGHT=0.05,SLOT_WEIGHT=0.2,SEPARATION_WEIGHT=0.1,ATTENTION_WEIGHT=0.05,CORRELATION_WEIGHT=0.0,MINIMUM_CENTER_SEPARATION=0.25,MINIMUM_VESSEL_AXIS=0.025,MAXIMUM_VESSEL_AXIS=0.23,SEED=0" \
  slurm/launch_train_voltage_vessel.sh)

SPATIAL_TRAIN=$(sbatch --parsable \
  --time=02:00:00 \
  --dependency="afterok:${DATA_JOB}" \
  --job-name="${SPATIAL_EXPERIMENT}-train" \
  --export="ALL,REPO_ROOT=${REPO_ROOT},CONFIG=${CONFIG},DATASET_DIR=${DATASET_DIR},EXPERIMENT=${SPATIAL_EXPERIMENT},ARCHITECTURE=spatial_slots,BASELINE_MODE=saved,CONDUCTIVITY_SCALE_MODE=positive_p995,POSITIVE_WEIGHT=1.0,BACKGROUND_WEIGHT=0.25,DICE_WEIGHT=0.05,SLOT_WEIGHT=0.5,SEPARATION_WEIGHT=0.15,ATTENTION_WEIGHT=0.05,CORRELATION_WEIGHT=0.10,MINIMUM_CENTER_SEPARATION=0.25,MINIMUM_VESSEL_AXIS=0.025,MAXIMUM_VESSEL_AXIS=0.23,SEED=0" \
  slurm/launch_train_voltage_vessel.sh)

COORD_SYNTH=$(sbatch --parsable \
  --time=02:00:00 --dependency="afterok:${COORD_TRAIN}" \
  --job-name="${COORD_EXPERIMENT}-synthetic" \
  --export="ALL,REPO_ROOT=${REPO_ROOT},CONFIG=${CONFIG},DATASET_DIR=${DATASET_DIR},EXPERIMENT=${COORD_EXPERIMENT},EVALUATION_NAME=evaluation_nonlinear,TARGET_KIND=clean" \
  slurm/launch_evaluate_faithful.sh)

COORD_REAL=$(sbatch --parsable \
  --time=02:00:00 --dependency="afterok:${COORD_TRAIN}" \
  --job-name="${COORD_EXPERIMENT}-real" \
  --export="ALL,REPO_ROOT=${REPO_ROOT},CONFIG=${CONFIG},DATASET_DIR=${DATASET_DIR},TEST_FILE=${REAL_TEST_FILE},EXPERIMENT=${COORD_EXPERIMENT},EVALUATION_NAME=evaluation_real_pvi,TARGET_KIND=real_subject006,SKIP_LM_CONTROL=1" \
  slurm/launch_evaluate_faithful.sh)

SLOT_SYNTH=$(sbatch --parsable \
  --time=02:00:00 --dependency="afterok:${SLOT_TRAIN}" \
  --job-name="${SLOT_EXPERIMENT}-synthetic" \
  --export="ALL,REPO_ROOT=${REPO_ROOT},CONFIG=${CONFIG},DATASET_DIR=${DATASET_DIR},EXPERIMENT=${SLOT_EXPERIMENT},EVALUATION_NAME=evaluation_nonlinear,TARGET_KIND=clean" \
  slurm/launch_evaluate_voltage_vessel.sh)

SLOT_REAL=$(sbatch --parsable \
  --time=02:00:00 --dependency="afterok:${SLOT_TRAIN}" \
  --job-name="${SLOT_EXPERIMENT}-real" \
  --export="ALL,REPO_ROOT=${REPO_ROOT},CONFIG=${CONFIG},DATASET_DIR=${DATASET_DIR},TEST_FILE=${REAL_TEST_FILE},EXPERIMENT=${SLOT_EXPERIMENT},EVALUATION_NAME=evaluation_real_pvi,TARGET_KIND=real_subject006" \
  slurm/launch_evaluate_voltage_vessel.sh)

SPATIAL_SYNTH=$(sbatch --parsable \
  --time=02:00:00 --dependency="afterok:${SPATIAL_TRAIN}" \
  --job-name="${SPATIAL_EXPERIMENT}-synthetic" \
  --export="ALL,REPO_ROOT=${REPO_ROOT},CONFIG=${CONFIG},DATASET_DIR=${DATASET_DIR},EXPERIMENT=${SPATIAL_EXPERIMENT},EVALUATION_NAME=evaluation_nonlinear,TARGET_KIND=clean" \
  slurm/launch_evaluate_voltage_vessel.sh)

SPATIAL_REAL=$(sbatch --parsable \
  --time=02:00:00 --dependency="afterok:${SPATIAL_TRAIN}" \
  --job-name="${SPATIAL_EXPERIMENT}-real" \
  --export="ALL,REPO_ROOT=${REPO_ROOT},CONFIG=${CONFIG},DATASET_DIR=${DATASET_DIR},TEST_FILE=${REAL_TEST_FILE},EXPERIMENT=${SPATIAL_EXPERIMENT},EVALUATION_NAME=evaluation_real_pvi,TARGET_KIND=real_subject006" \
  slurm/launch_evaluate_voltage_vessel.sh)

COORD_GIFS=$(sbatch --parsable \
  --time=02:00:00 --dependency="afterok:${COORD_SYNTH}:${COORD_REAL}" \
  --job-name="${COORD_EXPERIMENT}-gifs" \
  --export="ALL,REPO_ROOT=${REPO_ROOT},EXPERIMENT=${COORD_EXPERIMENT},EXPECTED_STAGES=2" \
  slurm/launch_make_stage_gifs.sh)

SLOT_GIFS=$(sbatch --parsable \
  --time=02:00:00 --dependency="afterok:${SLOT_SYNTH}:${SLOT_REAL}" \
  --job-name="${SLOT_EXPERIMENT}-gifs" \
  --export="ALL,REPO_ROOT=${REPO_ROOT},EXPERIMENT=${SLOT_EXPERIMENT},EXPECTED_STAGES=2" \
  slurm/launch_make_stage_gifs.sh)

SPATIAL_GIFS=$(sbatch --parsable \
  --time=02:00:00 --dependency="afterok:${SPATIAL_SYNTH}:${SPATIAL_REAL}" \
  --job-name="${SPATIAL_EXPERIMENT}-gifs" \
  --export="ALL,REPO_ROOT=${REPO_ROOT},EXPERIMENT=${SPATIAL_EXPERIMENT},EXPECTED_STAGES=2" \
  slurm/launch_make_stage_gifs.sh)

echo "dataset generation: ${DATA_JOB}"
echo "coordinate GCNM: train=${COORD_TRAIN} synthetic=${COORD_SYNTH} real=${COORD_REAL} gifs=${COORD_GIFS}"
echo "vessel slots: train=${SLOT_TRAIN} synthetic=${SLOT_SYNTH} real=${SLOT_REAL} gifs=${SLOT_GIFS}"
echo "spatial slots: train=${SPATIAL_TRAIN} synthetic=${SPATIAL_SYNTH} real=${SPATIAL_REAL} gifs=${SPATIAL_GIFS}"
