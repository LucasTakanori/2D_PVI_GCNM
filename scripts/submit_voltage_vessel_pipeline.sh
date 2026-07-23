#!/usr/bin/env bash
# Train and evaluate the voltage-conditioned two-vessel architecture.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"
mkdir -p logs

EXPERIMENT="${EXPERIMENT:-voltage_vessel_slots_hom_seed0}"
CONFIG="${CONFIG:-${REPO_ROOT}/configs/subject006_anatomical_gcnm.yaml}"
LINEAR_DIR="${LINEAR_DIR:-${REPO_ROOT}/data/subject006_anatomical_two_vessel_separated_full}"
NONLINEAR_DIR="${NONLINEAR_DIR:-${REPO_ROOT}/data/subject006_anatomical_two_vessel_separated_nonlinear_holdout}"
REAL_TEST_FILE="${REAL_TEST_FILE:-${REPO_ROOT}/data/subject006_gcnm_hdf/test.npz}"

LINEAR_DATA_JOB=$(sbatch --parsable \
  --time=02:00:00 \
  --job-name="${EXPERIMENT}-data" \
  --export="ALL,REPO_ROOT=${REPO_ROOT},CONFIG=${CONFIG},DATASET_DIR=${LINEAR_DIR},SYNTH_TRAIN=512,SYNTH_VALIDATION=128,SYNTH_TEST=128,SIMULATION_MODE=linearized,JACOBIAN_BANK_SIZE=8,VESSEL_COUNT=2,MINIMUM_VESSEL_GAP=0.06" \
  slurm/launch_generate_anatomical.sh)

NONLINEAR_DATA_JOB=$(sbatch --parsable \
  --time=02:00:00 \
  --job-name="${EXPERIMENT}-exact-data" \
  --export="ALL,REPO_ROOT=${REPO_ROOT},CONFIG=${CONFIG},DATASET_DIR=${NONLINEAR_DIR},SYNTH_TRAIN=1,SYNTH_VALIDATION=1,SYNTH_TEST=32,SIMULATION_MODE=nonlinear,JACOBIAN_BANK_SIZE=1,VESSEL_COUNT=2,MINIMUM_VESSEL_GAP=0.06" \
  slurm/launch_generate_anatomical.sh)

TRAIN_JOB=$(sbatch --parsable \
  --time=02:00:00 \
  --dependency="afterok:${LINEAR_DATA_JOB}" \
  --job-name="${EXPERIMENT}-train" \
  --export="ALL,REPO_ROOT=${REPO_ROOT},CONFIG=${CONFIG},DATASET_DIR=${LINEAR_DIR},EXPERIMENT=${EXPERIMENT},SEED=0,BASELINE_CONDUCTIVITY=0.7" \
  slurm/launch_train_voltage_vessel.sh)

NONLINEAR_EVAL_JOB=$(sbatch --parsable \
  --time=02:00:00 \
  --dependency="afterok:${TRAIN_JOB}:${NONLINEAR_DATA_JOB}" \
  --job-name="${EXPERIMENT}-nonlinear" \
  --export="ALL,REPO_ROOT=${REPO_ROOT},CONFIG=${CONFIG},DATASET_DIR=${NONLINEAR_DIR},EXPERIMENT=${EXPERIMENT},EVALUATION_NAME=evaluation_nonlinear,TARGET_KIND=clean" \
  slurm/launch_evaluate_voltage_vessel.sh)

LINEAR_EVAL_JOB=$(sbatch --parsable \
  --time=02:00:00 \
  --dependency="afterok:${TRAIN_JOB}" \
  --job-name="${EXPERIMENT}-linear" \
  --export="ALL,REPO_ROOT=${REPO_ROOT},CONFIG=${CONFIG},DATASET_DIR=${LINEAR_DIR},EXPERIMENT=${EXPERIMENT},EVALUATION_NAME=evaluation_linearized,TARGET_KIND=clean" \
  slurm/launch_evaluate_voltage_vessel.sh)

REAL_EVAL_JOB=$(sbatch --parsable \
  --time=02:00:00 \
  --dependency="afterok:${TRAIN_JOB}" \
  --job-name="${EXPERIMENT}-real" \
  --export="ALL,REPO_ROOT=${REPO_ROOT},CONFIG=${CONFIG},DATASET_DIR=$(dirname "${REAL_TEST_FILE}"),TEST_FILE=${REAL_TEST_FILE},EXPERIMENT=${EXPERIMENT},EVALUATION_NAME=evaluation_real_pvi,TARGET_KIND=real_subject006" \
  slurm/launch_evaluate_voltage_vessel.sh)

GIF_JOB=$(sbatch --parsable \
  --time=02:00:00 \
  --dependency="afterok:${NONLINEAR_EVAL_JOB}:${REAL_EVAL_JOB}" \
  --job-name="${EXPERIMENT}-gifs" \
  --export="ALL,REPO_ROOT=${REPO_ROOT},EXPERIMENT=${EXPERIMENT},EXPECTED_STAGES=2" \
  slurm/launch_make_stage_gifs.sh)

echo "experiment: ${EXPERIMENT}"
echo "separated two-vessel linearized data: ${LINEAR_DATA_JOB}"
echo "separated exact-nonlinear data: ${NONLINEAR_DATA_JOB}"
echo "voltage-conditioned two-stage training: ${TRAIN_JOB}"
echo "linearized evaluation: ${LINEAR_EVAL_JOB}"
echo "exact-nonlinear evaluation: ${NONLINEAR_EVAL_JOB}"
echo "real subject-006 voltage-only evaluation: ${REAL_EVAL_JOB}"
echo "comparison GIFs: ${GIF_JOB}"
