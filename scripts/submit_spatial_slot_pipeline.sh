#!/usr/bin/env bash
# Train and evaluate spatially attended, topology-preserving vessel slots.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"
mkdir -p logs

EXPERIMENT="${EXPERIMENT:-voltage_vessel_spatial_slots_corr010_hom_seed0}"
CONFIG="${CONFIG:-${REPO_ROOT}/configs/subject006_anatomical_gcnm.yaml}"
LINEAR_DIR="${LINEAR_DIR:-${REPO_ROOT}/data/subject006_anatomical_two_vessel_separated_full}"
NONLINEAR_DIR="${NONLINEAR_DIR:-${REPO_ROOT}/data/subject006_anatomical_two_vessel_separated_nonlinear_holdout}"
REAL_TEST_FILE="${REAL_TEST_FILE:-${REPO_ROOT}/data/subject006_gcnm_hdf/test.npz}"

TRAIN_JOB=$(sbatch --parsable \
  --time=02:00:00 \
  --job-name="${EXPERIMENT}-train" \
  --export="ALL,REPO_ROOT=${REPO_ROOT},CONFIG=${CONFIG},DATASET_DIR=${LINEAR_DIR},EXPERIMENT=${EXPERIMENT},ARCHITECTURE=spatial_slots,POSITIVE_WEIGHT=1.0,BACKGROUND_WEIGHT=0.25,DICE_WEIGHT=0.05,SLOT_WEIGHT=0.5,SEPARATION_WEIGHT=0.15,ATTENTION_WEIGHT=0.05,CORRELATION_WEIGHT=0.10,MINIMUM_CENTER_SEPARATION=0.25,SEED=0,BASELINE_CONDUCTIVITY=0.7" \
  slurm/launch_train_voltage_vessel.sh)

NONLINEAR_EVAL_JOB=$(sbatch --parsable \
  --time=02:00:00 \
  --dependency="afterok:${TRAIN_JOB}" \
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
echo "spatial-slot training: ${TRAIN_JOB}"
echo "exact-nonlinear evaluation: ${NONLINEAR_EVAL_JOB}"
echo "linearized evaluation: ${LINEAR_EVAL_JOB}"
echo "real subject-006 voltage-only evaluation: ${REAL_EVAL_JOB}"
echo "comparison GIFs: ${GIF_JOB}"
