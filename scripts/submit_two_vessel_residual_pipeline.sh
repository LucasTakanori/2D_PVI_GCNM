#!/usr/bin/env bash
# Generate all-two-vessel data, train a two-stage shallow residual GCNM, and evaluate it.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"
mkdir -p logs

EXPERIMENT="${EXPERIMENT:-faithful_two_vessel_shallow_residual_dice_hom_seed0}"
CONFIG="${CONFIG:-${REPO_ROOT}/configs/subject006_anatomical_gcnm.yaml}"
LINEAR_DIR="${LINEAR_DIR:-${REPO_ROOT}/data/subject006_anatomical_two_vessel_full}"
NONLINEAR_DIR="${NONLINEAR_DIR:-${REPO_ROOT}/data/subject006_anatomical_two_vessel_nonlinear_holdout}"
REAL_TEST_FILE="${REAL_TEST_FILE:-${REPO_ROOT}/data/subject006_gcnm_hdf/test.npz}"

LINEAR_DATA_JOB=$(sbatch --parsable \
  --time=02:00:00 \
  --job-name="${EXPERIMENT}-data" \
  --export="ALL,REPO_ROOT=${REPO_ROOT},CONFIG=${CONFIG},DATASET_DIR=${LINEAR_DIR},SYNTH_TRAIN=512,SYNTH_VALIDATION=128,SYNTH_TEST=128,SIMULATION_MODE=linearized,JACOBIAN_BANK_SIZE=8,VESSEL_COUNT=2" \
  slurm/launch_generate_anatomical.sh)

NONLINEAR_DATA_JOB=$(sbatch --parsable \
  --time=02:00:00 \
  --job-name="${EXPERIMENT}-exact-data" \
  --export="ALL,REPO_ROOT=${REPO_ROOT},CONFIG=${CONFIG},DATASET_DIR=${NONLINEAR_DIR},SYNTH_TRAIN=1,SYNTH_VALIDATION=1,SYNTH_TEST=32,SIMULATION_MODE=nonlinear,JACOBIAN_BANK_SIZE=1,VESSEL_COUNT=2" \
  slurm/launch_generate_anatomical.sh)

TRAIN_EXPORT="ALL,REPO_ROOT=${REPO_ROOT},CONFIG=${CONFIG},DATASET_DIR=${LINEAR_DIR},EXPERIMENT=${EXPERIMENT},OUTPUT_MODE=shallow_residual,USE_COORDINATES=1,POSITIVE_WEIGHT=1,BACKGROUND_WEIGHT=0.25,DICE_WEIGHT=0.05,HARD_BACKGROUND_WEIGHT=0.10,HARD_BACKGROUND_FRACTION=0.05,CHECKPOINT_MODE=composite,ITERATIONS=2,SEED=0,BASELINE_MODE=homogeneous,BASELINE_CONDUCTIVITY=0.7"
TRAIN_JOB=$(sbatch --parsable \
  --dependency="afterok:${LINEAR_DATA_JOB}" \
  --job-name="${EXPERIMENT}-train" \
  --export="${TRAIN_EXPORT}" \
  slurm/launch_train_faithful.sh)

COMMON_EXPORT="ALL,REPO_ROOT=${REPO_ROOT},CONFIG=${CONFIG},EXPERIMENT=${EXPERIMENT}"
NONLINEAR_EVAL_JOB=$(sbatch --parsable \
  --dependency="afterok:${TRAIN_JOB}:${NONLINEAR_DATA_JOB}" \
  --job-name="${EXPERIMENT}-nonlinear" \
  --export="${COMMON_EXPORT},DATASET_DIR=${NONLINEAR_DIR},EVALUATION_NAME=evaluation_nonlinear,TARGET_KIND=clean,ITERATIONS=2,SKIP_LM_CONTROL=0" \
  slurm/launch_evaluate_faithful.sh)

LINEAR_EVAL_JOB=$(sbatch --parsable \
  --dependency="afterok:${TRAIN_JOB}" \
  --job-name="${EXPERIMENT}-linear" \
  --export="${COMMON_EXPORT},DATASET_DIR=${LINEAR_DIR},EVALUATION_NAME=evaluation_linearized,TARGET_KIND=clean,ITERATIONS=2,SKIP_LM_CONTROL=0" \
  slurm/launch_evaluate_faithful.sh)

REAL_EVAL_JOB=$(sbatch --parsable \
  --time=02:00:00 \
  --dependency="afterok:${TRAIN_JOB}" \
  --job-name="${EXPERIMENT}-real" \
  --export="${COMMON_EXPORT},DATASET_DIR=$(dirname "${REAL_TEST_FILE}"),TEST_FILE=${REAL_TEST_FILE},EVALUATION_NAME=evaluation_real_pvi,TARGET_KIND=pvi_pseudo,ITERATIONS=2,SKIP_LM_CONTROL=1" \
  slurm/launch_evaluate_faithful.sh)

GIF_JOB=$(sbatch --parsable \
  --dependency="afterok:${NONLINEAR_EVAL_JOB}:${REAL_EVAL_JOB}" \
  --job-name="${EXPERIMENT}-gifs" \
  --export="${COMMON_EXPORT},EXPECTED_STAGES=2" \
  slurm/launch_make_stage_gifs.sh)

echo "experiment: ${EXPERIMENT}"
echo "two-vessel linearized data: ${LINEAR_DATA_JOB}"
echo "two-vessel exact-nonlinear data: ${NONLINEAR_DATA_JOB}"
echo "two-stage residual training: ${TRAIN_JOB}"
echo "linearized evaluation: ${LINEAR_EVAL_JOB}"
echo "exact-nonlinear evaluation: ${NONLINEAR_EVAL_JOB}"
echo "real subject-006 evaluation: ${REAL_EVAL_JOB}"
echo "comparison GIFs: ${GIF_JOB}"
