#!/usr/bin/env bash
# Submit one faithful GCNM variant and its three evaluation domains.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"
mkdir -p logs

EXPERIMENT="${EXPERIMENT:?set a unique EXPERIMENT name}"
CONFIG="${CONFIG:-${REPO_ROOT}/configs/subject006_anatomical_gcnm.yaml}"
LINEAR_DIR="${LINEAR_DIR:-${REPO_ROOT}/data/subject006_anatomical_full}"
NONLINEAR_DIR="${NONLINEAR_DIR:-${REPO_ROOT}/data/subject006_anatomical_nonlinear_holdout}"
REAL_TEST_FILE="${REAL_TEST_FILE:-${REPO_ROOT}/data/subject006_gcnm_hdf/test.npz}"
OUTPUT_MODE="${OUTPUT_MODE:-direct}"
USE_COORDINATES="${USE_COORDINATES:-0}"
POSITIVE_WEIGHT="${POSITIVE_WEIGHT:-0}"
BACKGROUND_WEIGHT="${BACKGROUND_WEIGHT:-0}"
CHECKPOINT_MODE="${CHECKPOINT_MODE:-loss}"
MAX_TRAIN="${MAX_TRAIN:-}"
MAX_VALIDATION="${MAX_VALIDATION:-}"
MAX_TEST="${MAX_TEST:-}"
EPOCHS="${EPOCHS:-}"
ITERATIONS="${ITERATIONS:-}"
SEED="${SEED:-0}"
BASELINE_MODE="${BASELINE_MODE:-homogeneous}"
BASELINE_CONDUCTIVITY="${BASELINE_CONDUCTIVITY:-0.7}"
RUN_REAL="${RUN_REAL:-1}"
RUN_LINEAR="${RUN_LINEAR:-1}"
RUN_GIFS="${RUN_GIFS:-0}"
SKIP_LM_CONTROL="${SKIP_LM_CONTROL:-0}"

COMMON_EXPORT="ALL,REPO_ROOT=${REPO_ROOT},CONFIG=${CONFIG},EXPERIMENT=${EXPERIMENT}"
TRAIN_EXPORT="${COMMON_EXPORT},DATASET_DIR=${LINEAR_DIR},OUTPUT_MODE=${OUTPUT_MODE},USE_COORDINATES=${USE_COORDINATES},POSITIVE_WEIGHT=${POSITIVE_WEIGHT},BACKGROUND_WEIGHT=${BACKGROUND_WEIGHT},CHECKPOINT_MODE=${CHECKPOINT_MODE},MAX_TRAIN=${MAX_TRAIN},MAX_VALIDATION=${MAX_VALIDATION},EPOCHS=${EPOCHS},ITERATIONS=${ITERATIONS},SEED=${SEED},BASELINE_MODE=${BASELINE_MODE},BASELINE_CONDUCTIVITY=${BASELINE_CONDUCTIVITY}"

TRAIN_JOB=$(sbatch --parsable \
  --job-name="${EXPERIMENT}-train" \
  --export="${TRAIN_EXPORT}" \
  slurm/launch_train_faithful.sh)

NONLINEAR_JOB=$(sbatch --parsable \
  --dependency="afterok:${TRAIN_JOB}" \
  --job-name="${EXPERIMENT}-nonlinear" \
  --export="${COMMON_EXPORT},DATASET_DIR=${NONLINEAR_DIR},EVALUATION_NAME=evaluation_nonlinear,TARGET_KIND=clean,MAX_TEST=${MAX_TEST},ITERATIONS=${ITERATIONS},SKIP_LM_CONTROL=${SKIP_LM_CONTROL}" \
  slurm/launch_evaluate_faithful.sh)

echo "experiment: ${EXPERIMENT}"
echo "training: ${TRAIN_JOB}"
echo "nonlinear evaluation: ${NONLINEAR_JOB}"

if [[ "${RUN_LINEAR}" == "1" ]]; then
  LINEAR_JOB=$(sbatch --parsable \
    --dependency="afterok:${TRAIN_JOB}" \
    --job-name="${EXPERIMENT}-linear" \
    --export="${COMMON_EXPORT},DATASET_DIR=${LINEAR_DIR},EVALUATION_NAME=evaluation_linearized,TARGET_KIND=clean,MAX_TEST=${MAX_TEST},ITERATIONS=${ITERATIONS},SKIP_LM_CONTROL=${SKIP_LM_CONTROL}" \
    slurm/launch_evaluate_faithful.sh)
  echo "linearized evaluation: ${LINEAR_JOB}"
fi

REAL_JOB=""
if [[ "${RUN_REAL}" == "1" ]]; then
  REAL_JOB=$(sbatch --parsable \
    --dependency="afterok:${TRAIN_JOB}" \
    --job-name="${EXPERIMENT}-real" \
    --export="${COMMON_EXPORT},DATASET_DIR=$(dirname "${REAL_TEST_FILE}"),TEST_FILE=${REAL_TEST_FILE},EVALUATION_NAME=evaluation_real_pvi,TARGET_KIND=pvi_pseudo,MAX_TEST=${MAX_TEST},ITERATIONS=${ITERATIONS},SKIP_LM_CONTROL=${SKIP_LM_CONTROL}" \
    slurm/launch_evaluate_faithful.sh)
  echo "real-PVI evaluation: ${REAL_JOB}"
fi

if [[ "${RUN_GIFS}" == "1" ]]; then
  if [[ -z "${REAL_JOB}" ]]; then
    echo "ERROR: RUN_GIFS=1 requires RUN_REAL=1" >&2
    exit 1
  fi
  if [[ -z "${ITERATIONS}" ]]; then
    echo "ERROR: RUN_GIFS=1 requires an explicit ITERATIONS value" >&2
    exit 1
  fi
  GIF_JOB=$(sbatch --parsable \
    --dependency="afterok:${NONLINEAR_JOB}:${REAL_JOB}" \
    --job-name="${EXPERIMENT}-gifs" \
    --export="${COMMON_EXPORT},EXPECTED_STAGES=${ITERATIONS}" \
    slurm/launch_make_stage_gifs.sh)
  echo "stage-comparison GIFs: ${GIF_JOB}"
fi
