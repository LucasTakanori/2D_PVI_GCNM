#!/usr/bin/env bash
# Evaluate completed voltage-vessel checkpoints without retraining.
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

NONLINEAR_EVAL_JOB=$(sbatch --parsable \
  --time=02:00:00 \
  --job-name="${EXPERIMENT}-nonlinear" \
  --export="ALL,REPO_ROOT=${REPO_ROOT},CONFIG=${CONFIG},DATASET_DIR=${NONLINEAR_DIR},EXPERIMENT=${EXPERIMENT},EVALUATION_NAME=evaluation_nonlinear,TARGET_KIND=clean" \
  slurm/launch_evaluate_voltage_vessel.sh)

LINEAR_EVAL_JOB=$(sbatch --parsable \
  --time=02:00:00 \
  --job-name="${EXPERIMENT}-linear" \
  --export="ALL,REPO_ROOT=${REPO_ROOT},CONFIG=${CONFIG},DATASET_DIR=${LINEAR_DIR},EXPERIMENT=${EXPERIMENT},EVALUATION_NAME=evaluation_linearized,TARGET_KIND=clean" \
  slurm/launch_evaluate_voltage_vessel.sh)

REAL_EVAL_JOB=$(sbatch --parsable \
  --time=02:00:00 \
  --job-name="${EXPERIMENT}-real" \
  --export="ALL,REPO_ROOT=${REPO_ROOT},CONFIG=${CONFIG},DATASET_DIR=$(dirname "${REAL_TEST_FILE}"),TEST_FILE=${REAL_TEST_FILE},EXPERIMENT=${EXPERIMENT},EVALUATION_NAME=evaluation_real_pvi,TARGET_KIND=real_subject006" \
  slurm/launch_evaluate_voltage_vessel.sh)

GIF_JOB=$(sbatch --parsable \
  --time=02:00:00 \
  --dependency="afterok:${NONLINEAR_EVAL_JOB}:${REAL_EVAL_JOB}" \
  --job-name="${EXPERIMENT}-gifs" \
  --export="ALL,REPO_ROOT=${REPO_ROOT},EXPERIMENT=${EXPERIMENT},EXPECTED_STAGES=2" \
  slurm/launch_make_stage_gifs.sh)

echo "exact-nonlinear evaluation retry: ${NONLINEAR_EVAL_JOB}"
echo "linearized evaluation retry: ${LINEAR_EVAL_JOB}"
echo "real subject-006 evaluation retry: ${REAL_EVAL_JOB}"
echo "comparison GIFs retry: ${GIF_JOB}"
