#!/usr/bin/env bash
# Submit data generation -> GPU training -> independent evaluation with dependencies.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"
mkdir -p logs

LINEAR_DIR="${LINEAR_DIR:-${REPO_ROOT}/data/subject006_anatomical_full}"
NONLINEAR_DIR="${NONLINEAR_DIR:-${REPO_ROOT}/data/subject006_anatomical_nonlinear_holdout}"
CONFIG="${CONFIG:-${REPO_ROOT}/configs/subject006_anatomical_gcnm.yaml}"

LINEAR_JOB=$(sbatch --parsable \
  --export="ALL,DATASET_DIR=${LINEAR_DIR},SYNTH_TRAIN=512,SYNTH_VALIDATION=128,SYNTH_TEST=128,SIMULATION_MODE=linearized,JACOBIAN_BANK_SIZE=8" \
  slurm/launch_generate_anatomical.sh)

NONLINEAR_JOB=$(sbatch --parsable \
  --job-name=gcnm-nonlinear-holdout \
  --export="ALL,DATASET_DIR=${NONLINEAR_DIR},SYNTH_TRAIN=1,SYNTH_VALIDATION=1,SYNTH_TEST=32,SIMULATION_MODE=nonlinear,JACOBIAN_BANK_SIZE=1" \
  slurm/launch_generate_anatomical.sh)

TRAIN_JOB=$(sbatch --parsable \
  --dependency="afterok:${LINEAR_JOB}" \
  --export="ALL,CONFIG=${CONFIG},DATASET_DIR=${LINEAR_DIR}" \
  slurm/launch_train_anatomical.sh)

EVAL_LINEAR_JOB=$(sbatch --parsable \
  --dependency="afterok:${TRAIN_JOB}" \
  --job-name=gcnm-eval-linear \
  --export="ALL,CONFIG=${CONFIG},DATASET_DIR=${LINEAR_DIR},EVAL_OUT_DIR=${REPO_ROOT}/data/subject006_anatomical_results/evaluation_linearized" \
  slurm/launch_evaluate_anatomical.sh)

EVAL_NONLINEAR_JOB=$(sbatch --parsable \
  --dependency="afterok:${TRAIN_JOB}:${NONLINEAR_JOB}" \
  --job-name=gcnm-eval-nonlinear \
  --export="ALL,CONFIG=${CONFIG},DATASET_DIR=${NONLINEAR_DIR},EVAL_OUT_DIR=${REPO_ROOT}/data/subject006_anatomical_results/evaluation_nonlinear" \
  slurm/launch_evaluate_anatomical.sh)

echo "linearized data job: ${LINEAR_JOB}"
echo "nonlinear holdout job: ${NONLINEAR_JOB}"
echo "training job: ${TRAIN_JOB} (after ${LINEAR_JOB})"
echo "linearized evaluation job: ${EVAL_LINEAR_JOB} (after ${TRAIN_JOB})"
echo "nonlinear evaluation job: ${EVAL_NONLINEAR_JOB} (after ${TRAIN_JOB}, ${NONLINEAR_JOB})"
