#!/usr/bin/env bash
# Evaluate the first successful two-stage GCNM on the new separated holdout.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"
mkdir -p logs

EXPERIMENT="faithful_background025_hom_seed0"
CONFIG="${REPO_ROOT}/configs/subject006_anatomical_gcnm.yaml"
DATASET_DIR="${REPO_ROOT}/data/subject006_anatomical_two_vessel_separated_nonlinear_holdout"
OUT_DIR="${REPO_ROOT}/data/faithful_results/${EXPERIMENT}/evaluation_nonlinear_separated"

JOB_ID=$(sbatch --parsable \
  --time=02:00:00 \
  --job-name="old-gcnm-separated" \
  --export="ALL,REPO_ROOT=${REPO_ROOT},CONFIG=${CONFIG},DATASET_DIR=${DATASET_DIR},EXPERIMENT=${EXPERIMENT},EVALUATION_NAME=evaluation_nonlinear_separated,OUT_DIR=${OUT_DIR},TARGET_KIND=clean,ITERATIONS=2,SKIP_LM_CONTROL=1" \
  slurm/launch_evaluate_faithful.sh)

echo "old GCNM separated-holdout evaluation: ${JOB_ID}"
