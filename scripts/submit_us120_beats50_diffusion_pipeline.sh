#!/usr/bin/env bash
# Submit the separate 1,000-whole-beat, 50-samples-per-beat US120 comparison.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"
source "${REPO_ROOT}/env/cluster.env"
mkdir -p logs

CONFIG="${CONFIG:-${REPO_ROOT}/configs/subject006_anatomical_gcnm.yaml}"
DATASET_DIR="${DATASET_DIR:-${REPO_ROOT}/data/finger_default_beats1000x50_US120_v1}"
EXPERIMENT="${EXPERIMENT:-finger_default_beats1000x50_diffusion_slots_US120_seed0}"
BASELINE_EXPERIMENT="${BASELINE_EXPERIMENT:-finger_default_diffusion_slots_corr010_mlp_seed0}"
REAL_TEST_FILE="${REAL_TEST_FILE:-${REPO_ROOT}/data/finger_default_anatomical_exact/subject006_test_default_finger_baseline.npz}"
RESULT_DIR="${REPO_ROOT}/data/faithful_results/${EXPERIMENT}"
MODEL_DIR="${REPO_ROOT}/models/faithful/${EXPERIMENT}"

for path in "${DATASET_DIR}" "${RESULT_DIR}" "${MODEL_DIR}"; do
  if [[ -e "${path}" ]]; then
    echo "ERROR: immutable output already exists: ${path}" >&2
    exit 1
  fi
done

DATA_JOB="$(sbatch --parsable \
  --export="ALL,REPO_ROOT=${REPO_ROOT},CONFIG=${CONFIG},DATASET_DIR=${DATASET_DIR}" \
  slurm/launch_generate_us120_beats50.sh)"

TRAIN_JOB="$(sbatch --parsable --time=10-00:00:00 \
  --dependency="afterok:${DATA_JOB}" --job-name="${EXPERIMENT}-train" \
  --export="ALL,REPO_ROOT=${REPO_ROOT},CONFIG=${CONFIG},DATASET_DIR=${DATASET_DIR},EXPERIMENT=${EXPERIMENT},ARCHITECTURE=beat_diffusion_slots,BASELINE_MODE=homogeneous,BASELINE_CONDUCTIVITY=0.7,CONDUCTIVITY_SCALE_MODE=positive_p995,POSITIVE_WEIGHT=1.0,BACKGROUND_WEIGHT=0.25,DICE_WEIGHT=0.05,SLOT_WEIGHT=0.5,SEPARATION_WEIGHT=0.15,ATTENTION_WEIGHT=0.05,CORRELATION_WEIGHT=0.10,MINIMUM_CENTER_SEPARATION=0.25,MINIMUM_VESSEL_AXIS=0.025,MAXIMUM_VESSEL_AXIS=0.23,SEED=0" \
  slurm/launch_train_voltage_vessel.sh)"

NEW_SYNTH_JOB="$(sbatch --parsable --time=10-00:00:00 \
  --dependency="afterok:${TRAIN_JOB}" --job-name="${EXPERIMENT}-exact" \
  --export="ALL,REPO_ROOT=${REPO_ROOT},CONFIG=${CONFIG},DATASET_DIR=${DATASET_DIR},EXPERIMENT=${EXPERIMENT},EVALUATION_NAME=comparison_exact_nonlinear_50samples,TARGET_KIND=clean" \
  slurm/launch_evaluate_voltage_vessel.sh)"
NEW_REAL_JOB="$(sbatch --parsable --time=10-00:00:00 \
  --dependency="afterok:${TRAIN_JOB}" --job-name="${EXPERIMENT}-real" \
  --export="ALL,REPO_ROOT=${REPO_ROOT},CONFIG=${CONFIG},DATASET_DIR=${DATASET_DIR},TEST_FILE=${REAL_TEST_FILE},EXPERIMENT=${EXPERIMENT},EVALUATION_NAME=comparison_real_50samples,TARGET_KIND=real_subject006" \
  slurm/launch_evaluate_voltage_vessel.sh)"

# Re-evaluate the accepted frame-wise model on the exact same 5,000 signed
# nonlinear samples and real voltage pack. These are fair paired comparisons.
BASELINE_SYNTH_JOB="$(sbatch --parsable --time=10-00:00:00 \
  --dependency="afterok:${DATA_JOB}" --job-name="baseline-beats50-exact" \
  --export="ALL,REPO_ROOT=${REPO_ROOT},CONFIG=${CONFIG},DATASET_DIR=${DATASET_DIR},EXPERIMENT=${BASELINE_EXPERIMENT},EVALUATION_NAME=comparison_exact_nonlinear_50samples,TARGET_KIND=clean" \
  slurm/launch_evaluate_voltage_vessel.sh)"
BASELINE_REAL_JOB="$(sbatch --parsable --time=10-00:00:00 \
  --dependency="afterok:${DATA_JOB}" --job-name="baseline-beats50-real" \
  --export="ALL,REPO_ROOT=${REPO_ROOT},CONFIG=${CONFIG},DATASET_DIR=${DATASET_DIR},TEST_FILE=${REAL_TEST_FILE},EXPERIMENT=${BASELINE_EXPERIMENT},EVALUATION_NAME=comparison_real_50samples,TARGET_KIND=real_subject006" \
  slurm/launch_evaluate_voltage_vessel.sh)"

# Consolidate exact-nonlinear accuracy, real-voltage residual, and the
# within-beat vessel-localization variation from both models.  BP performance
# can be added with --bp-json after the corresponding pvi_ml run completes.
REPORT_DIR="${REPO_ROOT}/data/faithful_results/us120_beats50_comparison"
REPORT_JOB="$(sbatch --parsable --time=01:00:00 \
  --dependency="afterok:${NEW_SYNTH_JOB}:${NEW_REAL_JOB}:${BASELINE_SYNTH_JOB}:${BASELINE_REAL_JOB}" \
  --job-name="${EXPERIMENT}-report" \
  --export="ALL,REPO_ROOT=${REPO_ROOT},NEW_ROOT=${REPO_ROOT}/data/faithful_results/${EXPERIMENT},BASELINE_ROOT=${REPO_ROOT}/data/faithful_results/${BASELINE_EXPERIMENT},REPORT_DIR=${REPORT_DIR}" \
  slurm/launch_summarize_us120_beats50.sh)"

echo "US120 beats50 dataset: ${DATA_JOB} (CPU only, no GPU requested)"
echo "signed diffusion training: ${TRAIN_JOB}"
echo "new exact-nonlinear evaluation: ${NEW_SYNTH_JOB}"
echo "new real-voltage evaluation: ${NEW_REAL_JOB}"
echo "accepted-model paired exact evaluation: ${BASELINE_SYNTH_JOB}"
echo "accepted-model paired real evaluation: ${BASELINE_REAL_JOB}"
echo "US120 50-sample comparison report: ${REPORT_JOB}"
