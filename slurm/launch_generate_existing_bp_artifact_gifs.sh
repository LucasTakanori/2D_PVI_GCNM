#!/bin/bash
# Backfill prediction-aligned GIFs for the six completed coordinate BP models.
# CPU-only: this reads saved Parquet/results and does not rerun either model.
#SBATCH --job-name=bp-artifact-gifs
#SBATCH --output=logs/bp-artifact-gifs_%A_%a.out
#SBATCH --error=logs/bp-artifact-gifs_%A_%a.err
#SBATCH --partition=ece_bst
#SBATCH --account=ece_bst
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32GB
#SBATCH --time=04:00:00

set -euo pipefail
module load gcc/11.2.0
REPO_ROOT="${REPO_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"
cd "${REPO_ROOT}"
source "${REPO_ROOT}/env/cluster.env"
source "$(dirname "$(dirname "${GCNM_PYTHON}")")/bin/activate"
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export MPLCONFIGDIR="/tmp/matplotlib-${SLURM_JOB_ID}-${SLURM_ARRAY_TASK_ID}"
mkdir -p "${MPLCONFIGDIR}"

TASK_ID="${SLURM_ARRAY_TASK_ID:?submit as array 0-5%6}"
SUBJECTS=(subject006 subject006 subject006 subject006 subject010 subject010)
OUTPUT_MODES=(waveform fiducials waveform fiducials waveform fiducials)
ARTIFACT_ROOTS=(
  artifacts/subject006_coordinate_direct_s1_s2_ds2_bp_v1
  artifacts/subject006_coordinate_direct_s1_s2_ds2_bp_v1
  artifacts/subject006_newton_coordinate_6ch_bp_v1
  artifacts/subject006_newton_coordinate_6ch_bp_v1
  artifacts/subject010_coordinate_direct_s1_s2_ds2_bp_v1
  artifacts/subject010_coordinate_direct_s1_s2_ds2_bp_v1
)
TARGETS=(
  gcnm-coordinate-3ch-crt-image-to-waveform
  gcnm-coordinate-3ch-crt-image-to-fiducials
  gcnm-newton_coordinate-6ch-crt-image-to-waveform
  gcnm-newton_coordinate-6ch-crt-image-to-fiducials
  gcnm-coordinate-3ch-crt-image-to-waveform
  gcnm-coordinate-3ch-crt-image-to-fiducials
)
COORDINATE_ROOTS=(
  gcnm_parquet/subject006_coordinate_direct_s1_s2_ds2_v1
  gcnm_parquet/subject006_coordinate_direct_s1_s2_ds2_v1
  gcnm_parquet/subject006_coordinate_direct_s1_s2_ds2_v1
  gcnm_parquet/subject006_coordinate_direct_s1_s2_ds2_v1
  gcnm_parquet/subject010_coordinate_direct_s1_s2_ds2_v1
  gcnm_parquet/subject010_coordinate_direct_s1_s2_ds2_v1
)

SUBJECT="${SUBJECTS[${TASK_ID}]}"
OUTPUT_MODE="${OUTPUT_MODES[${TASK_ID}]}"
ARTIFACT_MAIN="${REPO_ROOT}/${ARTIFACT_ROOTS[${TASK_ID}]}/${TARGETS[${TASK_ID}]}/main"
COORDINATE_ROOT="${REPO_ROOT}/${COORDINATE_ROOTS[${TASK_ID}]}"
REFERENCE_ROOT="${REPO_ROOT}/gcnm_parquet/us120_pilot_reference_image_v1"
SPLIT_MANIFEST="${REPO_ROOT}/data/splits/us120_pilot_subject006_subject010_mask05_v1.json"

echo "[$(date --iso-8601=seconds)] ${SUBJECT} ${OUTPUT_MODE} GIF backfill started"
python -u -m gcnm_pvi.bp_artifact_gifs \
  --artifact-main "${ARTIFACT_MAIN}" \
  --subject "${SUBJECT}" \
  --output-mode "${OUTPUT_MODE}" \
  --coordinate-root "${COORDINATE_ROOT}" \
  --reference-root "${REFERENCE_ROOT}" \
  --split-manifest "${SPLIT_MANIFEST}"
echo "[$(date --iso-8601=seconds)] ${SUBJECT} ${OUTPUT_MODE} GIF backfill complete"
