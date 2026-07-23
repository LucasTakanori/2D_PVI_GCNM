#!/bin/bash
# CPU-only: regenerate results/statistics/GIFs from selected best checkpoints.
#SBATCH --job-name=repair-s010-6ch-artifacts
#SBATCH --output=logs/repair-s010-6ch-artifacts_%A_%a.out
#SBATCH --error=logs/repair-s010-6ch-artifacts_%A_%a.err
#SBATCH --partition=ece_bst
#SBATCH --account=ece_bst
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=125GB
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

TASK_ID="${SLURM_ARRAY_TASK_ID:?submit as array 0-1%2}"
OUTPUT_MODES=(waveform fiducials)
OUTPUT_MODE="${OUTPUT_MODES[${TASK_ID}]}"
PVI_ML_ROOT="$(dirname "${REPO_ROOT}")/gia_bao/pvi_ml"
COORDINATE_ROOT="${REPO_ROOT}/gcnm_parquet/subject010_coordinate_direct_s1_s2_ds2_v1"
REFERENCE_ROOT="${REPO_ROOT}/gcnm_parquet/us120_pilot_reference_image_v1"
SPLIT_MANIFEST="${REPO_ROOT}/data/splits/us120_pilot_subject006_subject010_mask05_v1.json"
ARTIFACT_ROOT="${REPO_ROOT}/artifacts/subject010_newton_coordinate_6ch_bp_v1"

python -u -m gcnm_pvi.train_pvi_bp \
  --pvi-ml-root "${PVI_ML_ROOT}" \
  --parquet-root "${COORDINATE_ROOT}" \
  --reference-parquet-root "${REFERENCE_ROOT}" \
  --gif-reference-parquet-root "${REFERENCE_ROOT}" \
  --split-manifest "${SPLIT_MANIFEST}" \
  --subject subject010 --family newton_coordinate --architecture crt \
  --output-mode "${OUTPUT_MODE}" --channel-mode 6ch \
  --artifact-root "${ARTIFACT_ROOT}" --max-epochs 5000 --num-workers 8 \
  --postprocess-only
