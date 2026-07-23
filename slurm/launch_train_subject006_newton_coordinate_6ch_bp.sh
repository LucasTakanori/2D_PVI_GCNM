#!/bin/bash
# Two subject006 CRT experiments using Newton 3ch + coordinate-direct 3ch.
#SBATCH --job-name=s006-newton-coord-6ch
#SBATCH --output=logs/s006-newton-coord-6ch_%A_%a.out
#SBATCH --error=logs/s006-newton-coord-6ch_%A_%a.err
#SBATCH --partition=ece_bst
#SBATCH --account=ece_bst
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=250GB
#SBATCH --time=10-00:00:00
#SBATCH --gres=gpu:1

set -euo pipefail
module load CUDA/12.9.0
module load gcc/11.2.0
REPO_ROOT="${REPO_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"
cd "${REPO_ROOT}"
source "${REPO_ROOT}/env/cluster.env"
source "$(dirname "$(dirname "${GCNM_PYTHON}")")/bin/activate"
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK}"

TASK_ID="${SLURM_ARRAY_TASK_ID:?submit as array 0-1%2}"
OUTPUT_MODES=(waveform fiducials)
OUTPUT_MODE="${OUTPUT_MODES[${TASK_ID}]}"
COORDINATE_ROOT="${COORDINATE_ROOT:-${REPO_ROOT}/gcnm_parquet/subject006_coordinate_direct_s1_s2_ds2_v1}"
REFERENCE_ROOT="${REFERENCE_ROOT:-${REPO_ROOT}/gcnm_parquet/us120_pilot_reference_image_v1}"
SPLIT_MANIFEST="${SPLIT_MANIFEST:-${REPO_ROOT}/data/splits/us120_pilot_subject006_subject010_mask05_v1.json}"
ARTIFACT_ROOT="${ARTIFACT_ROOT:-${REPO_ROOT}/artifacts/subject006_newton_coordinate_6ch_bp_v1}"
PVI_ML_ROOT="${PVI_ML_ROOT:-$(dirname "${REPO_ROOT}")/gia_bao/pvi_ml}"
for required in \
  "${COORDINATE_ROOT}/validation.json" \
  "${REFERENCE_ROOT}/manifest.json" \
  "${SPLIT_MANIFEST}"; do
  [[ -f "${required}" ]] || { echo "missing paired input: ${required}" >&2; exit 2; }
done

echo "[$(date --iso-8601=seconds)] subject006 CRT ${OUTPUT_MODE} Newton3+coordinate3"
python -u -m gcnm_pvi.train_pvi_bp \
  --pvi-ml-root "${PVI_ML_ROOT}" \
  --parquet-root "${COORDINATE_ROOT}" \
  --reference-parquet-root "${REFERENCE_ROOT}" \
  --gif-reference-parquet-root "${REFERENCE_ROOT}" \
  --split-manifest "${SPLIT_MANIFEST}" \
  --subject subject006 --family newton_coordinate --architecture crt \
  --output-mode "${OUTPUT_MODE}" --channel-mode 6ch \
  --artifact-root "${ARTIFACT_ROOT}" --max-epochs 5000 --num-workers 8
echo "[$(date --iso-8601=seconds)] subject006 CRT ${OUTPUT_MODE} Newton3+coordinate3 complete"
