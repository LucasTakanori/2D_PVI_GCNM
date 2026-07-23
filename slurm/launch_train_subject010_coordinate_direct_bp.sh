#!/bin/bash
# Subject010 CRT waveform and fiducial experiments using [S1,S2,dS2/dt].
#SBATCH --job-name=coord-s010-bp
#SBATCH --output=logs/coord-s010-bp_%A_%a.out
#SBATCH --error=logs/coord-s010-bp_%A_%a.err
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
PARQUET_ROOT="${PARQUET_ROOT:-${REPO_ROOT}/gcnm_parquet/subject010_coordinate_direct_s1_s2_ds2_v1}"
GIF_REFERENCE_ROOT="${GIF_REFERENCE_ROOT:-${REPO_ROOT}/gcnm_parquet/us120_pilot_reference_image_v1}"
SPLIT_MANIFEST="${SPLIT_MANIFEST:-${REPO_ROOT}/data/splits/us120_pilot_subject006_subject010_mask05_v1.json}"
ARTIFACT_ROOT="${ARTIFACT_ROOT:-${REPO_ROOT}/artifacts/subject010_coordinate_direct_s1_s2_ds2_bp_v1}"
PVI_ML_ROOT="${PVI_ML_ROOT:-$(dirname "${REPO_ROOT}")/gia_bao/pvi_ml}"
[[ -f "${PARQUET_ROOT}/validation.json" ]] || { echo "validated subject010 Parquet missing" >&2; exit 2; }
[[ -f "${GIF_REFERENCE_ROOT}/manifest.json" ]] || { echo "GIF reference Parquet missing" >&2; exit 2; }

echo "[$(date --iso-8601=seconds)] subject010 CRT ${OUTPUT_MODE} [S1,S2,dS2/dt]"
python -u -m gcnm_pvi.train_pvi_bp \
  --pvi-ml-root "${PVI_ML_ROOT}" \
  --parquet-root "${PARQUET_ROOT}" \
  --gif-reference-parquet-root "${GIF_REFERENCE_ROOT}" \
  --split-manifest "${SPLIT_MANIFEST}" \
  --subject subject010 --family coordinate --architecture crt \
  --output-mode "${OUTPUT_MODE}" --channel-mode 3ch \
  --artifact-root "${ARTIFACT_ROOT}" --max-epochs 5000 --num-workers 8
echo "[$(date --iso-8601=seconds)] subject010 CRT ${OUTPUT_MODE} complete"
