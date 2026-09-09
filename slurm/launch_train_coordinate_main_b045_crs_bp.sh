#!/bin/bash
# One native pvi_ml CRS experiment per task, mirroring the frozen CRT matrix.
#SBATCH --job-name=coord91-crs-bp
#SBATCH --output=logs/coord91-crs-bp_%A_%a.out
#SBATCH --error=logs/coord91-crs-bp_%A_%a.err
#SBATCH --partition=ece_bst
#SBATCH --account=ece_bst
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=250GB
#SBATCH --time=10-00:00:00
#SBATCH --gres=gpu:1

set -euo pipefail
if [[ "${BP_CPU_POSTPROCESS:-0}" != "1" ]]; then
  module load CUDA/12.9.0
fi
module load gcc/11.2.0
REPO_ROOT="${REPO_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"
cd "${REPO_ROOT}"
source "${REPO_ROOT}/env/cluster.env"
source "$(dirname "$(dirname "${GCNM_PYTHON}")")/bin/activate"
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"
export OMP_NUM_THREADS="${BP_OMP_THREADS:-8}"
export MKL_NUM_THREADS="${BP_OMP_THREADS:-8}"
export OPENBLAS_NUM_THREADS="${BP_OMP_THREADS:-8}"
TASK_MANIFEST="${TASK_MANIFEST:?set TASK_MANIFEST}"
IFS=$'\t' read -r TASK_ID SUBJECT FAMILY ARCHITECTURE OUTPUT_MODE CHANNEL_MODE COORDINATE_REL REFERENCE_REGISTRY_REL TARGET MASK_KEY SEED \
  < <(sed -n "$((SLURM_ARRAY_TASK_ID + 2))p" "${TASK_MANIFEST}")
if [[ "${TASK_ID:-}" != "${SLURM_ARRAY_TASK_ID}" || "${ARCHITECTURE}" != "crs" || "${MASK_KEY}" != "mask05" ]]; then
  echo "invalid CRS task-manifest row ${SLURM_ARRAY_TASK_ID}" >&2
  exit 2
fi
COORDINATE_ROOT="${REPO_ROOT}/${COORDINATE_REL}"
REFERENCE_REGISTRY="${REPO_ROOT}/${REFERENCE_REGISTRY_REL}"
common=(
  --pvi-ml-root "${PVI_ML_ROOT:?set PVI_ML_ROOT}"
  --parquet-root "${COORDINATE_ROOT}"
  --gif-reference-hdf5-registry "${REFERENCE_REGISTRY}"
  --split-manifest "${SPLIT_MANIFEST:?set SPLIT_MANIFEST}"
  --subject "${SUBJECT}" --family "${FAMILY}" --architecture crs
  --output-mode "${OUTPUT_MODE}" --channel-mode "${CHANNEL_MODE}"
  --artifact-root "${ARTIFACT_ROOT:?set ARTIFACT_ROOT}"
  --max-epochs 5000 --num-workers "${BP_NUM_WORKERS:-8}"
  --seed "${SEED}" --deterministic
  --defer-artifact-gifs
)
if [[ "${FAMILY}" == "newton_coordinate" ]]; then
  common+=(--reference-hdf5-registry "${REFERENCE_REGISTRY}")
fi
if [[ "${BP_POSTPROCESS_ONLY:-0}" == "1" ]]; then
  common+=(--postprocess-only)
fi
echo "[$(date --iso-8601=seconds)] task=${TASK_ID} ${SUBJECT} ${TARGET} seed=${SEED}"
python -u -m gcnm_pvi.train_pvi_bp "${common[@]}"
echo "[$(date --iso-8601=seconds)] task=${TASK_ID} complete"
