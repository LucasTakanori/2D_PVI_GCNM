#!/bin/bash
# One independent BP experiment per task; the array is capped at four GPUs.
#SBATCH --job-name=us120-bp-pilot
#SBATCH --output=logs/us120-bp-pilot_%A_%a.out
#SBATCH --error=logs/us120-bp-pilot_%A_%a.err
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
export NUMEXPR_NUM_THREADS="${SLURM_CPUS_PER_TASK}"

TASK_MANIFEST="${TASK_MANIFEST:?set TASK_MANIFEST}"
IFS=$'\t' read -r TASK_ID SUBJECT FAMILY ARCHITECTURE OUTPUT_MODE CHANNEL_MODE PARQUET_ROOT TARGET MASK_KEY \
  < <(sed -n "$((SLURM_ARRAY_TASK_ID + 2))p" "${TASK_MANIFEST}")
MASK_KEY="${MASK_KEY%$'\r'}"
if [[ -z "${TASK_ID:-}" || "${TASK_ID}" != "${SLURM_ARRAY_TASK_ID}" ]]; then
  echo "task-manifest row mismatch for SLURM_ARRAY_TASK_ID=${SLURM_ARRAY_TASK_ID}" >&2
  exit 2
fi
if [[ "${MASK_KEY}" != "mask05" ]]; then
  echo "refusing non-mask05 task: ${MASK_KEY}" >&2
  exit 2
fi

echo "task=${TASK_ID} representation=gcnm subject=${SUBJECT} target=${TARGET}"
nvidia-smi
common=(
  --pvi-ml-root "${PVI_ML_ROOT:?set PVI_ML_ROOT}"
  --split-manifest "${SPLIT_MANIFEST:?set SPLIT_MANIFEST}"
  --subject "${SUBJECT}"
  --architecture "${ARCHITECTURE}"
  --output-mode "${OUTPUT_MODE}"
  --artifact-root "${ARTIFACT_ROOT:?set ARTIFACT_ROOT}"
  --max-epochs "${MAX_EPOCHS:-5000}"
  --num-workers "${NUM_WORKERS:-8}"
)
if [[ "${POSTPROCESS_ONLY:-0}" == 1 ]]; then
  common+=(--postprocess-only)
  echo "mode=postprocess-only (reuse finalized checkpoint; no training epochs)"
fi
python -u -m gcnm_pvi.train_pvi_bp \
  --parquet-root "${PARQUET_ROOT}" --family "${FAMILY}" \
  --channel-mode "${CHANNEL_MODE}" "${common[@]}"
