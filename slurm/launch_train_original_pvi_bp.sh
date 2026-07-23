#!/bin/bash
#SBATCH --job-name=original-pvi-bp
#SBATCH --output=logs/original-pvi-bp_%A_%a.out
#SBATCH --error=logs/original-pvi-bp_%A_%a.err
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
TASK_MANIFEST="${TASK_MANIFEST:?set TASK_MANIFEST}"
IFS=$'\t' read -r TASK_ID SUBJECT ARCHITECTURE OUTPUT_MODE TARGET MASK_KEY \
  < <(sed -n "$((SLURM_ARRAY_TASK_ID + 2))p" "${TASK_MANIFEST}")
if [[ -z "${TASK_ID:-}" || "${TASK_ID}" != "${SLURM_ARRAY_TASK_ID}" ]]; then
  echo "task-manifest row mismatch for SLURM_ARRAY_TASK_ID=${SLURM_ARRAY_TASK_ID}" >&2
  exit 2
fi
if [[ "${MASK_KEY}" != "mask05" ]]; then
  echo "refusing non-mask05 task: ${MASK_KEY}" >&2
  exit 2
fi
echo "task=${TASK_ID} subject=${SUBJECT} target=${TARGET} mask=${MASK_KEY}"
python -u -m gcnm_pvi.train_original_pvi_bp \
  --pvi-ml-root "${PVI_ML_ROOT:?set PVI_ML_ROOT}" \
  --registry "${REGISTRY:?set REGISTRY}" \
  --split-manifest "${SPLIT_MANIFEST:?set SPLIT_MANIFEST}" \
  --subject "${SUBJECT}" \
  --architecture "${ARCHITECTURE}" \
  --output-mode "${OUTPUT_MODE}" \
  --artifact-root "${ARTIFACT_ROOT:?set ARTIFACT_ROOT}" \
  --max-epochs "${MAX_EPOCHS:-5000}" --num-workers "${NUM_WORKERS:-8}"
