#!/bin/bash
# Eight CRT baselines: 2 subjects x image/BioZ x waveform/fiducials.
#SBATCH --job-name=us120-ref-bp
#SBATCH --output=logs/us120-ref-bp_%A_%a.out
#SBATCH --error=logs/us120-ref-bp_%A_%a.err
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
IFS=$'\t' read -r TASK_ID SUBJECT INPUT_MODE OUTPUT_MODE PARQUET_REL TARGET MASK_KEY \
  < <(sed -n "$((SLURM_ARRAY_TASK_ID + 2))p" "${TASK_MANIFEST}")
if [[ "${TASK_ID:-}" != "${SLURM_ARRAY_TASK_ID}" || "${MASK_KEY}" != "mask05" ]]; then
  echo "invalid task manifest row for ${SLURM_ARRAY_TASK_ID}" >&2
  exit 2
fi
PARQUET_ROOT="${REPO_ROOT}/${PARQUET_REL}"
if [[ ! -f "${PARQUET_ROOT}/manifest.json" || -e "${PARQUET_ROOT}/_INCOMPLETE" ]]; then
  echo "reference Parquet root failed preflight: ${PARQUET_ROOT}" >&2
  exit 2
fi

echo "task=${TASK_ID} subject=${SUBJECT} input=${INPUT_MODE} output=${OUTPUT_MODE} root=${PARQUET_ROOT}"
python -u -m gcnm_pvi.train_reference_parquet_bp \
  --pvi-ml-root "${PVI_ML_ROOT:?set PVI_ML_ROOT}" \
  --parquet-root "${PARQUET_ROOT}" \
  --split-manifest "${SPLIT_MANIFEST:?set SPLIT_MANIFEST}" \
  --subject "${SUBJECT}" --input-mode "${INPUT_MODE}" \
  --output-mode "${OUTPUT_MODE}" \
  --artifact-root "${ARTIFACT_ROOT:?set ARTIFACT_ROOT}" \
  --max-epochs "${MAX_EPOCHS:-5000}" --num-workers "${NUM_WORKERS:-8}"
