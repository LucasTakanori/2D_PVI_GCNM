#!/bin/bash
# CPU-only archived Newton image export, one subject per array task.
#SBATCH --job-name=ref91-export
#SBATCH --output=logs/ref91-export_%A_%a.out
#SBATCH --error=logs/ref91-export_%A_%a.err
#SBATCH --partition=ece_bst
#SBATCH --account=ece_bst
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=12GB
#SBATCH --time=10-00:00:00

set -euo pipefail
module load gcc/11.2.0
REPO_ROOT="${REPO_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"
cd "${REPO_ROOT}"
source "${REPO_ROOT}/env/cluster.env"
source "$(dirname "$(dirname "${GCNM_PYTHON}")")/bin/activate"
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
SUBJECT_MANIFEST="${SUBJECT_MANIFEST:?set SUBJECT_MANIFEST}"
SUBJECT="$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" "${SUBJECT_MANIFEST}")"
if [[ ! "${SUBJECT}" =~ ^subject[0-9]{3}$ ]]; then
  echo "invalid subject manifest row for task ${SLURM_ARRAY_TASK_ID}" >&2
  exit 2
fi
OUTPUT_ROOT="${REFERENCE_ROOT:?set REFERENCE_ROOT}/subject_parts/${SUBJECT}"
if [[ -f "${OUTPUT_ROOT}/manifest.json" && ! -e "${OUTPUT_ROOT}/_INCOMPLETE" ]]; then
  echo "reuse complete ${SUBJECT} reference part"
  exit 0
fi
if [[ -e "${OUTPUT_ROOT}" ]]; then
  echo "partial immutable reference export exists for ${SUBJECT}" >&2
  exit 2
fi
python -u -m gcnm_pvi.reference_parquet \
  --registry "${REPO_ROOT}/data/registries/main_b045_v1.json" \
  --output-root "${OUTPUT_ROOT}" --subject "${SUBJECT}" \
  --input-mode img --shard-rows 32
