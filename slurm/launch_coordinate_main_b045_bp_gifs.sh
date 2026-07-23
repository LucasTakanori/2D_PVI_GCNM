#!/bin/bash
# CPU-only GIF rendering after all native pvi_ml training artifacts exist.
#SBATCH --job-name=coord91-bgifs
#SBATCH --output=logs/coord91-bgifs_%A_%a.out
#SBATCH --error=logs/coord91-bgifs_%A_%a.err
#SBATCH --partition=ece_bst
#SBATCH --account=ece_bst
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64GB
#SBATCH --time=10-00:00:00

set -euo pipefail
module load gcc/11.2.0
REPO_ROOT="${REPO_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"
cd "${REPO_ROOT}"
source "${REPO_ROOT}/env/cluster.env"
source "$(dirname "$(dirname "${GCNM_PYTHON}")")/bin/activate"
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8
export MPLCONFIGDIR="${TMPDIR:-/tmp}/coord91-bgifs-${SLURM_JOB_ID}-${SLURM_ARRAY_TASK_ID}"
TASK_MANIFEST="${TASK_MANIFEST:?set TASK_MANIFEST}"
IFS=$'\t' read -r TASK_ID SUBJECT FAMILY ARCHITECTURE OUTPUT_MODE CHANNEL_MODE COORDINATE_REL REFERENCE_REGISTRY_REL TARGET MASK_KEY SEED \
  < <(sed -n "$((SLURM_ARRAY_TASK_ID + 2))p" "${TASK_MANIFEST}")
if [[ "${TASK_ID:-}" != "${SLURM_ARRAY_TASK_ID}" ]]; then
  echo "task-manifest row mismatch" >&2
  exit 2
fi
ARTIFACT_MAIN="${ARTIFACT_ROOT:?set ARTIFACT_ROOT}/${TARGET}/main"
python -u -m gcnm_pvi.bp_artifact_gifs \
  --artifact-main "${ARTIFACT_MAIN}" --subject "${SUBJECT}" \
  --output-mode "${OUTPUT_MODE}" \
  --coordinate-root "${REPO_ROOT}/${COORDINATE_REL}" \
  --reference-registry "${REPO_ROOT}/${REFERENCE_REGISTRY_REL}" \
  --split-manifest "${SPLIT_MANIFEST:?set SPLIT_MANIFEST}"
