#!/bin/bash
# High-throughput CPU GIF rendering for future coordinate BP artifact jobs.
# The original launcher remains unchanged for already-submitted job arrays.
#SBATCH --job-name=coord91-bgifs-fast
#SBATCH --output=logs/coord91-bgifs-fast_%A_%a.out
#SBATCH --error=logs/coord91-bgifs-fast_%A_%a.err
#SBATCH --partition=ece_bst
#SBATCH --account=ece_bst
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8GB
#SBATCH --time=04:00:00

set -euo pipefail
module load gcc/11.2.0
REPO_ROOT="${REPO_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"
cd "${REPO_ROOT}"
source "${REPO_ROOT}/env/cluster.env"
source "$(dirname "$(dirname "${GCNM_PYTHON}")")/bin/activate"
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"
# Each worker is already an independent process. Prevent nested BLAS pools.
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export MPLBACKEND=Agg
export MPLCONFIGDIR="${TMPDIR:-/tmp}/coord91-bgifs-fast-${SLURM_JOB_ID}-${SLURM_ARRAY_TASK_ID}"
mkdir -p "${MPLCONFIGDIR}"

TASK_ID="${SLURM_ARRAY_TASK_ID:?submit as an array}"
TASK_MANIFEST="${TASK_MANIFEST:?set TASK_MANIFEST}"
IFS=$'\t' read -r MANIFEST_TASK_ID SUBJECT FAMILY ARCHITECTURE OUTPUT_MODE CHANNEL_MODE COORDINATE_REL REFERENCE_REGISTRY_REL TARGET MASK_KEY SEED \
  < <(sed -n "$((TASK_ID + 2))p" "${TASK_MANIFEST}")
if [[ "${MANIFEST_TASK_ID:-}" != "${TASK_ID}" ]]; then
  echo "task-manifest row mismatch" >&2
  exit 2
fi

ARTIFACT_MAIN="${ARTIFACT_ROOT:?set ARTIFACT_ROOT}/${TARGET}/main"
python -u -m gcnm_pvi.bp_artifact_gifs_parallel \
  --artifact-main "${ARTIFACT_MAIN}" \
  --subject "${SUBJECT}" \
  --output-mode "${OUTPUT_MODE}" \
  --coordinate-root "${REPO_ROOT}/${COORDINATE_REL}" \
  --reference-registry "${REPO_ROOT}/${REFERENCE_REGISTRY_REL}" \
  --split-manifest "${SPLIT_MANIFEST:?set SPLIT_MANIFEST}" \
  --workers "${SLURM_CPUS_PER_TASK}" \
  --scratch-root "${TMPDIR:-/tmp}"
