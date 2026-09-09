#!/bin/bash
#SBATCH --job-name=synth-pf-final
#SBATCH --output=logs/synth-pf-final_%j.out
#SBATCH --error=logs/synth-pf-final_%j.err
#SBATCH --partition=ece_bst
#SBATCH --account=ece_bst
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64GB
#SBATCH --time=02:00:00

set -euo pipefail

REPO_ROOT="${REPO_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"
cd "${REPO_ROOT}"
source "${REPO_ROOT}/env/cluster.env"
export PYTHONPATH="${REPO_ROOT}:${PVI_SOLVER_ROOT}:${PYTHONPATH:-}"

OUTPUT_PARENT="${OUTPUT_PARENT:?set OUTPUT_PARENT to the completed ring export root}"
REFERENCE_ROOT="${REFERENCE_ROOT:?set REFERENCE_ROOT to the prior matched export}"
ARCHIVE_PATH="${ARCHIVE_PATH:-${OUTPUT_PARENT}.zip}"

if [[ -e "${ARCHIVE_PATH}" || -e "${ARCHIVE_PATH}.sha256" ]]; then
  echo "refusing to overwrite existing archive output" >&2
  exit 2
fi

"${GCNM_PYTHON}" -u scripts/finalize_gcnm_training_mesh_export.py \
  --root "${OUTPUT_PARENT}" \
  --reference-root "${REFERENCE_ROOT}"

"${GCNM_PYTHON}" -u scripts/create_parallel_zip.py \
  --source "${OUTPUT_PARENT}" \
  --output "${ARCHIVE_PATH}" \
  --workers "${SLURM_CPUS_PER_TASK}" \
  --in-flight 64

sha256sum "${ARCHIVE_PATH}" >"${ARCHIVE_PATH}.sha256"
