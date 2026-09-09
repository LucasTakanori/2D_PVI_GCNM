#!/usr/bin/env bash
# Validate the complete export and create the requested Zip64 archive.
#SBATCH --job-name=sixch-zip
#SBATCH --output=logs/sixch-zip_%j.out
#SBATCH --error=logs/sixch-zip_%j.err
#SBATCH --partition=ece_bst
#SBATCH --account=ece_bst
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16GB
#SBATCH --time=12:00:00

set -euo pipefail

REPO_ROOT="${REPO_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"
OUTPUT_ROOT="${OUTPUT_ROOT:?OUTPUT_ROOT must be set}"
ARCHIVE_PATH="${ARCHIVE_PATH:?ARCHIVE_PATH must be set}"
cd "${REPO_ROOT}"
source "${REPO_ROOT}/env/cluster.env"
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"

"${GCNM_PYTHON}" -u scripts/export_all_patient_six_channel_images.py \
  finalize --output-root "${OUTPUT_ROOT}"

archive_parent="$(dirname "${OUTPUT_ROOT}")"
archive_name="$(basename "${OUTPUT_ROOT}")"
if [[ -e "${ARCHIVE_PATH}" ]]; then
  echo "refusing to overwrite archive: ${ARCHIVE_PATH}" >&2
  exit 3
fi
cd "${archive_parent}"
zip -q -1 -r "${ARCHIVE_PATH}" "${archive_name}"
sha256sum "${ARCHIVE_PATH}" > "${ARCHIVE_PATH}.sha256"
ls -lh "${ARCHIVE_PATH}" "${ARCHIVE_PATH}.sha256"
