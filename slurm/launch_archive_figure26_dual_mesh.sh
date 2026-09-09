#!/bin/bash
#SBATCH --job-name=fig26-zip
#SBATCH --output=logs/fig26-zip_%j.out
#SBATCH --error=logs/fig26-zip_%j.err
#SBATCH --partition=ece_bst
#SBATCH --account=ece_bst
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64GB
#SBATCH --time=1-00:00:00

set -euo pipefail

REPO_ROOT="${REPO_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"
EXPORT_ROOT="${REPO_ROOT}/exports"
SOURCE_NAME="${FIGURE26_SOURCE_NAME:-figure26_dual_mesh_sensitivity_3beats_v1}"
SOURCE_ROOT="${EXPORT_ROOT}/${SOURCE_NAME}"
FINAL_ARCHIVE="${EXPORT_ROOT}/${SOURCE_NAME}.zip"
TEMP_ARCHIVE="${EXPORT_ROOT}/${SOURCE_NAME}.partial.zip"
CHECKSUM="${FINAL_ARCHIVE}.sha256"

if [[ ! -f "${SOURCE_ROOT}/_SUCCESS" ]]; then
  echo "validated source marker is missing: ${SOURCE_ROOT}/_SUCCESS" >&2
  exit 2
fi
if [[ -e "${FINAL_ARCHIVE}" || -e "${TEMP_ARCHIVE}" || -e "${CHECKSUM}" ]]; then
  echo "refusing to overwrite an existing archive, partial archive, or checksum" >&2
  exit 3
fi

cd "${EXPORT_ROOT}"
python3 -u "${REPO_ROOT}/scripts/create_parallel_zip.py" \
  --source "${SOURCE_ROOT}" --output "${TEMP_ARCHIVE}" \
  --workers 16 --in-flight 64
mv "${TEMP_ARCHIVE}" "${FINAL_ARCHIVE}"
sha256sum "${FINAL_ARCHIVE}" > "${CHECKSUM}"

ls -lh "${FINAL_ARCHIVE}" "${CHECKSUM}"
cat "${CHECKSUM}"
