#!/usr/bin/env bash
# Prepare, export all 91 subjects in parallel, validate, and archive.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"
source "${REPO_ROOT}/env/cluster.env"

OUTPUT_ROOT="${OUTPUT_ROOT:-${REPO_ROOT}/exports/all_patients_64ch_six_channel_10beats_v1}"
ARCHIVE_PATH="${ARCHIVE_PATH:-${OUTPUT_ROOT}.zip}"
COORDINATE_ROOT="${COORDINATE_ROOT:-${REPO_ROOT}/gcnm_parquet/coordinate_direct_main_b045_v1}"
REGISTRY="${REGISTRY:-${REPO_ROOT}/data/registries/main_b045_v1.json}"
ARRAY_CONCURRENCY="${ARRAY_CONCURRENCY:-16}"

for path in "${OUTPUT_ROOT}" "${ARCHIVE_PATH}" "${ARCHIVE_PATH}.sha256"; do
  if [[ -e "${path}" ]]; then
    echo "refusing to overwrite existing output: ${path}" >&2
    exit 2
  fi
done
mkdir -p logs

"${GCNM_PYTHON}" -u scripts/export_all_patient_six_channel_images.py prepare \
  --coordinate-root "${COORDINATE_ROOT}" \
  --registry "${REGISTRY}" \
  --output-root "${OUTPUT_ROOT}"

array_job="$(
  sbatch --parsable \
    --array="0-90%${ARRAY_CONCURRENCY}" \
    --export="ALL,REPO_ROOT=${REPO_ROOT},OUTPUT_ROOT=${OUTPUT_ROOT}" \
    slurm/launch_export_all_patient_six_channel_images.sh
)"
array_id="${array_job%%;*}"

archive_job="$(
  sbatch --parsable \
    --dependency="afterok:${array_id}" \
    --export="ALL,REPO_ROOT=${REPO_ROOT},OUTPUT_ROOT=${OUTPUT_ROOT},ARCHIVE_PATH=${ARCHIVE_PATH}" \
    slurm/launch_finalize_all_patient_six_channel_images.sh
)"

printf 'patient export array: %s\n' "${array_job}"
printf 'validation and archive: %s\n' "${archive_job}"
printf 'folder: %s\n' "${OUTPUT_ROOT}"
printf 'archive: %s\n' "${ARCHIVE_PATH}"
