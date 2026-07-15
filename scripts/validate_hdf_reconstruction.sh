#!/usr/bin/env bash
# Directly validate the subject-sized production inverse against pviHP HDF images.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=./_common.sh
source "${SCRIPT_DIR}/_common.sh"

H5_PATH="${1:-}"
OUT_JSON="${2:-${GCNM_ROOT}/data/subject006_hdf_reconstruction_validation.json}"
CONFIG="${GCNM_CONFIG:-${GCNM_ROOT}/configs/subject006_pvi08_production.yaml}"
FRAMES="${GCNM_VALIDATION_FRAMES:-500}"

if [[ -z "${H5_PATH}" ]]; then
  echo "Usage: bash scripts/validate_hdf_reconstruction.sh H5_PATH [OUT_JSON]" >&2
  exit 1
fi

"${GCNM_PYTHON}" -m gcnm_pvi.validate_hdf_reconstruction \
  --config "${CONFIG}" \
  --h5 "${H5_PATH}" \
  --frames "${FRAMES}" \
  --out-json "${OUT_JSON}"
