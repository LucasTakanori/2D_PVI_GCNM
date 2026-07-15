#!/usr/bin/env bash
# Validate production export vs masked HDF5 (grid-matched comparison).
#
# Usage:
#   bash scripts/validate_session.sh EXPORT_DIR H5_PATH
#
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=./_common.sh
source "${SCRIPT_DIR}/_common.sh"

EXPORT_DIR="${1:-${GCNM_EXPORT_DIR:-}}"
H5_PATH="${2:-${GCNM_H5_PATH:-}}"
COMPONENT="${GCNM_H5_COMPONENT:-pviHP}"
MAX_FRAMES="${GCNM_MAX_FRAMES:-500}"

if [[ -z "${EXPORT_DIR}" || -z "${H5_PATH}" ]]; then
  echo "Usage: bash scripts/validate_session.sh EXPORT_DIR H5_PATH" >&2
  exit 1
fi

echo "=== Validate export vs HDF5 ==="
echo "export=${EXPORT_DIR}"
echo "h5=${H5_PATH}"

"${GCNM_PYTHON}" -m gcnm_pvi.validate_session \
  --export-dir "${EXPORT_DIR}" \
  --h5 "${H5_PATH}" \
  --component "${COMPONENT}" \
  --max-frames "${MAX_FRAMES}"
