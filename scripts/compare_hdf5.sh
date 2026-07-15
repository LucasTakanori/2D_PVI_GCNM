#!/usr/bin/env bash
# Compare predicted img.npy against fundational_pvi HDF5 reference.
#
# Usage:
#   bash scripts/compare_hdf5.sh /path/to/img.npy /path/to/session.h5
#
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=./_common.sh
source "${SCRIPT_DIR}/_common.sh"

PRED_IMG="${1:-${GCNM_PRED_IMG:-}}"
H5_PATH="${2:-${GCNM_H5_PATH:-}}"
COMPONENT="${GCNM_H5_COMPONENT:-pviHP}"

if [[ -z "${PRED_IMG}" || -z "${H5_PATH}" ]]; then
  echo "Usage: bash scripts/compare_hdf5.sh PRED_IMG_NPY H5_PATH" >&2
  exit 1
fi

echo "=== GCNM-PVI HDF5 comparison ==="
echo "pred=${PRED_IMG}"
echo "ref=${H5_PATH}"

"${GCNM_PYTHON}" -m gcnm_pvi.compare_to_hdf5 \
  --pred-img-npy "${PRED_IMG}" \
  --h5 "${H5_PATH}" \
  --component "${COMPONENT}"
