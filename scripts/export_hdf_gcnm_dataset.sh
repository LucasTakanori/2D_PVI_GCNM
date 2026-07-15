#!/usr/bin/env bash
# Export subject006 aligned HDF frames into trial-disjoint GCNM packs.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=./_common.sh
source "${SCRIPT_DIR}/_common.sh"

H5_PATH="${1:-}"
OUT_DIR="${2:-${GCNM_ROOT}/data/subject006_gcnm_hdf}"
CONFIG="${GCNM_CONFIG:-${GCNM_ROOT}/configs/subject006_pvi08_production.yaml}"
PHASE_STRIDE="${GCNM_PHASE_STRIDE:-5}"

if [[ -z "${H5_PATH}" ]]; then
  echo "Usage: bash scripts/export_hdf_gcnm_dataset.sh H5_PATH [OUT_DIR]" >&2
  exit 1
fi

"${GCNM_PYTHON}" -m gcnm_pvi.export_hdf_gcnm_dataset \
  --config "${CONFIG}" \
  --h5 "${H5_PATH}" \
  --out-dir "${OUT_DIR}" \
  --phase-stride "${PHASE_STRIDE}"
