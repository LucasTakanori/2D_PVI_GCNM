#!/usr/bin/env bash
# Build 40×40 m2i mapping HDF5 from inverse mesh (interim until ring mesh arrives).
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=./_common.sh
source "${SCRIPT_DIR}/_common.sh"

CONFIG="${GCNM_CONFIG:-${GCNM_ROOT}/configs/finger_pvi08_production.yaml}"
IMG_SIZE="${GCNM_IMG_SIZE:-40}"
OUT="${GCNM_MAPPINGS_OUT:-}"

ARGS=(--config "${CONFIG}" --img-size "${IMG_SIZE}")
if [[ -n "${OUT}" ]]; then
  ARGS+=(--out "${OUT}")
fi

echo "=== Build m2i @ ${IMG_SIZE}×${IMG_SIZE} ==="
"${GCNM_PYTHON}" -m gcnm_pvi.build_mappings "${ARGS[@]}"
