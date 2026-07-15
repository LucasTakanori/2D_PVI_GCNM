#!/usr/bin/env bash
# Build GCNM training packs from ScioSpec .eit session or vmeas.npy.
#
# Usage examples:
#   bash scripts/export_dataset.sh /path/to/session /path/to/out
#   GCNM_VMEAS_NPY=/path/vmeas.npy bash scripts/export_dataset.sh "" /path/to/out
#
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=./_common.sh
source "${SCRIPT_DIR}/_common.sh"

SESSION_DIR="${1:-${GCNM_SESSION_DIR:-}}"
OUT_DIR="${2:-${GCNM_OUT_DIR:-}}"
CONFIG="${GCNM_CONFIG:-${GCNM_ROOT}/configs/finger_pvi08.yaml}"
VMEAS_NPY="${GCNM_VMEAS_NPY:-}"
MAX_FRAMES="${GCNM_MAX_FRAMES:-}"
STRIDE="${GCNM_STRIDE:-1}"
COMPONENT="${GCNM_COMPONENT:-hp}"

if [[ -z "${OUT_DIR}" ]]; then
  echo "Usage: bash scripts/export_dataset.sh SESSION_DIR OUT_DIR" >&2
  echo "   or: GCNM_VMEAS_NPY=... bash scripts/export_dataset.sh '' OUT_DIR" >&2
  exit 1
fi

ARGS=(--config "${CONFIG}" --out-dir "${OUT_DIR}" --component "${COMPONENT}" --stride "${STRIDE}")
if [[ -n "${VMEAS_NPY}" ]]; then
  ARGS+=(--vmeas-npy "${VMEAS_NPY}")
elif [[ -n "${SESSION_DIR}" ]]; then
  ARGS+=(--session-dir "${SESSION_DIR}")
else
  echo "ERROR: provide SESSION_DIR or set GCNM_VMEAS_NPY" >&2
  exit 1
fi
if [[ -n "${MAX_FRAMES}" ]]; then
  ARGS+=(--max-frames "${MAX_FRAMES}")
fi

echo "=== GCNM-PVI dataset export ==="
echo "out=${OUT_DIR}"

"${GCNM_PYTHON}" -m gcnm_pvi.export_gcnm_dataset "${ARGS[@]}"
