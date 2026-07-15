#!/usr/bin/env bash
# Run GCNM inference on a ScioSpec session -> sigma + images.
#
# Usage:
#   bash scripts/infer_gcnm.sh /path/to/session /path/to/output
#
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=./_common.sh
source "${SCRIPT_DIR}/_common.sh"

SESSION_DIR="${1:-${GCNM_SESSION_DIR:-}}"
OUT_DIR="${2:-${GCNM_OUT_DIR:-}}"
CONFIG="${GCNM_CONFIG:-${GCNM_ROOT}/configs/finger_pvi08.yaml}"

if [[ -z "${SESSION_DIR}" || -z "${OUT_DIR}" ]]; then
  echo "Usage: bash scripts/infer_gcnm.sh SESSION_DIR OUT_DIR" >&2
  exit 1
fi

echo "=== GCNM-PVI inference ==="
echo "session=${SESSION_DIR}"
echo "out=${OUT_DIR}"

"${GCNM_PYTHON}" -m gcnm_pvi.infer_gcnm \
  --config "${CONFIG}" \
  --session-dir "${SESSION_DIR}" \
  --out-dir "${OUT_DIR}" \
  "$@"
