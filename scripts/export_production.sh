#!/usr/bin/env bash
# Export one bioz session with production-like preprocessing (scionova 01+03 PVI path).
#
# Usage:
#   bash scripts/export_production.sh SESSION_BIOZ_DIR OUT_DIR
#
# Full session (all frames):
#   bash scripts/export_production.sh \
#     "/path/to/private/raw/subject/baseline/session/bioz" \
#     "/path/to/private/gcnm_export/subject_baseline_full"
#
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=./_common.sh
source "${SCRIPT_DIR}/_common.sh"

SESSION_DIR="${1:-${GCNM_SESSION_DIR:-}}"
OUT_DIR="${2:-${GCNM_OUT_DIR:-}}"
CONFIG="${GCNM_CONFIG:-${GCNM_ROOT}/configs/finger_pvi08_production.yaml}"
MAX_FRAMES="${GCNM_MAX_FRAMES:-}"
STRIDE="${GCNM_STRIDE:-1}"

if [[ -z "${SESSION_DIR}" || -z "${OUT_DIR}" ]]; then
  echo "Usage: bash scripts/export_production.sh SESSION_BIOZ_DIR OUT_DIR" >&2
  exit 1
fi

if [[ ! -f "${CONFIG}" ]]; then
  echo "ERROR: config not found: ${CONFIG}" >&2
  exit 1
fi

# Ensure 40×40 mappings exist
MAPPINGS=$("${GCNM_PYTHON}" -c "import yaml; from pathlib import Path; c=yaml.safe_load(open('${CONFIG}')); b=Path('${CONFIG}').parent; p=Path(c['mappings_h5']); print(str((b/p).resolve() if not p.is_absolute() else p))")
if [[ ! -f "${MAPPINGS}" ]]; then
  echo "Mappings not found: ${MAPPINGS}"
  echo "Run first: bash scripts/build_m2i_40.sh"
  exit 1
fi

ARGS=(--config "${CONFIG}" --session-dir "${SESSION_DIR}" --out-dir "${OUT_DIR}" --stride "${STRIDE}")
if [[ -n "${MAX_FRAMES}" ]]; then
  ARGS+=(--max-frames "${MAX_FRAMES}")
fi

echo "=== Production export ==="
echo "session=${SESSION_DIR}"
echo "out=${OUT_DIR}"
echo "config=${CONFIG}"

"${GCNM_PYTHON}" -m gcnm_pvi.export_production_session "${ARGS[@]}"
