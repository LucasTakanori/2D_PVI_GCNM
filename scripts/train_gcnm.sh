#!/usr/bin/env bash
# Train GCNM on PVI 8-el meshes (synthetic phantoms or prebuilt NPZ dataset).
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=./_common.sh
source "${SCRIPT_DIR}/_common.sh"

CONFIG="${GCNM_CONFIG:-${GCNM_ROOT}/configs/finger_pvi08.yaml}"
SAMPLES_DIR="${GCNM_SAMPLES_DIR:-}"

ARGS=(--config "${CONFIG}")
if [[ -n "${SAMPLES_DIR}" ]]; then
  ARGS+=(--samples-dir "${SAMPLES_DIR}")
fi

echo "=== GCNM-PVI training ==="
echo "config=${CONFIG}"
if [[ -n "${SAMPLES_DIR}" ]]; then
  echo "samples=${SAMPLES_DIR}"
fi

"${GCNM_PYTHON}" -m gcnm_pvi.train_gcnm "${ARGS[@]}" "$@"
