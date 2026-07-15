#!/usr/bin/env bash
# Evaluate trained GCNM vs LM baseline on held-out phantoms.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=./_common.sh
source "${SCRIPT_DIR}/_common.sh"

CONFIG="${GCNM_CONFIG:-${GCNM_ROOT}/configs/finger_pvi08.yaml}"

echo "=== GCNM-PVI evaluation ==="
echo "config=${CONFIG}"

"${GCNM_PYTHON}" -m gcnm_pvi.test_gcnm --config "${CONFIG}" "$@"
