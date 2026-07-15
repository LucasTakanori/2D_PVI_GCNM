#!/usr/bin/env bash
# Train GCNM on clean synthetic vascular conductivity truth.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=./_common.sh
source "${SCRIPT_DIR}/_common.sh"

CONFIG="${GCNM_CONFIG:-${GCNM_ROOT}/configs/subject006_anatomical_gcnm.yaml}"
DATASET_DIR="${1:-${GCNM_ROOT}/data/subject006_anatomical}"

MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/gcnm_mpl}" \
"${GCNM_PYTHON}" -m gcnm_pvi.train_anatomical_gcnm \
  --config "${CONFIG}" \
  --train "${DATASET_DIR}/train.npz" \
  --validation "${DATASET_DIR}/validation.npz" \
  "$@"
