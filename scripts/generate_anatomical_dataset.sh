#!/usr/bin/env bash
# Generate clean-truth, domain-randomized US120 vascular phantoms.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=./_common.sh
source "${SCRIPT_DIR}/_common.sh"

CONFIG="${GCNM_CONFIG:-${GCNM_ROOT}/configs/subject006_pvi08_production.yaml}"
OUT_DIR="${1:-${GCNM_ROOT}/data/subject006_anatomical}"
TRAIN="${GCNM_SYNTH_TRAIN:-64}"
VALIDATION="${GCNM_SYNTH_VALIDATION:-16}"
TEST="${GCNM_SYNTH_TEST:-16}"
SIMULATION_MODE="${GCNM_SYNTH_MODE:-linearized}"
JACOBIAN_BANK_SIZE="${GCNM_JACOBIAN_BANK_SIZE:-3}"

MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/gcnm_mpl}" \
"${GCNM_PYTHON}" -m gcnm_pvi.generate_anatomical_dataset \
  --config "${CONFIG}" \
  --out-dir "${OUT_DIR}" \
  --train "${TRAIN}" \
  --validation "${VALIDATION}" \
  --test "${TEST}" \
  --simulation-mode "${SIMULATION_MODE}" \
  --jacobian-bank-size "${JACOBIAN_BANK_SIZE}"
