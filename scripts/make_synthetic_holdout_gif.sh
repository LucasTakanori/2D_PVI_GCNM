#!/usr/bin/env bash
# Build the synthetic holdout PVI/GCNM comparison GIF.
#
# Usage:
#   bash scripts/make_synthetic_holdout_gif.sh
#   SAMPLES="0 1 2" bash scripts/make_synthetic_holdout_gif.sh
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=./_common.sh
source "${SCRIPT_DIR}/_common.sh"

EXPERIMENT="${EXPERIMENT:-faithful_background025_hom_seed0}"
SAMPLES="${SAMPLES:-0 1 2}"
OUT_FILE="${GCNM_ROOT}/reports/gcnm_pvi_latex/figures/synthetic_holdout_three_phantoms.gif"

export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/gcnm-mpl}"
"${GCNM_PYTHON}" "${GCNM_ROOT}/scripts/make_subject006_pvi_gcnm_gif.py" \
  --split synthetic \
  --experiment "${EXPERIMENT}" \
  --samples ${SAMPLES} \
  --out "${OUT_FILE}"
