#!/usr/bin/env bash
# Build the train-split PVI vs GCNM GIF after inference completes.
#
# Usage:
#   bash scripts/make_train_split_gif.sh
#   PERIODS="1 2 3" bash scripts/make_train_split_gif.sh
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=./_common.sh
source "${SCRIPT_DIR}/_common.sh"

EXPERIMENT="${EXPERIMENT:-faithful_background025_hom_seed0}"
PERIODS="${PERIODS:-1 2 3}"
PREDICTIONS="${GCNM_ROOT}/data/faithful_results/${EXPERIMENT}/evaluation_real_pvi_train/predictions.npz"

if [[ ! -f "${PREDICTIONS}" ]]; then
  echo "ERROR: missing predictions: ${PREDICTIONS}" >&2
  echo "Run: bash scripts/run_train_seen_inference.sh" >&2
  exit 1
fi

read -r -a PERIOD_ARRAY <<< "${PERIODS}"
SUBSET_FILE="${GCNM_ROOT}/data/subject006_gcnm_hdf/train_periods_${PERIOD_ARRAY[0]}_${PERIOD_ARRAY[1]}_${PERIOD_ARRAY[2]}.npz"
OUT_FILE="${GCNM_ROOT}/reports/gcnm_pvi_latex/figures/subject006_pvi_gcnm_train_split.gif"

export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/gcnm-mpl}"
"${GCNM_PYTHON}" "${GCNM_ROOT}/scripts/make_subject006_pvi_gcnm_gif.py" \
  --split train \
  --experiment "${EXPERIMENT}" \
  --meta "${SUBSET_FILE}" \
  --predictions "${PREDICTIONS}" \
  --periods ${PERIODS} \
  --out "${OUT_FILE}"
