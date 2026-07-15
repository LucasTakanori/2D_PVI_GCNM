#!/usr/bin/env bash
# Segment all PVI trials into MATLAB-compatible 50-point cardiac periods.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=./_common.sh
source "${SCRIPT_DIR}/_common.sh"

SESSION_DIR="${1:-${GCNM_SESSION_DIR:-}}"
OUT_DIR="${2:-${GCNM_OUT_DIR:-}}"
H5_PATH="${3:-${GCNM_H5_PATH:-}}"
CONFIG="${GCNM_CONFIG:-${GCNM_ROOT}/configs/subject006_pvi08_production.yaml}"
TRIALS="${GCNM_TRIALS:-}"
MIN_PEAK_DISTANCE="${GCNM_MIN_PEAK_DISTANCE:-0.5}"
PROMINENCE_FACTOR="${GCNM_PROMINENCE_FACTOR:-2.0}"

if [[ -z "${SESSION_DIR}" || -z "${OUT_DIR}" ]]; then
  echo "Usage: bash scripts/segment_session.sh SESSION_DIR OUT_DIR [REFERENCE_H5]" >&2
  exit 1
fi

ARGS=(--config "${CONFIG}" --session-dir "${SESSION_DIR}" --out-dir "${OUT_DIR}" \
  --min-peak-distance "${MIN_PEAK_DISTANCE}" --prominence-factor "${PROMINENCE_FACTOR}")
if [[ -n "${H5_PATH}" ]]; then ARGS+=(--h5 "${H5_PATH}"); fi
if [[ -n "${TRIALS}" ]]; then ARGS+=(--trials "${TRIALS}"); fi

"${GCNM_PYTHON}" -m gcnm_pvi.segment_session "${ARGS[@]}"
