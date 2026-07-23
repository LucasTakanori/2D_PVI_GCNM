#!/usr/bin/env bash
# Make held-out subject-006 and exact-nonlinear phantom stage-comparison GIFs.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=./_common.sh
source "${SCRIPT_DIR}/_common.sh"

EXPERIMENT="${EXPERIMENT:?set EXPERIMENT to the evaluated faithful model}"
EXPECTED_STAGES="${EXPECTED_STAGES:-10}"
SAMPLES="${SAMPLES:-0 1 2}"
PERIODS="${PERIODS:-}"
RESULTS_DIR="${GCNM_ROOT}/data/faithful_results/${EXPERIMENT}"
GIF_DIR="${RESULTS_DIR}/gifs"
SYNTHETIC_PREDICTIONS="${RESULTS_DIR}/evaluation_nonlinear/predictions.npz"
REAL_PREDICTIONS="${RESULTS_DIR}/evaluation_real_pvi/predictions.npz"
SYNTHETIC_ANATOMY_JSON="${SYNTHETIC_ANATOMY_JSON:-}"
if [[ -z "${SYNTHETIC_ANATOMY_JSON}" && "${EXPERIMENT}" == finger_default_* ]]; then
  candidate="${GCNM_ROOT}/data/finger_default_anatomical_exact/test_anatomy.json"
  if [[ -f "${candidate}" ]]; then
    SYNTHETIC_ANATOMY_JSON="${candidate}"
  fi
fi

for required in "${SYNTHETIC_PREDICTIONS}" "${REAL_PREDICTIONS}"; do
  if [[ ! -f "${required}" ]]; then
    echo "ERROR: missing predictions: ${required}" >&2
    exit 1
  fi
done

export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/gcnm-mpl-${SLURM_JOB_ID:-local}}"

# Deliberate word splitting turns the user-facing space-delimited sample list
# into individual argparse values.
# shellcheck disable=SC2086
SYNTHETIC_ARGS=(
  --split synthetic
  --experiment "${EXPERIMENT}"
  --predictions "${SYNTHETIC_PREDICTIONS}"
  --expected-stages "${EXPECTED_STAGES}"
  --out "${GIF_DIR}/phantom_truth_pvi_vs_${EXPECTED_STAGES}_stages.gif"
)
if [[ -n "${SYNTHETIC_ANATOMY_JSON}" ]]; then
  SYNTHETIC_ARGS+=(--synthetic-anatomy-json "${SYNTHETIC_ANATOMY_JSON}")
fi
"${GCNM_PYTHON}" "${GCNM_ROOT}/scripts/make_subject006_pvi_gcnm_gif.py" \
  "${SYNTHETIC_ARGS[@]}" \
  --samples ${SAMPLES}

REAL_ARGS=(
  --split test
  --experiment "${EXPERIMENT}"
  --predictions "${REAL_PREDICTIONS}"
  --expected-stages "${EXPECTED_STAGES}"
  --out "${GIF_DIR}/subject006_pvi_vs_${EXPECTED_STAGES}_stages.gif"
)
if [[ -n "${PERIODS}" ]]; then
  # shellcheck disable=SC2206
  PERIOD_ARRAY=(${PERIODS})
  REAL_ARGS+=(--periods "${PERIOD_ARRAY[@]}")
fi
"${GCNM_PYTHON}" "${GCNM_ROOT}/scripts/make_subject006_pvi_gcnm_gif.py" \
  "${REAL_ARGS[@]}"

echo "GIF outputs: ${GIF_DIR}"
