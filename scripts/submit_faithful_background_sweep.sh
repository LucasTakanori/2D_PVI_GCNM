#!/usr/bin/env bash
# Submit explicit background-loss ablations after choosing the vessel weight.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

POSITIVE_WEIGHT="${POSITIVE_WEIGHT:?set POSITIVE_WEIGHT selected by the prior sweep}"
SUFFIX="${SUFFIX:-}"

submit_variant() {
  local tag="$1"
  local background_weight="$2"
  EXPERIMENT="faithful_background${tag}${SUFFIX}" \
  OUTPUT_MODE=direct \
  USE_COORDINATES=1 \
  POSITIVE_WEIGHT="${POSITIVE_WEIGHT}" \
  BACKGROUND_WEIGHT="${background_weight}" \
  CHECKPOINT_MODE=composite \
  SKIP_LM_CONTROL=1 \
  BASELINE_MODE=homogeneous \
  BASELINE_CONDUCTIVITY=0.7 \
  SEED=0 \
  RUN_REAL=0 \
    bash scripts/submit_faithful_experiment.sh
}

submit_variant 025 0.25
submit_variant 05 0.5
submit_variant 1 1.0
