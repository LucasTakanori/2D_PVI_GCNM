#!/usr/bin/env bash
# Submit the coordinate ablation and vessel-weight sweep selected after B/C.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"
SUFFIX="${SUFFIX:-}"

submit_variant() {
  local name="$1"
  local positive_weight="$2"
  EXPERIMENT="${name}" \
  OUTPUT_MODE=direct \
  USE_COORDINATES=1 \
  POSITIVE_WEIGHT="${positive_weight}" \
  BACKGROUND_WEIGHT=0 \
  CHECKPOINT_MODE=loss \
  SKIP_LM_CONTROL=1 \
  RUN_REAL=0 \
    bash scripts/submit_faithful_experiment.sh
}

submit_variant "faithful_coords${SUFFIX}" 0
submit_variant "faithful_weight1${SUFFIX}" 1
submit_variant "faithful_weight2${SUFFIX}" 2
submit_variant "faithful_weight4${SUFFIX}" 4
submit_variant "faithful_weight8${SUFFIX}" 8
