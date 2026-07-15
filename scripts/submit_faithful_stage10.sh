#!/usr/bin/env bash
# Extend the selected faithful PVI-GCNM recipe to ten learned/physics stages.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export EXPERIMENT="${EXPERIMENT:-faithful_background025_hom_seed0_stage10}"
export OUTPUT_MODE="direct"
export USE_COORDINATES="1"
export POSITIVE_WEIGHT="1"
export BACKGROUND_WEIGHT="0.25"
export CHECKPOINT_MODE="composite"
export ITERATIONS="10"
export SEED="0"
export BASELINE_MODE="homogeneous"
export BASELINE_CONDUCTIVITY="0.7"
export RUN_LINEAR="0"
export RUN_REAL="1"
export RUN_GIFS="1"
export SKIP_LM_CONTROL="1"

bash "${SCRIPT_DIR}/submit_faithful_experiment.sh"
