#!/usr/bin/env bash
# Compatibility entry point for the corrected beats1000x50 pipeline.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec bash "${SCRIPT_DIR}/submit_beats50_mesh_training_and_parquet.sh" "$@"
