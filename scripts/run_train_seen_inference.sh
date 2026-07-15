#!/usr/bin/env bash
# Run faithful GCNM real-PVI inference on three consecutive train periods (seen data).
#
# Default periods: 1, 2, 3 from the leakage-safe train split.
# Writes:
#   data/faithful_results/<EXPERIMENT>/evaluation_real_pvi_train/predictions.npz
#
# Usage:
#   bash scripts/run_train_seen_inference.sh
#   PERIODS="4 5 6" bash scripts/run_train_seen_inference.sh
#   SLURM_TIME=3:00:00 bash scripts/run_train_seen_inference.sh
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=./_common.sh
source "${SCRIPT_DIR}/_common.sh"

EXPERIMENT="${EXPERIMENT:-faithful_background025_hom_seed0}"
CONFIG="${CONFIG:-${GCNM_ROOT}/configs/subject006_anatomical_gcnm.yaml}"
TRAIN_FILE="${TRAIN_FILE:-${GCNM_ROOT}/data/subject006_gcnm_hdf/train.npz}"
PERIODS="${PERIODS:-1 2 3}"
SUBMIT="${SUBMIT:-1}"
MAX_TEST="${MAX_TEST:-}"
SLURM_TIME="${SLURM_TIME:-3h}"

read -r -a PERIOD_ARRAY <<< "${PERIODS}"
SUBSET_FILE="${GCNM_ROOT}/data/subject006_gcnm_hdf/train_periods_${PERIOD_ARRAY[0]}_${PERIOD_ARRAY[1]}_${PERIOD_ARRAY[2]}.npz"
OUT_DIR="${GCNM_ROOT}/data/faithful_results/${EXPERIMENT}/evaluation_real_pvi_train"
MODELS_DIR="${GCNM_ROOT}/models/faithful/${EXPERIMENT}"

mkdir -p "$(dirname "${SUBSET_FILE}")" "${OUT_DIR}" logs

echo "=== train seen-data inference ==="
echo "experiment=${EXPERIMENT}"
echo "periods=${PERIODS}"
echo "subset=${SUBSET_FILE}"
echo "out=${OUT_DIR}"

TRAIN_FILE="${TRAIN_FILE}" SUBSET_FILE="${SUBSET_FILE}" PERIODS="${PERIODS}" \
"${GCNM_PYTHON}" - <<'PY'
import os
from pathlib import Path

import numpy as np

train_path = Path(os.environ["TRAIN_FILE"])
subset_path = Path(os.environ["SUBSET_FILE"])
periods = {int(value) for value in os.environ["PERIODS"].split()}

source = np.load(train_path)
mask = np.isin(source["period_id"], sorted(periods))
if not np.any(mask):
    raise SystemExit(f"no samples found for periods {sorted(periods)}")

subset = {key: source[key][mask] for key in source.files}
np.savez_compressed(subset_path, **subset)
selected_periods = sorted({int(value) for value in subset["period_id"]})
print(
    f"wrote {subset_path} with {int(mask.sum())} samples "
    f"for periods {selected_periods}"
)
PY

if [[ "${SUBMIT}" == "1" ]]; then
  EXPORT_VARS="ALL,REPO_ROOT=${GCNM_ROOT},CONFIG=${CONFIG},EXPERIMENT=${EXPERIMENT}"
  EXPORT_VARS+=",DATASET_DIR=${GCNM_ROOT}/data/subject006_gcnm_hdf"
  EXPORT_VARS+=",TEST_FILE=${SUBSET_FILE}"
  EXPORT_VARS+=",EVALUATION_NAME=evaluation_real_pvi_train"
  EXPORT_VARS+=",TARGET_KIND=pvi_pseudo"
  EXPORT_VARS+=",SKIP_LM_CONTROL=1"
  EXPORT_VARS+=",MAX_TEST=${MAX_TEST}"

  JOB_ID=$(sbatch --parsable \
    --job-name="${EXPERIMENT}-train-seen" \
    --time="${SLURM_TIME}" \
    --export="${EXPORT_VARS}" \
    "${GCNM_ROOT}/slurm/launch_evaluate_faithful.sh")
  echo "submitted Slurm job ${JOB_ID} (time limit ${SLURM_TIME})"
  echo "log: ${GCNM_ROOT}/logs/gcnm-faithful-eval_${JOB_ID}.out"
  exit 0
fi

export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/gcnm-mpl}"
ARGS=(
  --config "${CONFIG}"
  --test "${SUBSET_FILE}"
  --model-name "${EXPERIMENT}"
  --models-dir "${MODELS_DIR}"
  --out-dir "${OUT_DIR}"
  --target-kind pvi_pseudo
  --skip-lm-control
  --save-examples 0
)
if [[ -n "${MAX_TEST}" ]]; then
  ARGS+=(--max-test "${MAX_TEST}")
fi

"${GCNM_PYTHON}" -u -m gcnm_pvi.evaluate_faithful_gcnm "${ARGS[@]}"
