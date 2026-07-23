#!/bin/bash
# Four independent synthetic holdout GIFs: 2 families x HP/LP.
# Each task loads existing checkpoints and performs inference only.
#SBATCH --job-name=us120-synth-gif
#SBATCH --output=logs/us120-synth-gif_%A_%a.out
#SBATCH --error=logs/us120-synth-gif_%A_%a.err
#SBATCH --partition=ece_bst
#SBATCH --account=ece_bst
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=250GB
#SBATCH --time=10-00:00:00
#SBATCH --gres=gpu:1

set -euo pipefail
module load CUDA/12.9.0
module load gcc/11.2.0

REPO_ROOT="${REPO_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"
cd "${REPO_ROOT}"
source "${REPO_ROOT}/env/cluster.env"
source "$(dirname "$(dirname "${GCNM_PYTHON}")")/bin/activate"
export PYTHONPATH="${REPO_ROOT}:${PVI_SOLVER_ROOT}:${PYTHONPATH:-}"
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export GCNM_PHYSICS_WORKERS="${SLURM_CPUS_PER_TASK}"
export MPLCONFIGDIR="${TMPDIR:-/tmp}/matplotlib-gcnm"

TASK_ID="${SLURM_ARRAY_TASK_ID:?submit this script as array 0-3%4}"
FAMILIES=(coordinate coordinate global_voltage_slots global_voltage_slots)
COMPONENTS=(hp lp hp lp)
FAMILY="${FAMILIES[${TASK_ID}]}"
COMPONENT="${COMPONENTS[${TASK_ID}]}"
RING=US120

CONFIG="${REPO_ROOT}/configs/rings_b045/${RING}.yaml"
DATASET="${REPO_ROOT}/data/hp_lp_beats_US120_v1/${COMPONENT}/test.npz"
MODEL_NAME="${FAMILY}_${COMPONENT}_b045_${RING}_seed0"
CHECKPOINT_DIR="${REPO_ROOT}/models/hp_lp_us120_v1/${FAMILY}/${COMPONENT}/${RING}"
OUTPUT_DIR="${REPO_ROOT}/reports/hp_lp_us120_v1/synthetic_gifs"
OUTPUT="${OUTPUT_DIR}/${FAMILY}_${COMPONENT}_truth_newton_s1_s2.gif"

mkdir -p "${OUTPUT_DIR}" "${MPLCONFIGDIR}"
if [[ ! -f "${DATASET}" ]]; then
  echo "ERROR: missing synthetic test pack: ${DATASET}" >&2
  exit 1
fi
if [[ -e "${OUTPUT}" || -e "${OUTPUT%.gif}.json" ]]; then
  echo "ERROR: immutable GIF artifact already exists: ${OUTPUT}" >&2
  exit 1
fi

echo "family=${FAMILY} component=${COMPONENT} dataset=${DATASET} output=${OUTPUT}"
nvidia-smi --query-gpu=name,memory.total,memory.used,utilization.gpu --format=csv,noheader
python -u -m gcnm_pvi.three_beat_gifs \
  --kind synthetic \
  --family "${FAMILY}" \
  --component "${COMPONENT}" \
  --config "${CONFIG}" \
  --checkpoint-dir "${CHECKPOINT_DIR}" \
  --model-name "${MODEL_NAME}" \
  --dataset "${DATASET}" \
  --output "${OUTPUT}"

python - "${OUTPUT}" <<'PY'
import json
import sys
from pathlib import Path

gif = Path(sys.argv[1])
sidecar = gif.with_suffix(".json")
if not gif.is_file() or gif.stat().st_size == 0:
    raise SystemExit(f"missing or empty GIF: {gif}")
report = json.loads(sidecar.read_text(encoding="utf-8"))
if report.get("frames") != 150:
    raise SystemExit(f"expected 150 frames, found {report.get('frames')}")
print(json.dumps({"validated_gif": str(gif), "bytes": gif.stat().st_size, "frames": 150}))
PY
