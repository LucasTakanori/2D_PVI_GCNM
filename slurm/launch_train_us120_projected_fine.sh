#!/bin/bash
#SBATCH --job-name=US120-dual-train
#SBATCH --output=logs/US120-dual-train_%j.out
#SBATCH --error=logs/US120-dual-train_%j.err
#SBATCH --partition=ece_bst
#SBATCH --account=ece_bst
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=750GB
#SBATCH --time=10-00:00:00
#SBATCH --gres=gpu:1

set -euo pipefail

module load CUDA/12.9.0
module load gcc/11.2.0

REPO_ROOT="${REPO_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"
cd "${REPO_ROOT}"
source "${REPO_ROOT}/env/cluster.env"
source "$(dirname "$(dirname "${GCNM_PYTHON}")")/bin/activate"

export PYTHONUNBUFFERED=1
export PYTHONPATH="${REPO_ROOT}:${PVI_SOLVER_ROOT}:${PYTHONPATH:-}"
export GCNM_PHYSICS_WORKERS="${GCNM_PHYSICS_WORKERS:-${SLURM_CPUS_PER_TASK}}"
export GCNM_PHYSICS_EXECUTOR="${GCNM_PHYSICS_EXECUTOR:-process}"
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export MPLCONFIGDIR="${TMPDIR:-/tmp}/matplotlib-US120-dual-${SLURM_JOB_ID}"

DATA_ROOT="${DATA_ROOT:-${REPO_ROOT}/data/differential_US120_1000beats_clean_v1}"
CONFIG="${CONFIG:-${REPO_ROOT}/configs/rings_b045/US120.yaml}"
EXPERIMENT="${EXPERIMENT:-coordinate_direct_projected_fine}"
MODEL_ROOT="${MODEL_ROOT:-${REPO_ROOT}/models/differential_US120_1000beats_v1/${EXPERIMENT}}"
RESULT_ROOT="${RESULT_ROOT:-${REPO_ROOT}/data/differential_US120_1000beats_results_v1/${EXPERIMENT}}"
MAX_TRAIN="${MAX_TRAIN:-}"
MAX_VALIDATION="${MAX_VALIDATION:-}"

if [[ ! -f "${DATA_ROOT}/validation.json" || -e "${MODEL_ROOT}" || -e "${RESULT_ROOT}" ]]; then
  echo "validated data is missing or immutable output already exists" >&2
  exit 2
fi

ARGS=(
  --config "${CONFIG}"
  --train "${DATA_ROOT}/train.npz"
  --validation "${DATA_ROOT}/validation.npz"
  --model-name "${EXPERIMENT}"
  --models-dir "${MODEL_ROOT}"
  --results-dir "${RESULT_ROOT}"
  --physics-contract differential
  --physics-mesh-mode projected_fine
  --baseline-mode homogeneous
  --baseline-conductivity 0.7
  --output-mode direct
  --use-coordinates
  --positive-weight 1.0
  --background-weight 0.25
  --checkpoint-mode composite
  --iterations 2
  --epochs 150
  --patience 30
  --seed 0
  --batch-size 512
  --loader-workers 4
)
if [[ -n "${MAX_TRAIN}" ]]; then ARGS+=(--max-train "${MAX_TRAIN}"); fi
if [[ -n "${MAX_VALIDATION}" ]]; then ARGS+=(--max-validation "${MAX_VALIDATION}"); fi

echo "[$(date --iso-8601=seconds)] experiment=${EXPERIMENT} workers=${GCNM_PHYSICS_WORKERS} executor=${GCNM_PHYSICS_EXECUTOR}"
echo "[$(date --iso-8601=seconds)] train=${MAX_TRAIN:-40000} validation=${MAX_VALIDATION:-5000}"
nvidia-smi --query-gpu=index,name,memory.total,memory.free,utilization.gpu --format=csv
python -u -m gcnm_pvi.train_faithful_gcnm "${ARGS[@]}"
echo "[$(date --iso-8601=seconds)] projected-fine US120 training complete"
