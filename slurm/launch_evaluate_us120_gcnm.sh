#!/bin/bash
#SBATCH --job-name=US120-gcnm-eval
#SBATCH --output=logs/US120-gcnm-eval_%j.out
#SBATCH --error=logs/US120-gcnm-eval_%j.err
#SBATCH --partition=ece_bst
#SBATCH --account=ece_bst
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=750GB
#SBATCH --time=5-00:00:00
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
export MPLCONFIGDIR="${TMPDIR:-/tmp}/matplotlib-US120-eval-${SLURM_JOB_ID}"

DATA_ROOT="${DATA_ROOT:-${REPO_ROOT}/data/differential_US120_1000beats_clean_v1}"
CONFIG="${CONFIG:-${REPO_ROOT}/configs/rings_b045/US120.yaml}"
EXPERIMENT="${EXPERIMENT:-coordinate_direct_projected_fine}"
MODEL_ROOT="${MODEL_ROOT:-${REPO_ROOT}/models/differential_US120_1000beats_v1/${EXPERIMENT}}"
OUT_DIR="${OUT_DIR:-${REPO_ROOT}/data/differential_US120_1000beats_results_v1/${EXPERIMENT}/exact_nonlinear_test}"
MAX_TEST="${MAX_TEST:-}"
ALLOW_CONFIG_HASH_MISMATCH="${ALLOW_CONFIG_HASH_MISMATCH:-0}"
PHYSICS_MESH_MODE="${PHYSICS_MESH_MODE:-auto}"

if [[ ! -f "${MODEL_ROOT}/${EXPERIMENT}_0.pt" || ! -f "${MODEL_ROOT}/${EXPERIMENT}_1.pt" ]]; then
  echo "both trained stage checkpoints are required" >&2
  exit 2
fi
if [[ -e "${OUT_DIR}" ]]; then
  echo "immutable evaluation output already exists: ${OUT_DIR}" >&2
  exit 2
fi

ARGS=(
  --config "${CONFIG}"
  --test "${DATA_ROOT}/test.npz"
  --model-name "${EXPERIMENT}"
  --models-dir "${MODEL_ROOT}"
  --out-dir "${OUT_DIR}"
  --target-kind clean
  --baseline-mode homogeneous
  --baseline-conductivity 0.7
  --iterations 2
  --physics-mesh-mode "${PHYSICS_MESH_MODE}"
  --save-examples 8
  --skip-lm-control
)
if [[ -n "${MAX_TEST}" ]]; then ARGS+=(--max-test "${MAX_TEST}"); fi
if [[ "${ALLOW_CONFIG_HASH_MISMATCH}" == "1" ]]; then
  ARGS+=(--allow-config-hash-mismatch)
fi

echo "[$(date --iso-8601=seconds)] evaluating ${EXPERIMENT} on ${MAX_TEST:-5000} exact nonlinear frames"
python -u -m gcnm_pvi.evaluate_faithful_gcnm "${ARGS[@]}"
echo "[$(date --iso-8601=seconds)] US120 evaluation complete"
