#!/bin/bash
# One differential architecture experiment. Submit variants one job at a time.
#SBATCH --job-name=diff-gcnm-gate
#SBATCH --output=logs/diff-gcnm-gate_%j.out
#SBATCH --error=logs/diff-gcnm-gate_%j.err
#SBATCH --partition=ece_bst
#SBATCH --account=ece_bst
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
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

# Independent FEM frames use private runtimes. BLAS remains single-threaded so
# 50 physics lanes do not oversubscribe the 64 allocated CPU cores.
export GCNM_PHYSICS_WORKERS="${SLURM_CPUS_PER_TASK}"
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export MPLCONFIGDIR="${TMPDIR:-/tmp}/matplotlib-diff-gate-${SLURM_JOB_ID}"

VARIANT="${VARIANT:?set VARIANT=coordinate_direct|coordinate_global|coordinate_residual}"
case "${VARIANT}" in
  coordinate_direct)
    MODE=direct
    EXTRA=()
    ;;
  coordinate_global)
    MODE=direct
    EXTRA=(--use-voltage-mlp)
    ;;
  coordinate_residual)
    MODE=proposal_residual
    EXTRA=()
    ;;
  *)
    echo "unsupported differential architecture variant: ${VARIANT}" >&2
    exit 2
    ;;
esac

PACK_ROOT="${PACK_ROOT:-${REPO_ROOT}/data/differential_one_beat_US120_clean_v2}"
MODEL_ROOT="${MODEL_ROOT:-${REPO_ROOT}/models/differential_architecture_gate_v2/${VARIANT}}"
RESULT_ROOT="${RESULT_ROOT:-${REPO_ROOT}/data/differential_architecture_gate_results_v2/${VARIANT}}"
if [[ ! -f "${PACK_ROOT}/manifest.json" ]]; then
  echo "missing validated differential pilot pack: ${PACK_ROOT}" >&2
  exit 3
fi
if [[ -e "${MODEL_ROOT}" || -e "${RESULT_ROOT}" ]]; then
  echo "immutable output already exists for ${VARIANT}" >&2
  exit 4
fi

echo "[$(date --iso-8601=seconds)] starting ${VARIANT} on ${HOSTNAME}"
nvidia-smi --query-gpu=index,name,memory.total,memory.free,utilization.gpu --format=csv

python -u -m gcnm_pvi.train_faithful_gcnm \
  --config "${REPO_ROOT}/configs/rings_b045/US120.yaml" \
  --train "${PACK_ROOT}/train.npz" \
  --validation "${PACK_ROOT}/validation.npz" \
  --model-name "${VARIANT}" \
  --models-dir "${MODEL_ROOT}" \
  --results-dir "${RESULT_ROOT}" \
  --physics-contract differential \
  --baseline-mode homogeneous --baseline-conductivity 0.7 \
  --output-mode "${MODE}" --use-coordinates "${EXTRA[@]}" \
  --positive-weight 1.0 --background-weight 0.25 \
  --checkpoint-mode composite \
  --iterations 2 --epochs 500 --patience 100 --seed 0 \
  --batch-size 50 --loader-workers 0 --validation-is-train-copy

python -u -m gcnm_pvi.evaluate_faithful_gcnm \
  --config "${REPO_ROOT}/configs/rings_b045/US120.yaml" \
  --test "${PACK_ROOT}/test.npz" \
  --model-name "${VARIANT}" \
  --models-dir "${MODEL_ROOT}" \
  --out-dir "${RESULT_ROOT}/evaluation" \
  --iterations 2 --target-kind clean \
  --baseline-mode homogeneous --baseline-conductivity 0.7 \
  --save-examples 50

echo "[$(date --iso-8601=seconds)] completed ${VARIANT}"
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv
