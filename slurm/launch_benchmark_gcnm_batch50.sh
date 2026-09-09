#!/bin/bash
# Benchmark warmed streaming and one-second buffered GCNM reconstruction.
# This is inference-only and never writes checkpoints or modifies input data.
#SBATCH --job-name=gcnm-batch50
#SBATCH --output=logs/gcnm-batch50_%j.out
#SBATCH --error=logs/gcnm-batch50_%j.err
#SBATCH --partition=ece_bst
#SBATCH --account=ece_bst
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=250GB
#SBATCH --time=02:00:00
#SBATCH --gres=gpu:1

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${REPO_ROOT:-$(dirname "${SCRIPT_DIR}")}"
cd "${REPO_ROOT}"
source "${REPO_ROOT}/env/cluster.env"
source "$(dirname "$(dirname "${GCNM_PYTHON}")")/bin/activate"

module load CUDA/12.9.0
module load gcc/11.2.0

export PYTHONUNBUFFERED=1
export PYTHONPATH="${REPO_ROOT}:${PVI_SOLVER_ROOT}:${PYTHONPATH:-}"
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

CONFIG="${CONFIG:-${REPO_ROOT}/configs/rings_b045/US120.yaml}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-${REPO_ROOT}/models/differential_US120_1000beats_v1/coordinate_direct_projected_fine}"
MODEL_NAME="${MODEL_NAME:-coordinate_direct_projected_fine}"
FRAMES="${FRAMES:-50}"
WARMUPS="${WARMUPS:-1}"
REPEATS="${REPEATS:-1}"
REFERENCE_FRAMES="${REFERENCE_FRAMES:-1}"
PHYSICS_WORKERS="${PHYSICS_WORKERS:-${SLURM_CPUS_PER_TASK}}"
INFERENCE_BATCH_SIZE="${INFERENCE_BATCH_SIZE:-50}"
FORWARD_BACKEND="${FORWARD_BACKEND:-dense}"
OUTPUT="${OUTPUT:-}"

ARGS=(
  --config "${CONFIG}"
  --checkpoint-dir "${CHECKPOINT_DIR}"
  --model-name "${MODEL_NAME}"
  --frames "${FRAMES}"
  --chunk-sizes 1 50
  --warmups "${WARMUPS}"
  --repeats "${REPEATS}"
  --reference-frames "${REFERENCE_FRAMES}"
  --physics-workers "${PHYSICS_WORKERS}"
  --inference-batch-size "${INFERENCE_BATCH_SIZE}"
  --forward-backend "${FORWARD_BACKEND}"
  --target-hz 50
  --device cuda:0
)
if [[ -n "${SOURCE_HDF5:-}" ]]; then
  ARGS+=(--source-hdf5 "${SOURCE_HDF5}")
fi
if [[ -n "${INPUT_NPZ:-}" ]]; then
  ARGS+=(--input-npz "${INPUT_NPZ}")
fi
if [[ "${USE_REGISTRY_SOURCE:-0}" == "1" ]]; then
  ARGS+=(--use-registry-source --source-name "${SOURCE_NAME:-subject006_baseline}")
fi
if [[ -n "${OUTPUT}" ]]; then
  ARGS+=(--output "${OUTPUT}")
fi
if [[ "${COMPUTE_STAGE2_RESIDUALS:-0}" == "1" ]]; then
  ARGS+=(--compute-stage2-residuals)
fi
if [[ "${HASH_SOURCE_FILE:-0}" == "1" ]]; then
  ARGS+=(--hash-source-file)
fi
if [[ "${ALLOW_CONFIG_HASH_MISMATCH:-0}" == "1" ]]; then
  ARGS+=(--allow-config-hash-mismatch)
fi

echo "[$(date --iso-8601=seconds)] benchmarking warmed 1-frame and 50-frame GCNM inference"
python -u -m gcnm_pvi.benchmark_batch_inference "${ARGS[@]}"
echo "[$(date --iso-8601=seconds)] benchmark complete"
