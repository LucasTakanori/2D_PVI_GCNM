#!/bin/bash
# Optimized restart of the US120 global-voltage vessel-slot pilot export.
# Thirty-two persistent row-range workers share two GPUs.  Separate processes
# are required because the Python/FEM path does not scale across threads well.
#SBATCH --job-name=us120-vessel-fast
#SBATCH --output=logs/us120-vessel-fast_%j.out
#SBATCH --error=logs/us120-vessel-fast_%j.err
#SBATCH --partition=ece_bst
#SBATCH --account=ece_bst
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=500GB
#SBATCH --time=10-00:00:00
#SBATCH --gres=gpu:2

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
export GCNM_EXPORT_PHYSICS_WORKERS=1
export GCNM_INFERENCE_BATCH_SIZE=256
export GCNM_COMPUTE_STAGE2_RESIDUALS=0
export MPLCONFIGDIR="${TMPDIR:-/tmp}/matplotlib-gcnm"

OUTPUT_ROOT="${REPO_ROOT}/gcnm_parquet/us120_pilot_global_voltage_slots_hp_lp_v1"
if [[ -e "${OUTPUT_ROOT}" ]]; then
  echo "ERROR: immutable output exists: ${OUTPUT_ROOT}" >&2
  exit 1
fi

python -u -m gcnm_pvi.export_pvi_parquet \
  --registry "${REPO_ROOT}/data/registries/main_b045_v1.json" \
  --family global_voltage_slots \
  --checkpoint-root "${REPO_ROOT}/models/hp_lp_us120_v1/global_voltage_slots" \
  --output-root "${OUTPUT_ROOT}" --subjects subject006 subject010 \
  --session-workers 32 --row-range-size 16 --gpu-count 2 \
  --chunk-frames 2000 --batch-rows 8 \
  --shard-rows 8192 --compaction-batch-rows 64 \
  --progress-jsonl "${OUTPUT_ROOT}/progress.jsonl" --progress-every-batches 1

python -u -m gcnm_pvi.validate_pvi_parquet \
  --root "${OUTPUT_ROOT}" --max-rows 64
