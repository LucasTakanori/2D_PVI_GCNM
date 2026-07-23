#!/bin/bash
# One family export uses two GPUs and four session workers. Running coordinate
# and vessel exports together consumes the four-GPU/64-CPU/1000-GB ceiling.
#SBATCH --job-name=us120-hp-lp-export
#SBATCH --output=logs/us120-hp-lp-export_%A_%a.out
#SBATCH --error=logs/us120-hp-lp-export_%A_%a.err
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
export GCNM_EXPORT_PHYSICS_WORKERS=8
export MPLCONFIGDIR="${TMPDIR:-/tmp}/matplotlib-gcnm"

TASK_ID="${SLURM_ARRAY_TASK_ID:?submit this script as array 0-1}"
FAMILIES=(coordinate global_voltage_slots)
FAMILY="${FAMILIES[${TASK_ID}]}"
MODEL_ROOT="${MODEL_ROOT:-${REPO_ROOT}/models/hp_lp_us120_v1}"
PARQUET_ROOT="${PARQUET_ROOT:-${REPO_ROOT}/gcnm_parquet}"
OUTPUT_ROOT="${PARQUET_ROOT}/us120_pilot_${FAMILY}_hp_lp_v1"
if [[ -e "${OUTPUT_ROOT}" ]]; then
  echo "ERROR: immutable Parquet root exists: ${OUTPUT_ROOT}" >&2
  exit 1
fi

python -u -m gcnm_pvi.export_pvi_parquet \
  --registry "${REGISTRY:-${REPO_ROOT}/data/registries/main_b045_v1.json}" \
  --family "${FAMILY}" --checkpoint-root "${MODEL_ROOT}/${FAMILY}" \
  --output-root "${OUTPUT_ROOT}" --subjects subject006 subject010 \
  --session-workers 4 --gpu-count 2 --chunk-frames 512 --batch-rows 4 \
  --shard-rows 8192 --compaction-batch-rows 8

python -u -m gcnm_pvi.validate_pvi_parquet --root "${OUTPUT_ROOT}" --max-rows 64
