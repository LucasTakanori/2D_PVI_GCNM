#!/bin/bash
# One GPU array task exports all sessions assigned to one ring.
#SBATCH --job-name=coord91-export
#SBATCH --output=logs/coord91-export_%A_%a.out
#SBATCH --error=logs/coord91-export_%A_%a.err
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
export GCNM_INFERENCE_BATCH_SIZE=512
export GCNM_COMPUTE_STAGE2_RESIDUALS=0
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export MPLCONFIGDIR="${TMPDIR:-/tmp}/coord91-export-${SLURM_JOB_ID}-${SLURM_ARRAY_TASK_ID}"

RINGS=(US060 US065 US070 US075 US080 US085 US090 US095 US100 US105 US110 US115 US120 US125 US130)
RING="${RINGS[${SLURM_ARRAY_TASK_ID:?array task ID required}]}"
if [[ "${RING}" == "US120" ]]; then
  MODEL_ROOT="${REPO_ROOT}/models/differential_US120_1000beats_v1/coordinate_direct"
else
  MODEL_ROOT="${REPO_ROOT}/models/differential_main_b045_1000beats_v1/${RING}/coordinate_direct"
fi
OUTPUT_ROOT="${COORDINATE_ROOT:?set COORDINATE_ROOT}/ring_parts/${RING}"
if [[ -f "${OUTPUT_ROOT}/manifest.json" && ! -e "${OUTPUT_ROOT}/_INCOMPLETE" ]]; then
  echo "reuse complete ${RING} Parquet part"
  exit 0
fi
if [[ -e "${OUTPUT_ROOT}" ]]; then
  echo "partial immutable coordinate export exists for ${RING}" >&2
  exit 2
fi
python -u -m gcnm_pvi.export_coordinate_direct_parquet \
  --registry "${REPO_ROOT}/data/registries/main_b045_v1.json" \
  --checkpoint-dir "${MODEL_ROOT}" --model-name coordinate_direct \
  --output-root "${OUTPUT_ROOT}" --ring "${RING}" \
  --sessions baseline valsalva pressor \
  --session-workers 4 --physics-workers 4 --batch-rows 8 --shard-rows 8192
