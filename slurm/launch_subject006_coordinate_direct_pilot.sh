#!/bin/bash
# Real subject006 usability gate after the 1,000-beat training completes.
#SBATCH --job-name=coord-s006-pilot
#SBATCH --output=logs/coord-s006-pilot_%j.out
#SBATCH --error=logs/coord-s006-pilot_%j.err
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
export GCNM_PHYSICS_WORKERS="${SLURM_CPUS_PER_TASK}"
export GCNM_INFERENCE_BATCH_SIZE=512
export GCNM_STAGE2_RESIDUAL_STRIDE=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export MPLCONFIGDIR="${TMPDIR:-/tmp}/matplotlib-coord-s006-${SLURM_JOB_ID}"

MODEL_ROOT="${MODEL_ROOT:-${REPO_ROOT}/models/differential_US120_1000beats_v1/coordinate_direct}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${REPO_ROOT}/reports/subject006_coordinate_direct_US120_1000beats_v1}"
if [[ ! -f "${MODEL_ROOT}/coordinate_direct_0.pt" || ! -f "${MODEL_ROOT}/coordinate_direct_1.pt" ]]; then
  echo "completed two-stage coordinate-direct checkpoints are missing" >&2
  exit 2
fi
if [[ -e "${OUTPUT_ROOT}" ]]; then
  echo "immutable subject006 pilot output already exists" >&2
  exit 3
fi

python -u -m gcnm_pvi.subject_coordinate_direct_pilot \
  --registry "${REPO_ROOT}/data/registries/main_b045_v1.json" \
  --config "${REPO_ROOT}/configs/rings_b045/US120.yaml" \
  --checkpoint-dir "${MODEL_ROOT}" --model-name coordinate_direct \
  --subject subject006 --sessions baseline valsalva pressor \
  --mask-position middle --output-root "${OUTPUT_ROOT}"
