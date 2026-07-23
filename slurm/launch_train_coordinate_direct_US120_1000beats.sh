#!/bin/bash
# Production-scale US120 coordinate-direct training on 1,000 synthetic beats.
#SBATCH --job-name=coord-US120-1000
#SBATCH --output=logs/coord-US120-1000_%j.out
#SBATCH --error=logs/coord-US120-1000_%j.err
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
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export MPLCONFIGDIR="${TMPDIR:-/tmp}/matplotlib-coord-1000-${SLURM_JOB_ID}"

DATA_ROOT="${DATA_ROOT:-${REPO_ROOT}/data/differential_US120_1000beats_clean_v1}"
MODEL_ROOT="${MODEL_ROOT:-${REPO_ROOT}/models/differential_US120_1000beats_v1/coordinate_direct}"
RESULT_ROOT="${RESULT_ROOT:-${REPO_ROOT}/data/differential_US120_1000beats_results_v1/coordinate_direct}"
if [[ ! -f "${DATA_ROOT}/validation.json" || -e "${MODEL_ROOT}" || -e "${RESULT_ROOT}" ]]; then
  echo "validated 1,000-beat data is missing or immutable output already exists" >&2
  exit 2
fi

echo "[$(date --iso-8601=seconds)] training coordinate-direct on 40,000 frames"
nvidia-smi --query-gpu=index,name,memory.total,memory.free,utilization.gpu --format=csv
python -u -m gcnm_pvi.train_faithful_gcnm \
  --config "${REPO_ROOT}/configs/rings_b045/US120.yaml" \
  --train "${DATA_ROOT}/train.npz" --validation "${DATA_ROOT}/validation.npz" \
  --model-name coordinate_direct --models-dir "${MODEL_ROOT}" \
  --results-dir "${RESULT_ROOT}" \
  --physics-contract differential \
  --baseline-mode homogeneous --baseline-conductivity 0.7 \
  --output-mode direct --use-coordinates \
  --positive-weight 1.0 --background-weight 0.25 \
  --checkpoint-mode composite \
  --iterations 2 --epochs 150 --patience 30 --seed 0 \
  --batch-size 512 --loader-workers 4
echo "[$(date --iso-8601=seconds)] coordinate-direct 1,000-beat training complete"
