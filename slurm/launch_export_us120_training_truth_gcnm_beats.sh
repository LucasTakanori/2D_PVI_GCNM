#!/bin/bash
#SBATCH --job-name=us120-train-panels
#SBATCH --output=logs/us120-train-panels_%j.out
#SBATCH --error=logs/us120-train-panels_%j.err
#SBATCH --partition=ece_bst
#SBATCH --account=ece_bst
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=250GB
#SBATCH --time=02:00:00
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
export GCNM_COMPUTE_STAGE2_RESIDUALS=0
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export MPLCONFIGDIR="${TMPDIR:-/tmp}/matplotlib-us120-training-panels-${SLURM_JOB_ID}"

OUTPUT_ROOT="${OUTPUT_ROOT:-${REPO_ROOT}/figures/gcnm_training_20_anatomies_US120}"
if [[ -e "${OUTPUT_ROOT}" ]]; then
  echo "immutable output already exists: ${OUTPUT_ROOT}" >&2
  exit 3
fi

python -u scripts/export_us120_training_truth_gcnm_beats.py \
  --dataset "${REPO_ROOT}/data/differential_US120_1000beats_clean_v1/train.npz" \
  --source-hp "${REPO_ROOT}/data/hp_lp_beats_US120_v1/hp/train.npz" \
  --config "${REPO_ROOT}/configs/rings_b045/US120.yaml" \
  --checkpoint-dir "${REPO_ROOT}/models/differential_US120_1000beats_v1/coordinate_direct" \
  --model-name coordinate_direct \
  --output-root "${OUTPUT_ROOT}" \
  --count 20 --seed 20260802 --image-size 900
