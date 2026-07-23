#!/bin/bash
#SBATCH --job-name=gcnm-newton-pilot
#SBATCH --output=logs/gcnm-newton-pilot_%j.out
#SBATCH --error=logs/gcnm-newton-pilot_%j.err
#SBATCH --partition=ece_bst
#SBATCH --account=ece_bst
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128GB
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
export GCNM_PHYSICS_WORKERS="${GCNM_PHYSICS_WORKERS:-16}"
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

python -u -m gcnm_pvi.pilot_newton_compatible \
  --registry "${REGISTRY:-${REPO_ROOT}/data/registries/main_b045_v1.json}" \
  --checkpoint-root "${CHECKPOINT_ROOT:-${REPO_ROOT}/models/mesh_representations/beats1000x50_v1}" \
  --family "${FAMILY:?set FAMILY}" \
  --subject "${SUBJECT:?set SUBJECT}" \
  --sessions baseline valsalva pressor \
  --mask-position middle \
  --output-root "${OUTPUT_ROOT:?set OUTPUT_ROOT}"
