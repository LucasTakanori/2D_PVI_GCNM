#!/bin/bash
#SBATCH --job-name=fig26-dual
#SBATCH --output=logs/fig26-dual_%A_%a.out
#SBATCH --error=logs/fig26-dual_%A_%a.err
#SBATCH --partition=ece_bst
#SBATCH --account=ece_bst
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=200GB
#SBATCH --time=2-00:00:00
#SBATCH --gres=gpu:1

set -euo pipefail
module load CUDA/12.9.0
module load gcc/11.2.0
REPO_ROOT="${REPO_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"
cd "${REPO_ROOT}"
source "${REPO_ROOT}/env/cluster.env"
source "$(dirname "$(dirname "${GCNM_PYTHON}")")/bin/activate"
export PYTHONPATH="${REPO_ROOT}:${PVI_SOLVER_ROOT}:${PYTHONPATH:-}"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export GCNM_COMPUTE_STAGE2_RESIDUALS=0
export GCNM_INFERENCE_BATCH_SIZE=512

OUTPUT_ROOT="${DUAL_MESH_FIGURE26_ROOT:-${REPO_ROOT}/exports/figure26_dual_mesh_sensitivity_3beats_v1}"
python -u scripts/export_dual_mesh_figure26_dataset.py export-subject \
  --output-root "${OUTPUT_ROOT}" \
  --subject-index "${SLURM_ARRAY_TASK_ID:?array task ID required}" \
  --physics-workers 16 --inference-batch-size 512 --device cuda:0
