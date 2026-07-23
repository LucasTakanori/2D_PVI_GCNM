#!/bin/bash
#SBATCH --job-name=gcnm-export-bench
#SBATCH --output=logs/gcnm-export-bench_%j.out
#SBATCH --error=logs/gcnm-export-bench_%j.err
#SBATCH --partition=ece_bst
#SBATCH --account=ece_bst
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32GB
#SBATCH --time=00:30:00
#SBATCH --gres=gpu:1

set -euo pipefail
module load CUDA/12.9.0
module load gcc/11.2.0
REPO_ROOT="${REPO_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"
cd "${REPO_ROOT}"
source "${REPO_ROOT}/env/cluster.env"
source "$(dirname "$(dirname "${GCNM_PYTHON}")")/bin/activate"
export PYTHONPATH="${REPO_ROOT}:${PVI_SOLVER_ROOT}:${PYTHONPATH:-}"
export MPLCONFIGDIR="${TMPDIR:-/tmp}/pvi-gcnm-mplconfig"
export GCNM_PHYSICS_WORKERS="${GCNM_PHYSICS_WORKERS:-1}"
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
python -u scripts/benchmark_representation_export.py \
  --frames "${BENCHMARK_FRAMES:-8}" --family "${BENCHMARK_FAMILY:-coordinate}"
