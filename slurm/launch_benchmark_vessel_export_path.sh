#!/bin/bash
#SBATCH --job-name=vessel-export-profile
#SBATCH --output=logs/vessel-export-profile_%j.out
#SBATCH --error=logs/vessel-export-profile_%j.err
#SBATCH --partition=ece_bst
#SBATCH --account=ece_bst
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64GB
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
export MPLCONFIGDIR="${TMPDIR:-/tmp}/matplotlib-gcnm"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export GCNM_PHYSICS_WORKERS=16
export GCNM_INFERENCE_BATCH_SIZE=512
export GCNM_COMPUTE_STAGE2_RESIDUALS=0
python -u scripts/benchmark_vessel_export_path.py
