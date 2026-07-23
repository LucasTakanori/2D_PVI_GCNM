#!/bin/bash
#SBATCH --job-name=gcnm-session-bench
#SBATCH --output=logs/gcnm-session-bench_%j.out
#SBATCH --error=logs/gcnm-session-bench_%j.err
#SBATCH --partition=ece_bst
#SBATCH --account=ece_bst
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64GB
#SBATCH --time=04:00:00
#SBATCH --gres=gpu:2
set -euo pipefail
module load CUDA/12.9.0
module load gcc/11.2.0
REPO_ROOT="${REPO_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"
cd "${REPO_ROOT}"
source "${REPO_ROOT}/env/cluster.env"
source "$(dirname "$(dirname "${GCNM_PYTHON}")")/bin/activate"
export PYTHONPATH="${REPO_ROOT}:${PVI_SOLVER_ROOT}:${PYTHONPATH:-}"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
python -u scripts/benchmark_session_throughput.py \
  --family "${BENCHMARK_FAMILY:?set BENCHMARK_FAMILY}" \
  --frames "${BENCHMARK_FRAMES:-8192}" --chunk-frames "${BENCHMARK_CHUNK_FRAMES:-256}"
