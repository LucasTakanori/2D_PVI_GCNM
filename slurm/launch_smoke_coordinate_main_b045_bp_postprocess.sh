#!/bin/bash
# CPU-only proof that a completed BP checkpoint exports all native artifacts.
#SBATCH --job-name=coord91-bpost
#SBATCH --output=logs/coord91-bpost_%j.out
#SBATCH --error=logs/coord91-bpost_%j.err
#SBATCH --partition=ece_bst
#SBATCH --account=ece_bst
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64GB
#SBATCH --time=01:00:00

set -euo pipefail
REPO_ROOT="${REPO_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"
cd "${REPO_ROOT}"
export SLURM_ARRAY_TASK_ID="${SMOKE_TASK_ID:-0}"
export SLURM_CPUS_PER_TASK=4
# Match the immutable training contract. These workers only feed CPU
# postprocessing and do not alter the selected checkpoint.
export BP_NUM_WORKERS=3
export BP_OMP_THREADS=1
export BP_POSTPROCESS_ONLY=1
export BP_CPU_POSTPROCESS=1
bash slurm/launch_train_coordinate_main_b045_bp.sh
