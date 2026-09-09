#!/bin/bash
#SBATCH --job-name=fig26-final
#SBATCH --output=logs/fig26-final_%j.out
#SBATCH --error=logs/fig26-final_%j.err
#SBATCH --partition=ece_bst
#SBATCH --account=ece_bst
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16GB
#SBATCH --time=02:00:00

set -euo pipefail
REPO_ROOT="${REPO_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"
cd "${REPO_ROOT}"
source "${REPO_ROOT}/env/cluster.env"
source "$(dirname "$(dirname "${GCNM_PYTHON}")")/bin/activate"
export PYTHONPATH="${REPO_ROOT}:${PVI_SOLVER_ROOT}:${PYTHONPATH:-}"
OUTPUT_ROOT="${DUAL_MESH_FIGURE26_ROOT:-${REPO_ROOT}/exports/figure26_dual_mesh_sensitivity_3beats_v1}"
python -u scripts/export_dual_mesh_figure26_dataset.py finalize --output-root "${OUTPUT_ROOT}"
