#!/bin/bash
#SBATCH --job-name=gcnm-anatomy-eval
#SBATCH --output=logs/gcnm-anatomy-eval_%j.out
#SBATCH --error=logs/gcnm-anatomy-eval_%j.err
#SBATCH --partition=ece_bst
#SBATCH --account=ece_bst
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=250GB
#SBATCH --time=1-00:00:00
#SBATCH --gres=gpu:1

set -euo pipefail

module load CUDA/12.9.0
module load gcc/11.2.0

REPO_ROOT="${REPO_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"
cd "${REPO_ROOT}"
if [[ -f "${REPO_ROOT}/env/cluster.env" ]]; then source "${REPO_ROOT}/env/cluster.env"; fi
VENV_ROOT="${GCNM_VENV_ROOT:-$(dirname "$(dirname "${GCNM_PYTHON:-${REPO_ROOT}/.venv/bin/python}")")}"
source "${VENV_ROOT}/bin/activate"

export PYTHONUNBUFFERED=1
export PYTHONPATH="${REPO_ROOT}:${PVI_SOLVER_ROOT}:${PYTHONPATH:-}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/gcnm_mpl_${SLURM_JOB_ID}}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"

CONFIG="${CONFIG:-${REPO_ROOT}/configs/subject006_anatomical_gcnm.yaml}"
DATASET_DIR="${DATASET_DIR:-${REPO_ROOT}/data/subject006_anatomical_full}"
EVAL_OUT_DIR="${EVAL_OUT_DIR:-${REPO_ROOT}/data/subject006_anatomical_results/evaluation_linearized}"

echo "job=${SLURM_JOB_ID} host=$(hostname) test=${DATASET_DIR}/test.npz"
python -u -m gcnm_pvi.evaluate_anatomical_gcnm \
  --config "${CONFIG}" \
  --test "${DATASET_DIR}/test.npz" \
  --out-dir "${EVAL_OUT_DIR}" \
  --save-examples 8

deactivate
