#!/bin/bash
#SBATCH --job-name=gcnm-core-eval
#SBATCH --output=logs/gcnm-core-eval_%j.out
#SBATCH --error=logs/gcnm-core-eval_%j.err
#SBATCH --partition=ece_bst
#SBATCH --account=ece_bst
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=250GB
#SBATCH --time=02:00:00
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
export PYTHONPATH="${REPO_ROOT}:${PVI_SOLVER_ROOT:-}:${PYTHONPATH:-}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"

EXPERIMENT="${EXPERIMENT:?set EXPERIMENT}"
ARGS=(
  --config "${CONFIG:-${REPO_ROOT}/configs/subject006_anatomical_gcnm.yaml}"
  --test "${TEST_FILE:?set TEST_FILE}"
  --stage1 "${REPO_ROOT}/models/faithful/${EXPERIMENT}/${EXPERIMENT}_stage1.pt"
  --stage2 "${REPO_ROOT}/models/faithful/${EXPERIMENT}/${EXPERIMENT}_stage2.pt"
  --out-dir "${REPO_ROOT}/data/faithful_results/${EXPERIMENT}/${EVALUATION_NAME:?set EVALUATION_NAME}"
  --target-kind "${TARGET_KIND:?set TARGET_KIND}"
)
if [[ -n "${ANATOMY_FILE:-}" ]]; then ARGS+=(--anatomy "${ANATOMY_FILE}"); fi
if [[ -n "${STAGE1_MAXIMUM_AMPLITUDE_SCALE:-}" ]]; then
  ARGS+=(--stage1-maximum-amplitude-scale "${STAGE1_MAXIMUM_AMPLITUDE_SCALE}")
fi
if [[ -n "${STAGE2_MAXIMUM_AMPLITUDE_SCALE:-}" ]]; then
  ARGS+=(--stage2-maximum-amplitude-scale "${STAGE2_MAXIMUM_AMPLITUDE_SCALE}")
fi
python -u -m gcnm_pvi.evaluate_core_guided_gcnm "${ARGS[@]}"

deactivate
