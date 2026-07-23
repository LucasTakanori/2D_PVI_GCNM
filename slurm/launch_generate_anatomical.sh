#!/bin/bash
#SBATCH --job-name=gcnm-anatomy-data
#SBATCH --output=logs/gcnm-anatomy-data_%j.out
#SBATCH --error=logs/gcnm-anatomy-data_%j.err
#SBATCH --partition=ece_bst
#SBATCH --account=ece_bst
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=250GB
#SBATCH --time=1-00:00:00
#SBATCH --gres=gpu:1

set -euo pipefail

REPO_ROOT="${REPO_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"
cd "${REPO_ROOT}"
if [[ -f "${REPO_ROOT}/env/cluster.env" ]]; then source "${REPO_ROOT}/env/cluster.env"; fi
VENV_ROOT="${GCNM_VENV_ROOT:-$(dirname "$(dirname "${GCNM_PYTHON:-${REPO_ROOT}/.venv/bin/python}")")}"
source "${VENV_ROOT}/bin/activate"

export PYTHONUNBUFFERED=1
export PYTHONPATH="${REPO_ROOT}:${PVI_SOLVER_ROOT}:${PYTHONPATH:-}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/gcnm_mpl_${SLURM_JOB_ID}}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"

CONFIG="${CONFIG:-${REPO_ROOT}/configs/subject006_pvi08_production.yaml}"
DATASET_DIR="${DATASET_DIR:-${REPO_ROOT}/data/subject006_anatomical_full}"
SYNTH_TRAIN="${SYNTH_TRAIN:-512}"
SYNTH_VALIDATION="${SYNTH_VALIDATION:-128}"
SYNTH_TEST="${SYNTH_TEST:-128}"
SIMULATION_MODE="${SIMULATION_MODE:-linearized}"
JACOBIAN_BANK_SIZE="${JACOBIAN_BANK_SIZE:-8}"
VESSEL_COUNT="${VESSEL_COUNT:-mixed}"
MINIMUM_VESSEL_GAP="${MINIMUM_VESSEL_GAP:-0.0}"

echo "job=${SLURM_JOB_ID} host=$(hostname)"
echo "dataset=${DATASET_DIR} mode=${SIMULATION_MODE} vessels=${VESSEL_COUNT} minimum_gap=${MINIMUM_VESSEL_GAP} counts=${SYNTH_TRAIN}/${SYNTH_VALIDATION}/${SYNTH_TEST}"

python -u -m gcnm_pvi.generate_anatomical_dataset \
  --config "${CONFIG}" \
  --out-dir "${DATASET_DIR}" \
  --train "${SYNTH_TRAIN}" \
  --validation "${SYNTH_VALIDATION}" \
  --test "${SYNTH_TEST}" \
  --simulation-mode "${SIMULATION_MODE}" \
  --jacobian-bank-size "${JACOBIAN_BANK_SIZE}" \
  --vessel-count "${VESSEL_COUNT}" \
  --minimum-vessel-gap "${MINIMUM_VESSEL_GAP}"

deactivate
