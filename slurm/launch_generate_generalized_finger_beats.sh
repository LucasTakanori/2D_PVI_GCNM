#!/bin/bash
#SBATCH --job-name=gcnm-beat-data-v2
#SBATCH --output=logs/gcnm-beat-data-v2_%j.out
#SBATCH --error=logs/gcnm-beat-data-v2_%j.err
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
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/gcnm_beat_data_${SLURM_JOB_ID}}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"

CONFIG="${CONFIG:-${REPO_ROOT}/configs/subject006_anatomical_gcnm.yaml}"
SIMULATOR_ROOT="${SIMULATOR_ROOT:-$(dirname "${REPO_ROOT}")/Finger-Conductivity-Simulator}"
DATASET_DIR="${DATASET_DIR:-${REPO_ROOT}/data/finger_generalized_beats_US120_v2}"

python -u -m gcnm_pvi.generate_generalized_finger_beats \
  --config "${CONFIG}" \
  --simulator-root "${SIMULATOR_ROOT}" \
  --finger-config "${SIMULATOR_ROOT}/configs/default_finger.json" \
  --out-dir "${DATASET_DIR}" \
  --train-beats "${TRAIN_BEATS:-1000}" \
  --validation-beats "${VALIDATION_BEATS:-200}" \
  --test-beats "${TEST_BEATS:-8}" \
  --train-phases-per-beat "${TRAIN_PHASES_PER_BEAT:-4}" \
  --validation-phases-per-beat "${VALIDATION_PHASES_PER_BEAT:-4}" \
  --jacobian-bank-size "${JACOBIAN_BANK_SIZE:-32}" \
  --seed "${DATA_SEED:-20260718}"

deactivate
