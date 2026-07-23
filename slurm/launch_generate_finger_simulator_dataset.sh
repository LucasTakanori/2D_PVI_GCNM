#!/bin/bash
#SBATCH --job-name=gcnm-finger-data
#SBATCH --output=logs/gcnm-finger-data_%j.out
#SBATCH --error=logs/gcnm-finger-data_%j.err
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

CONFIG="${CONFIG:-${REPO_ROOT}/configs/subject006_anatomical_gcnm.yaml}"
SIMULATOR_ROOT="${SIMULATOR_ROOT:-$(dirname "${REPO_ROOT}")/Finger-Conductivity-Simulator}"
DATASET_DIR="${DATASET_DIR:-${REPO_ROOT}/data/finger_default_anatomical_exact}"

python -u -m gcnm_pvi.generate_finger_simulator_dataset \
  --config "${CONFIG}" \
  --simulator-root "${SIMULATOR_ROOT}" \
  --finger-config "${SIMULATOR_ROOT}/configs/default_finger.json" \
  --out-dir "${DATASET_DIR}" \
  --train "${SYNTH_TRAIN:-512}" \
  --validation "${SYNTH_VALIDATION:-128}" \
  --test "${SYNTH_TEST:-32}" \
  --seed "${SEED:-20260716}"

deactivate
