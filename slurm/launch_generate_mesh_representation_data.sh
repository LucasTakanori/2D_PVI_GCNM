#!/bin/bash
#SBATCH --job-name=gcnm-mesh-data
#SBATCH --output=logs/gcnm-mesh-data_%A_%a.out
#SBATCH --error=logs/gcnm-mesh-data_%A_%a.err
#SBATCH --partition=ece_bst
#SBATCH --account=ece_bst
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=250GB
#SBATCH --time=10-00:00:00
set -euo pipefail
module load gcc/11.2.0
REPO_ROOT="${REPO_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"
cd "${REPO_ROOT}"
source "${REPO_ROOT}/env/cluster.env"
source "$(dirname "$(dirname "${GCNM_PYTHON}")")/bin/activate"
export PYTHONPATH="${REPO_ROOT}:${PVI_SOLVER_ROOT}:${PYTHONPATH:-}"
export OMP_NUM_THREADS="${TASK_CPUS:-${SLURM_CPUS_PER_TASK}}"

RINGS=(US060 US065 US070 US075 US080 US085 US090 US095 US100 US105 US110 US115 US120 US125 US130)
RING="${RINGS[${SLURM_ARRAY_TASK_ID}]}"
FAMILY="${FAMILY:?set FAMILY=coordinate, global_voltage_slots, or diffusion}"
CONFIG="${REPO_ROOT}/configs/rings_b045/${RING}.yaml"
DATASET_DIR="${REPO_ROOT}/data/mesh_training/${FAMILY}/${RING}"
if [[ -e "${DATASET_DIR}" ]]; then
  echo "ERROR: immutable dataset already exists: ${DATASET_DIR}" >&2
  exit 1
fi
if [[ "${FAMILY}" == "coordinate" ]]; then
  python -u -m gcnm_pvi.generate_anatomical_dataset \
    --config "${CONFIG}" --out-dir "${DATASET_DIR}" \
    --train 512 --validation 128 --test 128 --seed 20260712 \
    --simulation-mode linearized --jacobian-bank-size 8 --vessel-count mixed
elif [[ "${FAMILY}" == "global_voltage_slots" ]]; then
  python -u -m gcnm_pvi.generate_anatomical_dataset \
    --config "${CONFIG}" --out-dir "${DATASET_DIR}" \
    --train 512 --validation 128 --test 128 --seed 20260712 \
    --simulation-mode linearized --jacobian-bank-size 8 \
    --vessel-count 2 --minimum-vessel-gap 0.06
elif [[ "${FAMILY}" == "diffusion" ]]; then
  python -u -m gcnm_pvi.generate_finger_simulator_dataset \
    --config "${CONFIG}" \
    --simulator-root "${SIMULATOR_ROOT:-$(dirname "${REPO_ROOT}")/Finger-Conductivity-Simulator}" \
    --finger-config "${SIMULATOR_ROOT:-$(dirname "${REPO_ROOT}")/Finger-Conductivity-Simulator}/configs/default_finger.json" \
    --out-dir "${DATASET_DIR}" --train 512 --validation 128 --test 32 \
    --seed 20260716 --training-mode linearized --test-mode nonlinear \
    --jacobian-bank-size 8 --skip-real-pack
else
  echo "invalid FAMILY=${FAMILY}" >&2
  exit 2
fi
