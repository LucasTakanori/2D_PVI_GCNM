#!/bin/bash
# Generate the one shared 200-anatomy/1,000-retained-beat HP/LP US120 pack.
# Exact validation/test forward solves are CPU work; no GPU is requested.
#SBATCH --job-name=us120-hp-lp-data
#SBATCH --output=logs/us120-hp-lp-data_%j.out
#SBATCH --error=logs/us120-hp-lp-data_%j.err
#SBATCH --partition=ece_bst
#SBATCH --account=ece_bst
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=1000GB
#SBATCH --time=10-00:00:00

set -euo pipefail
module load gcc/11.2.0

REPO_ROOT="${REPO_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"
cd "${REPO_ROOT}"
source "${REPO_ROOT}/env/cluster.env"
source "$(dirname "$(dirname "${GCNM_PYTHON}")")/bin/activate"
export PYTHONPATH="${REPO_ROOT}:${PVI_SOLVER_ROOT}:${PYTHONPATH:-}"
# One exact FEM solve is active at a time in this version. Let its dense/sparse
# kernels use the full CPU allocation; generation does not reserve a GPU.
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export NUMEXPR_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export MPLCONFIGDIR="${TMPDIR:-/tmp}/matplotlib-gcnm"

DATASET_ROOT="${DATASET_ROOT:-${REPO_ROOT}/data/hp_lp_beats_US120_v1}"
if [[ -e "${DATASET_ROOT}" ]]; then
  echo "ERROR: immutable HP/LP dataset exists: ${DATASET_ROOT}" >&2
  exit 1
fi

python -u -m gcnm_pvi.generate_hp_lp_beat_dataset \
  --config "${CONFIG:-${REPO_ROOT}/configs/rings_b045/US120.yaml}" \
  --output-root "${DATASET_ROOT}" \
  --train-anatomies 160 --validation-anatomies 20 --test-anatomies 20 \
  --train-mode linearized --validation-mode nonlinear --test-mode nonlinear \
  --linearization-reference anatomy --jacobian-bank-size 0 --seed 20260719 \
  --maximum-linearized-component-nrmse 0.50 \
  --minimum-linearized-component-correlation 0.80

python -u -m gcnm_pvi.validate_hp_lp_beat_dataset \
  --root "${DATASET_ROOT}" --output "${DATASET_ROOT}/validation.json"
