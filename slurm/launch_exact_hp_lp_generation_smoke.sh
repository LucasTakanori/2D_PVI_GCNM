#!/bin/bash
# One exact validation/test anatomy proves nonlinear solves and acceleration
# metrics before the 200-anatomy production generation is released.
#SBATCH --job-name=hp-lp-exact-smoke
#SBATCH --output=logs/hp-lp-exact-smoke_%j.out
#SBATCH --error=logs/hp-lp-exact-smoke_%j.err
#SBATCH --partition=ece_bst
#SBATCH --account=ece_bst
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=250GB
#SBATCH --time=10-00:00:00

set -euo pipefail
module load gcc/11.2.0
REPO_ROOT="${REPO_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"
cd "${REPO_ROOT}"
source "${REPO_ROOT}/env/cluster.env"
source "$(dirname "$(dirname "${GCNM_PYTHON}")")/bin/activate"
export PYTHONPATH="${REPO_ROOT}:${PVI_SOLVER_ROOT}:${PYTHONPATH:-}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export MPLCONFIGDIR="${TMPDIR:-/tmp}/matplotlib-gcnm"

SMOKE_ROOT="${TMPDIR:-/tmp}/pvi-gcnm-hp-lp-exact-${SLURM_JOB_ID}"
python -u -m gcnm_pvi.generate_hp_lp_beat_dataset \
  --config "${REPO_ROOT}/configs/rings_b045/US120.yaml" \
  --output-root "${SMOKE_ROOT}" \
  --train-anatomies 1 --validation-anatomies 1 --test-anatomies 1 \
  --train-mode linearized --validation-mode nonlinear --test-mode nonlinear \
  --linearization-reference anatomy --jacobian-bank-size 0 --seed 20260720 \
  --maximum-linearized-component-nrmse 0.50 \
  --minimum-linearized-component-correlation 0.80
python -u -m gcnm_pvi.validate_hp_lp_beat_dataset \
  --root "${SMOKE_ROOT}" --allow-unverified-exact \
  --output "${SMOKE_ROOT}/validation.json"
