#!/bin/bash
# Continue after already-completed FEM shards/merge: validate and make view.
#SBATCH --job-name=diff-cohort-finalize
#SBATCH --output=logs/diff-cohort-finalize_%j.out
#SBATCH --error=logs/diff-cohort-finalize_%j.err
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
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export MPLCONFIGDIR="${TMPDIR:-/tmp}/matplotlib-diff-finalize-${SLURM_JOB_ID}"

MERGED_ROOT="${REPO_ROOT}/data/full_band_beats_US120_differential_generalization_v2"
DIFFERENTIAL_ROOT="${REPO_ROOT}/data/differential_generalization_US120_clean_v2"
if [[ ! -f "${MERGED_ROOT}/metadata.json" || -e "${MERGED_ROOT}/validation.json" || -e "${DIFFERENTIAL_ROOT}" ]]; then
  echo "missing merged source or immutable final output already exists" >&2
  exit 2
fi

python -u -m gcnm_pvi.validate_hp_lp_beat_dataset \
  --root "${MERGED_ROOT}" --allow-unverified-exact \
  --output "${MERGED_ROOT}/validation.json"
python -u -m gcnm_pvi.make_differential_cohort_pack \
  --config "${REPO_ROOT}/configs/rings_b045/US120.yaml" \
  --dataset-root "${MERGED_ROOT}" \
  --output-root "${DIFFERENTIAL_ROOT}" \
  --voltage-key V_clean_delta_reference
