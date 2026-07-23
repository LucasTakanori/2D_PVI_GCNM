#!/bin/bash
# Reuse the validated HP+LP synthetic source; no FEM regeneration and no GPU.
#SBATCH --job-name=diff-1000-prepare
#SBATCH --output=logs/diff-1000-prepare_%j.out
#SBATCH --error=logs/diff-1000-prepare_%j.err
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
export MPLCONFIGDIR="${TMPDIR:-/tmp}/matplotlib-diff-1000-${SLURM_JOB_ID}"

SOURCE_ROOT="${SOURCE_ROOT:-${REPO_ROOT}/data/hp_lp_beats_US120_v1}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${REPO_ROOT}/data/differential_US120_1000beats_clean_v1}"
if [[ ! -f "${SOURCE_ROOT}/validation.json" || -e "${OUTPUT_ROOT}" ]]; then
  echo "validated source is missing or immutable output already exists" >&2
  exit 2
fi

python - "${SOURCE_ROOT}/validation.json" <<'PY'
import json,sys
report=json.load(open(sys.argv[1]))
if report.get("status") != "pass" or report.get("anatomy_counts") != {
    "train": 160, "validation": 20, "test": 20
}:
    raise SystemExit("source HP/LP validation is not the accepted 160/20/20 cohort")
PY

python -u -m gcnm_pvi.make_differential_cohort_pack \
  --config "${REPO_ROOT}/configs/rings_b045/US120.yaml" \
  --dataset-root "${SOURCE_ROOT}" --output-root "${OUTPUT_ROOT}" \
  --voltage-key V_clean_delta_reference
python -u -m gcnm_pvi.validate_differential_cohort_pack \
  --root "${OUTPUT_ROOT}" --output "${OUTPUT_ROOT}/validation.json"
