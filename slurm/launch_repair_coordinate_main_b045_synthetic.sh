#!/bin/bash
# Validate completed ring packs and create only missing differential views.
#SBATCH --job-name=coord91-srepair
#SBATCH --output=logs/coord91-srepair_%A_%a.out
#SBATCH --error=logs/coord91-srepair_%A_%a.err
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
RINGS=(US060 US065 US070 US075 US080 US085 US090 US095 US100 US105 US110 US115 US120 US125 US130)
RING="${RINGS[${SLURM_ARRAY_TASK_ID:?array task ID required}]}"
if [[ "${RING}" == "US120" ]]; then
  SOURCE_ROOT="${REPO_ROOT}/data/hp_lp_beats_US120_v1"
  DIFFERENTIAL_ROOT="${REPO_ROOT}/data/differential_US120_1000beats_clean_v1"
else
  SOURCE_ROOT="${REPO_ROOT}/data/hp_lp_beats_main_b045_v1/${RING}"
  DIFFERENTIAL_ROOT="${REPO_ROOT}/data/differential_main_b045_1000beats_clean_v1/${RING}"
fi
CONFIG="${REPO_ROOT}/configs/rings_b045/${RING}.yaml"
if [[ -e "${SOURCE_ROOT}/_INCOMPLETE" || ! -f "${SOURCE_ROOT}/metadata.json" ]]; then
  echo "synthetic source is incomplete for ${RING}" >&2
  exit 2
fi
if [[ ! -f "${SOURCE_ROOT}/validation.json" ]]; then
  python -u -m gcnm_pvi.validate_hp_lp_beat_dataset \
    --root "${SOURCE_ROOT}" --output "${SOURCE_ROOT}/validation.json"
fi
if [[ ! -e "${DIFFERENTIAL_ROOT}" ]]; then
  python -u -m gcnm_pvi.make_differential_cohort_pack \
    --config "${CONFIG}" --dataset-root "${SOURCE_ROOT}" \
    --output-root "${DIFFERENTIAL_ROOT}" --voltage-key V_clean_delta_reference
fi
if [[ -e "${DIFFERENTIAL_ROOT}/_INCOMPLETE" || ! -f "${DIFFERENTIAL_ROOT}/manifest.json" ]]; then
  echo "differential pack is incomplete for ${RING}" >&2
  exit 3
fi
if [[ ! -f "${DIFFERENTIAL_ROOT}/validation.json" ]]; then
  python -u -m gcnm_pvi.validate_differential_cohort_pack \
    --root "${DIFFERENTIAL_ROOT}" --output "${DIFFERENTIAL_ROOT}/validation.json"
fi
echo "[$(date --iso-8601=seconds)] ${RING} synthetic source and differential pack validated"
