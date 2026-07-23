#!/bin/bash
# One CPU array task generates the accepted 1,000-beat/50-sample pack for one ring.
#SBATCH --job-name=coord91-synth
#SBATCH --output=logs/coord91-synth_%A_%a.out
#SBATCH --error=logs/coord91-synth_%A_%a.err
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
export NUMEXPR_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export MPLCONFIGDIR="${TMPDIR:-/tmp}/coord91-synth-${SLURM_JOB_ID}-${SLURM_ARRAY_TASK_ID}"

RINGS=(US060 US065 US070 US075 US080 US085 US090 US095 US100 US105 US110 US115 US120 US125 US130)
RING="${RINGS[${SLURM_ARRAY_TASK_ID:?array task ID required}]}"
CONFIG="${REPO_ROOT}/configs/rings_b045/${RING}.yaml"
if [[ "${RING}" == "US120" ]]; then
  SOURCE_ROOT="${REPO_ROOT}/data/hp_lp_beats_US120_v1"
  DIFFERENTIAL_ROOT="${REPO_ROOT}/data/differential_US120_1000beats_clean_v1"
else
  SOURCE_ROOT="${REPO_ROOT}/data/hp_lp_beats_main_b045_v1/${RING}"
  DIFFERENTIAL_ROOT="${REPO_ROOT}/data/differential_main_b045_1000beats_clean_v1/${RING}"
fi

if [[ -f "${SOURCE_ROOT}/validation.json" && -f "${DIFFERENTIAL_ROOT}/validation.json" ]]; then
  echo "[$(date --iso-8601=seconds)] reuse validated immutable pack ${RING}"
  exit 0
fi
if [[ -e "${SOURCE_ROOT}" || -e "${DIFFERENTIAL_ROOT}" ]]; then
  echo "partial or unvalidated immutable output exists for ${RING}" >&2
  exit 2
fi

echo "[$(date --iso-8601=seconds)] generating ${RING}: 200 anatomies, 1000 beats, 50000 frames"
python -u -m gcnm_pvi.generate_hp_lp_beat_dataset \
  --config "${CONFIG}" --output-root "${SOURCE_ROOT}" \
  --train-anatomies 160 --validation-anatomies 20 --test-anatomies 20 \
  --train-mode linearized --validation-mode nonlinear --test-mode nonlinear \
  --linearization-reference anatomy --jacobian-bank-size 0 --seed 20260719 \
  --maximum-linearized-component-nrmse 0.50 \
  --minimum-linearized-component-correlation 0.80
python -u -m gcnm_pvi.validate_hp_lp_beat_dataset \
  --root "${SOURCE_ROOT}" --output "${SOURCE_ROOT}/validation.json"
python -u -m gcnm_pvi.make_differential_cohort_pack \
  --config "${CONFIG}" --dataset-root "${SOURCE_ROOT}" \
  --output-root "${DIFFERENTIAL_ROOT}" --voltage-key V_clean_delta_reference
python -u -m gcnm_pvi.validate_differential_cohort_pack \
  --root "${DIFFERENTIAL_ROOT}" --output "${DIFFERENTIAL_ROOT}/validation.json"
echo "[$(date --iso-8601=seconds)] validated ${RING} synthetic pack complete"
