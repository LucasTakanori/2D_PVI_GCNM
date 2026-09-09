#!/bin/bash
# One held-out truth/Newton/S1/S2 GIF per ring checkpoint.
#SBATCH --job-name=coord91-sgif
#SBATCH --output=logs/coord91-sgif_%A_%a.out
#SBATCH --error=logs/coord91-sgif_%A_%a.err
#SBATCH --partition=ece_bst
#SBATCH --account=ece_bst
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=250GB
#SBATCH --time=10-00:00:00
#SBATCH --gres=gpu:1

set -euo pipefail
module load CUDA/12.9.0
module load gcc/11.2.0
REPO_ROOT="${REPO_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"
cd "${REPO_ROOT}"
source "${REPO_ROOT}/env/cluster.env"
source "$(dirname "$(dirname "${GCNM_PYTHON}")")/bin/activate"
export PYTHONPATH="${REPO_ROOT}:${PVI_SOLVER_ROOT}:${PYTHONPATH:-}"
export GCNM_PHYSICS_WORKERS="${SLURM_CPUS_PER_TASK}"
export GCNM_INFERENCE_BATCH_SIZE=512
export GCNM_COMPUTE_STAGE2_RESIDUALS=1
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export MPLCONFIGDIR="${TMPDIR:-/tmp}/coord91-sgif-${SLURM_JOB_ID}-${SLURM_ARRAY_TASK_ID}"
RINGS=(US060 US065 US070 US075 US080 US085 US090 US095 US100 US105 US110 US115 US120 US125 US130)
RING="${RINGS[${SLURM_ARRAY_TASK_ID:?array task ID required}]}"
if [[ "${RING}" == "US120" ]]; then
  DATA_ROOT="${REPO_ROOT}/data/differential_US120_1000beats_clean_v1"
  MODEL_ROOT="${REPO_ROOT}/models/differential_US120_1000beats_v1/coordinate_direct"
else
  DATA_ROOT="${REPO_ROOT}/data/differential_main_b045_1000beats_clean_v1/${RING}"
  MODEL_ROOT="${REPO_ROOT}/models/differential_main_b045_1000beats_v1/${RING}/coordinate_direct"
fi
REPORT_ROOT="${REPO_ROOT}/reports/coordinate_main_b045_1000beats_v1/${RING}"
OUTPUT="${REPORT_ROOT}/synthetic_exact_truth_newton_s1_s2_three_beats.gif"
if [[ -f "${OUTPUT}" && -f "${OUTPUT%.gif}.json" ]]; then
  echo "reuse complete ${RING} synthetic GIF"
  exit 0
fi
if [[ -e "${OUTPUT}" || -e "${OUTPUT%.gif}.json" ]]; then
  echo "partial synthetic GIF output for ${RING}" >&2
  exit 2
fi
mkdir -p "${REPORT_ROOT}"
python -u -m gcnm_pvi.synthetic_three_beat_gifs \
  --kind synthetic --family coordinate --component full \
  --config "${REPO_ROOT}/configs/rings_b045/${RING}.yaml" \
  --checkpoint-dir "${MODEL_ROOT}" --model-name coordinate_direct \
  --dataset "${DATA_ROOT}/test.npz" --first-beat 0 \
  --output "${OUTPUT}" --frame-ms 90
