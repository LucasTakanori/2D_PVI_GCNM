#!/bin/bash
# Held-out exact synthetic truth/Newton/S1/S2 GIF after production training.
#SBATCH --job-name=coord-synth-gif
#SBATCH --output=logs/coord-synth-gif_%j.out
#SBATCH --error=logs/coord-synth-gif_%j.err
#SBATCH --partition=ece_bst
#SBATCH --account=ece_bst
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
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
export GCNM_STAGE2_RESIDUAL_STRIDE=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export MPLCONFIGDIR="${TMPDIR:-/tmp}/matplotlib-coord-synth-gif-${SLURM_JOB_ID}"

DATA_ROOT="${DATA_ROOT:-${REPO_ROOT}/data/differential_US120_1000beats_clean_v1}"
MODEL_ROOT="${MODEL_ROOT:-${REPO_ROOT}/models/differential_US120_1000beats_v1/coordinate_direct}"
REPORT_ROOT="${REPORT_ROOT:-${REPO_ROOT}/reports/coordinate_direct_US120_1000beats_v1}"
OUTPUT="${REPORT_ROOT}/synthetic_exact_truth_newton_s1_s2_three_beats.gif"
if [[ -e "${OUTPUT}" || -e "${OUTPUT%.gif}.json" ]]; then
  echo "immutable synthetic GIF already exists" >&2
  exit 3
fi
mkdir -p "${REPORT_ROOT}"
python -u -m gcnm_pvi.three_beat_gifs \
  --kind synthetic --family coordinate --component full \
  --config "${REPO_ROOT}/configs/rings_b045/US120.yaml" \
  --checkpoint-dir "${MODEL_ROOT}" --model-name coordinate_direct \
  --dataset "${DATA_ROOT}/test.npz" --first-beat 0 \
  --output "${OUTPUT}" --frame-ms 90
echo "[$(date --iso-8601=seconds)] synthetic held-out GIF complete: ${OUTPUT}"
