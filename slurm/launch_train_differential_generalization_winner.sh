#!/bin/bash
# Train only the one-beat gate winner, then run clean/augmented/GIF checks.
#SBATCH --job-name=diff-gcnm-generalize
#SBATCH --output=logs/diff-gcnm-generalize_%j.out
#SBATCH --error=logs/diff-gcnm-generalize_%j.err
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
export MPLCONFIGDIR="${TMPDIR:-/tmp}/matplotlib-diff-generalize-${SLURM_JOB_ID}"

DATA_ROOT="${DATA_ROOT:-${REPO_ROOT}/data/differential_generalization_US120_clean_v2}"
MODEL_ROOT="${MODEL_ROOT:-${REPO_ROOT}/models/differential_generalization_US120_v2/coordinate_direct}"
RESULT_ROOT="${RESULT_ROOT:-${REPO_ROOT}/data/differential_generalization_US120_results_v2/coordinate_direct}"
REPORT_ROOT="${REPORT_ROOT:-${REPO_ROOT}/reports/differential_generalization_US120_v2}"
if [[ ! -f "${DATA_ROOT}/manifest.json" || -e "${MODEL_ROOT}" || -e "${RESULT_ROOT}" ]]; then
  echo "missing validated data or immutable model/result output already exists" >&2
  exit 2
fi
mkdir -p "${REPORT_ROOT}"
GIF_PATH="${REPORT_ROOT}/coordinate_direct_truth_newton_s1_s2_three_beats.gif"
if [[ -e "${GIF_PATH}" || -e "${GIF_PATH%.gif}.json" ]]; then
  echo "immutable visual output already exists" >&2
  exit 3
fi

echo "[$(date --iso-8601=seconds)] starting coordinate-direct generalization"
nvidia-smi --query-gpu=index,name,memory.total,memory.free,utilization.gpu --format=csv

python -u -m gcnm_pvi.train_faithful_gcnm \
  --config "${REPO_ROOT}/configs/rings_b045/US120.yaml" \
  --train "${DATA_ROOT}/train.npz" \
  --validation "${DATA_ROOT}/validation.npz" \
  --model-name coordinate_direct \
  --models-dir "${MODEL_ROOT}" \
  --results-dir "${RESULT_ROOT}" \
  --physics-contract differential \
  --baseline-mode homogeneous --baseline-conductivity 0.7 \
  --output-mode direct --use-coordinates \
  --positive-weight 1.0 --background-weight 0.25 \
  --checkpoint-mode composite \
  --iterations 2 --epochs 150 --patience 30 --seed 0 \
  --batch-size 512 --loader-workers 4

python -u -m gcnm_pvi.evaluate_faithful_gcnm \
  --config "${REPO_ROOT}/configs/rings_b045/US120.yaml" \
  --test "${DATA_ROOT}/test.npz" \
  --model-name coordinate_direct --models-dir "${MODEL_ROOT}" \
  --out-dir "${RESULT_ROOT}/evaluation_clean" \
  --iterations 2 --target-kind clean --baseline-mode homogeneous \
  --baseline-conductivity 0.7 --save-examples 24

python -u -m gcnm_pvi.evaluate_faithful_gcnm \
  --config "${REPO_ROOT}/configs/rings_b045/US120.yaml" \
  --test "${DATA_ROOT}/test_augmented.npz" \
  --model-name coordinate_direct --models-dir "${MODEL_ROOT}" \
  --out-dir "${RESULT_ROOT}/evaluation_augmented" \
  --iterations 2 --target-kind clean --baseline-mode homogeneous \
  --baseline-conductivity 0.7 --save-examples 8

python -u -m gcnm_pvi.synthetic_three_beat_gifs \
  --kind synthetic --family coordinate --component full \
  --config "${REPO_ROOT}/configs/rings_b045/US120.yaml" \
  --checkpoint-dir "${MODEL_ROOT}" --model-name coordinate_direct \
  --dataset "${DATA_ROOT}/test.npz" --first-beat 0 \
  --output "${GIF_PATH}" --frame-ms 90

python -u -m gcnm_pvi.report_differential_generalization \
  --result-root "${RESULT_ROOT}" \
  --gif-sidecar "${GIF_PATH%.gif}.json" \
  --output "${REPORT_ROOT}/summary.md"

echo "[$(date --iso-8601=seconds)] generalization, robustness, and GIF complete"
