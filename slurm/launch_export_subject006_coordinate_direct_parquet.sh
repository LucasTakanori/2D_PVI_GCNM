#!/bin/bash
# Full subject006 mask05 export for the selected [S1,S2,dS2/dt] BP contract.
#SBATCH --job-name=coord-s006-parquet
#SBATCH --output=logs/coord-s006-parquet_%j.out
#SBATCH --error=logs/coord-s006-parquet_%j.err
#SBATCH --partition=ece_bst
#SBATCH --account=ece_bst
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=500GB
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
export GCNM_INFERENCE_BATCH_SIZE=512
export GCNM_COMPUTE_STAGE2_RESIDUALS=0
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export MPLCONFIGDIR="${TMPDIR:-/tmp}/matplotlib-coord-parquet-${SLURM_JOB_ID}"

MODEL_ROOT="${MODEL_ROOT:-${REPO_ROOT}/models/differential_US120_1000beats_v1/coordinate_direct}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${REPO_ROOT}/gcnm_parquet/subject006_coordinate_direct_s1_s2_ds2_v1}"
SPLIT_MANIFEST="${SPLIT_MANIFEST:-${REPO_ROOT}/data/splits/us120_pilot_subject006_subject010_mask05_v1.json}"
REGISTRY="${REGISTRY:-${REPO_ROOT}/data/registries/main_b045_v1.json}"
RESUME="${RESUME:-0}"
if [[ ! -f "${MODEL_ROOT}/coordinate_direct_0.pt" || ! -f "${MODEL_ROOT}/coordinate_direct_1.pt" ]]; then
  echo "coordinate-direct checkpoints are missing" >&2
  exit 2
fi
if [[ "${RESUME}" == 1 ]]; then
  [[ -f "${OUTPUT_ROOT}/_INCOMPLETE" ]] || {
    echo "resume requested without an incomplete output root: ${OUTPUT_ROOT}" >&2
    exit 3
  }
elif [[ -e "${OUTPUT_ROOT}" ]]; then
  echo "immutable Parquet output already exists: ${OUTPUT_ROOT}" >&2
  exit 3
fi

RESUME_ARGS=()
[[ "${RESUME}" == 1 ]] && RESUME_ARGS+=(--resume)
echo "[$(date --iso-8601=seconds)] exporting all subject006 mask05 rows"
python -u -m gcnm_pvi.export_coordinate_direct_parquet \
  --registry "${REGISTRY}" \
  --checkpoint-dir "${MODEL_ROOT}" --model-name coordinate_direct \
  --output-root "${OUTPUT_ROOT}" --subjects subject006 \
  --sessions baseline valsalva pressor "${RESUME_ARGS[@]}" \
  --session-workers 3 --physics-workers 20 --batch-rows 8
python -u -m gcnm_pvi.validate_coordinate_direct_parquet \
  --root "${OUTPUT_ROOT}" --split-manifest "${SPLIT_MANIFEST}"
echo "[$(date --iso-8601=seconds)] coordinate-direct Parquet export validated"
