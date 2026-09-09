#!/bin/bash
#SBATCH --job-name=mesh-truth-gcnm
#SBATCH --output=logs/mesh-truth-gcnm_%A_%a.out
#SBATCH --error=logs/mesh-truth-gcnm_%A_%a.err
#SBATCH --partition=ece_bst
#SBATCH --account=ece_bst
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=200GB
#SBATCH --time=02:00:00
#SBATCH --gres=gpu:1
# Three export lanes use 48 CPUs/3 GPUs; the fourth 16-CPU/GPU lane remains
# available for the concurrent PW training job on the 64-CPU/4-GPU node.
#SBATCH --array=0-14%3

set -euo pipefail
module load CUDA/12.9.0
module load gcc/11.2.0

REPO_ROOT="${REPO_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"
cd "${REPO_ROOT}"
source "${REPO_ROOT}/env/cluster.env"
source "$(dirname "$(dirname "${GCNM_PYTHON}")")/bin/activate"
export PYTHONPATH="${REPO_ROOT}:${PVI_SOLVER_ROOT}:${PYTHONPATH:-}"
export GCNM_PHYSICS_WORKERS="${SLURM_CPUS_PER_TASK}"
export GCNM_INFERENCE_BATCH_SIZE=128
export GCNM_COMPUTE_STAGE2_RESIDUALS=0
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export MPLCONFIGDIR="${TMPDIR:-/tmp}/matplotlib-mesh-truth-gcnm-${SLURM_ARRAY_JOB_ID}-${SLURM_ARRAY_TASK_ID}"

RINGS=(US060 US065 US070 US075 US080 US085 US090 US095 US100 US105 US110 US115 US120 US125 US130)
RING="${RINGS[${SLURM_ARRAY_TASK_ID}]}"
OUTPUT_PARENT="${OUTPUT_PARENT:-${REPO_ROOT}/figures/gcnm_training_one_beat_per_mesh}"
OUTPUT_ROOT="${OUTPUT_PARENT}/${RING}"
SELECTION_CSV="${SELECTION_CSV:-${REPO_ROOT}/figures/gcnm_training_two_visible_arteries_selection.csv}"
EXPERIMENT="${EXPERIMENT:-coordinate_direct}"
PHYSICS_MESH_MODE="${PHYSICS_MESH_MODE:-coarse}"
ANATOMY_ID="$(awk -F, -v ring="${RING}" 'NR > 1 && $1 == ring {print $2; exit}' "${SELECTION_CSV}")"

if [[ -z "${ANATOMY_ID}" ]]; then
  echo "no selected anatomy ID for ${RING} in ${SELECTION_CSV}" >&2
  exit 4
fi

if [[ "${RING}" == "US120" ]]; then
  DATASET="${REPO_ROOT}/data/differential_US120_1000beats_clean_v1/train.npz"
  SOURCE_HP="${REPO_ROOT}/data/hp_lp_beats_US120_v1/hp/train.npz"
  MODEL_ROOT="${REPO_ROOT}/models/differential_US120_1000beats_v1/${EXPERIMENT}"
else
  DATASET="${REPO_ROOT}/data/differential_main_b045_1000beats_clean_v1/${RING}/train.npz"
  SOURCE_HP="${REPO_ROOT}/data/hp_lp_beats_main_b045_v1/${RING}/hp/train.npz"
  MODEL_ROOT="${REPO_ROOT}/models/differential_main_b045_1000beats_v1/${RING}/${EXPERIMENT}"
fi

if [[ -e "${OUTPUT_ROOT}" ]]; then
  echo "immutable ring output already exists: ${OUTPUT_ROOT}" >&2
  exit 3
fi

python -u scripts/export_us120_training_truth_gcnm_beats.py \
  --ring "${RING}" \
  --dataset "${DATASET}" \
  --source-hp "${SOURCE_HP}" \
  --config "${REPO_ROOT}/configs/rings_b045/${RING}.yaml" \
  --checkpoint-dir "${MODEL_ROOT}" \
  --model-name "${EXPERIMENT}" \
  --physics-mesh-mode "${PHYSICS_MESH_MODE}" \
  --output-root "${OUTPUT_ROOT}" \
  --anatomy-id "${ANATOMY_ID}" --image-size 900 \
  --shared-color-limit 0.03
