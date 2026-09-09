#!/bin/bash
# One array task trains one family/component pipeline on one GPU. Submit 0-3%4
# so HP/LP x coordinate/vessel use all four GPUs without sharing GPU memory.
#SBATCH --job-name=us120-hp-lp-gcnm
#SBATCH --output=logs/us120-hp-lp-gcnm_%A_%a.out
#SBATCH --error=logs/us120-hp-lp-gcnm_%A_%a.err
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
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export GCNM_PHYSICS_WORKERS="${SLURM_CPUS_PER_TASK}"
export MPLCONFIGDIR="${TMPDIR:-/tmp}/matplotlib-gcnm"

TASK_ID="${SLURM_ARRAY_TASK_ID:?submit this script as array 0-3}"
FAMILIES=(coordinate coordinate global_voltage_slots global_voltage_slots)
COMPONENTS=(hp lp hp lp)
FAMILY="${FAMILIES[${TASK_ID}]}"
COMPONENT="${COMPONENTS[${TASK_ID}]}"
RING=US120
CONFIG="${CONFIG:-${REPO_ROOT}/configs/rings_b045/${RING}.yaml}"
DATASET_ROOT="${DATASET_ROOT:-${REPO_ROOT}/data/hp_lp_beats_US120_v1}"
MODEL_ROOT="${MODEL_ROOT:-${REPO_ROOT}/models/hp_lp_us120_v1}"
RESULT_ROOT="${RESULT_ROOT:-${REPO_ROOT}/data/hp_lp_training_results_us120_v1}"
MODEL_NAME="${FAMILY}_${COMPONENT}_b045_${RING}_seed0"
MODEL_DIR="${MODEL_ROOT}/${FAMILY}/${COMPONENT}/${RING}"
RESULT_DIR="${RESULT_ROOT}/${FAMILY}/${COMPONENT}/${RING}"
DATA_DIR="${DATASET_ROOT}/${COMPONENT}"
GIF_DIR="${GIF_DIR:-${REPO_ROOT}/reports/hp_lp_us120_v1/synthetic_gifs}"

if [[ ! -f "${DATASET_ROOT}/validation.json" ]]; then
  echo "ERROR: validated synthetic dataset missing: ${DATASET_ROOT}" >&2
  exit 1
fi
if [[ -e "${MODEL_DIR}" || -e "${RESULT_DIR}" ]]; then
  echo "ERROR: immutable model/result exists for ${FAMILY}/${COMPONENT}" >&2
  exit 1
fi

echo "family=${FAMILY} component=${COMPONENT} dataset=${DATA_DIR}"
nvidia-smi
if [[ "${FAMILY}" == coordinate ]]; then
  python -u -m gcnm_pvi.train_faithful_gcnm \
    --config "${CONFIG}" --train "${DATA_DIR}/train.npz" \
    --validation "${DATA_DIR}/validation.npz" \
    --model-name "${MODEL_NAME}" --models-dir "${MODEL_DIR}" \
    --results-dir "${RESULT_DIR}" --output-mode direct --use-coordinates \
    --positive-weight 1 --background-weight 0.25 \
    --checkpoint-mode composite --seed 0 --baseline-mode homogeneous \
    --baseline-conductivity 0.7 --iterations 2
  python -u -m gcnm_pvi.evaluate_faithful_gcnm \
    --config "${CONFIG}" --test "${DATA_DIR}/test.npz" \
    --model-name "${MODEL_NAME}" --models-dir "${MODEL_DIR}" \
    --out-dir "${RESULT_DIR}/exact_nonlinear_test" --iterations 2 \
    --target-kind clean --baseline-mode homogeneous \
    --baseline-conductivity 0.7 --skip-lm-control --save-examples 0
else
  python -u -m gcnm_pvi.train_voltage_vessel_gcnm \
    --config "${CONFIG}" --train "${DATA_DIR}/train.npz" \
    --validation "${DATA_DIR}/validation.npz" \
    --train-anatomy "${DATA_DIR}/train_anatomy.json" \
    --validation-anatomy "${DATA_DIR}/validation_anatomy.json" \
    --model-name "${MODEL_NAME}" --models-dir "${MODEL_DIR}" \
    --results-dir "${RESULT_DIR}" --architecture beat_voltage_slots \
    --baseline-mode homogeneous --baseline-conductivity 0.7 \
    --conductivity-scale-mode global_p995 \
    --positive-weight 1 --background-weight 0.25 --dice-weight 0.05 \
    --slot-weight 0.2 --separation-weight 0.1 --attention-weight 0.05 \
    --correlation-weight 0 --minimum-center-separation 0.25 \
    --minimum-vessel-axis 0.05 --maximum-vessel-axis 0.23 --seed 0
  python -u -m gcnm_pvi.evaluate_voltage_vessel_gcnm \
    --config "${CONFIG}" --test "${DATA_DIR}/test.npz" \
    --localizer "${MODEL_DIR}/${MODEL_NAME}_localizer.pt" \
    --refiner "${MODEL_DIR}/${MODEL_NAME}_refiner.pt" \
    --out-dir "${RESULT_DIR}/exact_nonlinear_test" --target-kind clean
fi

python -u -m gcnm_pvi.mesh_run_manifest --record \
  --family "${FAMILY}" --ring "${RING}" --config "${CONFIG}" \
  --model-dir "${MODEL_DIR}" --output "${MODEL_DIR}/run_manifest.json"

mkdir -p "${GIF_DIR}"
python -u -m gcnm_pvi.synthetic_three_beat_gifs \
  --kind synthetic --family "${FAMILY}" --component "${COMPONENT}" \
  --config "${CONFIG}" --checkpoint-dir "${MODEL_DIR}" \
  --model-name "${MODEL_NAME}" --dataset "${DATA_DIR}/test.npz" \
  --output "${GIF_DIR}/${FAMILY}_${COMPONENT}_truth_newton_s1_s2.gif"
