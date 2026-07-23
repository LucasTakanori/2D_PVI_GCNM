#!/bin/bash
#SBATCH --job-name=gcnm-mesh-train
#SBATCH --output=logs/gcnm-mesh-train_%A_%a.out
#SBATCH --error=logs/gcnm-mesh-train_%A_%a.err
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
export OMP_NUM_THREADS="${TASK_CPUS:-${SLURM_CPUS_PER_TASK}}"

RINGS=(US060 US065 US070 US075 US080 US085 US090 US095 US100 US105 US110 US115 US120 US125 US130)
RING="${RINGS[${SLURM_ARRAY_TASK_ID}]}"
FAMILY="${FAMILY:?set FAMILY=coordinate, global_voltage_slots, or diffusion}"
CONFIG="${REPO_ROOT}/configs/rings_b045/${RING}.yaml"
DATASET_DIR="${REPO_ROOT}/data/mesh_training/${FAMILY}/${RING}"
MODEL_NAME="${FAMILY}_b045_${RING}_seed0"
MODEL_DIR="${REPO_ROOT}/models/mesh_representations/${FAMILY}/${RING}"
RESULT_DIR="${REPO_ROOT}/data/mesh_training_results/${FAMILY}/${RING}"
if [[ -e "${MODEL_DIR}" || -e "${RESULT_DIR}" ]]; then
  echo "ERROR: immutable model/result output exists for ${FAMILY}/${RING}" >&2
  exit 1
fi
if [[ "${FAMILY}" == "coordinate" ]]; then
  python -u -m gcnm_pvi.train_faithful_gcnm \
    --config "${CONFIG}" --train "${DATASET_DIR}/train.npz" \
    --validation "${DATASET_DIR}/validation.npz" --model-name "${MODEL_NAME}" \
    --models-dir "${MODEL_DIR}" --results-dir "${RESULT_DIR}" \
    --output-mode direct --use-coordinates --positive-weight 1 \
    --background-weight 0.25 --checkpoint-mode composite --seed 0 \
    --baseline-mode homogeneous --baseline-conductivity 0.7 --iterations 2
elif [[ "${FAMILY}" == "global_voltage_slots" ]]; then
  python -u -m gcnm_pvi.train_voltage_vessel_gcnm \
    --config "${CONFIG}" --train "${DATASET_DIR}/train.npz" \
    --validation "${DATASET_DIR}/validation.npz" \
    --train-anatomy "${DATASET_DIR}/train_anatomy.json" \
    --validation-anatomy "${DATASET_DIR}/validation_anatomy.json" \
    --model-name "${MODEL_NAME}" --models-dir "${MODEL_DIR}" --results-dir "${RESULT_DIR}" \
    --architecture mean_pool_dense_refiner --baseline-mode homogeneous \
    --baseline-conductivity 0.7 --conductivity-scale-mode global_p995 \
    --positive-weight 1 --background-weight 0.25 --dice-weight 0.05 \
    --slot-weight 0.2 --separation-weight 0.1 --attention-weight 0.05 \
    --correlation-weight 0 --minimum-center-separation 0.25 \
    --minimum-vessel-axis 0.05 --maximum-vessel-axis 0.23 --seed 0
elif [[ "${FAMILY}" == "diffusion" ]]; then
  python -u -m gcnm_pvi.train_voltage_vessel_gcnm \
    --config "${CONFIG}" --train "${DATASET_DIR}/train.npz" \
    --validation "${DATASET_DIR}/validation.npz" \
    --train-anatomy "${DATASET_DIR}/train_anatomy.json" \
    --validation-anatomy "${DATASET_DIR}/validation_anatomy.json" \
    --model-name "${MODEL_NAME}" --models-dir "${MODEL_DIR}" --results-dir "${RESULT_DIR}" \
    --architecture diffusion_slots --baseline-mode saved \
    --conductivity-scale-mode positive_p995 --positive-weight 1 --background-weight 0.25 \
    --dice-weight 0.05 --slot-weight 0.5 --separation-weight 0.15 \
    --attention-weight 0.05 --correlation-weight 0.10 \
    --minimum-center-separation 0.25 --minimum-vessel-axis 0.025 \
    --maximum-vessel-axis 0.23 --seed 0
else
  echo "invalid FAMILY=${FAMILY}" >&2
  exit 2
fi
python -m gcnm_pvi.mesh_run_manifest --record --family "${FAMILY}" --ring "${RING}" \
  --config "${CONFIG}" --model-dir "${MODEL_DIR}" --output "${MODEL_DIR}/run_manifest.json"
