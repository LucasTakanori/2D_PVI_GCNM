#!/bin/bash
# Retrain the 14 non-US120 ring GCNMs with projected fine-mesh physics.
# US120 is reused from its completed, separately validated pilot.
# Two independent lanes keep one ring on each GPU and advance sequentially.
#SBATCH --job-name=gcnm-dual-15rings
#SBATCH --output=logs/gcnm-dual-15rings_%j.out
#SBATCH --error=logs/gcnm-dual-15rings_%j.err
#SBATCH --partition=ece_bst
#SBATCH --account=ece_bst
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=450GB
#SBATCH --time=10-00:00:00
#SBATCH --gres=gpu:2

set -euo pipefail

module load CUDA/12.9.0
module load gcc/11.2.0

REPO_ROOT="${REPO_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"
cd "${REPO_ROOT}"
source "${REPO_ROOT}/env/cluster.env"
source "$(dirname "$(dirname "${GCNM_PYTHON}")")/bin/activate"

export PYTHONUNBUFFERED=1
export PYTHONPATH="${REPO_ROOT}:${PVI_SOLVER_ROOT}:${PYTHONPATH:-}"
export GCNM_PHYSICS_EXECUTOR=process
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export MPLCONFIGDIR="${TMPDIR:-/tmp}/gcnm-dual-15rings-${SLURM_JOB_ID}"

EXPERIMENT="${EXPERIMENT:-coordinate_direct_projected_fine}"
RINGS=(US060 US065 US070 US075 US080 US085 US090 US095 US100 US105 US110 US115 US125 US130)
IFS=',' read -r -a GPUS <<< "${CUDA_VISIBLE_DEVICES:-0,1}"
if [[ "${#GPUS[@]}" -ne 2 ]]; then
  echo "expected two visible GPUs, found ${#GPUS[@]}: ${CUDA_VISIBLE_DEVICES:-unset}" >&2
  exit 2
fi

workers_per_lane=$((SLURM_CPUS_PER_TASK / 2))
if [[ "${workers_per_lane}" -lt 1 ]]; then
  echo "no CPU workers available for the two training lanes" >&2
  exit 2
fi

LOG_ROOT="${REPO_ROOT}/logs/coordinate_main_b045_projected_fine"
mkdir -p "${LOG_ROOT}"

train_ring() {
  local ring="$1"
  local gpu="$2"
  local data_root="${REPO_ROOT}/data/differential_main_b045_1000beats_clean_v1/${ring}"
  local model_root="${REPO_ROOT}/models/differential_main_b045_1000beats_v1/${ring}/${EXPERIMENT}"
  local result_root="${REPO_ROOT}/data/differential_main_b045_1000beats_results_v1/${ring}/${EXPERIMENT}"
  local report="${result_root}/${EXPERIMENT}_training_report.json"

  if [[ -f "${model_root}/${EXPERIMENT}_0.pt" \
        && -f "${model_root}/${EXPERIMENT}_1.pt" \
        && -f "${report}" ]]; then
    echo "[$(date --iso-8601=seconds)] reuse complete ${ring}"
    return 0
  fi
  if [[ ! -f "${data_root}/validation.json" ]]; then
    echo "validated dataset is missing for ${ring}: ${data_root}" >&2
    return 2
  fi
  if [[ -e "${model_root}" || -e "${result_root}" ]]; then
    echo "refusing to overwrite partial immutable output for ${ring}" >&2
    return 2
  fi

  echo "[$(date --iso-8601=seconds)] start ${ring} on GPU ${gpu} with ${workers_per_lane} physics workers"
  CUDA_VISIBLE_DEVICES="${gpu}" \
  GCNM_PHYSICS_WORKERS="${workers_per_lane}" \
  python -u -m gcnm_pvi.train_faithful_gcnm \
    --config "${REPO_ROOT}/configs/rings_b045/${ring}.yaml" \
    --train "${data_root}/train.npz" \
    --validation "${data_root}/validation.npz" \
    --model-name "${EXPERIMENT}" \
    --models-dir "${model_root}" \
    --results-dir "${result_root}" \
    --physics-contract differential \
    --physics-mesh-mode projected_fine \
    --baseline-mode homogeneous \
    --baseline-conductivity 0.7 \
    --output-mode direct \
    --use-coordinates \
    --positive-weight 1.0 \
    --background-weight 0.25 \
    --checkpoint-mode composite \
    --iterations 2 \
    --epochs 150 \
    --patience 30 \
    --seed 0 \
    --batch-size 512 \
    --loader-workers 2
  echo "[$(date --iso-8601=seconds)] complete ${ring} on GPU ${gpu}"
}

run_lane() {
  local lane="$1"
  local gpu="${GPUS[$lane]}"
  local index
  local ring
  for index in "${!RINGS[@]}"; do
    if (( index % 2 != lane )); then
      continue
    fi
    ring="${RINGS[$index]}"
    train_ring "${ring}" "${gpu}" \
      >"${LOG_ROOT}/${ring}_${SLURM_JOB_ID}.out" \
      2>"${LOG_ROOT}/${ring}_${SLURM_JOB_ID}.err"
  done
}

echo "[$(date --iso-8601=seconds)] projected-fine rollout begins"
echo "contract: F_f(P sigma_c), J_f(P sigma_c) P, coarse graph/output"
echo "GPUs=${CUDA_VISIBLE_DEVICES:-0,1} workers_per_lane=${workers_per_lane}"
nvidia-smi --query-gpu=index,name,memory.total,memory.free,utilization.gpu --format=csv

run_lane 0 &
lane0_pid=$!
run_lane 1 &
lane1_pid=$!

failed=0
if ! wait "${lane0_pid}"; then failed=1; fi
if ! wait "${lane1_pid}"; then failed=1; fi
if [[ "${failed}" -ne 0 ]]; then
  echo "one or more projected-fine ring trainings failed" >&2
  exit 3
fi

us120_model="${REPO_ROOT}/models/differential_US120_1000beats_v1/${EXPERIMENT}"
us120_result="${REPO_ROOT}/data/differential_US120_1000beats_results_v1/${EXPERIMENT}"
if [[ ! -f "${us120_model}/${EXPERIMENT}_0.pt" \
      || ! -f "${us120_model}/${EXPERIMENT}_1.pt" \
      || ! -f "${us120_result}/${EXPERIMENT}_training_report.json" ]]; then
  echo "completed US120 projected-fine pilot is missing" >&2
  exit 4
fi

echo "[$(date --iso-8601=seconds)] all 15 projected-fine ring GCNMs are complete"
