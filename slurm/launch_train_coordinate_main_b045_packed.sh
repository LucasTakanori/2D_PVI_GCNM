#!/bin/bash
# Pack the 14 missing ring GCNMs onto four NVL GPUs (3-4 models/GPU).
#SBATCH --job-name=coord91-gcnm
#SBATCH --output=logs/coord91-gcnm_%j.out
#SBATCH --error=logs/coord91-gcnm_%j.err
#SBATCH --partition=ece_bst
#SBATCH --account=ece_bst
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=1000GB
#SBATCH --time=10-00:00:00
#SBATCH --gres=gpu:4

set -euo pipefail
module load CUDA/12.9.0
module load gcc/11.2.0
REPO_ROOT="${REPO_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"
cd "${REPO_ROOT}"
source "${REPO_ROOT}/env/cluster.env"
source "$(dirname "$(dirname "${GCNM_PYTHON}")")/bin/activate"
export PYTHONPATH="${REPO_ROOT}:${PVI_SOLVER_ROOT}:${PYTHONPATH:-}"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export GCNM_PHYSICS_WORKERS=4
export MPLCONFIGDIR="${TMPDIR:-/tmp}/coord91-gcnm-${SLURM_JOB_ID}"

RINGS=(US060 US065 US070 US075 US080 US085 US090 US095 US100 US105 US110 US115 US125 US130)
IFS=',' read -r -a GPUS <<< "${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
if [[ "${#GPUS[@]}" -ne 4 ]]; then
  echo "expected four visible GPUs, found ${#GPUS[@]}" >&2
  exit 2
fi
mkdir -p "${REPO_ROOT}/logs/coordinate_main_b045_training"
pids=()
for index in "${!RINGS[@]}"; do
  ring="${RINGS[${index}]}"
  gpu="${GPUS[$((index % 4))]}"
  data_root="${REPO_ROOT}/data/differential_main_b045_1000beats_clean_v1/${ring}"
  model_root="${REPO_ROOT}/models/differential_main_b045_1000beats_v1/${ring}/coordinate_direct"
  result_root="${REPO_ROOT}/data/differential_main_b045_1000beats_results_v1/${ring}/coordinate_direct"
  if [[ -f "${model_root}/coordinate_direct_0.pt" && -f "${model_root}/coordinate_direct_1.pt" ]]; then
    echo "reuse complete ${ring} checkpoint"
    continue
  fi
  if [[ ! -f "${data_root}/validation.json" || -e "${model_root}" || -e "${result_root}" ]]; then
    echo "missing validated input or partial output for ${ring}" >&2
    exit 2
  fi
  (
    export CUDA_VISIBLE_DEVICES="${gpu}"
    echo "[$(date --iso-8601=seconds)] ${ring} training on GPU ${gpu}"
    python -u -m gcnm_pvi.train_faithful_gcnm \
      --config "${REPO_ROOT}/configs/rings_b045/${ring}.yaml" \
      --train "${data_root}/train.npz" --validation "${data_root}/validation.npz" \
      --model-name coordinate_direct --models-dir "${model_root}" \
      --results-dir "${result_root}" --physics-contract differential \
      --baseline-mode homogeneous --baseline-conductivity 0.7 \
      --output-mode direct --use-coordinates \
      --positive-weight 1.0 --background-weight 0.25 \
      --checkpoint-mode composite --iterations 2 --epochs 150 --patience 30 \
      --seed 0 --batch-size 512 --loader-workers 0
    echo "[$(date --iso-8601=seconds)] ${ring} training complete"
  ) >"${REPO_ROOT}/logs/coordinate_main_b045_training/${ring}_${SLURM_JOB_ID}.out" \
    2>"${REPO_ROOT}/logs/coordinate_main_b045_training/${ring}_${SLURM_JOB_ID}.err" &
  pids+=("$!")
done

failed=0
for pid in "${pids[@]}"; do
  if ! wait "${pid}"; then failed=1; fi
done
if [[ "${failed}" -ne 0 ]]; then
  echo "one or more ring trainings failed" >&2
  exit 3
fi
echo "[$(date --iso-8601=seconds)] all 15 ring checkpoints available (US120 reused)"
