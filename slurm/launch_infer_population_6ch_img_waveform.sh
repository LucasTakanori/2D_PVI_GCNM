#!/bin/bash
#SBATCH --job-name=gcnm6-infer
#SBATCH --output=logs/gcnm_population_6ch_exact_pvi/gcnm6-infer_%x_%j.out
#SBATCH --error=logs/gcnm_population_6ch_exact_pvi/gcnm6-infer_%x_%j.err
#SBATCH --partition=ece_bst
#SBATCH --account=ece_bst
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=250GB
#SBATCH --time=1-00:00:00
#SBATCH --gres=gpu:1

set -euo pipefail
module load CUDA/12.9.0
module load gcc/11.2.0

REPO_ROOT="${REPO_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"
cd "${REPO_ROOT}"
source "${REPO_ROOT}/env/cluster.env"
source "$(dirname "$(dirname "${GCNM_PYTHON}")")/bin/activate"
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4

EXPERIMENT="${EXPERIMENT:-crt}"
SPLIT_MODE="${SPLIT_MODE:-within}"
case "${EXPERIMENT}" in
  crt|samba) ;;
  *) echo "EXPERIMENT must be crt or samba" >&2; exit 2 ;;
esac
case "${SPLIT_MODE}" in
  within|disjoint) ;;
  *) echo "SPLIT_MODE must be within or disjoint" >&2; exit 2 ;;
esac

PVI_ML_ROOT="${PVI_ML_ROOT:-$(dirname "${REPO_ROOT}")/gia_bao/pvi_ml}"
CACHE_ROOT="${CACHE_ROOT:-/mmfs1/scratch/${USER}/gcnm_population_6ch/img2wave_main_within_seed42_exact_pvi_v3}"
ARTIFACT_ROOT="${REPO_ROOT}/artifacts"
RUNTIME_ROOT="${RUNTIME_ROOT:-/tmp/${USER}/gcnm_population_6ch_inference/${SLURM_JOB_ID}}"

case "${CACHE_ROOT}" in
  /mmfs1/scratch/${USER}/*) ;;
  *) echo "Parquet cache must be in scratch: ${CACHE_ROOT}" >&2; exit 2 ;;
esac
case "${RUNTIME_ROOT}" in
  /tmp/${USER}/*) ;;
  *) echo "runtime files must use node-local /tmp: ${RUNTIME_ROOT}" >&2; exit 2 ;;
esac

mkdir -p "${RUNTIME_ROOT}/tmp" "${RUNTIME_ROOT}/xdg" "${RUNTIME_ROOT}/torch"
export TMPDIR="${RUNTIME_ROOT}/tmp"
export XDG_CACHE_HOME="${RUNTIME_ROOT}/xdg"
export TORCH_HOME="${RUNTIME_ROOT}/torch"

test -f "${CACHE_ROOT}/_SUCCESS"
test -f "${CACHE_ROOT}/manifest.json"
case "${SPLIT_MODE}:${EXPERIMENT}" in
  within:crt) TARGET="crt-gcnm6ch-img-to-waveform-exact-pvi" ;;
  within:samba) TARGET="samba-gcnm6ch-img-to-waveform-exact-pvi" ;;
  disjoint:crt) TARGET="pd13-crt-gcnm6ch-img-to-waveform-exact-pvi" ;;
  disjoint:samba) TARGET="pd17-samba-gcnm6ch-img-to-waveform-exact-pvi" ;;
esac
test -f "${ARTIFACT_ROOT}/${TARGET}/main/checkpoints/dataset_lazy_checkpoints.pth"

INFERENCE_ARGS=()
if [[ "${ALLOW_CHECKPOINT_CHANGE:-0}" == "1" ]]; then
  INFERENCE_ARGS+=(--allow-checkpoint-change)
fi

python -u -m gcnm_pvi.infer_population_6ch_bp \
  --experiment "${EXPERIMENT}" \
  --pvi-ml-root "${PVI_ML_ROOT}" \
  --cache-root "${CACHE_ROOT}" \
  --artifact-root "${ARTIFACT_ROOT}" \
  --split-mode "${SPLIT_MODE}" \
  --runtime-root "${RUNTIME_ROOT}" \
  --seed 42 \
  --batch-size 32 \
  --num-workers 16 \
  --device cuda:0 \
  --checkpoint-name dataset_lazy_checkpoints.pth \
  --parity-tolerance 1e-5 \
  "${INFERENCE_ARGS[@]}"
