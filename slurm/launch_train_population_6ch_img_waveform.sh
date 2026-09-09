#!/bin/bash
#SBATCH --job-name=gcnm6-img2wave
#SBATCH --output=logs/gcnm6-img2wave_%A_%a.out
#SBATCH --error=logs/gcnm6-img2wave_%A_%a.err
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
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4

case "${SLURM_ARRAY_TASK_ID}" in
  0) experiment=crt ;;
  1) experiment=samba ;;
  *) echo "invalid task ${SLURM_ARRAY_TASK_ID}" >&2; exit 2 ;;
esac

PVI_ML_ROOT="${PVI_ML_ROOT:-$(dirname "${REPO_ROOT}")/gia_bao/pvi_ml}"
SPLIT_MODE="${SPLIT_MODE:-within}"
# Deliberately do not accept an environment override: model artifacts must
# always use the normal PVI repository artifact root.  Scratch is data-only.
ARTIFACT_ROOT="${REPO_ROOT}/artifacts"
RUNTIME_ROOT="${RUNTIME_ROOT:-/tmp/${USER}/gcnm_population_6ch/${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}}"
NUM_WORKERS="${NUM_WORKERS:-16}"

case "${SPLIT_MODE}" in
  within|disjoint) ;;
  *) echo "SPLIT_MODE must be within or disjoint" >&2; exit 2 ;;
esac

if [[ "${SPLIT_MODE}" == "disjoint" ]]; then
  case "${experiment}" in
    crt)
      CACHE_ROOT="${CRT_CACHE_ROOT:-${CACHE_ROOT:-/mmfs1/scratch/${USER}/gcnm_population_6ch/pd13_img2wave_legacy_disjoint_exact_pvi_view_v4}}"
      ;;
    samba)
      CACHE_ROOT="${SAMBA_CACHE_ROOT:-${CACHE_ROOT:-/mmfs1/scratch/${USER}/gcnm_population_6ch/pd17_img2wave_legacy_disjoint_exact_pvi_view_v4}}"
      ;;
  esac
else
  CACHE_ROOT="${CACHE_ROOT:-/mmfs1/scratch/${USER}/gcnm_population_6ch/img2wave_main_within_seed42_exact_pvi_v3}"
fi

case "${CACHE_ROOT}" in
  /mmfs1/scratch/${USER}/*) ;;
  *) echo "Parquet cache must be in scratch: ${CACHE_ROOT}" >&2; exit 2 ;;
esac
[[ "$(readlink -m "${ARTIFACT_ROOT}")" == "$(readlink -m "${REPO_ROOT}/artifacts")" ]] || {
  echo "PVI artifacts must use the repository artifact root: ${ARTIFACT_ROOT}" >&2
  exit 2
}
case "${RUNTIME_ROOT}" in
  /tmp/${USER}/*) ;;
  *) echo "runtime files must use node-local /tmp: ${RUNTIME_ROOT}" >&2; exit 2 ;;
esac
mkdir -p "${ARTIFACT_ROOT}" "${RUNTIME_ROOT}/tmp" "${RUNTIME_ROOT}/xdg" "${RUNTIME_ROOT}/torch"
export TMPDIR="${RUNTIME_ROOT}/tmp"
export XDG_CACHE_HOME="${RUNTIME_ROOT}/xdg"
export TORCH_HOME="${RUNTIME_ROOT}/torch"

test -f "${CACHE_ROOT}/_SUCCESS"
test -f "${CACHE_ROOT}/manifest.json"
python -u -m gcnm_pvi.train_population_6ch_bp \
  --experiment "${experiment}" \
  --pvi-ml-root "${PVI_ML_ROOT}" \
  --cache-root "${CACHE_ROOT}" \
  --artifact-root "${ARTIFACT_ROOT}" \
  --split-mode "${SPLIT_MODE}" \
  --seed 42 \
  --batch-size 32 \
  --num-workers "${NUM_WORKERS}" \
  --max-epochs 500
