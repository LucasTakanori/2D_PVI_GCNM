#!/bin/bash
#SBATCH --job-name=gcnm6-pd-view
#SBATCH --output=logs/gcnm_population_6ch_disjoint/gcnm6-pd-view_%A_%a.out
#SBATCH --error=logs/gcnm_population_6ch_disjoint/gcnm6-pd-view_%A_%a.err
#SBATCH --partition=ece_bst
#SBATCH --account=ece_bst
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=250GB
#SBATCH --time=2-00:00:00
#SBATCH --array=0-1

set -euo pipefail
module load gcc/11.2.0
REPO_ROOT="${REPO_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"
cd "${REPO_ROOT}"
source "${REPO_ROOT}/env/cluster.env"
source "$(dirname "$(dirname "${GCNM_PYTHON}")")/bin/activate"
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

SCRATCH_BASE="${SCRATCH_BASE:-/mmfs1/scratch/${USER}/gcnm_population_6ch}"
BASE_CACHE_ROOT="${BASE_CACHE_ROOT:-${SCRATCH_BASE}/img2wave_main_within_seed42_exact_pvi_v3}"
COMPLETION_ROOT="${COMPLETION_ROOT:-${SCRATCH_BASE}/img2wave_legacy_pd_missing_payload_v1}"
SOURCE_SPLIT_MANIFEST="${SOURCE_SPLIT_MANIFEST:-${REPO_ROOT}/data/manifests/pw_population_within_mask05_seed42_v1.json}"
RUNTIME_ROOT="${RUNTIME_ROOT:-/tmp/${USER}/gcnm_population_6ch/pd_view_${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}}"

case "${SLURM_ARRAY_TASK_ID}" in
  0)
    label=pd13
    REFERENCE_CHECKPOINT="${PD13_CHECKPOINT:-$(dirname "${REPO_ROOT}")/ml-experiments/archived/artifacts_old/ps13-crt-img-to-waveform/checkpoints/dataset_lazy_checkpoints.pth}"
    SPLIT_MANIFEST="${PD13_SPLIT_MANIFEST:-${SCRATCH_BASE}/manifests/pd13_population_disjoint_legacy_v1.json}"
    CACHE_ROOT="${CRT_CACHE_ROOT:-${SCRATCH_BASE}/pd13_img2wave_legacy_disjoint_exact_pvi_view_v4}"
    ;;
  1)
    label=pd17
    REFERENCE_CHECKPOINT="${PD17_CHECKPOINT:-$(dirname "${REPO_ROOT}")/ml-experiments/archived/artifacts_old/ps17-samba-img-to-waveform/checkpoints/dataset_lazy_checkpoints.pth}"
    SPLIT_MANIFEST="${PD17_SPLIT_MANIFEST:-${SCRATCH_BASE}/manifests/pd17_population_disjoint_legacy_v1.json}"
    CACHE_ROOT="${SAMBA_CACHE_ROOT:-${SCRATCH_BASE}/pd17_img2wave_legacy_disjoint_exact_pvi_view_v4}"
    ;;
  *) echo "invalid task ${SLURM_ARRAY_TASK_ID}" >&2; exit 2 ;;
esac

for path in "${SCRATCH_BASE}" "${BASE_CACHE_ROOT}" "${COMPLETION_ROOT}" "${SPLIT_MANIFEST}" "${CACHE_ROOT}"; do
  case "${path}" in
    /mmfs1/scratch/${USER}/*) ;;
    *) echo "generated cache data must remain in scratch: ${path}" >&2; exit 2 ;;
  esac
done
case "${RUNTIME_ROOT}" in
  /tmp/${USER}/*) ;;
  *) echo "runtime files must use node-local /tmp: ${RUNTIME_ROOT}" >&2; exit 2 ;;
esac
mkdir -p "$(dirname "${SPLIT_MANIFEST}")" "${RUNTIME_ROOT}/tmp" "${RUNTIME_ROOT}/xdg"
export TMPDIR="${RUNTIME_ROOT}/tmp"
export XDG_CACHE_HOME="${RUNTIME_ROOT}/xdg"

test -f "${BASE_CACHE_ROOT}/_SUCCESS"
test -f "${BASE_CACHE_ROOT}/manifest.json"
test -f "${COMPLETION_ROOT}/_SUCCESS"
test -f "${COMPLETION_ROOT}/manifest.json"
test -f "${SOURCE_SPLIT_MANIFEST}"
test -f "${REFERENCE_CHECKPOINT}"

if [[ ! -f "${SPLIT_MANIFEST}" ]]; then
  python -u scripts/build_population_disjoint_split.py \
    --source-manifest "${SOURCE_SPLIT_MANIFEST}" \
    --reference-checkpoint "${REFERENCE_CHECKPOINT}" \
    --all-identities-active \
    --output "${SPLIT_MANIFEST}" \
    --seed 42 \
    --test-size 0.1
fi

if [[ -f "${CACHE_ROOT}/_SUCCESS" && -f "${CACHE_ROOT}/manifest.json" ]]; then
  echo "validated ${label} disjoint index view already exists: ${CACHE_ROOT}"
  exit 0
fi

resume=()
if [[ -f "${CACHE_ROOT}/_INCOMPLETE" ]]; then
  resume+=(--resume)
fi
python -u scripts/build_population_6ch_disjoint_view.py \
  --source-root "${BASE_CACHE_ROOT}" \
  --additional-source-root "${COMPLETION_ROOT}" \
  --split-manifest "${SPLIT_MANIFEST}" \
  --output-root "${CACHE_ROOT}" \
  --epochs 501 \
  --batch-size 32 \
  --cluster-size 30 \
  --seed 42 \
  "${resume[@]}"
