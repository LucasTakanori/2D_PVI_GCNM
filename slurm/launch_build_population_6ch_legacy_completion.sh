#!/bin/bash
#SBATCH --job-name=gcnm6-pd-payload
#SBATCH --output=logs/gcnm_population_6ch_disjoint/gcnm6-pd-payload_%j.out
#SBATCH --error=logs/gcnm_population_6ch_disjoint/gcnm6-pd-payload_%j.err
#SBATCH --partition=ece_bst
#SBATCH --account=ece_bst
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=250GB
#SBATCH --time=10-00:00:00

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
PD13_SPLIT_MANIFEST="${PD13_SPLIT_MANIFEST:-${SCRATCH_BASE}/manifests/pd13_population_disjoint_legacy_v1.json}"
PD13_CHECKPOINT="${PD13_CHECKPOINT:-$(dirname "${REPO_ROOT}")/ml-experiments/archived/artifacts_old/ps13-crt-img-to-waveform/checkpoints/dataset_lazy_checkpoints.pth}"
COORDINATE_ROOT="${COORDINATE_ROOT:-${REPO_ROOT}/gcnm_parquet/coordinate_direct_main_b045_v1}"
REGISTRY="${REGISTRY:-${REPO_ROOT}/data/registries/main_b045_v1.json}"
RUNTIME_ROOT="${RUNTIME_ROOT:-/tmp/${USER}/gcnm_population_6ch/pd_payload_${SLURM_JOB_ID}}"

for path in "${SCRATCH_BASE}" "${BASE_CACHE_ROOT}" "${COMPLETION_ROOT}" "${PD13_SPLIT_MANIFEST}"; do
  case "${path}" in
    /mmfs1/scratch/${USER}/*) ;;
    *) echo "generated Parquet data must remain in scratch: ${path}" >&2; exit 2 ;;
  esac
done
case "${RUNTIME_ROOT}" in
  /tmp/${USER}/*) ;;
  *) echo "runtime files must use node-local /tmp: ${RUNTIME_ROOT}" >&2; exit 2 ;;
esac
mkdir -p "$(dirname "${PD13_SPLIT_MANIFEST}")" "${RUNTIME_ROOT}/tmp" "${RUNTIME_ROOT}/xdg"
export TMPDIR="${RUNTIME_ROOT}/tmp"
export XDG_CACHE_HOME="${RUNTIME_ROOT}/xdg"

test -f "${BASE_CACHE_ROOT}/_SUCCESS"
test -f "${BASE_CACHE_ROOT}/manifest.json"
test -f "${SOURCE_SPLIT_MANIFEST}"
test -f "${PD13_CHECKPOINT}"
test -f "${COORDINATE_ROOT}/manifest.json"
test -f "${REGISTRY}"

if [[ ! -f "${PD13_SPLIT_MANIFEST}" ]]; then
  python -u scripts/build_population_disjoint_split.py \
    --source-manifest "${SOURCE_SPLIT_MANIFEST}" \
    --reference-checkpoint "${PD13_CHECKPOINT}" \
    --all-identities-active \
    --output "${PD13_SPLIT_MANIFEST}" \
    --seed 42 \
    --test-size 0.1
fi

if [[ -f "${COMPLETION_ROOT}/_SUCCESS" && -f "${COMPLETION_ROOT}/manifest.json" ]]; then
  echo "validated legacy payload completion already exists: ${COMPLETION_ROOT}"
  exit 0
fi

resume=()
if [[ -f "${COMPLETION_ROOT}/_INCOMPLETE" ]]; then
  resume+=(--resume)
fi
python -u scripts/build_population_6ch_cache.py \
  --coordinate-root "${COORDINATE_ROOT}" \
  --registry "${REGISTRY}" \
  --split-manifest "${PD13_SPLIT_MANIFEST}" \
  --skip-sample-ids-from-cache-root "${BASE_CACHE_ROOT}" \
  --output-root "${COMPLETION_ROOT}" \
  --workers 16 \
  --row-group-size 1 \
  "${resume[@]}"
