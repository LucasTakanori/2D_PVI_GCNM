#!/bin/bash
#SBATCH --job-name=gcnm6-exact-cache
#SBATCH --output=logs/gcnm6-exact-cache_%j.out
#SBATCH --error=logs/gcnm6-exact-cache_%j.err
#SBATCH --partition=ece_bst
#SBATCH --account=ece_bst
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128GB
#SBATCH --time=2-00:00:00

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

SOURCE_CACHE_ROOT="${SOURCE_CACHE_ROOT:-/mmfs1/scratch/${USER}/gcnm_population_6ch/pw_main_within_seed42_v1}"
CACHE_ROOT="${CACHE_ROOT:-/mmfs1/scratch/${USER}/gcnm_population_6ch/img2wave_main_within_seed42_exact_pvi_v3}"
RUNTIME_ROOT="${RUNTIME_ROOT:-/tmp/${USER}/gcnm_population_6ch/cache_${SLURM_JOB_ID}}"

for path in "${SOURCE_CACHE_ROOT}" "${CACHE_ROOT}"; do
  case "${path}" in
    /mmfs1/scratch/${USER}/*) ;;
    *) echo "Parquet cache must be in scratch: ${path}" >&2; exit 2 ;;
  esac
done
case "${RUNTIME_ROOT}" in
  /tmp/${USER}/*) ;;
  *) echo "runtime files must use node-local /tmp: ${RUNTIME_ROOT}" >&2; exit 2 ;;
esac
mkdir -p "${RUNTIME_ROOT}/tmp" "${RUNTIME_ROOT}/xdg"
export TMPDIR="${RUNTIME_ROOT}/tmp"
export XDG_CACHE_HOME="${RUNTIME_ROOT}/xdg"

test -f "${SOURCE_CACHE_ROOT}/_SUCCESS"
test -f "${SOURCE_CACHE_ROOT}/manifest.json"

if [[ -f "${CACHE_ROOT}/_SUCCESS" && -f "${CACHE_ROOT}/manifest.json" ]]; then
  echo "validated exact-PVI cache already exists: ${CACHE_ROOT}"
  exit 0
fi

resume=()
if [[ -f "${CACHE_ROOT}/_INCOMPLETE" ]]; then
  resume+=(--resume)
fi

python -u scripts/build_population_6ch_exact_pvi_cache.py \
  --source-root "${SOURCE_CACHE_ROOT}" \
  --output-root "${CACHE_ROOT}" \
  --workers 8 \
  --epochs 501 \
  --batch-size 32 \
  --cluster-size 48 \
  --seed 42 \
  "${resume[@]}"
