#!/bin/bash
#SBATCH --job-name=gcnm6-exact-bench
#SBATCH --output=logs/gcnm6-exact-bench_%j.out
#SBATCH --error=logs/gcnm6-exact-bench_%j.err
#SBATCH --partition=ece_bst
#SBATCH --account=ece_bst
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96GB
#SBATCH --time=04:00:00

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

SOURCE_FILE="${SOURCE_FILE:-/mmfs1/scratch/${USER}/gcnm_population_6ch/pw_main_within_seed42_v1/train/US060-part-00000-train.parquet}"
BENCHMARK_ROOT="${BENCHMARK_ROOT:-/mmfs1/scratch/${USER}/gcnm_population_6ch/benchmarks/exact_pvi_rg1_1024_v1}"
RUNTIME_ROOT="${RUNTIME_ROOT:-/mmfs1/scratch/${USER}/gcnm_population_6ch/runtime/benchmark_${SLURM_JOB_ID}}"
BENCHMARK_WORKERS="${BENCHMARK_WORKERS:-16}"

for path in "${SOURCE_FILE}" "${BENCHMARK_ROOT}" "${RUNTIME_ROOT}"; do
  case "${path}" in
    /mmfs1/scratch/${USER}/*) ;;
    *) echo "refusing non-scratch benchmark I/O path: ${path}" >&2; exit 2 ;;
  esac
done
mkdir -p "${RUNTIME_ROOT}/tmp" "${RUNTIME_ROOT}/xdg"
export TMPDIR="${RUNTIME_ROOT}/tmp"
export XDG_CACHE_HOME="${RUNTIME_ROOT}/xdg"

python -u scripts/benchmark_population_6ch_exact_io.py \
  --source-file "${SOURCE_FILE}" \
  --scratch-root "${BENCHMARK_ROOT}" \
  --rows 1024 \
  --workers "${BENCHMARK_WORKERS}"
