#!/bin/bash
# Run five minutes after an exporter starts; wait briefly if its first session
# shard has not closed yet, then compare live output metadata to source HDF5.
#SBATCH --job-name=gcnm-live-check
#SBATCH --output=logs/gcnm-live-check_%j.out
#SBATCH --error=logs/gcnm-live-check_%j.err
#SBATCH --partition=ece_bst
#SBATCH --account=ece_bst
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64GB
#SBATCH --time=01:00:00

set -euo pipefail
module load gcc/11.2.0
REPO_ROOT="${REPO_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"
cd "${REPO_ROOT}"
source "${REPO_ROOT}/env/cluster.env"
source "$(dirname "$(dirname "${GCNM_PYTHON}")")/bin/activate"
export PYTHONPATH="${REPO_ROOT}:${PVI_SOLVER_ROOT}:${PYTHONPATH:-}"
python -u -m gcnm_pvi.check_live_parquet_export \
  --root "${OUTPUT_ROOT:?set OUTPUT_ROOT}" \
  --registry "${REGISTRY:-${REPO_ROOT}/data/registries/main_b045_v1.json}" \
  --timeout-seconds 1200
