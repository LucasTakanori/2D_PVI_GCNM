#!/bin/bash
# Merge hard-linked ring shards and validate against source HDF5/sample IDs.
#SBATCH --job-name=coord91-final
#SBATCH --output=logs/coord91-final_%j.out
#SBATCH --error=logs/coord91-final_%j.err
#SBATCH --partition=ece_bst
#SBATCH --account=ece_bst
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=1000GB
#SBATCH --time=10-00:00:00

set -euo pipefail
module load gcc/11.2.0
REPO_ROOT="${REPO_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"
cd "${REPO_ROOT}"
source "${REPO_ROOT}/env/cluster.env"
source "$(dirname "$(dirname "${GCNM_PYTHON}")")/bin/activate"
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"
REGISTRY="${REPO_ROOT}/data/registries/main_b045_v1.json"
COORDINATE_ROOT="${COORDINATE_ROOT:?set COORDINATE_ROOT}"
SPLIT_MANIFEST="${SPLIT_MANIFEST:?set SPLIT_MANIFEST}"

if [[ ! -f "${COORDINATE_ROOT}/manifest.json" ]]; then
  python -u -m gcnm_pvi.full_coordinate_rollout merge-coordinate \
    --root "${COORDINATE_ROOT}" --registry "${REGISTRY}"
fi
python -u -m gcnm_pvi.validate_coordinate_direct_parquet \
  --root "${COORDINATE_ROOT}" --split-manifest "${SPLIT_MANIFEST}"
python -u -m gcnm_pvi.full_coordinate_rollout preflight-hdf5 \
  --coordinate-root "${COORDINATE_ROOT}" --registry "${REGISTRY}" \
  --split-manifest "${SPLIT_MANIFEST}"
