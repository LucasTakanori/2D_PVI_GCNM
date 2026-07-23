#!/bin/bash
#SBATCH --job-name=compare-storage-parity
#SBATCH --output=logs/compare-storage-parity_%j.out
#SBATCH --error=logs/compare-storage-parity_%j.err
#SBATCH --partition=ece_bst
#SBATCH --account=ece_bst
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8GB
#SBATCH --time=00:20:00

set -euo pipefail
module load gcc/11.2.0
REPO_ROOT="${REPO_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"
cd "${REPO_ROOT}"
source "${REPO_ROOT}/env/cluster.env"
source "$(dirname "$(dirname "${GCNM_PYTHON}")")/bin/activate"
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"
python scripts/compare_subject006_storage_parity.py \
  --artifact-root artifacts/subject006_hdf5_parquet_seeded_parity_v1 \
  --seeds 0 1 2 \
  --output reports/subject006_hdf5_parquet_seeded_parity_v1.json
