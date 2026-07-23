#!/bin/bash
# CPU-only visual verification of literal serialized Parquet S1/S2 rows.
#SBATCH --job-name=coord-parquet-gifs
#SBATCH --output=logs/coord-parquet-gifs_%j.out
#SBATCH --error=logs/coord-parquet-gifs_%j.err
#SBATCH --partition=ece_bst
#SBATCH --account=ece_bst
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64GB
#SBATCH --time=10-00:00:00

set -euo pipefail
module load gcc/11.2.0
REPO_ROOT="${REPO_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"
cd "${REPO_ROOT}"
source "${REPO_ROOT}/env/cluster.env"
source "$(dirname "$(dirname "${GCNM_PYTHON}")")/bin/activate"
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"
export MPLCONFIGDIR="${TMPDIR:-/tmp}/matplotlib-coord-parquet-gifs-${SLURM_JOB_ID}"

PARQUET_ROOT="${PARQUET_ROOT:-${REPO_ROOT}/gcnm_parquet/subject006_coordinate_direct_s1_s2_ds2_v1}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${REPO_ROOT}/reports/subject006_coordinate_direct_parquet_gifs_v1}"
if [[ ! -f "${PARQUET_ROOT}/validation.json" ]]; then
  echo "validated coordinate-direct Parquet root is missing" >&2
  exit 2
fi
python -u -m gcnm_pvi.coordinate_direct_parquet_gifs \
  --registry "${REPO_ROOT}/data/registries/main_b045_v1.json" \
  --parquet-root "${PARQUET_ROOT}" --output-root "${OUTPUT_ROOT}" \
  --subject subject006 --sessions baseline valsalva pressor
echo "[$(date --iso-8601=seconds)] literal Parquet GIF verification complete"
