#!/bin/bash
# Eight CPU-only exported-data GIFs: 2 families x 2 subjects x HP/LP.
#SBATCH --job-name=us120-real-gif
#SBATCH --output=logs/us120-real-gif_%A_%a.out
#SBATCH --error=logs/us120-real-gif_%A_%a.err
#SBATCH --partition=ece_bst
#SBATCH --account=ece_bst
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=16GB
#SBATCH --time=10-00:00:00

set -euo pipefail
module load gcc/11.2.0
REPO_ROOT="${REPO_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"
cd "${REPO_ROOT}"
source "${REPO_ROOT}/env/cluster.env"
source "$(dirname "$(dirname "${GCNM_PYTHON}")")/bin/activate"
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export MPLCONFIGDIR="${TMPDIR:-/tmp}/matplotlib-gcnm"

TASK_ID="${SLURM_ARRAY_TASK_ID:?submit this script as array 0-7%8}"
FAMILIES=(coordinate coordinate coordinate coordinate global_voltage_slots global_voltage_slots global_voltage_slots global_voltage_slots)
SUBJECTS=(subject006 subject006 subject010 subject010 subject006 subject006 subject010 subject010)
COMPONENTS=(hp lp hp lp hp lp hp lp)
FAMILY="${FAMILIES[${TASK_ID}]}"
SUBJECT="${SUBJECTS[${TASK_ID}]}"
COMPONENT="${COMPONENTS[${TASK_ID}]}"
HDF5="${PVI_DATA_ROOT}/main/${SUBJECT}_baseline_masked.h5"
PARQUET_ROOT="${PARQUET_ROOT:-${REPO_ROOT}/gcnm_parquet}"
FAMILY_PARQUET="${PARQUET_ROOT}/us120_pilot_${FAMILY}_hp_lp_v1"
GIF_ROOT="${GIF_ROOT:-${REPO_ROOT}/reports/hp_lp_us120_v1/real_gifs}"
mkdir -p "${GIF_ROOT}"

python -u -m gcnm_pvi.three_beat_gifs \
  --kind real --family "${FAMILY}" --component "${COMPONENT}" \
  --parquet-root "${FAMILY_PARQUET}" --hdf5 "${HDF5}" \
  --output "${GIF_ROOT}/${FAMILY}_${SUBJECT}_baseline_${COMPONENT}_newton_s1_s2.gif"
