#!/bin/bash
# Deterministic HDF5-versus-Parquet CRT waveform storage control.
#SBATCH --job-name=s006-storage-parity
#SBATCH --output=logs/s006-storage-parity_%A_%a.out
#SBATCH --error=logs/s006-storage-parity_%A_%a.err
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
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export PYTHONHASHSEED=0

TASK_ID="${SLURM_ARRAY_TASK_ID:?submit as array 0-5%2}"
SEED="$((TASK_ID / 2))"
STORAGE_TASK="$((TASK_ID % 2))"
PVI_ML_ROOT="${PVI_ML_ROOT:-$(dirname "${REPO_ROOT}")/gia_bao/pvi_ml}"
REGISTRY="${REPO_ROOT}/data/registries/main_b045_v1.json"
SPLIT="${REPO_ROOT}/data/splits/us120_pilot_subject006_subject010_mask05_v1.json"
PARQUET="${REPO_ROOT}/gcnm_parquet/us120_pilot_reference_image_v1"
ARTIFACT_ROOT="${REPO_ROOT}/artifacts/subject006_hdf5_parquet_seeded_parity_v1/seed${SEED}"

if [[ "${STORAGE_TASK}" == "0" ]]; then
  echo "[$(date --iso-8601=seconds)] seeded native HDF5 control seed=${SEED}"
  python -u -m gcnm_pvi.train_original_pvi_bp \
    --pvi-ml-root "${PVI_ML_ROOT}" --registry "${REGISTRY}" \
    --split-manifest "${SPLIT}" --subject subject006 --input-mode img \
    --architecture crt --output-mode waveform --artifact-root "${ARTIFACT_ROOT}" \
    --max-epochs 5000 --num-workers 0 --seed "${SEED}" --deterministic
else
  echo "[$(date --iso-8601=seconds)] seeded reference-Parquet control seed=${SEED}"
  python -u -m gcnm_pvi.train_reference_parquet_bp \
    --pvi-ml-root "${PVI_ML_ROOT}" --parquet-root "${PARQUET}" \
    --split-manifest "${SPLIT}" --subject subject006 --input-mode img \
    --output-mode waveform --artifact-root "${ARTIFACT_ROOT}" \
    --max-epochs 5000 --num-workers 0 --seed "${SEED}" --deterministic
fi
