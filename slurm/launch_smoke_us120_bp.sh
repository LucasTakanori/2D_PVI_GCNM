#!/bin/bash
# Four BP smoke tasks cover CRT/CRS x waveform/fiducials and checkpoint resume.
#SBATCH --job-name=us120-bp-smoke
#SBATCH --output=logs/us120-bp-smoke_%A_%a.out
#SBATCH --error=logs/us120-bp-smoke_%A_%a.err
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

TASK_ID="${SLURM_ARRAY_TASK_ID:?submit this script as array 0-3%4}"
ARCHITECTURES=(crt crt crs crs)
OUTPUT_MODES=(waveform fiducials waveform fiducials)
ARCHITECTURE="${ARCHITECTURES[${TASK_ID}]}"
OUTPUT_MODE="${OUTPUT_MODES[${TASK_ID}]}"
PARQUET_ROOT="${PARQUET_ROOT:-${REPO_ROOT}/gcnm_parquet/us120_pilot_coordinate_hp_lp_v1}"
ARTIFACT_ROOT="${SMOKE_ARTIFACT_ROOT:-${REPO_ROOT}/artifacts/pvi_bp_us120_hp_lp_smoke_v1}"
SPLIT_MANIFEST="${SPLIT_MANIFEST:-${REPO_ROOT}/data/splits/us120_pilot_subject006_subject010_mask05_v1.json}"
PVI_ML_ROOT="${PVI_ML_ROOT:-$(dirname "${REPO_ROOT}")/gia_bao/pvi_ml}"

common=(
  --pvi-ml-root "${PVI_ML_ROOT}"
  --parquet-root "${PARQUET_ROOT}"
  --split-manifest "${SPLIT_MANIFEST}"
  --subject subject006 --family coordinate
  --architecture "${ARCHITECTURE}" --output-mode "${OUTPUT_MODE}"
  --channel-mode 3ch --artifact-root "${ARTIFACT_ROOT}"
  --max-epochs 1 --num-workers 8
)

echo "first one-epoch pass: ${ARCHITECTURE}/${OUTPUT_MODE}"
python -u -m gcnm_pvi.train_pvi_bp "${common[@]}"
echo "resume pass: ${ARCHITECTURE}/${OUTPUT_MODE}"
python -u -m gcnm_pvi.train_pvi_bp "${common[@]}"

verification="${ARTIFACT_ROOT}/gcnm-coordinate-3ch-${ARCHITECTURE}-image-to-${OUTPUT_MODE}/main/verification/subject006.json"
python -c 'import json,sys; d=json.load(open(sys.argv[1])); assert d["status"]=="pass" and d["checkpoint_epoch"] >= 1, d' "${verification}"
