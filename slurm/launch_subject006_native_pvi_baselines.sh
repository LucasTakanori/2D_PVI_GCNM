#!/bin/bash
# Strict subject006-only native HDF5 PVI baselines: image and BioZ waveform CRT.
#SBATCH --job-name=s006-pvi-native
#SBATCH --output=logs/s006-pvi-native_%A_%a.out
#SBATCH --error=logs/s006-pvi-native_%A_%a.err
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

PVI_ML_ROOT="${PVI_ML_ROOT:-$(dirname "${REPO_ROOT}")/gia_bao/pvi_ml}"
REGISTRY="${REGISTRY:-${REPO_ROOT}/data/registries/main_b045_v1.json}"
SPLIT_MANIFEST="${SPLIT_MANIFEST:-${REPO_ROOT}/data/splits/us120_pilot_subject006_subject010_mask05_v1.json}"
ARTIFACT_ROOT="${ARTIFACT_ROOT:-${REPO_ROOT}/artifacts/pvi_ml_subject006_native_v1}"
for required in "${PVI_ML_ROOT}" "${REGISTRY}" "${SPLIT_MANIFEST}"; do
  [[ -e "${required}" ]] || { echo "missing native baseline input: ${required}" >&2; exit 2; }
done
git -C "${PVI_ML_ROOT}" diff --quiet
git -C "${PVI_ML_ROOT}" diff --cached --quiet

TASK_ID="${SLURM_ARRAY_TASK_ID:?submit exactly as array 0-1%2}"
INPUT_MODES=(img bioz)
if [[ "${TASK_ID}" -lt 0 || "${TASK_ID}" -gt 1 ]]; then
  echo "this launcher permits only task 0 or 1" >&2
  exit 2
fi
INPUT_MODE="${INPUT_MODES[${TASK_ID}]}"
TARGET="subject006-crt-${INPUT_MODE}-to-waveform"
if [[ -e "${ARTIFACT_ROOT}/${TARGET}" ]]; then
  echo "immutable native model folder already exists: ${ARTIFACT_ROOT}/${TARGET}" >&2
  exit 2
fi

export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"
export PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
echo "pvi_ml_commit=$(git -C "${PVI_ML_ROOT}" rev-parse HEAD)"
echo "subject=subject006 input=${INPUT_MODE} target=waveform artifact=${TARGET}"

python -B -u -m gcnm_pvi.train_original_pvi_bp \
  --pvi-ml-root "${PVI_ML_ROOT}" \
  --registry "${REGISTRY}" \
  --split-manifest "${SPLIT_MANIFEST}" \
  --subject subject006 --architecture crt --input-mode "${INPUT_MODE}" \
  --output-mode waveform --target "${TARGET}" \
  --artifact-root "${ARTIFACT_ROOT}" \
  --max-epochs 5000 --num-workers 0 --native-artifacts-only
