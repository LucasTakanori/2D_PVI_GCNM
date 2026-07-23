#!/bin/bash
# CPU-only pvi_ml artifact export for finalized reference checkpoints.
# Array indices 0-5 correspond to the six image/subject006 tasks that launched
# before the lazy-dataset metadata contract was corrected.
#SBATCH --job-name=us120-ref-repair
#SBATCH --output=logs/us120-ref-repair_%A_%a.out
#SBATCH --error=logs/us120-ref-repair_%A_%a.err
#SBATCH --partition=ece_bst
#SBATCH --account=ece_bst
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64GB
#SBATCH --time=01:00:00

set -euo pipefail
module load gcc/11.2.0
REPO_ROOT="${REPO_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"
cd "${REPO_ROOT}"
source "${REPO_ROOT}/env/cluster.env"
source "$(dirname "$(dirname "${GCNM_PYTHON}")")/bin/activate"
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK}"

# Make direct/manual repair submissions reproducible instead of depending on
# environment variables inherited from the original submission wrapper.
TASK_MANIFEST="${TASK_MANIFEST:-${REPO_ROOT}/data/manifests/us120_reference_parquet_bp_v1.tsv}"
PVI_ML_ROOT="${PVI_ML_ROOT:-$(dirname "${REPO_ROOT}")/gia_bao/pvi_ml}"
SPLIT_MANIFEST="${SPLIT_MANIFEST:-${REPO_ROOT}/data/splits/us120_pilot_subject006_subject010_mask05_v1.json}"
ARTIFACT_ROOT="${ARTIFACT_ROOT:-${REPO_ROOT}/artifacts/pvi_bp_us120_reference_parquet_v1}"
for required in "${TASK_MANIFEST}" "${PVI_ML_ROOT}" "${SPLIT_MANIFEST}"; do
  [[ -e "${required}" ]] || { echo "missing repair input: ${required}" >&2; exit 2; }
done

TASK_ID="${SLURM_ARRAY_TASK_ID:?submit as array 0-5%4}"
IFS=$'\t' read -r MANIFEST_ID SUBJECT INPUT_MODE OUTPUT_MODE PARQUET_REL TARGET MASK_KEY \
  < <(sed -n "$((TASK_ID + 2))p" "${TASK_MANIFEST}")
if [[ "${MANIFEST_ID}" != "${TASK_ID}" || "${MASK_KEY}" != "mask05" ]]; then
  echo "invalid repair manifest row ${TASK_ID}" >&2
  exit 2
fi

python -u -m gcnm_pvi.train_reference_parquet_bp \
  --pvi-ml-root "${PVI_ML_ROOT}" \
  --parquet-root "${REPO_ROOT}/${PARQUET_REL}" \
  --split-manifest "${SPLIT_MANIFEST}" \
  --subject "${SUBJECT}" --input-mode "${INPUT_MODE}" --output-mode "${OUTPUT_MODE}" \
  --artifact-root "${ARTIFACT_ROOT}" \
  --num-workers 8 --postprocess-only
