#!/usr/bin/env bash
# Submit the complete accepted dependency chain. Re-running is fail-closed.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"
source "${REPO_ROOT}/env/cluster.env"
mkdir -p logs data/manifests

REGISTRY="${REPO_ROOT}/data/registries/main_b045_v1.json"
SPLIT_MANIFEST="${REPO_ROOT}/data/manifests/pvi_subject_splits_mask05_v1.json"
COORDINATE_ROOT="${REPO_ROOT}/gcnm_parquet/coordinate_direct_main_b045_v1"
ARTIFACT_ROOT="${REPO_ROOT}/artifacts/coordinate_main_b045_bp_v1"
PVI_ML_ROOT="$(dirname "${REPO_ROOT}")/gia_bao/pvi_ml"
TASK_MANIFEST="${REPO_ROOT}/data/manifests/coordinate_main_b045_bp_364_v1.tsv"
SUBJECT_MANIFEST="${REPO_ROOT}/data/manifests/coordinate_main_b045_subjects_v1.txt"

for path in "${REGISTRY}" "${SPLIT_MANIFEST}" "${PVI_ML_ROOT}"; do
  [[ -e "${path}" ]] || { echo "required path missing: ${path}" >&2; exit 2; }
done
for path in "${COORDINATE_ROOT}" "${ARTIFACT_ROOT}"; do
  [[ ! -e "${path}" ]] || { echo "immutable rollout path already exists: ${path}" >&2; exit 2; }
done

"${GCNM_PYTHON}" -m gcnm_pvi.full_coordinate_rollout build-manifests \
  --registry "${REGISTRY}" --output-root "${REPO_ROOT}/data/manifests"
[[ "$(($(wc -l < "${TASK_MANIFEST}") - 1))" -eq 364 ]] || {
  echo "BP manifest does not contain exactly 364 runs" >&2; exit 2;
}
mkdir -p "${COORDINATE_ROOT}"
printf 'coordinate rollout export in progress\n' > "${COORDINATE_ROOT}/_INCOMPLETE"

synthetic_job="$(sbatch --parsable --array=0-14%4 \
  --export="ALL,REPO_ROOT=${REPO_ROOT}" \
  slurm/launch_generate_coordinate_main_b045_synthetic.sh)"
gcnm_job="$(sbatch --parsable --dependency="afterok:${synthetic_job}" \
  --export="ALL,REPO_ROOT=${REPO_ROOT}" \
  slurm/launch_train_coordinate_main_b045_packed.sh)"
synthetic_gif_job="$(sbatch --parsable --array=0-14%4 --dependency="afterok:${gcnm_job}" \
  --export="ALL,REPO_ROOT=${REPO_ROOT}" \
  slurm/launch_coordinate_main_b045_synthetic_gifs.sh)"
coordinate_export_job="$(sbatch --parsable --array=0-14%4 --dependency="afterok:${synthetic_gif_job}" \
  --export="ALL,REPO_ROOT=${REPO_ROOT},COORDINATE_ROOT=${COORDINATE_ROOT}" \
  slurm/launch_export_coordinate_main_b045_ring.sh)"
finalize_job="$(sbatch --parsable --dependency="afterok:${coordinate_export_job}" \
  --export="ALL,REPO_ROOT=${REPO_ROOT},COORDINATE_ROOT=${COORDINATE_ROOT},SPLIT_MANIFEST=${SPLIT_MANIFEST}" \
  slurm/launch_finalize_coordinate_main_b045_parquet.sh)"
bp_job="$(sbatch --parsable --dependency="afterok:${finalize_job}" \
  --export="ALL,REPO_ROOT=${REPO_ROOT},TASK_MANIFEST=${TASK_MANIFEST},PVI_ML_ROOT=${PVI_ML_ROOT},SPLIT_MANIFEST=${SPLIT_MANIFEST},ARTIFACT_ROOT=${ARTIFACT_ROOT}" \
  slurm/launch_train_coordinate_main_b045_bp_packed.sh)"
gif_job="$(sbatch --parsable --array=0-363%16 --dependency="afterok:${bp_job}" \
  --export="ALL,REPO_ROOT=${REPO_ROOT},TASK_MANIFEST=${TASK_MANIFEST},SPLIT_MANIFEST=${SPLIT_MANIFEST},ARTIFACT_ROOT=${ARTIFACT_ROOT}" \
  slurm/launch_coordinate_main_b045_bp_gifs_parallel.sh)"

python - "${REPO_ROOT}/reports/CURRENT_JOB_LEDGER_2026-07-21.md" \
  "${synthetic_job}" "${gcnm_job}" "${synthetic_gif_job}" "${coordinate_export_job}" \
  "${finalize_job}" "${bp_job}" "${gif_job}" <<'PY'
from datetime import datetime, timezone
from pathlib import Path
import sys
path=Path(sys.argv[1]); jobs=sys.argv[2:]
names=['synthetic ring packs','coordinate GCNM training','synthetic GCNM GIFs','coordinate Parquet export','Parquet/HDF5 merge validation','364 CRT BP models','364 BP artifact GIF tasks']
with path.open('a', encoding='utf-8') as f:
    f.write(f"\n## 91-subject coordinate rollout — {datetime.now(timezone.utc).isoformat()}\n\n")
    f.write('| Job | Description | Dependency status |\n|---:|---|---|\n')
    for job,name in zip(jobs,names): f.write(f'| `{job}` | {name} | queued dependency chain |\n')
PY

echo "synthetic packs: ${synthetic_job}"
echo "15 coordinate GCNMs (US120 reused): ${gcnm_job}"
echo "synthetic GIFs: ${synthetic_gif_job}"
echo "coordinate Parquet: ${coordinate_export_job}"
echo "Parquet/HDF5 validation gate: ${finalize_job}"
echo "364 CRT BP experiments: ${bp_job}"
echo "prediction-aligned GIFs: ${gif_job}"
