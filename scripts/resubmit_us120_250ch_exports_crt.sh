#!/usr/bin/env bash
# Resume the two failed 250-channel exports and submit their dependent CRT jobs.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"
mkdir -p logs

REGISTRY="${REPO_ROOT}/data/registries/us120_subject006_subject010_250ch_v1.json"
MODEL_ROOT="${REPO_ROOT}/models/differential_US120_1000beats_v1/coordinate_direct_250ch"
SPLIT_MANIFEST="${REPO_ROOT}/data/splits/us120_pilot_subject006_subject010_mask05_v1.json"
REFERENCE_ROOT="${REPO_ROOT}/gcnm_parquet/us120_pilot_reference_image_v1"

S006_PARQUET="${REPO_ROOT}/gcnm_parquet/subject006_coordinate_direct_250ch_s1_s2_ds2_v1"
S010_PARQUET="${REPO_ROOT}/gcnm_parquet/subject010_coordinate_direct_250ch_s1_s2_ds2_v1"
S006_COORD_ARTIFACTS="${REPO_ROOT}/artifacts/subject006_coordinate_direct_250ch_bp_v1"
S006_FUSED_ARTIFACTS="${REPO_ROOT}/artifacts/subject006_newton_coordinate_250ch_6ch_bp_v1"
S010_COORD_ARTIFACTS="${REPO_ROOT}/artifacts/subject010_coordinate_direct_250ch_bp_v1"
S010_FUSED_ARTIFACTS="${REPO_ROOT}/artifacts/subject010_newton_coordinate_250ch_6ch_bp_v1"

for required in \
  "${REGISTRY}" \
  "${MODEL_ROOT}/coordinate_direct_0.pt" \
  "${MODEL_ROOT}/coordinate_direct_1.pt" \
  "${SPLIT_MANIFEST}" \
  "${REFERENCE_ROOT}/manifest.json" \
  "${S006_PARQUET}/_INCOMPLETE" \
  "${S010_PARQUET}/_INCOMPLETE"; do
  [[ -e "${required}" ]] || {
    echo "required recovery input is missing: ${required}" >&2
    exit 2
  }
done

for output in \
  "${S006_COORD_ARTIFACTS}" \
  "${S006_FUSED_ARTIFACTS}" \
  "${S010_COORD_ARTIFACTS}" \
  "${S010_FUSED_ARTIFACTS}"; do
  [[ ! -e "${output}" ]] || {
    echo "refusing to overwrite artifact output: ${output}" >&2
    exit 3
  }
done

if [[ "${DRY_RUN:-0}" == 1 ]]; then
  echo "250-channel export and CRT recovery preflight passed"
  exit 0
fi

s006_export_job="$(
  sbatch --parsable \
    --job-name=coord250r-s006-export \
    --output=logs/coord250r-s006-export_%j.out \
    --error=logs/coord250r-s006-export_%j.err \
    --export="ALL,REPO_ROOT=${REPO_ROOT},MODEL_ROOT=${MODEL_ROOT},OUTPUT_ROOT=${S006_PARQUET},SPLIT_MANIFEST=${SPLIT_MANIFEST},REGISTRY=${REGISTRY},RESUME=1" \
    slurm/launch_export_subject006_coordinate_direct_parquet.sh
)"
s006_export_id="${s006_export_job%%;*}"

s010_export_job="$(
  sbatch --parsable \
    --job-name=coord250r-s010-export \
    --output=logs/coord250r-s010-export_%j.out \
    --error=logs/coord250r-s010-export_%j.err \
    --export="ALL,REPO_ROOT=${REPO_ROOT},MODEL_ROOT=${MODEL_ROOT},OUTPUT_ROOT=${S010_PARQUET},SPLIT_MANIFEST=${SPLIT_MANIFEST},REGISTRY=${REGISTRY},RESUME=1" \
    slurm/launch_export_subject010_coordinate_direct_parquet.sh
)"
s010_export_id="${s010_export_job%%;*}"

s006_coord_job="$(
  sbatch --parsable --array=0-1%2 \
    --dependency="afterok:${s006_export_id}" \
    --job-name=coord250r-s006-crt3 \
    --output=logs/coord250r-s006-crt3_%A_%a.out \
    --error=logs/coord250r-s006-crt3_%A_%a.err \
    --export="ALL,REPO_ROOT=${REPO_ROOT},PARQUET_ROOT=${S006_PARQUET},SPLIT_MANIFEST=${SPLIT_MANIFEST},ARTIFACT_ROOT=${S006_COORD_ARTIFACTS}" \
    slurm/launch_train_subject006_coordinate_direct_bp.sh
)"

s006_fused_job="$(
  sbatch --parsable --array=0-1%2 \
    --dependency="afterok:${s006_export_id}" \
    --job-name=coord250r-s006-crt6 \
    --output=logs/coord250r-s006-crt6_%A_%a.out \
    --error=logs/coord250r-s006-crt6_%A_%a.err \
    --export="ALL,REPO_ROOT=${REPO_ROOT},COORDINATE_ROOT=${S006_PARQUET},REFERENCE_ROOT=${REFERENCE_ROOT},SPLIT_MANIFEST=${SPLIT_MANIFEST},ARTIFACT_ROOT=${S006_FUSED_ARTIFACTS}" \
    slurm/launch_train_subject006_newton_coordinate_6ch_bp.sh
)"

s010_coord_job="$(
  sbatch --parsable --array=0-1%2 \
    --dependency="afterok:${s010_export_id}" \
    --job-name=coord250r-s010-crt3 \
    --output=logs/coord250r-s010-crt3_%A_%a.out \
    --error=logs/coord250r-s010-crt3_%A_%a.err \
    --export="ALL,REPO_ROOT=${REPO_ROOT},PARQUET_ROOT=${S010_PARQUET},SPLIT_MANIFEST=${SPLIT_MANIFEST},ARTIFACT_ROOT=${S010_COORD_ARTIFACTS}" \
    slurm/launch_train_subject010_coordinate_direct_bp.sh
)"

s010_fused_job="$(
  sbatch --parsable --array=0-1%2 \
    --dependency="afterok:${s010_export_id}" \
    --job-name=coord250r-s010-crt6 \
    --output=logs/coord250r-s010-crt6_%A_%a.out \
    --error=logs/coord250r-s010-crt6_%A_%a.err \
    --export="ALL,REPO_ROOT=${REPO_ROOT},COORDINATE_ROOT=${S010_PARQUET},REFERENCE_ROOT=${REFERENCE_ROOT},SPLIT_MANIFEST=${SPLIT_MANIFEST},ARTIFACT_ROOT=${S010_FUSED_ARTIFACTS}" \
    slurm/launch_train_subject010_newton_coordinate_6ch_bp.sh
)"

echo "subject006 250-channel export recovery: ${s006_export_job}"
echo "subject010 250-channel export recovery: ${s010_export_job}"
echo "subject006 coordinate 3ch CRT waveform and fiducials: ${s006_coord_job}"
echo "subject006 fused 6ch CRT waveform and fiducials: ${s006_fused_job}"
echo "subject010 coordinate 3ch CRT waveform and fiducials: ${s010_coord_job}"
echo "subject010 fused 6ch CRT waveform and fiducials: ${s010_fused_job}"
