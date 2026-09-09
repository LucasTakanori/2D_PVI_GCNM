#!/usr/bin/env bash
# Train a 250-wide coordinate GCNM, export subject006/010, and run all
# coordinate-only 3ch and Newton+coordinate 6ch CRT target combinations.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"
mkdir -p logs

CONFIG="${REPO_ROOT}/configs/rings_b045/US120_250ch.yaml"
REGISTRY="${REPO_ROOT}/data/registries/us120_subject006_subject010_250ch_v1.json"
DATA_ROOT="${REPO_ROOT}/data/differential_US120_1000beats_clean_v1"
MODEL_ROOT="${REPO_ROOT}/models/differential_US120_1000beats_v1/coordinate_direct_250ch"
RESULT_ROOT="${REPO_ROOT}/data/differential_US120_1000beats_results_v1/coordinate_direct_250ch"
SPLIT_MANIFEST="${REPO_ROOT}/data/splits/us120_pilot_subject006_subject010_mask05_v1.json"
REFERENCE_ROOT="${REPO_ROOT}/gcnm_parquet/us120_pilot_reference_image_v1"

S006_PARQUET="${REPO_ROOT}/gcnm_parquet/subject006_coordinate_direct_250ch_s1_s2_ds2_v1"
S010_PARQUET="${REPO_ROOT}/gcnm_parquet/subject010_coordinate_direct_250ch_s1_s2_ds2_v1"
S006_COORD_ARTIFACTS="${REPO_ROOT}/artifacts/subject006_coordinate_direct_250ch_bp_v1"
S006_FUSED_ARTIFACTS="${REPO_ROOT}/artifacts/subject006_newton_coordinate_250ch_6ch_bp_v1"
S010_COORD_ARTIFACTS="${REPO_ROOT}/artifacts/subject010_coordinate_direct_250ch_bp_v1"
S010_FUSED_ARTIFACTS="${REPO_ROOT}/artifacts/subject010_newton_coordinate_250ch_6ch_bp_v1"

for required in \
  "${CONFIG}" \
  "${REGISTRY}" \
  "${DATA_ROOT}/validation.json" \
  "${DATA_ROOT}/train.npz" \
  "${DATA_ROOT}/validation.npz" \
  "${SPLIT_MANIFEST}" \
  "${REFERENCE_ROOT}/manifest.json"; do
  [[ -e "${required}" ]] || {
    echo "required input is missing: ${required}" >&2
    exit 2
  }
done

for output in \
  "${MODEL_ROOT}" \
  "${RESULT_ROOT}" \
  "${S006_PARQUET}" \
  "${S010_PARQUET}" \
  "${S006_COORD_ARTIFACTS}" \
  "${S006_FUSED_ARTIFACTS}" \
  "${S010_COORD_ARTIFACTS}" \
  "${S010_FUSED_ARTIFACTS}"; do
  [[ ! -e "${output}" ]] || {
    echo "refusing to overwrite immutable output: ${output}" >&2
    exit 3
  }
done

if [[ "${DRY_RUN:-0}" == 1 ]]; then
  echo "preflight passed for the US120 250-channel GCNM-to-CRT ablation"
  echo "planned outputs:"
  printf '  %s\n' \
    "${MODEL_ROOT}" "${RESULT_ROOT}" "${S006_PARQUET}" "${S010_PARQUET}" \
    "${S006_COORD_ARTIFACTS}" "${S006_FUSED_ARTIFACTS}" \
    "${S010_COORD_ARTIFACTS}" "${S010_FUSED_ARTIFACTS}"
  exit 0
fi

train_job="$(
  sbatch --parsable \
    --job-name=coord250-US120 \
    --output=logs/coord250-US120_%j.out \
    --error=logs/coord250-US120_%j.err \
    --export="ALL,REPO_ROOT=${REPO_ROOT},CONFIG=${CONFIG},DATA_ROOT=${DATA_ROOT},MODEL_ROOT=${MODEL_ROOT},RESULT_ROOT=${RESULT_ROOT}" \
    slurm/launch_train_coordinate_direct_US120_1000beats.sh
)"
train_id="${train_job%%;*}"

s006_export_job="$(
  sbatch --parsable \
    --dependency="afterok:${train_id}" \
    --job-name=coord250-s006-export \
    --output=logs/coord250-s006-export_%j.out \
    --error=logs/coord250-s006-export_%j.err \
    --export="ALL,REPO_ROOT=${REPO_ROOT},MODEL_ROOT=${MODEL_ROOT},OUTPUT_ROOT=${S006_PARQUET},SPLIT_MANIFEST=${SPLIT_MANIFEST},REGISTRY=${REGISTRY}" \
    slurm/launch_export_subject006_coordinate_direct_parquet.sh
)"
s006_export_id="${s006_export_job%%;*}"

s010_export_job="$(
  sbatch --parsable \
    --dependency="afterok:${train_id}" \
    --job-name=coord250-s010-export \
    --output=logs/coord250-s010-export_%j.out \
    --error=logs/coord250-s010-export_%j.err \
    --export="ALL,REPO_ROOT=${REPO_ROOT},MODEL_ROOT=${MODEL_ROOT},OUTPUT_ROOT=${S010_PARQUET},SPLIT_MANIFEST=${SPLIT_MANIFEST},REGISTRY=${REGISTRY}" \
    slurm/launch_export_subject010_coordinate_direct_parquet.sh
)"
s010_export_id="${s010_export_job%%;*}"

s006_coord_job="$(
  sbatch --parsable --array=0-1%2 \
    --dependency="afterok:${s006_export_id}" \
    --job-name=coord250-s006-crt3 \
    --output=logs/coord250-s006-crt3_%A_%a.out \
    --error=logs/coord250-s006-crt3_%A_%a.err \
    --export="ALL,REPO_ROOT=${REPO_ROOT},PARQUET_ROOT=${S006_PARQUET},SPLIT_MANIFEST=${SPLIT_MANIFEST},ARTIFACT_ROOT=${S006_COORD_ARTIFACTS}" \
    slurm/launch_train_subject006_coordinate_direct_bp.sh
)"

s006_fused_job="$(
  sbatch --parsable --array=0-1%2 \
    --dependency="afterok:${s006_export_id}" \
    --job-name=coord250-s006-crt6 \
    --output=logs/coord250-s006-crt6_%A_%a.out \
    --error=logs/coord250-s006-crt6_%A_%a.err \
    --export="ALL,REPO_ROOT=${REPO_ROOT},COORDINATE_ROOT=${S006_PARQUET},REFERENCE_ROOT=${REFERENCE_ROOT},SPLIT_MANIFEST=${SPLIT_MANIFEST},ARTIFACT_ROOT=${S006_FUSED_ARTIFACTS}" \
    slurm/launch_train_subject006_newton_coordinate_6ch_bp.sh
)"

s010_coord_job="$(
  sbatch --parsable --array=0-1%2 \
    --dependency="afterok:${s010_export_id}" \
    --job-name=coord250-s010-crt3 \
    --output=logs/coord250-s010-crt3_%A_%a.out \
    --error=logs/coord250-s010-crt3_%A_%a.err \
    --export="ALL,REPO_ROOT=${REPO_ROOT},PARQUET_ROOT=${S010_PARQUET},SPLIT_MANIFEST=${SPLIT_MANIFEST},ARTIFACT_ROOT=${S010_COORD_ARTIFACTS}" \
    slurm/launch_train_subject010_coordinate_direct_bp.sh
)"

s010_fused_job="$(
  sbatch --parsable --array=0-1%2 \
    --dependency="afterok:${s010_export_id}" \
    --job-name=coord250-s010-crt6 \
    --output=logs/coord250-s010-crt6_%A_%a.out \
    --error=logs/coord250-s010-crt6_%A_%a.err \
    --export="ALL,REPO_ROOT=${REPO_ROOT},COORDINATE_ROOT=${S010_PARQUET},REFERENCE_ROOT=${REFERENCE_ROOT},SPLIT_MANIFEST=${SPLIT_MANIFEST},ARTIFACT_ROOT=${S010_FUSED_ARTIFACTS}" \
    slurm/launch_train_subject010_newton_coordinate_6ch_bp.sh
)"

echo "250-channel GCNM training: ${train_job}"
echo "subject006 export: ${s006_export_job}"
echo "subject010 export: ${s010_export_job}"
echo "subject006 coordinate 3ch CRT waveform/fiducials: ${s006_coord_job}"
echo "subject006 fused 6ch CRT waveform/fiducials: ${s006_fused_job}"
echo "subject010 coordinate 3ch CRT waveform/fiducials: ${s010_coord_job}"
echo "subject010 fused 6ch CRT waveform/fiducials: ${s010_fused_job}"
