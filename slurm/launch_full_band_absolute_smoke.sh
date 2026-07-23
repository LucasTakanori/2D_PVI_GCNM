#!/bin/bash
# One small CPU-only contract test for absolute V -> absolute conductivity.
# This is not the production synthetic cohort and does not request a GPU.
#SBATCH --job-name=full-abs-smoke
#SBATCH --output=logs/full-abs-smoke_%j.out
#SBATCH --error=logs/full-abs-smoke_%j.err
#SBATCH --partition=ece_bst
#SBATCH --account=ece_bst
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=1000GB
#SBATCH --time=10-00:00:00

set -euo pipefail
module load gcc/11.2.0

REPO_ROOT="${REPO_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"
cd "${REPO_ROOT}"
source "${REPO_ROOT}/env/cluster.env"
source "$(dirname "$(dirname "${GCNM_PYTHON}")")/bin/activate"
export PYTHONPATH="${REPO_ROOT}:${PVI_SOLVER_ROOT}:${PYTHONPATH:-}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export NUMEXPR_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export MPLCONFIGDIR="${TMPDIR:-/tmp}/matplotlib-gcnm"

SMOKE_ROOT="${SMOKE_ROOT:-${REPO_ROOT}/data/full_band_beats_US120_v2_smoke}"
if [[ -e "${SMOKE_ROOT}" ]]; then
  echo "ERROR: immutable smoke root already exists: ${SMOKE_ROOT}" >&2
  exit 2
fi

python -u -m gcnm_pvi.generate_hp_lp_beat_dataset \
  --config "${REPO_ROOT}/configs/rings_b045/US120.yaml" \
  --output-root "${SMOKE_ROOT}" \
  --train-anatomies 1 --validation-anatomies 1 --test-anatomies 1 \
  --train-mode linearized --validation-mode linearized --test-mode linearized \
  --linearization-reference anatomy --jacobian-bank-size 0 --seed 20260721

python -u -m gcnm_pvi.validate_hp_lp_beat_dataset \
  --root "${SMOKE_ROOT}" --allow-unverified-exact \
  --output "${SMOKE_ROOT}/validation.json"

python - "${SMOKE_ROOT}" <<'PY'
import json
import sys
from pathlib import Path

import numpy as np

root = Path(sys.argv[1])
with np.load(root / "full" / "test.npz") as full:
    report = {
        "training_pair": "V_absolute_filtered -> sigma_absolute",
        "frames": int(len(full["sigma"])),
        "sigma_is_absolute": bool(np.array_equal(full["sigma"], full["sigma_absolute"])),
        "voltage_is_absolute": bool(np.array_equal(full["V"], full["V_absolute_filtered"])),
        "absolute_identity_max_error": float(np.max(np.abs(
            full["sigma_reference"] + full["sigma_delta_reference"] - full["sigma"]
        ))),
    }
(root / "absolute_contract_smoke.json").write_text(
    json.dumps(report, indent=2) + "\n", encoding="utf-8"
)
print(json.dumps(report, indent=2))
PY
