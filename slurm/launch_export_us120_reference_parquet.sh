#!/bin/bash
# Two CPU-only exports: archived Newton images and BioZ impedance.
#SBATCH --job-name=us120-ref-parquet
#SBATCH --output=logs/us120-ref-parquet_%A_%a.out
#SBATCH --error=logs/us120-ref-parquet_%A_%a.err
#SBATCH --partition=ece_bst
#SBATCH --account=ece_bst
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=500GB
#SBATCH --time=10-00:00:00

set -euo pipefail
module load gcc/11.2.0
REPO_ROOT="${REPO_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"
cd "${REPO_ROOT}"
source "${REPO_ROOT}/env/cluster.env"
source "$(dirname "$(dirname "${GCNM_PYTHON}")")/bin/activate"
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export ARROW_NUM_THREADS="${SLURM_CPUS_PER_TASK}"

TASK_ID="${SLURM_ARRAY_TASK_ID:?submit as array 0-1%2}"
MODES=(img bioz)
ROOT_NAMES=(us120_pilot_reference_image_v1 us120_pilot_reference_bioz_v1)
SHARD_ROWS=(32 2048)
MODE="${MODES[${TASK_ID}]}"
OUTPUT_ROOT="${REPO_ROOT}/gcnm_parquet/${ROOT_NAMES[${TASK_ID}]}"
HASH_MANIFEST="${REPO_ROOT}/gcnm_parquet/us120_pilot_coordinate_hp_lp_v1/manifest.json"

echo "input_mode=${MODE} output_root=${OUTPUT_ROOT} cpus=${SLURM_CPUS_PER_TASK}"
python -u -m gcnm_pvi.reference_parquet \
  --registry "${REPO_ROOT}/data/registries/main_b045_v1.json" \
  --output-root "${OUTPUT_ROOT}" \
  --subject subject006 --subject subject010 \
  --input-mode "${MODE}" --shard-rows "${SHARD_ROWS[${TASK_ID}]}" \
  --source-hash-manifest "${HASH_MANIFEST}"

python - "${OUTPUT_ROOT}" <<'PY'
import json, sys
from pathlib import Path
import pyarrow.dataset as ds
root = Path(sys.argv[1])
manifest = json.loads((root / "manifest.json").read_text())
rows = ds.dataset(str(root / "shards"), format="parquet").count_rows()
if rows != 3382 or manifest["row_count"] != 3382:
    raise SystemExit(f"expected 3382 rows, found parquet={rows}, manifest={manifest['row_count']}")
if set(manifest["subjects"]) != {"subject006", "subject010"}:
    raise SystemExit(f"unexpected subjects: {manifest['subjects']}")
print(json.dumps({"validated_root": str(root), "rows": rows, "input_mode": manifest["input_mode"]}))
PY
