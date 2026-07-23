# Coordinate GCNM 91-subject rollout runbook

This is the reproducible path for the accepted b045 coordinate-direct
experiment. Generated participant data, synthetic arrays, checkpoints, Parquet
shards, logs, and BP artifacts are intentionally not committed.

## Experiment contract

- 15 b045 rings: `US060` through `US130` in 5 mm increments.
- 91 subjects and 216 source HDF5 sessions from the main cohort.
- One ring-specific coordinate-direct GCNM checkpoint per mesh.
- 1,000 synthetic beats per mesh, with 50 samples per beat.
- Coordinate representation: `[S1, S2, dS2/dt]`.
- Combined representation: archived Newton `[PVI_HP, dPVI_LP/dt,
  d2PVI_LP/dt2]` plus coordinate `[S1, S2, dS2/dt]`.
- `mask05`, frozen subject-internal sample splits, and the original `pvi_ml`
  CRT training protocol.
- BP matrix: 91 subjects × 2 channel contracts × 2 targets = 364 runs.

The archived Newton channels for six-channel training are read directly from
the source HDF5 files. They are not copied into a second Parquet dataset.

## 1. Prepare the environment

Create the Python environment described in the repository `README.md`, then
create the private cluster file:

```bash
mkdir -p env
cp cluster.env.example env/cluster.env
```

Set at least `GCNM_PYTHON`, `PVI_SOLVER_ROOT`, and `PVI_DATA_ROOT` in
`env/cluster.env`. The sibling `pvi_ml` checkout must be available at
`../gia_bao/pvi_ml`, or the launch scripts must be adjusted to its location.

Activate the same environment for login-node checks:

```bash
source env/cluster.env
source "$(dirname "$(dirname "$GCNM_PYTHON")")/bin/activate"
export PYTHONPATH="$PWD:$PVI_SOLVER_ROOT:${PYTHONPATH:-}"
export MPLCONFIGDIR="${TMPDIR:-/tmp}/gcnm-mpl"
```

## 2. Build and validate cohort metadata

Generate portable ring configs and the subject/session registry:

```bash
python -m gcnm_pvi.ring_configs
python -m gcnm_pvi.mesh_registry \
  --data-root "$PVI_DATA_ROOT" \
  --output data/registries/main_b045_v1.json
```

The registry command fails if a session cannot be matched to a ring config,
forward mesh, inverse mesh, or 40×40 element-to-image mapping. Its accepted
main-cohort contract is 91 subjects, 216 sessions, and 15 rings.

Freeze sample splits. Pass the original checkpoint roots in preferred search
order when they are available; subjects without a recoverable split use the
deterministic seed-42 fallback:

```bash
python -m gcnm_pvi.pvi_splits \
  --registry data/registries/main_b045_v1.json \
  --checkpoint-root /path/to/original/pvi_ml/artifacts \
  --output data/manifests/pvi_subject_splits_mask05_v1.json \
  --test-size 0.1 \
  --seed 42
```

Create and inspect the ring, subject, and 364-task manifests:

```bash
python -m gcnm_pvi.full_coordinate_rollout build-manifests \
  --registry data/registries/main_b045_v1.json \
  --output-root data/manifests

test "$(($(wc -l < data/manifests/coordinate_main_b045_bp_364_v1.tsv) - 1))" \
  -eq 364
```

## 3. Verify the code before submission

```bash
python -m compileall -q gcnm_pvi scripts
bash -n scripts/*.sh slurm/*.sh
python -m pytest -q
```

The current verified suite contains 122 tests.

## 4. Submit a fresh immutable rollout

The fresh submission is fail-closed: it refuses to overwrite an existing
coordinate Parquet root or BP artifact root.

```bash
bash scripts/submit_coordinate_main_b045_91_subject_rollout.sh
```

The dependency chain is:

1. Generate 15 ring-specific synthetic packs.
2. Train the ring-specific coordinate-direct GCNMs in a packed four-GPU job.
3. Generate synthetic truth/Newton/S1/S2 GIFs.
4. Export one coordinate Parquet part per ring.
5. Merge and validate all coordinate parts against source HDF5 and frozen
   splits.
6. Train the 364 CRT BP experiments using 16 workers across four GPUs.
7. Generate prediction-aligned artifact GIFs.

The production launchers request up to four GPUs, 64 CPUs, 1,000 GB of memory,
and a 10-day wall limit where the full workload needs it. Packed BP training
runs four independent CRT processes per GPU. The Parquet exporter batches
inference, deduplicates repeated source frames, and disables optional stage-2
residual diagnostics during export.

## 5. Resume BP training safely

If the coordinate dataset passed final validation but the BP packed job was
interrupted, use:

```bash
bash scripts/submit_resume_coordinate_main_b045_bp.sh
```

This first runs a CPU checkpoint-postprocessing smoke test. The packed runner
then skips artifact directories that already satisfy the completion contract
and reruns only missing or failed experiments. A dependent GIF array renders
the completed models afterward.

To launch the BP matrix directly after an independently validated export:

```bash
bash scripts/submit_coordinate_main_b045_hdf5_bp.sh
```

## 6. Manual preflight

Before any standalone BP submission, validate the coordinate rows, the HDF5
Newton pairing, sample identities, and frozen split membership:

```bash
python -m gcnm_pvi.validate_coordinate_direct_parquet \
  --root gcnm_parquet/coordinate_direct_main_b045_v1 \
  --split-manifest data/manifests/pvi_subject_splits_mask05_v1.json

python -m gcnm_pvi.full_coordinate_rollout preflight-hdf5 \
  --coordinate-root gcnm_parquet/coordinate_direct_main_b045_v1 \
  --registry data/registries/main_b045_v1.json \
  --split-manifest data/manifests/pvi_subject_splits_mask05_v1.json
```

The preflight rejects sample-order differences, missing sessions, split
leakage, incompatible tensor shapes, or a mismatch between coordinate rows and
the source HDF5 windows.

## 7. Outputs

The untracked production roots are:

```text
data/hp_lp_beats_main_b045_v1/                 synthetic ring packs
data/differential_main_b045_1000beats_clean_v1/
models/differential_main_b045_1000beats_v1/    GCNM checkpoints
gcnm_parquet/coordinate_direct_main_b045_v1/   coordinate S1/S2 rows
artifacts/coordinate_main_b045_bp_v1/          pvi_ml-compatible BP artifacts
logs/                                           Slurm and per-task logs
reports/coordinate_main_b045_1000beats_v1/     synthetic GIF reports
```

Each BP model uses the standard `pvi_ml` artifact structure and adds a `gifs/`
directory. The GIF panels align the model prediction with BP truth, archived
PVI/Newton frames from HDF5, coordinate S1/S2, and the temporal derivative.

Do not delete an output root while any dependent Slurm job is queued or
running. Do not reuse a partially written immutable root: retain it for
diagnosis, move it to an explicitly named archive if needed, and submit to a
new versioned path.
