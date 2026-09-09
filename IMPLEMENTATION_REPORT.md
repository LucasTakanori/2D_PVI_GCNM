# GCNM-PVI Implementation Report

**Date:** 2026-07-12
**Location:** repository root (`2D_PVI_GCNM/`)
**Status:** Subject006/US120 production parity, trial-disjoint data export, and a held-out GCNM engineering smoke run are complete.

> Current authoritative summary: `SUBJECT006_STATUS.md`. Older milestone sections
> below are retained as implementation history and may describe work that was
> pending at the time.
>
> Repository split note (2026-09-08): real-session ingestion, validation, and
> BP-facing export commands now live in the sibling `pvi_gcnm_bp_pipeline`
> repository under the `pvi_gcnm_bp` namespace. GCNM physics, synthetic data,
> training, evaluation, and runtime remain here.

## Current production-data status — 2026-07-12

The raw-data state is maintained in `pvi_gcnm_bp_pipeline/RAW_DATA_PROGRESS.md`;
this section supersedes older references below to raw data being on another
machine or all output being 32×32.

| Area | Current state |
|------|---------------|
| Raw data | Available at `common/data/pvi_data/raw/`; all 11 subject006 baseline trials were processed. |
| Production preprocessing | Exact sampled MATLAB-cache parity for remap, filtering, current sign, resistance/reactance, and movmean. |
| 40×40 output | Real `b045/US120` ring mesh exported for subject006 with 40×40 `m2i`, Laplacian, and `c2f`. |
| HDF5 validation | 500 frame-matched images: correlation 0.9936, mask IoU 1.0. |
| Period alignment | 1,298 HDF periods matched across 11 trials; LP channel correlation >0.9998. |
| Dataset and GCNM | 9,710 trial-disjoint samples exported; train/checkpoint/held-out inference smoke test passed. |

The mesh entrypoint remains `scripts/export_ring_mesh.sh` here. Real-session
entrypoints are `scripts/export_production.sh` and `scripts/validate_session.sh`
in `pvi_gcnm_bp_pipeline`, using the core
`configs/finger_pvi08_production.yaml` (`meas_pattern: [0, 2]`,
`hyper_pvi: 5e-4`). Run order, when authorized, is:

```bash
bash scripts/setup_env.sh
export GCNM_CONFIG=configs/subject006_pvi08_production.yaml
bash scripts/export_ring_mesh.sh US120 data/ring_meshes/subject006_US120
cd ../pvi_gcnm_bp_pipeline
bash scripts/export_production.sh "<bioz_dir>" "<out_dir>"
bash scripts/validate_session.sh "<out_dir>" "<masked.h5>"
```

---

## 1. Objective

Adapt the 2D GCNM prototype to the PVI 8-electrode EIT stack so that:

1. Forward physics, Jacobian, and mesh format come from PVI (not PyEIT).
2. Training produces iterative LM + GCN conductivity reconstructions on the FEM element graph.
3. A path exists from raw ScioSpec `.eit` → `vmeas` → Newton pseudo-labels → GCNM training/inference → **40×40 images** comparable to HDF5 `pviHP/img`.

---

## 2. What was built

### 2.1 Python packages after the repository split

| Package | Purpose |
|------|---------|
| `gcnm_pvi` (this repository) | FEM physics, meshes, graphs, synthetic generators, GCNM models, training, evaluation, and reconstruction runtime |
| `pvi_gcnm_bp` (sibling repository) | ScioSpec/HDF ingestion, participant splits, representation/Parquet export, CRT/SAMBA training, BP inference, and artifact compatibility |

The original module-by-module inventory below this point is historical. See
`README.md` and `REPOSITORY_BOUNDARY.md` for the current package contract.

### 2.2 Configuration

| File | Mode | Use case |
|------|------|----------|
| `configs/finger_pvi08.yaml` | `absolute`, `perturbation` phantoms | Synthetic training on tank mesh |
| `configs/finger_pvi08_differential.yaml` | `differential` | Real ScioSpec sessions |

Mesh paths target **`pvi_solver/_data/_mesh08_r64/`** (corrected from non-existent `_mesh64_pvi08` in original zip):

- Forward: `tank2d_mdl_64_refined.h5`
- Inverse: `tank2d_mdl_64.h5`
- Mappings: `tank2d_mdl_64_mappings.h5` → **32×32** image from **432** elements

### 2.3 Shell scripts (required entrypoints)

| Script | Action |
|--------|--------|
| `scripts/_common.sh` | Sources `env/cluster.env`, sets `PYTHONPATH`, `GCNM_PYTHON` |
| `scripts/setup_env.sh` | `pip install -r requirements.txt` |
| `scripts/smoke_test.sh` | `python -m gcnm_pvi.smoke_test` |
| `scripts/train_gcnm.sh` | Training (`GCNM_CONFIG`, `GCNM_SAMPLES_DIR` optional) |
| `scripts/test_gcnm.sh` | Held-out evaluation |
| `pvi_gcnm_bp_pipeline/scripts/export_dataset.sh` | ScioSpec → NPZ dataset |
| `pvi_gcnm_bp_pipeline/scripts/infer_gcnm.sh` | Trained models → images |
| `pvi_gcnm_bp_pipeline/scripts/compare_hdf5.sh` | Compare `img.npy` vs HDF5 |

### 2.4 Supporting files

- `requirements.txt` — numpy&lt;2, scipy, h5py, torch, torch_geometric, pyyaml, matplotlib
- `env/cluster.env` — cluster paths + default venv
- `README.md` — updated usage (scripts-first)
- `PVI_GCNM_ADAPTATION_PLAN.md` — prior analysis (unchanged)

---

## 3. Physics and protocol (verified in prior session)

| Item | Value |
|------|-------|
| Electrodes | 8 |
| Stim pattern | skip-3 `(0, 3)` |
| Meas pattern | adjacent `(0, 1)` |
| Channels | **32** (4 meas × 8 stim) |
| Jacobian rank (tiny mesh) | **20** independent measurements |
| LM normalization | divide J and residual by \|V_hom\| |

### Differential imaging (production parity)

`PviDifferentialPhysics` implements:

1. `calibrate(vmeas_ref)` — `sigma_alpha` scaling (matches `pvi_inv_calibrate`)
2. `differential_residual(vmeas_t)` — ΔV relative to reference frame
3. `lm_update_differential()` — fixed Jacobian at init σ + Laplacian `hyper_pvi`
4. `newton_step_production()` — one-step Newton for pseudo-labels

---

## 4. Data flow

```text
Raw ScioSpec/HDF (pvi_gcnm_bp)
        -> validated voltage/sample packs
        -> GCNM training/runtime (gcnm_pvi)
        -> ring-specific S1/S2 reconstructions
        -> Parquet + CRT/SAMBA + BP artifacts (pvi_gcnm_bp)
```

The sibling BP repository owns the real-session export, comparison, and
inference wrappers. This core repository owns model training and reconstruction
mathematics.

---

## 5. Changes from original `gcnm_pvi.zip`

| Gap in zip | Resolution |
|------------|------------|
| Wrong mesh path `_mesh64_pvi08` | → `_mesh08_r64` via `paths.py` + YAML |
| No differential / calibrate | → `PviDifferentialPhysics` |
| No Laplacian R in LM | → `MeshMappings.rtr()`, `hyper_pvi` in config |
| No m2i / image output | → `gcnm_mesh_maps`, `gcnm_image`, export/infer |
| No `.eit` reader | → `pvi_gcnm_bp.sciospec_reader` in the sibling repository |
| Phantom-only σ range | → `perturbation` phantom mode |
| Monolithic scripts | → package + YAML config + shell scripts |
| numpy 2 breakage | → `pvi_compat.py` |

---

## 6. Verification state

| Check | State |
|-------|-------|
| Package imports | Written; smoke test covers full chain |
| Tiny mesh forward + Jacobian rank=20 | Passed in prior debug run (~0.2 s/Jacobian after import) |
| Full smoke test (`SMOKE TEST PASSED`) | **Pending** — run `bash scripts/smoke_test.sh` (slow on login node due to torch import + 12 LM Jacobians + 2 GCN iterations) |
| Real mesh training | **Pending** — run `bash scripts/train_gcnm.sh` on compute node |
| ScioSpec `.eit` → vmeas | **Pending** — needs raw files from other machine; validate against MATLAB `SCIOSPEC.read` |
| HDF5 image MSE | **Pending** — after export/infer on matched session |

Prior partial smoke output (before user interrupted):

```
[1/6] protocol OK: 32 channels
[2/6] forward+Jacobian OK, rank(J)=20
[3/6] LM update OK
[4/6] differential calibrate OK
[5/6] learning ... (in progress)
```

---

## 7. How to run (summary)

```bash
cd /path/to/2D_PVI_GCNM
bash scripts/setup_env.sh      # once
bash scripts/smoke_test.sh     # verify
bash scripts/train_gcnm.sh     # synthetic
```

For ScioSpec ingestion and inference, switch to the BP pipeline repository:

```bash
cd ../pvi_gcnm_bp_pipeline
bash scripts/export_dataset.sh /path/to/session /path/to/out
cd ../2D_GCNM
export GCNM_CONFIG=configs/finger_pvi08_differential.yaml
export GCNM_SAMPLES_DIR=/path/to/out/npz
bash scripts/train_gcnm.sh
cd ../pvi_gcnm_bp_pipeline
bash scripts/infer_gcnm.sh /path/to/session /path/to/infer_out
bash scripts/compare_hdf5.sh /path/to/infer_out/img.npy /path/to/session.h5
```

Recommend **compute node** (`salloc`) for training and full mesh Jacobian work.

---

## 8. Open items (not blocking scaffolding)

1. **Finger/ring mesh HDF5** — production geometry; current tank `_mesh08_r64` is structurally correct for 8-el protocol but not finger anatomy.
2. **ScioSpec channel mapping validation** — maintained in `pvi_gcnm_bp`; confirm against hardware `MeasurementChannels` / remap tables.
3. **Exact scionova preprocessing** — maintained and validated in the sibling BP pipeline.
4. **Jacobian speed** — `pvi_forward.compute_jacobian` is Python loop; vectorization needed for large-scale training on login nodes.
5. **Real-session inference** — batching and online scheduling are owned by the sibling BP pipeline and consume the public core runtime.
6. **Differential calibrate per session vs per frame** — current train script calibrates once on frame 0; time-series may need session-level reference frame selection matching `scionova_03_segmentation.m`.

---

## 9. Related PVI reference files

| Path | Role |
|------|------|
| `pvi_solver/pvi_forward.py` | FEM forward + Jacobian |
| `pvi_solver/pvi_inverse.py` | Production inverse (MATLAB port) |
| `experiments/scionova_01_processing.m` | Raw → processed |
| `experiments/scionova_03_segmentation.m` | `pvi_inv_make` + `pvi_inv_solve` |
| `external_packages/SCIOSPEC.m` | Reference `.eit` reader |
| `common/data/pvi_data/main/` | ~216 masked HDF5 sessions |

---

## 10. File inventory (new/changed in this implementation)

```text
2D_GCNM/
├── gcnm_pvi/                  GCNM physics, models, training, evaluation, runtime
├── configs/                   GCNM and ring configurations
├── scripts/                   core generation/training/evaluation entrypoints
├── slurm/                     core cluster launchers
├── tests/                     numerical and boundary regressions
├── README.md
├── REPOSITORY_BOUNDARY.md
└── IMPLEMENTATION_REPORT.md   this historical implementation record
```

Real-data ingestion, Parquet, BP training, and artifact workflows are inventoried
in the sibling `pvi_gcnm_bp_pipeline` repository.

---

## 11. Subject006 verified checkpoint (2026-07-12)

The earlier pending/raw-data notes above are superseded by the completed subject006
workflow documented in `SUBJECT006_STATUS.md`.

- Confirmed ring size: US120, 1,330 inverse elements.
- Exact raw MATLAB-cache parity established for sampled frames and channel remap.
- Corrected production inverse to unweighted `J`, `lambda² * Rrec`, and differential
  first-frame output.
- Corrected the 40×40 mapping from C-order to MATLAB column-major pixel ordering.
- Direct 500-frame HDF image correlation: 0.9936; mask IoU: 1.0.
- Recovered 1,298-period order from 1,310 automatic candidates across 11 trials.
- Built 9,710 clean samples with disjoint trial splits.
- Completed a one-epoch graph training/checkpoint/inference smoke test on held-out
  trials 10 and 11 (image correlation 0.9828 to PVI pseudo-labels).

The main unresolved item is now scientific supervision, not data plumbing: PVI
pseudo-labels can validate or distill the production inverse but cannot demonstrate
that GCNM is better than that inverse.

---

## 12. Faithful nonlinear PVI-GCNM completion (2026-07-12)

The earlier sections describe the original scaffolding and contain historical
pending items. The current production research path now additionally includes:

- `iterative_physics.py`: per-stage differential forward residual, Jacobian, LM
  solve, positivity projection, voltage residual, and clamp diagnostics;
- `train_faithful_gcnm.py`: direct and proposal-residual greedy training,
  homogeneous or oracle baseline contracts, coordinate/loss ablations, composite
  checkpointing, reproducible seed, and artifact hashes;
- `evaluate_faithful_gcnm.py`: saved Newton, matched iterative-LM-only control,
  faithful GCN stages, exact nonlinear voltage consistency, synthetic clean truth,
  and real-PVI pseudo-reference modes;
- fixed-resource `.sh`/`sbatch` launchers preserving one GPU, 16 CPUs, 250 GB,
  one node/task, `ece_bst`, and a one-day time limit; and
- five passing unit tests plus completed end-to-end Slurm smoke jobs.

All deployable experiments use a homogeneous 0.7 S/m baseline in both training and
inference. Saved synthetic anatomy is labeled as an oracle ablation because it
contains unavailable static vessel information.

On the 32-sample exact nonlinear fine-mesh holdout, the selected
`faithful_background025_hom_seed0` stage-2 model achieved image correlation 0.5574,
mean per-sample correlation 0.5667, image RMSE 0.004324, localization Dice 0.4853,
background RMS 0.001276, and post-stage voltage residual RMS 3.18e-6. The matched
iterative LM stage 2 achieved correlation 0.2908, RMSE 0.005033, Dice 0.3939, and
background RMS 0.000589. This is a Pareto result, not proof of human anatomical
recovery.

The full subject-6 real evaluation uses 1,600 phases from held-out trials 10--11
and applies voltage sign -1 to convert the saved MATLAB display convention back to
the physical inversion convention. Its target is explicitly a production Newton
pseudo-reference. Selected faithful stage 1 obtained 0.6039 global pseudo-reference
correlation and reduced mean voltage residual from 2.36e-5 to 1.68e-5. Stage 2
dropped to 0.1694 correlation and increased voltage residual to 2.63e-5, exposing
synthetic-to-real distribution shift. Correlation with that target measures inverse
parity/domain shift, not anatomical correctness; independent phantom or ultrasound
truth remains the principal scientific next step.
