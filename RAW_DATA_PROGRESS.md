# Raw ScioSpec Data — Progress Log

**Last updated:** 2026-07-12  
**Status:** Subject006 US120 production parity and trial-disjoint export are complete; exact archived manual peak merges remain unavailable.

> Current authoritative summary: `SUBJECT006_STATUS.md`. This file also retains
> older intermediate observations for provenance.

---

## Recommended work plan (your directions)

| Step | Task | Status |
|------|------|--------|
| 1 | Export subject006 with exact production preprocessing | **Complete**; corrected 500-frame pilot contains 450 post-trim frames |
| 2 | Construct and verify 40×40 US120 mapping | **Complete**; MATLAB pixel ordering and mask IoU 1.0 |
| 3 | Validate inverse vs frame-matched HDF5 | **Complete**; image correlation 0.9936 over 500 frames |
| 4 | Create trial-disjoint packs and exercise GCNM | **Complete as engineering smoke test**; scientific supervision remains |

---

## Raw data location

```
/home/lsanc68/ece_bst_link/common/data/pvi_data/raw/
├── subject001 … subject008/
│   └── baseline/<timestamp>/bioz/*.eit + bioz.setUp
```

**HDF5 reference:**
```
/home/lsanc68/ece_bst_link/common/data/pvi_data/main/subject006_baseline_masked.h5
  data/pviHP/img        (40, 40, 64900)
  data/pviHP/resistance (32, 64900)
```

**Important:** HDF5 time axis = cardiac-period interpolation (867 periods × 50 pts). Export uses **raw frame index** (~6000 frames @ 50 Hz per bioz trial). Exact frame-by-frame match requires scionova_02 alignment + segmentation (not yet ported).

---

## Production preprocessing (now implemented)

Mirrors **scionova_01** + **scionova_03** PVI path:

| Step | MATLAB | Python module |
|------|--------|---------------|
| `make_eit` + complex vmeas | `SCIOSPEC.make_eit` | `sciospec_reader.session_to_vmeas` |
| idx0 trim after movmean detrend | `SIGNALOPERATIONS.doFFT` (window ≈ fs) | `production_preprocess.trim_idx0` |
| 5 Hz butter lowpass (real + imag) | `scionova_01` | `production_preprocess.butter_lowpass_complex` |
| 1-step Newton on `real(vmeas)` | `pvi_inv_make/solve`, λ=5e-4 | `gcnm_differential.reconstruct_newton_series` |
| `meas_pattern = [0, 2]` | `PIPELINECONFIGS` strict | `configs/finger_pvi08_production.yaml` |
| HP/LP movmean window=100 | `scionova_03` | `production_preprocess.hp_lp_movmean` |
| R/X = -real/imag / I | `scionova_03` | `production_preprocess.resistance_reactance` |

**Config:** `configs/finger_pvi08_production.yaml`

Subject-specific config: `configs/subject006_pvi08_production.yaml`.
Complex ScioSpec extraction is now preserved through preprocessing, including
nonzero reactance output.

---

## 40×40 image mapping

| Source | m2i shape | Image | Notes |
|--------|-----------|-------|-------|
| Tank `_mesh08_r64` (32×32) | (1024, 432) | 32×32 | Old pilot export |
| **Interim 40×40 builder** | (1600, 432) | 40×40 | `fem2d_mesh2img.py` + `build_m2i_40.sh` |
| **Production ring mesh** | (1600, 1330) | 40×40 | Confirmed subject006 `b045/US120` export |

The PVI repository contains `mesh_collection_b035.mat` and `mesh_collection_b045.mat`, each with `US060`–`US150`. Subject006's historical HDF5 has 1,364 finite pixels, an exact zero-XOR match unique to `US120` in both collections. The subject006 manifest records this evidence. Resistance/vmeas metrics remain unaligned until the cardiac segmentation timeline is reproduced.

---

## New files (this update)

| File | Purpose |
|------|---------|
| `gcnm_pvi/production_preprocess.py` | scionova-like vmeas + HP/LP |
| `gcnm_pvi/fem2d_mesh2img.py` | Build m2i @ N×N (includes a no-Shapely fallback) |
| `gcnm_pvi/ring_mesh_collection.py` | Export real MATLAB ring meshes and mappings to PVI HDF5 |
| `gcnm_pvi/build_mappings.py` | Write `*_mappings_40.h5` |
| `gcnm_pvi/export_production_session.py` | Full production export CLI |
| `gcnm_pvi/validate_session.py` | Grid-matched HDF5 validation + JSON report |
| `configs/finger_pvi08_production.yaml` | Generic production params |
| `configs/subject006_pvi08_production.yaml` | Confirmed real-ring subject006 params |
| `scripts/export_ring_mesh.sh` | Export one `US###` ring entry |
| `scripts/build_m2i_40.sh` | One-time mapping build |
| `scripts/export_production.sh` | Full/longer session export |
| `scripts/validate_session.sh` | Compare export vs HDF5 |

**Earlier fixes still in place:** `.eit` 8-stim parser, mesh impedance normalize, `sciospec_reader` frame limits.

---

## Commands to run (in order)

### 1. Export a 40×40 real ring mesh
```bash
bash scripts/export_ring_mesh.sh US120 data/ring_meshes/subject006_US120
export GCNM_CONFIG=configs/subject006_pvi08_production.yaml
```

### 2. Export full subject006 baseline trial (6000 frames — use compute node)
```bash
bash scripts/export_production.sh \
  "/home/lsanc68/ece_bst_link/common/data/pvi_data/raw/subject006/baseline/20241217 17.16.06/bioz" \
  "/home/lsanc68/ece_bst_link/common/data/pvi_data/gcnm_export/subject006_baseline_full"
```

Pilot subset first:
```bash
export GCNM_MAX_FRAMES=500
export GCNM_STRIDE=1
bash scripts/export_production.sh "<bioz_dir>" "<out_dir>"
```

### 3. Validate vs HDF5 (before any training)
```bash
bash scripts/validate_session.sh \
  "/home/lsanc68/ece_bst_link/common/data/pvi_data/gcnm_export/subject006_baseline_full" \
  "/home/lsanc68/ece_bst_link/common/data/pvi_data/main/subject006_baseline_masked.h5"
```

Check `validation_report.json` in export dir:
- `resistance.mean_channel_corr` — target > 0.5 (indicative; time bases differ)
- `img.mse` — only meaningful after ring mesh or alignment

### 4. (Later) Batch + train — only after step 3 acceptable

---

## Pilot export (previous session)

```
gcnm_export/subject001_baseline_pilot/   # 5 frames, butter HP, 32×32, non-production config
```

Superseded by production path above.

The current real-ring pilot is at `data/subject006_ring_pilot5/`. It produced complex
32-channel measurements, `(1330, 5)` element reconstructions, `(40, 40, 5)`
images, and nonzero reactance. Its validation report has image-mask IoU 1.0;
same-index signal metrics are not pass/fail evidence because the HDF5 timeline is
cardiac-period interpolated.

---

## Blockers for exact production parity

1. **Multi-trial merge + alignment** — baseline HDF5 merges multiple bioz trials via scionova_02/04  
2. **NOVA sync chop** — not ported (scionova_01)  
3. **SCIOSPEC.remap** — partial (8-channels assumed in first 8 of 32)

---

## Completed subject006 checkpoint (2026-07-12)

The preceding waiting/blocker section is superseded:

- Raw `.eit` and MATLAB full caches are present for all 11 subject006 baseline trials.
- The US120 real-ring export and corrected 500-frame production pilot have run.
- The direct frame-matched HDF validation reaches correlation 0.9936 and mask IoU 1.0.
- The HDF period order, trial assignments, leakage-safe packs, and GCNM engineering
  smoke run are complete.

See `SUBJECT006_STATUS.md` and the JSON reports under `data/` for current results.
The remaining raw-parity limitation is exact reproduction of manual peak merges;
it does not block HDF-based trial-disjoint dataset construction.
