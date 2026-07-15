# PVI Ring → 2D GCNM Adaptation Plan

**Project goal:** Replace (or improve upon) the PVI deterministic 1-step Newton conductivity
reconstruction with the **2D GCNM** deep-learning approach, using our own ScioSpec/PVI data
from the wearable 8-electrode ring.

**Repos involved:**

| Repo | Role |
|------|------|
| `2D_GCNM/` | GCNM training + inference scripts (Injun Lee) — **target method** |
| `Peripheral-Vascular-Impedance-Imaging/` | FEM forward/inverse solver, ScioSpec interface — **physics** |
| `fundational_pvi/` | HDF5 data loading, masks, splits — **reference for processed data only** |

**Date:** 2026-07-06

---

## Table of Contents

1. [What We Are Trying to Achieve](#1-what-we-are-trying-to-achieve)
2. [The Physics and Mathematics](#2-the-physics-and-mathematics)
3. [Our Data: Channels, Formats, and What Each Contains](#3-our-data-channels-formats-and-what-each-contains)
4. [What PVI Does Today (Deterministic Solver)](#4-what-pvi-does-today-deterministic-solver)
5. [What 2D GCNM Does (Deep Learning Solver)](#5-what-2d-gcnm-does-deep-learning-solver)
6. [Side-by-Side Comparison](#6-side-by-side-comparison)
7. [Current GCNM Code vs Our Setup (Gaps)](#7-current-gcnm-code-vs-our-setup-gaps)
8. [What We Have, What We Need, What Is Missing](#8-what-we-have-what-we-need-what-is-missing)
9. [How to Get Each Required Artifact](#9-how-to-get-each-required-artifact)
10. [Adaptation Plan (Phased)](#10-adaptation-plan-phased)
11. [Training Strategies](#11-training-strategies)
12. [Evaluation Plan](#12-evaluation-plan)
13. [Immediate Next Steps](#13-immediate-next-steps)
14. [References in Our Codebase](#14-references-in-our-codebase)

---

## 1. What We Are Trying to Achieve

### The imaging problem

We wear a ring with **8 electrodes** on the arm. At each time frame, the ScioSpec device:

1. Injects current between **adjacent electrode pairs** (8 stimulation patterns)
2. Measures voltages on the boundary (32 differential channels)
3. Asks: **what is the electrical conductivity σ(x, y) inside the arm cross-section?**

The output we want is a **2D conductivity image** — a map of how conductive each region
inside the ring is. Blood vessels, muscle, and skin have different conductivity; changes
in conductivity over the cardiac cycle relate to blood pressure and vascular dynamics.

### The method comparison

| Approach | Type | Output |
|----------|------|--------|
| **PVI (current)** | Deterministic 1-step Gauss–Newton | 40×40 conductivity image per frame |
| **GCNM (target)** | 10 iterations of LM + learned GCN correction | Element σ → projected 40×40 image |

### Research question

> Can a Graph Convolutional Network, iteratively correcting Newton updates on the FEM mesh,
> produce **better** conductivity images than a single Newton step — for our 8-electrode,
> 32-channel PVI ring?

This is well-defined and publishable: same physics, same measurements, different reconstruction.

---

## 2. The Physics and Mathematics

### 2.1 The forward problem (physics)

Inside the 2D domain Ω (arm cross-section):

$$\nabla \cdot (\sigma \nabla u) = 0$$

At electrodes (Complete Electrode Model, CEM):

- Current is injected between electrode pairs
- Voltages are measured between electrode pairs

After finite-element discretization:

$$A(\sigma)\, \Phi = I_{\text{stim}}$$

Boundary measurements:

$$V = M\, \Phi_{\text{elecs}}$$

| Symbol | Meaning |
|--------|---------|
| σ | Conductivity per mesh element — **what we want to recover** |
| u (Φ) | Electric potential field inside the domain |
| V | 32-channel boundary measurement vector — **what we measure** |
| M | Electrode extraction / differential measurement matrix |
| A(σ) | FEM stiffness matrix (depends on σ) |

### 2.2 The inverse problem

Given measured voltages V_meas, find σ such that:

$$\min_\sigma \; \|V(\sigma) - V_{\text{meas}}\|^2 + \lambda^2 \|R(\sigma - \sigma_{\text{ref}})\|^2$$

| Term | Role |
|------|------|
| Data fidelity ‖V(σ) − V_meas‖² | Match boundary measurements |
| Regularization λ²‖R(σ − σ_ref)‖² | Smoothness prior (Laplacian R) |
| λ | Regularization strength (5×10⁻⁴ in the strict production pipeline) |

**Why it is hard:** Only 32 boundary values constrain thousands of interior unknowns.
The problem is **nonlinear** and **severely ill-posed**.

### 2.3 Linearization (shared by both PVI and GCNM)

Linearize around current estimate σ₀:

$$V(\sigma_0 + \Delta\sigma) \approx V(\sigma_0) + J\, \Delta\sigma$$

where **J = ∂V/∂σ** is the Jacobian (sensitivity matrix, size 32 × N_elements).

The **one-step Gauss–Newton / Levenberg–Marquardt** update:

For the strict MATLAB production path, the stored projected Laplacian is already
the reconstruction regularizer `Rrec = (c2fᵀ Rfwd c2f)/2`; it is not squared again:

$$D_1 = (J^T J + \lambda^2 R_{rec})^{-1} J^T$$

$$\Delta\sigma_t = -D_1(V_t - V_0)$$

$$\sigma_{\text{new}} = \sigma_0 - \alpha\, \Delta\sigma$$

PVI applies this **once**. The original GCNM applies iterative LM updates with a
GCN correction; how to formulate those iterations for trial-relative PVI data is
an explicit remaining research decision.

### 2.4 Differential (time-varying) imaging

For real recordings, we do not use absolute voltages. We use **changes relative to a
reference frame** (first frame or baseline):

$$\Delta V_t = V_t - V_{\text{ref}}$$

This removes systematic offsets (electrode contact, absolute impedance level) and isolates
physiological changes (blood volume shifts within the cardiac cycle).

---

## 3. Our Data: Channels, Formats, and What Each Contains

### 3.1 Raw ScioSpec data (`.eit` files) — **primary source, we have all of these**

Each session folder contains numbered `.eit` frame files. Example structure
(`sample_frame_bioz08.eit`):

```
MeasurementChannels: 1,2,3,4,5,6,7,8          ← 8 active electrodes
MeasurementChannelsIndependentFromInjectionPattern: 1..32  ← 32 hardware channels
Excitation patterns: 1-4, 2-5, 3-6, 4-7, 5-8, 6-1, 7-2, 8-3  ← 8 adjacent injections
Per stim: 64 values (32 complex voltages = 32 real + 32 imaginary)
Frequency: 50 kHz
Frame rate: ~50 Hz
```

**Processing chain (raw → reconstruction input):**

```
.eit frame
  → read + remap electrode channels
  → make_eit()  [differential voltage extraction]
  → vmeas (32,)   ← THIS is the GCNM input V
```

The existing code for this lives in:
- `Peripheral-Vascular-Impedance-Imaging/python_port/sciospec_interface/Models/Frame.py`
- `.../Models/DataProcessorv2.py` (live acquisition; same math applies offline)

### 3.2 Processed HDF5 recordings (`fundational_pvi` loader) — **reference / cross-check**

Location: `$PVI_DATA_ROOT/main/{subject}_{session}_masked.h5`
(~216 sessions available)

```
data/
  bp/
    signal          (1, T)           blood pressure waveform
  pviHP/                          high-pass component (cardiac-frequency)
    resistance      (32, T)          per-channel resistance — NOT raw vmeas
    reactance       (32, T)          per-channel reactance  — NOT raw vmeas
    signal          (1, T)           aggregate bioimpedance
    img             (40, 40, T)      1-step Newton conductivity image ← PVI output
  pviLP/                          low-pass component (same structure)
masks/
  mask01..mask15    clean sequence intervals (period indices, 1-based)
metadata/
  subject, session, num_periods (~440–870), period_length (50 frames)
```

**Important:** The HDF5 `resistance`/`reactance` (32, T) are processed impedance channels
used for BP prediction in `fundational_pvi`. They are **not** the same as the `vmeas`
vector that PVI/GCNM feed into the Newton solver. For GCNM, generate `vmeas` from raw
`.eit` files.

The HDF5 `img` field **is** the PVI Newton output — use it to verify our offline
preprocessing pipeline is correct.

### 3.3 Channel count summary

| Quantity | Count | Source |
|----------|-------|--------|
| Physical electrodes | **8** | Ring hardware |
| Stimulation patterns | **8** | Adjacent pairs |
| Hardware voltage channels per stim | **32** | ScioSpec device |
| Differential EIT measurements (vmeas) | **32** | `make_eit()` output — **GCNM input V** |
| FEM inverse mesh elements | ~1,000–1,500 | Ring mesh (depends on refinement) |
| Display image pixels | **40×40** | `m2i` projection of element σ |
| Internal solver grid | 32×32 | Used in some PVI Python paths |

> **Verify early:** `elec_configs.num_meas_total` must equal **32** when built from
> your exact excitation sequence. If it prints 40, the protocol config is wrong.

### 3.4 What GCNM requires per sample (from Injun Lee)

| Field | Shape | Description |
|-------|-------|-------------|
| `edge_index` | `[2, n_graph_edges]` | Element adjacency on inverse FEM mesh |
| `x` | `[n_mesh_elements]` | Initial σ estimate (homogeneous, all 1.0) |
| `y` | `[n_mesh_elements]` | Ground-truth σ (training only) |
| `V` | `[n_measurements]` | Boundary vmeas vector (32,) |

During training, `computeLMUpdates` concatenates `[σ, δσ]` → feature matrix
`[n_mesh_elements, 2]` fed to the GCN.

---

## 4. What PVI Does Today (Deterministic Solver)

```
32-channel vmeas  →  calibrate vs reference frame
                  →  build FEM mesh + Jacobian J
                  →  ONE linear solve: Δσ = D1·dV + D3
                  →  σ on FEM elements
                  →  m2i projection
                  →  40×40 conductivity image  (stored as HDF5 img)
```

### Step by step

1. **Mesh:** Triangulate the 2D arm cross-section. σ lives on each triangle (element).
2. **Forward model:** FEM + CEM solves A(σ)Φ = I and computes simulated voltages.
3. **Jacobian:** J = ∂V/∂σ — sensitivity of each of the 32 measurements to each element.
4. **Calibration:** ΔV = V_t − V_ref using a reference (baseline) frame.
5. **One Newton step:** Single linear solve → updated element conductivities.
6. **Image:** Project element σ to 40×40 pixel grid via `m2i` matrix.

### Key properties

- No learning, no training data needed
- Fast (one linear solve per frame)
- Heavily regularized → smooth but blurry images
- Deterministic: same input always gives same output
- This is what produced `data/pviHP/img` in our HDF5 files

### Code locations

| Component | Path |
|-----------|------|
| Forward solver | `python_port/pvi_solver/pvi_forward.py` |
| Inverse solver | `python_port/pvi_solver/pvi_inverse.py` |
| Mesh loader | `python_port/pvi_solver/pvi_mesh2d.py` |
| Electrode config | `python_port/pvi_solver/pvi_configs.py` |
| MATLAB reference | `fem/pvi_inv_analysis.m` |

---

## 5. What 2D GCNM Does (Deep Learning Solver)

GCNM solves the **same inverse problem** but replaces "one deterministic Newton step"
with **learned corrections on the mesh graph**, repeated over 10 outer iterations.

```
For k = 0, 1, ..., 9:
  1. LM update:   δσ_k = (J^T J + λI)^{-1} J^T (V(σ_k) - V_meas)   [same physics as PVI]
  2. Features:    h_k = [σ_k, δσ_k]  per element
  3. GCN:         σ_{k+1} = GCN_θ(h_k, edge_index)                 [learned correction]
  4. Train GCN_θ to minimize ||σ_{k+1} - σ_true||²                 [supervised, training only]
```

### What each component does

| Component | Role |
|-----------|------|
| `V` | 32 boundary measurements |
| `x` (initial σ) | Homogeneous guess (all 1.0) |
| `computeLMUpdates` | Same physics as PVI: Jacobian + linear solve → δσ |
| `edge_index` | Graph connecting mesh elements that share an edge |
| `GCNBlock` | Neural network: `[σ, δσ]` per element → improved σ |
| `y` (ground truth) | True σ — used **only during training** |

### Training loop (from `GCNM_training.py`)

For each outer iteration k = 0, …, 9:

1. Compute LM update δσ_k from current σ_k and V
2. Form features [σ_k, δσ_k] per element
3. **Train GCN** to minimize ‖GCN(σ_k, δσ_k) − σ_true‖²
4. Apply trained GCN to all samples → new σ_{k+1}
5. Save model `models/sample_models_k.pt`
6. Repeat

At **inference**, only V and the saved models are needed — no ground truth.

### Intuition

- **PVI:** "What does linearized physics say?" — asked once.
- **GCNM:** "What does linearized physics say?" then "What correction should I make,
  given what my neighboring elements look like?" — asked 10 times, correction learned
  from examples.

### Current code limitations

The received code (`GCNM_training.py`, `GCNM_testing.py`) is a **research prototype**:

- Uses **PyEIT** (not PVI solver) for forward/Jacobian
- Uses **32-electrode circle/thorax** mesh (not our 8-electrode ring)
- Generates **synthetic** elliptical anomalies (not real data)
- No separate data-loading module (noted by Injun Lee)
- Requires: `PyEIT`, `PyTorch`, `torch_geometric`

---

## 6. Side-by-Side Comparison

| | PVI (1-step Newton) | GCNM |
|--|---------------------|------|
| **Problem solved** | Recover σ(x,y) from 32 boundary channels | Same |
| **Physics used** | FEM forward + Jacobian J | Same J for δσ computation |
| **How interior is filled** | Laplacian regularizer R only | GCN message passing on mesh neighbors |
| **Iterations** | 1 linear solve | 10 × (LM + GCN) |
| **Learning** | None | GCN weights trained on (σ_true, V) pairs |
| **Training data needed** | No | Yes |
| **Deterministic** | Yes | No (learned) |
| **Output** | 40×40 image via m2i | Element σ → same m2i → 40×40 image |
| **Current status** | Working on our data | Synthetic PyEIT only; needs adaptation |

---

## 7. Current GCNM Code vs Our Setup (Gaps)

| | Current `2D_GCNM` | Our PVI ring |
|--|-------------------|--------------|
| Electrodes | 32 (PyEIT circle) | **8** |
| Measurements | PyEIT protocol (~hundreds) | **32 channels** |
| Mesh | PyEIT circle/thorax | PVI custom ring FEM mesh |
| Forward solver | PyEIT `EITForward` | `PviForward` (CEM FEM) |
| Ground truth | Synthetic elliptical anomalies | Not available for real data |
| Image output | Direct tripcolor on mesh | 40×40 via `m2i` projection |
| Data loading | Inline synthetic generation | Need offline ScioSpec preprocessor |
| Dependencies | PyEIT, torch_geometric | Not in `fundational_pvi` venv |

---

## 8. What We Have, What We Need, What Is Missing

### ✅ What we have

| Item | Location / Notes |
|------|-----------------|
| Raw ScioSpec `.eit` files | All sessions — **can generate anything** |
| Processed HDF5 recordings | `$PVI_DATA_ROOT/main/*.h5` (216 sessions) |
| PVI FEM forward/inverse (Python) | `Peripheral-Vascular-Impedance-Imaging/python_port/pvi_solver/` |
| ScioSpec interface (Python) | `.../python_port/sciospec_interface/` |
| MATLAB ScioSpec loader | `external_packages/SCIOSPEC.m` |
| MATLAB ring mesh (16-el version) | `data/validation_model.mat` (`ring2d_S4L_fwd/inv`) |
| Mesh + mappings (phantom) | `python_port/pvi_solver/_data/_mesh16_r160/` |
| Mappings mat file | `data/mesh_default_160.mat` (c2f, m2i) |
| GCNM prototype code | `2D_GCNM/GCNM_training.py`, `GCNM_testing.py` |
| HDF5 Newton images (reference) | `data/pviHP/img` — for verification |
| fundational_pvi data loader | For cross-check and downstream BP only |

### 🔧 What we need to build / obtain

| Item | Priority | How |
|------|----------|-----|
| **vmeas (32, T) per session** | P0 — blocker | Offline `.eit` → `make_eit()` batch processor |
| **8-electrode ring inverse mesh (HDF5)** | P0 — blocker | Export from MATLAB mesh pipeline |
| **m2i mapping matrix** | P0 — blocker | From `mesh_default_160.mat` or mapping `.h5` |
| **edge_index** | P0 | Build once from ring mesh elements |
| **sigma_elem pseudo-labels** | P1 — for training | PVI Newton on generated vmeas |
| **GCNM data loader module** | P1 | New `src/data_adapter.py` |
| **PyEIT → PviForward swap** | P1 | Modify `computeLMUpdates` |
| **GCNM training packs (NPZ)** | P2 | Batch exporter per frame |
| **Adapted GCNM scripts** | P2 | Refactor monolithic scripts |

### ❌ What is missing

| Item | Impact | Resolution |
|------|--------|------------|
| **8-el ring mesh in HDF5** | Cannot run PVI/GCNM on correct geometry | Export from MATLAB; `make_ring` is a stub in Python |
| **Offline `.eit` batch reader** | Cannot generate vmeas at scale | Port `SCIOSPEC.read` or wrap MATLAB |
| **Confirmed vmeas ↔ HDF5 equivalence** | Risk of wrong inputs | Verify 1 session vs HDF5 `img` |
| **GCNM deps in environment** | Cannot run GCNM | Install `pyeit`, `torch_geometric` |
| **Ground truth σ for real data** | Cannot do supervised training on real frames without proxy | Use Newton σ as pseudo-labels |
| **Separate data-loading module** | Noted by Injun Lee as missing | Build as part of adaptation |
| **Protocol verification** | Wrong J if pattern mismatches | Confirm `num_meas_total == 32` |

---

## 9. How to Get Each Required Artifact

### 9.1 `vmeas` — boundary measurements (GCNM input `V`)

**Source:** Raw `.eit` files

```python
# Per frame (offline, same math as live DataProcessor):
frame.remap(elec_configs.channel_idx)
vmeas = frame.make_eit(
    elec_configs.potential.extractor,
    num_elecs,
    elec_configs.num_meas_total,
    elec_configs.num_meas_per_stim,
)
# vmeas.shape = (32, 1)
```

Read `excitation_sequence` and `meas_pattern` from each `.eit` file header — do not
hardcode. MATLAB alternative: `ensemble = SCIOSPEC.read(session_folder, [])`.

**Save as:** `vmeas` array shape `(32, T)` per session.

### 9.2 `sigma_elem` — element conductivity (GCNM training label `y`)

**Source:** PVI 1-step Newton on generated vmeas

```python
pvi_inverse.calibrate(vmeas_t0=vmeas[:, 0])   # reference frame
for t in range(T):
    sigma_t = pvi_inverse.solve(vmeas_tk=vmeas[:, t])
```

**Save as:** `sigma_elem` array shape `(N_elements, T)`.

**Note:** This is a **pseudo-label** (Newton output), not absolute ground truth.
GCNM learns to improve upon it.

### 9.3 `sigma_img` — 40×40 display image

**Source:** `m2i` projection of `sigma_elem`

```python
img_t = m2i @ sigma_elem[:, t]   # reshape to (40, 40) or (32, 32)
```

**Verify against:** `data/pviHP/img` in the corresponding HDF5 session.

### 9.4 `edge_index` — mesh element graph

**Source:** Ring inverse mesh elements (build once)

```python
# Same logic as GCNM_training.py lines 386-394:
for each pair of elements (i, j):
    if i != j and share a node:
        adj[i, j] = 1
edge_index = torch.tensor(np.array(adj.nonzero()), dtype=torch.long)
```

### 9.5 Ring FEM mesh + mappings

**Source:** MATLAB mesh pipeline

- Check: `data/validation_model.mat` (16-el ring — may need 8-el variant)
- Check: `data/mesh_default_160.mat` (has `m2i`, `c2f`, `mesh_pvi16_fwd/inv`)
- Export to HDF5 using `pvi_mesh2d.load_mesh_hdf5` format
- Fingerprint tool: `python -m src.scripts.inspect_mesh_files <dir> --recursive --ring-elecs 8`

### 9.6 GCNM training packs (NPZ format)

```python
np.savez_compressed(
    f"sample_{idx:05d}.npz",
    sigma=sigma_elem[:, t],    # (N_elements,)
    V=vmeas[:, t],             # (32,)
    bkg=1.0,
)
```

---

## 10. Adaptation Plan (Phased)

### Phase 0 — Environment and repo layout (1 day)

```
2D_GCNM/
├── PVI_GCNM_ADAPTATION_PLAN.md   ← this file
├── requirements.txt              # torch, torch_geometric, pyeit, scipy, h5py
├── env/cluster.env               # PVI_DATA_ROOT, mesh paths
├── src/
│   ├── sciospec_reader.py        # offline .eit → frames
│   ├── vmeas_exporter.py         # frames → vmeas time series
│   ├── mesh_utils.py             # load ring mesh, build edge_index
│   ├── data_adapter.py           # pack GCNM tensors
│   ├── gcnm_model.py             # GCNBlock, computeLMUpdates (extracted)
│   ├── train.py
│   └── test.py
└── configs/default.yaml
```

Install dependencies:

```bash
pip install torch torch_geometric pyeit scipy h5py matplotlib
```

### Phase 1 — Offline ScioSpec preprocessing (2–3 days) `[P0]`

**Goal:** `.eit` → `vmeas` + Newton `sigma` for one session; verify against HDF5.

1. Build offline `.eit` batch reader (Python port of `SCIOSPEC.read` or MATLAB wrapper)
2. Configure `PviElecConfigs` from `.eit` file headers (8 electrodes, adjacent)
3. Confirm `num_meas_total == 32`
4. Run `make_eit()` per frame → `vmeas (32, T)`
5. Run PVI Newton → `sigma_elem (N_elements, T)` + `img (40, 40, T)`
6. **Verification gate:** compare output `img` to HDF5 `data/pviHP/img` for same session
   - Must match within tolerance before proceeding

**Deliverable:** `preprocessed/{session_name}/vmeas.npy`, `sigma_elem.npy`, `img.npy`

### Phase 2 — Ring mesh export (1–2 days) `[P0]`

**Goal:** Correct 8-electrode ring mesh in HDF5 + m2i mapping.

1. Locate 8-el ring mesh in MATLAB (`mesh_default_160.mat`, team mesh files)
2. Export inverse mesh + forward mesh + `m2i`/`c2f` to HDF5
3. Verify Python `PviForward.compute_jacobian()` matches MATLAB output on test case
4. Build `edge_index` from inverse mesh

**Deliverable:** `mesh/ring_inv.h5`, `mesh/ring_fwd.h5`, `mesh/ring_mappings.h5`, `mesh/edge_index.pt`

### Phase 3 — GCNM core adaptation (3–5 days) `[P1]`

**Goal:** Replace PyEIT with PVI physics in GCNM scripts.

| Change | From | To |
|--------|------|----|
| Mesh | `mesh.create(n_el=32, fd=circle)` | `load_mesh_hdf5('ring_inv.h5')` |
| Forward/Jacobian | `EITForward.compute_jac()` | `PviForward.compute_jacobian()` |
| LM update | `computeLMUpdates` with PyEIT | Same logic, PVI Jacobian |
| Protocol | `protocol.create(32, ...)` | `PviElecConfigs(8 stim patterns, meas_pattern)` |
| Electrodes | 32 | **8** |
| Measurements | PyEIT count | **32** |
| Image output | tripcolor on mesh | m2i → 40×40 |

Extract functions from monolithic scripts into `src/gcnm_model.py` and `src/data_adapter.py`.

**Deliverable:** Adapted training script that loads preprocessed NPZ packs.

### Phase 4 — Synthetic smoke test (1–2 days) `[P1]`

**Goal:** Validate adapted GCNM loop before touching real data.

1. Generate synthetic anomalies on ring mesh (or PyEIT circle with n_el=8 as fallback)
2. Run full 10-iteration GCNM training
3. Confirm models save, inference runs, metrics compute
4. Visual inspection of predicted vs true σ

**Deliverable:** Baseline GCNM results on synthetic ring-like data.

### Phase 5 — Real data training (3–5 days) `[P2]`

**Goal:** Train GCNM on real pseudo-labeled data.

1. Batch preprocess N sessions (start with 10–20 subjects)
2. Pack into GCNM NPZ format with Newton `sigma_elem` as `y`
3. Train 10-iteration GCNM
4. Subject-held-out split (train subjects 001–080, test 081+)

**Deliverable:** Trained `models/*.pt` on real PVI data.

### Phase 6 — Evaluation and comparison (2–3 days) `[P2]`

See [Section 12](#12-evaluation-plan).

**Deliverable:** Comparison report: GCNM images vs Newton images.

---

## 11. Training Strategies

### Option A — Real-data pseudo-supervised (primary)

- Labels `y` = PVI Newton `sigma_elem`
- GCN learns to refine Newton using mesh graph structure
- **Pro:** Directly targets improvement over current pipeline
- **Con:** Labels are noisy (Newton is already approximate)

### Option B — Synthetic pretrain + real fine-tune

- Pretrain on synthetic anomalies (forward model on ring mesh)
- Fine-tune on real `(vmeas, sigma_newton)` pairs
- **Pro:** More stable; GCN learns general anomaly shapes first
- **Con:** Domain gap between synthetic and real

### Option C — Synthetic bootstrap first (recommended start)

1. Run adapted GCNM on **synthetic** ring data → validate code path
2. Plug in real preprocessed packs from Phase 1
3. Compare Option A vs B on held-out subjects

### Recommended order: C → A → (optional) B

---

## 12. Evaluation Plan

### Metrics available for real data

| Metric | Available? | Formula / Method |
|--------|------------|-----------------|
| RE_sigma vs true σ | ❌ No absolute truth | — |
| RE_sigma vs Newton img | ✅ Yes | ‖σ_GCNM − σ_Newton‖₁ / ‖σ_Newton‖₁ |
| MSE vs Newton img | ✅ Yes | mean((σ_GCNM − σ_Newton)²) |
| RE_voltage (data consistency) | ✅ Yes | ‖U(σ_GCNM) − V_meas‖₂ / ‖V_meas‖₂ |
| Visual quality | ✅ Yes | Side-by-side 40×40 images |
| Downstream BP correlation | ✅ Yes | Feed GCNM img vs Newton img to frozen PVI core |

### Comparison protocol

For each test frame:

1. **Input:** same `vmeas` (32,)
2. **PVI output:** 1-step Newton image (from HDF5 `img` or recomputed)
3. **GCNM output:** 10-iteration learned image
4. **Report:** visual comparison + RE_voltage + RE_sigma vs Newton

### Train/test split

- **Subject-held-out:** train on subjects 001–080, test on 081–091
- Use `GraphBipartitePartitioner` from `fundational_pvi` for frame-level splits
  within sessions (no temporal leakage)
- Use `masks/mask05` or `mask10` for clean cardiac cycles only

---

## 13. Immediate Next Steps

| # | Task | Owner | Blocker? |
|---|------|-------|----------|
| 1 | Confirm `num_meas_total == 32` for our excitation sequence | Student | Yes |
| 2 | Build offline `.eit` → `vmeas` exporter for **one session** | Student | Yes |
| 3 | Verify exported `img` matches HDF5 `data/pviHP/img` | Student | Yes (gate) |
| 4 | Export 8-el ring mesh + m2i to HDF5 | Team / MATLAB | Yes |
| 5 | Install GCNM deps; run original synthetic smoke test | Student | No |
| 6 | Swap PyEIT → PviForward in `computeLMUpdates` | Student | After 4 |
| 7 | Batch preprocess 10–20 sessions → NPZ packs | Student | After 1–3 |
| 8 | Train GCNM; compare vs Newton on held-out subjects | Student | After 6–7 |

**Critical path:** Steps 1 → 2 → 3 → 4 → 6 → 7 → 8

---

## 14. References in Our Codebase

### 2D GCNM (target method)

| File | Purpose |
|------|---------|
| `2D_GCNM/GCNM_training.py` | Synthetic data generation + 10-iter GCNM training |
| `2D_GCNM/GCNM_testing.py` | Inference + RE_sigma, MSE, RE_voltage metrics |

Key functions: `initializeDataset`, `computeLMUpdates`, `GCNBlock`, `trainModel`,
`applyModel`.

### PVI physics solver

| File | Purpose |
|------|---------|
| `python_port/pvi_solver/pvi_forward.py` | FEM forward + `compute_jacobian()` |
| `python_port/pvi_solver/pvi_inverse.py` | 1-step Newton inverse |
| `python_port/pvi_solver/pvi_mesh2d.py` | Mesh load/save HDF5 |
| `python_port/pvi_solver/pvi_configs.py` | 8-el electrode protocol |
| `fem/pvi_inv_analysis.m` | MATLAB reference inverse |

### ScioSpec interface

| File | Purpose |
|------|---------|
| `python_port/sciospec_interface/Models/Frame.py` | `make_eit()` → vmeas |
| `python_port/sciospec_interface/Models/DataProcessorv2.py` | Live frame processing |
| `python_port/sciospec_interface/PVI/pvi_inverse.py` | Streaming reconstruction |
| `external_packages/SCIOSPEC.m` | MATLAB `.eit` batch loader |

### fundational_pvi (data reference only)

| File | Purpose |
|------|---------|
| `src/pipeline/data_extraction.py` | `PviRawDataset` HDF5 loader |
| `src/scripts/inspect_mesh_files.py` | Fingerprint mesh HDF5 files |
| `src/models/eit_recon.py` | Learned recon (different approach) |
| `PLAN.md` §3.4, §14.4 | EIT reconstruction track notes |

### Data locations

| Path | Contents |
|------|----------|
| Raw `.eit` sessions | ScioSpec export folders (we have all) |
| `$PVI_DATA_ROOT/main/*.h5` | Processed recordings (216 sessions) |
| `data/validation_model.mat` | Ring mesh MATLAB (16-el) |
| `data/mesh_default_160.mat` | Mesh + m2i mappings |
| `data/sample_frame_bioz08.eit` | Example raw frame format |

### External dependencies (GCNM)

- [PyEIT](https://github.com/eitcom/pyEIT) — forward problem (to be replaced by PVI)
- [PyTorch](https://pytorch.org/) — training
- [torch_geometric](https://pytorch-geometric.readthedocs.io/) — GCN layers

---

## Appendix A: End-to-End Target Pipeline

```
Raw ScioSpec .eit files
        │
        ▼
  [Offline reader]
        │
        ▼
  vmeas (32, T)  ──────────────────────────────┐
        │                                         │
        ▼                                         │
  PVI Newton (reference / pseudo-label)           │
        │                                         │
        ▼                                         │
  sigma_elem (N_elem, T)                         │
        │                                         │
        ├──── GCNM TRAINING ────┐                 │
        │   y = sigma_elem      │                 │
        │   V = vmeas           │                 │
        │   edge_index (mesh)   │                 │
        │                       ▼                 │
        │              Train GCNM (10 iterations) │
        │                       │                 │
        │                       ▼                 │
        └───────────── GCNM INFERENCE ────────────┘
                                │
                                ▼
                    sigma_elem (GCNM prediction)
                                │
                                ▼
                    m2i projection → 40×40 image
                                │
                                ▼
                    Compare vs PVI Newton image
```

---

## Appendix B: Context from Collaboration (Hyeuknam / Injun Lee)

> The current `training.py` and `testing.py` files both include the code for generating
> training and testing data. At the moment, there is no separate data-loading module for
> externally generated samples. Use the current implementation as a starting point, then
> adapt together as the project direction becomes more specific.

Required libraries: **PyEIT**, **PyTorch**, **torch_geometric**.

Required data formats per sample:

- `edge_index`: `[2, n_graph_edges]`
- `x`: initial σ estimate, `[n_mesh_elements]`
- `y`: ground-truth σ, `[n_mesh_elements]`
- `V`: measured voltages, `[n_measurements]`

`computeLMUpdates` computes δσ; `[σ, δσ]` concatenated → `[n_mesh_elements, 2]` GCN input.

---

*This document consolidates the analysis from the 2026-07-06 planning session.
Update as phases complete and verification gates pass.*
