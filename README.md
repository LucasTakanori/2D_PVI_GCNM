# 2D PVI-GCNM

Graph-Convolutional Newton Methods for two-dimensional peripheral vascular
impedance (PVI) imaging with an eight-electrode wearable ring.

This repository contains the code, experiment entrypoints, documentation, and
compiled technical report for adapting Injun Lee's 2D GCNM idea to the PVI
complete-electrode finite-element solver. It does **not** distribute participant
data, generated datasets, meshes, model checkpoints, scheduler logs, or Python
environments.

## Scientific objective

Electrical impedance tomography estimates an internal conductivity change
`delta_sigma` from boundary voltage changes `delta_v`. For subject 6 the inverse
mesh has 1,330 elements, while each frame contains only 32 voltage measurements.
The problem is therefore strongly underdetermined.

The faithful iterative method implemented here alternates between nonlinear PVI
physics and a graph network. At stage `k`,

```text
predicted_delta_v = F(sigma_baseline + delta_sigma_k) - F(sigma_baseline)
residual          = predicted_delta_v - measured_delta_v
J_k               = J(sigma_baseline + delta_sigma_k)
```

and the physical direction is

```text
(J_k.T J_k + hyper_pvi^2 R + lambda_lm I) p_k = -J_k.T residual.
```

The faithful-core graph input is `[delta_sigma_k, p_k]`. A fresh GCN is trained
greedily for every stage. The same physics routine also runs an iterative-LM-only
control, so improvement from repeated Newton steps can be separated from
improvement produced by the learned graph prior.

The deployable experiments use a homogeneous `0.7 S/m` baseline in both training
and inference. The saved synthetic anatomical baseline contains the true static
vessel locations and is therefore available only as an explicitly labeled oracle
upper-bound ablation; it is never silently substituted at real-data inference.

## What is included

```text
gcnm_pvi/                       Python physics, models, data tools, training and evaluation
configs/                        PVI and subject-6 YAML configurations
scripts/                        User-facing shell and Slurm submission entrypoints
slurm/                          Batch launchers with fixed project resources
tests/                          Unit tests for nonlinear stage recomputation and graph output
reports/gcnm_pvi_latex/         LaTeX source, static figures, and compiled report PDF
cluster.env.example             Portable environment-variable template
ANATOMICAL_GCNM.md              Clean synthetic supervision design
SUBJECT006_STATUS.md            Subject-6 geometry and production-port evidence
IMPLEMENTATION_REPORT.md        Implementation history and verified state
```

The original third-party Injun scripts are not redistributed here. Their method
is audited and described in the technical report.

## Required external components

1. Python 3.10 or newer.
2. Dependencies from `requirements.txt`.
3. The public PVI solver repository:

   ```bash
   git clone https://github.com/Sanchez-Research-Lab/Peripheral-Vascular-Impedance-Imaging.git
   ```

4. Authorized access to the private PVI data if reproducing real-data exports.
5. Slurm access for production experiments.

Create an environment and expose the external solver:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt

export PVI_SOLVER_ROOT=/path/to/Peripheral-Vascular-Impedance-Imaging/python_port/pvi_solver
export GCNM_VENV_ROOT="$PWD/.venv"
export GCNM_PYTHON="$GCNM_VENV_ROOT/bin/python"
export PYTHONPATH="$PWD:$PVI_SOLVER_ROOT:${PYTHONPATH:-}"
```

On a cluster, the same values can be placed in an untracked `env/cluster.env`.
See `cluster.env.example`.

## Complete project workflow

The current production workflow is the 91-subject b045 coordinate-direct
rollout. Its exact environment, registry, split, validation, submission,
resume, and output commands are in
[`docs/COORDINATE_ROLLOUT_RUNBOOK.md`](docs/COORDINATE_ROLLOUT_RUNBOOK.md).
The sections below retain the earlier subject-6 faithful-GCNM workflow because
it documents how the selected coordinate architecture was established.

### 1. Export a ring mesh

The subject-6 ring was identified as `US120`. Export the forward mesh, inverse
mesh, projected Laplacian, coarse-to-fine map, and 40x40 image mapping from the
sibling PVI repository:

```bash
bash scripts/export_ring_mesh.sh US120 data/ring_meshes/subject006_US120
```

Expected private outputs:

```text
data/ring_meshes/subject006_US120/ring_US120_fwd.h5
data/ring_meshes/subject006_US120/ring_US120_inv.h5
data/ring_meshes/subject006_US120/ring_US120_mappings_40.h5
```

### 2. Verify the production inverse

```bash
bash scripts/validate_hdf_reconstruction.sh /path/to/subject006_baseline_masked.h5
```

The verified subject-6 port obtained a 1.0000 finite-mask IoU and 0.9936 image
correlation across 500 matched frames. This validates the Python/MATLAB inverse
and pixel ordering; it does not establish anatomical accuracy.

### 3. Build clean synthetic supervision

The clean target is finite-element vascular conductivity, not a PVI Newton image.
Training voltages are created on the refined 5,320-element mesh and inverted on
the 1,330-element mesh to avoid an inverse crime.

Production generation, training, and evaluation can be submitted with:

```bash
bash scripts/submit_anatomical_pipeline.sh
```

The saved dataset fields used by the faithful method are:

| Field | Meaning |
|---|---|
| `sigma` | Clean inverse-mesh vascular conductivity change |
| `sigma_baseline` | Randomized skin/fat/muscle/bone/blood baseline |
| `V` | Noisy 32-channel differential voltage |
| `V_clean` | Voltage before acquisition perturbations |
| `newton` | Historical fixed one-step feature, retained as a baseline only |

### 4. Test the faithful implementation

```bash
python -m unittest discover -s tests -v
```

The tests verify zero residual, physical sign, nonlinear residual descent,
stage-2 recomputation at the stage-1 state, and zero-initialized residual output.

### 5. Submit one faithful experiment

All model training and inference is launched through `.sh` files and `sbatch`.
The faithful direct-output core is:

```bash
EXPERIMENT=faithful_direct \
OUTPUT_MODE=direct \
BASELINE_MODE=homogeneous \
BASELINE_CONDUCTIVITY=0.7 \
USE_COORDINATES=0 \
POSITIVE_WEIGHT=0 \
BACKGROUND_WEIGHT=0 \
CHECKPOINT_MODE=loss \
bash scripts/submit_faithful_experiment.sh
```

This submits one training job followed by linearized, exact nonlinear, and
real-PVI evaluations. Evaluation includes the matched iterative-LM-only control.
Checkpoints bind the baseline policy and physics hyperparameters, and evaluation
rejects incompatible baseline modes.

### 6. Run the ablation sequence

Change one methodological component at a time:

| Experiment | Output | Coordinates | Vessel weight | Background penalty | Checkpoint |
|---|---|---:|---:|---:|---|
| `faithful_direct` | direct | no | 0 | 0 | MSE |
| `faithful_residual` | current + LM + correction | no | 0 | 0 | MSE |
| `faithful_coords` | selected output | yes | 0 | 0 | MSE |
| `faithful_weighted` | selected output | yes | 8 | 0 | weighted MSE |
| `faithful_background` | selected output | yes | selected alpha | positive | composite |

Example residual ablation:

```bash
EXPERIMENT=faithful_residual \
OUTPUT_MODE=proposal_residual \
USE_COORDINATES=0 \
POSITIVE_WEIGHT=0 \
BACKGROUND_WEIGHT=0 \
CHECKPOINT_MODE=loss \
bash scripts/submit_faithful_experiment.sh
```

Every experiment uses separate model/result directories, preserving previous
scientific evidence.

After selecting the vessel weight, run the explicit background-loss sweep:

```bash
POSITIVE_WEIGHT=1 SUFFIX=_hom_seed0 \
  bash scripts/submit_faithful_background_sweep.sh
```

This evaluates background coefficients `0.25`, `0.5`, and `1.0` with the
composite NRMSE/background/Dice checkpoint criterion.

### 7. Evaluate the correct quantities

Each evaluation reports, per LM and GCNM stage:

- element and mapped-image RMSE;
- element and image correlation;
- localization Dice at the true vessel volume;
- background RMS and vessel mean;
- nonlinear voltage residual before and after each stage;
- conductivity-floor clamp count.

Synthetic nonlinear holdouts provide independent conductivity ground truth. Real
PVI images do not: agreement with them measures reproduction of the existing
Newton inverse only.

The real HDF pack follows the MATLAB display-sign convention. Real evaluation
explicitly multiplies saved `V` by `-1` before physical conductivity inversion and
records that choice in `report.json`; synthetic solver voltages use sign `+1`.

### 8. Rebuild the report

The committed report already includes static figures. To regenerate figures from
private experiment artifacts:

```bash
cd reports/gcnm_pvi_latex
MPLCONFIGDIR=/tmp/gcnm-mpl \
PVI_SUBJECT006_H5=/path/to/subject006_baseline_masked.h5 \
  "$GCNM_PYTHON" make_figures.py

FINAL_FAITHFUL_EXPERIMENT=faithful_background025_hom_seed0 \
MPLCONFIGDIR=/tmp/gcnm-mpl \
  "$GCNM_PYTHON" make_faithful_figures.py
```

Compile with Tectonic:

```bash
XDG_CACHE_HOME=/tmp/tectonic-cache \
XDG_CONFIG_HOME=/tmp/tectonic-config \
tectonic --keep-logs --keep-intermediates main.tex
```

Open `reports/gcnm_pvi_latex/main.pdf` for the mathematical derivation, original
code audit, experiment tables, output comparisons, limitations, and research plan.

## Historical faithful-pilot Slurm resources

The original faithful-pilot launchers preserve the allocation used by those
verified subject-6 experiments:

| Resource | Value |
|---|---|
| Partition/account | `ece_bst` / `ece_bst` |
| Nodes/tasks | 1 / 1 |
| CPUs | 16 |
| Memory | 250 GB |
| GPUs | 1 |
| Wall time | 1 day |

Model architecture and optimizer settings remain in
`configs/subject006_anatomical_gcnm.yaml`: two stages, three 64-channel hidden
layers, learning rate 0.001, batch size 16, at most 150 epochs, patience 15.

The current packed 91-subject rollout has a separate resource contract: up to
four GPUs, 64 CPUs, 1,000 GB of memory, and a 10-day wall limit. See the
production runbook for its job-by-job allocation and resume behavior.

## Reproduced results

Before faithful per-stage recomputation, the fixed-feature anatomical experiment
obtained the following exact nonlinear holdout results over 32 samples:

| Method | Image correlation | Image RMSE | Dice | Background RMS |
|---|---:|---:|---:|---:|
| Fixed production Newton | 0.265 | 0.00506 | 0.295 | 0.00063 |
| Fixed-feature GCN stage 1 | 0.388 | 0.00502 | 0.307 | 0.00202 |
| Fixed-feature GCN stage 2 | 0.463 | 0.00541 | 0.359 | 0.00281 |

The correlation improvement came with increasing false background. The implemented
faithful pipeline recomputes the nonlinear voltage residual, Jacobian, and LM
proposal at every graph stage. Its principal exact nonlinear results are:

| Method | Image correlation | Image RMSE | Dice | Background RMS |
|---|---:|---:|---:|---:|
| Saved production Newton | 0.265 | 0.00506 | 0.295 | 0.000626 |
| Iterative LM stage 2 | 0.291 | 0.00503 | 0.394 | 0.000589 |
| Faithful direct homogeneous core | 0.458 | 0.00484 | 0.230 | 0.000581 |
| Direct + coordinates | 0.561 | 0.00444 | 0.471 | 0.000865 |
| Coordinates + vessel weight 1 | 0.557 | 0.00430 | 0.543 | 0.001541 |
| **Selected: plus background weight 0.25** | 0.557 | 0.00432 | 0.485 | 0.001276 |

The selected model balances spatial correlation, RMSE, localization, background
energy, and nonlinear voltage consistency; larger background penalties reduce
background further but erase vessel support. These are seed-0 synthetic results,
not uncertainty intervals or human anatomical validation. The report contains the
complete direct/residual, coordinate, vessel-weight, background, and oracle-baseline
tables plus best/median/worst output comparisons.

On all 1,600 held-out subject-6 phases, the selected model's stage-1 output has
0.604 global image correlation with the PVI Newton pseudo-reference and reduces
mean voltage residual from `2.36e-5` to `1.68e-5`. Stage 2 falls to 0.169 pseudo-
reference correlation and increases voltage residual to `2.63e-5`, exposing a
synthetic-to-real domain shift. These numbers do not say that Newton is anatomical
truth; they show why intermediate outputs and independent phantom/ultrasound
validation are required.

## Data and publication policy

The public repository intentionally excludes:

- raw or processed participant data;
- generated NPZ/NumPy arrays and HDF5/MATLAB meshes;
- model checkpoints;
- scheduler logs and runtime output;
- virtual environments and private cluster path files.

The `.gitignore` enforces these exclusions. The committed report PDF and static
figure PDFs are derived research documentation and contain no raw acquisition
files.

## Further documentation

- `reports/gcnm_pvi_latex/main.pdf` -- complete technical report
- `ANATOMICAL_GCNM.md` -- anatomical phantom and clean-supervision design
- `SUBJECT006_STATUS.md` -- ring-size and production reconstruction evidence
- `IMPLEMENTATION_REPORT.md` -- implementation and verification history
- `PVI_GCNM_ADAPTATION_PLAN.md` -- original adaptation analysis
- `docs/COORDINATE_ROLLOUT_RUNBOOK.md` -- accepted 91-subject production path
