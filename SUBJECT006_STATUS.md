# Subject 6 PVI → 2D GCNM status

**Updated:** 2026-07-12

## Problem being solved

The PVI ring provides 32 boundary-voltage measurements from eight electrodes, but
the desired output is a conductivity change on 1,330 interior FEM elements (and a
40×40 display image). The inverse problem is severely underdetermined. Production
PVI uses one regularized linearized Newton step. The adaptation goal is to retain
that PVI physics update and let a graph convolutional network learn spatial
corrections on the element-adjacency graph.

## Subject and mesh

- Subject: `subject006`, baseline session.
- Ring: **US120** (12.0 cm identifier).
- Evidence: the US120 40×40 mapping has 1,364 finite pixels and zero mask XOR
  against `subject006_baseline_masked.h5`; it is the unique matching size in the
  available ring collections.
- Inverse mesh: 738 nodes, 1,330 elements.
- Refined forward mesh: 2,805 nodes, 5,320 elements.

## Production inverse reproduced

The exact external MATLAB path is:

1. Build the unweighted coarse-mesh Jacobian at homogeneous conductivity 0.7.
2. Use the refined projected Laplacian
   `Rrec = (c2f.T @ Rfwd @ c2f) / 2`.
3. Compute `D1 = solve(J.T @ J + lambda² Rrec, J.T)`, with lambda = 5e-4.
4. Return trial-relative conductivity `-D1 @ (vmeas - vmeas[:, 0])`.

This is implemented in `gcnm_pvi/gcnm_differential.py`. It does not use voltage
normalization or `laplace.T @ laplace`; those belong to the experimental iterative
GCNM LM path, not the production pseudo-label inverse.

The Python image mapping also now follows MATLAB column-major pixel enumeration.
Comparison with the collection's native 32×32 mapping is exact to numerical
precision after this correction.

## Validation results

Direct reconstruction from 500 aligned HDF high-pass resistance frames gives:

| Metric | Result |
|---|---:|
| Finite pixels | 1,364 |
| Mask IoU | 1.0000 |
| Global image correlation | 0.9936 |
| Median per-frame correlation | 0.9914 |
| Minimum per-frame correlation | 0.9776 |
| RMSE | 6.86e-4 |
| Normalized RMSE | 0.1131 |
| Optimal scale | 0.9907 |

This check is independent of raw/cardiac-period timing because both the inverse
and high-pass operation are linear. Report:
`data/subject006_hdf_reconstruction_validation.json`.

Raw MATLAB cache parity is exact for sampled voltage frames, channel remapping,
the signed 10 mA current, resistance/reactance conversion, the 5 Hz filter, and
centered-shrink `movmean` behavior.

## Period alignment

- Automatic detection produced 1,310 candidate periods across all 11 trials.
- The HDF contains 1,298 periods of 50 phase points.
- Monotonic LP-feature matching selects 1,298 candidates and skips 12.
- LP resistance mean channel correlation: 0.99982.
- LP reactance mean channel correlation: 0.99999.
- HP correlation is lower (~0.67) because the archived pipeline included manual
  peak edits that merge some spurious automatic intervals.

The recovered order and per-trial counts are sufficient to create acquisition-
trial-disjoint splits. Exact raw re-segmentation of every HP boundary still needs
the missing manual peak-review record or a dedicated merge algorithm.

## Leakage-safe dataset

`data/subject006_gcnm_hdf/` contains compact production pseudo-label packs built
from the HDF `mask01` clean periods. Ten phases per clean period are retained.

| Split | Acquisition trials | Samples |
|---|---|---:|
| Train | 1–7 | 6,930 |
| Validation | 8–9 | 1,180 |
| Test | 10–11 | 1,600 |

All phases from a heartbeat stay in one split. Labels are deterministic PVI
one-step outputs, so this dataset is appropriate for data-path testing and model
distillation, not for claiming that GCNM improves on PVI.

## GCNM smoke result

A one-iteration, one-epoch, 17-parameter CPU model was trained on 8 examples and
validated on 2 examples. On four examples from held-out trials 10 and 11:

| Metric | Result |
|---|---:|
| Element correlation | 0.9812 |
| Image correlation | 0.9828 |
| Element MSE | 9.56e-6 |
| Image MSE | 8.47e-6 |

This proves mesh loading, graph construction, fixed Newton features, training,
checkpointing, and held-out inference work together. It is not a scientific
performance result because the Newton feature and target are the same PVI
pseudo-label.

## What remains

1. The clean synthetic/anatomical supervision path is implemented and its full
   Slurm training/evaluation chain is submitted. Independent experimental ground
   truth remains necessary before making physiological claims.
2. Define the real-data fine-tuning formulation. A fixed differential Newton feature is
   naturally one iteration; a true multi-iteration method needs a nonlinear
   current-estimate update and careful reference handling.
3. Tune the clean-truth GCNM loss after the completed Slurm run: stage 1 improves
   nonlinear-holdout correlation and slightly improves RMSE, while stage 2
   over-amplifies background despite better localization.
4. Evaluate on real recordings using independent targets: forward-voltage
   residuals, controlled phantoms, vessel localization, or downstream BP metrics.
5. If exact raw-to-HDF reproduction is required, recover the manual peak merges.

## Reproducible commands

```bash
# Corrected 500-frame raw export
"$GCNM_PYTHON" \
  -m gcnm_pvi.export_production_session \
  --config configs/subject006_pvi08_production.yaml \
  --session-dir "/path/to/subject006/baseline/trial/bioz" \
  --out-dir data/subject006_ring_pilot500 --max-frames 500

# Frame-matched inverse validation
bash scripts/validate_hdf_reconstruction.sh \
  /path/to/subject006_baseline_masked.h5

# Leakage-safe dataset
bash scripts/export_hdf_gcnm_dataset.sh \
  /path/to/subject006_baseline_masked.h5
```
