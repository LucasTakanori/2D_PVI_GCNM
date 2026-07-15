# Clean-truth anatomical GCNM training

**Updated:** 2026-07-12

## Purpose

This is the scientific training path for producing cleaner vascular conductivity
representations. It does **not** use PVI/HDF images as labels. The deterministic
PVI image is retained only as a real-data baseline and preprocessing-parity check.

The supervised target is the clean conductivity change assigned directly to the
US120 inverse-mesh elements. The graph network receives a noisy physical
Gauss–Newton reconstruction as one input feature and learns a residual correction.

## Phantom construction

One anatomical parameter draw defines:

- outer skin and subcutaneous-fat thickness;
- skin, fat, muscle, bone and blood conductivity;
- an off-center elliptical bone;
- one or two randomly positioned elliptical vessels;
- a random cardiac-phase conductivity increase in each vessel.

The same parameters are rasterized independently on:

- the 5,320-element refined forward mesh used to generate voltages;
- the 1,330-element inverse mesh used for clean training labels.

This avoids the previous coarse-to-fine shape bug and reduces inverse crime.

## Simulated acquisition variation

Training data randomizes anatomical background, contact impedance, channel gain,
current gain, white noise and correlated channel noise. Every clean target spans
at least six inverse elements.

Two simulation modes are available:

- `linearized`: precompute randomized anatomical Jacobians on the refined mesh,
  then generate many targets efficiently. This is used for training.
- `nonlinear`: run separate refined-mesh FEM solves for baseline and dynamic
  anatomy. This is used as an independent holdout.

The stored `newton` array is the sign-correct physical Gauss–Newton direction
`+D1 ΔV`. MATLAB PVI displays `-D1 ΔV`, which is its historical image convention
and is not the physical conductivity sign used for clean supervision.

## Model

The residual GCN uses five element features:

1. current GCN estimate;
2. noisy Newton conductivity update;
3. normalized x coordinate;
4. normalized y coordinate;
5. normalized radial coordinate.

The final graph layer starts at zero, so the initial network output is exactly the
Newton input. Training therefore learns only corrections supported by clean
conductivity targets. Vessel elements receive additional loss weight to prevent
the sparse all-zero solution.

## Evaluation

The evaluator reports both Newton and GCNM results against clean truth:

- element and 40×40 image RMSE;
- element and image correlation;
- vessel localization Dice at the true vessel volume;
- background RMS noise;
- mean reconstructed vessel contrast.

It also saves truth/Newton/GCNM comparison figures and all element predictions.

## Slurm workflow

All production-size runs use one GPU, 16 CPUs, 250 GB, one node/task, the
`ece_bst` partition/account, and a one-day time limit so they fit before the
scheduled maintenance window.

```bash
bash scripts/submit_anatomical_pipeline.sh
```

The submitted dependency chain is:

| Job | ID | Purpose |
|---|---:|---|
The original jobs 426824–426828 were cancelled before starting because their
ten-day time requests overlapped scheduled maintenance. The one-day replacement
chain is:

| Job | ID | Latest state at resubmission |
|---|---:|---|
| Linearized data | 426833 | Completed successfully |
| Nonlinear holdout | 426834 | Completed successfully |
| GPU training | 426835 | Completed successfully |
| Linearized evaluation | 426836 | Completed successfully |
| Nonlinear evaluation | 426837 | Completed successfully |

All jobs exited `0:0` with empty error logs.

## First full results

The first clean-truth run improves spatial detection but is not yet a uniformly
better reconstruction. On the exact 32-sample nonlinear holdout:

| Metric | Newton | GCNM stage 1 | GCNM stage 2 |
|---|---:|---:|---:|
| Element correlation | 0.175 | 0.373 | 0.445 |
| Image correlation | 0.265 | 0.388 | 0.463 |
| Localization Dice | 0.295 | 0.307 | 0.359 |
| Element RMSE | 0.00354 | 0.00345 | 0.00365 |
| Image RMSE | 0.00506 | 0.00502 | 0.00541 |
| Background RMS | 0.00063 | 0.00202 | 0.00281 |

Stage 1 is the best balanced result: it roughly doubles correlation while slightly
improving RMSE. Stage 2 further improves localization and correlation but
over-amplifies background and worsens RMSE. Example images show that GCNM detects
the correct broad vessel region where Newton is nearly blank, but it often merges
separate vessels into a diffuse lobe and introduces a central negative artifact.

Therefore this historical run demonstrates learned spatial information beyond the
Newton image, but not the desired clean internal structure. It motivated the
faithful per-stage implementation and the completed coordinate, vessel-weight,
background-loss, and composite-checkpoint ablations documented below.

## Smoke-test interpretation

A local 3/1/1 nonlinear dataset verified the full clean-label code path. It is too
small to establish generalization: the one-sample held-out result overfit and did
not beat Newton. That negative smoke result is retained rather than presented as
an improvement. Scientific conclusions must use the Slurm-scale linearized and
nonlinear reports.

## Important limitation

Differential voltage primarily identifies time-varying conductivity, so this model
targets pulsatile vessels. Static skin/fat/bone layers are randomized forward-model
nuisances, not claimed reconstructed anatomy. Recovering static tissue structure
would require a separate absolute-EIT branch with much stronger calibration and
independent experimental validation.

## Faithful per-stage physics implementation

The historical results above used one fixed Newton feature in both learned stages.
The new faithful path in `iterative_physics.py`, `train_faithful_gcnm.py`, and
`evaluate_faithful_gcnm.py` recomputes the differential forward prediction,
Jacobian, residual, and LM direction at every stage. The evaluator runs the same
nonlinear iterations without a GCN as a physics-only control.

Two baseline policies are deliberately separated:

- `homogeneous`: use 0.7 S/m in training and real inference. This is the deployable
  policy and the default.
- `saved`: use the exact simulated anatomical baseline, including static vessel
  locations. This is an oracle upper-bound experiment and is not available for
  real acquisitions.

On the 32-sample exact nonlinear holdout, the seed-0 homogeneous direct-output
model produced:

| Method | Image correlation | Image RMSE | Dice | Background RMS |
|---|---:|---:|---:|---:|
| Saved fixed Newton | 0.265 | 0.00506 | 0.295 | 0.000626 |
| Iterative LM stage 2 | 0.291 | 0.00503 | **0.394** | 0.000589 |
| Faithful direct GCN stage 2 | **0.458** | **0.00484** | 0.230 | **0.000581** |

This is a real Pareto trade-off. The GCN gives the best correlation and RMSE while
maintaining low background, but iterative LM gives substantially better support
Dice and much stronger voltage consistency. The report therefore presents both
estimators rather than declaring a universal winner.

Coordinates materially improved the deployable direct model. A vessel-weight
sweep selected `alpha=1`, followed by background coefficients 0.25, 0.5, and 1.0:

| Stage-2 model | Image correlation | Image RMSE | Dice | Background RMS | Post-stage voltage RMS |
|---|---:|---:|---:|---:|---:|
| Direct + coordinates | 0.561 | 0.00444 | 0.471 | 0.000865 | 1.73e-6 |
| Coordinates + alpha 1 | 0.557 | 0.00430 | 0.543 | 0.001541 | 4.62e-6 |
| Alpha 1 + background 0.25 | 0.557 | 0.00432 | 0.485 | 0.001276 | 3.18e-6 |
| Alpha 1 + background 0.5 | 0.530 | 0.00446 | 0.462 | 0.000992 | 1.87e-6 |
| Alpha 1 + background 1.0 | 0.527 | 0.00453 | 0.427 | 0.000845 | 1.58e-6 |

`faithful_background025_hom_seed0` is the selected balanced model: it retains the
best weighted-model correlation and near-best RMSE while reducing background and
voltage mismatch relative to the unpenalized alpha-1 checkpoint. Coefficients 0.5
and 1.0 form the expected background-versus-localization Pareto frontier. These are
single-seed synthetic results; physical phantom and independent anatomical
validation remain necessary.
