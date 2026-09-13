# GCNM runtime 0.3.1: numerical correction and live-data validation

This release fixes a numerical failure in the optimized physics update. The
released projected-fine 15-ring weights remain unchanged; no retraining is
needed to use the corrected solver. This is not evidence that those weights
generalize to every live acquisition condition.

## What changed in the core

The equation remains `(J.T @ J + lambda**2 * R + lambda_LM * I) p = -J.T @ r`.
The model configuration, graph architecture, target scale, regularization,
conductivity floor, and checkpoint tensors are unchanged.

The former low-rank implementation could subtract nearly equal quantities in
the regularizer's null-mode Schur complement and amplify roundoff through
unchecked iterative refinement. On a recorded transition this produced an
element magnitude of approximately 40 million S/m.

The replacement solves the same regularized least-squares problem using an
SVD in measurement space. Its null-mode denominator is evaluated as a weighted
sum of squares, avoiding the unstable subtraction. Finite-value and scaled
stationarity checks protect the result; a rank-checked augmented least-squares
solve is an exceptional fallback for the same objective, not extra damping.
An unobservable null mode raises an error rather than inventing a solution.

The implementation marker is `measurement_svd_v1`. Batch metadata now includes
stage-1 conductivity-floor counts and stage-2 physics-step RMS. These are
diagnostics, not new clipping operations or physiological quality thresholds.

## Validation completed on 12 September 2026

- Replaying all 1,844 originally stored live model-input tensors produced
  finite results without using the augmented fallback. The problematic
  transition became approximately 4.59642 S/m, agreeing with an independent
  augmented solve rather than the unstable saved result.
- On ten selected ordinary/transition frames, the maximum difference from
  the independent augmented solve was `9.54e-7 S/m`, relative L2 `7.06e-8`.
  This checks forward agreement as well as the stationarity check; a small
  backward error alone is insufficient for ill-conditioned systems.
- Earlier experimental exports were replayed using their original signed
  voltage inputs: nine US075 subjects, 450 frames. The maximum image difference
  was `5.03e-8 S/m`, aggregate relative L2 `4.00e-7` over finite pixels.
- A 50-frame US120 same-input test of the real-time resident runtime against
  standalone reconstruction produced bit-identical stage-2 outputs.
- Corrected sparse/batched versus dense-reference reconstruction was checked
  on 50 test inputs for each of the 15 ring models: 750 frames, all finite,
  maximum absolute element difference `4.66e-10 S/m`, worst per-ring relative
  L2 `4.43e-8`, and no stage-1 conductivity-floor events or dense-solver warnings.
  Fourteen rings used `differential_main_b045_1000beats_clean_v1/<ring>/test.npz`.
  The US120 file is absent there, so its separately labeled comparison used
  `differential_US120_1000beats_clean_v1/test.npz`. This is same-input numerical
  parity, not a claim that the two corpus locations are interchangeable.
- Full application-worker replay verified raw-frame retention, invalid warm-up
  marking, 250-frame reference epochs, and sequence/timestamp/reference-event
  alignment between Newton and GCNM in Both mode.

On a difficult 250-frame live-data interval spanning a reference boundary,
the app and standalone outputs were bit-identical when both used the actual
call sizes `[50, 49, 50, 50, 50, 1]`. Comparing with five arbitrary 50-frame
calls instead changed stage-1 outputs by at most `5.96e-8 S/m` and stage-2
outputs by `1.04e-4 S/m` (relative L2 `1.06e-6`, on outputs reaching 4.53 S/m).
The input tensor was identical and conductivity-floor counts were unchanged;
batch-shape roundoff was amplified by the nonlinear stage. The portable
validator therefore matches call boundaries rather than relaxing tolerances.
This diagnostic does not establish physiological validity of that interval.

The private-data regression is `tests/test_recorded_gcnm_stability.py`. Configure
`GCNM_REGRESSION_BUNDLE` and `GCNM_REGRESSION_SESSION` to run it. Recordings and
deployment weights are intentionally not committed.

## What changed in the companion real-time app

Update **both repositories**, then rerun the app's `setup_gcnm.ps1` with the
core repository root as `-GcnmSourcePath`. Keep the existing deployment bundle.
The app requires the corrected solver marker and records source/model hashes.

The corrected experimental-input convention is real voltage
**reference minus current**, with no fitted amplitude factor. GCNM has a
continuous third-order causal 5 Hz low-pass and explicit filter warm-up. Its
default reference policy resets every 250 accepted native frames after warm-up
(five seconds at 50 Hz); retaining a session reference is a selectable option.
Manual Baseline, recording start, and acquisition discontinuities also begin
new epochs. Fifty-frame inference chunks do not themselves reset references.

Both mode captures the two methods' references at the same acquired frame and
displays paired frames at matching timestamps. Their reference vectors are
stored separately because their filters differ. Newton-only reconstruction
remains unchanged. Filter-settling raw rows are retained, with GCNM results
explicitly invalid rather than displayed as meaningful reconstructions.

The default ten-second automatic-SVD ROI warm-up cannot finish within a
five-second reference epoch. The app warns without changing the ROI settings;
choose whole-image/manual ROI, a shorter ROI warm-up, or session reference
explicitly if that overlay is needed.

## Important remaining limitations

After the current hardware-notch, low-pass, and 250-frame reference pipeline
was replayed from each uploaded file's start, the reconstructions still had
large, predominantly negative changes:

| Recording | Accepted frames | Maximum absolute change | Negative element fraction |
| --- | ---: | ---: | ---: |
| First upload (1,296 raw frames) | 1,097 | 4.53341 S/m | 92.92% |
| Second upload (548 raw frames) | 349 | 0.385317 S/m | 94.86% |

Each replay starts the filters from the file's first sample. Unrecorded frames
and filter state before recording cannot be recovered. These are reproducible
software diagnostics, **not validated artery reconstructions**. Contact drift,
physical electrode mapping/polarity, acquisition decoding/scaling, and model
distribution mismatch still need matched measurements or phantom validation.
Do not force positive images or fit a voltage gain to conceal the discrepancy.

The archived GCNM exporter referenced each selected **five-beat** window,
whereas the original Newton workflow used a trial-level reference. Archived
experimental voltages were zero-phase filtered and beat-resampled. A causal,
native-50-Hz stream with five-second epochs is a documented approximation,
not identical preprocessing; causal versus double-pass filtering changes both
phase and magnitude response.

The companion app contains `scripts/validate_gcnm_recording.py` and
`docs/GCNM_TROUBLESHOOTING.md` with Windows commands, H5 schema details, and
source-machine checks. A software-parity pass establishes agreement on the
same input, not anatomical accuracy or a guarantee that future live problems
must originate in the ring.
