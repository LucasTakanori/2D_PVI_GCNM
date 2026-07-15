# 2D PVI-GCNM Presentation Outline

Recommended length: four main slides, with one optional conclusions slide.

---

## Slide 1 — Original 2D Graph-Convolutional Newton Method

### Problem being solved

- Reconstruct a two-dimensional conductivity distribution from boundary EIT
  voltages.
- The inverse problem is nonlinear, ill-conditioned, and underdetermined.
- Original experimental setup: 32 electrodes, 928 voltage measurements, and
  synthetic ellipse phantoms.

### Physics update

At iteration or learned stage $k$, the original code computes the nonlinear
voltage residual

$$
r_k = F(\sigma_k)-V_{\mathrm{meas}},
$$

where $F(\sigma_k)$ is the FEM forward solution for the current conductivity
estimate $\sigma_k$. The Jacobian is

$$
J_k = \left.\frac{\partial F}{\partial \sigma}\right|_{\sigma_k}.
$$

A regularized Levenberg--Marquardt/Newton direction is then obtained from

$$
p_k
=
-\left(J_k^{\mathsf T}J_k+\lambda I\right)^{-1}
J_k^{\mathsf T}r_k.
$$

Equivalently, it solves the normal equation

$$
\left(J_k^{\mathsf T}J_k+\lambda I\right)p_k
=-J_k^{\mathsf T}r_k.
$$

### Graph-network update

Each FEM element becomes a graph node. Mesh-neighboring elements are connected by
graph edges. The GCN receives the current conductivity and physics proposal:

$$
H_k=[\sigma_k,p_k].
$$

A separate GCN is trained for each stage:

$$
\sigma_{k+1}
=
\operatorname{GCN}_{\theta_k}(H_k,G).
$$

### Main message

> Original GCNM is a learned iterative inverse solver. It does not directly map
> voltages to an image: physics first proposes an update, and the graph network
> learns how that update should be spatially regularized on the FEM mesh.

Suggested visual:
[`algorithm_flow.pdf`](reports/gcnm_pvi_latex/figures/algorithm_flow.pdf)

This figure now places all three repositories in execution order: Injun's original
2D GCNM, the production PVI repository, and our faithful PVI-GCNM adaptation.

![Three-repository algorithm flow](reports/gcnm_pvi_latex/figures/algorithm_flow.png)

---

## Slide 2 — Peripheral Vascular Impedance Imaging

### Problem being solved

- Estimate pulsatile conductivity changes inside a limb using an eight-electrode
  wearable ring.
- Subject 6 uses the US120 ring geometry.
- The inverse mesh contains 1,330 elements, but each frame contains only 32 voltage
  measurements.

The measurement-to-unknown ratio is therefore

$$
\frac{32}{1330}\approx 0.024.
$$

The differential forward problem is

$$
\Delta V
=
F(\sigma_b+\Delta\sigma)-F(\sigma_b),
$$

where

- $\sigma_b$ is the baseline conductivity;
- $\Delta\sigma$ is the pulsatile conductivity change; and
- $\Delta V$ is the measured boundary-voltage change.

For a small conductivity change, the linear approximation is

$$
\Delta V\approx J(\sigma_b)\Delta\sigma.
$$

### Original PVI processing pipeline

```text
ScioSpec acquisition
        ↓
Channel remapping and temporal filtering
        ↓
32 differential-voltage measurements
        ↓
One-step regularized Newton inverse
        ↓
1,330-element conductivity estimate
        ↓
40 × 40 PVI visualization
```

### Production inverse

The original PVI reconstruction approximately solves

$$
\widehat{\Delta\sigma}_{\mathrm{Newton}}
=
\left(J^{\mathsf T}J+h_{\mathrm{PVI}}^2R\right)^{-1}
J^{\mathsf T}\Delta V,
$$

where $R$ is the projected spatial regularizer.

### Verified implementation

- Forward mesh: 5,320 elements.
- Inverse mesh: 1,330 elements.
- MATLAB/Python comparison over 500 frames:
  - image correlation: $0.9936$;
  - finite-mask intersection over union: $1.000$.

### Main limitation

> The production Newton reconstruction is useful as a physics baseline, but it is
> noisy and diffuse. It is also not anatomical ground truth, so training only to
> reproduce PVI images cannot demonstrate improved vessel recovery.

Suggested visual:
[`measurement_comparison.pdf`](reports/gcnm_pvi_latex/figures/measurement_comparison.pdf)

---

## Slide 3 — Our Faithful PVI-GCNM Adaptation

### What we changed

- Replaced the original PyEIT model with the PVI complete-electrode FEM solver.
- Changed the reconstruction target from absolute conductivity $\sigma$ to
  differential vascular conductivity $\Delta\sigma$.
- Used clean synthetic vessel conductivity as supervision instead of noisy PVI
  Newton images.
- Generated voltages on a 5,320-element forward mesh and reconstructed on a
  separate 1,330-element inverse mesh to reduce the inverse crime.
- Used the same deployable homogeneous baseline in training and inference:

$$
\sigma_b=0.7\ \mathrm{S/m}.
$$

- Restored the defining original-GCNM mechanism: nonlinear physics is recomputed
  after every learned stage.

### Stage 1: nonlinear differential prediction

For the current estimate $\Delta\sigma_k$,

$$
\widehat{\Delta V}_k
=
F(\sigma_b+\Delta\sigma_k)-F(\sigma_b).
$$

The voltage residual is

$$
r_k
=
\widehat{\Delta V}_k-\Delta V_{\mathrm{meas}}.
$$

### Stage 2: recompute the Jacobian

$$
J_k
=
\left.\frac{\partial F}{\partial \sigma}
\right|_{\sigma_b+\Delta\sigma_k}.
$$

### Stage 3: compute the physical proposal

The implemented LM direction solves

$$
\left(
J_k^{\mathsf T}J_k
+h_{\mathrm{PVI}}^2R
+\lambda_{\mathrm{LM}}I
\right)p_k
=
-J_k^{\mathsf T}r_k.
$$

### Stage 4: graph update

The faithful-core graph features are

$$
H_k=[\Delta\sigma_k,p_k].
$$

The coordinate ablation adds the element centroid and radius:

$$
H_k=[\Delta\sigma_k,p_k,x,y,r].
$$

The direct-output model predicts

$$
\Delta\sigma_{k+1}
=
\operatorname{GCN}_{\theta_k}(H_k,G).
$$

The proposal-residual alternative predicts

$$
\Delta\sigma_{k+1}
=
\Delta\sigma_k+p_k+c_{\theta_k}(H_k,G).
$$

### Controlled experiments completed

- Iterative LM without a GCN.
- Direct versus proposal-residual output.
- With and without coordinates $(x,y,r)$.
- Vessel weights $\alpha\in\{0,1,2,4,8\}$.
- Background penalties $\beta\in\{0.25,0.5,1.0\}$.
- Saved anatomical oracle versus homogeneous deployable baseline.

The weighted reconstruction objective is

$$
\mathcal L_{\mathrm{reconstruction}}
=
\frac{1}{SK}
\sum_{s=1}^{S}\sum_{i=1}^{K}
w_{s,i}
\left(\widehat{\Delta\sigma}_{s,i}-\Delta\sigma^*_{s,i}\right)^2.
$$

The explicit background penalty is

$$
\mathcal L_{\mathrm{background}}
=
\beta
\frac{
\sum_{s,i}(1-m_{s,i})\widehat{\Delta\sigma}_{s,i}^{,2}
}{
\sum_{s,i}(1-m_{s,i})+\epsilon
}.
$$

Suggested visual:
[`faithful_ablation_comparison.pdf`](reports/gcnm_pvi_latex/figures/faithful_ablation_comparison.pdf)

---

## Slide 4 — Results and Reconstruction Examples

### Exact nonlinear synthetic holdout

The nonlinear holdout contains 32 independent fine-mesh phantoms that were not
used for training or checkpoint selection.

| Method | Image correlation $\uparrow$ | Image RMSE $\downarrow$ | Dice $\uparrow$ | Background RMS $\downarrow$ |
|---|---:|---:|---:|---:|
| Saved Newton | 0.265 | 0.00506 | 0.295 | 0.000626 |
| Iterative LM stage 2 | 0.291 | 0.00503 | 0.394 | 0.000589 |
| Earlier fixed-feature GCN stage 2 | 0.463 | 0.00541 | 0.359 | 0.002810 |
| Direct GCN + coordinates | **0.561** | 0.00444 | 0.471 | 0.000865 |
| Selected $\alpha=1,\beta=0.25$ | 0.557 | **0.00432** | **0.485** | 0.001276 |

### Metric definitions

Image RMSE is

$$
\operatorname{RMSE}
=
\sqrt{
\frac{1}{SK}
\sum_{s=1}^{S}\sum_{i=1}^{K}
\left(
\widehat{\Delta\sigma}_{s,i}-\Delta\sigma^*_{s,i}
\right)^2
}.
$$

Background RMS is

$$
\operatorname{BG\text{-}RMS}
=
\sqrt{
\frac{
\sum_{s,i}(1-m_{s,i})\widehat{\Delta\sigma}_{s,i}^{,2}
}{
\sum_{s,i}(1-m_{s,i})
}
}.
$$

Localization Dice at the true vessel volume is

$$
\operatorname{Dice}
=
\frac{
2\left|\widehat{\mathcal S}\cap\mathcal S^*\right|
}{
\left|\widehat{\mathcal S}\right|+\left|\mathcal S^*\right|
}.
$$

### Real subject-6 PVI evaluation

The selected model was evaluated on 1,600 phases from held-out trials 10 and 11.
The comparison target is the existing PVI Newton reconstruction, not anatomical
ground truth.

**Selected stage 1**

$$
\operatorname{corr}
(\widehat{\Delta\sigma}_{\mathrm{GCN1}},
\Delta\sigma_{\mathrm{PVI}})
=0.604.
$$

The mean physical voltage residual decreases:

$$
2.36\times10^{-5}
\longrightarrow
1.68\times10^{-5}.
$$

**Selected stage 2**

$$
\operatorname{corr}
(\widehat{\Delta\sigma}_{\mathrm{GCN2}},
\Delta\sigma_{\mathrm{PVI}})
=0.169.
$$

The voltage residual increases:

$$
1.68\times10^{-5}
\longrightarrow
2.63\times10^{-5}.
$$

### Main conclusion

> The faithful PVI-GCNM substantially improves structural recovery on synthetic
> nonlinear conductivity truth. However, the second stage does not transfer
> reliably to the real PVI domain. Intermediate representations and voltage
> consistency must therefore remain visible, and anatomical claims require an
> independent phantom or ultrasound reference.

Suggested visual:
[`faithful_output_gallery.pdf`](reports/gcnm_pvi_latex/figures/faithful_output_gallery.pdf)

---

## Optional Slide 5 — Conclusions and Next Steps

### What has been completed

- Subject-6 US120 ring geometry and production inverse reproduced.
- MATLAB/Python production parity verified.
- Clean anatomical synthetic supervision generated.
- Faithful per-stage PVI forward/Jacobian recomputation implemented.
- Iterative-LM, architecture, coordinate, vessel-weight, baseline, and background
  ablations completed.
- Synthetic ground-truth and full real-PVI output comparisons completed.

### Next scientific steps

1. Introduce physics-aware stopping so real inference can stop before stage-2
   voltage consistency deteriorates.
2. Run multiple random seeds and report confidence intervals.
3. Increase anatomical, electrode, contact-impedance, and acquisition-domain
   randomization.
4. Test additional subjects and ring geometries.
5. Validate vessel localization with a physical phantom and co-registered
   ultrasound.

Suggested visual:
[`faithful_real_pvi_gallery.pdf`](reports/gcnm_pvi_latex/figures/faithful_real_pvi_gallery.pdf)
