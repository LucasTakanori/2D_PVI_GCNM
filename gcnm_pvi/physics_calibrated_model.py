"""Direction-anchored core-guided GCNM with voltage-space amplitude fitting.

The learned graph network predicts a spatial *shape* around the current LM
direction.  A one-dimensional least-squares solve then determines the physical
conductivity amplitude that best explains the measured differential voltage.
This separates localization from amplitude and removes the synthetic target
scale as the only source of real-data activation.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from gcnm_pvi.core_guided_model import CoreGuidedGCNMStage


class PhysicsCalibratedCoreStage(CoreGuidedGCNMStage):
    """Core-guided dense shape estimator anchored to the LM direction.

    The inherited output head starts at zero.  Consequently a new stage starts
    from its physics direction rather than an empty image, while the GCN learns
    a bounded shape correction.  The returned tensor is dimensionless and is
    converted to a physical map by :func:`calibrate_shape_batch`.
    """

    def __init__(self, *, correction_limit: float = 1.0) -> None:
        super().__init__(
            node_features=6,
            hidden=64,
            residual=False,
            correction_limit=correction_limit,
        )

    def forward(self, data, *, return_aux: bool = False):
        output = super().forward(data, return_aux=return_aux)
        if return_aux:
            learned, core_logits, attention = output
            return data.x[:, 1:2] + learned, core_logits, attention
        return data.x[:, 1:2] + output


@dataclass
class CalibratedBatch:
    """Differentiable voltage calibration result for a PyG graph batch."""

    prediction: torch.Tensor
    correction: torch.Tensor
    amplitude: torch.Tensor
    predicted_voltage: torch.Tensor
    residual_voltage: torch.Tensor
    residual_rms: torch.Tensor
    target_rms: torch.Tensor
    relative_mse: torch.Tensor


def calibrate_shape_batch(
    raw_shape: torch.Tensor,
    batch,
    *,
    conductivity_scale: float,
    maximum_amplitude: float,
    voltage_floor: float = 1e-8,
) -> CalibratedBatch:
    """Fit one non-negative physical amplitude per graph.

    Each graph must provide:

    ``jacobian``
        Shape ``(1, measurements, elements)`` before batching.
    ``voltage_target``
        The remaining voltage that this stage should explain.
    ``current_physical``
        The accumulated differential conductivity before this stage.

    For unit-RMS shape ``q`` the fitted amplitude is

    ``a = clamp((Jq)^T v / (||Jq||^2 + eps), 0, a_max)``.
    """

    graphs = int(batch.num_graphs)
    if graphs <= 0 or raw_shape.numel() % graphs:
        raise ValueError("raw shape cannot be divided into equal graph maps")
    elements = raw_shape.numel() // graphs
    raw_physical = raw_shape.reshape(graphs, elements) * float(conductivity_scale)
    shape_rms = torch.sqrt(torch.mean(raw_physical**2, dim=1, keepdim=True))
    unit_shape = raw_physical / shape_rms.clamp_min(1e-8)

    jacobian = batch.jacobian.reshape(graphs, -1, elements).to(
        dtype=raw_shape.dtype
    )
    voltage_target = batch.voltage_target.reshape(graphs, -1).to(
        dtype=raw_shape.dtype
    )
    current = batch.current_physical.reshape(graphs, elements).to(
        dtype=raw_shape.dtype
    )
    unit_voltage = torch.bmm(jacobian, unit_shape.unsqueeze(-1)).squeeze(-1)
    numerator = torch.sum(unit_voltage * voltage_target, dim=1)
    denominator = torch.sum(unit_voltage**2, dim=1).clamp_min(1e-20)
    amplitude = torch.clamp(
        numerator / denominator,
        min=0.0,
        max=float(maximum_amplitude),
    )
    correction = amplitude[:, None] * unit_shape
    predicted_voltage = amplitude[:, None] * unit_voltage
    residual_voltage = predicted_voltage - voltage_target
    residual_rms = torch.sqrt(torch.mean(residual_voltage**2, dim=1))
    target_power = torch.mean(voltage_target**2, dim=1)
    target_rms = torch.sqrt(target_power)
    relative_mse = torch.mean(residual_voltage**2, dim=1) / (
        target_power + float(voltage_floor) ** 2
    )
    prediction = current + correction
    return CalibratedBatch(
        prediction=prediction.reshape(-1, 1),
        correction=correction.reshape(-1, 1),
        amplitude=amplitude,
        predicted_voltage=predicted_voltage,
        residual_voltage=residual_voltage,
        residual_rms=residual_rms,
        target_rms=target_rms,
        relative_mse=relative_mse,
    )
