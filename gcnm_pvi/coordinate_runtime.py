"""Stable graph/model helpers shared by GCNM training and inference.

This module intentionally contains no training loop or command-line entrypoint.
Deployable reconstruction code can therefore load coordinate GCNM checkpoints
without importing :mod:`gcnm_pvi.train_faithful_gcnm`.
"""

from __future__ import annotations

import numpy as np
import torch
from torch_geometric.data import Data

from gcnm_pvi.gcnm_model import (
    GCNBlock,
    PhysicsProposalResidualGCNBlock,
    PositiveOutput,
    ShallowPhysicsResidualGCNBlock,
    VoltageConditionedGCNBlock,
)


def make_coordinate_graphs(
    truth: np.ndarray,
    current: np.ndarray,
    direction: np.ndarray,
    positions: np.ndarray,
    edge_index,
    *,
    scale: float,
    use_coordinates: bool,
    positive_weight: float,
    voltage: np.ndarray | None = None,
    voltage_scale: float = 1.0,
    reference: np.ndarray | None = None,
) -> list[Data]:
    """Build the node-feature graphs used by faithful coordinate GCNMs."""

    dataset: list[Data] = []
    for index in range(len(truth)):
        target = truth[index] / scale
        feature_columns = [current[index] / scale, direction[index] / scale]
        if use_coordinates:
            feature_columns.extend(positions.T)
        features = np.column_stack(feature_columns)
        normalized_reference = None if reference is None else reference[index] / scale
        supervised_signal = (
            target if normalized_reference is None else target - normalized_reference
        )
        magnitude = np.abs(supervised_signal)
        nonzero = magnitude[magnitude > 0]
        support_reference = float(np.median(nonzero)) if nonzero.size else 1.0
        weights = 1.0 + positive_weight * np.clip(
            magnitude / max(support_reference, 1e-8), 0.0, 2.0
        )
        graph = Data(
            edge_index=edge_index,
            x=torch.tensor(features, dtype=torch.float32),
            y=torch.tensor(target[:, None], dtype=torch.float32),
            weights=torch.tensor(weights[:, None], dtype=torch.float32),
            background=torch.tensor((magnitude <= 1e-8)[:, None]),
        )
        if normalized_reference is not None:
            graph.reference = torch.tensor(
                normalized_reference[:, None], dtype=torch.float32
            )
        if voltage is not None:
            graph.voltage = torch.tensor(
                voltage[index][None, :] / max(float(voltage_scale), 1e-12),
                dtype=torch.float32,
            )
        dataset.append(graph)
    return dataset


def build_coordinate_model(
    mode: str,
    channels: list[int],
    in_channels: int,
    *,
    use_voltage_mlp: bool = False,
    measurements: int = 32,
    voltage_latent: int = 64,
):
    """Construct the exact coordinate GCN architecture stored in checkpoints."""

    if use_voltage_mlp:
        if mode not in {"direct", "positive_direct", "proposal_residual"}:
            raise ValueError(
                "the voltage-MLP front end supports direct or proposal-residual output"
            )
        model = VoltageConditionedGCNBlock(
            channels,
            node_features=in_channels,
            measurements=measurements,
            voltage_latent=voltage_latent,
            residual_around_proposal=mode == "proposal_residual",
        ).float()
        return PositiveOutput(model).float() if mode == "positive_direct" else model
    if mode in {"direct", "positive_direct"}:
        model = GCNBlock(channels, in_channels=in_channels).float()
        return PositiveOutput(model).float() if mode == "positive_direct" else model
    if mode == "proposal_residual":
        return PhysicsProposalResidualGCNBlock(
            channels,
            in_channels=in_channels,
        ).float()
    if mode == "shallow_residual":
        return ShallowPhysicsResidualGCNBlock(
            channels,
            in_channels=in_channels,
        ).float()
    raise ValueError(f"unknown output mode {mode}")


def predict_coordinate_graphs(model, graphs: list[Data], scale: float) -> np.ndarray:
    """Run frame-wise coordinate GCN inference in the model's current device."""

    device = next(model.parameters()).device
    model.eval()
    output = []
    with torch.no_grad():
        for sample in graphs:
            output.append(model(sample.to(device)).squeeze().cpu().numpy() * scale)
    return np.stack(output)
