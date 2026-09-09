"""Stable graph and prediction helpers for vessel-parameter GCNMs."""

from __future__ import annotations

import numpy as np
import torch
from torch_geometric.data import Data


def make_vessel_graphs(
    truth: np.ndarray,
    current: np.ndarray,
    direction: np.ndarray,
    voltage: np.ndarray,
    vessel_parameters: np.ndarray,
    positions: np.ndarray,
    edge_index,
    *,
    conductivity_scale: float,
    voltage_scale: float,
    initial_parameters: np.ndarray | None = None,
    tissue_labels: np.ndarray | None = None,
    voltage_template: np.ndarray | None = None,
    context_direction: np.ndarray | None = None,
    voltage_clean: np.ndarray | None = None,
    beat_id: np.ndarray | None = None,
    voltage_input_mode: str = "global_scale",
    voltage_rms_reference: float | None = None,
) -> list[Data]:
    """Build the exact graph contract used by vessel-localizer checkpoints."""

    graphs = []
    for index in range(len(truth)):
        target = truth[index] / conductivity_scale
        if voltage_input_mode == "beat_normalized":
            template = (
                voltage[index]
                if voltage_template is None
                else voltage_template[index]
            )
            phase_rms = max(float(np.sqrt(np.mean(voltage[index] ** 2))), 1e-12)
            template_rms = max(float(np.sqrt(np.mean(template**2))), 1e-12)
            reference = max(float(voltage_rms_reference or 1.0), 1e-12)
            normalized_voltage = np.clip(voltage[index] / phase_rms, -8.0, 8.0)
            normalized_template = np.clip(template / template_rms, -8.0, 8.0)
            log_rms = float(np.clip(np.log(phase_rms / reference), -5.0, 5.0))
            context = (
                direction[index]
                if context_direction is None
                else context_direction[index]
            )
            context_scale = max(
                float(np.percentile(np.abs(context), 95.0)), 1e-12
            )
            normalized_context = np.clip(np.abs(context) / context_scale, 0.0, 5.0)
            features = np.column_stack(
                (
                    current[index] / conductivity_scale,
                    direction[index] / conductivity_scale,
                    normalized_context,
                    positions[:, 0],
                    positions[:, 1],
                )
            )
            geometry_features = np.column_stack(
                (normalized_context, positions[:, 0], positions[:, 1])
            )
        else:
            normalized_voltage = voltage[index] / voltage_scale
            normalized_template = normalized_voltage
            log_rms = 0.0
            features = np.column_stack(
                (
                    current[index] / conductivity_scale,
                    direction[index] / conductivity_scale,
                    positions[:, 0],
                    positions[:, 1],
                )
            )
            geometry_features = np.column_stack(
                (np.abs(direction[index]) / conductivity_scale, positions)
            )
        graph = Data(
            x=torch.tensor(features, dtype=torch.float32),
            geometry_x=torch.tensor(geometry_features, dtype=torch.float32),
            edge_index=edge_index,
            y=torch.tensor(target[:, None], dtype=torch.float32),
            voltage=torch.tensor(normalized_voltage[None, :], dtype=torch.float32),
            voltage_template=torch.tensor(
                normalized_template[None, :], dtype=torch.float32
            ),
            voltage_log_rms=torch.tensor([[log_rms]], dtype=torch.float32),
            voltage_clean=torch.tensor(
                (
                    voltage[index]
                    if voltage_clean is None
                    else voltage_clean[index]
                )[None, :],
                dtype=torch.float32,
            ),
            beat_id=torch.tensor(
                [index if beat_id is None else int(beat_id[index])],
                dtype=torch.long,
            ),
            vessel_parameters=torch.tensor(
                vessel_parameters[index][None, :, :], dtype=torch.float32
            ),
            muscle_gate=torch.tensor(
                (
                    (tissue_labels[index] == 3).astype(np.float32)
                    if tissue_labels is not None
                    else np.ones(len(target), dtype=np.float32)
                )[:, None],
                dtype=torch.float32,
            ),
        )
        if initial_parameters is not None:
            graph.initial_parameters = torch.tensor(
                initial_parameters[index][None, :, :], dtype=torch.float32
            )
        graphs.append(graph)
    return graphs


def predict_vessel_graphs(model, graphs, scale: float) -> np.ndarray:
    """Run frame-wise vessel GCN inference."""

    model.eval()
    device = next(model.parameters()).device
    predictions = []
    with torch.no_grad():
        for graph in graphs:
            predictions.append(model(graph.to(device)).cpu().numpy().ravel() * scale)
    return np.stack(predictions)


def predict_vessel_graphs_with_parameters(
    model, graphs, scale: float
) -> tuple[np.ndarray, np.ndarray]:
    """Run vessel GCN inference and return its explicit vessel parameters."""

    model.eval()
    device = next(model.parameters()).device
    predictions, parameters = [], []
    with torch.no_grad():
        for graph in graphs:
            output = model(graph.to(device), return_parameters=True)
            predictions.append(output[0].cpu().numpy().ravel() * scale)
            parameters.append(output[1].cpu().numpy()[0])
    return np.stack(predictions), np.stack(parameters)
