"""Beat-normalized, signed two-vessel models for cross-subject PVI transfer.

The models factor the input voltage into a scale-free spatial pattern and a
single log-RMS amplitude.  Beat-template features determine vessel geometry;
the current phase determines signed conductivity amplitude.  This prevents a
small real voltage vector from being interpreted as "no vessel" merely because
the synthetic training pack used a larger global voltage scale.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_add_pool, global_mean_pool
from torch_geometric.utils import softmax

from gcnm_pvi.voltage_vessel_model import (
    rasterize_diffusion_slots,
    rasterize_vessel_slots,
)


def _batch_index(data) -> torch.Tensor:
    batch = getattr(data, "batch", None)
    if batch is None:
        batch = torch.zeros(
            data.x.shape[0], dtype=torch.long, device=data.x.device
        )
    return batch


class BeatVoltageContext(torch.nn.Module):
    """Encode phase shape/amplitude separately from the beat template."""

    def __init__(self, measurements: int = 32, latent: int = 64) -> None:
        super().__init__()
        self.phase = torch.nn.Sequential(
            torch.nn.Linear(measurements + 1, 128),
            torch.nn.LayerNorm(128),
            torch.nn.GELU(),
            torch.nn.Linear(128, latent),
        )
        self.template = torch.nn.Sequential(
            torch.nn.Linear(measurements, 128),
            torch.nn.LayerNorm(128),
            torch.nn.GELU(),
            torch.nn.Linear(128, latent),
        )

    def forward(self, data) -> tuple[torch.Tensor, torch.Tensor]:
        voltage = data.voltage if data.voltage.ndim > 1 else data.voltage[None, :]
        template = (
            data.voltage_template
            if data.voltage_template.ndim > 1
            else data.voltage_template[None, :]
        )
        log_rms = (
            data.voltage_log_rms
            if data.voltage_log_rms.ndim > 1
            else data.voltage_log_rms.reshape(1, 1)
        )
        phase = self.phase(torch.cat((voltage, log_rms), dim=1))
        geometry = self.template(template)
        return phase, geometry


def _decode_geometry(
    raw: torch.Tensor,
    *,
    minimum_axis: float,
    maximum_axis: float,
) -> torch.Tensor:
    center = 0.70 * torch.tanh(raw[..., 0:2])
    axes = float(minimum_axis) + (
        float(maximum_axis) - float(minimum_axis)
    ) * torch.sigmoid(raw[..., 2:4])
    angle = math.pi * torch.tanh(raw[..., 4:5])
    return torch.cat((center, axes, angle), dim=-1)


def _signed_amplitude(raw: torch.Tensor) -> torch.Tensor:
    # Normalized conductivity targets are normally within +/-1.  The modest
    # extra headroom covers augmented high-amplitude beats without clipping.
    return 1.5 * torch.tanh(raw)


class BeatMeanPoolVesselLocalizer(torch.nn.Module):
    """Two signed ellipses: beat template for geometry, phase for amplitude."""

    def __init__(
        self,
        *,
        hidden: int = 64,
        slots: int = 2,
        minimum_axis: float = 0.025,
        maximum_axis: float = 0.23,
        raster_temperature: float = 0.05,
    ) -> None:
        super().__init__()
        if slots != 2:
            raise ValueError("the vascular contract requires exactly two slots")
        self.slots = slots
        self.minimum_axis = float(minimum_axis)
        self.maximum_axis = float(maximum_axis)
        self.raster_temperature = float(raster_temperature)
        self.voltage_context = BeatVoltageContext()
        self.node_projection = torch.nn.Linear(3, hidden)
        self.conv1 = GCNConv(hidden, hidden)
        self.conv2 = GCNConv(hidden, hidden)
        self.norm1 = torch.nn.LayerNorm(hidden)
        self.norm2 = torch.nn.LayerNorm(hidden)
        self.geometry_head = torch.nn.Sequential(
            torch.nn.Linear(hidden + 64, 128),
            torch.nn.GELU(),
            torch.nn.Linear(128, slots * 5),
        )
        self.amplitude_head = torch.nn.Sequential(
            torch.nn.Linear(hidden + 128, 128),
            torch.nn.GELU(),
            torch.nn.Linear(128, slots),
        )

    def _geometry_embedding(self, data) -> tuple[torch.Tensor, torch.Tensor]:
        batch = _batch_index(data)
        h0 = F.gelu(self.node_projection(data.geometry_x))
        h1 = h0 + F.gelu(self.norm1(self.conv1(h0, data.edge_index)))
        h2 = h1 + F.gelu(self.norm2(self.conv2(h1, data.edge_index)))
        return global_mean_pool(h2, batch), batch

    def forward(self, data, *, return_parameters: bool = False):
        phase_latent, template_latent = self.voltage_context(data)
        pooled, batch = self._geometry_embedding(data)
        base = torch.cat((pooled, template_latent), dim=1)
        geometry = _decode_geometry(
            self.geometry_head(base).reshape(-1, self.slots, 5),
            minimum_axis=self.minimum_axis,
            maximum_axis=self.maximum_axis,
        )
        amplitude = _signed_amplitude(
            self.amplitude_head(torch.cat((base, phase_latent), dim=1))
            .reshape(-1, self.slots, 1)
        )
        parameters = torch.cat((geometry, amplitude), dim=-1)
        prediction, masks = rasterize_vessel_slots(
            data.x[:, -2:],
            batch,
            parameters,
            temperature=self.raster_temperature,
        )
        if return_parameters:
            return prediction, parameters, masks
        return prediction


class BeatSpatialDiffusionLocalizer(torch.nn.Module):
    """Two signed artery cores with positive muscle-gated diffusion halos."""

    def __init__(
        self,
        *,
        hidden: int = 64,
        slots: int = 2,
        minimum_axis: float = 0.025,
        maximum_axis: float = 0.23,
        minimum_diffusion_length: float = 0.03,
        maximum_diffusion_length: float = 0.35,
        raster_temperature: float = 0.05,
    ) -> None:
        super().__init__()
        if slots != 2:
            raise ValueError("the vascular contract requires exactly two slots")
        self.hidden = int(hidden)
        self.slots = int(slots)
        self.minimum_axis = float(minimum_axis)
        self.maximum_axis = float(maximum_axis)
        self.minimum_diffusion_length = float(minimum_diffusion_length)
        self.maximum_diffusion_length = float(maximum_diffusion_length)
        self.raster_temperature = float(raster_temperature)
        self.voltage_context = BeatVoltageContext()
        self.node_projection = torch.nn.Linear(3, hidden)
        self.conv1 = GCNConv(hidden, hidden)
        self.conv2 = GCNConv(hidden, hidden)
        self.norm1 = torch.nn.LayerNorm(hidden)
        self.norm2 = torch.nn.LayerNorm(hidden)
        self.key_projection = torch.nn.Linear(hidden, hidden)
        self.template_query = torch.nn.Linear(64, hidden)
        self.slot_queries = torch.nn.Parameter(torch.empty(slots, hidden))
        geometry_features = hidden + 64 + 2 + hidden
        self.geometry_head = torch.nn.Sequential(
            torch.nn.Linear(geometry_features, 128),
            torch.nn.GELU(),
            torch.nn.Linear(128, 7),
        )
        self.amplitude_head = torch.nn.Sequential(
            torch.nn.Linear(geometry_features + 64, 128),
            torch.nn.GELU(),
            torch.nn.Linear(128, 1),
        )
        torch.nn.init.normal_(self.slot_queries, std=0.02)

    def forward(self, data, *, return_parameters: bool = False):
        batch = _batch_index(data)
        phase_latent, template_latent = self.voltage_context(data)
        h0 = F.gelu(self.node_projection(data.geometry_x))
        h1 = h0 + F.gelu(self.norm1(self.conv1(h0, data.edge_index)))
        h2 = h1 + F.gelu(self.norm2(self.conv2(h1, data.edge_index)))
        keys = self.key_projection(h2)
        base_query = self.template_query(template_latent)
        coordinates = data.x[:, -2:]
        parameters, attention = [], []
        for slot in range(self.slots):
            slot_query = self.slot_queries[slot][None, :].expand(
                len(template_latent), -1
            )
            query = base_query + slot_query
            logits = torch.sum(keys * query[batch], dim=1) / math.sqrt(self.hidden)
            weights = softmax(logits, batch)
            pooled = global_add_pool(weights[:, None] * h2, batch)
            center_hint = global_add_pool(weights[:, None] * coordinates, batch)
            base = torch.cat(
                (pooled, template_latent, center_hint, slot_query), dim=1
            )
            raw = self.geometry_head(base)
            geometry = _decode_geometry(
                raw[:, 0:5],
                minimum_axis=self.minimum_axis,
                maximum_axis=self.maximum_axis,
            )
            diffusion_fraction = 0.5 * torch.sigmoid(raw[:, 5:6])
            diffusion_length = self.minimum_diffusion_length + (
                self.maximum_diffusion_length - self.minimum_diffusion_length
            ) * torch.sigmoid(raw[:, 6:7])
            amplitude = _signed_amplitude(
                self.amplitude_head(torch.cat((base, phase_latent), dim=1))
            )
            parameters.append(
                torch.cat(
                    (geometry, amplitude, diffusion_fraction, diffusion_length),
                    dim=1,
                )
            )
            attention.append(weights)
        decoded = torch.stack(parameters, dim=1)
        prediction, core_masks, _halo_masks = rasterize_diffusion_slots(
            coordinates,
            batch,
            decoded,
            data.muscle_gate,
            temperature=self.raster_temperature,
        )
        if return_parameters:
            return prediction, decoded, core_masks, torch.stack(attention, dim=1)
        return prediction


class BeatVesselParameterRefiner(torch.nn.Module):
    """Stage-2 bounded slot refinement; never adds the raw LM image."""

    def __init__(
        self,
        *,
        diffusion: bool,
        hidden: int = 64,
        slots: int = 2,
        minimum_axis: float = 0.025,
        maximum_axis: float = 0.23,
        minimum_diffusion_length: float = 0.03,
        maximum_diffusion_length: float = 0.35,
        raster_temperature: float = 0.05,
    ) -> None:
        super().__init__()
        self.diffusion = bool(diffusion)
        self.hidden = int(hidden)
        self.slots = int(slots)
        self.parameter_count = 8 if diffusion else 6
        self.minimum_axis = float(minimum_axis)
        self.maximum_axis = float(maximum_axis)
        self.minimum_diffusion_length = float(minimum_diffusion_length)
        self.maximum_diffusion_length = float(maximum_diffusion_length)
        self.raster_temperature = float(raster_temperature)
        self.voltage_context = BeatVoltageContext()
        self.node_projection = torch.nn.Linear(5, hidden)
        self.conv1 = GCNConv(hidden, hidden)
        self.conv2 = GCNConv(hidden, hidden)
        self.norm1 = torch.nn.LayerNorm(hidden)
        self.norm2 = torch.nn.LayerNorm(hidden)
        self.key_projection = torch.nn.Linear(hidden, hidden)
        self.template_query = torch.nn.Linear(64, hidden)
        self.phase_query = torch.nn.Linear(64, hidden)
        self.parameter_query = torch.nn.Linear(self.parameter_count, hidden)
        self.slot_queries = torch.nn.Parameter(torch.empty(slots, hidden))
        features = hidden + 128 + 2 + self.parameter_count + hidden
        self.delta_head = torch.nn.Sequential(
            torch.nn.Linear(features, 128),
            torch.nn.GELU(),
            torch.nn.Linear(128, self.parameter_count),
        )
        torch.nn.init.normal_(self.slot_queries, std=0.02)
        torch.nn.init.zeros_(self.delta_head[-1].weight)
        torch.nn.init.zeros_(self.delta_head[-1].bias)

    def _apply_delta(
        self, initial: torch.Tensor, raw: torch.Tensor
    ) -> torch.Tensor:
        # Geometry is a beat property.  Stage 2 may correct it, but cannot move
        # a vessel across the finger in response to one noisy phase.
        center = torch.clamp(
            initial[..., 0:2] + 0.05 * torch.tanh(raw[..., 0:2]),
            min=-0.70,
            max=0.70,
        )
        axes = torch.clamp(
            initial[..., 2:4] + 0.03 * torch.tanh(raw[..., 2:4]),
            min=self.minimum_axis,
            max=self.maximum_axis,
        )
        angle_unwrapped = initial[..., 4:5] + 0.30 * torch.tanh(raw[..., 4:5])
        angle = torch.atan2(torch.sin(angle_unwrapped), torch.cos(angle_unwrapped))
        amplitude = torch.clamp(
            initial[..., 5:6] + 0.60 * torch.tanh(raw[..., 5:6]),
            min=-1.5,
            max=1.5,
        )
        output = [center, axes, angle, amplitude]
        if self.diffusion:
            fraction = torch.clamp(
                initial[..., 6:7] + 0.10 * torch.tanh(raw[..., 6:7]),
                min=0.0,
                max=0.5,
            )
            length = torch.clamp(
                initial[..., 7:8] + 0.04 * torch.tanh(raw[..., 7:8]),
                min=self.minimum_diffusion_length,
                max=self.maximum_diffusion_length,
            )
            output.extend((fraction, length))
        return torch.cat(output, dim=-1)

    def forward(self, data, *, return_parameters: bool = False):
        batch = _batch_index(data)
        initial = data.initial_parameters.reshape(
            -1, self.slots, self.parameter_count
        )
        phase_latent, template_latent = self.voltage_context(data)
        h0 = F.gelu(self.node_projection(data.x))
        h1 = h0 + F.gelu(self.norm1(self.conv1(h0, data.edge_index)))
        h2 = h1 + F.gelu(self.norm2(self.conv2(h1, data.edge_index)))
        keys = self.key_projection(h2)
        base_query = self.template_query(template_latent) + self.phase_query(
            phase_latent
        )
        coordinates = data.x[:, -2:]
        deltas, attention = [], []
        for slot in range(self.slots):
            slot_query = self.slot_queries[slot][None, :].expand(
                len(template_latent), -1
            )
            query = (
                base_query
                + self.parameter_query(initial[:, slot])
                + slot_query
            )
            logits = torch.sum(keys * query[batch], dim=1) / math.sqrt(self.hidden)
            weights = softmax(logits, batch)
            pooled = global_add_pool(weights[:, None] * h2, batch)
            center_hint = global_add_pool(weights[:, None] * coordinates, batch)
            features = torch.cat(
                (
                    pooled,
                    template_latent,
                    phase_latent,
                    center_hint,
                    initial[:, slot],
                    slot_query,
                ),
                dim=1,
            )
            deltas.append(self.delta_head(features))
            attention.append(weights)
        parameters = self._apply_delta(initial, torch.stack(deltas, dim=1))
        if self.diffusion:
            prediction, masks, _halo = rasterize_diffusion_slots(
                coordinates,
                batch,
                parameters,
                data.muscle_gate,
                temperature=self.raster_temperature,
            )
        else:
            prediction, masks = rasterize_vessel_slots(
                coordinates,
                batch,
                parameters,
                temperature=self.raster_temperature,
            )
        if return_parameters:
            return prediction, parameters, masks, torch.stack(attention, dim=1)
        return prediction
