"""Voltage-conditioned two-vessel localizer and physics residual refiner."""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_add_pool, global_mean_pool
from torch_geometric.utils import softmax


class VoltageEncoder(torch.nn.Module):
    """Encode the complete differential-voltage vector into global context."""

    def __init__(self, measurements: int = 32, hidden: int = 128, latent: int = 64):
        super().__init__()
        self.network = torch.nn.Sequential(
            torch.nn.Linear(measurements, hidden),
            torch.nn.LayerNorm(hidden),
            torch.nn.GELU(),
            torch.nn.Linear(hidden, hidden),
            torch.nn.GELU(),
            torch.nn.Linear(hidden, latent),
        )

    def forward(self, voltage: torch.Tensor) -> torch.Tensor:
        return self.network(voltage)


def rasterize_vessel_slots(
    coordinates: torch.Tensor,
    graph_index: torch.Tensor,
    parameters: torch.Tensor,
    *,
    temperature: float = 0.05,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Rasterize two smooth positive ellipses onto batched graph elements.

    Parameters are ``(x, y, axis_a, axis_b, angle, amplitude)`` in normalized
    mesh coordinates and normalized conductivity units.
    """

    per_node = parameters[graph_index]
    dx = coordinates[:, None, 0] - per_node[:, :, 0]
    dy = coordinates[:, None, 1] - per_node[:, :, 1]
    angle = per_node[:, :, 4]
    cosine, sine = torch.cos(angle), torch.sin(angle)
    local_x = cosine * dx + sine * dy
    local_y = -sine * dx + cosine * dy
    distance_squared = (
        (local_x / per_node[:, :, 2].clamp_min(1e-4)) ** 2
        + (local_y / per_node[:, :, 3].clamp_min(1e-4)) ** 2
    )
    masks = torch.sigmoid((1.0 - distance_squared) / float(temperature))
    conductivity = torch.sum(masks * per_node[:, :, 5], dim=1, keepdim=True)
    return conductivity, masks


def rasterize_diffusion_slots(
    coordinates: torch.Tensor,
    graph_index: torch.Tensor,
    parameters: torch.Tensor,
    muscle_gate: torch.Tensor,
    *,
    temperature: float = 0.05,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Rasterize two artery cores plus tissue-gated Gaussian diffusion halos.

    Parameters are ``(x, y, axis_a, axis_b, angle, amplitude,
    diffusion_fraction, diffusion_length)`` in normalized mesh units.
    """

    per_node = parameters[graph_index]
    dx = coordinates[:, None, 0] - per_node[:, :, 0]
    dy = coordinates[:, None, 1] - per_node[:, :, 1]
    angle = per_node[:, :, 4]
    cosine, sine = torch.cos(angle), torch.sin(angle)
    local_x = cosine * dx + sine * dy
    local_y = -sine * dx + cosine * dy
    axis_a = per_node[:, :, 2].clamp_min(1e-4)
    axis_b = per_node[:, :, 3].clamp_min(1e-4)
    radius = torch.sqrt(
        (local_x / axis_a) ** 2 + (local_y / axis_b) ** 2 + 1e-12
    )
    core_masks = torch.sigmoid((1.0 - radius**2) / float(temperature))
    effective_axis = torch.sqrt(axis_a * axis_b)
    distance_from_wall = torch.relu(radius - 1.0) * effective_axis
    diffusion_length = per_node[:, :, 7].clamp_min(1e-4)
    halos = torch.exp(-0.5 * (distance_from_wall / diffusion_length) ** 2)
    gate = muscle_gate.reshape(-1, 1).clamp(0.0, 1.0)
    halo_masks = (1.0 - core_masks) * halos * gate
    amplitude = per_node[:, :, 5]
    diffusion_fraction = per_node[:, :, 6]
    conductivity = torch.sum(
        amplitude * (core_masks + diffusion_fraction * halo_masks),
        dim=1,
        keepdim=True,
    )
    return conductivity, core_masks, halo_masks


def decode_diffusion_parameters(
    raw: torch.Tensor,
    *,
    minimum_axis: float = 0.05,
    maximum_axis: float = 0.23,
    minimum_diffusion_length: float = 0.03,
    maximum_diffusion_length: float = 0.35,
) -> torch.Tensor:
    """Decode bounded two-vessel geometry, amplitude, and halo parameters."""

    vessel = VoltageConditionedVesselLocalizer.decode_parameters(
        raw[..., :6], minimum_axis, maximum_axis
    )
    diffusion_fraction = 0.5 * torch.sigmoid(raw[..., 6:7])
    diffusion_length = float(minimum_diffusion_length) + (
        float(maximum_diffusion_length) - float(minimum_diffusion_length)
    ) * torch.sigmoid(raw[..., 7:8])
    return torch.cat((vessel, diffusion_fraction, diffusion_length), dim=-1)


class VoltageConditionedVesselLocalizer(torch.nn.Module):
    """Predict two explicit vascular ellipses from voltage and physics features."""

    def __init__(
        self,
        *,
        measurements: int = 32,
        node_features: int = 4,
        hidden: int = 64,
        voltage_latent: int = 64,
        slots: int = 2,
        raster_temperature: float = 0.05,
        minimum_axis: float = 0.05,
        maximum_axis: float = 0.23,
    ):
        super().__init__()
        if slots != 2:
            raise ValueError("the current vascular contract requires exactly two slots")
        self.slots = slots
        self.raster_temperature = float(raster_temperature)
        self.minimum_axis = float(minimum_axis)
        self.maximum_axis = float(maximum_axis)
        self.voltage_encoder = VoltageEncoder(measurements, 128, voltage_latent)
        self.node_projection = torch.nn.Linear(node_features, hidden)
        self.conv1 = GCNConv(hidden, hidden)
        self.conv2 = GCNConv(hidden, hidden)
        self.norm1 = torch.nn.LayerNorm(hidden)
        self.norm2 = torch.nn.LayerNorm(hidden)
        self.context = torch.nn.Sequential(
            torch.nn.Linear(hidden + voltage_latent, 128),
            torch.nn.GELU(),
            torch.nn.Linear(128, 128),
            torch.nn.GELU(),
        )
        self.slot_head = torch.nn.Linear(128, slots * 6)

    @staticmethod
    def decode_parameters(
        raw: torch.Tensor,
        minimum_axis: float = 0.05,
        maximum_axis: float = 0.23,
    ) -> torch.Tensor:
        center = 0.65 * torch.tanh(raw[..., 0:2])
        axes = float(minimum_axis) + (
            float(maximum_axis) - float(minimum_axis)
        ) * torch.sigmoid(raw[..., 2:4])
        angle = math.pi * torch.tanh(raw[..., 4:5])
        amplitude = 1.5 * torch.sigmoid(raw[..., 5:6])
        return torch.cat((center, axes, angle, amplitude), dim=-1)

    def forward(self, data, *, return_parameters: bool = False):
        batch = getattr(data, "batch", None)
        if batch is None:
            batch = torch.zeros(
                data.x.shape[0], dtype=torch.long, device=data.x.device
            )
        voltage = data.voltage
        if voltage.ndim == 1:
            voltage = voltage[None, :]
        z_voltage = self.voltage_encoder(voltage)
        h0 = F.gelu(self.node_projection(data.x))
        h1 = h0 + F.gelu(self.norm1(self.conv1(h0, data.edge_index)))
        h2 = h1 + F.gelu(self.norm2(self.conv2(h1, data.edge_index)))
        pooled = global_mean_pool(h2, batch)
        context = self.context(torch.cat((pooled, z_voltage), dim=1))
        raw_parameters = self.slot_head(context).reshape(-1, self.slots, 6)
        parameters = self.decode_parameters(
            raw_parameters, self.minimum_axis, self.maximum_axis
        )
        conductivity, masks = rasterize_vessel_slots(
            data.x[:, 2:4],
            batch,
            parameters,
            temperature=self.raster_temperature,
        )
        if return_parameters:
            return conductivity, parameters, masks
        return conductivity


class VoltageConditionedPhysicsRefiner(torch.nn.Module):
    """A shallow positive residual correction conditioned on all 32 voltages."""

    def __init__(
        self,
        *,
        measurements: int = 32,
        node_features: int = 4,
        hidden: int = 64,
        voltage_latent: int = 64,
        correction_limit: float = 0.5,
    ):
        super().__init__()
        self.correction_limit = float(correction_limit)
        self.voltage_encoder = VoltageEncoder(measurements, 128, voltage_latent)
        self.node_projection = torch.nn.Linear(
            node_features + voltage_latent, hidden
        )
        self.conv1 = GCNConv(hidden, hidden)
        self.conv2 = GCNConv(hidden, hidden)
        self.norm1 = torch.nn.LayerNorm(hidden)
        self.norm2 = torch.nn.LayerNorm(hidden)
        self.output_head = torch.nn.Linear(node_features + voltage_latent + 3 * hidden, 1)
        torch.nn.init.zeros_(self.output_head.weight)
        torch.nn.init.zeros_(self.output_head.bias)

    def forward(self, data):
        batch = getattr(data, "batch", None)
        if batch is None:
            batch = torch.zeros(
                data.x.shape[0], dtype=torch.long, device=data.x.device
            )
        voltage = data.voltage
        if voltage.ndim == 1:
            voltage = voltage[None, :]
        z_voltage = self.voltage_encoder(voltage)
        node_voltage = z_voltage[batch]
        raw = torch.cat((data.x, node_voltage), dim=1)
        h0 = F.gelu(self.node_projection(raw))
        h1 = h0 + F.gelu(self.norm1(self.conv1(h0, data.edge_index)))
        h2 = h1 + F.gelu(self.norm2(self.conv2(h1, data.edge_index)))
        features = torch.cat((raw, h0, h1, h2), dim=1)
        correction = self.correction_limit * torch.tanh(self.output_head(features))
        # Vascular pulse targets are non-negative.  The physics direction is an
        # input, not an automatically added signed lobe.
        return torch.clamp(data.x[:, 0:1] + correction, min=0.0)


class SpatialAttentionVesselLocalizer(torch.nn.Module):
    """Use one learned spatial attention distribution for each vessel slot.

    Unlike global mean pooling, the two attention pools retain where the graph
    response occurred.  The resulting slots are still rendered as positive
    ellipses, so the anatomical output contract remains explicit.
    """

    def __init__(
        self,
        *,
        measurements: int = 32,
        node_features: int = 4,
        hidden: int = 64,
        voltage_latent: int = 64,
        slots: int = 2,
        raster_temperature: float = 0.05,
        minimum_axis: float = 0.05,
        maximum_axis: float = 0.23,
    ):
        super().__init__()
        if slots != 2:
            raise ValueError("the current vascular contract requires exactly two slots")
        self.hidden = int(hidden)
        self.slots = int(slots)
        self.raster_temperature = float(raster_temperature)
        self.minimum_axis = float(minimum_axis)
        self.maximum_axis = float(maximum_axis)
        self.voltage_encoder = VoltageEncoder(measurements, 128, voltage_latent)
        self.node_projection = torch.nn.Linear(node_features, hidden)
        self.conv1 = GCNConv(hidden, hidden)
        self.conv2 = GCNConv(hidden, hidden)
        self.norm1 = torch.nn.LayerNorm(hidden)
        self.norm2 = torch.nn.LayerNorm(hidden)
        self.key_projection = torch.nn.Linear(hidden, hidden)
        self.voltage_query = torch.nn.Linear(voltage_latent, hidden)
        self.slot_queries = torch.nn.Parameter(torch.empty(slots, hidden))
        context_features = hidden + voltage_latent + 2 + hidden
        self.slot_head = torch.nn.Sequential(
            torch.nn.Linear(context_features, 128),
            torch.nn.GELU(),
            torch.nn.Linear(128, 6),
        )
        torch.nn.init.normal_(self.slot_queries, std=0.02)

    def _node_embeddings(self, data):
        h0 = F.gelu(self.node_projection(data.x))
        h1 = h0 + F.gelu(self.norm1(self.conv1(h0, data.edge_index)))
        h2 = h1 + F.gelu(self.norm2(self.conv2(h1, data.edge_index)))
        return h2

    def forward(self, data, *, return_parameters: bool = False):
        batch = getattr(data, "batch", None)
        if batch is None:
            batch = torch.zeros(
                data.x.shape[0], dtype=torch.long, device=data.x.device
            )
        voltage = data.voltage
        if voltage.ndim == 1:
            voltage = voltage[None, :]
        z_voltage = self.voltage_encoder(voltage)
        hidden = self._node_embeddings(data)
        keys = self.key_projection(hidden)
        voltage_queries = self.voltage_query(z_voltage)
        parameters = []
        attention = []
        coordinates = data.x[:, 2:4]
        for slot_index in range(self.slots):
            query = voltage_queries + self.slot_queries[slot_index][None, :]
            logits = torch.sum(keys * query[batch], dim=1) / math.sqrt(self.hidden)
            weights = softmax(logits, batch)
            pooled = global_add_pool(weights[:, None] * hidden, batch)
            center_hint = global_add_pool(weights[:, None] * coordinates, batch)
            slot_query = self.slot_queries[slot_index][None, :].expand(
                len(z_voltage), -1
            )
            context = torch.cat(
                (pooled, z_voltage, center_hint, slot_query), dim=1
            )
            parameters.append(self.slot_head(context))
            attention.append(weights)
        raw_parameters = torch.stack(parameters, dim=1)
        decoded = VoltageConditionedVesselLocalizer.decode_parameters(
            raw_parameters, self.minimum_axis, self.maximum_axis
        )
        conductivity, masks = rasterize_vessel_slots(
            coordinates,
            batch,
            decoded,
            temperature=self.raster_temperature,
        )
        if return_parameters:
            return conductivity, decoded, masks, torch.stack(attention, dim=1)
        return conductivity


class SpatialAttentionVesselDiffusionLocalizer(SpatialAttentionVesselLocalizer):
    """Localize two artery cores and predict a muscle diffusion halo per slot."""

    def __init__(
        self,
        *,
        measurements: int = 32,
        node_features: int = 4,
        hidden: int = 64,
        voltage_latent: int = 64,
        slots: int = 2,
        raster_temperature: float = 0.05,
        minimum_axis: float = 0.05,
        maximum_axis: float = 0.23,
        minimum_diffusion_length: float = 0.03,
        maximum_diffusion_length: float = 0.35,
    ):
        super().__init__(
            measurements=measurements,
            node_features=node_features,
            hidden=hidden,
            voltage_latent=voltage_latent,
            slots=slots,
            raster_temperature=raster_temperature,
            minimum_axis=minimum_axis,
            maximum_axis=maximum_axis,
        )
        context_features = hidden + voltage_latent + 2 + hidden
        self.slot_head = torch.nn.Sequential(
            torch.nn.Linear(context_features, 128),
            torch.nn.GELU(),
            torch.nn.Linear(128, 8),
        )
        self.minimum_diffusion_length = float(minimum_diffusion_length)
        self.maximum_diffusion_length = float(maximum_diffusion_length)

    def forward(self, data, *, return_parameters: bool = False):
        batch = getattr(data, "batch", None)
        if batch is None:
            batch = torch.zeros(
                data.x.shape[0], dtype=torch.long, device=data.x.device
            )
        voltage = data.voltage if data.voltage.ndim > 1 else data.voltage[None, :]
        z_voltage = self.voltage_encoder(voltage)
        hidden = self._node_embeddings(data)
        keys = self.key_projection(hidden)
        voltage_queries = self.voltage_query(z_voltage)
        parameters, attention = [], []
        coordinates = data.x[:, 2:4]
        for slot_index in range(self.slots):
            query = voltage_queries + self.slot_queries[slot_index][None, :]
            logits = torch.sum(keys * query[batch], dim=1) / math.sqrt(self.hidden)
            weights = softmax(logits, batch)
            pooled = global_add_pool(weights[:, None] * hidden, batch)
            center_hint = global_add_pool(weights[:, None] * coordinates, batch)
            slot_query = self.slot_queries[slot_index][None, :].expand(
                len(z_voltage), -1
            )
            context = torch.cat((pooled, z_voltage, center_hint, slot_query), dim=1)
            parameters.append(self.slot_head(context))
            attention.append(weights)
        decoded = decode_diffusion_parameters(
            torch.stack(parameters, dim=1),
            minimum_axis=self.minimum_axis,
            maximum_axis=self.maximum_axis,
            minimum_diffusion_length=self.minimum_diffusion_length,
            maximum_diffusion_length=self.maximum_diffusion_length,
        )
        conductivity, core_masks, _halo_masks = rasterize_diffusion_slots(
            coordinates,
            batch,
            decoded,
            data.muscle_gate,
            temperature=self.raster_temperature,
        )
        if return_parameters:
            return conductivity, decoded, core_masks, torch.stack(attention, dim=1)
        return conductivity


class SpatialVesselParameterRefiner(torch.nn.Module):
    """Refine the two stage-1 vessel parameter sets without losing topology."""

    def __init__(
        self,
        *,
        measurements: int = 32,
        node_features: int = 4,
        hidden: int = 64,
        voltage_latent: int = 64,
        slots: int = 2,
        raster_temperature: float = 0.05,
        minimum_axis: float = 0.05,
        maximum_axis: float = 0.23,
    ):
        super().__init__()
        self.hidden = int(hidden)
        self.slots = int(slots)
        self.raster_temperature = float(raster_temperature)
        self.minimum_axis = float(minimum_axis)
        self.maximum_axis = float(maximum_axis)
        self.voltage_encoder = VoltageEncoder(measurements, 128, voltage_latent)
        self.node_projection = torch.nn.Linear(node_features, hidden)
        self.conv1 = GCNConv(hidden, hidden)
        self.conv2 = GCNConv(hidden, hidden)
        self.norm1 = torch.nn.LayerNorm(hidden)
        self.norm2 = torch.nn.LayerNorm(hidden)
        self.key_projection = torch.nn.Linear(hidden, hidden)
        self.voltage_query = torch.nn.Linear(voltage_latent, hidden)
        self.parameter_query = torch.nn.Linear(6, hidden)
        self.slot_queries = torch.nn.Parameter(torch.empty(slots, hidden))
        context_features = hidden + voltage_latent + 2 + 6 + hidden
        self.delta_head = torch.nn.Sequential(
            torch.nn.Linear(context_features, 128),
            torch.nn.GELU(),
            torch.nn.Linear(128, 6),
        )
        torch.nn.init.normal_(self.slot_queries, std=0.02)
        torch.nn.init.zeros_(self.delta_head[-1].weight)
        torch.nn.init.zeros_(self.delta_head[-1].bias)

    @staticmethod
    def _apply_delta(
        initial: torch.Tensor,
        raw_delta: torch.Tensor,
        minimum_axis: float = 0.05,
        maximum_axis: float = 0.23,
    ) -> torch.Tensor:
        center = torch.clamp(
            initial[..., 0:2] + 0.15 * torch.tanh(raw_delta[..., 0:2]),
            min=-0.65,
            max=0.65,
        )
        axes = torch.clamp(
            initial[..., 2:4] + 0.06 * torch.tanh(raw_delta[..., 2:4]),
            min=float(minimum_axis),
            max=float(maximum_axis),
        )
        angle_unwrapped = initial[..., 4:5] + 0.75 * torch.tanh(
            raw_delta[..., 4:5]
        )
        angle = torch.atan2(torch.sin(angle_unwrapped), torch.cos(angle_unwrapped))
        amplitude = torch.clamp(
            initial[..., 5:6] + 0.75 * torch.tanh(raw_delta[..., 5:6]),
            min=0.0,
            max=1.5,
        )
        return torch.cat((center, axes, angle, amplitude), dim=-1)

    def forward(self, data, *, return_parameters: bool = False):
        batch = getattr(data, "batch", None)
        if batch is None:
            batch = torch.zeros(
                data.x.shape[0], dtype=torch.long, device=data.x.device
            )
        voltage = data.voltage
        if voltage.ndim == 1:
            voltage = voltage[None, :]
        initial = data.initial_parameters.reshape(-1, self.slots, 6)
        z_voltage = self.voltage_encoder(voltage)
        h0 = F.gelu(self.node_projection(data.x))
        h1 = h0 + F.gelu(self.norm1(self.conv1(h0, data.edge_index)))
        h2 = h1 + F.gelu(self.norm2(self.conv2(h1, data.edge_index)))
        keys = self.key_projection(h2)
        base_voltage_query = self.voltage_query(z_voltage)
        coordinates = data.x[:, 2:4]
        raw_deltas = []
        attention = []
        for slot_index in range(self.slots):
            parameter_query = self.parameter_query(initial[:, slot_index])
            query = (
                base_voltage_query
                + parameter_query
                + self.slot_queries[slot_index][None, :]
            )
            logits = torch.sum(keys * query[batch], dim=1) / math.sqrt(self.hidden)
            weights = softmax(logits, batch)
            pooled = global_add_pool(weights[:, None] * h2, batch)
            center_hint = global_add_pool(weights[:, None] * coordinates, batch)
            slot_query = self.slot_queries[slot_index][None, :].expand(
                len(z_voltage), -1
            )
            context = torch.cat(
                (
                    pooled,
                    z_voltage,
                    center_hint,
                    initial[:, slot_index],
                    slot_query,
                ),
                dim=1,
            )
            raw_deltas.append(self.delta_head(context))
            attention.append(weights)
        raw_delta = torch.stack(raw_deltas, dim=1)
        parameters = self._apply_delta(
            initial, raw_delta, self.minimum_axis, self.maximum_axis
        )
        conductivity, masks = rasterize_vessel_slots(
            coordinates,
            batch,
            parameters,
            temperature=self.raster_temperature,
        )
        if return_parameters:
            return conductivity, parameters, masks, torch.stack(attention, dim=1)
        return conductivity


class SpatialVesselDiffusionRefiner(SpatialVesselParameterRefiner):
    """Refine two artery cores and their tissue-gated diffusion halos."""

    def __init__(
        self,
        *,
        measurements: int = 32,
        node_features: int = 4,
        hidden: int = 64,
        voltage_latent: int = 64,
        slots: int = 2,
        raster_temperature: float = 0.05,
        minimum_axis: float = 0.05,
        maximum_axis: float = 0.23,
        minimum_diffusion_length: float = 0.03,
        maximum_diffusion_length: float = 0.35,
    ):
        super().__init__(
            measurements=measurements,
            node_features=node_features,
            hidden=hidden,
            voltage_latent=voltage_latent,
            slots=slots,
            raster_temperature=raster_temperature,
            minimum_axis=minimum_axis,
            maximum_axis=maximum_axis,
        )
        self.parameter_query = torch.nn.Linear(8, hidden)
        context_features = hidden + voltage_latent + 2 + 8 + hidden
        self.delta_head = torch.nn.Sequential(
            torch.nn.Linear(context_features, 128),
            torch.nn.GELU(),
            torch.nn.Linear(128, 8),
        )
        torch.nn.init.zeros_(self.delta_head[-1].weight)
        torch.nn.init.zeros_(self.delta_head[-1].bias)
        self.minimum_diffusion_length = float(minimum_diffusion_length)
        self.maximum_diffusion_length = float(maximum_diffusion_length)

    def _apply_diffusion_delta(
        self, initial: torch.Tensor, raw_delta: torch.Tensor
    ) -> torch.Tensor:
        vessel = self._apply_delta(
            initial[..., :6],
            raw_delta[..., :6],
            self.minimum_axis,
            self.maximum_axis,
        )
        diffusion_fraction = torch.clamp(
            initial[..., 6:7] + 0.20 * torch.tanh(raw_delta[..., 6:7]),
            min=0.0,
            max=0.5,
        )
        diffusion_length = torch.clamp(
            initial[..., 7:8] + 0.08 * torch.tanh(raw_delta[..., 7:8]),
            min=self.minimum_diffusion_length,
            max=self.maximum_diffusion_length,
        )
        return torch.cat((vessel, diffusion_fraction, diffusion_length), dim=-1)

    def forward(self, data, *, return_parameters: bool = False):
        batch = getattr(data, "batch", None)
        if batch is None:
            batch = torch.zeros(
                data.x.shape[0], dtype=torch.long, device=data.x.device
            )
        voltage = data.voltage if data.voltage.ndim > 1 else data.voltage[None, :]
        initial = data.initial_parameters.reshape(-1, self.slots, 8)
        z_voltage = self.voltage_encoder(voltage)
        h0 = F.gelu(self.node_projection(data.x))
        h1 = h0 + F.gelu(self.norm1(self.conv1(h0, data.edge_index)))
        h2 = h1 + F.gelu(self.norm2(self.conv2(h1, data.edge_index)))
        keys = self.key_projection(h2)
        base_voltage_query = self.voltage_query(z_voltage)
        coordinates = data.x[:, 2:4]
        raw_deltas, attention = [], []
        for slot_index in range(self.slots):
            query = (
                base_voltage_query
                + self.parameter_query(initial[:, slot_index])
                + self.slot_queries[slot_index][None, :]
            )
            logits = torch.sum(keys * query[batch], dim=1) / math.sqrt(self.hidden)
            weights = softmax(logits, batch)
            pooled = global_add_pool(weights[:, None] * h2, batch)
            center_hint = global_add_pool(weights[:, None] * coordinates, batch)
            slot_query = self.slot_queries[slot_index][None, :].expand(
                len(z_voltage), -1
            )
            context = torch.cat(
                (
                    pooled,
                    z_voltage,
                    center_hint,
                    initial[:, slot_index],
                    slot_query,
                ),
                dim=1,
            )
            raw_deltas.append(self.delta_head(context))
            attention.append(weights)
        parameters = self._apply_diffusion_delta(
            initial, torch.stack(raw_deltas, dim=1)
        )
        conductivity, core_masks, _halo_masks = rasterize_diffusion_slots(
            coordinates,
            batch,
            parameters,
            data.muscle_gate,
            temperature=self.raster_temperature,
        )
        if return_parameters:
            return conductivity, parameters, core_masks, torch.stack(attention, dim=1)
        return conductivity
