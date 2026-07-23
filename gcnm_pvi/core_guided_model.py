"""Two-stage dense GCNM with an auxiliary, beat-stable artery localizer.

The conductivity image remains a free per-element graph output.  The artery
heads are auxiliary: they force the hidden representation to retain two
vascular regions without restricting the final image to two analytic blobs.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.utils import softmax


class CoreGuidedGCNMStage(torch.nn.Module):
    """One learned GCNM stage with shared beat context and dense output.

    Node features follow the fixed contract

    ``(current, LM direction, beat-context magnitude, x, y, radius)``.

    The context branch only sees ``(beat-context magnitude, x, y, radius)``.
    Consequently its two artery attentions are identical for all phases that
    share a beat template.  The map branch still sees the signed per-frame LM
    direction and can represent artery cores, muscle diffusion, and weaker
    non-arterial conductivity changes.
    """

    def __init__(
        self,
        *,
        node_features: int = 6,
        hidden: int = 64,
        residual: bool = False,
        correction_limit: float = 0.5,
    ) -> None:
        super().__init__()
        if node_features != 6:
            raise ValueError("the core-guided feature contract has six channels")
        self.node_features = int(node_features)
        self.hidden = int(hidden)
        self.residual = bool(residual)
        self.correction_limit = float(correction_limit)

        # Beat-stable geometry branch: |template physics| + (x, y, r).
        self.context_projection = torch.nn.Linear(4, hidden)
        self.context_conv1 = GCNConv(hidden, hidden)
        self.context_conv2 = GCNConv(hidden, hidden)
        self.context_norm1 = torch.nn.LayerNorm(hidden)
        self.context_norm2 = torch.nn.LayerNorm(hidden)
        self.core_head = torch.nn.Linear(hidden, 1)
        self.slot_head = torch.nn.Linear(hidden, 2)

        # Dense signed-conductivity branch.
        self.map_projection = torch.nn.Linear(node_features + hidden, hidden)
        self.map_conv1 = GCNConv(hidden, hidden)
        self.map_conv2 = GCNConv(hidden, hidden)
        self.map_norm1 = torch.nn.LayerNorm(hidden)
        self.map_norm2 = torch.nn.LayerNorm(hidden)
        self.output_head = torch.nn.Linear(node_features + 4 * hidden, 1)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        for module in (
            self.context_projection,
            self.context_conv1,
            self.context_conv2,
            self.context_norm1,
            self.context_norm2,
            self.core_head,
            self.slot_head,
            self.map_projection,
            self.map_conv1,
            self.map_conv2,
            self.map_norm1,
            self.map_norm2,
        ):
            module.reset_parameters()
        # A new stage starts at zero (stage 1) or at the previous image
        # (stage 2).  This avoids an arbitrary initial artifact field.
        torch.nn.init.zeros_(self.output_head.weight)
        torch.nn.init.zeros_(self.output_head.bias)

    def _batch_index(self, data) -> torch.Tensor:
        batch = getattr(data, "batch", None)
        if batch is None:
            batch = torch.zeros(
                data.x.shape[0], dtype=torch.long, device=data.x.device
            )
        return batch

    def _context(self, data) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        edge_index = data.edge_index
        raw = data.x[:, 2:6]
        h0 = F.gelu(self.context_projection(raw))
        h1 = h0 + F.gelu(
            self.context_norm1(self.context_conv1(h0, edge_index))
        )
        h2 = h1 + F.gelu(
            self.context_norm2(self.context_conv2(h1, edge_index))
        )
        core_logits = self.core_head(h2)
        slot_logits = self.slot_head(h2)
        attention = softmax(slot_logits, self._batch_index(data))
        return h2, core_logits, attention

    def forward(self, data, *, return_aux: bool = False):
        context, core_logits, attention = self._context(data)
        raw = data.x
        edge_index = data.edge_index
        h0 = F.gelu(self.map_projection(torch.cat((raw, context), dim=1)))
        h1 = h0 + F.gelu(self.map_norm1(self.map_conv1(h0, edge_index)))
        h2 = h1 + F.gelu(self.map_norm2(self.map_conv2(h1, edge_index)))
        features = torch.cat((raw, context, h0, h1, h2), dim=1)
        correction = self.correction_limit * torch.tanh(
            self.output_head(features)
        )
        prediction = raw[:, 0:1] + correction if self.residual else correction
        if return_aux:
            return prediction, core_logits, attention
        return prediction

    def copy_context_from(self, source: "CoreGuidedGCNMStage") -> None:
        """Copy the beat-level artery representation into a later stage."""

        for name in (
            "context_projection",
            "context_conv1",
            "context_conv2",
            "context_norm1",
            "context_norm2",
            "core_head",
            "slot_head",
        ):
            getattr(self, name).load_state_dict(getattr(source, name).state_dict())

    def freeze_context(self) -> None:
        """Keep artery locations fixed while stage 2 learns a dense correction."""

        for name in (
            "context_projection",
            "context_conv1",
            "context_conv2",
            "context_norm1",
            "context_norm2",
            "core_head",
            "slot_head",
        ):
            for parameter in getattr(self, name).parameters():
                parameter.requires_grad_(False)
