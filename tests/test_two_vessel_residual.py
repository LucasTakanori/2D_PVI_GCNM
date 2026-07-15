"""Tests for forced two-vessel data and shallow residual GCN refinement."""

import numpy as np
import torch
from torch_geometric.data import Batch, Data

from gcnm_pvi.anatomical_phantoms import sample_anatomy
from gcnm_pvi.gcnm_model import ShallowPhysicsResidualGCNBlock
from gcnm_pvi.train_faithful_gcnm import _soft_support_dice_loss


def _edge_index(nodes: int) -> torch.Tensor:
    left = torch.arange(nodes - 1, dtype=torch.long)
    right = left + 1
    return torch.stack(
        (torch.cat((left, right)), torch.cat((right, left))),
        dim=0,
    )


def test_forced_two_vessel_sampling_is_deterministic():
    first = np.random.default_rng(31)
    second = np.random.default_rng(31)
    draws_a = [sample_anatomy(first, vessel_count=2).to_dict() for _ in range(12)]
    draws_b = [sample_anatomy(second, vessel_count=2).to_dict() for _ in range(12)]
    assert draws_a == draws_b
    assert all(len(draw["vessels"]) == 2 for draw in draws_a)


def test_shallow_residual_initializes_to_physics_proposal():
    nodes = 9
    features = torch.randn(nodes, 5)
    features[:, 0] = 0.20
    features[:, 1] = 0.07
    sample = Data(x=features, edge_index=_edge_index(nodes))
    model = ShallowPhysicsResidualGCNBlock([16, 16, 16], in_channels=5)
    prediction = model(sample)
    torch.testing.assert_close(prediction, torch.full((nodes, 1), 0.27))


def test_soft_dice_penalizes_merged_background_support():
    nodes = 12
    truth = torch.zeros(nodes, 1)
    truth[[1, 2, 9, 10]] = 1.0
    base = Data(
        x=torch.zeros(nodes, 5),
        y=truth,
        edge_index=_edge_index(nodes),
    )
    batch = Batch.from_data_list([base])
    separated = truth.clone()
    merged = torch.ones_like(truth)
    assert _soft_support_dice_loss(separated, batch) < _soft_support_dice_loss(
        merged, batch
    )
