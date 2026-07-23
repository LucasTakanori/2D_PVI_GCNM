"""Tests for the physics-proposal residual graph model."""

from __future__ import annotations

import unittest

import torch
from torch_geometric.data import Batch, Data

from gcnm_pvi.gcnm_model import (
    PositiveOutput,
    PhysicsProposalResidualGCNBlock,
    VoltageConditionedGCNBlock,
)
from gcnm_pvi.train_faithful_gcnm import _loss


class FaithfulModelTests(unittest.TestCase):
    def test_zero_initialized_model_equals_current_plus_update(self) -> None:
        edge_index = torch.tensor(
            [[0, 1, 1, 2, 2, 0], [1, 0, 2, 1, 0, 2]], dtype=torch.long
        )
        features = torch.tensor(
            [[0.10, 0.03], [-0.02, 0.01], [0.04, -0.02]], dtype=torch.float32
        )
        model = PhysicsProposalResidualGCNBlock([4], in_channels=2).float()
        prediction = model(Data(x=features, edge_index=edge_index)).squeeze()
        torch.testing.assert_close(prediction, features[:, 0] + features[:, 1])

    def test_voltage_mlp_broadcasts_one_context_per_graph(self) -> None:
        edge_index = torch.tensor(
            [[0, 1, 1, 2, 2, 0], [1, 0, 2, 1, 0, 2]], dtype=torch.long
        )
        graphs = [
            Data(
                x=torch.randn(3, 5),
                edge_index=edge_index,
                voltage=torch.full((1, 32), float(index)),
            )
            for index in range(2)
        ]
        batch = Batch.from_data_list(graphs)
        original = batch.x.clone()
        model = VoltageConditionedGCNBlock([8, 8], node_features=5).float()
        prediction = model(batch)
        assert prediction.shape == (6, 1)
        torch.testing.assert_close(batch.x, original)
        prediction.square().mean().backward()
        assert model.voltage_encoder[0].weight.grad is not None

    def test_voltage_conditioned_residual_starts_at_physics_proposal(self) -> None:
        edge_index = torch.tensor(
            [[0, 1, 1, 2, 2, 0], [1, 0, 2, 1, 0, 2]], dtype=torch.long
        )
        graph = Data(
            x=torch.tensor([[0.1, 0.03], [-0.02, 0.01], [0.04, -0.02]]),
            edge_index=edge_index,
            voltage=torch.randn(1, 32),
        )
        model = VoltageConditionedGCNBlock(
            [8], node_features=2, residual_around_proposal=True
        ).float()
        prediction = model(graph).squeeze()
        torch.testing.assert_close(prediction, graph.x[:, 0] + graph.x[:, 1])

    def test_positive_output_never_returns_nonpositive_conductivity(self) -> None:
        class Negative(torch.nn.Module):
            def forward(self, data):
                return -100.0 * torch.ones((len(data.x), 1))

        graph = Data(x=torch.zeros(3, 1))
        prediction = PositiveOutput(Negative())(graph)
        assert torch.all(prediction > 0)

    def test_balanced_signed_loss_backpropagates_through_negative_support(self) -> None:
        class Identity(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.scale = torch.nn.Parameter(torch.tensor(0.5))

            def forward(self, batch):
                return batch.x[:, :1] * self.scale

        graph = Data(
            x=torch.tensor([[1.0], [-1.0], [0.0], [0.0]]),
            y=torch.tensor([[1.0], [-1.0], [0.0], [0.0]]),
            weights=torch.ones(4, 1),
            background=torch.tensor([[False], [False], [True], [True]]),
        )
        batch = Batch.from_data_list([graph])
        model = Identity()
        loss = _loss(
            model,
            batch,
            background_weight=0.0,
            balanced_support=True,
            correlation_weight=0.1,
            amplitude_weight=0.1,
        )
        loss.backward()
        assert model.scale.grad is not None
        assert torch.isfinite(model.scale.grad)

    def test_absolute_background_penalty_is_error_not_zero_conductivity(self) -> None:
        class FirstFeature(torch.nn.Module):
            def forward(self, batch):
                return batch.x[:, :1]

        target = torch.tensor([[0.7], [0.6], [0.9]])
        graph = Data(
            x=target.clone(),
            y=target.clone(),
            reference=target.clone(),
            weights=torch.ones_like(target),
            background=torch.ones_like(target, dtype=torch.bool),
        )
        loss = _loss(
            FirstFeature(),
            Batch.from_data_list([graph]),
            background_weight=10.0,
        )
        torch.testing.assert_close(loss, torch.tensor(0.0))

    def test_relative_amplitude_operates_on_dynamic_field(self) -> None:
        class FirstFeature(torch.nn.Module):
            def forward(self, batch):
                return batch.x[:, :1]

        resting = torch.full((4, 1), 0.7)
        target = resting + torch.tensor([[0.0], [0.02], [-0.01], [0.0]])
        common = {
            "y": target,
            "reference": resting,
            "weights": torch.ones_like(target),
            "background": torch.tensor([[True], [False], [False], [True]]),
        }
        exact = Batch.from_data_list([Data(x=target.clone(), **common)])
        static = Batch.from_data_list([Data(x=resting.clone(), **common)])
        exact_loss = _loss(
            FirstFeature(), exact, 0.25, relative_amplitude_weight=0.1
        )
        static_loss = _loss(
            FirstFeature(), static, 0.25, relative_amplitude_weight=0.1
        )
        assert exact_loss < static_loss

    def test_temporal_loss_compares_frames_after_common_reference(self) -> None:
        class FirstFeature(torch.nn.Module):
            def forward(self, batch):
                return batch.x[:, :1]

        targets = [
            torch.tensor([[0.7], [0.5]]),
            torch.tensor([[0.72], [0.49]]),
            torch.tensor([[0.69], [0.53]]),
        ]
        exact = Batch.from_data_list(
            [
                Data(
                    x=target.clone(),
                    y=target,
                    weights=torch.ones_like(target),
                    background=torch.zeros_like(target, dtype=torch.bool),
                )
                for target in targets
            ]
        )
        static = Batch.from_data_list(
            [
                Data(
                    x=targets[0].clone(),
                    y=target,
                    weights=torch.ones_like(target),
                    background=torch.zeros_like(target, dtype=torch.bool),
                )
                for target in targets
            ]
        )
        exact_loss = _loss(
            FirstFeature(),
            exact,
            0.0,
            temporal_delta_weight=1.0,
            temporal_group_size=3,
        )
        static_loss = _loss(
            FirstFeature(),
            static,
            0.0,
            temporal_delta_weight=1.0,
            temporal_group_size=3,
        )
        assert exact_loss < static_loss


if __name__ == "__main__":
    unittest.main()
