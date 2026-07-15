"""Tests for the physics-proposal residual graph model."""

from __future__ import annotations

import unittest

import torch
from torch_geometric.data import Data

from gcnm_pvi.gcnm_model import PhysicsProposalResidualGCNBlock


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


if __name__ == "__main__":
    unittest.main()
