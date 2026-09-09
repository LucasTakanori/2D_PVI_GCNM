"""Parity gates for moving deployable helpers out of training entrypoints."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from gcnm_pvi.coordinate_runtime import (
    build_coordinate_model,
    make_coordinate_graphs,
)
from gcnm_pvi.train_faithful_gcnm import (
    _make_dataset as legacy_make_coordinate_graphs,
)
from gcnm_pvi.train_faithful_gcnm import _model as legacy_build_coordinate_model
from gcnm_pvi.train_voltage_vessel_gcnm import _graphs as legacy_make_vessel_graphs
from gcnm_pvi.vessel_runtime import make_vessel_graphs


def _assert_graphs_equal(left, right) -> None:
    assert len(left) == len(right)
    for actual, expected in zip(left, right):
        assert set(actual.keys()) == set(expected.keys())
        for key in actual.keys():
            actual_value = actual[key]
            expected_value = expected[key]
            if isinstance(actual_value, torch.Tensor):
                torch.testing.assert_close(actual_value, expected_value, rtol=0, atol=0)
            else:
                assert actual_value == expected_value


def test_coordinate_graph_contract_matches_preserved_training_helper() -> None:
    truth = np.asarray([[0.0, 0.02, -0.01], [0.03, 0.0, -0.02]])
    current = 0.5 * truth
    direction = -0.25 * truth
    reference = np.full_like(truth, 0.7)
    positions = np.asarray([[-0.5, 0.2], [0.0, -0.3], [0.5, 0.1]])
    voltage = np.arange(8, dtype=np.float64).reshape(2, 4)
    edge_index = torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]], dtype=torch.long)
    kwargs = {
        "scale": 0.2,
        "use_coordinates": True,
        "positive_weight": 1.0,
        "voltage": voltage,
        "voltage_scale": 2.0,
        "reference": reference,
    }
    _assert_graphs_equal(
        make_coordinate_graphs(
            truth, current, direction, positions, edge_index, **kwargs
        ),
        legacy_make_coordinate_graphs(
            truth, current, direction, positions, edge_index, **kwargs
        ),
    )


def test_coordinate_model_factory_matches_preserved_training_helper() -> None:
    for mode in ("direct", "positive_direct", "proposal_residual", "shallow_residual"):
        torch.manual_seed(17)
        public = build_coordinate_model(mode, [8, 8], 5)
        torch.manual_seed(17)
        legacy = legacy_build_coordinate_model(mode, [8, 8], 5)
        assert type(public) is type(legacy)
        assert public.state_dict().keys() == legacy.state_dict().keys()
        for key in public.state_dict():
            torch.testing.assert_close(
                public.state_dict()[key], legacy.state_dict()[key], rtol=0, atol=0
            )


def test_vessel_graph_contract_matches_preserved_training_helper() -> None:
    nodes = 4
    truth = np.zeros((2, nodes), dtype=np.float64)
    current = truth.copy()
    direction = np.asarray(
        [[-1.0, -0.5, 0.5, 1.0], [1.0, 0.5, -0.5, -1.0]]
    )
    voltage = np.asarray(
        [np.linspace(-2.0, 2.0, 6), np.linspace(-3.0, 3.0, 6)]
    )
    positions = np.column_stack(
        (np.linspace(-0.5, 0.5, nodes), np.linspace(0.5, -0.5, nodes))
    )
    edge_index = torch.tensor([[0, 1, 2, 3], [1, 2, 3, 0]], dtype=torch.long)
    parameters = np.zeros((2, 2, 6), dtype=np.float32)
    kwargs = {
        "conductivity_scale": 0.2,
        "voltage_scale": 1.5,
        "voltage_template": voltage,
        "context_direction": direction,
        "voltage_clean": voltage,
        "beat_id": np.asarray([4, 5]),
        "voltage_input_mode": "beat_normalized",
        "voltage_rms_reference": 1.0,
    }
    _assert_graphs_equal(
        make_vessel_graphs(
            truth,
            current,
            direction,
            voltage,
            parameters,
            positions,
            edge_index,
            **kwargs,
        ),
        legacy_make_vessel_graphs(
            truth,
            current,
            direction,
            voltage,
            parameters,
            positions,
            edge_index,
            **kwargs,
        ),
    )


def test_deployable_representations_do_not_import_training_entrypoints() -> None:
    source = (
        Path(__file__).resolve().parents[1] / "gcnm_pvi" / "representations.py"
    ).read_text(encoding="utf-8")
    assert "train_faithful_gcnm" not in source
    assert "train_voltage_vessel_gcnm" not in source
