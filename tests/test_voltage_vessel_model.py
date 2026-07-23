import numpy as np
import torch
from torch_geometric.data import Batch, Data

from gcnm_pvi.anatomical_phantoms import sample_anatomy
from gcnm_pvi.train_voltage_vessel_gcnm import _correlation_loss
from gcnm_pvi.voltage_vessel_model import (
    SpatialAttentionVesselLocalizer,
    SpatialAttentionVesselDiffusionLocalizer,
    SpatialVesselDiffusionRefiner,
    SpatialVesselParameterRefiner,
    VoltageConditionedPhysicsRefiner,
    VoltageConditionedVesselLocalizer,
)


def _batch(graphs: int = 2, nodes: int = 20) -> Batch:
    source = torch.arange(nodes - 1, dtype=torch.long)
    edge_index = torch.stack(
        (torch.cat((source, source + 1)), torch.cat((source + 1, source)))
    )
    return Batch.from_data_list(
        [
            Data(
                x=torch.randn(nodes, 4),
                edge_index=edge_index,
                voltage=torch.randn(1, 32),
            )
            for _ in range(graphs)
        ]
    )


def test_localizer_outputs_two_positive_vessel_slots():
    prediction, parameters, masks = VoltageConditionedVesselLocalizer()(
        _batch(), return_parameters=True
    )
    assert prediction.shape == (40, 1)
    assert parameters.shape == (2, 2, 6)
    assert masks.shape == (40, 2)
    assert torch.all(prediction >= 0)
    assert torch.all((parameters[:, :, 2:4] >= 0.05) & (parameters[:, :, 2:4] <= 0.23))


def test_refiner_starts_at_nonnegative_stage_one_map():
    batch = _batch(graphs=1)
    batch.x[:, 0] = torch.linspace(0.0, 1.0, len(batch.x))
    model = VoltageConditionedPhysicsRefiner()
    prediction = model(batch)
    torch.testing.assert_close(prediction[:, 0], batch.x[:, 0])
    assert torch.all(prediction >= 0)


def test_localizer_supports_unbatched_inference_used_between_stages():
    graph = _batch(graphs=1).to_data_list()[0]
    prediction, parameters, masks = VoltageConditionedVesselLocalizer()(
        graph, return_parameters=True
    )
    assert prediction.shape == (20, 1)
    assert parameters.shape == (1, 2, 6)
    assert masks.shape == (20, 2)


def test_two_vessel_sampler_enforces_boundary_gap():
    rng = np.random.default_rng(7)
    anatomy = sample_anatomy(rng, vessel_count=2, minimum_vessel_gap=0.06)
    first, second = anatomy.vessels
    center_distance = np.hypot(
        first.center_x - second.center_x, first.center_y - second.center_y
    )
    required = (
        max(first.axis_a, first.axis_b)
        + max(second.axis_a, second.axis_b)
        + 0.06
    )
    assert center_distance > required


def test_spatial_attention_localizer_has_distinct_normalized_slot_attention():
    batch = _batch(graphs=2)
    prediction, parameters, masks, attention = SpatialAttentionVesselLocalizer()(
        batch, return_parameters=True
    )
    assert prediction.shape == (40, 1)
    assert parameters.shape == (2, 2, 6)
    assert masks.shape == (40, 2)
    assert attention.shape == (40, 2)
    for graph_index in range(2):
        selected = batch.batch == graph_index
        torch.testing.assert_close(
            attention[selected].sum(dim=0), torch.ones(2), atol=1e-5, rtol=1e-5
        )


def test_parameter_refiner_starts_from_stage_one_slots_exactly():
    graph = _batch(graphs=1).to_data_list()[0]
    initial = torch.tensor(
        [[[-0.2, 0.1, 0.10, 0.08, 0.2, 0.5], [0.3, -0.2, 0.12, 0.09, -0.4, 0.7]]]
    )
    graph.initial_parameters = initial
    prediction, parameters, masks, attention = SpatialVesselParameterRefiner()(
        graph, return_parameters=True
    )
    torch.testing.assert_close(parameters, initial)
    assert prediction.shape == (20, 1)
    assert masks.shape == (20, 2)
    assert attention.shape == (20, 2)
    assert torch.all(prediction >= 0)


def test_diffusion_localizer_outputs_bounded_core_and_halo_parameters():
    batch = _batch(graphs=2)
    batch.muscle_gate = torch.ones((len(batch.x), 1))
    prediction, parameters, masks, attention = (
        SpatialAttentionVesselDiffusionLocalizer()(batch, return_parameters=True)
    )
    assert prediction.shape == (40, 1)
    assert parameters.shape == (2, 2, 8)
    assert masks.shape == (40, 2)
    assert attention.shape == (40, 2)
    assert torch.all(prediction >= 0)
    assert torch.all((parameters[:, :, 6] >= 0.0) & (parameters[:, :, 6] <= 0.5))
    assert torch.all((parameters[:, :, 7] >= 0.03) & (parameters[:, :, 7] <= 0.35))


def test_diffusion_refiner_starts_from_stage_one_parameters_exactly():
    graph = _batch(graphs=1).to_data_list()[0]
    graph.muscle_gate = torch.ones((len(graph.x), 1))
    initial = torch.tensor(
        [[
            [-0.2, 0.1, 0.10, 0.08, 0.2, 0.5, 0.12, 0.14],
            [0.3, -0.2, 0.12, 0.09, -0.4, 0.7, 0.18, 0.11],
        ]]
    )
    graph.initial_parameters = initial
    prediction, parameters, masks, attention = SpatialVesselDiffusionRefiner()(
        graph, return_parameters=True
    )
    torch.testing.assert_close(parameters, initial)
    assert prediction.shape == (20, 1)
    assert masks.shape == (20, 2)
    assert attention.shape == (20, 2)
    assert torch.all(prediction >= 0)


def test_correlation_loss_penalizes_spatially_shifted_vessels():
    graph = Data(
        x=torch.zeros((4, 1)),
        y=torch.tensor([[0.0], [1.0], [0.0], [0.5]]),
    )
    batch = Batch.from_data_list([graph])
    aligned = batch.y.clone()
    shifted = torch.tensor([[1.0], [0.0], [0.5], [0.0]])
    assert _correlation_loss(aligned, batch) < _correlation_loss(shifted, batch)
