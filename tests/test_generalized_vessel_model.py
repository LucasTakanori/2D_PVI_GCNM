import numpy as np
import torch
from torch_geometric.data import Batch, Data

from gcnm_pvi.generalized_vessel_model import (
    BeatMeanPoolVesselLocalizer,
    BeatSpatialDiffusionLocalizer,
    BeatVesselParameterRefiner,
)
from gcnm_pvi.train_voltage_vessel_gcnm import _graphs


def _graph(nodes: int = 20) -> Data:
    source = torch.arange(nodes - 1, dtype=torch.long)
    edge_index = torch.stack(
        (torch.cat((source, source + 1)), torch.cat((source + 1, source)))
    )
    coordinates = torch.stack(
        (torch.linspace(-0.8, 0.8, nodes), torch.linspace(0.7, -0.7, nodes)),
        dim=1,
    )
    return Data(
        x=torch.cat((torch.randn(nodes, 3), coordinates), dim=1),
        geometry_x=torch.cat((torch.randn(nodes, 1), coordinates), dim=1),
        edge_index=edge_index,
        voltage=torch.randn(1, 32),
        voltage_template=torch.randn(1, 32),
        voltage_log_rms=torch.zeros(1, 1),
        muscle_gate=torch.ones(nodes, 1),
    )


def test_beat_voltage_slots_factor_geometry_from_phase_amplitude():
    first = _graph()
    second = first.clone()
    second.voltage = -2.0 * first.voltage
    second.voltage_log_rms = torch.ones(1, 1)
    batch = Batch.from_data_list([first, second])
    prediction, parameters, masks = BeatMeanPoolVesselLocalizer()(
        batch, return_parameters=True
    )
    assert prediction.shape == (40, 1)
    assert parameters.shape == (2, 2, 6)
    assert masks.shape == (40, 2)
    torch.testing.assert_close(parameters[0, :, :5], parameters[1, :, :5])
    assert torch.all(torch.abs(parameters[:, :, 5]) <= 1.5)


def test_beat_diffusion_slots_output_signed_bounded_parameters():
    prediction, parameters, masks, attention = BeatSpatialDiffusionLocalizer()(
        Batch.from_data_list([_graph(), _graph()]), return_parameters=True
    )
    assert prediction.shape == (40, 1)
    assert parameters.shape == (2, 2, 8)
    assert masks.shape == (40, 2)
    assert attention.shape == (40, 2)
    assert torch.all(torch.abs(parameters[:, :, 5]) <= 1.5)
    assert torch.all((parameters[:, :, 6] >= 0.0) & (parameters[:, :, 6] <= 0.5))
    assert torch.all((parameters[:, :, 7] >= 0.03) & (parameters[:, :, 7] <= 0.35))


def test_beat_refiner_starts_exactly_at_stage_one_parameters():
    graph = _graph()
    initial = torch.tensor(
        [[
            [-0.2, -0.2, 0.10, 0.08, 0.2, -0.4, 0.12, 0.14],
            [0.3, -0.2, 0.12, 0.09, -0.4, 0.7, 0.18, 0.11],
        ]]
    )
    graph.initial_parameters = initial
    prediction, parameters, masks, attention = BeatVesselParameterRefiner(
        diffusion=True
    )(graph, return_parameters=True)
    torch.testing.assert_close(parameters, initial)
    assert prediction.shape == (20, 1)
    assert masks.shape == (20, 2)
    assert attention.shape == (20, 2)


def test_beat_graph_normalization_separates_shape_from_rms():
    nodes = 6
    base_voltage = np.linspace(-2.0, 2.0, 32)[None, :]
    voltage = np.concatenate((base_voltage, 3.0 * base_voltage), axis=0)
    truth = np.zeros((2, nodes))
    direction = np.tile(np.linspace(-1.0, 1.0, nodes), (2, 1))
    positions = np.column_stack(
        (np.linspace(-0.5, 0.5, nodes), np.linspace(0.5, -0.5, nodes))
    )
    source = torch.arange(nodes - 1, dtype=torch.long)
    edge_index = torch.stack(
        (torch.cat((source, source + 1)), torch.cat((source + 1, source)))
    )
    graphs = _graphs(
        truth,
        truth,
        direction,
        voltage,
        np.zeros((2, 2, 6), dtype=np.float32),
        positions,
        edge_index,
        conductivity_scale=1.0,
        voltage_scale=1.0,
        voltage_template=voltage,
        context_direction=direction,
        voltage_input_mode="beat_normalized",
        voltage_rms_reference=float(np.sqrt(np.mean(base_voltage**2))),
    )
    torch.testing.assert_close(graphs[0].voltage, graphs[1].voltage)
    assert graphs[1].voltage_log_rms > graphs[0].voltage_log_rms
    assert graphs[0].x.shape[1] == 5
    assert graphs[0].geometry_x.shape[1] == 3
