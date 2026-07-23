import torch
from torch_geometric.data import Batch, Data

from gcnm_pvi.physics_calibrated_model import (
    PhysicsCalibratedCoreStage,
    calibrate_shape_batch,
)


def _graph(nodes: int = 8) -> Data:
    source = torch.arange(nodes - 1, dtype=torch.long)
    edge_index = torch.stack(
        (torch.cat((source, source + 1)), torch.cat((source + 1, source)))
    )
    x = torch.randn(nodes, 6)
    x[:, 1] = torch.linspace(-0.5, 0.5, nodes)
    return Data(x=x, edge_index=edge_index)


def test_new_physics_calibrated_stage_starts_from_lm_direction():
    batch = Batch.from_data_list([_graph(), _graph()])
    model = PhysicsCalibratedCoreStage(correction_limit=1.0)
    prediction, core_logits, attention = model(batch, return_aux=True)
    torch.testing.assert_close(prediction, batch.x[:, 1:2])
    assert core_logits.shape == (16, 1)
    assert attention.shape == (16, 2)


def test_batch_calibration_recovers_physical_amplitude_and_adds_current():
    graph = Data(
        x=torch.zeros(2, 6),
        edge_index=torch.tensor([[0, 1], [1, 0]], dtype=torch.long),
        jacobian=torch.eye(2, dtype=torch.float32)[None, :, :],
        voltage_target=torch.tensor([[0.5, 0.5]], dtype=torch.float32),
        current_physical=torch.tensor([[0.1, -0.1]], dtype=torch.float32),
    )
    batch = Batch.from_data_list([graph])
    calibrated = calibrate_shape_batch(
        torch.ones(2, 1),
        batch,
        conductivity_scale=1.0,
        maximum_amplitude=2.0,
    )
    torch.testing.assert_close(calibrated.amplitude, torch.tensor([0.5]))
    torch.testing.assert_close(
        calibrated.correction[:, 0], torch.tensor([0.5, 0.5])
    )
    torch.testing.assert_close(
        calibrated.prediction[:, 0], torch.tensor([0.6, 0.4])
    )
    torch.testing.assert_close(
        calibrated.residual_rms, torch.zeros(1), atol=1e-7, rtol=0
    )


def test_amplitude_limit_is_in_physical_conductivity_units():
    graph = Data(
        x=torch.zeros(2, 6),
        edge_index=torch.tensor([[0, 1], [1, 0]], dtype=torch.long),
        jacobian=torch.eye(2, dtype=torch.float32)[None, :, :],
        voltage_target=torch.tensor([[10.0, 10.0]], dtype=torch.float32),
        current_physical=torch.zeros(1, 2),
    )
    calibrated = calibrate_shape_batch(
        torch.ones(2, 1),
        Batch.from_data_list([graph]),
        conductivity_scale=0.01,
        maximum_amplitude=0.25,
    )
    torch.testing.assert_close(calibrated.amplitude, torch.tensor([0.25]))
    torch.testing.assert_close(
        torch.sqrt(torch.mean(calibrated.correction[:, 0] ** 2)),
        torch.tensor(0.25),
    )
