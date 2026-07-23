import numpy as np
import torch
from torch_geometric.data import Batch, Data

from gcnm_pvi.core_guided_model import CoreGuidedGCNMStage
from gcnm_pvi.core_guided_physics import (
    linear_amplitude_calibration,
    normalized_context_direction,
)
from gcnm_pvi.evaluate_core_guided_gcnm import _beat_templates
from gcnm_pvi.generate_multisubject_beat_dataset import (
    _rank_one_template,
    _selected_phases,
)
from gcnm_pvi.train_core_guided_gcnm import _load_arrays


def _batch(graphs: int = 2, nodes: int = 20) -> Batch:
    source = torch.arange(nodes - 1, dtype=torch.long)
    edge_index = torch.stack(
        (torch.cat((source, source + 1)), torch.cat((source + 1, source)))
    )
    return Batch.from_data_list(
        [
            Data(x=torch.randn(nodes, 6), edge_index=edge_index)
            for _ in range(graphs)
        ]
    )


def test_stage_one_starts_at_zero_with_two_normalized_attentions():
    batch = _batch()
    prediction, core_logits, attention = CoreGuidedGCNMStage(
        residual=False, correction_limit=2.0
    )(batch, return_aux=True)
    torch.testing.assert_close(prediction, torch.zeros_like(prediction))
    assert core_logits.shape == (40, 1)
    assert attention.shape == (40, 2)
    for graph_index in range(2):
        selected = batch.batch == graph_index
        torch.testing.assert_close(
            attention[selected].sum(dim=0), torch.ones(2), atol=1e-5, rtol=1e-5
        )


def test_stage_two_copies_and_freezes_context_and_starts_at_current_map():
    stage1 = CoreGuidedGCNMStage(residual=False, correction_limit=2.0)
    stage2 = CoreGuidedGCNMStage(residual=True, correction_limit=0.5)
    stage2.copy_context_from(stage1)
    stage2.freeze_context()
    batch = _batch(graphs=1)
    current = torch.linspace(-0.5, 0.5, len(batch.x))
    batch.x[:, 0] = current
    prediction = stage2(batch)
    torch.testing.assert_close(prediction[:, 0], current)
    assert not any(
        parameter.requires_grad for parameter in stage2.context_projection.parameters()
    )
    for first, second in zip(
        stage1.context_conv1.parameters(), stage2.context_conv1.parameters()
    ):
        torch.testing.assert_close(first, second)


def test_context_normalization_is_sign_invariant():
    direction = np.array([[0.0, -2.0, 1.0, 4.0]])
    np.testing.assert_allclose(
        normalized_context_direction(direction),
        normalized_context_direction(-direction),
    )


def test_linear_amplitude_calibration_recovers_known_scale():
    jacobian = np.array([[1.0, 0.0], [0.0, 2.0]])
    maps = np.array([[1.0, 1.0], [2.0, -1.0]])
    measured = 0.5 * (maps @ jacobian.T)
    calibrated, scale, residual = linear_amplitude_calibration(
        maps, measured, jacobian
    )
    np.testing.assert_allclose(scale, 0.5)
    np.testing.assert_allclose(calibrated, 0.5 * maps)
    np.testing.assert_allclose(residual, 0.0, atol=1e-12)


def test_rank_one_template_rejects_phase_independent_offset():
    temporal = np.array([-1.0, 0.0, 1.0, 0.5])
    spatial = np.array([1.0, -2.0, 0.5])
    offset = np.array([9.0, 8.0, 7.0])
    beat = temporal[:, None] * spatial[None, :] + offset[None, :]
    template = _rank_one_template(beat)
    assert abs(np.corrcoef(template, spatial)[0, 1]) > 0.999


def test_trial_template_is_shared_across_noisy_beats():
    rng = np.random.default_rng(4)
    spatial = np.array([1.0, -2.0, 0.5])
    temporal = np.array([-1.0, 0.0, 1.0, 0.5])
    first = temporal[:, None] * spatial[None, :] + 0.02 * rng.normal(size=(4, 3))
    second = -temporal[:, None] * spatial[None, :] + 0.02 * rng.normal(size=(4, 3))
    voltage = np.concatenate((first, second))
    period = np.repeat([10, 11], 4)
    trial = np.ones(8, dtype=int)
    templates = _beat_templates(voltage, period, trial)
    np.testing.assert_allclose(
        templates, np.broadcast_to(templates[0], templates.shape)
    )
    assert abs(np.corrcoef(templates[0], spatial)[0, 1]) > 0.99


def test_phase_selection_includes_peak_trough_and_near_zero():
    waveform = np.array([-0.4, -0.1, 0.0, 0.3, 1.0, 0.2, -0.2, -0.3])
    phases = np.arange(len(waveform))
    selected = _selected_phases(
        waveform, phases, count=4, rng=np.random.default_rng(3)
    )
    assert int(np.argmax(waveform)) in selected
    assert int(np.argmin(waveform)) in selected
    assert int(np.argmin(np.abs(waveform))) in selected


def test_legacy_single_phase_archive_uses_measured_voltage_as_context(tmp_path):
    archive = tmp_path / "legacy.npz"
    voltage = np.arange(12, dtype=np.float32).reshape(3, 4)
    np.savez_compressed(
        archive,
        sigma=np.zeros((3, 5), dtype=np.float32),
        V=voltage,
        tissue_labels=np.zeros((3, 5), dtype=np.uint8),
    )
    arrays = _load_arrays(archive)
    np.testing.assert_array_equal(arrays["V_template"], voltage)
    assert not np.shares_memory(arrays["V_template"], arrays["V"])
