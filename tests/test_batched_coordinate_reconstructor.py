from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
import torch

import gcnm_pvi.representations as representations
from gcnm_pvi.representations import (
    CoordinateReconstructor,
    _ParallelPhysics,
    _resolve_physics_mesh_mode,
    _validate_physics_hashes,
)


ELEMENTS = 4
MEASUREMENTS = 32


def _direction(current: np.ndarray, measured: np.ndarray) -> np.ndarray:
    signal = measured[:, :ELEMENTS]
    if np.count_nonzero(current) == 0:
        return signal * 0.01
    return signal * 0.02 - current * 0.1


def _residual(current: np.ndarray, measured: np.ndarray) -> np.ndarray:
    return np.sqrt(np.mean((current - measured[:, :ELEMENTS] * 0.005) ** 2, axis=1))


class _ColumnModel(torch.nn.Module):
    def __init__(self, *, residual: bool) -> None:
        super().__init__()
        self.marker = torch.nn.Parameter(torch.zeros(()))
        self.residual = residual

    def forward(self, graph):
        output = graph.x[:, 1:2]
        if self.residual:
            output = graph.x[:, 0:1] + output
        return output + self.marker * 0.0


class _FixedStage:
    def solve_many(self, measured: np.ndarray):
        current = np.zeros((len(measured), ELEMENTS), dtype=np.float64)
        direction = _direction(current, measured)
        baseline_voltages = np.zeros_like(measured)
        return direction, [], baseline_voltages


class _ParallelStage:
    def __init__(self, workers: int = 4) -> None:
        self.workers = workers
        self.closed = False

    def active_workers(self, count: int, workers: int | None = None) -> int:
        requested = self.workers if workers is None else int(workers)
        if requested <= 0:
            raise ValueError("physics_workers must be a positive integer")
        if requested > self.workers:
            raise ValueError("physics_workers exceeds the resident pool size")
        return min(count, requested)

    def lm_directions(self, baseline, current, measured, **kwargs):
        direction = _direction(current, measured)
        diagnostics = [
            SimpleNamespace(voltage_residual_rms=value)
            for value in _residual(current, measured)
        ]
        return direction, diagnostics, kwargs["baseline_voltages"]

    def residual_rms(self, baseline, current, measured, **kwargs):
        return _residual(current, measured), kwargs["baseline_voltages"], 0

    def close(self) -> None:
        self.closed = True


@pytest.fixture
def reconstructor(monkeypatch) -> CoordinateReconstructor:
    instance = CoordinateReconstructor.__new__(CoordinateReconstructor)
    instance.runtime = {
        "mappings": SimpleNamespace(
            num_elements=ELEMENTS, laplace=np.eye(ELEMENTS)
        ),
        "edge_index": torch.empty((2, 0), dtype=torch.long),
    }
    instance.positions = np.zeros((ELEMENTS, 3), dtype=np.float32)
    instance.checkpoints = [
        {
            "scale": 1.0,
            "use_voltage_mlp": False,
            "physics_contract": {
                "num_measurements": MEASUREMENTS,
                "minimum_conductivity": 1e-4,
            },
        }
        for _ in range(2)
    ]
    instance.models = [_ColumnModel(residual=False), _ColumnModel(residual=True)]
    instance.baseline = np.full(ELEMENTS, 0.7, dtype=np.float64)
    instance.fixed_stage = _FixedStage()
    instance.parallel_physics = _ParallelStage()
    instance.stage_physics = object()
    instance.nonlinear_solver = None
    instance.cfg = SimpleNamespace(
        hyper_pvi=5e-4, lambda_lm=0.0, lm_step_size=1.0
    )
    instance._reconstruction_lock = representations.threading.Lock()

    def fake_directions(
        physics, baseline, current, measured, *, baseline_voltages=None, **kwargs
    ):
        direction = _direction(current, measured)
        diagnostics = [
            SimpleNamespace(voltage_residual_rms=value)
            for value in _residual(current, measured)
        ]
        if baseline_voltages is None:
            baseline_voltages = np.zeros_like(measured)
        return direction, diagnostics, baseline_voltages

    def fake_residual(
        physics, baseline, current, measured, *, baseline_voltages=None, **kwargs
    ):
        if baseline_voltages is None:
            baseline_voltages = np.zeros_like(measured)
        return _residual(current, measured), baseline_voltages, 0

    monkeypatch.setattr(representations, "dataset_lm_directions", fake_directions)
    monkeypatch.setattr(
        representations, "dataset_voltage_residual_rms", fake_residual
    )
    return instance


def _voltage(frames: int) -> np.ndarray:
    values = np.arange(frames * MEASUREMENTS, dtype=np.float64)
    return values.reshape(frames, MEASUREMENTS) / 1000.0 - 0.4


@pytest.mark.parametrize("frames", [1, 7, 50])
def test_batch_matches_framewise_reference_in_original_order(
    reconstructor: CoordinateReconstructor, frames: int
) -> None:
    voltage = _voltage(frames)
    reference = reconstructor.reconstruct_reference(voltage)
    stage_1, stage_2, diagnostics = reconstructor.reconstruct_batch(
        voltage,
        model_batch_size=50,
        physics_workers=3,
        compute_stage2_residuals=True,
    )

    np.testing.assert_allclose(stage_1, reference[0], rtol=0.0, atol=0.0)
    np.testing.assert_allclose(stage_2, reference[1], rtol=0.0, atol=0.0)
    np.testing.assert_allclose(
        diagnostics["stage_1_forward_voltage_rms"],
        reference[2]["stage_1_forward_voltage_rms"],
    )
    np.testing.assert_allclose(
        diagnostics["stage_2_forward_voltage_rms"],
        reference[2]["stage_2_forward_voltage_rms"],
    )
    np.testing.assert_array_equal(diagnostics["frame_indices"], np.arange(frames))
    np.testing.assert_array_equal(
        diagnostics["stage_1_forward_voltage_rms_indices"], np.arange(frames)
    )
    np.testing.assert_array_equal(
        diagnostics["stage_2_forward_voltage_rms_indices"], np.arange(frames)
    )
    assert diagnostics["batch"] == {
        "frames": frames,
        "measurements": MEASUREMENTS,
        "model_batch_size": 50,
        "physics_workers": min(frames, 3),
    }


def test_residual_stride_reports_source_frame_indices(
    reconstructor: CoordinateReconstructor,
) -> None:
    voltage = _voltage(50)
    reference = reconstructor.reconstruct_reference(voltage)
    _, _, diagnostics = reconstructor.reconstruct_batch(
        voltage, stage2_residual_stride=8
    )
    expected_indices = np.array([0, 8, 16, 24, 32, 40, 48, 49])
    np.testing.assert_array_equal(
        diagnostics["stage_2_forward_voltage_rms_indices"], expected_indices
    )
    np.testing.assert_allclose(
        diagnostics["stage_2_forward_voltage_rms"],
        reference[2]["stage_2_forward_voltage_rms"][expected_indices],
    )


def test_optional_stage_2_residual_is_diagnostic_only(
    reconstructor: CoordinateReconstructor,
) -> None:
    voltage = _voltage(7)
    with_residual = reconstructor.reconstruct_batch(
        voltage, compute_stage2_residuals=True
    )
    without_residual = reconstructor.reconstruct_batch(
        voltage, compute_stage2_residuals=False
    )
    np.testing.assert_array_equal(without_residual[0], with_residual[0])
    np.testing.assert_array_equal(without_residual[1], with_residual[1])
    assert without_residual[2]["stage_2_forward_voltage_rms"].size == 0
    assert without_residual[2]["stage_2_forward_voltage_rms_indices"].size == 0
    assert without_residual[2]["stage_2_forward_voltage_rms_stride"] == 0


@pytest.mark.parametrize(
    ("voltage", "message"),
    [
        (np.zeros(MEASUREMENTS), "shape"),
        (np.zeros((0, MEASUREMENTS)), "at least one frame"),
        (np.zeros((2, MEASUREMENTS - 1)), "checkpoint requires"),
        (np.full((2, MEASUREMENTS), np.nan), "non-finite"),
    ],
)
def test_invalid_voltage_batches_are_rejected(
    reconstructor: CoordinateReconstructor, voltage: np.ndarray, message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        reconstructor.reconstruct_batch(voltage)


@pytest.mark.parametrize(
    ("kwargs", "error"),
    [
        ({"model_batch_size": 0}, ValueError),
        ({"physics_workers": 5}, ValueError),
        ({"stage2_residual_stride": 0}, ValueError),
        ({"compute_stage2_residuals": "yes"}, TypeError),
    ],
)
def test_invalid_batch_controls_are_rejected(
    reconstructor: CoordinateReconstructor, kwargs: dict, error: type[Exception]
) -> None:
    with pytest.raises(error):
        reconstructor.reconstruct_batch(_voltage(2), **kwargs)


def test_context_manager_releases_resident_workers(
    reconstructor: CoordinateReconstructor,
) -> None:
    with reconstructor as active:
        assert active is reconstructor
    assert reconstructor.parallel_physics.closed


def test_physics_mode_is_inferred_and_strictly_enforced() -> None:
    legacy = [{"physics_contract": {}}, {"physics_contract": {}}]
    projected = [
        {"physics_contract": {"physics_mesh_mode": "projected_fine"}},
        {"physics_contract": {"physics_mesh_mode": "projected_fine"}},
    ]
    assert _resolve_physics_mesh_mode(legacy, None) == "coarse"
    assert _resolve_physics_mesh_mode(projected, None) == "projected_fine"
    assert _resolve_physics_mesh_mode(projected, "projected_fine") == "projected_fine"
    with pytest.raises(ValueError, match="was trained with"):
        _resolve_physics_mesh_mode(projected, "coarse")
    assert (
        _resolve_physics_mesh_mode(
            projected, "coarse", allow_mismatch=True
        )
        == "coarse"
    )
    with pytest.raises(ValueError, match="disagree"):
        _resolve_physics_mesh_mode([legacy[0], projected[0]], None)


def test_projected_fine_contract_validates_forward_mesh_hash(monkeypatch) -> None:
    monkeypatch.setattr(
        representations,
        "sha256_file",
        lambda path: f"hash:{path}",
    )
    cfg = SimpleNamespace(
        mesh_inv_h5="inverse.h5",
        mesh_fwd_h5="forward.h5",
        mappings_h5="mappings.h5",
    )
    contract = {
        "physics_mesh_mode": "projected_fine",
        "config_sha256": "hash:config.yaml",
        "mesh_inverse_sha256": "hash:inverse.h5",
        "mesh_forward_sha256": "hash:forward.h5",
        "mappings_sha256": "hash:mappings.h5",
    }
    _validate_physics_hashes(
        contract,
        cfg,
        representations.Path("config.yaml"),
        effective_physics_mesh_mode="projected_fine",
    )
    contract["mesh_forward_sha256"] = "wrong"
    with pytest.raises(ValueError, match="mesh_forward_sha256"):
        _validate_physics_hashes(
            contract,
            cfg,
            representations.Path("config.yaml"),
            effective_physics_mesh_mode="projected_fine",
        )


def test_parallel_pool_is_capped_and_cannot_be_reused_after_close(monkeypatch) -> None:
    monkeypatch.setattr(representations, "_available_cpu_workers", lambda: 4)
    pool = _ParallelPhysics(object(), workers=20)
    assert pool.workers == 4
    assert pool.active_workers(50, 2) == 2
    assert pool.active_workers(1) == 1
    with pytest.raises(ValueError, match="resident pool"):
        pool.active_workers(50, 5)
    pool.close()
    pool.close()
    with pytest.raises(RuntimeError, match="closed"):
        pool.lm_directions(
            np.zeros((1, 1)),
            np.zeros((1, 1)),
            np.zeros((1, 1)),
            regularizer=None,
            hyper_pvi=0.0,
            lambda_lm=0.0,
            step_size=1.0,
            minimum_conductivity=1e-4,
            baseline_voltages=np.zeros((1, 1)),
            system_solver=None,
        )
