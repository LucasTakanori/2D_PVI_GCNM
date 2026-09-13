"""Opt-in regression using private recordings, never included in the repository.

Set GCNM_REGRESSION_BUNDLE to the deployed 15-ring bundle and
GCNM_REGRESSION_SESSION to the 2026-09-12 21-48-58 session.h5, then run this
module with pytest. The amplitude bound detects the known numerical explosion;
it is NOT a physiological validity criterion.
"""
from pathlib import Path
import os

import h5py
import numpy as np
import pytest
import torch

from gcnm_pvi.representations import CoordinateReconstructor


@pytest.mark.integration
def test_recorded_transition_matches_independent_augmented_solve(monkeypatch):
    bundle_value = os.environ.get("GCNM_REGRESSION_BUNDLE")
    session_value = os.environ.get("GCNM_REGRESSION_SESSION")
    if not bundle_value or not session_value:
        pytest.skip("private recorded-session regression paths not configured")
    bundle = Path(bundle_value)
    requested = np.array([99, 300, 649, 650, 651, 652, 653, 654, 655, 1100])
    with h5py.File(session_value, "r") as handle:
        sequences = handle["sequence"][:]
        indices = np.flatnonzero(np.isin(sequences, requested))
        np.testing.assert_array_equal(sequences[indices], requested)
        voltage = handle["gcnm_input_voltage"][indices]
        saved = handle["element_conductivity_change"][indices]

    previous_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        with CoordinateReconstructor(
            bundle / "configs/rings_b045/US075.yaml",
            bundle / "weights/US075",
            "coordinate_direct_projected_fine",
            device="cpu",
            physics_workers=2,
            forward_backend="sparse",
        ) as model:
            kwargs = dict(model_batch_size=50, physics_workers=2,
                          compute_stage2_residuals=False)
            stage1, stage2, _ = model.reconstruct_batch(voltage, **kwargs)
            monkeypatch.setattr(model.nonlinear_solver, "solve",
                                model.nonlinear_solver._augmented_solve)
            oracle1, oracle2, _ = model.reconstruct_batch(voltage, **kwargs)
    finally:
        torch.set_num_threads(previous_threads)

    assert np.isfinite(stage2).all()
    assert np.max(np.abs(stage2)) < 10.0
    np.testing.assert_array_equal(stage1, oracle1)
    np.testing.assert_allclose(stage2, oracle2, rtol=2e-5, atol=2e-6)
    ordinary = np.isin(requested, [99, 300, 1100])
    np.testing.assert_allclose(stage2[ordinary], saved[ordinary],
                               rtol=2e-5, atol=2e-6)
