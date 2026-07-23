"""Unit tests for simulator-to-PVI dataset geometry and phase selection."""

import numpy as np

from gcnm_pvi.generate_finger_simulator_dataset import (
    _ellipse_from_boundary,
    _select_phase,
)


def test_boundary_fit_recovers_rotated_ellipse():
    angle = 0.37
    phase = np.linspace(0.0, 2.0 * np.pi, 512, endpoint=False)
    aligned = np.column_stack((0.16 * np.cos(phase), 0.08 * np.sin(phase)))
    cosine, sine = np.cos(angle), np.sin(angle)
    rotation = np.array([[cosine, -sine], [sine, cosine]])
    points = aligned @ rotation.T + np.array([0.21, -0.18])
    fitted = _ellipse_from_boundary(points)
    np.testing.assert_allclose(
        [fitted["center_x"], fitted["center_y"]], [0.21, -0.18], atol=1e-8
    )
    np.testing.assert_allclose(
        [fitted["axis_a"], fitted["axis_b"]], [0.16, 0.08], atol=1e-8
    )


def test_phase_selection_avoids_resting_zero_frames():
    waveform = np.array([0.0, 0.02, 0.20, 0.55, 1.0, 0.1])
    rng = np.random.default_rng(9)
    selected = [_select_phase(waveform, rng, 0.05) for _ in range(50)]
    assert all(waveform[index] >= 0.05 for index in selected)
    assert len(set(selected)) > 1
