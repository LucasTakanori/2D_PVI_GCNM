import numpy as np
from types import SimpleNamespace

from gcnm_pvi.generate_hp_lp_beat_dataset import (
    FRAMES_PER_BEAT,
    RETAINED_BEATS,
    TOTAL_BEATS,
    _anatomy_linearized_voltage_sequence,
    _measurement_effects,
    _production_newton_control,
    decompose_and_retain,
    retained_frame_indices,
    split_anatomy_ids,
)


class _LinearPhysics:
    def __init__(self, matrix):
        self.matrix = np.asarray(matrix, dtype=float)

    def _forward(self, sigma):
        voltage = self.matrix @ np.asarray(sigma)
        return SimpleNamespace(results=SimpleNamespace(vmeas=voltage))

    def jacobian_from_forward(self, forward):
        return self.matrix


def test_anatomy_matched_linearization_is_exact_for_linear_physics():
    matrix = np.array([[1.0, -2.0], [0.5, 3.0]])
    baseline = np.array([0.7, 0.6])
    sigma = np.array([[0.7, 0.6], [0.71, 0.58], [0.66, 0.63]])
    result = _anatomy_linearized_voltage_sequence(
        _LinearPhysics(matrix), baseline, sigma
    )
    np.testing.assert_allclose(result, sigma @ matrix.T)


def test_newton_control_does_not_apply_a_second_sign_flip():
    signed_inverse = np.array([[-2.0, 1.0], [0.5, -3.0]])
    voltage = np.array([[1.0, 4.0], [-2.0, 0.25]])
    np.testing.assert_allclose(
        _production_newton_control(signed_inverse, voltage),
        voltage @ signed_inverse.T,
    )


def test_anatomy_split_occurs_before_frames_and_never_overlaps():
    train, validation, test = split_anatomy_ids(160, 20, 20, seed=42)
    sets = [set(item.anatomy_ids.tolist()) for item in (train, validation, test)]
    assert [len(item) for item in sets] == [160, 20, 20]
    assert not sets[0] & sets[1]
    assert not sets[0] & sets[2]
    assert not sets[1] & sets[2]
    assert set.union(*sets) == set(range(200))


def test_padding_crop_keeps_exactly_five_beats_at_fifty_samples():
    indices = retained_frame_indices()
    assert len(indices) == RETAINED_BEATS * FRAMES_PER_BEAT == 250
    assert indices[0] == 50
    assert indices[-1] == 299


def test_continuous_decomposition_retains_signed_hp_lp_and_common_reference():
    frames = TOTAL_BEATS * FRAMES_PER_BEAT
    time = np.arange(frames, dtype=np.float64)
    cardiac = np.sin(2 * np.pi * time / FRAMES_PER_BEAT)
    drift = 0.20 * np.sin(2 * np.pi * (time - 50) / 300)
    sigma = 0.7 + cardiac[:, None] * np.array([[0.02, -0.01]]) + drift[:, None]
    voltage = cardiac[:, None] * np.array([[2e-4, -3e-4]]) + drift[:, None] * 1e-3
    amplitudes = np.column_stack((0.05 * cardiac, -0.04 * cardiac + drift * 0.01))
    resting = np.array([0.61, 0.52])
    result = decompose_and_retain(
        sigma,
        voltage,
        voltage,
        amplitudes,
        sigma_resting=resting,
        voltage_raw=voltage + 1.0,
        voltage_clean_raw=voltage,
    )
    for component in ("hp", "lp"):
        values = result[component]
        assert values["sigma"].shape == (250, 2)
        assert values["V"].shape == (250, 2)
        assert values["V_template"].shape == (250, 2)
        np.testing.assert_allclose(values["sigma"][0], 0.0, atol=1e-12)
        np.testing.assert_allclose(values["V"][0], 0.0, atol=1e-12)
        assert np.min(values["sigma"]) < 0 < np.max(values["sigma"])
    full = result["full"]
    np.testing.assert_allclose(
        full["sigma_reference"] + full["sigma_delta_reference"], full["sigma"]
    )
    np.testing.assert_allclose(
        full["sigma"], full["sigma_absolute"]
    )
    np.testing.assert_allclose(
        full["sigma_delta_reference"],
        result["hp"]["sigma"] + result["lp"]["sigma"],
    )
    np.testing.assert_allclose(
        full["V_delta_reference"],
        result["hp"]["V"] + result["lp"]["V"],
        atol=1e-12,
    )
    np.testing.assert_allclose(full["V"], full["V_absolute_filtered"])
    np.testing.assert_allclose(
        full["sigma_resting"], np.broadcast_to(resting, full["sigma_resting"].shape)
    )
    np.testing.assert_allclose(
        full["V_absolute_raw"] - full["V_clean_absolute_raw"], 1.0
    )


def test_component_templates_are_constant_inside_each_beat():
    frames = TOTAL_BEATS * FRAMES_PER_BEAT
    time = np.arange(frames)
    sigma = 0.7 + 0.01 * np.sin(2 * np.pi * time / 50)[:, None]
    voltage = np.column_stack(
        (
            np.sin(2 * np.pi * time / 50),
            2 * np.sin(2 * np.pi * time / 50 + 0.2),
        )
    )
    amplitudes = np.column_stack((voltage[:, 0], voltage[:, 1]))
    result = decompose_and_retain(sigma, voltage, voltage, amplitudes)
    for component in ("hp", "lp"):
        templates = result[component]["V_template"]
        for beat in range(RETAINED_BEATS):
            selected = templates[beat * 50 : (beat + 1) * 50]
            np.testing.assert_allclose(selected, np.broadcast_to(selected[0], selected.shape))


def test_measurement_effects_return_reproducible_acquisition_metadata():
    clean = np.linspace(-1e-4, 2e-4, 350)[:, None] * np.ones((1, 3))
    noise = {
        "channel_gain_sd": 0.0,
        "current_gain_sd": 0.0,
        "offset_rel": 0.0,
        "contact_drift_sd": 0.0,
        "white_noise_rel": 0.0,
        "correlated_noise_rel": 0.0,
        "noise_floor": 0.0,
        "artifact_probability": 0.0,
        "artifact_scale": 0.0,
    }
    measured, metadata = _measurement_effects(
        clean, np.random.default_rng(7), noise
    )
    np.testing.assert_allclose(measured, clean)
    assert metadata["current_gain"] == 1.0
    assert metadata["channel_gain"] == [1.0, 1.0, 1.0]
    assert metadata["artifact_events"] == []
    assert "not resampled" in metadata["note"]
