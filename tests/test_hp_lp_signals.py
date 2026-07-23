import numpy as np
import pytest

from gcnm_pvi.hp_lp_signals import (
    butterworth_lowpass_5hz,
    centered_difference,
    matlab_movmean,
    referenced_hp_lp,
    resistance_to_voltage,
)


def test_even_matlab_movmean_uses_previous_center_and_shrunk_endpoints():
    values = np.arange(1.0, 7.0)
    # k=4: i=0 sees [1,2]; i=1 sees [1,2,3]; i=2 sees [1,2,3,4],
    # and the final point sees [4,5,6].
    expected = np.array([1.5, 2.0, 2.5, 3.5, 4.5, 5.0])
    np.testing.assert_allclose(matlab_movmean(values, 4), expected)


def test_movmean_honors_requested_axis():
    values = np.arange(12.0).reshape(3, 4)
    result = matlab_movmean(values, 3, axis=0)
    np.testing.assert_allclose(result[0], (values[0] + values[1]) / 2)
    np.testing.assert_allclose(result[1], values.mean(axis=0))


def test_referenced_components_preserve_sum_and_signed_values():
    phase = np.linspace(0, 8 * np.pi, 350)
    full = 2.0 + 0.2 * np.sin(phase) + np.linspace(-0.4, 0.5, len(phase))
    components = referenced_hp_lp(full[:, None], window=100, reference_index=50)
    np.testing.assert_allclose(components.hp + components.lp, components.full, atol=1e-12)
    assert np.min(components.hp) < 0 < np.max(components.hp)
    assert np.min(components.lp) < 0 < np.max(components.lp)
    np.testing.assert_allclose(components.full[50], 0.0)
    np.testing.assert_allclose(components.hp[50], 0.0)
    np.testing.assert_allclose(components.lp[50], 0.0)


def test_voltage_sign_and_centered_differences_match_pvi_ml():
    resistance = np.array([-2.0, 0.0, 3.0])
    np.testing.assert_allclose(resistance_to_voltage(resistance), [-0.02, 0.0, 0.03])
    values = np.arange(5.0)
    first = centered_difference(values)
    second = centered_difference(first)
    np.testing.assert_allclose(first, [0, 1, 1, 1, 0])
    np.testing.assert_allclose(second, [0, 0.5, 0, -0.5, 0])


def test_butterworth_rejects_invalid_sampling_and_attenuates_high_frequency():
    with pytest.raises(ValueError):
        butterworth_lowpass_5hz(np.ones(100), sampling_rate_hz=10)
    time = np.arange(500) / 50.0
    low = np.sin(2 * np.pi * time)
    high = 0.5 * np.sin(2 * np.pi * 12 * time)
    filtered = butterworth_lowpass_5hz(low + high, sampling_rate_hz=50)
    assert np.sqrt(np.mean((filtered - low) ** 2)) < 0.05
