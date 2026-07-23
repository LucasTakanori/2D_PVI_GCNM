import numpy as np

from gcnm_pvi.coordinate_direct_representation import (
    centered_temporal_difference,
    compose_s1_s2_ds2,
    representation_diagnostics,
)
from gcnm_pvi.subject_coordinate_direct_pilot import (
    physical_full_differential_voltage,
)


def test_three_channel_contract_is_s1_s2_and_centered_s2_derivative():
    s1 = np.zeros((2, 2, 5), dtype=np.float32)
    s2 = np.broadcast_to(np.arange(5, dtype=np.float32), (2, 2, 5)).copy()
    channels = compose_s1_s2_ds2(s1, s2)
    assert channels.shape == (3, 2, 2, 5)
    np.testing.assert_array_equal(channels[0], s1)
    np.testing.assert_array_equal(channels[1], s2)
    np.testing.assert_array_equal(
        channels[2, 0, 0], np.array([0, 1, 1, 1, 0], dtype=np.float32)
    )


def test_derivative_preserves_static_outside_mesh_nan_mask():
    values = np.ones((2, 2, 4), dtype=np.float32)
    values[0, 1] = np.nan
    derivative = centered_temporal_difference(values)
    assert np.all(np.isnan(derivative[0, 1]))
    assert np.all(np.isfinite(derivative[1, 1]))


def test_representation_diagnostics_reports_nonzero_signed_channels():
    s1 = np.array([[[-1.0, 0.0, 1.0]]])
    s2 = np.array([[[-2.0, 0.0, 2.0]]])
    report = representation_diagnostics(compose_s1_s2_ds2(s1, s2))
    assert report["shape"] == [3, 1, 1, 3]
    assert report["all_channels_nonzero"]
    assert report["s2"]["negative_fraction"] > 0


def test_real_resistance_conversion_references_then_uses_physical_fem_sign():
    hp = np.array([[10.0, 11.0, 8.0], [4.0, 6.0, 3.0]])
    lp = np.array([[2.0, 4.0, 5.0], [1.0, 0.0, 2.0]])
    voltage = physical_full_differential_voltage(hp, lp)
    expected = -0.01 * np.array([[0.0, 0.0], [3.0, 1.0], [1.0, 0.0]])
    np.testing.assert_allclose(voltage, expected)
