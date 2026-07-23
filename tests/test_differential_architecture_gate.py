from __future__ import annotations

import numpy as np

from gcnm_pvi.evaluate_faithful_gcnm import _newton_relative
from gcnm_pvi.make_differential_pilot_pack import _select_complete_beat
from gcnm_pvi.make_differential_cohort_pack import _load_split


def test_select_complete_beat_preserves_all_50_samples_in_order():
    anatomy = np.repeat([4, 4], 50)
    beat = np.repeat([20, 21], 50)
    sample = np.tile(np.arange(50), 2)
    permutation = np.random.default_rng(3).permutation(100)
    selected, anatomy_id, beat_id = _select_complete_beat(
        anatomy[permutation],
        beat[permutation],
        sample[permutation],
        anatomy_id=4,
        beat_id=21,
    )
    assert anatomy_id == 4 and beat_id == 21
    np.testing.assert_array_equal(sample[permutation][selected], np.arange(50))


def _metrics(**updates):
    values = {
        "element_nrmse": 0.8,
        "image_correlation": 0.4,
        "localization_dice_at_true_volume": 0.3,
        "support_sign_accuracy": 0.7,
        "prediction_to_truth_rms": 0.6,
        "frame_difference_rms_ratio": 0.5,
    }
    values.update(updates)
    return values


def test_gate_is_relative_to_newton_and_does_not_require_perfection():
    newton = _metrics()
    candidate = _metrics(
        element_nrmse=0.7,
        image_correlation=0.5,
        localization_dice_at_true_volume=0.35,
        support_sign_accuracy=0.75,
        prediction_to_truth_rms=0.7,
        frame_difference_rms_ratio=0.6,
    )
    result = _newton_relative(
        candidate,
        newton,
        candidate_voltage_residual=2.0e-6,
        newton_voltage_residual=3.0e-6,
    )
    assert result["classification"] == "promising_relative_to_production_newton"
    assert result["improvements"] == 7


def test_gate_rejects_blank_representation_even_if_one_metric_improves():
    result = _newton_relative(
        _metrics(prediction_to_truth_rms=0.001, frame_difference_rms_ratio=0.001),
        _metrics(),
        candidate_voltage_residual=1.0e-6,
        newton_voltage_residual=2.0e-6,
    )
    assert result["classification"].startswith("failed_nonfinite_or_blank")


def test_legacy_hp_lp_source_is_summed_into_one_full_differential_view(tmp_path):
    identities = {
        "anatomy_id": np.array([3, 3], dtype=np.int32),
        "beat_id": np.array([15, 15], dtype=np.int32),
        "sample_index": np.array([0, 1], dtype=np.int16),
    }
    for component, factor in (("hp", 1.0), ("lp", 2.0)):
        root = tmp_path / component
        root.mkdir()
        np.savez_compressed(
            root / "train.npz",
            sigma=np.full((2, 4), factor, dtype=np.float32),
            V_clean=np.full((2, 3), 10.0 * factor, dtype=np.float32),
            V=np.full((2, 3), 20.0 * factor, dtype=np.float32),
            **identities,
        )
    values, hashes = _load_split(
        tmp_path,
        "train",
        source_schema="pvi-gcnm-continuous-hp-lp-beats-v1",
        voltage_key="V_clean_delta_reference",
    )
    np.testing.assert_array_equal(values["sigma"], np.full((2, 4), 3.0))
    np.testing.assert_array_equal(values["voltage"], np.full((2, 3), 30.0))
    np.testing.assert_array_equal(values["augmented"], np.full((2, 3), 60.0))
    assert set(hashes) == {"hp", "lp"}
