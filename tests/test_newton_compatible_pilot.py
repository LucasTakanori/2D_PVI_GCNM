import h5py
import numpy as np

from gcnm_pvi.pilot_newton_compatible import centered_difference, run_session_pilot
from gcnm_pvi.production_preprocess import movmean


class _Mappings:
    @staticmethod
    def elem_to_image(elements):
        return np.repeat(np.asarray(elements), 40 * 40, axis=0)


class _Reconstructor:
    mappings = _Mappings()

    def __init__(self):
        self.measured = []

    def reconstruct(self, voltage):
        values = np.asarray(voltage)
        self.measured.append(values.copy())
        scalar = np.mean(values, axis=1, keepdims=True)
        residual = np.sqrt(np.mean(values**2, axis=1))
        return scalar, 2.0 * scalar, {
            "stage_1_forward_voltage_rms": residual,
            "stage_2_forward_voltage_rms": 0.5 * residual,
        }


def test_centered_difference_matches_pvi_ml_nan_and_padding_contract():
    values = np.array([np.nan, 1.0, 3.0, 7.0, np.nan])
    assert np.array_equal(centered_difference(values), [0.0, 1.5, 3.0, -1.5, 0.0])


def test_pilot_reconstructs_hp_and_stable_dynamic_voltage_separately(tmp_path):
    periods = 6
    frames = periods * 50
    source = tmp_path / "subject006_baseline_masked.h5"
    hp = np.tile(np.linspace(-2.0, 2.0, frames), (32, 1))
    lp_change = np.tile(np.linspace(-1.0, 1.0, frames), (32, 1))
    lp = 100.0 + lp_change
    base_image = np.tile(np.linspace(-1.0, 1.0, frames), (40, 40, 1))
    with h5py.File(source, "w") as handle:
        handle.create_dataset("masks/mask05", data=np.array([[1, 5], [2, 6]]))
        handle.create_dataset("data/pviHP/resistance", data=hp)
        handle.create_dataset("data/pviLP/resistance", data=lp)
        handle.create_dataset("data/pviHP/img", data=base_image)
        handle.create_dataset("data/pviLP/img", data=2.0 * base_image)

    reconstructor = _Reconstructor()
    report = run_session_pilot(
        {
            "source_hdf5": str(source),
            "source_name": "subject006_baseline",
            "subject": "subject006",
            "session": "baseline",
            "ring": "US120",
        },
        reconstructor,
        tmp_path / "pilot",
        mask_position="first",
    )

    hp_measured, dynamic_measured = reconstructor.measured
    np.testing.assert_allclose(hp_measured, 0.01 * hp[:, :250].T)
    full_voltage = 0.01 * (hp + lp)
    expected_dynamic = full_voltage - movmean(full_voltage, 250, axis=1)
    np.testing.assert_allclose(dynamic_measured, expected_dynamic[:, :250].T)
    assert report["voltage_contract"]["bp_input_candidate"] == [
        "dynamic_stage2",
        "centered_difference(dynamic_stage1)",
        "centered_difference(centered_difference(dynamic_stage1))",
    ]
    assert report["hp_stage_diagnostics"]["stage_2_lower_residual_fraction"] == 1.0
