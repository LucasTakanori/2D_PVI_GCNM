import h5py
import numpy as np

from gcnm_pvi.audit_voltage_distributions import audit_registry


def _session(path, hp, lp):
    with h5py.File(path, "w") as handle:
        handle.create_dataset("data/pviHP/resistance", data=np.asarray(hp))
        handle.create_dataset("data/pviLP/resistance", data=np.asarray(lp))
        frames = np.asarray(hp).shape[1]
        handle.create_dataset("metadata/period_length", data=1)
        handle.create_dataset("masks/mask05", data=np.array([[1, frames]]))


def test_cohort_audit_weights_subjects_not_frames(tmp_path):
    short = tmp_path / "subject001_baseline.h5"
    long = tmp_path / "subject002_baseline.h5"
    _session(short, np.ones((2, 3)) * 100, np.zeros((2, 3)))
    _session(long, np.ones((2, 300)) * 300, np.zeros((2, 300)))
    registry = {
        "records": [
            {
                "subject": "subject001",
                "ring": "US120",
                "source_name": "subject001_baseline",
                "source_hdf5": str(short),
                "source_order": 0,
            },
            {
                "subject": "subject002",
                "ring": "US120",
                "source_name": "subject002_baseline",
                "source_hdf5": str(long),
                "source_order": 1,
            },
        ]
    }
    report = audit_registry(registry)
    offsets = report["global_subject_weighted"]["hp_voltage"]["offset"]
    # Resistance 100 and 300 become 1 V and 3 V. Equal subject weighting is 2 V,
    # not the roughly 2.98 V frame-weighted result.
    np.testing.assert_allclose(offsets["subject_mean"], [2.0, 2.0])
    assert report["subject_count"] == 2
    assert report["session_count"] == 2
    assert report["rings_subject_weighted"]["US120"]["hp_voltage"]["subject_count"] == 2


def test_audit_preserves_negative_ranges_and_lp_differences(tmp_path):
    source = tmp_path / "subject003_baseline.h5"
    hp = np.array([[-200.0, 0.0, 100.0, -50.0]])
    lp = np.array([[0.0, 100.0, 400.0, 900.0]])
    _session(source, hp, lp)
    registry = {
        "records": [
            {
                "subject": "subject003",
                "ring": "US100",
                "source_name": "subject003_baseline",
                "source_hdf5": str(source),
                "source_order": 0,
            }
        ]
    }
    report = audit_registry(registry)
    subject = report["subjects"]["subject003"]["signals"]
    assert subject["hp_voltage"]["negative_minimum"] == [-2.0]
    assert subject["hp_voltage"]["negative_maximum"] == [-0.5]
    np.testing.assert_allclose(subject["d_lp"]["offset"], [1.5])
    assert "lp_referenced_mask05" in subject
