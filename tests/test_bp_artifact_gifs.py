import csv

import numpy as np
import pytest

from gcnm_pvi.bp_artifact_gifs import _load_results, select_median_error_examples


def test_results_parser_preserves_native_prediction_target_order(tmp_path):
    path = tmp_path / "subject006_results.csv"
    header = ["pred_1", "pred_2", "target_1", "target_2"]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        writer.writerow([70.0, 120.0, 72.0, 118.0])
        writer.writerow([71.0, 119.0, 73.0, 121.0])
    predictions, targets = _load_results(path, "fiducials")
    np.testing.assert_array_equal(predictions, [[70.0, 120.0], [71.0, 119.0]])
    np.testing.assert_array_equal(targets, [[72.0, 118.0], [73.0, 121.0]])


def test_selects_lower_median_error_test_sample_per_session():
    metadata = [
        {"sample_id": "a", "session": "baseline"},
        {"sample_id": "b", "session": "baseline"},
        {"sample_id": "c", "session": "baseline"},
        {"sample_id": "d", "session": "pressor"},
    ]
    predictions = np.asarray([[0.0], [2.0], [1.0], [4.0]])
    targets = np.zeros_like(predictions)
    selected = select_median_error_examples(metadata, predictions, targets)
    assert [(item["session"], item["sample_id"]) for item in selected] == [
        ("baseline", "c"),
        ("pressor", "d"),
    ]
    assert selected[0]["result_index"] == 2


def test_rejects_result_metadata_length_mismatch():
    with pytest.raises(ValueError, match="do not align"):
        select_median_error_examples(
            [{"sample_id": "a", "session": "baseline"}],
            np.zeros((2, 2)),
            np.zeros((2, 2)),
        )
