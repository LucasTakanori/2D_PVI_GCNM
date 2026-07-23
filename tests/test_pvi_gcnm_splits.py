import pytest
import torch

from gcnm_pvi.pvi_splits import (
    deterministic_subject_split,
    recover_subject_split,
    stable_sample_id,
    validate_split,
)


def _row(start, stop):
    return {
        "subject": "subject001",
        "session": "baseline",
        "source_name": "subject001_baseline",
        "mask_start": start,
        "mask_stop": stop,
        "sample_id": stable_sample_id("subject001_baseline", "mask05", start, stop),
    }


def test_stable_ids_and_overlap_components_are_leakage_safe():
    rows = [_row(0, 5), _row(1, 6), _row(10, 15), _row(11, 16)]
    manifest = deterministic_subject_split(rows, test_size=0.25, seed=42)
    validate_split(rows, manifest["assignments"])
    assert manifest["counts"]["train"] == 2
    assert manifest["counts"]["test"] == 2
    assert stable_sample_id("subject001_baseline", "mask05", 0, 5) == rows[0]["sample_id"]


def test_validation_rejects_overlapping_cross_partition_windows():
    rows = [_row(0, 5), _row(1, 6)]
    assignments = {rows[0]["sample_id"]: "train", rows[1]["sample_id"]: "test"}
    with pytest.raises(ValueError, match="overlapping windows"):
        validate_split(rows, assignments)


def test_one_session_fallback_matches_pvi_ml_seed42_overlap_removal():
    rows = [_row(start, start + 5) for start in range(30)]
    manifest = deterministic_subject_split(rows, test_size=0.1, seed=42)

    train_starts = {
        row["mask_start"]
        for row in rows
        if manifest["assignments"][row["sample_id"]] == "train"
    }
    test_starts = {
        row["mask_start"]
        for row in rows
        if manifest["assignments"][row["sample_id"]] == "test"
    }
    excluded_starts = {
        row["mask_start"]
        for row in rows
        if manifest["assignments"][row["sample_id"]] == "excluded"
    }
    assert train_starts == set(range(15)) | {16, 17, 18}
    assert test_starts == {23, 27}
    assert excluded_starts == {15, 19, 20, 21, 22, 24, 25, 26, 28, 29}
    assert manifest["counts"] == {"excluded": 10, "test": 2, "train": 18}
    validate_split(rows, manifest["assignments"])


def test_recovery_preserves_historical_excluded_windows(tmp_path):
    rows = [
        {**_row(0, 5), "num_periods": 20},
        {**_row(1, 6), "num_periods": 20},
        {**_row(10, 15), "num_periods": 20},
    ]
    checkpoint = tmp_path / "subject001_checkpoints.pth"
    torch.save(
        {
            "dataset": {
                "active_mask": [(0, 5), (1, 6), (10, 15)],
                "train_mask": [(0, 5)],
                "test_mask": [(10, 15)],
            }
        },
        checkpoint,
    )
    manifest = recover_subject_split(rows, checkpoint)
    assert manifest["counts"] == {"train": 1, "test": 1, "excluded": 1}
    assert manifest["assignments"][rows[1]["sample_id"]] == "excluded"
