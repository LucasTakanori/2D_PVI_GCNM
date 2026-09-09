import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import torch
from torch.utils.data import TensorDataset

from gcnm_pvi.infer_population_6ch_bp import (
    assert_checkpoint_accuracy_parity,
    _load_results,
    _subject_slices,
    make_original_order_loader,
    original_pvi_test_order,
)
from gcnm_pvi.train_population_6ch_bp import (
    finalize_selected_checkpoint,
    promote_checkpoint_file,
)


def _write_metadata_cache(tmp_path):
    cache_rows = [
        {
            "sample_id": "s2-a",
            "subject": "subject002",
            "session": "baseline",
            "source_name": "subject002_baseline",
            "mask_start": 7,
            "mask_stop": 12,
        },
        {
            "sample_id": "s1-b",
            "subject": "subject001",
            "session": "valsalva",
            "source_name": "subject001_valsalva",
            "mask_start": 9,
            "mask_stop": 14,
        },
        {
            "sample_id": "s1-a",
            "subject": "subject001",
            "session": "baseline",
            "source_name": "subject001_baseline",
            "mask_start": 2,
            "mask_stop": 7,
        },
    ]
    parquet_path = tmp_path / "test" / "part.parquet"
    parquet_path.parent.mkdir()
    pq.write_table(pa.Table.from_pylist(cache_rows), parquet_path, row_group_size=1)

    identities = [
        {**cache_rows[2], "assignment": "test"},
        {**cache_rows[1], "assignment": "test"},
        {**cache_rows[0], "assignment": "test"},
    ]
    split_path = tmp_path / "split.json"
    split_path.write_text(
        json.dumps(
            {
                "schema": "pvi-gcnm-population-within-split-v1",
                "identities": identities,
            }
        )
    )
    split_digest = hashlib.sha256(split_path.read_bytes()).hexdigest()
    dataset = SimpleNamespace(
        root=tmp_path,
        manifest={
            "counts": {"test": 3},
            "files": {"test": ["test/part.parquet"]},
            "split_manifest": str(split_path),
            "split_manifest_sha256": split_digest,
        },
    )
    return dataset


def test_original_pvi_inference_order_maps_cache_rows_by_identity(tmp_path):
    dataset = _write_metadata_cache(tmp_path)
    local_indices, rows = original_pvi_test_order(dataset)

    assert local_indices == [2, 1, 0]
    assert [row["sample_id"] for row in rows] == ["s1-a", "s1-b", "s2-a"]
    assert _subject_slices(rows) == {
        "subject001": slice(0, 2),
        "subject002": slice(2, 3),
    }


def test_original_order_loader_is_sequential_not_training_schedule():
    dataset = SimpleNamespace(
        subsets={"test": TensorDataset(torch.arange(5, dtype=torch.int64))}
    )
    loader = make_original_order_loader(
        dataset,
        [4, 1, 3, 0, 2],
        batch_size=2,
        num_workers=0,
    )
    observed = torch.cat([batch[0] for batch in loader]).tolist()
    assert observed == [4, 1, 3, 0, 2]


def test_load_results_requires_native_pvi_100_column_schema(tmp_path):
    columns = [
        *[f"pred_{index}" for index in range(1, 51)],
        *[f"target_{index}" for index in range(1, 51)],
    ]
    path = tmp_path / "dataset_lazy_results.csv"
    values = np.arange(200, dtype=np.float32).reshape(2, 100)
    import pandas as pd

    pd.DataFrame(values, columns=columns).to_csv(path, index=False)
    predictions, targets = _load_results(path)
    assert predictions.shape == targets.shape == (2, 50)
    assert torch.equal(predictions, torch.from_numpy(values[:, :50]))
    assert torch.equal(targets, torch.from_numpy(values[:, 50:]))


def test_checkpoint_accuracy_parity_accepts_the_selected_model_metric():
    checkpoint = {
        "epoch": 148,
        "tracker": {"epoch": [147, 148], "test_accuracy": [0.90, 0.91233]},
    }
    report = assert_checkpoint_accuracy_parity(
        checkpoint,
        {"bp_accuracy": 0.9123301},
        tolerance=1e-5,
    )
    assert report["absolute_difference"] < 1e-6


def test_checkpoint_accuracy_parity_rejects_results_from_another_checkpoint():
    checkpoint = {
        "epoch": 148,
        "tracker": {"epoch": [148], "test_accuracy": [0.91233]},
    }
    with pytest.raises(RuntimeError, match="does not reproduce"):
        assert_checkpoint_accuracy_parity(
            checkpoint,
            {"bp_accuracy": 0.90684},
            tolerance=1e-5,
        )


def test_checkpoint_promotion_atomically_replaces_periodic_checkpoint(tmp_path):
    current = tmp_path / "dataset_lazy_checkpoints.pth"
    best = tmp_path / "dataset_lazy_checkpoints_best.pth"
    torch.save({"epoch": 190, "model": {"value": torch.tensor([190])}}, current)
    torch.save({"epoch": 148, "model": {"value": torch.tensor([148])}}, best)

    report = promote_checkpoint_file(best, current, expected_epoch=148)
    promoted = torch.load(current, map_location="cpu", weights_only=True)
    assert report["epoch"] == 148
    assert promoted["epoch"] == 148
    assert promoted["model"]["value"].item() == 148


def test_workflow_finalization_promotes_the_best_state_selected_in_memory(tmp_path):
    checkpoint_dir = tmp_path / "checkpoints"
    checkpoint_dir.mkdir()
    current = checkpoint_dir / "dataset_lazy_checkpoints.pth"
    best = checkpoint_dir / "dataset_lazy_checkpoints_best.pth"
    torch.save({"epoch": 190, "model": {"value": torch.tensor([190])}}, current)
    torch.save({"epoch": 148, "model": {"value": torch.tensor([148])}}, best)

    class Manager:
        def generate_artifact_path(
            self, *, core_name, artifact_name, extension, suffix=None
        ):
            stem = f"{core_name}_{artifact_name}"
            if suffix:
                stem += f"_{suffix}"
            return checkpoint_dir / f"{stem}.{extension}"

    class Logger:
        def update(self, status):
            assert status == "terminal"

        def export(self, path):
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text('{"status":"terminal"}\n')

    class Tracker:
        def export(self, path):
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("epoch\n148\n")

    workflow = SimpleNamespace(
        epoch=148,
        path_manager=Manager(),
        dataset=SimpleNamespace(name="dataset_lazy"),
        checkpoint=SimpleNamespace(create=lambda name: None),
        logger=Logger(),
        tracker=Tracker(),
        status="best",
    )
    report = finalize_selected_checkpoint(workflow)
    promoted = torch.load(current, map_location="cpu", weights_only=True)
    assert report["selection"] == "best"
    assert promoted["epoch"] == 148
    assert Path(report["config"]).is_file()
    assert Path(report["history"]).is_file()
