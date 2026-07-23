import json

import h5py
import numpy as np
import pytest

from gcnm_pvi.pvi_splits import stable_sample_id
from gcnm_pvi.reference_parquet import export_reference_parquet
from gcnm_pvi.reference_parquet_dataset import PviReferenceParquetDataset


def _source(path):
    frames = 500
    periods = frames // 50
    with h5py.File(path, "w") as h:
        h.create_dataset("masks/mask05", data=np.array([[1, 5], [6, 10]]))
        h.create_dataset("data/bp/signal", data=np.arange(frames, dtype=float)[None])
        for component, offset in (("pviHP", 0), ("pviLP", 10000)):
            h.create_dataset(f"data/{component}/img", data=(np.arange(40 * 40 * frames).reshape(40, 40, frames) + offset))
            h.create_dataset(f"data/{component}/reactance", data=(np.arange(32 * frames).reshape(32, frames) + offset + 20000))
            h.create_dataset(f"data/{component}/resistance", data=(np.arange(32 * frames).reshape(32, frames) + offset + 40000))
        h.create_dataset("stats/pviHP/duration", data=np.arange(periods, dtype=float)[None])
        h.create_dataset("stats/pviHP/tMax", data=(np.arange(periods, dtype=float) + 10)[None])


@pytest.mark.parametrize("mode", ["img", "bioz"])
def test_reference_export_and_loader(tmp_path, mode):
    source = tmp_path / "subject006_baseline_masked.h5"
    _source(source)
    registry = tmp_path / "registry.json"
    registry.write_text(json.dumps({"records": [{
        "subject": "subject006", "session": "baseline",
        "source_name": "subject006_baseline", "source_hdf5": str(source),
        "source_order": 0, "exclusion_reason": None,
    }]}))
    root = tmp_path / mode
    manifest = export_reference_parquet(
        registry_path=registry, output_root=root, subjects=["subject006"],
        input_mode=mode, shard_rows=1, source_hash_manifest=None,
    )
    assert manifest["row_count"] == 2
    ids = [stable_sample_id("subject006_baseline", "mask05", i, i + 5) for i in (0, 5)]
    split = {"assignments": {ids[0]: "train", ids[1]: "test"}}
    dataset = PviReferenceParquetDataset(root, "subject006", "waveform", split).build()
    dataset.get_partition()
    params = dataset.get_params_shallow()
    assert params["counts"]["num_train"] == 1
    assert params["counts"]["num_test"] == 1
    assert params["raw_stats"]["num_seq05"] == 2
    sample = dataset[0]
    assert sample["bp"].tolist() == list(np.arange(200, 250, dtype=float))
    assert sample["stats"].shape == (2, 5)
    assert dataset.state_dict()["train_sample_ids"] == [ids[0]]
    if mode == "img":
        assert sample["pviHP"].shape == (1, 40, 40, 250)
        assert sample["pviHP"][0, 0, 0].tolist() == list(np.arange(250, dtype=float))
    else:
        assert sample["pviHP"].shape == (64, 250)
        assert sample["pviHP"][0].tolist() == list(np.arange(20000, 20250, dtype=float))
        assert sample["pviHP"][32].tolist() == list(np.arange(40000, 40250, dtype=float))


def test_fiducials_are_derived_from_stored_waveform(tmp_path):
    source = tmp_path / "subject006_baseline_masked.h5"
    _source(source)
    registry = tmp_path / "registry.json"
    registry.write_text(json.dumps({"records": [{
        "subject": "subject006", "session": "baseline",
        "source_name": "subject006_baseline", "source_hdf5": str(source),
        "source_order": 0, "exclusion_reason": None,
    }]}))
    root = tmp_path / "bioz"
    export_reference_parquet(registry_path=registry, output_root=root, subjects=["subject006"], input_mode="bioz", shard_rows=2, source_hash_manifest=None)
    ids = [stable_sample_id("subject006_baseline", "mask05", i, i + 5) for i in (0, 5)]
    ds = PviReferenceParquetDataset(root, "subject006", "fiducials", {"assignments": {ids[0]: "train", ids[1]: "test"}}).build()
    assert ds[0]["bp"].tolist() == [200.0, 249.0]
