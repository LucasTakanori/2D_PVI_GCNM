import json
from pathlib import Path

import h5py
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch

from gcnm_pvi.export_coordinate_direct_parquet import (
    _physical_window_voltage,
    _schema,
)
from gcnm_pvi.pvi_parquet_dataset import PviParquetCompositeDataset
from gcnm_pvi.newton_coordinate_parquet_dataset import (
    PviNewtonCoordinateCombinedDataset,
)
from gcnm_pvi.newton_coordinate_hdf5_dataset import (
    PviNewtonCoordinateHdf5Dataset,
    validate_coordinate_hdf5_contract,
)
from gcnm_pvi.train_pvi_bp import _install_representation_preprocessor


def _fixed(values):
    flat = pa.array(np.asarray(values, dtype=np.float32).reshape(-1))
    return pa.FixedSizeListArray.from_arrays(flat, int(np.prod(values.shape[1:])))


def test_physical_voltage_sums_hp_lp_references_each_window_and_flips_saved_sign():
    hp = np.array([[[1.0], [2.0], [4.0]], [[10.0], [8.0], [7.0]]])
    lp = np.array([[[2.0], [5.0], [9.0]], [[3.0], [4.0], [8.0]]])
    voltage = _physical_window_voltage(hp, lp)
    assert np.allclose(voltage[:, 0], 0.0)
    assert np.allclose(voltage[0, :, 0], [0.0, -0.04, -0.10])
    assert np.allclose(voltage[1, :, 0], [0.0, 0.01, -0.02])


def test_coordinate_direct_loader_and_bp_preprocessor_make_s1_s2_ds2(tmp_path: Path):
    root = tmp_path / "parquet"
    (root / "shards").mkdir(parents=True)
    s1 = np.ones((1, 1, 40, 40, 250), dtype=np.float32)
    time = np.arange(250, dtype=np.float32)
    s2 = np.broadcast_to(time, (1, 1, 40, 40, 250)).copy()
    bp = np.arange(50, dtype=np.float32)[None]
    stats = np.ones((1, 2, 5), dtype=np.float32)
    table = pa.Table.from_arrays(
        [
            _fixed(s1), _fixed(s2), _fixed(bp), _fixed(stats),
            pa.array(["subject006"]), pa.array(["baseline"]),
            pa.array(["subject006_baseline"]), pa.array(["sample-a"]),
            pa.array([0], type=pa.int32()), pa.array([0], type=pa.int32()),
            pa.array([5], type=pa.int32()), pa.array([10], type=pa.int32()),
        ],
        schema=_schema(),
    )
    pq.write_table(table, root / "shards" / "part-00000.parquet")
    manifest = {
        "schema": "pvi-gcnm-coordinate-direct-parquet-v1",
        "mask_key": "mask05",
        "bp_channel_contract": ["s1", "s2", "d_s2_dt"],
        "tensor_shapes": {
            "s1": [1, 40, 40, 250], "s2": [1, 40, 40, 250],
            "bp_waveform": [50], "stats": [2, 5],
        },
    }
    (root / "manifest.json").write_text(json.dumps(manifest))
    split = {"assignments": {"sample-a": "train"}}
    dataset = PviParquetCompositeDataset(root, "subject006", split_manifest=split).build()
    sample = dataset[0]
    assert torch.equal(sample["pviHP"], torch.from_numpy(s1[0]))
    assert torch.equal(sample["pviLP"], torch.from_numpy(s2[0]))
    assert dataset.shapes["input"] == (1, 40, 40, 250)

    class Model:
        nan_values = 0.0

        @staticmethod
        def _compute_diff(x):
            return torch.nn.functional.pad((x[..., 2:] - x[..., :-2]) / 2, (1, 1))

    model = Model()
    assert _install_representation_preprocessor(model, manifest) == "s1_s2_d_s2_dt"
    output = model._process_sequence(
        {"pviHP": sample["pviHP"].unsqueeze(0), "pviLP": sample["pviLP"].unsqueeze(0)}
    )
    assert output.shape == (1, 3, 40, 40, 250)
    assert torch.all(output[:, 0] == 1)
    assert torch.all(output[:, 1, ..., 17] == 17)
    assert torch.all(output[:, 2, ..., 1:-1] == 1)
    assert torch.all(output[:, 2, ..., (0, -1)] == 0)


def test_paired_newton_coordinate_loader_builds_exact_six_channel_contract(tmp_path: Path):
    coordinate_root = tmp_path / "coordinate"
    reference_root = tmp_path / "reference"
    (coordinate_root / "shards").mkdir(parents=True)
    (reference_root / "shards").mkdir(parents=True)
    rows = 2
    time = np.arange(250, dtype=np.float32)
    s1 = np.ones((rows, 1, 40, 40, 250), dtype=np.float32)
    s2 = np.broadcast_to(time, (rows, 1, 40, 40, 250)).copy()
    newton_hp = np.full_like(s1, 10.0)
    newton_lp = np.broadcast_to(time**2, s1.shape).copy()
    bp = np.stack((np.arange(50, dtype=np.float32), np.arange(50, dtype=np.float32) + 1))
    stats = np.ones((rows, 2, 5), dtype=np.float32)
    ids = ["sample-a", "sample-b"]
    metadata = [
        pa.array(["subject006"] * rows), pa.array(["baseline"] * rows),
        pa.array(["subject006_baseline"] * rows), pa.array(ids),
        pa.array([0, 0], type=pa.int32()), pa.array([0, 5], type=pa.int32()),
        pa.array([5, 10], type=pa.int32()), pa.array([10, 10], type=pa.int32()),
    ]
    coordinate = pa.Table.from_arrays(
        [_fixed(s1), _fixed(s2), _fixed(bp), _fixed(stats), *metadata], schema=_schema()
    )
    pq.write_table(coordinate, coordinate_root / "shards" / "part.parquet")
    (coordinate_root / "manifest.json").write_text(json.dumps({
        "schema": "pvi-gcnm-coordinate-direct-parquet-v1", "mask_key": "mask05",
        "bp_channel_contract": ["s1", "s2", "d_s2_dt"],
        "tensor_shapes": {"s1": [1, 40, 40, 250], "s2": [1, 40, 40, 250],
                          "bp_waveform": [50], "stats": [2, 5]},
    }))
    reference = pa.table({
        "pviHP": _fixed(newton_hp), "pviLP": _fixed(newton_lp),
        "bp_waveform": _fixed(bp), "stats": _fixed(stats),
        "subject": metadata[0], "session": metadata[1], "source_name": metadata[2],
        "sample_id": metadata[3], "source_order": metadata[4],
        "mask_start": metadata[5], "mask_stop": metadata[6], "num_periods": metadata[7],
    })
    pq.write_table(reference, reference_root / "shards" / "part.parquet")
    (reference_root / "manifest.json").write_text(json.dumps({
        "schema": "pvi-reference-parquet-v1", "input_mode": "img",
        "tensor_shapes": {"pviHP": [1, 40, 40, 250], "pviLP": [1, 40, 40, 250],
                          "bp_waveform": [50], "stats": [2, 5]},
    }))
    split = {"assignments": {ids[0]: "train", ids[1]: "test"}}
    dataset = PviNewtonCoordinateCombinedDataset(
        reference_root, coordinate_root, "subject006", "waveform", split
    ).build()
    dataset.get_partition()
    sample = dataset[0]
    assert dataset.shapes["input"] == (2, 40, 40, 250)
    assert torch.all(sample["pviHP"][0] == 10)
    assert torch.all(sample["pviHP"][1] == 1)
    assert torch.equal(sample["pviLP"][1], torch.from_numpy(s2[0, 0]))

    class Model:
        nan_values = 0.0

        @staticmethod
        def _compute_diff(x):
            return torch.nn.functional.pad((x[..., 2:] - x[..., :-2]) / 2, (1, 1))

    model = Model()
    contract = _install_representation_preprocessor(model, dataset.manifest)
    assert contract == "newton_hp_dlp_ddlp_plus_s1_s2_d_s2_dt"
    output = model._process_sequence(
        {"pviHP": sample["pviHP"].unsqueeze(0), "pviLP": sample["pviLP"].unsqueeze(0)}
    )
    assert output.shape == (1, 6, 40, 40, 250)
    assert torch.all(output[:, 0] == 10)
    assert torch.all(output[:, 3] == 1)
    assert torch.all(output[:, 4, ..., 17] == 17)
    assert torch.all(output[:, 5, ..., 1:-1] == 1)


def test_hdf5_newton_coordinate_loader_matches_parquet_pairing(tmp_path: Path):
    coordinate_root = tmp_path / "coordinate"
    (coordinate_root / "shards").mkdir(parents=True)
    rows = 2
    time = np.arange(250, dtype=np.float32)
    s1 = np.ones((rows, 1, 40, 40, 250), dtype=np.float32)
    s2 = np.broadcast_to(time, (rows, 1, 40, 40, 250)).copy()
    bp_signal = np.arange(500, dtype=np.float32)
    bp = np.stack((bp_signal[200:250], bp_signal[450:500]))
    stats = np.ones((rows, 2, 5), dtype=np.float32)
    ids = [
        "subject006_baseline:mask05:0:5",
        "subject006_baseline:mask05:5:10",
    ]
    # Use production stable IDs rather than encoding assumptions in the test.
    from gcnm_pvi.pvi_splits import stable_sample_id

    ids = [
        stable_sample_id("subject006_baseline", "mask05", 0, 5),
        stable_sample_id("subject006_baseline", "mask05", 5, 10),
    ]
    metadata = [
        pa.array(["subject006"] * rows),
        pa.array(["baseline"] * rows),
        pa.array(["subject006_baseline"] * rows),
        pa.array(ids),
        pa.array([0, 0], type=pa.int32()),
        pa.array([0, 5], type=pa.int32()),
        pa.array([5, 10], type=pa.int32()),
        pa.array([10, 10], type=pa.int32()),
    ]
    coordinate = pa.Table.from_arrays(
        [_fixed(s1), _fixed(s2), _fixed(bp), _fixed(stats), *metadata],
        schema=_schema(),
    )
    pq.write_table(coordinate, coordinate_root / "shards" / "part.parquet")
    (coordinate_root / "manifest.json").write_text(
        json.dumps(
            {
                "schema": "pvi-gcnm-coordinate-direct-parquet-v1",
                "mask_key": "mask05",
                "bp_channel_contract": ["s1", "s2", "d_s2_dt"],
                "tensor_shapes": {
                    "s1": [1, 40, 40, 250],
                    "s2": [1, 40, 40, 250],
                    "bp_waveform": [50],
                    "stats": [2, 5],
                },
            }
        )
    )
    hdf5_path = tmp_path / "subject006_baseline_masked.h5"
    hp_frames = np.full((40, 40, 500), 10.0, dtype=np.float64)
    lp_frames = np.broadcast_to(
        np.arange(500, dtype=np.float64), (40, 40, 500)
    ).copy()
    with h5py.File(hdf5_path, "w") as handle:
        handle.create_dataset("masks/mask05", data=[[1, 5], [6, 10]])
        handle.create_dataset("data/pviHP/img", data=hp_frames)
        handle.create_dataset("data/pviLP/img", data=lp_frames)
        handle.create_dataset("data/bp/signal", data=bp_signal[None])
        handle.create_dataset("stats/pviHP/duration", data=np.ones((1, 10)))
        handle.create_dataset("stats/pviHP/tMax", data=np.ones((1, 10)))
        handle.create_dataset("metadata/num_periods", data=10)
    registry_path = tmp_path / "registry.json"
    registry_path.write_text(
        json.dumps(
            {
                "records": [
                    {
                        "subject": "subject006",
                        "session": "baseline",
                        "source_name": "subject006_baseline",
                        "source_hdf5": str(hdf5_path),
                        "source_order": 0,
                        "exclusion_reason": None,
                    }
                ]
            }
        )
    )
    split = {"assignments": {ids[0]: "train", ids[1]: "test"}}
    split_path = tmp_path / "split.json"
    split_path.write_text(json.dumps(split))
    dataset = PviNewtonCoordinateHdf5Dataset(
        registry_path,
        coordinate_root,
        "subject006",
        "waveform",
        split,
    ).build()
    dataset.get_partition()
    sample = dataset[1]
    assert dataset.manifest["reference_storage"] == "source_hdf5"
    assert dataset.shapes["input"] == (2, 40, 40, 250)
    assert torch.all(sample["pviHP"][0] == 10)
    assert torch.all(sample["pviHP"][1] == 1)
    assert torch.equal(sample["pviLP"][0, 0, 0], torch.arange(250, 500))
    assert torch.equal(sample["pviLP"][1], torch.from_numpy(s2[1, 0]))
    report = validate_coordinate_hdf5_contract(
        coordinate_root, registry_path, split_path
    )
    assert report == {
        "status": "pass",
        "rows": 2,
        "subjects": 1,
        "sessions": 1,
        "newton_storage": "source_hdf5",
    }
