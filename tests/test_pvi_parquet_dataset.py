import json

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch

from gcnm_pvi.export_pvi_parquet import _schema
from gcnm_pvi.pvi_parquet_dataset import PviParquetCompositeDataset
from gcnm_pvi.pvi_splits import stable_sample_id


def _representation(tmp_path):
    root = tmp_path / "coordinate_hp_lp_v3"
    shard = root / "shards" / "part.parquet"
    shard.parent.mkdir(parents=True)
    pvi_size = 40 * 40 * 250
    rows = []
    for start, offset in ((0, 0.0), (5, 100.0)):
        waveform = np.linspace(70 + offset, 120 + offset, 50, dtype=np.float32)
        rows.append(
            {
                "hp_s1": np.full(pvi_size, 1 + offset, np.float32).tolist(),
                "hp_s2": np.full(pvi_size, 2 + offset, np.float32).tolist(),
                "lp_s1": np.full(pvi_size, 3 + offset, np.float32).tolist(),
                "lp_s2": np.full(pvi_size, 4 + offset, np.float32).tolist(),
                "bp_waveform": waveform.tolist(),
                "stats": np.arange(10, dtype=np.float32).tolist(),
                "subject": "subject001",
                "session": "baseline",
                "source_name": "subject001_baseline",
                "sample_id": stable_sample_id(
                    "subject001_baseline", "mask05", start, start + 5
                ),
                "source_order": 0,
                "mask_start": start,
                "mask_stop": start + 5,
                "num_periods": 10,
            }
        )
    pq.write_table(pa.Table.from_pylist(rows, schema=_schema()), shard, compression="zstd")
    manifest = {
        "schema": "pvi-gcnm-hp-lp-parquet-v3",
        "mask_key": "mask05",
        "tensor_shapes": {
            key: [1, 40, 40, 250]
            for key in ("hp_s1", "hp_s2", "lp_s1", "lp_s2")
        }
        | {"bp_waveform": [50], "stats": [2, 5]},
    }
    (root / "manifest.json").write_text(json.dumps(manifest))
    split = {
        "assignments": {rows[0]["sample_id"]: "train", rows[1]["sample_id"]: "test"}
    }
    return root, rows, split


def test_parquet_loader_waveform_fiducials_and_three_channel_layout(tmp_path):
    root, _rows, split = _representation(tmp_path)
    waveform_ds = PviParquetCompositeDataset(
        root,
        ["subject001"],
        output_mode="waveform",
        channel_mode="3ch",
        split_manifest=split,
    ).build()
    waveform_ds.set_partition()
    assert waveform_ds.input_mode.value == "img"
    assert waveform_ds.output_mode.value == "waveform"
    assert waveform_ds.mask_key.value == "mask05"
    sample = waveform_ds[0]
    assert waveform_ds.shapes == {
        "input": (1, 40, 40, 250),
        "output": (50,),
        "stats": (2, 5),
    }
    assert set(sample) == {"bp", "pviHP", "pviLP", "stats"}
    assert tuple(sample["pviHP"].shape) == (1, 40, 40, 250)
    assert tuple(sample["pviLP"].shape) == (1, 40, 40, 250)
    assert torch.all(sample["pviHP"] == 2.0)
    assert torch.all(sample["pviLP"] == 4.0)
    assert np.isclose(sample["bp"].min().item(), 70.0)

    fresh_ds = PviParquetCompositeDataset(
        root, ["subject001"], output_mode="waveform", split_manifest=split
    ).build()
    fresh_ds.load_state_dict(fresh_ds.state_dict())
    assert fresh_ds.train_mask == [] and fresh_ds.test_mask == []

    fid_ds = PviParquetCompositeDataset(
        root, "subject001", output_mode="fiducials", split_manifest=split
    ).build()
    fid_ds.set_partition()
    assert fid_ds.shapes["output"] == (2,)
    assert np.allclose(fid_ds[0]["bp"].numpy(), [70.0, 120.0])


def test_six_channel_loader_feeds_unchanged_pvi_ml_channel_contract(tmp_path):
    root, _rows, split = _representation(tmp_path)
    dataset = PviParquetCompositeDataset(
        root, "subject001", channel_mode="6ch", split_manifest=split
    ).build()
    sample = dataset[0]
    assert dataset.shapes["input"] == (2, 40, 40, 250)
    assert torch.all(sample["pviHP"][0] == 1.0)
    assert torch.all(sample["pviHP"][1] == 2.0)
    assert torch.all(sample["pviLP"][0] == 3.0)
    assert torch.all(sample["pviLP"][1] == 4.0)

    # This is the unmodified BasePviLearner operation: concatenate HP, dLP,
    # and d2LP.  The loader alone creates the requested six-channel ablation.
    def diff(x):
        dx = (x[..., 2:] - x[..., :-2]) / 2
        return torch.nn.functional.pad(dx, (1, 1), mode="constant")

    hp = sample["pviHP"].unsqueeze(0)
    lp = sample["pviLP"].unsqueeze(0)
    model_input = torch.cat((hp, diff(lp), diff(diff(lp))), dim=1)
    assert model_input.shape == (1, 6, 40, 40, 250)
    assert torch.all(model_input[:, 0] == 1.0)
    assert torch.all(model_input[:, 1] == 2.0)
