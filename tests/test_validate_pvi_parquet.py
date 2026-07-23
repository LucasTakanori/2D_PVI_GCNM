import json

import h5py
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from gcnm_pvi.export_pvi_parquet import _fixed_list, _schema
from gcnm_pvi.pvi_splits import stable_sample_id
from gcnm_pvi.validate_pvi_parquet import validate


def _write_source(path, periods=7):
    bp = np.arange(periods * 50, dtype=np.float32).reshape(periods, 50)
    duration = np.arange(periods, dtype=np.float32) + 1
    t_max = np.arange(periods, dtype=np.float32) + 11
    with h5py.File(path, "w") as handle:
        handle.create_dataset("metadata/num_periods", data=periods)
        handle.create_dataset(
            "masks/mask05", data=np.array([[1, 5], [2, 6], [3, 7]])
        )
        handle.create_dataset("data/bp/signal", data=bp)
        handle.create_dataset("stats/pviHP/duration", data=duration)
        handle.create_dataset("stats/pviHP/tMax", data=t_max)
    return bp, np.vstack((duration, t_max))


def _write_shard(path, source_name, bp, stats):
    starts = np.arange(3, dtype=np.int32)
    stops = starts + 5
    count = len(starts)
    pvi_width = 40 * 40 * 250
    images = np.ones((count, pvi_width), dtype=np.float32)
    waveforms = np.stack([bp[stop - 1] for stop in stops])
    statistics = np.stack([stats[:, start:stop].reshape(-1) for start, stop in zip(starts, stops)])
    table = pa.Table.from_arrays(
        [
            _fixed_list(images * 4, pvi_width),
            _fixed_list(images * 3, pvi_width),
            _fixed_list(images * 2, pvi_width),
            _fixed_list(images, pvi_width),
            _fixed_list(waveforms, 50),
            _fixed_list(statistics, 10),
            pa.array(["subject001"] * count),
            pa.array(["baseline"] * count),
            pa.array([source_name] * count),
            pa.array(
                [
                    stable_sample_id(source_name, "mask05", int(start), int(stop))
                    for start, stop in zip(starts, stops)
                ]
            ),
            pa.array([0] * count, type=pa.int32()),
            pa.array(starts, type=pa.int32()),
            pa.array(stops, type=pa.int32()),
            pa.array([len(bp)] * count, type=pa.int32()),
        ],
        schema=_schema(),
    )
    pq.write_table(table, path, row_group_size=1)


def test_validation_counts_metadata_and_reads_only_requested_rows(tmp_path):
    root = tmp_path / "coordinate_v2"
    shards = root / "shards"
    shards.mkdir(parents=True)
    source = tmp_path / "subject001_baseline_masked.h5"
    bp, stats = _write_source(source)
    _write_shard(shards / "subject001_baseline.parquet", "subject001_baseline", bp, stats)
    manifest = {
        "row_count": 3,
        "source_sessions": [{"source_hdf5": str(source)}],
        "tensor_shapes": {
            "hp_s1": [1, 40, 40, 250],
            "hp_s2": [1, 40, 40, 250],
            "lp_s1": [1, 40, 40, 250],
            "lp_s2": [1, 40, 40, 250],
            "bp_waveform": [50],
            "stats": [2, 5],
        },
    }
    (root / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

    report = validate(root, max_rows=2)

    assert report["rows"] == 3
    assert report["rows_checked_against_hdf5"] == 2
    assert report["status"] == "pass"


def test_validation_rejects_manifest_row_count_mismatch(tmp_path):
    root = tmp_path / "coordinate_v2"
    shards = root / "shards"
    shards.mkdir(parents=True)
    source = tmp_path / "subject001_baseline_masked.h5"
    bp, stats = _write_source(source)
    _write_shard(shards / "subject001_baseline.parquet", "subject001_baseline", bp, stats)
    (root / "manifest.json").write_text(
        json.dumps({"row_count": 4, "source_sessions": [], "tensor_shapes": {}}),
        encoding="utf-8",
    )

    try:
        validate(root, max_rows=1)
    except ValueError as error:
        assert "row count differs" in str(error)
    else:
        raise AssertionError("row-count mismatch was not rejected")
