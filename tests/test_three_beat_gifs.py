import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from gcnm_pvi.three_beat_gifs import (
    _read_exported_sample,
    _shared_limit,
    three_consecutive_beat_indices,
)


def test_three_beat_selection_is_chronological_and_anatomy_local():
    anatomy = np.repeat([8, 9], 5 * 50)
    beats = np.concatenate(
        [
            np.repeat(np.arange(base, base + 5), 50)
            for base in (40, 45)
        ]
    )
    samples = np.tile(np.arange(50), 10)
    indices = three_consecutive_beat_indices(
        anatomy, beats, samples, anatomy=9, first_beat=1
    )
    assert len(indices) == 150
    assert np.all(anatomy[indices] == 9)
    np.testing.assert_array_equal(np.unique(beats[indices]), [46, 47, 48])
    np.testing.assert_array_equal(samples[indices[:50]], np.arange(50))


def test_three_beat_selection_rejects_incomplete_beat():
    anatomy = np.zeros(149, dtype=int)
    beats = np.repeat([0, 1, 2], [50, 50, 49])
    samples = np.concatenate((np.arange(50), np.arange(50), np.arange(49)))
    with pytest.raises(ValueError, match="complete 50-sample"):
        three_consecutive_beat_indices(anatomy, beats, samples)


def test_shared_limit_ignores_nan_mesh_exterior():
    panels = np.array([[[[np.nan, -1.0], [2.0, np.nan]]]])
    assert 1.9 < _shared_limit(panels) <= 2.0


def test_exported_sample_reader_loads_one_matching_row_group(tmp_path):
    shards = tmp_path / "shards"
    shards.mkdir()
    table = pa.table(
        {
            "sample_id": ["a", "b", "target", "d"],
            "hp_s1": pa.array([[1.0], [2.0], [3.0], [4.0]]),
            "hp_s2": pa.array([[5.0], [6.0], [7.0], [8.0]]),
        }
    )
    pq.write_table(table, shards / "part.parquet", row_group_size=2)
    selected = _read_exported_sample(shards, "target", ["hp_s1", "hp_s2"])
    assert selected.num_rows == 1
    assert selected["sample_id"][0].as_py() == "target"
    assert selected["hp_s1"][0].as_py() == [3.0]
