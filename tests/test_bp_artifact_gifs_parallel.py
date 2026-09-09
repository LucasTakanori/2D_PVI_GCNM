from functools import partial

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from PIL import Image

from gcnm_pvi import bp_artifact_gifs as legacy
from gcnm_pvi import bp_artifact_gifs_parallel as parallel


def test_frame_chunks_are_balanced_contiguous_and_complete():
    chunks = parallel._frame_chunks(150, 16)
    assert len(chunks) == 16
    assert max(map(len, chunks)) - min(map(len, chunks)) == 1
    assert [offset for chunk in chunks for offset in chunk] == list(range(150))


def test_frame_chunks_do_not_create_empty_workers():
    assert parallel._frame_chunks(3, 8) == [[0], [1], [2]]


@pytest.mark.parametrize("frame_count,workers", [(0, 1), (1, 0)])
def test_frame_chunks_reject_invalid_limits(frame_count, workers):
    with pytest.raises(ValueError):
        parallel._frame_chunks(frame_count, workers)


def test_parallel_wrapper_restores_original_renderer(monkeypatch):
    original = legacy._render_prediction_gif
    observed = {}

    def fake_generate(**kwargs):
        observed["renderer"] = legacy._render_prediction_gif
        return {"examples": [], "kwargs": kwargs}

    monkeypatch.setattr(legacy, "generate_bp_artifact_gifs", fake_generate)
    report = parallel.generate_bp_artifact_gifs_parallel(
        workers=4,
        scratch_root=None,
        artifact_main="artifact",
    )
    assert isinstance(observed["renderer"], partial)
    assert observed["renderer"].func is parallel._render_prediction_gif_parallel
    assert report["kwargs"]["artifact_main"] == "artifact"
    assert legacy._render_prediction_gif is original


def test_read_one_targets_the_matching_parquet_row_group(tmp_path):
    shard = tmp_path / "part-00000.parquet"
    pq.write_table(
        pa.table(
            {
                "sample_id": ["sample-a", "sample-b", "sample-c"],
                "image": [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]],
            }
        ),
        shard,
        row_group_size=2,
    )

    table = legacy._read_one(
        tmp_path,
        "sample-c",
        ["image"],
        shard_paths=[shard],
    )

    assert table.num_rows == 1
    assert table["sample_id"][0].as_py() == "sample-c"
    assert table["image"][0].as_py() == [5.0, 6.0]


def test_parallel_renderer_writes_ordered_animated_gif(tmp_path):
    panels = [
        np.full((4, 4, 250), fill_value=index + 1, dtype=np.float32)
        for index in range(5)
    ]
    output = tmp_path / "parallel.gif"
    parallel._render_prediction_gif_parallel(
        output,
        panels,
        ["HP", "LP", "S1", "S2", "dS2/dt"],
        output_mode="waveform",
        prediction=np.linspace(70, 120, 50, dtype=np.float32),
        target=np.linspace(72, 118, 50, dtype=np.float32),
        waveform=np.linspace(72, 118, 50, dtype=np.float32),
        heading="test",
        workers=2,
        scratch_root=tmp_path,
        frame_offsets=[0, 1],
    )
    with Image.open(output) as gif:
        assert gif.n_frames == 2
        assert gif.size == (1377, 760)
