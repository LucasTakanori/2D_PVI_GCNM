import json

import h5py
import numpy as np
import pyarrow.parquet as pq

import gcnm_pvi.export_pvi_parquet as export_module
from gcnm_pvi.export_pvi_parquet import (
    _frame_beat_templates,
    compact_session_shards,
    export_session,
    merge_session_ranges,
)


class _Mappings:
    @staticmethod
    def elem_to_image(elements):
        return np.repeat(np.asarray(elements), 40 * 40, axis=0)


class _Reconstructor:
    mappings = _Mappings()

    def __init__(self, first=1.0, second=2.0):
        self.first = first
        self.second = second
        self.voltage = []

    def reconstruct(self, voltage):
        voltage = np.asarray(voltage)
        self.voltage.append(voltage.copy())
        count = len(voltage)
        return (
            np.full((count, 1), self.first, dtype=np.float32),
            np.full((count, 1), self.second, dtype=np.float32),
            {
                "stage_1_forward_voltage_rms": np.ones(count),
                "stage_2_forward_voltage_rms": np.ones(count),
            },
        )


class _BeatContextReconstructor(_Reconstructor):
    requires_beat_context = True

    def __init__(self, first=1.0, second=2.0):
        super().__init__(first, second)
        self.templates = []

    def reconstruct(self, voltage, voltage_template):
        self.templates.append(np.asarray(voltage_template).copy())
        return super().reconstruct(voltage)


class _SignalReconstructor(_Reconstructor):
    """Produce input-dependent outputs so chunk/order errors are observable."""

    def reconstruct(self, voltage):
        voltage = np.asarray(voltage, dtype=np.float64)
        self.voltage.append(voltage.copy())
        signal = np.mean(voltage, axis=1, keepdims=True).astype(np.float32)
        residual = np.sqrt(np.mean(voltage * voltage, axis=1))
        return (
            signal,
            2.0 * signal + np.float32(0.125),
            {
                "stage_1_forward_voltage_rms": residual,
                "stage_2_forward_voltage_rms": 2.0 * residual,
            },
        )


def _source(path, hp, lp, *, periods=5, masks=None):
    frames = periods * 50
    if masks is None:
        masks = np.array([[1, 5]])
    with h5py.File(path, "w") as handle:
        handle.create_dataset("metadata/num_periods", data=periods)
        handle.create_dataset("metadata/period_length", data=50)
        handle.create_dataset("masks/mask05", data=np.asarray(masks))
        handle.create_dataset("data/pviHP/resistance", data=hp)
        handle.create_dataset("data/pviLP/resistance", data=lp)
        handle.create_dataset("data/bp/signal", data=np.arange(frames))
        handle.create_dataset("stats/pviHP/duration", data=np.arange(periods))
        handle.create_dataset("stats/pviHP/tMax", data=np.arange(periods) + 10)


def _record(source):
    return {
        "source_hdf5": str(source),
        "source_name": "subject001_baseline",
        "subject": "subject001",
        "session": "baseline",
        "source_order": 0,
    }


def test_export_separates_hp_lp_references_and_writes_four_literal_stages(tmp_path):
    frames = 250
    source = tmp_path / "subject001_baseline_masked.h5"
    time = np.arange(frames, dtype=np.float64)
    hp = np.broadcast_to(time[None, :], (32, frames))
    lp = np.broadcast_to((100 + 2 * time)[None, :], (32, frames))
    _source(source, hp, lp)
    output = tmp_path / "session.parquet"
    hp_model = _Reconstructor(1, 2)
    lp_model = _Reconstructor(3, 4)
    report = export_session(_record(source), hp_model, lp_model, output)

    measured_hp = np.concatenate(hp_model.voltage)
    measured_lp = np.concatenate(lp_model.voltage)
    assert measured_hp.shape == measured_lp.shape == (frames, 32)
    np.testing.assert_allclose(measured_hp[:, 0], 0.01 * time)
    np.testing.assert_allclose(measured_lp[:, 0], 0.02 * time)
    np.testing.assert_allclose(measured_hp[0], 0.0)
    np.testing.assert_allclose(measured_lp[0], 0.0)

    table = pq.read_table(output)
    assert table.schema.names[:4] == ["hp_s1", "hp_s2", "lp_s1", "lp_s2"]
    assert "pviHP" not in table.schema.names and "pviLP" not in table.schema.names
    assert np.allclose(table["hp_s1"][0].as_py(), 1.0)
    assert np.allclose(table["hp_s2"][0].as_py(), 2.0)
    assert np.allclose(table["lp_s1"][0].as_py(), 3.0)
    assert np.allclose(table["lp_s2"][0].as_py(), 4.0)
    assert report["rows"] == 1
    assert report["reconstructed_window_frames"] == 250


def test_export_supplies_one_rank_one_template_per_component_beat(tmp_path):
    periods = 5
    frames = periods * 50
    source = tmp_path / "subject006_baseline_masked.h5"
    resistance = np.empty((32, frames), dtype=np.float64)
    channel = np.linspace(0.5, 1.5, 32)
    for beat in range(periods):
        waveform = np.sin(np.linspace(0, 2 * np.pi, 50, endpoint=False) + beat)
        resistance[:, beat * 50 : (beat + 1) * 50] = channel[:, None] * waveform
    _source(source, resistance, 2 * resistance)

    hp_model = _BeatContextReconstructor(1, 2)
    lp_model = _BeatContextReconstructor(3, 4)
    export_session(
        _record(source),
        hp_model,
        lp_model,
        tmp_path / "beat-context.parquet",
        chunk_frames=73,
    )
    for model in (hp_model, lp_model):
        measured = np.concatenate(model.voltage)
        supplied = np.concatenate(model.templates)
        expected = _frame_beat_templates(measured)
        np.testing.assert_allclose(supplied, expected)
        for beat in range(periods):
            beat_templates = supplied[beat * 50 : (beat + 1) * 50]
            np.testing.assert_allclose(
                beat_templates,
                np.broadcast_to(beat_templates[0], beat_templates.shape),
            )


def test_overlapping_windows_get_distinct_first_frame_references(tmp_path):
    periods = 6
    frames = periods * 50
    source = tmp_path / "subject002_baseline_masked.h5"
    time = np.arange(frames, dtype=np.float64)
    resistance = np.broadcast_to(time[None, :], (32, frames))
    with h5py.File(source, "w") as handle:
        handle.create_dataset("metadata/num_periods", data=periods)
        handle.create_dataset("metadata/period_length", data=50)
        handle.create_dataset("masks/mask05", data=np.array([[1, 5], [2, 6]]))
        handle.create_dataset("data/pviHP/resistance", data=resistance)
        handle.create_dataset("data/pviLP/resistance", data=resistance)
        handle.create_dataset("data/bp/signal", data=np.arange(frames))
        handle.create_dataset("stats/pviHP/duration", data=np.arange(periods))
        handle.create_dataset("stats/pviHP/tMax", data=np.arange(periods))
    hp_model = _Reconstructor()
    lp_model = _Reconstructor()
    export_session(_record(source), hp_model, lp_model, tmp_path / "overlap.parquet")
    windows = np.concatenate(hp_model.voltage).reshape(2, 250, 32)
    np.testing.assert_allclose(windows[:, 0], 0.0)
    np.testing.assert_allclose(windows[0, 1, 0], 0.01)
    np.testing.assert_allclose(windows[1, 1, 0], 0.01)


def test_export_is_equivalent_across_frame_chunks_and_row_batches(tmp_path):
    """Throughput tuning must not alter values, ordering, IDs, or reports."""

    periods = 7
    frames = periods * 50
    source = tmp_path / "subject001_multibatch_masked.h5"
    time = np.arange(frames, dtype=np.float64)
    channel = np.arange(1, 33, dtype=np.float64)[:, None]
    hp = channel * (0.2 + np.sin(time[None, :] / 17.0))
    lp = channel * (1.5 + time[None, :] / 1000.0)
    masks = np.array([[1, 5], [2, 6], [3, 7]])
    _source(source, hp, lp, periods=periods, masks=masks)

    small_path = tmp_path / "small-batches.parquet"
    large_path = tmp_path / "large-batches.parquet"
    small_report = export_session(
        _record(source),
        _SignalReconstructor(),
        _SignalReconstructor(),
        small_path,
        chunk_frames=37,
        batch_rows=1,
    )
    large_report = export_session(
        _record(source),
        _SignalReconstructor(),
        _SignalReconstructor(),
        large_path,
        chunk_frames=2048,
        batch_rows=16,
    )

    small = pq.read_table(small_path).combine_chunks()
    large = pq.read_table(large_path).combine_chunks()
    assert small.equals(large)
    assert small["mask_start"].to_pylist() == [0, 1, 2]
    assert small["mask_stop"].to_pylist() == [5, 6, 7]
    assert small["sample_id"].to_pylist() == large["sample_id"].to_pylist()
    assert small_report == large_report


def test_row_ranges_compact_in_order_and_merge_to_one_logical_session(tmp_path):
    periods = 7
    frames = periods * 50
    source = tmp_path / "subject001_ranges_masked.h5"
    time = np.arange(frames, dtype=np.float64)
    channel = np.arange(1, 33, dtype=np.float64)[:, None]
    hp = channel * np.sin(time[None, :] / 19.0)
    lp = channel * (0.7 + time[None, :] / 500.0)
    masks = np.array([[1, 5], [2, 6], [3, 7]])
    _source(source, hp, lp, periods=periods, masks=masks)
    record = {**_record(source), "_source_hdf5_sha256": "fixed-source-hash"}

    full_path = tmp_path / "full.parquet"
    full_report = export_session(
        record,
        _SignalReconstructor(),
        _SignalReconstructor(),
        full_path,
        batch_rows=3,
    )

    export_root = tmp_path / "range-export"
    staged_reports = []
    for start, stop in ((0, 2), (2, 3)):
        staging = export_root / "session_shards" / "subject001_baseline" / (
            f"rows-{start:06d}-{stop:06d}.parquet"
        )
        report = export_session(
            record,
            _SignalReconstructor(),
            _SignalReconstructor(),
            staging,
            batch_rows=1,
            row_start=start,
            row_stop=stop,
        )
        staged_reports.append(
            {
                **report,
                "subject": "subject001",
                "session": "baseline",
                "source_name": "subject001_baseline",
                "ring": "US120",
                "staging_shard": str(staging.resolve()),
            }
        )

    compacted, _inventory = compact_session_shards(
        staged_reports, export_root, target_rows=8192, batch_rows=1
    )
    merged = merge_session_ranges(compacted)
    assert len(merged) == 1
    assert merged[0]["rows"] == 3
    assert merged[0]["mask05_windows"] == 3
    assert merged[0]["required_source_frames"] == full_report["required_source_frames"]
    assert merged[0]["components"] == full_report["components"]

    range_table = pq.read_table(export_root / "shards" / "part-00000.parquet")
    full_table = pq.read_table(full_path)
    assert range_table.combine_chunks().equals(full_table.combine_chunks())


def test_export_progress_jsonl_reports_rate_and_eta(
    tmp_path, monkeypatch, capsys
):
    periods = 7
    frames = periods * 50
    source = tmp_path / "subject001_progress_masked.h5"
    time = np.arange(frames, dtype=np.float64)
    resistance = np.broadcast_to(time[None, :], (32, frames))
    masks = np.array([[1, 5], [2, 6], [3, 7]])
    _source(source, resistance, 2.0 * resistance, periods=periods, masks=masks)

    # One start timestamp, two reported batches, then session completion.
    clock = iter([10.0, 14.0, 16.0, 18.0])
    monkeypatch.setattr(export_module.time, "monotonic", lambda: next(clock))
    progress_path = tmp_path / "progress.jsonl"
    output_path = tmp_path / "progress.parquet"
    export_session(
        _record(source),
        _Reconstructor(),
        _Reconstructor(),
        output_path,
        chunk_frames=512,
        batch_rows=1,
        progress_path=progress_path,
        progress_every_batches=2,
    )

    events = [json.loads(line) for line in progress_path.read_text().splitlines()]
    assert [event["event"] for event in events] == [
        "session_started",
        "session_progress",
        "session_progress",
        "session_completed",
    ]
    started, halfway, finished, completed = events
    assert started["rows_total"] == 3
    assert started["batches_total"] == 3
    assert started["chunk_frames"] == 512
    assert started["batch_rows"] == 1

    assert halfway["batch"] == 2
    assert halfway["rows_done"] == 2
    assert halfway["elapsed_seconds"] == 4.0
    assert halfway["rows_per_second"] == 0.5
    assert halfway["eta_seconds"] == 2.0
    assert finished["batch"] == 3
    assert finished["rows_done"] == finished["rows_total"] == 3
    assert finished["elapsed_seconds"] == 6.0
    assert finished["rows_per_second"] == 0.5
    assert finished["eta_seconds"] == 0.0

    assert completed["rows"] == 3
    assert completed["elapsed_seconds"] == 8.0
    assert completed["output"] == str(output_path.resolve())
    for event in events:
        assert event["subject"] == "subject001"
        assert event["session"] == "baseline"
        assert event["source_name"] == "subject001_baseline"
        assert event["timestamp_utc"].endswith("+00:00")
        assert isinstance(event["pid"], int)

    stdout_events = [
        json.loads(line.removeprefix("progress "))
        for line in capsys.readouterr().out.splitlines()
    ]
    assert stdout_events == events
