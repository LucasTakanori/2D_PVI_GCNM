from datetime import datetime, timezone

import numpy as np
import pytest
import torch

from gcnm_pvi.benchmark_batch_inference import (
    _load_voltage,
    _run_chunks,
    build_parser,
    code_provenance,
    default_output_path,
    latency_summary,
    parity_metrics,
    selected_data_sha256,
    write_json_exclusive,
)


def test_cli_defaults_cover_streaming_and_one_second_batch():
    args = build_parser().parse_args([])
    assert args.frames == 50
    assert args.chunk_sizes == [1, 50]
    assert args.target_hz == 50.0
    assert args.physics_workers == 16
    assert args.inference_batch_size == 50
    assert not args.compute_stage2_residuals
    assert args.npz_key == "V"
    assert args.model_name == "coordinate_direct_projected_fine"
    assert args.forward_backend == "dense"
    assert not args.allow_config_hash_mismatch
    assert args.max_parity_relative_l2 == 1e-6
    assert args.max_parity_absolute == 1e-7


def test_latency_and_parity_reporting():
    summary = latency_summary([0.1, 0.2, 0.3])
    assert summary["count"] == 3
    assert summary["total_seconds"] == pytest.approx(0.6)
    assert summary["median_seconds"] == pytest.approx(0.2)
    parity = parity_metrics(np.array([[1.0, 2.0]]), np.array([[1.0, 1.0]]))
    assert parity["maximum_absolute"] == pytest.approx(1.0)
    assert parity["mean_absolute"] == pytest.approx(0.5)


def test_selected_data_hash_covers_shape_dtype_and_values():
    values = np.arange(12, dtype=np.float64).reshape(3, 4)
    assert selected_data_sha256(values) == selected_data_sha256(values.copy())
    assert selected_data_sha256(values) != selected_data_sha256(values.astype(np.float32))
    assert selected_data_sha256(values) != selected_data_sha256(values.reshape(4, 3))


def test_npz_beat_provenance_is_validated_and_reported(tmp_path):
    path = tmp_path / "beat.npz"
    np.savez(
        path,
        V=np.arange(160, dtype=np.float32).reshape(5, 32),
        anatomy_id=np.full(5, 8),
        beat_id=np.full(5, 40),
        sample_index=np.arange(5),
    )
    args = build_parser().parse_args(
        ["--input-npz", str(path), "--frames", "5"]
    )
    voltage, provenance, _, source = _load_voltage(args)
    assert voltage.shape == (5, 32)
    assert source == path
    assert provenance["beat_audit"] == {
        "anatomy_id": 8,
        "beat_id": 40,
        "sample_index_first": 0,
        "sample_index_last": 4,
        "ordered_consecutive": True,
        "single_anatomy": True,
        "single_beat": True,
    }

    np.savez(
        path,
        V=np.arange(160, dtype=np.float32).reshape(5, 32),
        anatomy_id=np.full(5, 8),
        beat_id=np.array([40, 40, 41, 41, 41]),
        sample_index=np.arange(5),
    )
    with pytest.raises(ValueError, match="one anatomy and one beat"):
        _load_voltage(args)


def test_output_is_timestamped_and_never_overwritten(tmp_path):
    moment = datetime(2026, 9, 9, 12, 30, 0, 123456, tzinfo=timezone.utc)
    assert default_output_path(moment).name == "gcnm_batch50_20260909T123000.123456Z.json"
    output = tmp_path / "benchmark.json"
    write_json_exclusive(output, {"answer": 42})
    with pytest.raises(FileExistsError):
        write_json_exclusive(output, {"answer": 43})


def test_code_provenance_gracefully_reports_non_repository(tmp_path):
    provenance = code_provenance(tmp_path)
    assert provenance["available"] is False
    assert provenance["repository_root"] == str(tmp_path)
    assert provenance["error"]


def test_run_chunks_returns_one_record_per_chunk_in_input_order():
    class StubReconstructor:
        def __init__(self):
            self.calls = []

        def reconstruct_batch(self, voltage, **controls):
            self.calls.append((voltage.copy(), controls))
            return (
                voltage[:, :1].copy(),
                (2.0 * voltage[:, :1]).copy(),
                {"timings_seconds": {"total": 0.001}},
            )

    voltage = np.arange(10, dtype=np.float64).reshape(5, 2)
    stub = StubReconstructor()
    result = _run_chunks(
        stub,
        voltage,
        2,
        torch.device("cpu"),
        model_batch_size=50,
        physics_workers=16,
        compute_stage2_residuals=False,
    )
    assert len(result) == 6
    stage_1, stage_2, latencies, total, chunk_frames, components = result
    np.testing.assert_array_equal(np.concatenate(stage_1), voltage[:, :1])
    np.testing.assert_array_equal(np.concatenate(stage_2), 2.0 * voltage[:, :1])
    assert chunk_frames == [2, 2, 1]
    assert len(latencies) == 3
    assert total >= sum(latencies)
    assert components == {"total": [0.001, 0.001, 0.001]}
    assert [call[1]["physics_workers"] for call in stub.calls] == [2, 2, 1]


def test_slurm_launcher_has_inference_resource_contract():
    launcher = (
        default_output_path().parents[2]
        / "slurm/launch_benchmark_gcnm_batch50.sh"
    ).read_text(encoding="utf-8")
    assert "#SBATCH --gres=gpu:1" in launcher
    assert "#SBATCH --cpus-per-task=16" in launcher
    assert "#SBATCH --mem=250GB" in launcher
    assert 'source "${REPO_ROOT}/env/cluster.env"' in launcher
    assert "--chunk-sizes 1 50" in launcher
    assert '--forward-backend "${FORWARD_BACKEND}"' in launcher
