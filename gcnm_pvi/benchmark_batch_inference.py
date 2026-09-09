"""Benchmark warmed one-frame and buffered-batch GCNM reconstruction.

The benchmark exercises the public :class:`CoordinateReconstructor` exactly as
an online caller would.  It intentionally lives outside the reconstruction
runtime so timing instrumentation cannot change the model or physics path.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import resource
import statistics
import subprocess
import sys
import time
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Sequence

import h5py
import numpy as np
import torch

from gcnm_pvi.mesh_registry import sha256_file
from gcnm_pvi.representations import CoordinateReconstructor


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT_NPZ = ROOT / "data/differential_US120_1000beats_clean_v1/test.npz"


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be positive")
    return parsed


def build_parser() -> argparse.ArgumentParser:
    """Build the CLI parser separately so its contract can be unit tested."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs/rings_b045/US120.yaml",
        help="Ring YAML used by the reconstruction runtime.",
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        default=(
            ROOT
            / "models/differential_US120_1000beats_v1/coordinate_direct_projected_fine"
        ),
    )
    parser.add_argument("--model-name", default="coordinate_direct_projected_fine")
    parser.add_argument(
        "--physics-mesh-mode",
        choices=("auto", "coarse", "projected_fine"),
        default="auto",
    )
    parser.add_argument(
        "--forward-backend",
        choices=("dense", "sparse"),
        default="dense",
        help="Numerical backend for the unchanged FEM forward equations.",
    )
    parser.add_argument(
        "--cross-backend-reference",
        choices=("none", "dense", "sparse"),
        default="none",
        help=(
            "After timing, reconstruct the same frames with another FEM backend "
            "and report direct end-to-end output parity."
        ),
    )
    parser.add_argument("--device", default="auto")

    source = parser.add_mutually_exclusive_group()
    source.add_argument(
        "--source-hdf5",
        type=Path,
        help="HDF5 containing resistance channels by sample.",
    )
    source.add_argument(
        "--input-npz",
        type=Path,
        help="NPZ containing an already scaled frames-by-channels voltage array.",
    )
    source.add_argument(
        "--use-registry-source",
        action="store_true",
        help="Resolve source-name from registry instead of using the default test NPZ.",
    )
    parser.add_argument(
        "--registry",
        type=Path,
        default=ROOT / "data/registries/main_b045_v1.json",
        help="Used to resolve the default HDF5 and, unless overridden, config.",
    )
    parser.add_argument("--source-name", default="subject006_baseline")
    parser.add_argument("--hdf5-dataset", default="data/pviHP/resistance")
    parser.add_argument("--npz-key", default="V")
    parser.add_argument("--voltage-scale", type=float, default=0.01)
    parser.add_argument("--start-frame", type=int, default=0)
    parser.add_argument("--frames", type=_positive_int, default=50)

    parser.add_argument(
        "--chunk-sizes",
        nargs="+",
        type=_positive_int,
        default=[1, 50],
        help="Caller-side frame counts; 1 and 50 compare streaming with 1-s buffering.",
    )
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--repeats", type=_positive_int, default=1)
    parser.add_argument("--physics-workers", type=_positive_int, default=16)
    parser.add_argument("--inference-batch-size", type=_positive_int, default=50)
    parser.add_argument(
        "--reference-frames",
        type=int,
        default=1,
        help="Dense historical frame-wise parity sample; zero disables it.",
    )
    parser.add_argument("--target-hz", type=float, default=50.0)
    parser.add_argument(
        "--compute-stage2-residuals",
        action="store_true",
        help="Include optional final forward-residual diagnostics in timed calls.",
    )
    parser.add_argument(
        "--allow-config-hash-mismatch",
        action="store_true",
        help=(
            "Permit a stale checkpoint config hash after auditing mesh hashes, "
            "hyper_pvi, lambda_lm, and graph connectivity."
        ),
    )
    parser.add_argument("--max-parity-relative-l2", type=float, default=1e-6)
    parser.add_argument(
        "--max-parity-absolute",
        type=float,
        default=1e-7,
        help="Maximum allowed conductivity difference in S/m.",
    )
    parser.add_argument(
        "--hash-source-file",
        action="store_true",
        help="Also SHA256 the complete source file (the selected-frame hash is always saved).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Exclusive JSON destination; defaults to reports/inference_benchmarks.",
    )
    return parser


def _percentile(values: Sequence[float], percentile: float) -> float:
    if not values:
        return math.nan
    return float(np.percentile(np.asarray(values, dtype=np.float64), percentile))


def latency_summary(values: Sequence[float]) -> dict[str, float | int]:
    """Return stable latency statistics in seconds."""

    if not values:
        raise ValueError("latency summary requires at least one value")
    return {
        "count": len(values),
        "total_seconds": float(sum(values)),
        "mean_seconds": float(statistics.fmean(values)),
        "median_seconds": float(statistics.median(values)),
        "p95_seconds": _percentile(values, 95.0),
        "minimum_seconds": float(min(values)),
        "maximum_seconds": float(max(values)),
    }


def parity_metrics(candidate: np.ndarray, reference: np.ndarray) -> dict[str, float]:
    """Quantify numerical equivalence without hiding a zero reference norm."""

    candidate = np.asarray(candidate, dtype=np.float64)
    reference = np.asarray(reference, dtype=np.float64)
    if candidate.shape != reference.shape:
        raise ValueError(
            f"parity shapes differ: candidate={candidate.shape}, reference={reference.shape}"
        )
    difference = candidate - reference
    return {
        "relative_l2": float(
            np.linalg.norm(difference)
            / max(float(np.linalg.norm(reference)), 1e-12)
        ),
        "maximum_absolute": float(np.max(np.abs(difference))) if difference.size else 0.0,
        "mean_absolute": float(np.mean(np.abs(difference))) if difference.size else 0.0,
    }


def selected_data_sha256(voltage: np.ndarray) -> str:
    """Hash shape, dtype, and values of the exact frames sent to the model."""

    contiguous = np.ascontiguousarray(voltage)
    digest = hashlib.sha256()
    digest.update(str(contiguous.shape).encode("ascii"))
    digest.update(contiguous.dtype.str.encode("ascii"))
    digest.update(memoryview(contiguous).cast("B"))
    return digest.hexdigest()


def default_output_path(now: datetime | None = None) -> Path:
    moment = now or datetime.now(timezone.utc)
    stamp = moment.strftime("%Y%m%dT%H%M%S.%fZ")
    return ROOT / "reports/inference_benchmarks" / f"gcnm_batch50_{stamp}.json"


def write_json_exclusive(path: Path, payload: dict) -> None:
    """Write a result once; benchmark evidence is never silently replaced."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")


def _process_peak_rss_bytes() -> int:
    peak = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    # Linux reports KiB; macOS reports bytes.
    return peak if sys.platform == "darwin" else peak * 1024


def _synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


@contextmanager
def _runtime_environment(args: argparse.Namespace):
    values = {
        "GCNM_PHYSICS_WORKERS": str(args.physics_workers),
        "GCNM_INFERENCE_BATCH_SIZE": str(args.inference_batch_size),
        "GCNM_COMPUTE_STAGE2_RESIDUALS": (
            "1" if args.compute_stage2_residuals else "0"
        ),
    }
    old = {key: os.environ.get(key) for key in values}
    os.environ.update(values)
    try:
        yield values
    finally:
        for key, value in old.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def _load_registry_record(path: Path, source_name: str) -> dict:
    registry = json.loads(Path(path).read_text(encoding="utf-8"))
    matches = [row for row in registry["records"] if row["source_name"] == source_name]
    if len(matches) != 1:
        raise ValueError(
            f"expected one registry row for {source_name!r}, found {len(matches)}"
        )
    return matches[0]


def _load_voltage(args: argparse.Namespace) -> tuple[np.ndarray, dict, Path, Path]:
    if args.start_frame < 0:
        raise ValueError("start-frame cannot be negative")
    if args.input_npz is not None or (
        args.source_hdf5 is None and not args.use_registry_source
    ):
        source_path = (args.input_npz or DEFAULT_INPUT_NPZ).resolve()
        with np.load(source_path, allow_pickle=False) as archive:
            if args.npz_key not in archive:
                raise KeyError(f"NPZ has no array named {args.npz_key!r}")
            all_voltage = np.asarray(archive[args.npz_key], dtype=np.float64)
            selection = slice(args.start_frame, args.start_frame + args.frames)
            selected_metadata = {
                key: np.asarray(archive[key][selection])
                for key in ("anatomy_id", "beat_id", "sample_index")
                if key in archive
            }
        if all_voltage.ndim != 2:
            raise ValueError("NPZ voltage must be a frames-by-channels 2-D array")
        voltage = all_voltage[selection]
        provenance = {"kind": "npz_voltage", "array_key": args.npz_key}
        if selected_metadata:
            missing = {
                "anatomy_id", "beat_id", "sample_index"
            } - selected_metadata.keys()
            if missing:
                raise ValueError(
                    "NPZ must provide all beat-audit arrays when any are present; "
                    f"missing {sorted(missing)}"
                )
            anatomy = selected_metadata["anatomy_id"]
            beat = selected_metadata["beat_id"]
            sample_index = selected_metadata["sample_index"]
            if len(anatomy) != args.frames:
                raise ValueError("NPZ beat-audit arrays are shorter than selected voltage")
            anatomy_values = np.unique(anatomy)
            beat_values = np.unique(beat)
            if len(anatomy_values) != 1 or len(beat_values) != 1:
                raise ValueError(
                    "selected NPZ frames must belong to one anatomy and one beat"
                )
            if len(sample_index) > 1 and not np.all(np.diff(sample_index) == 1):
                raise ValueError("selected NPZ sample_index must be strictly consecutive")
            provenance["beat_audit"] = {
                "anatomy_id": int(anatomy_values[0]),
                "beat_id": int(beat_values[0]),
                "sample_index_first": int(sample_index[0]),
                "sample_index_last": int(sample_index[-1]),
                "ordered_consecutive": True,
                "single_anatomy": True,
                "single_beat": True,
            }
        config = args.config
    else:
        record = None
        if args.source_hdf5 is None:
            record = _load_registry_record(args.registry, args.source_name)
            source_path = Path(record["source_hdf5"]).resolve()
        else:
            source_path = args.source_hdf5.resolve()
        with h5py.File(source_path, "r") as handle:
            dataset = handle[args.hdf5_dataset]
            stop = args.start_frame + args.frames
            resistance = np.asarray(
                dataset[:, args.start_frame:stop], dtype=np.float64
            )
        voltage = args.voltage_scale * resistance.T
        provenance = {
            "kind": "hdf5_resistance",
            "dataset": args.hdf5_dataset,
            "voltage_scale": args.voltage_scale,
            "registry": str(args.registry.resolve()) if record is not None else None,
            "source_name": args.source_name if record is not None else None,
        }
        config = args.config
    if len(voltage) != args.frames:
        raise ValueError(
            f"requested {args.frames} frames at {args.start_frame}, loaded {len(voltage)}"
        )
    if not np.all(np.isfinite(voltage)):
        raise ValueError("selected voltage contains non-finite values")
    return (
        np.ascontiguousarray(voltage, dtype=np.float64),
        provenance,
        Path(config),
        source_path,
    )


def _resolve_device(value: str) -> torch.device:
    if value == "auto":
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    device = torch.device(value)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"CUDA device requested but CUDA is unavailable: {value}")
    return device


def _run_chunks(
    reconstructor: CoordinateReconstructor,
    voltage: np.ndarray,
    chunk_size: int,
    device: torch.device,
    *,
    model_batch_size: int,
    physics_workers: int,
    compute_stage2_residuals: bool,
) -> tuple[
    list[np.ndarray],
    list[np.ndarray],
    list[float],
    float,
    list[int],
    dict[str, list[float]],
]:
    stage_1, stage_2, chunk_latencies = [], [], []
    chunk_frames: list[int] = []
    component_latencies: dict[str, list[float]] = {}
    _synchronize(device)
    total_started = time.perf_counter()
    for first in range(0, len(voltage), chunk_size):
        chunk = voltage[first : first + chunk_size]
        _synchronize(device)
        started = time.perf_counter()
        result = reconstructor.reconstruct_batch(
            chunk,
            model_batch_size=model_batch_size,
            physics_workers=min(physics_workers, len(chunk)),
            compute_stage2_residuals=compute_stage2_residuals,
        )
        _synchronize(device)
        chunk_latencies.append(time.perf_counter() - started)
        stage_1.append(result[0])
        stage_2.append(result[1])
        chunk_frames.append(len(chunk))
        for name, seconds in result[2].get("timings_seconds", {}).items():
            component_latencies.setdefault(name, []).append(float(seconds))
    _synchronize(device)
    total_seconds = time.perf_counter() - total_started
    return (
        stage_1,
        stage_2,
        chunk_latencies,
        total_seconds,
        chunk_frames,
        component_latencies,
    )


def _memory_begin(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)


def _memory_end(device: torch.device) -> dict[str, int | None]:
    result: dict[str, int | None] = {
        "process_peak_rss_bytes": _process_peak_rss_bytes(),
        "cuda_peak_allocated_bytes": None,
        "cuda_peak_reserved_bytes": None,
    }
    if device.type == "cuda":
        result["cuda_peak_allocated_bytes"] = int(torch.cuda.max_memory_allocated(device))
        result["cuda_peak_reserved_bytes"] = int(torch.cuda.max_memory_reserved(device))
    return result


def benchmark_scenario(
    reconstructor: CoordinateReconstructor,
    voltage: np.ndarray,
    *,
    chunk_size: int,
    warmups: int,
    repeats: int,
    target_hz: float,
    device: torch.device,
    model_batch_size: int,
    physics_workers: int,
    compute_stage2_residuals: bool,
) -> tuple[dict, tuple[np.ndarray, np.ndarray]]:
    """Run one chunking scenario and retain the last outputs for parity."""

    if warmups < 0:
        raise ValueError("warmups cannot be negative")
    for _ in range(warmups):
        warm_voltage = voltage[: min(chunk_size, len(voltage))]
        reconstructor.reconstruct_batch(
            warm_voltage,
            model_batch_size=model_batch_size,
            physics_workers=min(physics_workers, len(warm_voltage)),
            compute_stage2_residuals=compute_stage2_residuals,
        )
        _synchronize(device)

    _memory_begin(device)
    totals: list[float] = []
    all_chunks: list[float] = []
    all_chunk_frames: list[int] = []
    all_components: dict[str, list[float]] = {}
    last_output: tuple[np.ndarray, np.ndarray] | None = None
    for _ in range(repeats):
        stage_1, stage_2, chunks, total, chunk_frames, components = _run_chunks(
            reconstructor,
            voltage,
            chunk_size,
            device,
            model_batch_size=model_batch_size,
            physics_workers=physics_workers,
            compute_stage2_residuals=compute_stage2_residuals,
        )
        last_output = (np.concatenate(stage_1), np.concatenate(stage_2))
        totals.append(total)
        all_chunks.extend(chunks)
        all_chunk_frames.extend(chunk_frames)
        for name, values in components.items():
            all_components.setdefault(name, []).extend(values)
    assert last_output is not None
    timed_frames = len(voltage) * repeats
    total_timed_seconds = float(sum(totals))
    frames_per_second = timed_frames / total_timed_seconds
    mean_chunk_seconds = statistics.fmean(all_chunks)
    budget_fractions = [
        seconds / (frames / target_hz)
        for seconds, frames in zip(all_chunks, all_chunk_frames)
    ]
    report = {
        "chunk_size_frames": chunk_size,
        "warmup_calls": warmups,
        "repeats": repeats,
        "timed_frames": timed_frames,
        "total_latency": latency_summary(totals),
        "chunk_latency": latency_summary(all_chunks),
        "component_latency": {
            name: latency_summary(values)
            for name, values in sorted(all_components.items())
        },
        "frames_per_second": frames_per_second,
        "effective_hz": frames_per_second,
        "per_frame_latency_seconds": total_timed_seconds / timed_frames,
        "target_hz": target_hz,
        "meets_target_hz": frames_per_second >= target_hz,
        "compute_budget_fraction_at_target_hz": (
            statistics.fmean(budget_fractions)
        ),
        "p95_compute_budget_fraction_at_target_hz": (
            _percentile(budget_fractions, 95.0)
        ),
        "maximum_compute_budget_fraction_at_target_hz": (
            max(budget_fractions)
        ),
        "p95_meets_target_hz": _percentile(budget_fractions, 95.0) <= 1.0,
        "maximum_meets_target_hz": max(budget_fractions) <= 1.0,
        "buffer_fill_seconds_at_target_hz": chunk_size / target_hz,
        "estimated_buffered_delivery_seconds": (
            chunk_size / target_hz + mean_chunk_seconds
        ),
        "peak_memory": _memory_end(device),
    }
    return report, last_output


def _checkpoint_hashes(args: argparse.Namespace) -> dict[str, str]:
    return {
        f"stage_{stage + 1}": sha256_file(
            args.checkpoint_dir / f"{args.model_name}_{stage}.pt"
        )
        for stage in range(2)
    }


def _file_identity(path: Path, *, complete_hash: bool) -> dict:
    stat = path.stat()
    result = {
        "path": str(path),
        "size_bytes": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
        "sha256": None,
    }
    if complete_hash:
        result["sha256"] = sha256_file(path)
    return result


def code_provenance(root: Path = ROOT) -> dict:
    """Report the exact Git state, or an explicit reason it is unavailable."""

    try:
        def git(*arguments: str) -> str:
            return subprocess.run(
                ["git", *arguments],
                cwd=root,
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip()

        status = git("status", "--porcelain", "--untracked-files=normal")
        return {
            "available": True,
            "repository_root": str(Path(root).resolve()),
            "head_commit": git("rev-parse", "HEAD"),
            "branch": git("rev-parse", "--abbrev-ref", "HEAD"),
            "dirty": bool(status),
            "status_porcelain": status.splitlines(),
        }
    except (OSError, subprocess.CalledProcessError) as error:
        return {
            "available": False,
            "repository_root": str(Path(root).resolve()),
            "error": str(error),
        }


def run(args: argparse.Namespace) -> tuple[dict, Path]:
    if args.reference_frames < 0:
        raise ValueError("reference-frames cannot be negative")
    if args.target_hz <= 0:
        raise ValueError("target-hz must be positive")
    if args.max_parity_relative_l2 < 0 or args.max_parity_absolute < 0:
        raise ValueError("parity thresholds cannot be negative")
    if args.warmups < 0:
        raise ValueError("warmups cannot be negative")
    chunk_sizes = list(dict.fromkeys(args.chunk_sizes))
    if 1 not in chunk_sizes or 50 not in chunk_sizes:
        raise ValueError("chunk-sizes must include both 1 and 50")
    voltage, provenance, config, source_path = _load_voltage(args)
    device = _resolve_device(args.device)

    with _runtime_environment(args) as runtime_environment:
        _synchronize(device)
        _memory_begin(device)
        initialization_started = time.perf_counter()
        requested_physics_mode = (
            None if args.physics_mesh_mode == "auto" else args.physics_mesh_mode
        )
        with CoordinateReconstructor(
            config,
            args.checkpoint_dir,
            args.model_name,
            device=device,
            physics_mesh_mode=requested_physics_mode,
            allow_config_hash_mismatch=args.allow_config_hash_mismatch,
            physics_workers=args.physics_workers,
            forward_backend=args.forward_backend,
        ) as reconstructor:
            _synchronize(device)
            initialization_seconds = time.perf_counter() - initialization_started
            initialization_memory = _memory_end(device)
            physics_mesh_mode = reconstructor.physics_mesh_mode

            scenarios: dict[int, dict] = {}
            outputs: dict[int, tuple[np.ndarray, np.ndarray]] = {}
            for chunk_size in chunk_sizes:
                scenario, output = benchmark_scenario(
                    reconstructor,
                    voltage,
                    chunk_size=chunk_size,
                    warmups=args.warmups,
                    repeats=args.repeats,
                    target_hz=args.target_hz,
                    device=device,
                    model_batch_size=args.inference_batch_size,
                    physics_workers=args.physics_workers,
                    compute_stage2_residuals=args.compute_stage2_residuals,
                )
                scenarios[chunk_size] = scenario
                outputs[chunk_size] = output

            framewise = outputs[1]
            for chunk_size in chunk_sizes:
                scenarios[chunk_size]["parity_vs_framewise"] = {
                    f"stage_{stage + 1}": parity_metrics(
                        outputs[chunk_size][stage], framewise[stage]
                    )
                    for stage in range(2)
                }

            reference_count = min(args.reference_frames, len(voltage))
            algorithm_reference_report = None
            if reference_count:
                _synchronize(device)
                started = time.perf_counter()
                reference = reconstructor.reconstruct_reference(voltage[:reference_count])
                _synchronize(device)
                reference_seconds = time.perf_counter() - started
                algorithm_reference_report = {
                    "frames": reference_count,
                    "latency_seconds": reference_seconds,
                    "parity": {
                        f"stage_{stage + 1}": parity_metrics(
                            framewise[stage][:reference_count], reference[stage]
                        )
                        for stage in range(2)
                    },
                }

        cross_backend_report = None
        if args.cross_backend_reference != "none":
            if args.cross_backend_reference == args.forward_backend:
                raise ValueError(
                    "cross-backend-reference must differ from forward-backend"
                )
            _synchronize(device)
            started = time.perf_counter()
            with CoordinateReconstructor(
                config,
                args.checkpoint_dir,
                args.model_name,
                device=device,
                physics_mesh_mode=requested_physics_mode,
                allow_config_hash_mismatch=args.allow_config_hash_mismatch,
                physics_workers=args.physics_workers,
                forward_backend=args.cross_backend_reference,
            ) as reference_reconstructor:
                reference_stage_1, reference_stage_2, _ = (
                    reference_reconstructor.reconstruct_batch(
                        voltage,
                        model_batch_size=args.inference_batch_size,
                        physics_workers=args.physics_workers,
                        compute_stage2_residuals=False,
                    )
                )
            _synchronize(device)
            cross_backend_report = {
                "candidate_backend": args.forward_backend,
                "reference_backend": args.cross_backend_reference,
                "frames": len(voltage),
                "latency_seconds": time.perf_counter() - started,
                "parity": {
                    "stage_1": parity_metrics(
                        outputs[50][0], reference_stage_1
                    ),
                    "stage_2": parity_metrics(
                        outputs[50][1], reference_stage_2
                    ),
                },
            }

    parity_items = [
        metric
        for scenario in scenarios.values()
        for metric in scenario["parity_vs_framewise"].values()
    ]
    if algorithm_reference_report is not None:
        parity_items.extend(algorithm_reference_report["parity"].values())
    if cross_backend_report is not None:
        parity_items.extend(cross_backend_report["parity"].values())
    maximum_parity_relative = max(item["relative_l2"] for item in parity_items)
    maximum_parity_absolute = max(item["maximum_absolute"] for item in parity_items)
    parity_pass = (
        maximum_parity_relative <= args.max_parity_relative_l2
        and maximum_parity_absolute <= args.max_parity_absolute
    )

    timestamp = datetime.now(timezone.utc)
    report = {
        "schema_version": 1,
        "created_utc": timestamp.isoformat(),
        "purpose": "warmed one-frame versus 50-frame buffered GCNM inference",
        "code": code_provenance(),
        "initialization": {
            "latency_seconds": initialization_seconds,
            "peak_memory": initialization_memory,
        },
        "runtime": {
            "device": str(device),
            "device_name": (
                torch.cuda.get_device_name(device)
                if device.type == "cuda"
                else platform.processor()
            ),
            "torch_version": torch.__version__,
            "torch_cuda_version": torch.version.cuda,
            "python_version": platform.python_version(),
            "platform": platform.platform(),
            "physics_mesh_mode": physics_mesh_mode,
            "forward_backend": args.forward_backend,
            "allow_config_hash_mismatch": args.allow_config_hash_mismatch,
            "environment": runtime_environment,
        },
        "inputs": {
            "config": str(config.resolve()),
            "config_sha256": sha256_file(config),
            "checkpoint_dir": str(args.checkpoint_dir.resolve()),
            "model_name": args.model_name,
            "checkpoint_sha256": _checkpoint_hashes(args),
            "source": _file_identity(source_path, complete_hash=args.hash_source_file),
            "provenance": provenance,
            "start_frame": args.start_frame,
            "frames": len(voltage),
            "channels": voltage.shape[1],
            "selected_voltage_sha256": selected_data_sha256(voltage),
        },
        "scenarios": {str(key): value for key, value in scenarios.items()},
        "framewise_algorithm_reference": algorithm_reference_report,
        "cross_backend_reference": cross_backend_report,
        "comparison": {
            "batch50_speedup_vs_framewise": (
                scenarios[1]["per_frame_latency_seconds"]
                / scenarios[50]["per_frame_latency_seconds"]
            ),
            "batch50_effective_hz": scenarios[50]["effective_hz"],
            "batch50_meets_target_hz": scenarios[50]["meets_target_hz"],
        },
        "acceptance": {
            "parity": {
                "maximum_relative_l2": maximum_parity_relative,
                "maximum_absolute": maximum_parity_absolute,
                "relative_l2_at_most": args.max_parity_relative_l2,
                "absolute_at_most_s_per_m": args.max_parity_absolute,
                "pass": parity_pass,
            },
            "batch50_target_hz": {
                "target_hz": args.target_hz,
                "mean_pass": scenarios[50]["meets_target_hz"],
                "p95_pass": scenarios[50]["p95_meets_target_hz"],
                "maximum_pass": scenarios[50]["maximum_meets_target_hz"],
            },
            "pass": parity_pass,
        },
    }
    output = args.output.resolve() if args.output else default_output_path(timestamp)
    write_json_exclusive(output, report)
    return report, output


def main(argv: Iterable[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    report, output = run(args)
    print(json.dumps(report, indent=2, sort_keys=True), flush=True)
    print(f"saved immutable benchmark report: {output}", flush=True)
    if not report["acceptance"]["parity"]["pass"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
