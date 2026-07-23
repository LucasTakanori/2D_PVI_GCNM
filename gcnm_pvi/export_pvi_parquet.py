"""Export immutable HP/LP, two-stage GCNM representations from PVI HDF5.

HP and LP are reconstructed by separately trained GCNMs.  Each ``mask05``
window uses its first archived frame as the shared component reference, and
the Parquet columns retain literal component/stage names: ``hp_s1``,
``hp_s2``, ``lp_s1``, and ``lp_s2``.
"""

from __future__ import annotations

import argparse
import json
import multiprocessing
import os
import time
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

import h5py
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch

from gcnm_pvi.mesh_registry import sha256_file
from gcnm_pvi.hp_lp_signals import component_reference, resistance_to_voltage
from gcnm_pvi.pvi_splits import stable_sample_id
from gcnm_pvi.representations import (
    CoordinateReconstructor,
    DiffusionSlotReconstructor,
    GlobalVoltageVesselSlotReconstructor,
    rasterize_frames,
)


SCHEMA_VERSION = 3
PERIOD_LENGTH = 50
WINDOW_PERIODS = 5
IMAGE_SIDE = 40
_WORKER_DEVICE: str | None = None
_WORKER_RECONSTRUCTORS: dict[tuple[str, str, str], object] = {}


def _progress_event(progress_path: Path | None, event: str, **fields) -> None:
    """Emit one immediately visible human line and one append-only JSON event."""

    payload = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "event": event,
        "pid": os.getpid(),
        **fields,
    }
    print("progress " + json.dumps(payload, sort_keys=True), flush=True)
    if progress_path is not None:
        line = (json.dumps(payload, sort_keys=True) + "\n").encode("utf-8")
        descriptor = os.open(
            Path(progress_path), os.O_CREAT | os.O_APPEND | os.O_WRONLY, 0o600
        )
        try:
            os.write(descriptor, line)
        finally:
            os.close(descriptor)


def _stable_metric(value: float) -> float:
    """Normalize insignificant float32 reduction-order noise in manifests."""

    return float(np.round(float(value), 7))


def _rank_one_beat_template(voltage: np.ndarray) -> np.ndarray:
    """Match the synthetic generator's rank-one 50-sample beat context."""

    values = np.asarray(voltage, dtype=np.float64)
    centered = values - np.mean(values, axis=0, keepdims=True)
    if not np.any(np.abs(centered) > 1e-15):
        return np.zeros(values.shape[1], dtype=np.float64)
    u, singular, vh = np.linalg.svd(centered, full_matrices=False)
    spatial = vh[0].copy()
    temporal = singular[0] * u[:, 0].copy()
    anchor = int(np.argmax(np.abs(spatial)))
    if spatial[anchor] < 0:
        spatial *= -1.0
        temporal *= -1.0
    return temporal[int(np.argmax(temporal))] * spatial


def _frame_beat_templates(voltage: np.ndarray) -> np.ndarray:
    """Repeat one denoised template across each complete archived beat."""

    values = np.asarray(voltage, dtype=np.float64)
    if len(values) % PERIOD_LENGTH:
        raise ValueError("beat-context inference requires complete 50-sample periods")
    output = np.empty_like(values)
    for start in range(0, len(values), PERIOD_LENGTH):
        template = _rank_one_beat_template(values[start : start + PERIOD_LENGTH])
        output[start : start + PERIOD_LENGTH] = template
    return output


def _schema() -> pa.Schema:
    pvi_length = IMAGE_SIDE * IMAGE_SIDE * PERIOD_LENGTH * WINDOW_PERIODS
    return pa.schema(
        [
            pa.field("hp_s1", pa.list_(pa.float32(), pvi_length)),
            pa.field("hp_s2", pa.list_(pa.float32(), pvi_length)),
            pa.field("lp_s1", pa.list_(pa.float32(), pvi_length)),
            pa.field("lp_s2", pa.list_(pa.float32(), pvi_length)),
            pa.field("bp_waveform", pa.list_(pa.float32(), PERIOD_LENGTH)),
            pa.field("stats", pa.list_(pa.float32(), 2 * WINDOW_PERIODS)),
            pa.field("subject", pa.string()),
            pa.field("session", pa.string()),
            pa.field("source_name", pa.string()),
            pa.field("sample_id", pa.string()),
            pa.field("source_order", pa.int32()),
            pa.field("mask_start", pa.int32()),
            pa.field("mask_stop", pa.int32()),
            pa.field("num_periods", pa.int32()),
        ]
    )


def _validate_images(stage_1: np.ndarray, stage_2: np.ndarray) -> dict:
    if stage_1.shape != stage_2.shape or stage_1.shape[:2] != (40, 40):
        raise ValueError(f"invalid rasterized stage shapes {stage_1.shape}, {stage_2.shape}")
    finite_1, finite_2 = np.isfinite(stage_1), np.isfinite(stage_2)
    if not np.array_equal(finite_1, finite_2):
        raise ValueError("stage finite masks differ")
    pixel_ever_finite = np.any(finite_1, axis=2)
    pixel_always_finite = np.all(finite_1, axis=2)
    if not np.array_equal(pixel_ever_finite, pixel_always_finite):
        raise ValueError("NaNs vary over time; they must occur only outside the mesh")
    rms_1 = float(np.sqrt(np.mean(stage_1[finite_1] ** 2)))
    rms_2 = float(np.sqrt(np.mean(stage_2[finite_2] ** 2)))
    if not np.isfinite(rms_1 + rms_2) or rms_1 <= 0 or rms_2 <= 0:
        raise ValueError("stage activation RMS must be finite and non-zero")
    return {
        "stage_1_activation_rms": rms_1,
        "stage_2_activation_rms": rms_2,
        "stage_1_activation_sumsq": float(np.sum(stage_1[finite_1] ** 2)),
        "stage_2_activation_sumsq": float(np.sum(stage_2[finite_2] ** 2)),
        "stage_1_activation_count": int(np.count_nonzero(finite_1)),
        "stage_2_activation_count": int(np.count_nonzero(finite_2)),
    }


def _fixed_list(values: np.ndarray, width: int) -> pa.FixedSizeListArray:
    values = np.ascontiguousarray(values, dtype=np.float32).reshape(-1)
    return pa.FixedSizeListArray.from_arrays(pa.array(values, type=pa.float32()), width)


def _write_row_batch(
    writer: pq.ParquetWriter,
    record: dict,
    selected: np.ndarray,
    hp_s1: np.ndarray,
    hp_s2: np.ndarray,
    lp_s1: np.ndarray,
    lp_s2: np.ndarray,
    bp: np.ndarray,
    stats: np.ndarray,
    num_periods: int,
) -> None:
    """Write one bounded batch without constructing Python float lists."""

    schema = _schema()
    pvi_width = IMAGE_SIDE * IMAGE_SIDE * PERIOD_LENGTH * WINDOW_PERIODS
    count = len(selected)
    waveforms = np.stack([bp[stop - 1] for _start, stop in selected])
    statistics = np.stack(
        [stats[:, start:stop].reshape(-1) for start, stop in selected]
    )
    sample_ids = [
        stable_sample_id(record["source_name"], "mask05", start, stop)
        for start, stop in selected
    ]
    table = pa.Table.from_arrays(
        [
            _fixed_list(hp_s1.reshape(count, -1), pvi_width),
            _fixed_list(hp_s2.reshape(count, -1), pvi_width),
            _fixed_list(lp_s1.reshape(count, -1), pvi_width),
            _fixed_list(lp_s2.reshape(count, -1), pvi_width),
            _fixed_list(waveforms, PERIOD_LENGTH),
            _fixed_list(statistics, 2 * WINDOW_PERIODS),
            pa.array([record["subject"]] * count, type=pa.string()),
            pa.array([record["session"]] * count, type=pa.string()),
            pa.array([record["source_name"]] * count, type=pa.string()),
            pa.array(sample_ids, type=pa.string()),
            pa.array([int(record["source_order"])] * count, type=pa.int32()),
            pa.array(selected[:, 0], type=pa.int32()),
            pa.array(selected[:, 1], type=pa.int32()),
            pa.array([num_periods] * count, type=pa.int32()),
        ],
        schema=schema,
    )
    writer.write_table(table)


def _reconstruct_component(
    reconstructor,
    windows: np.ndarray,
    *,
    chunk_frames: int,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Reconstruct a bounded ``rows x 250 x measurements`` component batch."""

    values = np.asarray(windows, dtype=np.float64)
    if values.ndim != 3 or values.shape[1] != PERIOD_LENGTH * WINDOW_PERIODS:
        raise ValueError(f"invalid component window batch {values.shape}")
    # The first retained frame is frozen by the synthetic contract.  Do this
    # separately for every overlapping mask05 window.
    values = component_reference(values, reference_index=0, axis=1)
    flat = values.reshape(-1, values.shape[-1])
    templates = None
    if getattr(reconstructor, "requires_beat_context", False):
        templates = np.stack([_frame_beat_templates(window) for window in values])
        templates = templates.reshape(flat.shape)
    stage_1_parts, stage_2_parts = [], []
    residual_1, residual_2 = [], []
    residual_2_stride = None
    for start in range(0, len(flat), chunk_frames):
        stop = min(start + chunk_frames, len(flat))
        if templates is None:
            stage_1, stage_2, report = reconstructor.reconstruct(flat[start:stop])
        else:
            stage_1, stage_2, report = reconstructor.reconstruct(
                flat[start:stop], templates[start:stop]
            )
        stage_1_parts.append(stage_1)
        stage_2_parts.append(stage_2)
        residual_1.extend(report["stage_1_forward_voltage_rms"])
        residual_2.extend(report["stage_2_forward_voltage_rms"])
        chunk_stride = int(report.get("stage_2_forward_voltage_rms_stride", 1))
        if residual_2_stride is None:
            residual_2_stride = chunk_stride
        elif residual_2_stride != chunk_stride:
            raise RuntimeError("stage-2 diagnostic stride changed within one export")
    rows = len(values)
    stage_1_img = rasterize_frames(
        reconstructor.mappings, np.concatenate(stage_1_parts)
    )
    stage_2_img = rasterize_frames(
        reconstructor.mappings, np.concatenate(stage_2_parts)
    )
    checks = _validate_images(stage_1_img, stage_2_img)
    shape = (IMAGE_SIDE, IMAGE_SIDE, rows, PERIOD_LENGTH * WINDOW_PERIODS)
    stage_1_rows = np.moveaxis(stage_1_img.reshape(shape), 2, 0)
    stage_2_rows = np.moveaxis(stage_2_img.reshape(shape), 2, 0)
    return stage_1_rows, stage_2_rows, {
        **checks,
        "stage_1_forward_voltage_rms_values": residual_1,
        "stage_2_forward_voltage_rms_values": residual_2,
        "stage_2_forward_voltage_rms_stride": int(residual_2_stride or 1),
        "input_voltage_rms_values": np.sqrt(np.mean(flat * flat, axis=1)).tolist(),
    }


def export_session(
    record: dict,
    hp_reconstructor,
    lp_reconstructor,
    output: Path,
    *,
    chunk_frames: int = 256,
    batch_rows: int = 8,
    progress_path: Path | None = None,
    progress_every_batches: int = 1,
    row_start: int = 0,
    row_stop: int | None = None,
) -> dict:
    if progress_every_batches <= 0:
        raise ValueError("progress_every_batches must be positive")
    source_path = Path(record["source_hdf5"])
    with h5py.File(source_path, "r") as handle:
        num_periods = int(np.asarray(handle["metadata/num_periods"]).item())
        period_length = int(np.asarray(handle["metadata/period_length"]).item())
        if period_length != PERIOD_LENGTH:
            raise ValueError(f"{source_path} period length is {period_length}, expected 50")
        masks = np.asarray(handle["masks/mask05"], dtype=np.int64)
        masks[:, 0] -= 1
        if np.any(masks[:, 1] - masks[:, 0] != WINDOW_PERIODS):
            raise ValueError(f"{source_path} contains a non-five-period mask05 window")
        resistance_hp = np.asarray(handle["data/pviHP/resistance"], dtype=np.float64)
        resistance_lp = np.asarray(handle["data/pviLP/resistance"], dtype=np.float64)
        if resistance_hp.shape != resistance_lp.shape:
            raise ValueError(f"{source_path} HP/LP resistance shapes differ")
        bp = np.asarray(handle["data/bp/signal"], dtype=np.float32).reshape(num_periods, period_length)
        stats = np.vstack(
            (
                np.asarray(handle["stats/pviHP/duration"]).reshape(-1),
                np.asarray(handle["stats/pviHP/tMax"]).reshape(-1),
            )
        ).astype(np.float32)

    session_rows = len(masks)
    row_stop = session_rows if row_stop is None else int(row_stop)
    row_start = int(row_start)
    if row_start < 0 or row_stop <= row_start or row_stop > session_rows:
        raise ValueError(
            f"invalid mask row range [{row_start}, {row_stop}) for {session_rows} rows"
        )
    masks = masks[row_start:row_stop]

    output.parent.mkdir(parents=True, exist_ok=True)
    metrics = {
        component: {
            "stage_1_activation_rms": [],
            "stage_2_activation_rms": [],
            "stage_1_activation_sumsq": 0.0,
            "stage_2_activation_sumsq": 0.0,
            "stage_1_activation_count": 0,
            "stage_2_activation_count": 0,
            "stage_1_forward_voltage_rms": [],
            "stage_2_forward_voltage_rms": [],
            "stage_2_forward_voltage_rms_stride": None,
            "input_voltage_rms": [],
        }
        for component in ("hp", "lp")
    }
    started = time.monotonic()
    total_batches = (len(masks) + batch_rows - 1) // batch_rows
    _progress_event(
        progress_path,
        "session_started",
        subject=record["subject"],
        session=record["session"],
        source_name=record["source_name"],
        rows_total=len(masks),
        session_rows_total=session_rows,
        row_start=row_start,
        row_stop=row_stop,
        batches_total=total_batches,
        chunk_frames=chunk_frames,
        batch_rows=batch_rows,
        device=_WORKER_DEVICE,
    )
    with pq.ParquetWriter(output, _schema(), compression="zstd") as writer:
        for first in range(0, len(masks), batch_rows):
            selected = masks[first : first + batch_rows]
            frame_rows = np.stack(
                [
                    np.arange(start * period_length, stop * period_length)
                    for start, stop in selected
                ]
            )
            hp_voltage = resistance_to_voltage(resistance_hp[:, frame_rows]).transpose(1, 2, 0)
            lp_voltage = resistance_to_voltage(resistance_lp[:, frame_rows]).transpose(1, 2, 0)
            hp_s1, hp_s2, hp_report = _reconstruct_component(
                hp_reconstructor, hp_voltage, chunk_frames=chunk_frames
            )
            lp_s1, lp_s2, lp_report = _reconstruct_component(
                lp_reconstructor, lp_voltage, chunk_frames=chunk_frames
            )
            _write_row_batch(
                writer,
                record,
                selected,
                hp_s1,
                hp_s2,
                lp_s1,
                lp_s2,
                bp,
                stats,
                num_periods,
            )
            for component, report in (("hp", hp_report), ("lp", lp_report)):
                metrics[component]["stage_1_activation_rms"].append(
                    report["stage_1_activation_rms"]
                )
                metrics[component]["stage_2_activation_rms"].append(
                    report["stage_2_activation_rms"]
                )
                for stage in ("stage_1", "stage_2"):
                    metrics[component][f"{stage}_activation_sumsq"] += report[
                        f"{stage}_activation_sumsq"
                    ]
                    metrics[component][f"{stage}_activation_count"] += report[
                        f"{stage}_activation_count"
                    ]
                metrics[component]["stage_1_forward_voltage_rms"].extend(
                    report["stage_1_forward_voltage_rms_values"]
                )
                metrics[component]["stage_2_forward_voltage_rms"].extend(
                    report["stage_2_forward_voltage_rms_values"]
                )
                stride = int(report["stage_2_forward_voltage_rms_stride"])
                current_stride = metrics[component]["stage_2_forward_voltage_rms_stride"]
                if current_stride is None:
                    metrics[component]["stage_2_forward_voltage_rms_stride"] = stride
                elif current_stride != stride:
                    raise RuntimeError("stage-2 diagnostic stride changed within a session")
                metrics[component]["input_voltage_rms"].extend(
                    report["input_voltage_rms_values"]
                )
            batch_index = first // batch_rows + 1
            if batch_index % progress_every_batches == 0 or batch_index == total_batches:
                elapsed = max(time.monotonic() - started, 1e-9)
                rows_done = min(first + len(selected), len(masks))
                rate = rows_done / elapsed
                eta = (len(masks) - rows_done) / rate if rate > 0 else None
                gpu_memory = None
                if torch.cuda.is_available() and _WORKER_DEVICE is not None:
                    gpu_memory = int(torch.cuda.max_memory_allocated(_WORKER_DEVICE))
                _progress_event(
                    progress_path,
                    "session_progress",
                    subject=record["subject"],
                    session=record["session"],
                    source_name=record["source_name"],
                    batch=batch_index,
                    batches_total=total_batches,
                    rows_done=rows_done,
                    rows_total=len(masks),
                    session_rows_total=session_rows,
                    row_start=row_start,
                    row_stop=row_stop,
                    elapsed_seconds=elapsed,
                    rows_per_second=rate,
                    eta_seconds=eta,
                    gpu_max_memory_bytes=gpu_memory,
                    device=_WORKER_DEVICE,
                )
    summary = {}
    for component, component_metrics in metrics.items():
        summary[component] = {
            "stage_1_activation_rms": _stable_metric(
                np.sqrt(
                    component_metrics["stage_1_activation_sumsq"]
                    / component_metrics["stage_1_activation_count"]
                )
            ),
            "stage_2_activation_rms": _stable_metric(
                np.sqrt(
                    component_metrics["stage_2_activation_sumsq"]
                    / component_metrics["stage_2_activation_count"]
                )
            ),
            "stage_1_forward_voltage_rms": _stable_metric(
                np.mean(component_metrics["stage_1_forward_voltage_rms"])
            ),
            "stage_1_forward_voltage_rms_samples": len(
                component_metrics["stage_1_forward_voltage_rms"]
            ),
            "stage_2_forward_voltage_rms": (
                _stable_metric(np.mean(component_metrics["stage_2_forward_voltage_rms"]))
                if component_metrics["stage_2_forward_voltage_rms"]
                else None
            ),
            "stage_2_forward_voltage_rms_samples": len(
                component_metrics["stage_2_forward_voltage_rms"]
            ),
            "stage_2_forward_voltage_rms_stride": int(
                component_metrics["stage_2_forward_voltage_rms_stride"] or 1
            ),
            "input_voltage_rms": _stable_metric(
                np.mean(component_metrics["input_voltage_rms"])
            ),
            "input_voltage_rms_samples": len(component_metrics["input_voltage_rms"]),
        }
    required_periods = np.unique(
        np.concatenate([np.arange(start, stop) for start, stop in masks])
    )
    report = {
        "source_hdf5": str(source_path.resolve()),
        "source_hdf5_sha256": record.get("_source_hdf5_sha256") or sha256_file(source_path),
        "source_frames": num_periods * period_length,
        "required_source_frames": len(required_periods) * period_length,
        "reconstructed_window_frames": len(masks) * period_length * WINDOW_PERIODS,
        "mask05_windows": len(masks),
        "rows": len(masks),
        "row_start": row_start,
        "row_stop": row_stop,
        "_required_periods": required_periods.tolist(),
        "components": summary,
    }
    _progress_event(
        progress_path,
        "session_completed",
        subject=record["subject"],
        session=record["session"],
        source_name=record["source_name"],
        rows=len(masks),
        session_rows_total=session_rows,
        row_start=row_start,
        row_stop=row_stop,
        elapsed_seconds=time.monotonic() - started,
        output=str(output.resolve()),
        device=_WORKER_DEVICE,
    )
    return report


def _init_session_worker(gpu_count: int) -> None:
    global _WORKER_DEVICE
    identity = multiprocessing.current_process()._identity
    worker_index = (identity[0] - 1) if identity else 0
    _WORKER_DEVICE = f"cuda:{worker_index % gpu_count}"
    os.environ["GCNM_PHYSICS_WORKERS"] = os.environ.get(
        "GCNM_EXPORT_PHYSICS_WORKERS", "1"
    )


def _export_session_worker(payload: dict) -> dict:
    record = payload["record"]
    family = payload["family"]
    ring = record["ring"]
    reconstructors = {}
    for component in ("hp", "lp"):
        key = (family, component, ring)
        reconstructor = _WORKER_RECONSTRUCTORS.get(key)
        if reconstructor is not None:
            reconstructors[component] = reconstructor
            continue
        model_name = f"{family}_{component}_b045_{ring}_seed0"
        model_dir = Path(payload["checkpoint_root"]) / component / ring
        if family == "coordinate":
            reconstructor = CoordinateReconstructor(
                Path(record["config"]), model_dir, model_name, device=_WORKER_DEVICE
            )
        elif family == "global_voltage_slots":
            reconstructor = GlobalVoltageVesselSlotReconstructor(
                Path(record["config"]),
                model_dir / f"{model_name}_localizer.pt",
                model_dir / f"{model_name}_refiner.pt",
                device=_WORKER_DEVICE,
            )
        elif family == "diffusion":
            prior_root = Path(payload["prior_root"])
            reconstructor = DiffusionSlotReconstructor(
                Path(record["config"]),
                model_dir / f"{model_name}_localizer.pt",
                model_dir / f"{model_name}_refiner.pt",
                prior_root / ring / "default_finger_prior.npz",
                device=_WORKER_DEVICE,
            )
        else:
            raise ValueError(f"unsupported representation family {family}")
        _WORKER_RECONSTRUCTORS[key] = reconstructor
        reconstructors[component] = reconstructor
    if family == "global_voltage_slots":
        # HP and LP use different learned weights but exactly the same ring
        # physics.  These objects are read-only during inference and the two
        # components run sequentially inside a worker, so sharing avoids a
        # duplicate baseline solve, regularizer eigensystem, and FEM cache.
        reconstructors["lp"].fixed_stage = reconstructors["hp"].fixed_stage
        reconstructors["lp"].nonlinear_solver = reconstructors["hp"].nonlinear_solver
        reconstructors["lp"].parallel_physics = reconstructors["hp"].parallel_physics
    raw_row_start = payload.get("row_start")
    row_start = 0 if raw_row_start is None else int(raw_row_start)
    row_stop = payload.get("row_stop")
    if row_stop is None:
        shard = (
            Path(payload["output_root"])
            / "session_shards"
            / f"{record['source_name']}.parquet"
        )
    else:
        shard = (
            Path(payload["output_root"])
            / "session_shards"
            / record["source_name"]
            / f"rows-{row_start:06d}-{int(row_stop):06d}.parquet"
        )
    report = export_session(
        record,
        reconstructors["hp"],
        reconstructors["lp"],
        shard,
        chunk_frames=int(payload["chunk_frames"]),
        batch_rows=int(payload["batch_rows"]),
        progress_path=(
            None if payload.get("progress_path") is None else Path(payload["progress_path"])
        ),
        progress_every_batches=int(payload.get("progress_every_batches", 1)),
        row_start=row_start,
        row_stop=None if row_stop is None else int(row_stop),
    )
    return {
        **report,
        "subject": record["subject"],
        "session": record["session"],
        "source_name": record["source_name"],
        "ring": ring,
        "staging_shard": str(shard.resolve()),
    }


def merge_session_ranges(reports: list[dict]) -> list[dict]:
    """Merge ordered row-range reports back into six logical source sessions."""

    grouped: dict[str, list[dict]] = {}
    for report in reports:
        grouped.setdefault(report["source_name"], []).append(report)
    merged_reports = []
    for source_name, parts in grouped.items():
        parts.sort(key=lambda item: int(item.get("row_start", 0)))
        expected_start = 0
        for part in parts:
            if int(part.get("row_start", 0)) != expected_start:
                raise RuntimeError(f"non-contiguous export ranges for {source_name}")
            expected_start = int(part.get("row_stop", expected_start + part["rows"]))
        first = parts[0]
        required_periods = sorted(
            {period for part in parts for period in part.get("_required_periods", [])}
        )
        rows = sum(int(part["rows"]) for part in parts)
        components = {}
        for component in ("hp", "lp"):
            values = [part["components"][component] for part in parts]
            activation_weights = np.asarray([part["rows"] for part in parts], dtype=float)
            summary = {}
            for stage in ("stage_1", "stage_2"):
                activation = np.asarray(
                    [value[f"{stage}_activation_rms"] for value in values], dtype=float
                )
                summary[f"{stage}_activation_rms"] = _stable_metric(
                    np.sqrt(np.average(activation * activation, weights=activation_weights))
                )
                metric = f"{stage}_forward_voltage_rms"
                sample_key = f"{metric}_samples"
                sample_weights = np.asarray([value[sample_key] for value in values], dtype=float)
                summary[metric] = (
                    _stable_metric(
                        np.average(
                            [value[metric] for value in values], weights=sample_weights
                        )
                    )
                    if np.sum(sample_weights) > 0
                    else None
                )
                summary[sample_key] = int(np.sum(sample_weights))
            strides = {value["stage_2_forward_voltage_rms_stride"] for value in values}
            if len(strides) != 1:
                raise RuntimeError(f"mixed diagnostic strides for {source_name}/{component}")
            summary["stage_2_forward_voltage_rms_stride"] = int(strides.pop())
            input_weights = np.asarray(
                [value["input_voltage_rms_samples"] for value in values], dtype=float
            )
            summary["input_voltage_rms"] = _stable_metric(
                np.average(
                    [value["input_voltage_rms"] for value in values], weights=input_weights
                )
            )
            summary["input_voltage_rms_samples"] = int(np.sum(input_weights))
            components[component] = summary
        shards = []
        for part in parts:
            for shard in part.get("shards", []):
                if not shards or shards[-1] != shard:
                    shards.append(shard)
        merged = {
            key: first[key]
            for key in (
                "subject",
                "session",
                "source_name",
                "ring",
                "source_hdf5",
                "source_hdf5_sha256",
                "source_frames",
            )
        }
        merged.update(
            {
                "required_source_frames": len(required_periods) * PERIOD_LENGTH,
                "reconstructed_window_frames": sum(
                    int(part["reconstructed_window_frames"]) for part in parts
                ),
                "mask05_windows": rows,
                "rows": rows,
                "components": components,
                "shards": shards,
            }
        )
        merged_reports.append(merged)
    return merged_reports


def compact_session_shards(
    reports: list[dict],
    output_root: Path,
    *,
    target_rows: int = 8192,
    batch_rows: int = 16,
) -> tuple[list[dict], list[dict]]:
    """Stream staging sessions into deterministic, bounded final shards.

    All staging files are retained until every final writer has closed and the
    final metadata row total has been verified.  Thus a failed compaction leaves
    the export's ``_INCOMPLETE`` marker and its recoverable session shards.
    """
    if target_rows <= 0 or batch_rows <= 0:
        raise ValueError("target_rows and batch_rows must be positive")
    if not reports:
        raise ValueError("cannot compact zero session reports")
    staging_paths = [Path(report["staging_shard"]) for report in reports]
    missing = [path for path in staging_paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(missing[0])

    output_root = Path(output_root)
    final_root = output_root / "shards"
    final_root.mkdir(parents=True, exist_ok=False)
    schema = pq.ParquetFile(staging_paths[0]).schema_arrow
    compacted_reports = []
    final_inventory: list[dict] = []
    writer = None
    current_path = None
    current_rows = 0
    total_rows = 0

    def open_writer():
        nonlocal writer, current_path, current_rows
        current_path = final_root / f"part-{len(final_inventory):05d}.parquet"
        writer = pq.ParquetWriter(current_path, schema, compression="zstd")
        current_rows = 0

    def close_writer():
        nonlocal writer, current_path, current_rows
        if writer is None:
            return
        writer.close()
        metadata_rows = int(pq.ParquetFile(current_path).metadata.num_rows)
        if metadata_rows != current_rows:
            raise RuntimeError(
                f"compacted shard row mismatch for {current_path}: "
                f"metadata={metadata_rows}, written={current_rows}"
            )
        final_inventory.append(
            {"path": str(current_path.resolve()), "rows": metadata_rows}
        )
        writer = None
        current_path = None
        current_rows = 0

    try:
        for report, staging_path in zip(reports, staging_paths):
            parquet = pq.ParquetFile(staging_path)
            if not parquet.schema_arrow.equals(schema):
                raise ValueError(f"staging shard schema differs: {staging_path}")
            expected_rows = int(report["rows"])
            if int(parquet.metadata.num_rows) != expected_rows:
                raise ValueError(
                    f"staging shard row mismatch for {staging_path}: "
                    f"metadata={parquet.metadata.num_rows}, report={expected_rows}"
                )
            session_final_paths: list[str] = []
            session_rows = 0
            for batch in parquet.iter_batches(batch_size=batch_rows):
                table = pa.Table.from_batches([batch], schema=schema)
                offset = 0
                while offset < len(table):
                    if writer is None:
                        open_writer()
                    take = min(len(table) - offset, target_rows - current_rows)
                    writer.write_table(table.slice(offset, take))
                    final_path = str(current_path.resolve())
                    if not session_final_paths or session_final_paths[-1] != final_path:
                        session_final_paths.append(final_path)
                    offset += take
                    current_rows += take
                    session_rows += take
                    total_rows += take
                    if current_rows == target_rows:
                        close_writer()
            if session_rows != expected_rows:
                raise RuntimeError(
                    f"streamed {session_rows} rows for {staging_path}, expected {expected_rows}"
                )
            compacted = dict(report)
            compacted.pop("staging_shard", None)
            compacted["shards"] = session_final_paths
            compacted_reports.append(compacted)
        close_writer()
    finally:
        if writer is not None:
            writer.close()

    expected_total = sum(int(report["rows"]) for report in reports)
    metadata_total = sum(int(item["rows"]) for item in final_inventory)
    if total_rows != expected_total or metadata_total != expected_total:
        raise RuntimeError(
            "compacted total row mismatch: "
            f"streamed={total_rows}, metadata={metadata_total}, expected={expected_total}"
        )

    for staging_path in staging_paths:
        staging_path.unlink()
    staging_roots = {path.parent for path in staging_paths}
    for staging_root in staging_roots:
        try:
            staging_root.rmdir()
        except OSError:
            pass
    return compacted_reports, final_inventory


def _hashes(paths: list[Path]) -> dict[str, str]:
    return {str(path.resolve()): sha256_file(path) for path in paths}


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registry", type=Path, default=root / "data/registries/main_b045_v1.json")
    parser.add_argument(
        "--family",
        choices=["coordinate", "global_voltage_slots", "diffusion"],
        required=True,
    )
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--checkpoint-root", type=Path, required=True)
    parser.add_argument("--prior-root", type=Path)
    parser.add_argument("--subjects", nargs="*")
    parser.add_argument("--sessions", nargs="*")
    parser.add_argument("--chunk-frames", type=int, default=256)
    parser.add_argument("--batch-rows", type=int, default=8)
    parser.add_argument("--session-workers", type=int, default=1)
    parser.add_argument(
        "--row-range-size",
        type=int,
        default=0,
        help="split each session into deterministic mask-row tasks; 0 keeps one task/session",
    )
    parser.add_argument("--gpu-count", type=int, default=1)
    parser.add_argument("--shard-rows", type=int, default=8192)
    parser.add_argument("--compaction-batch-rows", type=int, default=16)
    parser.add_argument("--progress-jsonl", type=Path)
    parser.add_argument("--progress-every-batches", type=int, default=1)
    args = parser.parse_args()
    if args.output_root.exists():
        raise FileExistsError(f"immutable output root already exists: {args.output_root}")
    args.output_root.mkdir(parents=True, exist_ok=False)
    incomplete = args.output_root / "_INCOMPLETE"
    incomplete.write_text("export in progress\n", encoding="utf-8")
    if args.progress_jsonl is None:
        args.progress_jsonl = args.output_root / "progress.jsonl"
    registry = json.loads(args.registry.read_text(encoding="utf-8"))
    selected = registry["records"]
    if args.subjects:
        selected = [r for r in selected if r["subject"] in set(args.subjects)]
    if args.sessions:
        selected = [r for r in selected if r["session"] in set(args.sessions)]
    if not selected:
        raise ValueError("selection contains zero sessions")
    if args.row_range_size < 0:
        raise ValueError("row range size cannot be negative")
    if args.row_range_size and args.session_workers <= 1:
        raise ValueError("row-range export requires more than one worker")
    if args.row_range_size:
        # A range worker must not re-hash the same large HDF5 file.  Resolve the
        # immutable source hashes once in the parent and pass them to all tasks.
        selected = [
            {
                **record,
                "_source_hdf5_sha256": sha256_file(Path(record["source_hdf5"])),
            }
            for record in selected
        ]
    _progress_event(
        args.progress_jsonl,
        "export_started",
        family=args.family,
        sessions=len(selected),
        session_workers=args.session_workers,
        gpu_count=args.gpu_count,
        chunk_frames=args.chunk_frames,
        batch_rows=args.batch_rows,
        row_range_size=args.row_range_size,
        inference_batch_size=int(os.environ.get("GCNM_INFERENCE_BATCH_SIZE", "64")),
    )
    if args.family == "diffusion" and args.prior_root is None:
        raise ValueError("diffusion export requires --prior-root")
    reports = []
    checkpoint_paths: list[Path] = []
    for ring in sorted({record["ring"] for record in selected}):
        for component in ("hp", "lp"):
            model_name = f"{args.family}_{component}_b045_{ring}_seed0"
            model_dir = args.checkpoint_root / component / ring
            suffixes = (
                ("0.pt", "1.pt")
                if args.family == "coordinate"
                else ("localizer.pt", "refiner.pt")
            )
            checkpoint_paths.extend(
                model_dir / f"{model_name}_{suffix}" for suffix in suffixes
            )
    missing_checkpoints = [path for path in checkpoint_paths if not path.is_file()]
    if missing_checkpoints:
        raise FileNotFoundError(f"required checkpoint not found: {missing_checkpoints[0]}")

    if args.session_workers > 1:
        if not torch.cuda.is_available():
            raise RuntimeError("parallel session export requires CUDA")
        payloads = []
        for record in selected:
            ranges: list[tuple[int, int] | tuple[None, None]]
            if args.row_range_size:
                with h5py.File(record["source_hdf5"], "r") as handle:
                    row_count = int(handle["masks/mask05"].shape[0])
                ranges = [
                    (start, min(start + args.row_range_size, row_count))
                    for start in range(0, row_count, args.row_range_size)
                ]
            else:
                ranges = [(None, None)]
            for row_start, row_stop in ranges:
                payloads.append(
                    {
                        "record": record,
                        "family": args.family,
                        "checkpoint_root": str(args.checkpoint_root),
                        "prior_root": (
                            None if args.prior_root is None else str(args.prior_root)
                        ),
                        "output_root": str(args.output_root),
                        "chunk_frames": args.chunk_frames,
                        "batch_rows": args.batch_rows,
                        "row_start": row_start,
                        "row_stop": row_stop,
                        "progress_path": (
                            str(args.progress_jsonl)
                            if args.progress_jsonl is not None
                            else None
                        ),
                        "progress_every_batches": args.progress_every_batches,
                    }
                )
        context = multiprocessing.get_context("spawn")
        with ProcessPoolExecutor(
            max_workers=min(args.session_workers, len(payloads)),
            mp_context=context,
            initializer=_init_session_worker,
            initargs=(args.gpu_count,),
        ) as executor:
            reports = list(executor.map(_export_session_worker, payloads))
    else:
        by_ring_component = {}
        for record in selected:
            ring = record["ring"]
            for component in ("hp", "lp"):
                key = (ring, component)
                if key in by_ring_component:
                    continue
                model_name = f"{args.family}_{component}_b045_{ring}_seed0"
                model_dir = args.checkpoint_root / component / ring
                if args.family == "coordinate":
                    reconstructor = CoordinateReconstructor(
                        record["config"], model_dir, model_name
                    )
                elif args.family == "global_voltage_slots":
                    reconstructor = GlobalVoltageVesselSlotReconstructor(
                        record["config"],
                        model_dir / f"{model_name}_localizer.pt",
                        model_dir / f"{model_name}_refiner.pt",
                    )
                elif args.family == "diffusion":
                    if args.prior_root is None:
                        raise ValueError("diffusion export requires --prior-root")
                    reconstructor = DiffusionSlotReconstructor(
                        record["config"],
                        model_dir / f"{model_name}_localizer.pt",
                        model_dir / f"{model_name}_refiner.pt",
                        args.prior_root / ring / "default_finger_prior.npz",
                    )
                else:
                    raise ValueError(f"unsupported representation family {args.family}")
                by_ring_component[key] = reconstructor
            shard = (
                args.output_root
                / "session_shards"
                / f"{record['source_name']}.parquet"
            )
            report = export_session(
                record,
                by_ring_component[(ring, "hp")],
                by_ring_component[(ring, "lp")],
                shard,
                chunk_frames=args.chunk_frames,
                batch_rows=args.batch_rows,
                progress_path=args.progress_jsonl,
                progress_every_batches=args.progress_every_batches,
            )
            reports.append(
                {
                    **report,
                    "source_name": record["source_name"],
                    "ring": ring,
                    "staging_shard": str(shard.resolve()),
                }
            )
    _progress_event(
        args.progress_jsonl,
        "inference_completed",
        family=args.family,
        sessions=len(selected),
        tasks=len(reports),
        rows=sum(int(item["rows"]) for item in reports),
    )
    _progress_event(
        args.progress_jsonl,
        "compaction_started",
        family=args.family,
        sessions=len(reports),
    )
    reports, parquet_shards = compact_session_shards(
        reports,
        args.output_root,
        target_rows=args.shard_rows,
        batch_rows=args.compaction_batch_rows,
    )
    if args.row_range_size:
        reports = merge_session_ranges(reports)
    manifest = {
        "schema": "pvi-gcnm-hp-lp-parquet-v3",
        "version": SCHEMA_VERSION,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "representation_family": args.family,
        "mesh_variant": "b045",
        "stage_columns": {
            "hp_s1": "HP stage 1",
            "hp_s2": "HP stage 2",
            "lp_s1": "LP stage 1",
            "lp_s2": "LP stage 2",
        },
        "source_signal": {
            "hp_resistance": "data/pviHP/resistance",
            "lp_resistance": "data/pviLP/resistance",
            "sampling_space": "archived cardiac periods resampled to 50 samples per beat",
            "reference_rule": "subtract first frame independently within each mask05 window",
        },
        "beat_context": (
            "rank-one template computed independently from each archived 50-sample beat"
            if args.family == "global_voltage_slots"
            else None
        ),
        "voltage_convention": "delta_V = -I * delta_R; I = -0.01 A",
        "mask_key": "mask05",
        "tensor_shapes": {
            "hp_s1": [1, 40, 40, 250],
            "hp_s2": [1, 40, 40, 250],
            "lp_s1": [1, 40, 40, 250],
            "lp_s2": [1, 40, 40, 250],
            "bp_waveform": [50],
            "stats": [2, 5],
        },
        "row_count": sum(item["rows"] for item in reports),
        "session_count": len(reports),
        "export_execution": {
            "workers": args.session_workers,
            "row_range_size": args.row_range_size,
            "chunk_frames": args.chunk_frames,
            "batch_rows": args.batch_rows,
            "inference_batch_size": int(
                os.environ.get("GCNM_INFERENCE_BATCH_SIZE", "64")
            ),
            "stage_2_forward_residual_stride": (
                0
                if os.environ.get("GCNM_COMPUTE_STAGE2_RESIDUALS", "1").lower()
                in {"0", "false", "no"}
                else int(os.environ.get("GCNM_STAGE2_RESIDUAL_STRIDE", "1"))
            ),
            "fem_implementation": (
                "cached topology/CEM, vectorized assembly, combined direct+reciprocal solve"
            ),
            "stage_2_forward_residual_scope": (
                "disabled during inference; run as post-export validation"
                if os.environ.get("GCNM_COMPUTE_STAGE2_RESIDUALS", "1").lower()
                in {"0", "false", "no"}
                else "deterministic validation subset; all frames reconstructed"
            ),
        },
        "shard_target_rows": args.shard_rows,
        "parquet_shards": parquet_shards,
        "registry": str(args.registry.resolve()),
        "registry_sha256": sha256_file(args.registry),
        "workbook_sha256": registry["workbook_sha256"],
        "checkpoint_hashes": _hashes(checkpoint_paths),
        "config_hashes": _hashes(sorted({Path(record["config"]) for record in selected})),
        "mesh_hashes": _hashes(
            sorted(
                {
                    Path(record[key])
                    for record in selected
                    for key in ("mesh_forward", "mesh_inverse", "mapping_40")
                }
            )
        ),
        "source_sessions": reports,
        "excluded_sessions": [],
    }
    (args.output_root / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    incomplete.unlink()
    _progress_event(
        args.progress_jsonl,
        "export_completed",
        family=args.family,
        sessions=len(reports),
        rows=manifest["row_count"],
        shards=len(parquet_shards),
    )
    print(json.dumps({"rows": manifest["row_count"], "sessions": len(reports)}, indent=2))


if __name__ == "__main__":
    main()
