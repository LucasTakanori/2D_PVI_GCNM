"""Export full-differential coordinate-direct S1/S2 mask05 representations."""

from __future__ import annotations

import argparse
import json
import multiprocessing
import os
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

import h5py
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from gcnm_pvi.export_pvi_parquet import compact_session_shards
from gcnm_pvi.mesh_registry import sha256_file
from gcnm_pvi.pvi_splits import stable_sample_id
from gcnm_pvi.representations import CoordinateReconstructor, rasterize_frames


PERIOD_LENGTH = 50
WINDOW_PERIODS = 5
IMAGE_SIDE = 40
STIM_CURRENT_A = -0.01
SCHEMA = "pvi-gcnm-coordinate-direct-parquet-v1"


def _schema() -> pa.Schema:
    image_width = IMAGE_SIDE * IMAGE_SIDE * PERIOD_LENGTH * WINDOW_PERIODS
    return pa.schema(
        [
            pa.field("s1", pa.list_(pa.float32(), image_width)),
            pa.field("s2", pa.list_(pa.float32(), image_width)),
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


def _fixed_list(values: np.ndarray, width: int) -> pa.FixedSizeListArray:
    flat = np.ascontiguousarray(values, dtype=np.float32).reshape(-1)
    return pa.FixedSizeListArray.from_arrays(pa.array(flat, type=pa.float32()), width)


def _physical_window_voltage(
    resistance_hp: np.ndarray, resistance_lp: np.ndarray
) -> np.ndarray:
    """Return physical-sign delta voltage, referenced per mask05 window."""

    resistance = np.asarray(resistance_hp, dtype=np.float64) + np.asarray(
        resistance_lp, dtype=np.float64
    )
    saved_voltage = -STIM_CURRENT_A * resistance
    return -(saved_voltage - saved_voltage[:, :1, :])


def _stage_rows(reconstructor, voltage: np.ndarray) -> tuple[np.ndarray, np.ndarray, dict]:
    rows, frames, measurements = voltage.shape
    flat = voltage.reshape(rows * frames, measurements)
    stage_1, stage_2, report = reconstructor.reconstruct(flat)
    image_1 = rasterize_frames(reconstructor.mappings, stage_1)
    image_2 = rasterize_frames(reconstructor.mappings, stage_2)
    shape = (IMAGE_SIDE, IMAGE_SIDE, rows, frames)
    image_1 = np.moveaxis(image_1.reshape(shape), 2, 0).astype(np.float32)
    image_2 = np.moveaxis(image_2.reshape(shape), 2, 0).astype(np.float32)
    if image_1.shape != (rows, IMAGE_SIDE, IMAGE_SIDE, frames):
        raise ValueError(f"unexpected S1 shape {image_1.shape}")
    if not np.array_equal(np.isfinite(image_1), np.isfinite(image_2)):
        raise ValueError("S1 and S2 finite masks differ")
    finite = np.isfinite(image_1)
    rms_1 = float(np.sqrt(np.mean(image_1[finite].astype(np.float64) ** 2)))
    rms_2 = float(np.sqrt(np.mean(image_2[finite].astype(np.float64) ** 2)))
    if not np.isfinite(rms_1 + rms_2) or min(rms_1, rms_2) <= 0:
        raise ValueError("S1/S2 activation RMS must be finite and nonzero")
    return image_1, image_2, {
        "s1_rms": rms_1,
        "s2_rms": rms_2,
        "s1_forward_voltage_rms": float(
            np.mean(report["stage_1_forward_voltage_rms"])
        ),
        "s2_forward_voltage_rms": (
            float(np.mean(report["stage_2_forward_voltage_rms"]))
            if len(report["stage_2_forward_voltage_rms"])
            else None
        ),
    }


def export_session(
    record: dict,
    reconstructor: CoordinateReconstructor,
    output: Path,
    *,
    batch_rows: int = 8,
) -> dict:
    source = Path(record["source_hdf5"])
    with h5py.File(source, "r") as handle:
        num_periods = int(np.asarray(handle["metadata/num_periods"]).item())
        period_length = int(np.asarray(handle["metadata/period_length"]).item())
        if period_length != PERIOD_LENGTH:
            raise ValueError(f"{source} period length is not 50")
        masks = np.asarray(handle["masks/mask05"], dtype=np.int64)
        masks[:, 0] -= 1
        if np.any(masks[:, 1] - masks[:, 0] != WINDOW_PERIODS):
            raise ValueError(f"{source} contains a non-five-period mask05 window")
        hp = np.asarray(handle["data/pviHP/resistance"], dtype=np.float64)
        lp = np.asarray(handle["data/pviLP/resistance"], dtype=np.float64)
        bp = np.asarray(handle["data/bp/signal"], dtype=np.float32).reshape(
            num_periods, period_length
        )
        stats = np.vstack(
            (
                np.asarray(handle["stats/pviHP/duration"]).reshape(-1),
                np.asarray(handle["stats/pviHP/tMax"]).reshape(-1),
            )
        ).astype(np.float32)

    output.parent.mkdir(parents=True, exist_ok=True)
    image_width = IMAGE_SIDE * IMAGE_SIDE * PERIOD_LENGTH * WINDOW_PERIODS
    metric_rows = []
    with pq.ParquetWriter(output, _schema(), compression="zstd") as writer:
        for first in range(0, len(masks), batch_rows):
            selected = masks[first : first + batch_rows]
            frame_rows = np.stack(
                [
                    np.arange(start * PERIOD_LENGTH, stop * PERIOD_LENGTH)
                    for start, stop in selected
                ]
            )
            hp_windows = hp[:, frame_rows].transpose(1, 2, 0)
            lp_windows = lp[:, frame_rows].transpose(1, 2, 0)
            voltage = _physical_window_voltage(hp_windows, lp_windows)
            s1, s2, metrics = _stage_rows(reconstructor, voltage)
            metric_rows.append(metrics)
            waveforms = np.stack([bp[stop - 1] for _start, stop in selected])
            statistics = np.stack(
                [stats[:, start:stop].reshape(-1) for start, stop in selected]
            )
            sample_ids = [
                stable_sample_id(record["source_name"], "mask05", start, stop)
                for start, stop in selected
            ]
            count = len(selected)
            table = pa.Table.from_arrays(
                [
                    _fixed_list(s1, image_width),
                    _fixed_list(s2, image_width),
                    _fixed_list(waveforms, PERIOD_LENGTH),
                    _fixed_list(statistics, 2 * WINDOW_PERIODS),
                    pa.array([record["subject"]] * count),
                    pa.array([record["session"]] * count),
                    pa.array([record["source_name"]] * count),
                    pa.array(sample_ids),
                    pa.array([int(record["source_order"])] * count, type=pa.int32()),
                    pa.array(selected[:, 0], type=pa.int32()),
                    pa.array(selected[:, 1], type=pa.int32()),
                    pa.array([num_periods] * count, type=pa.int32()),
                ],
                schema=_schema(),
            )
            writer.write_table(table)
            print(
                json.dumps(
                    {
                        "event": "session_progress",
                        "source_name": record["source_name"],
                        "rows_done": min(first + count, len(masks)),
                        "rows_total": len(masks),
                    }
                ),
                flush=True,
            )
    summary = {
        key: float(np.mean([item[key] for item in metric_rows if item[key] is not None]))
        if any(item[key] is not None for item in metric_rows)
        else None
        for key in metric_rows[0]
    }
    return {
        "subject": record["subject"],
        "session": record["session"],
        "source_name": record["source_name"],
        "ring": record["ring"],
        "source_hdf5": str(source.resolve()),
        "source_hdf5_sha256": sha256_file(source),
        "source_frames": num_periods * PERIOD_LENGTH,
        "mask05_windows": len(masks),
        "rows": len(masks),
        "metrics": summary,
        "staging_shard": str(output.resolve()),
    }


def _worker(payload: dict) -> dict:
    os.environ["GCNM_PHYSICS_WORKERS"] = str(payload["physics_workers"])
    reconstructor = CoordinateReconstructor(
        Path(payload["record"]["config"]),
        Path(payload["checkpoint_dir"]),
        payload["model_name"],
        device="cuda:0",
    )
    return export_session(
        payload["record"],
        reconstructor,
        Path(payload["output"]),
        batch_rows=int(payload["batch_rows"]),
    )


def _expected_session_rows(record: dict) -> tuple[int, int]:
    """Return source period and mask05 counts without loading image tensors."""

    with h5py.File(record["source_hdf5"], "r") as handle:
        periods = int(np.asarray(handle["metadata/num_periods"]).item())
        rows = int(len(handle["masks/mask05"]))
    return periods, rows


def _recover_staging_report(record: dict, output: Path) -> dict | None:
    """Recover a complete session shard left by an interrupted ring export."""

    periods, expected_rows = _expected_session_rows(record)
    if not output.is_file():
        return None
    try:
        parquet_rows = int(pq.ParquetFile(output).metadata.num_rows)
    except (OSError, pa.ArrowInvalid):
        return None
    if parquet_rows != expected_rows:
        return None
    source = Path(record["source_hdf5"])
    return {
        "subject": record["subject"],
        "session": record["session"],
        "source_name": record["source_name"],
        "ring": record["ring"],
        "source_hdf5": str(source.resolve()),
        "source_hdf5_sha256": sha256_file(source),
        "source_frames": periods * PERIOD_LENGTH,
        "mask05_windows": expected_rows,
        "rows": expected_rows,
        "metrics": {"recovered_complete_staging_shard": True},
        "staging_shard": str(output.resolve()),
    }


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registry", type=Path, default=root / "data/registries/main_b045_v1.json")
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--model-name", default="coordinate_direct")
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--subjects", nargs="+", default=[])
    parser.add_argument(
        "--ring",
        choices=[
            "US060", "US065", "US070", "US075", "US080", "US085", "US090",
            "US095", "US100", "US105", "US110", "US115", "US120", "US125",
            "US130",
        ],
        help="export every included subject registered to this ring",
    )
    parser.add_argument("--sessions", nargs="+", default=["baseline", "valsalva", "pressor"])
    parser.add_argument("--session-workers", type=int, default=3)
    parser.add_argument("--physics-workers", type=int, default=20)
    parser.add_argument("--batch-rows", type=int, default=8)
    parser.add_argument("--shard-rows", type=int, default=8192)
    parser.add_argument(
        "--resume",
        action="store_true",
        help="reuse exact-row-count session shards under an incomplete output root",
    )
    args = parser.parse_args()

    incomplete = args.output_root / "_INCOMPLETE"
    if args.resume:
        if not incomplete.is_file() or (args.output_root / "manifest.json").exists():
            raise RuntimeError("resume requires an incomplete root without a final manifest")
        if (args.output_root / "shards").exists():
            raise RuntimeError("resume refuses a root with partial compacted shards")
    else:
        if args.output_root.exists():
            raise FileExistsError(f"immutable output root exists: {args.output_root}")
        args.output_root.mkdir(parents=True)
        incomplete.write_text("export in progress\n", encoding="utf-8")
    registry = json.loads(args.registry.read_text(encoding="utf-8"))
    records = registry.get("records", registry.get("sessions", []))
    if args.ring and args.subjects:
        parser.error("use either --ring or --subjects, not both")
    if args.ring:
        args.subjects = sorted(
            {
                record["subject"]
                for record in records
                if record["ring"] == args.ring and not record.get("exclusion_reason")
            }
        )
    if not args.subjects:
        parser.error("--ring or at least one --subjects value is required")
    selected = [
        record
        for record in records
        if record["subject"] in set(args.subjects)
        and record["session"] in set(args.sessions)
    ]
    selected.sort(key=lambda item: int(item["source_order"]))
    expected = [
        record
        for record in records
        if record["subject"] in set(args.subjects)
        and record["session"] in set(args.sessions)
        and not record.get("exclusion_reason")
    ]
    if {
        (record["subject"], record["session"], record["source_name"])
        for record in selected
    } != {
        (record["subject"], record["session"], record["source_name"])
        for record in expected
    }:
        raise ValueError("registry selection lost one or more requested source sessions")
    found_subjects = {record["subject"] for record in selected}
    missing_subjects = sorted(set(args.subjects) - found_subjects)
    if missing_subjects:
        raise ValueError(f"registry selection contains no session for {missing_subjects}")
    checkpoints = [
        args.checkpoint_dir / f"{args.model_name}_{stage}.pt" for stage in range(2)
    ]
    for checkpoint in checkpoints:
        if not checkpoint.is_file():
            raise FileNotFoundError(checkpoint)

    reports_by_source = {}
    payloads = []
    repair_destinations = {}
    for record in selected:
        destination = (
            args.output_root / "session_shards" / f"{record['source_name']}.parquet"
        )
        recovered = _recover_staging_report(record, destination) if args.resume else None
        if recovered is not None:
            reports_by_source[record["source_name"]] = recovered
            print(json.dumps({"event": "reuse_complete_session", "source_name": record["source_name"], "rows": recovered["rows"]}), flush=True)
            continue
        temporary = destination.with_suffix(".repair.parquet") if args.resume else destination
        if temporary.exists():
            raise FileExistsError(f"stale repair shard exists: {temporary}")
        payloads.append(
            {
                "record": record,
                "checkpoint_dir": str(args.checkpoint_dir),
                "model_name": args.model_name,
                "output": str(temporary),
                "batch_rows": args.batch_rows,
                "physics_workers": args.physics_workers,
            }
        )
        repair_destinations[record["source_name"]] = destination
    generated_reports = []
    if payloads and args.session_workers > 1:
        context = multiprocessing.get_context("spawn")
        with ProcessPoolExecutor(
            max_workers=min(args.session_workers, len(payloads)), mp_context=context
        ) as executor:
            generated_reports = list(executor.map(_worker, payloads))
    elif payloads:
        generated_reports = [_worker(payload) for payload in payloads]
    for report in generated_reports:
        source_name = report["source_name"]
        if args.resume:
            temporary = Path(report["staging_shard"])
            destination = repair_destinations[source_name]
            temporary.replace(destination)
            report["staging_shard"] = str(destination.resolve())
        reports_by_source[source_name] = report
    reports = [reports_by_source[record["source_name"]] for record in selected]
    reports, shards = compact_session_shards(
        reports, args.output_root, target_rows=args.shard_rows, batch_rows=8
    )
    manifest = {
        "schema": SCHEMA,
        "version": 1,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "representation_family": "coordinate_direct",
        "mesh_variant": "b045",
        "mask_key": "mask05",
        "stored_fields": {"s1": "coordinate-direct stage 1", "s2": "coordinate-direct stage 2"},
        "bp_channel_contract": ["s1", "s2", "d_s2_dt"],
        "derivative": "centered temporal difference of S2 with zero-padded boundaries",
        "source_signal": {
            "resistance": "data/pviHP/resistance + data/pviLP/resistance",
            "window_reference": "subtract first frame independently in each mask05 window",
            "physical_voltage": "negative of saved PVI voltage after reference",
            "stim_current_a": STIM_CURRENT_A,
        },
        "tensor_shapes": {
            "s1": [1, 40, 40, 250],
            "s2": [1, 40, 40, 250],
            "bp_waveform": [50],
            "stats": [2, 5],
        },
        "row_count": sum(int(report["rows"]) for report in reports),
        "session_count": len(reports),
        "source_sessions": reports,
        "parquet_shards": shards,
        "checkpoint_hashes": {
            str(path.resolve()): sha256_file(path) for path in checkpoints
        },
        "registry": str(args.registry.resolve()),
        "registry_sha256": sha256_file(args.registry),
        "workbook_sha256": registry.get("workbook_sha256"),
        "export_execution": {
            "session_workers": min(args.session_workers, len(payloads)),
            "physics_workers_per_session": args.physics_workers,
            "inference_batch_size": int(os.environ.get("GCNM_INFERENCE_BATCH_SIZE", "512")),
            "stage_2_forward_residuals": "disabled during export",
        },
        "excluded_sessions": [],
    }
    (args.output_root / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    incomplete.unlink()
    print(json.dumps({"status": "pass", "rows": manifest["row_count"]}, indent=2))


if __name__ == "__main__":
    main()
