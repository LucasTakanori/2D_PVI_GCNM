"""Render subject GIFs from the literal serialized coordinate-direct Parquet rows."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import numpy as np

from gcnm_pvi.coordinate_direct_representation import (
    compose_s1_s2_ds2,
    representation_diagnostics,
)
from gcnm_pvi.mesh_registry import sha256_file
from gcnm_pvi.pilot_newton_compatible import comparison_metrics
from gcnm_pvi.pvi_splits import stable_sample_id
from gcnm_pvi.subject_coordinate_direct_pilot import _render_gif
from gcnm_pvi.three_beat_gifs import _read_exported_sample


PERIOD_LENGTH = 50


def _image(table, column: str) -> np.ndarray:
    values = table[column][0].values.to_numpy(zero_copy_only=False)
    return np.asarray(values, dtype=np.float32).reshape(1, 40, 40, 250)[0]


def render_session(record: dict, parquet_root: Path, output_root: Path) -> dict:
    source = Path(record["source_hdf5"])
    with h5py.File(source, "r") as handle:
        masks = np.asarray(handle["masks/mask05"], dtype=np.int64)
        masks[:, 0] -= 1
        mask_row = len(masks) // 2
        start, stop = (int(value) for value in masks[mask_row])
        if stop - start != 5:
            raise ValueError("selected Parquet GIF row is not mask05")
        frames = slice(start * PERIOD_LENGTH, stop * PERIOD_LENGTH)
        archived_hp = np.asarray(handle["data/pviHP/img"][:, :, frames], dtype=np.float32)
        archived_lp = np.asarray(handle["data/pviLP/img"][:, :, frames], dtype=np.float32)
    sample_id = stable_sample_id(record["source_name"], "mask05", start, stop)
    table = _read_exported_sample(parquet_root / "shards", sample_id, ["s1", "s2"])
    s1 = _image(table, "s1")
    s2 = _image(table, "s2")
    channels = compose_s1_s2_ds2(s1, s2)
    d_s2 = channels[2]
    session_root = output_root / record["source_name"]
    session_root.mkdir(parents=True, exist_ok=False)
    gif = session_root / "parquet_archived_pvi_s1_s2_ds2_three_beats.gif"
    _render_gif(
        gif,
        [archived_hp, archived_lp, s1, s2, d_s2],
        ["Archived PVI HP", "Archived PVI LP", "Parquet S1", "Parquet S2", "Parquet dS2/dt"],
        source_name=f"{record['source_name']} | literal Parquet row",
    )
    report = {
        "schema": "pvi-gcnm-coordinate-direct-parquet-gif-v1",
        "subject": record["subject"],
        "session": record["session"],
        "source_name": record["source_name"],
        "source_hdf5": str(source.resolve()),
        "source_hdf5_sha256": sha256_file(source),
        "parquet_root": str(parquet_root.resolve()),
        "parquet_manifest_sha256": sha256_file(parquet_root / "manifest.json"),
        "sample_id": sample_id,
        "mask05_row": mask_row,
        "mask_period_bounds_zero_based": [start, stop],
        "channels": ["s1", "s2", "d_s2_dt"],
        "representation_diagnostics": representation_diagnostics(channels),
        "archived_pvi_is_behavioral_reference_not_truth": True,
        "behavioral_comparisons": {
            "s1_vs_archived_hp": comparison_metrics(s1, archived_hp),
            "s2_vs_archived_hp": comparison_metrics(s2, archived_hp),
        },
        "gif": str(gif.resolve()),
    }
    (session_root / "report.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    return report


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registry", type=Path, default=root / "data/registries/main_b045_v1.json")
    parser.add_argument("--parquet-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--subject", default="subject006")
    parser.add_argument("--sessions", nargs="+", default=["baseline", "valsalva", "pressor"])
    args = parser.parse_args()
    if args.output_root.exists():
        raise FileExistsError(f"immutable Parquet GIF output exists: {args.output_root}")
    manifest = json.loads((args.parquet_root / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("schema") != "pvi-gcnm-coordinate-direct-parquet-v1":
        raise ValueError("Parquet GIF requires coordinate-direct v1 schema")
    registry = json.loads(args.registry.read_text(encoding="utf-8"))
    records = registry.get("records", registry.get("sessions", []))
    selected = [
        record for record in records
        if record["subject"] == args.subject and record["session"] in set(args.sessions)
    ]
    selected.sort(key=lambda record: args.sessions.index(record["session"]))
    if len(selected) != len(args.sessions):
        raise ValueError("registry is missing a requested subject/session")
    args.output_root.mkdir(parents=True, exist_ok=False)
    reports = [render_session(record, args.parquet_root, args.output_root) for record in selected]
    summary = {
        "schema": "pvi-gcnm-coordinate-direct-parquet-gif-summary-v1",
        "subject": args.subject,
        "sessions": args.sessions,
        "gifs": [report["gif"] for report in reports],
        "reports": reports,
    }
    (args.output_root / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({"status": "pass", "gifs": summary["gifs"]}, indent=2))


if __name__ == "__main__":
    main()
