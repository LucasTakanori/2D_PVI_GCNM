"""Check completed shards while a four-field HP/LP export is running."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import h5py
import numpy as np
import pyarrow.parquet as pq

from gcnm_pvi.pvi_splits import stable_sample_id


def _available_parquet(root: Path) -> list[Path]:
    staging = sorted((root / "session_shards").glob("*.parquet"))
    final = sorted((root / "shards").glob("*.parquet"))
    return staging or final


def _wait_for_shards(root: Path, timeout_seconds: int) -> list[Path]:
    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        paths = _available_parquet(root)
        if paths:
            return paths
        time.sleep(30)
    raise TimeoutError(f"no completed Parquet shard appeared under {root}")


def check(root: Path, registry_path: Path, timeout_seconds: int = 1200) -> dict:
    root = Path(root)
    paths = _wait_for_shards(root, timeout_seconds)
    registry = json.loads(Path(registry_path).read_text(encoding="utf-8"))
    by_source = {record["source_name"]: record for record in registry["records"]}
    checked = []
    for path in paths[:3]:
        parquet = pq.ParquetFile(path)
        names = parquet.schema_arrow.names
        stage_fields = ["hp_s1", "hp_s2", "lp_s1", "lp_s2"]
        if names[:4] != stage_fields or "pviHP" in names or "pviLP" in names:
            raise ValueError(f"incorrect live schema in {path}: {names}")
        table = parquet.read_row_group(
            0,
            columns=[
                *stage_fields, "bp_waveform", "stats", "source_name",
                "sample_id", "mask_start", "mask_stop",
            ],
        ).slice(0, 1)
        if len(table) != 1:
            raise ValueError(f"completed shard contains no rows: {path}")
        row = {name: table[name][0].as_py() for name in table.column_names}
        stages = {
            name: np.asarray(row[name], dtype=np.float32).reshape(40, 40, 250)
            for name in stage_fields
        }
        finite = np.isfinite(stages["hp_s1"])
        if any(not np.array_equal(finite, np.isfinite(value)) for value in stages.values()):
            raise ValueError(f"four-field finite masks differ in {path}")
        if not np.array_equal(np.any(finite, axis=2), np.all(finite, axis=2)):
            raise ValueError(f"time-varying NaN mask in {path}")
        rms = {
            name: float(np.sqrt(np.mean(value[np.isfinite(value)] ** 2)))
            for name, value in stages.items()
        }
        if not all(np.isfinite(value) and value > 0 for value in rms.values()):
            raise ValueError(f"invalid live stage RMS in {path}")

        source_name = row["source_name"]
        record = by_source[source_name]
        start, stop = int(row["mask_start"]), int(row["mask_stop"])
        expected_id = stable_sample_id(source_name, "mask05", start, stop)
        if row["sample_id"] != expected_id:
            raise ValueError(f"sample ID mismatch in {path}")
        with h5py.File(record["source_hdf5"], "r") as handle:
            bp = np.asarray(handle["data/bp/signal"], dtype=np.float32).reshape(-1, 50)
            expected_bp = bp[stop - 1]
            expected_stats = np.vstack(
                (
                    np.asarray(handle["stats/pviHP/duration"]).reshape(-1),
                    np.asarray(handle["stats/pviHP/tMax"]).reshape(-1),
                )
            )[:, start:stop].astype(np.float32).reshape(-1)
        np.testing.assert_array_equal(np.asarray(row["bp_waveform"]), expected_bp)
        np.testing.assert_array_equal(np.asarray(row["stats"]), expected_stats)
        checked.append(
            {
                "path": str(path.resolve()),
                "source_name": source_name,
                "sample_id": row["sample_id"],
                "stage_rms": rms,
                "bp_stats_source_parity": True,
            }
        )
    return {
        "schema": "pvi-gcnm-live-export-check-v1",
        "root": str(root.resolve()),
        "completed_shards_visible": len(paths),
        "checked": checked,
        "valid": True,
    }


def main() -> None:
    project = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument(
        "--registry",
        type=Path,
        default=project / "data/registries/main_b045_v1.json",
    )
    parser.add_argument("--timeout-seconds", type=int, default=1200)
    args = parser.parse_args()
    report = check(args.root, args.registry, args.timeout_seconds)
    output = args.root / "live_validation_5min.json"
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
