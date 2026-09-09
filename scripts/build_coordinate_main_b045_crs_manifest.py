#!/usr/bin/env python3
"""Mirror the frozen 364-run coordinate CRT matrix with CRS as the only change."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-tsv", type=Path, required=True)
    parser.add_argument("--source-json", type=Path, required=True)
    parser.add_argument("--output-tsv", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    args = parser.parse_args()

    with args.source_tsv.open(newline="", encoding="utf-8") as stream:
        source_rows = list(csv.DictReader(stream, delimiter="\t"))
    if len(source_rows) != 364:
        raise ValueError(f"expected 364 source runs, found {len(source_rows)}")
    if [int(row["task_id"]) for row in source_rows] != list(range(364)):
        raise ValueError("source task IDs are not contiguous")
    if {row["architecture"] for row in source_rows} != {"crt"}:
        raise ValueError("source matrix is not CRT-only")
    if {row["seed"] for row in source_rows} != {"0"}:
        raise ValueError("source matrix does not use the frozen seed 0")
    if {row["mask_key"] for row in source_rows} != {"mask05"}:
        raise ValueError("source matrix does not use the frozen mask05 split")

    rows = []
    for source in source_rows:
        row = dict(source)
        row["architecture"] = "crs"
        row["target"] = source["target"].replace("-crt-", "-crs-")
        if row["target"] == source["target"]:
            raise ValueError(f"CRT token is absent from target {source['target']}")
        rows.append(row)

    args.output_tsv.parent.mkdir(parents=True, exist_ok=True)
    if args.output_tsv.exists() or args.output_json.exists():
        raise FileExistsError("refusing to overwrite an existing CRS manifest")
    with args.output_tsv.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=list(rows[0]),
            delimiter="\t",
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)

    source_payload = json.loads(args.source_json.read_text(encoding="utf-8"))
    payload = {
        **source_payload,
        "schema": "pvi-gcnm-coordinate-main-b045-crs-bp-matrix-v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "architecture": "crs",
        "source_crt_manifest": str(args.source_tsv.resolve()),
        "source_crt_manifest_sha256": sha256(args.source_tsv),
        "experimental_change": "downstream architecture only: CRT replaced by CRS",
        "runs": rows,
    }
    args.output_json.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "runs": len(rows),
                "subjects": len({row["subject"] for row in rows}),
                "seed": sorted({row["seed"] for row in rows}),
                "output": str(args.output_tsv),
            }
        )
    )


if __name__ == "__main__":
    main()
