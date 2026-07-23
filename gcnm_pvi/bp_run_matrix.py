"""Build auditable pilot or production single-subject BP run matrices."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


FAMILIES = ("coordinate", "global_voltage_slots")
ARCHITECTURES = ("crt", "crs")
OUTPUT_MODES = ("waveform", "fiducials")
MASK_KEY = "mask05"


def renumber_task_ids(rows: list[dict]) -> list[dict]:
    """Assign contiguous array IDs after any architecture/family filtering."""

    output = []
    for task_id, row in enumerate(rows):
        output.append({**row, "task_id": task_id})
    return output


def subjects_from_registry(registry: dict) -> list[str]:
    return sorted(
        {
            str(record["subject"])
            for record in registry["records"]
            if not record.get("exclusion_reason")
        }
    )


def build_run_matrix(
    subjects: list[str],
    coordinate_root: str | Path,
    global_voltage_slots_root: str | Path,
    *,
    families: tuple[str, ...] = FAMILIES,
    channel_mode: str = "3ch",
) -> list[dict]:
    """Return one deterministic row per GCNM family/model/target/subject run."""
    if channel_mode not in {"3ch", "6ch"}:
        raise ValueError("channel_mode must be 3ch or 6ch")
    invalid = set(families) - set(FAMILIES)
    if invalid:
        raise ValueError(f"invalid GCNM families: {sorted(invalid)}")
    roots = {
        "coordinate": str(Path(coordinate_root).resolve()),
        "global_voltage_slots": str(Path(global_voltage_slots_root).resolve()),
    }
    rows: list[dict] = []
    # Keep subject blocks contiguous so the prioritized pilot starts with
    # subject006 and only then schedules subject010.
    for subject in sorted(subjects):
        for family in families:
            for architecture in ARCHITECTURES:
                for output_mode in OUTPUT_MODES:
                    target = (
                        f"gcnm-{family}-{channel_mode}-{architecture}"
                        f"-image-to-{output_mode}"
                    )
                    rows.append(
                        {
                            "task_id": len(rows),
                            "subject": subject,
                            "family": family,
                            "architecture": architecture,
                            "output_mode": output_mode,
                            "channel_mode": channel_mode,
                            "parquet_root": roots[family],
                            "target": target,
                            "mask_key": MASK_KEY,
                        }
                    )
    return rows


def build_original_run_matrix(subjects: list[str]) -> list[dict]:
    """Return split-matched archived-Newton baselines for the same BP protocol."""
    rows: list[dict] = []
    for architecture in ARCHITECTURES:
        for output_mode in OUTPUT_MODES:
            target = f"original-pvi-{architecture}-image-to-{output_mode}"
            for subject in sorted(subjects):
                rows.append(
                    {
                        "task_id": len(rows),
                        "subject": subject,
                        "architecture": architecture,
                        "output_mode": output_mode,
                        "target": target,
                        "mask_key": MASK_KEY,
                    }
                )
    return rows


def build_pilot_run_matrix(
    subjects: list[str],
    coordinate_root: str | Path,
    global_voltage_slots_root: str | Path,
    *,
    channel_mode: str = "3ch",
) -> list[dict]:
    """Combine GCNM and required split-matched Newton pilot experiments."""
    rows = []
    for row in build_run_matrix(
        subjects,
        coordinate_root,
        global_voltage_slots_root,
        channel_mode=channel_mode,
    ):
        rows.append(
            {
                "task_id": len(rows),
                "representation": "gcnm",
                "subject": row["subject"],
                "family": row["family"],
                "architecture": row["architecture"],
                "output_mode": row["output_mode"],
                "channel_mode": row["channel_mode"],
                "parquet_root": row["parquet_root"],
                "target": row["target"],
                "mask_key": row["mask_key"],
            }
        )
    for row in build_original_run_matrix(subjects):
        rows.append(
            {
                "task_id": len(rows),
                "representation": "original",
                "subject": row["subject"],
                "family": "original_pvi",
                "architecture": row["architecture"],
                "output_mode": row["output_mode"],
                "channel_mode": "3ch",
                "parquet_root": "-",
                "target": row["target"],
                "mask_key": row["mask_key"],
            }
        )
    return rows


def write_manifests(
    rows: list[dict], tsv_path: Path, json_path: Path, *, representation: str
) -> None:
    if not rows:
        raise ValueError("cannot write an empty BP run matrix")
    fieldnames = list(rows[0])
    if any(list(row) != fieldnames for row in rows):
        raise ValueError("BP matrix rows do not share one stable schema")
    tsv_path.parent.mkdir(parents=True, exist_ok=True)
    with tsv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=fieldnames, delimiter="\t", lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(rows)

    payload = {
        "schema": "pvi-gcnm-bp-run-matrix-v2",
        "representation": representation,
        "mask_key": MASK_KEY,
        "periods_per_sample": 5,
        "frames_per_period": 50,
        "frames_per_sample": 250,
        "subject_count": len({row["subject"] for row in rows}),
        "run_count": len(rows),
        "resource_contract": {
            "array_max_concurrent": 4,
            "gpus_per_run": 1,
            "cpus_per_run": 16,
            "memory_gb_per_run": 250,
            "data_loader_workers_per_run": 8,
            "maximum_concurrent_request": {"gpus": 4, "cpus": 64, "memory_gb": 1000},
        },
        "runs": rows,
    }
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registry", type=Path, required=True)
    parser.add_argument(
        "--mode", choices=["gcnm", "original", "pilot"], default="gcnm"
    )
    parser.add_argument("--coordinate-root", type=Path)
    parser.add_argument("--global-voltage-slots-root", type=Path)
    parser.add_argument("--family", action="append", choices=FAMILIES, default=[])
    parser.add_argument("--architecture", action="append", choices=ARCHITECTURES, default=[])
    parser.add_argument("--channel-mode", choices=["3ch", "6ch"], default="3ch")
    parser.add_argument("--subject", action="append", default=[])
    parser.add_argument("--tsv", type=Path, required=True)
    parser.add_argument("--json", type=Path, required=True)
    parser.add_argument("--expected-subjects", type=int)
    parser.add_argument("--expected-runs", type=int)
    args = parser.parse_args()

    registry = json.loads(args.registry.read_text(encoding="utf-8"))
    available = subjects_from_registry(registry)
    if args.subject:
        requested = sorted(set(args.subject))
        missing = sorted(set(requested) - set(available))
        if missing:
            raise ValueError(f"requested subjects absent from registry: {missing}")
        subjects = requested
    else:
        subjects = available
    if args.expected_subjects is not None and len(subjects) != args.expected_subjects:
        raise ValueError(
            f"selected {len(subjects)} subjects; expected {args.expected_subjects}"
        )

    if args.mode in {"gcnm", "pilot"}:
        if args.coordinate_root is None or args.global_voltage_slots_root is None:
            parser.error("gcnm mode requires both Parquet roots")
        if args.mode == "pilot":
            if args.family:
                parser.error("pilot mode always compares both GCNM families")
            rows = build_pilot_run_matrix(
                subjects,
                args.coordinate_root,
                args.global_voltage_slots_root,
                channel_mode=args.channel_mode,
            )
            representation = f"us120-pilot-gcnm-{args.channel_mode}-and-original"
        else:
            families = tuple(args.family) if args.family else FAMILIES
            rows = build_run_matrix(
                subjects,
                args.coordinate_root,
                args.global_voltage_slots_root,
                families=families,
                channel_mode=args.channel_mode,
            )
            if args.architecture:
                rows = [row for row in rows if row["architecture"] in set(args.architecture)]
            representation = f"gcnm-{args.channel_mode}"
    else:
        if args.family:
            parser.error("--family is not valid in original mode")
        rows = build_original_run_matrix(subjects)
        representation = "archived-newton-pvi"
    if args.expected_runs is not None and len(rows) != args.expected_runs:
        raise ValueError(f"built {len(rows)} runs; expected {args.expected_runs}")
    rows = renumber_task_ids(rows)
    write_manifests(rows, args.tsv, args.json, representation=representation)
    print(f"wrote {len(rows)} mask05 runs for {len(subjects)} subjects")


if __name__ == "__main__":
    main()
