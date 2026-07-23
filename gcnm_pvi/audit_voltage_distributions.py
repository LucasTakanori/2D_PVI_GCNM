#!/usr/bin/env python3
"""Audit archived cohort voltage ranges without fitting the simulator to them.

Every subject contributes one summary regardless of its number of sessions or
periods.  The result is therefore suitable for checking blind synthetic
coverage without allowing subjects with longer recordings to dominate.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import h5py
import numpy as np

from gcnm_pvi.hp_lp_signals import centered_difference, resistance_to_voltage
from gcnm_pvi.mesh_registry import sha256_file


QUANTILES = (0.001, 0.01, 0.05, 0.5, 0.95, 0.99, 0.999)
SIGNALS = (
    "absolute_voltage",
    "hp_voltage",
    "lp_voltage",
    "d_lp",
    "dd_lp",
    "hp_referenced_mask05",
    "lp_referenced_mask05",
    "d_lp_referenced_mask05",
    "dd_lp_referenced_mask05",
)


def _finite_summary(values: np.ndarray) -> dict:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 2:
        raise ValueError(f"expected channels x frames, received {array.shape}")
    finite = np.isfinite(array)
    if not np.all(finite):
        raise ValueError("archived resistance audit encountered non-finite values")
    negative = array < 0

    def per_channel(function) -> list[float | None]:
        return [function(array[channel]) for channel in range(array.shape[0])]

    negative_min = []
    negative_max = []
    for channel in range(array.shape[0]):
        selected = array[channel, negative[channel]]
        negative_min.append(float(np.min(selected)) if len(selected) else None)
        negative_max.append(float(np.max(selected)) if len(selected) else None)
    quantiles = np.quantile(array, QUANTILES, axis=1).T
    return {
        "frames": int(array.shape[1]),
        "channels": int(array.shape[0]),
        "offset": per_channel(lambda x: float(np.mean(x))),
        "rms": per_channel(lambda x: float(np.sqrt(np.mean(x * x)))),
        "minimum": per_channel(lambda x: float(np.min(x))),
        "maximum": per_channel(lambda x: float(np.max(x))),
        "negative_fraction": [float(value) for value in np.mean(negative, axis=1)],
        "negative_minimum": negative_min,
        "negative_maximum": negative_max,
        "quantiles": {
            f"q{value:g}": [float(item) for item in quantiles[:, index]]
            for index, value in enumerate(QUANTILES)
        },
    }


def _load_subject(records: Iterable[dict]) -> dict[str, np.ndarray]:
    hp_parts: list[np.ndarray] = []
    lp_parts: list[np.ndarray] = []
    hp_window_parts: list[np.ndarray] = []
    lp_window_parts: list[np.ndarray] = []
    d_lp_window_parts: list[np.ndarray] = []
    dd_lp_window_parts: list[np.ndarray] = []
    for record in records:
        with h5py.File(record["source_hdf5"], "r") as handle:
            hp = resistance_to_voltage(handle["data/pviHP/resistance"][...])
            lp = resistance_to_voltage(handle["data/pviLP/resistance"][...])
            masks = np.asarray(handle["masks/mask05"], dtype=np.int64)
            masks[:, 0] -= 1
            period_length = int(np.asarray(handle["metadata/period_length"]).item())
        if hp.shape != lp.shape or hp.ndim != 2:
            raise ValueError(
                f"{record['source_hdf5']} has incompatible HP/LP shapes {hp.shape}/{lp.shape}"
            )
        hp_parts.append(hp)
        lp_parts.append(lp)
        for start, stop in masks:
            frames = slice(int(start) * period_length, int(stop) * period_length)
            hp_window = hp[:, frames] - hp[:, frames][:, :1]
            lp_window = lp[:, frames] - lp[:, frames][:, :1]
            hp_window_parts.append(hp_window)
            lp_window_parts.append(lp_window)
            d_lp = centered_difference(lp_window, axis=1)
            d_lp_window_parts.append(d_lp)
            dd_lp_window_parts.append(centered_difference(d_lp, axis=1))
    hp = np.concatenate(hp_parts, axis=1)
    lp = np.concatenate(lp_parts, axis=1)
    return {
        "absolute_voltage": hp + lp,
        "hp_voltage": hp,
        "lp_voltage": lp,
        "d_lp": centered_difference(lp, axis=1),
        "dd_lp": centered_difference(centered_difference(lp, axis=1), axis=1),
        "hp_referenced_mask05": np.concatenate(hp_window_parts, axis=1),
        "lp_referenced_mask05": np.concatenate(lp_window_parts, axis=1),
        "d_lp_referenced_mask05": np.concatenate(d_lp_window_parts, axis=1),
        "dd_lp_referenced_mask05": np.concatenate(dd_lp_window_parts, axis=1),
    }


def _aggregate_subject_summaries(subjects: list[dict]) -> dict:
    """Aggregate arrays of per-channel metrics with one vote per subject."""

    if not subjects:
        raise ValueError("cannot aggregate zero subjects")
    output: dict = {"subject_count": len(subjects)}

    def finite_columns(raw: np.ndarray, reducer) -> list[float | None]:
        result: list[float | None] = []
        for channel in range(raw.shape[1]):
            selected = raw[:, channel]
            selected = selected[np.isfinite(selected)]
            result.append(float(reducer(selected)) if len(selected) else None)
        return result

    for metric in (
        "offset",
        "rms",
        "minimum",
        "maximum",
        "negative_fraction",
        "negative_minimum",
        "negative_maximum",
    ):
        raw = np.asarray(
            [
                [np.nan if item is None else item for item in subject[metric]]
                for subject in subjects
            ],
            dtype=np.float64,
        )
        output[metric] = {
            "subject_mean": finite_columns(raw, np.mean),
            "subject_median": finite_columns(raw, np.median),
            "across_subject_minimum": finite_columns(raw, np.min),
            "across_subject_maximum": finite_columns(raw, np.max),
        }
    output["quantiles"] = {}
    for key in subjects[0]["quantiles"]:
        raw = np.asarray([subject["quantiles"][key] for subject in subjects])
        output["quantiles"][key] = {
            "subject_mean": np.mean(raw, axis=0).tolist(),
            "subject_median": np.median(raw, axis=0).tolist(),
        }
    return output


def audit_registry(registry: dict, *, include_subjects: bool = True) -> dict:
    records = registry.get("records", [])
    grouped: dict[str, list[dict]] = defaultdict(list)
    for record in records:
        grouped[record["subject"]].append(record)
    if not grouped:
        raise ValueError("registry contains zero sessions")

    subject_reports: dict[str, dict] = {}
    for index, subject in enumerate(sorted(grouped)):
        subject_records = sorted(grouped[subject], key=lambda item: item["source_order"])
        values = _load_subject(subject_records)
        subject_reports[subject] = {
            "ring": subject_records[0]["ring"],
            "sessions": [item["source_name"] for item in subject_records],
            "signals": {name: _finite_summary(values[name]) for name in SIGNALS},
        }
        print(
            f"audited {subject} ({index + 1}/{len(grouped)}, "
            f"{sum(item['frames'] for item in subject_reports[subject]['signals'].values())} signal-frames)",
            flush=True,
        )

    global_summary = {
        signal: _aggregate_subject_summaries(
            [subject_reports[subject]["signals"][signal] for subject in sorted(subject_reports)]
        )
        for signal in SIGNALS
    }
    rings: dict[str, dict] = {}
    for ring in sorted({report["ring"] for report in subject_reports.values()}):
        selected = [report for report in subject_reports.values() if report["ring"] == ring]
        rings[ring] = {
            signal: _aggregate_subject_summaries(
                [report["signals"][signal] for report in selected]
            )
            for signal in SIGNALS
        }
    return {
        "schema": "pvi-cohort-voltage-audit-v2",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "policy": "each subject contributes one per-channel summary regardless of session/frame count",
        "role": "coverage audit only; not synthetic-generator fitting",
        "voltage_convention": "delta_V = -I * delta_R; I = -0.01 A",
        "subject_count": len(subject_reports),
        "session_count": len(records),
        "signals": list(SIGNALS),
        "global_subject_weighted": global_summary,
        "rings_subject_weighted": rings,
        "subjects": subject_reports if include_subjects else None,
    }


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--registry", type=Path, default=root / "data/registries/main_b045_v1.json"
    )
    parser.add_argument(
        "--output", type=Path, default=root / "reports/cohort_voltage_audit_v2.json"
    )
    parser.add_argument(
        "--reuse-source-hashes-from",
        type=Path,
        help="reuse hashes from a prior audit with the identical registry hash",
    )
    parser.add_argument("--omit-subject-details", action="store_true")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"immutable audit output already exists: {args.output}")
    registry = json.loads(args.registry.read_text(encoding="utf-8"))
    report = audit_registry(registry, include_subjects=not args.omit_subject_details)
    report["registry"] = str(args.registry.resolve())
    report["registry_sha256"] = sha256_file(args.registry)
    report["workbook_sha256"] = registry.get("workbook_sha256")
    if args.reuse_source_hashes_from is not None:
        previous = json.loads(
            args.reuse_source_hashes_from.read_text(encoding="utf-8")
        )
        if previous.get("registry_sha256") != report["registry_sha256"]:
            raise ValueError("prior audit registry hash does not match")
        previous_hashes = previous.get("source_hdf5_hashes", {})
        expected_names = {record["source_name"] for record in registry["records"]}
        if set(previous_hashes) != expected_names:
            raise ValueError("prior audit source hash set does not match registry")
        report["source_hdf5_hashes"] = previous_hashes
        report["source_hash_provenance"] = str(
            args.reuse_source_hashes_from.resolve()
        )
    else:
        report["source_hdf5_hashes"] = {
            record["source_name"]: sha256_file(Path(record["source_hdf5"]))
            for record in registry["records"]
        }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "output": str(args.output),
                "subjects": report["subject_count"],
                "sessions": report["session_count"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
