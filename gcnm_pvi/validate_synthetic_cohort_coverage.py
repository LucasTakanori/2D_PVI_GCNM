#!/usr/bin/env python3
"""Check blind synthetic HP/LP voltage coverage against the 91-subject audit."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from gcnm_pvi.hp_lp_signals import centered_difference


SIGNAL_MAP = {
    "hp": "hp_referenced_mask05",
    "lp": "lp_referenced_mask05",
    "d_lp": "d_lp_referenced_mask05",
    "dd_lp": "dd_lp_referenced_mask05",
}


def _sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _synthetic_signals(root: Path) -> dict[str, np.ndarray]:
    hp_parts: list[np.ndarray] = []
    lp_parts: list[np.ndarray] = []
    d_parts: list[np.ndarray] = []
    dd_parts: list[np.ndarray] = []
    for split in ("train", "validation", "test"):
        with np.load(root / "hp" / f"{split}.npz") as hp_source, np.load(
            root / "lp" / f"{split}.npz"
        ) as lp_source:
            hp = np.asarray(hp_source["V"], dtype=np.float64)
            lp = np.asarray(lp_source["V"], dtype=np.float64)
            anatomy_hp = np.asarray(hp_source["anatomy_id"])
            anatomy_lp = np.asarray(lp_source["anatomy_id"])
        if hp.shape != lp.shape or not np.array_equal(anatomy_hp, anatomy_lp):
            raise ValueError(f"{split} HP/LP voltage rows are not aligned")
        hp_parts.append(hp)
        lp_parts.append(lp)
        for anatomy in np.unique(anatomy_lp):
            selected = np.flatnonzero(anatomy_lp == anatomy)
            lp_window = lp[selected]
            derivative = centered_difference(lp_window, axis=0)
            d_parts.append(derivative)
            dd_parts.append(centered_difference(derivative, axis=0))
    return {
        "hp": np.concatenate(hp_parts, axis=0),
        "lp": np.concatenate(lp_parts, axis=0),
        "d_lp": np.concatenate(d_parts, axis=0),
        "dd_lp": np.concatenate(dd_parts, axis=0),
    }


def _real_scales(signal: dict) -> tuple[float, float]:
    rms = np.asarray(signal["rms"]["subject_median"], dtype=np.float64)
    low = np.asarray(signal["quantiles"]["q0.01"]["subject_median"], dtype=np.float64)
    high = np.asarray(signal["quantiles"]["q0.99"]["subject_median"], dtype=np.float64)
    return float(np.median(rms)), float(np.median(high - low))


def coverage_report(
    synthetic: dict[str, np.ndarray],
    audit: dict,
    *,
    minimum_scale_ratio: float,
    maximum_scale_ratio: float,
    minimum_negative_fraction: float,
    maximum_negative_fraction: float,
) -> dict:
    if audit.get("schema") != "pvi-cohort-voltage-audit-v2":
        raise ValueError("coverage requires the mask05-referenced cohort audit v2")
    global_real = audit["global_subject_weighted"]
    reports = {}
    failures = []
    for synthetic_name, real_name in SIGNAL_MAP.items():
        values = np.asarray(synthetic[synthetic_name], dtype=np.float64)
        if values.ndim != 2 or not np.all(np.isfinite(values)):
            raise ValueError(f"synthetic {synthetic_name} is not a finite samples x channels array")
        real_rms, real_span = _real_scales(global_real[real_name])
        synthetic_rms = float(np.median(np.sqrt(np.mean(values * values, axis=0))))
        quantiles = np.quantile(values, (0.01, 0.99), axis=0)
        synthetic_span = float(np.median(quantiles[1] - quantiles[0]))
        rms_ratio = synthetic_rms / max(real_rms, 1e-30)
        span_ratio = synthetic_span / max(real_span, 1e-30)
        negative_fraction = float(np.mean(values < 0))
        reports[synthetic_name] = {
            "real_signal": real_name,
            "samples": int(values.shape[0]),
            "channels": int(values.shape[1]),
            "synthetic_channel_median_rms_v": synthetic_rms,
            "real_subject_median_channel_median_rms_v": real_rms,
            "rms_scale_ratio": rms_ratio,
            "synthetic_channel_median_q01_q99_span_v": synthetic_span,
            "real_subject_median_channel_median_q01_q99_span_v": real_span,
            "span_scale_ratio": span_ratio,
            "negative_fraction": negative_fraction,
        }
        for metric, ratio in (("rms", rms_ratio), ("q01_q99_span", span_ratio)):
            if not minimum_scale_ratio <= ratio <= maximum_scale_ratio:
                failures.append(
                    f"{synthetic_name} {metric} scale ratio {ratio:.6g} is outside "
                    f"[{minimum_scale_ratio}, {maximum_scale_ratio}]"
                )
        if not minimum_negative_fraction <= negative_fraction <= maximum_negative_fraction:
            failures.append(
                f"{synthetic_name} negative fraction {negative_fraction:.6g} is outside "
                f"[{minimum_negative_fraction}, {maximum_negative_fraction}]"
            )
    channels = {report["channels"] for report in reports.values()}
    if len(channels) != 1 or channels.pop() != 32:
        failures.append("synthetic coverage must contain exactly 32 voltage channels")
    return {
        "schema": "pvi-gcnm-synthetic-cohort-coverage-v1",
        "status": "pass" if not failures else "fail",
        "policy": (
            "blind gross-coverage gate only; cohort measurements do not fit or rescale "
            "the synthetic generator"
        ),
        "thresholds": {
            "minimum_scale_ratio": minimum_scale_ratio,
            "maximum_scale_ratio": maximum_scale_ratio,
            "minimum_negative_fraction": minimum_negative_fraction,
            "maximum_negative_fraction": maximum_negative_fraction,
        },
        "signals": reports,
        "failures": failures,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--cohort-audit", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--minimum-scale-ratio", type=float, default=0.02)
    parser.add_argument("--maximum-scale-ratio", type=float, default=50.0)
    parser.add_argument("--minimum-negative-fraction", type=float, default=0.001)
    parser.add_argument("--maximum-negative-fraction", type=float, default=0.999)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"immutable coverage report exists: {args.output}")
    audit = json.loads(args.cohort_audit.read_text(encoding="utf-8"))
    report = coverage_report(
        _synthetic_signals(args.dataset_root),
        audit,
        minimum_scale_ratio=args.minimum_scale_ratio,
        maximum_scale_ratio=args.maximum_scale_ratio,
        minimum_negative_fraction=args.minimum_negative_fraction,
        maximum_negative_fraction=args.maximum_negative_fraction,
    )
    report.update(
        {
            "dataset_root": str(args.dataset_root.resolve()),
            "dataset_metadata_sha256": _sha256(args.dataset_root / "metadata.json"),
            "cohort_audit": str(args.cohort_audit.resolve()),
            "cohort_audit_sha256": _sha256(args.cohort_audit),
        }
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    if report["status"] != "pass":
        raise RuntimeError("; ".join(report["failures"]))


if __name__ == "__main__":
    main()
