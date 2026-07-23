"""Pilot a two-stage GCNM representation as an alternative to Newton PVI.

The archived Newton BP tensor is ``[HP, d(LP)/dt, d2(LP)/dt2]``.  The proposed
GCNM tensor is deliberately different: ``[S2, d(S1)/dt, d2(S1)/dt2]``, where
S1 and S2 are iterative reconstructions of the same correctly referenced
voltage.  This pilot compares two reference choices: the exact archived HP
voltage and a stable five-beat moving-reference voltage formed from HP + LP.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import h5py
import numpy as np

from gcnm_pvi.export_pvi_parquet import (
    _frame_beat_templates,
    _validate_images,
)
from gcnm_pvi.production_preprocess import movmean
from gcnm_pvi.representations import (
    CoordinateReconstructor,
    GlobalVoltageVesselSlotReconstructor,
    rasterize_frames,
)


PERIOD_LENGTH = 50
STIM_CURRENT_A = -0.01


def centered_difference(values: np.ndarray) -> np.ndarray:
    """Match pvi_ml's centered difference and zero padding exactly."""

    array = np.nan_to_num(np.asarray(values), nan=0.0)
    output = np.zeros_like(array)
    output[..., 1:-1] = (array[..., 2:] - array[..., :-2]) / 2.0
    return output


def _correlation(left: np.ndarray, right: np.ndarray) -> float:
    valid = np.isfinite(left) & np.isfinite(right)
    a = np.asarray(left[valid], dtype=np.float64)
    b = np.asarray(right[valid], dtype=np.float64)
    if not len(a):
        return float("nan")
    a -= np.mean(a)
    b -= np.mean(b)
    denominator = float(np.linalg.norm(a) * np.linalg.norm(b))
    return float(np.dot(a, b) / denominator) if denominator > 0 else float("nan")


def comparison_metrics(candidate: np.ndarray, newton: np.ndarray) -> dict:
    """Compare behavior without requiring pixel equality to Newton PVI."""

    candidate = np.asarray(candidate, dtype=np.float64)
    newton = np.asarray(newton, dtype=np.float64)
    finite_candidate = np.isfinite(candidate)
    finite_newton = np.isfinite(newton)
    joint = finite_candidate & finite_newton
    if not np.any(joint):
        raise ValueError("candidate and Newton image have no jointly finite values")
    frame_correlations = [
        _correlation(candidate[..., frame], newton[..., frame])
        for frame in range(candidate.shape[-1])
    ]
    finite_frame_correlations = [
        value for value in frame_correlations if np.isfinite(value)
    ]
    candidate_envelope = np.sqrt(np.nanmean(candidate**2, axis=(0, 1)))
    newton_envelope = np.sqrt(np.nanmean(newton**2, axis=(0, 1)))
    mask_candidate = finite_candidate[..., 0]
    mask_newton = finite_newton[..., 0]
    union = int(np.count_nonzero(mask_candidate | mask_newton))
    candidate_values = candidate[joint]
    newton_values = newton[joint]
    design = np.column_stack((candidate_values, np.ones_like(candidate_values)))
    scale, offset = np.linalg.lstsq(design, newton_values, rcond=None)[0]
    fitted = scale * candidate_values + offset
    reference_std = max(float(np.std(newton_values)), 1e-12)
    return {
        "correlation": _correlation(candidate, newton),
        "absolute_correlation": abs(_correlation(candidate, newton)),
        "median_frame_correlation": float(np.median(finite_frame_correlations))
        if finite_frame_correlations
        else float("nan"),
        "temporal_rms_envelope_correlation": _correlation(
            candidate_envelope, newton_envelope
        ),
        "affine_nrmse": float(
            np.sqrt(np.mean((fitted - newton_values) ** 2)) / reference_std
        ),
        "affine_scale": float(scale),
        "affine_offset": float(offset),
        "candidate_rms": float(np.sqrt(np.mean(candidate_values**2))),
        "newton_rms": float(np.sqrt(np.mean(newton_values**2))),
        "mask_iou": float(
            np.count_nonzero(mask_candidate & mask_newton) / union
        )
        if union
        else float("nan"),
    }


def _reconstruct(
    reconstructor,
    voltage: np.ndarray,
    *,
    chunk_frames: int,
) -> tuple[np.ndarray, np.ndarray, dict]:
    templates = (
        _frame_beat_templates(voltage)
        if getattr(reconstructor, "requires_beat_context", False)
        else None
    )
    stage_1_parts: list[np.ndarray] = []
    stage_2_parts: list[np.ndarray] = []
    residual_1: list[float] = []
    residual_2: list[float] = []
    for first in range(0, len(voltage), chunk_frames):
        stop = min(first + chunk_frames, len(voltage))
        if templates is None:
            stage_1, stage_2, report = reconstructor.reconstruct(voltage[first:stop])
        else:
            stage_1, stage_2, report = reconstructor.reconstruct(
                voltage[first:stop], templates[first:stop]
            )
        stage_1_parts.append(stage_1)
        stage_2_parts.append(stage_2)
        residual_1.extend(np.asarray(report["stage_1_forward_voltage_rms"]).tolist())
        residual_2.extend(np.asarray(report["stage_2_forward_voltage_rms"]).tolist())
    stage_1_img = rasterize_frames(
        reconstructor.mappings, np.concatenate(stage_1_parts)
    )
    stage_2_img = rasterize_frames(
        reconstructor.mappings, np.concatenate(stage_2_parts)
    )
    activation = _validate_images(stage_1_img, stage_2_img)
    residual_1_array = np.asarray(residual_1, dtype=np.float64)
    residual_2_array = np.asarray(residual_2, dtype=np.float64)
    return stage_1_img, stage_2_img, {
        **activation,
        "stage_1_forward_voltage_rms": float(np.mean(residual_1_array)),
        "stage_2_forward_voltage_rms": float(np.mean(residual_2_array)),
        "stage_2_lower_residual_fraction": float(
            np.mean(residual_2_array <= residual_1_array)
        ),
    }


def _save_figure(path: Path, arrays: dict[str, np.ndarray]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    newton_hp = arrays["newton_hp"]
    frame = int(np.nanargmax(np.sqrt(np.nanmean(newton_hp**2, axis=(0, 1)))))
    rows = [
        (
            "HP direct",
            arrays["newton_hp"],
            arrays["dynamic_stage1"],
            arrays["dynamic_stage2"],
        ),
        (
            "first temporal difference",
            centered_difference(arrays["newton_lp"]),
            centered_difference(arrays["dynamic_stage1"]),
            centered_difference(arrays["dynamic_stage2"]),
        ),
        (
            "second temporal difference",
            centered_difference(centered_difference(arrays["newton_lp"])),
            centered_difference(centered_difference(arrays["dynamic_stage1"])),
            centered_difference(centered_difference(arrays["dynamic_stage2"])),
        ),
    ]
    figure, axes = plt.subplots(3, 3, figsize=(12, 10), constrained_layout=True)
    for row_index, (label, newton, stage_1, stage_2) in enumerate(rows):
        images = (newton[..., frame], stage_1[..., frame], stage_2[..., frame])
        for column, (title, image) in enumerate(
            zip(("Newton", "GCNM stage 1", "GCNM stage 2"), images)
        ):
            limit = max(float(np.nanpercentile(np.abs(image), 99)), 1e-12)
            shown = axes[row_index, column].imshow(
                image, cmap="RdBu_r", vmin=-limit, vmax=limit, origin="lower"
            )
            axes[row_index, column].set_title(f"{label}: {title}")
            axes[row_index, column].axis("off")
            figure.colorbar(shown, ax=axes[row_index, column], shrink=0.72)
    figure.suptitle(f"Newton-compatible GCNM pilot, frame {frame}")
    figure.savefig(path, dpi=160)
    plt.close(figure)


def run_session_pilot(
    record: dict,
    reconstructor,
    output_dir: Path,
    *,
    mask_position: str = "middle",
    chunk_frames: int = 256,
) -> dict:
    """Reconstruct one exact mask05 window with two differential references."""

    source_path = Path(record["source_hdf5"])
    with h5py.File(source_path, "r") as handle:
        masks = np.asarray(handle["masks/mask05"], dtype=np.int64)
        masks[:, 0] -= 1
        if mask_position == "first":
            mask_index = 0
        elif mask_position == "middle":
            mask_index = len(masks) // 2
        elif mask_position == "last":
            mask_index = len(masks) - 1
        else:
            raise ValueError("mask_position must be first, middle, or last")
        mask_start, mask_stop = (int(value) for value in masks[mask_index])
        if mask_stop - mask_start != 5:
            raise ValueError("pilot requires a five-period mask05 window")
        frame_slice = slice(mask_start * PERIOD_LENGTH, mask_stop * PERIOD_LENGTH)
        resistance_hp_all = np.asarray(
            handle["data/pviHP/resistance"], dtype=np.float64
        )
        resistance_lp_all = np.asarray(
            handle["data/pviLP/resistance"], dtype=np.float64
        )
        newton_hp = np.asarray(
            handle["data/pviHP/img"][:, :, frame_slice], dtype=np.float32
        )
        newton_lp = np.asarray(
            handle["data/pviLP/img"][:, :, frame_slice], dtype=np.float32
        )

    # Production convention: delta V = -I delta R and I = -0.01 A.
    voltage_hp = (-STIM_CURRENT_A * resistance_hp_all[:, frame_slice]).T
    voltage_full = -STIM_CURRENT_A * (resistance_hp_all + resistance_lp_all)
    # A centered five-beat reference is computed over the complete session, not
    # independently per sample window.  Consequently an overlapping archived
    # frame always has one stable GCNM input while the large DC/contact offset
    # is removed.  HP-only remains a conservative comparison arm.
    dynamic_all = voltage_full - movmean(
        voltage_full, 5 * PERIOD_LENGTH, axis=1
    )
    voltage_dynamic = dynamic_all[:, frame_slice].T
    hp_stage_1, hp_stage_2, hp_report = _reconstruct(
        reconstructor, voltage_hp, chunk_frames=chunk_frames
    )
    dynamic_stage_1, dynamic_stage_2, dynamic_report = _reconstruct(
        reconstructor, voltage_dynamic, chunk_frames=chunk_frames
    )
    arrays = {
        "hp_stage1": hp_stage_1.astype(np.float32),
        "hp_stage2": hp_stage_2.astype(np.float32),
        "dynamic_stage1": dynamic_stage_1.astype(np.float32),
        "dynamic_stage2": dynamic_stage_2.astype(np.float32),
        "newton_hp": newton_hp,
        "newton_lp": newton_lp,
    }
    output_dir.mkdir(parents=True, exist_ok=False)
    np.savez_compressed(output_dir / "representations.npz", **arrays)
    _save_figure(output_dir / "newton_gcnm_comparison.png", arrays)

    d_newton_lp = centered_difference(newton_lp)
    dd_newton_lp = centered_difference(d_newton_lp)
    d_dynamic_stage_1 = centered_difference(dynamic_stage_1)
    dd_dynamic_stage_1 = centered_difference(d_dynamic_stage_1)
    d_hp_stage_1 = centered_difference(hp_stage_1)
    dd_hp_stage_1 = centered_difference(d_hp_stage_1)

    def stage_relationship(stage_1, stage_2):
        finite = np.isfinite(stage_1) & np.isfinite(stage_2)
        rms_1 = float(np.sqrt(np.mean(stage_1[finite] ** 2)))
        difference_rms = float(
            np.sqrt(np.mean((stage_2[finite] - stage_1[finite]) ** 2))
        )
        return {
            "correlation": _correlation(stage_1, stage_2),
            "difference_rms": difference_rms,
            "difference_to_stage1_rms": difference_rms / max(rms_1, 1e-12),
        }

    report = {
        "schema": "gcnm-newton-compatible-pilot-v1",
        "source_name": record["source_name"],
        "subject": record["subject"],
        "session": record["session"],
        "ring": record["ring"],
        "mask05_index": mask_index,
        "mask_start": mask_start,
        "mask_stop": mask_stop,
        "frames": int(len(voltage_hp)),
        "voltage_contract": {
            "primary": "-I * (R_HP + R_LP) minus a centered 250-frame session moving mean",
            "comparison": "-I * data/pviHP/resistance",
            "stim_current_a": STIM_CURRENT_A,
            "bp_input_candidate": [
                "dynamic_stage2",
                "centered_difference(dynamic_stage1)",
                "centered_difference(centered_difference(dynamic_stage1))",
            ],
        },
        "voltage_rms": {
            "hp": float(np.sqrt(np.mean(voltage_hp**2))),
            "dynamic": float(np.sqrt(np.mean(voltage_dynamic**2))),
        },
        "hp_stage_diagnostics": hp_report,
        "dynamic_stage_diagnostics": dynamic_report,
        "stage_relationship": {
            "hp": stage_relationship(hp_stage_1, hp_stage_2),
            "dynamic": stage_relationship(dynamic_stage_1, dynamic_stage_2),
        },
        "comparisons": {
            "primary_s2_vs_newton_hp": comparison_metrics(
                dynamic_stage_2, newton_hp
            ),
            "primary_ds1_vs_d_newton_lp": comparison_metrics(
                d_dynamic_stage_1, d_newton_lp
            ),
            "primary_dds1_vs_dd_newton_lp": comparison_metrics(
                dd_dynamic_stage_1, dd_newton_lp
            ),
            "hp_only_s2_vs_newton_hp": comparison_metrics(
                hp_stage_2, newton_hp
            ),
            "hp_only_ds1_vs_d_newton_lp": comparison_metrics(
                d_hp_stage_1, d_newton_lp
            ),
            "hp_only_dds1_vs_dd_newton_lp": comparison_metrics(
                dd_hp_stage_1, dd_newton_lp
            ),
        },
        "artifact": str((output_dir / "representations.npz").resolve()),
        "figure": str((output_dir / "newton_gcnm_comparison.png").resolve()),
    }
    (output_dir / "report.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    return report


def _make_reconstructor(family: str, record: dict, checkpoint_root: Path):
    ring = record["ring"]
    model_name = f"{family}_b045_{ring}_seed0"
    model_dir = checkpoint_root / family / ring
    if family == "coordinate":
        return CoordinateReconstructor(
            Path(record["config"]), model_dir, model_name
        )
    if family == "global_voltage_slots":
        return GlobalVoltageVesselSlotReconstructor(
            Path(record["config"]),
            model_dir / f"{model_name}_localizer.pt",
            model_dir / f"{model_name}_refiner.pt",
        )
    raise ValueError(f"unsupported pilot family {family}")


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--registry", type=Path, default=root / "data/registries/main_b045_v1.json"
    )
    parser.add_argument(
        "--checkpoint-root",
        type=Path,
        default=root / "models/mesh_representations/beats1000x50_v1",
    )
    parser.add_argument(
        "--family", choices=["coordinate", "global_voltage_slots"], required=True
    )
    parser.add_argument("--subject", required=True)
    parser.add_argument("--sessions", nargs="*", default=["baseline", "valsalva", "pressor"])
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument(
        "--mask-position", choices=["first", "middle", "last"], default="middle"
    )
    parser.add_argument("--chunk-frames", type=int, default=256)
    args = parser.parse_args()
    if args.output_root.exists():
        raise FileExistsError(f"immutable pilot root exists: {args.output_root}")
    registry = json.loads(args.registry.read_text(encoding="utf-8"))
    selected = [
        record
        for record in registry["records"]
        if record["subject"] == args.subject and record["session"] in set(args.sessions)
    ]
    if not selected:
        raise ValueError("pilot selection contains zero sessions")
    args.output_root.mkdir(parents=True, exist_ok=False)
    reports = []
    reconstructors: dict[str, object] = {}
    for record in selected:
        ring = record["ring"]
        if ring not in reconstructors:
            reconstructors[ring] = _make_reconstructor(
                args.family, record, args.checkpoint_root
            )
        reports.append(
            run_session_pilot(
                record,
                reconstructors[ring],
                args.output_root / record["source_name"],
                mask_position=args.mask_position,
                chunk_frames=args.chunk_frames,
            )
        )
    summary = {
        "schema": "gcnm-newton-compatible-pilot-summary-v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "family": args.family,
        "subject": args.subject,
        "sessions": [report["source_name"] for report in reports],
        "reports": reports,
    }
    (args.output_root / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({"family": args.family, "subject": args.subject, "sessions": len(reports)}, indent=2))


if __name__ == "__main__":
    main()
