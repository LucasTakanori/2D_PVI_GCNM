#!/usr/bin/env python3
"""Reconstruct selected real subject windows with production coordinate-direct GCNM."""

from __future__ import annotations

import argparse
import json
from io import BytesIO
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from gcnm_pvi.coordinate_direct_representation import (
    CHANNEL_NAMES,
    centered_temporal_difference,
    compose_s1_s2_ds2,
    representation_diagnostics,
)
from gcnm_pvi.mesh_registry import sha256_file
from gcnm_pvi.pilot_newton_compatible import comparison_metrics
from gcnm_pvi.representations import CoordinateReconstructor, rasterize_frames


PERIOD_LENGTH = 50
STIM_CURRENT_A = -0.01


def physical_full_differential_voltage(
    resistance_hp: np.ndarray,
    resistance_lp: np.ndarray,
) -> np.ndarray:
    """Convert real PVI resistance to the synthetic physical-voltage sign."""

    resistance = np.asarray(resistance_hp, dtype=np.float64) + np.asarray(
        resistance_lp, dtype=np.float64
    )
    saved_pvi_voltage = -STIM_CURRENT_A * resistance
    referenced_saved = saved_pvi_voltage - saved_pvi_voltage[:, :1]
    return -referenced_saved.T


def _panel_limit(values: np.ndarray) -> float:
    finite = np.abs(values[np.isfinite(values)])
    return max(float(np.quantile(finite, 0.995)), 1e-8)


def _render_gif(
    output: Path,
    panels: list[np.ndarray],
    titles: list[str],
    *,
    source_name: str,
    first_frame: int = 50,
    frames: int = 150,
    frame_ms: int = 90,
) -> None:
    limits = [_panel_limit(values[..., first_frame : first_frame + frames]) for values in panels]
    images: list[Image.Image] = []
    for offset in range(frames):
        index = first_frame + offset
        figure, axes = plt.subplots(1, len(panels), figsize=(3.0 * len(panels), 3.6), dpi=100)
        for axis, values, title, limit in zip(axes, panels, titles, limits):
            shown = axis.imshow(
                values[..., index],
                cmap="RdBu_r",
                vmin=-limit,
                vmax=limit,
                origin="upper",
            )
            axis.set_title(f"{title}\n±{limit:.2e} S/m", fontsize=9)
            axis.axis("off")
            figure.colorbar(shown, ax=axis, orientation="horizontal", fraction=0.06, pad=0.06)
        figure.suptitle(
            f"{source_name} | displayed beat {offset // 50 + 1}/3, sample {offset % 50 + 1}/50",
            fontsize=11,
        )
        figure.subplots_adjust(left=0.01, right=0.99, top=0.82, bottom=0.13, wspace=0.08)
        buffer = BytesIO()
        figure.savefig(buffer, format="png", facecolor="white")
        plt.close(figure)
        buffer.seek(0)
        images.append(Image.open(buffer).convert("P", palette=Image.Palette.ADAPTIVE))
    output.parent.mkdir(parents=True, exist_ok=True)
    images[0].save(
        output,
        save_all=True,
        append_images=images[1:],
        duration=frame_ms,
        loop=0,
        optimize=False,
    )


def _envelope_cv(values: np.ndarray) -> float:
    envelope = np.sqrt(np.nanmean(values.astype(np.float64) ** 2, axis=(0, 1)))
    return float(np.std(envelope) / max(np.mean(envelope), 1e-12))


def reconstruct_session(
    record: dict,
    reconstructor: CoordinateReconstructor,
    output_root: Path,
    *,
    mask_position: str,
) -> dict:
    source_path = Path(record["source_hdf5"])
    with h5py.File(source_path, "r") as handle:
        masks = np.asarray(handle["masks/mask05"], dtype=np.int64)
        masks[:, 0] -= 1
        positions = {"first": 0, "middle": len(masks) // 2, "last": len(masks) - 1}
        mask_index = positions[mask_position]
        start, stop = (int(value) for value in masks[mask_index])
        if stop - start != 5:
            raise ValueError(f"{record['source_name']} selected mask is not five beats")
        frame_slice = slice(start * PERIOD_LENGTH, stop * PERIOD_LENGTH)
        hp_r = np.asarray(handle["data/pviHP/resistance"][:, frame_slice], dtype=np.float64)
        lp_r = np.asarray(handle["data/pviLP/resistance"][:, frame_slice], dtype=np.float64)
        archived_hp = np.asarray(handle["data/pviHP/img"][:, :, frame_slice], dtype=np.float32)
        archived_lp = np.asarray(handle["data/pviLP/img"][:, :, frame_slice], dtype=np.float32)

    physical_voltage = physical_full_differential_voltage(hp_r, lp_r)
    stage_1, stage_2, residuals = reconstructor.reconstruct(physical_voltage)
    s1 = rasterize_frames(reconstructor.mappings, stage_1).astype(np.float32)
    s2 = rasterize_frames(reconstructor.mappings, stage_2).astype(np.float32)
    channels = compose_s1_s2_ds2(s1, s2).astype(np.float32)
    d_s2 = channels[2]
    finite = np.isfinite(s1) & np.isfinite(s2)
    s1_rms = float(np.sqrt(np.mean(s1[finite].astype(np.float64) ** 2)))
    difference_rms = float(
        np.sqrt(np.mean((s2[finite].astype(np.float64) - s1[finite]) ** 2))
    )
    diagnostics = representation_diagnostics(channels)
    diagnostics["temporal_envelope_cv"] = {
        name: _envelope_cv(channels[index])
        for index, name in enumerate(CHANNEL_NAMES)
    }
    diagnostics["s2_minus_s1_rms"] = difference_rms
    diagnostics["s2_minus_s1_to_s1_rms"] = difference_rms / max(s1_rms, 1e-12)
    diagnostics["physical_voltage_rms_v"] = float(
        np.sqrt(np.mean(physical_voltage**2))
    )
    diagnostics["stage_1_forward_voltage_residual_rms_v"] = float(
        np.mean(residuals["stage_1_forward_voltage_rms"])
    )
    diagnostics["stage_2_forward_voltage_residual_rms_v"] = float(
        np.mean(residuals["stage_2_forward_voltage_rms"])
    )
    nondegenerate = bool(
        diagnostics["all_channels_nonzero"]
        and diagnostics["temporal_envelope_cv"]["s2"] > 0.01
        and diagnostics["temporal_envelope_cv"]["d_s2_dt"] > 0.01
        and diagnostics["s2_minus_s1_to_s1_rms"] > 0.01
    )

    session_root = output_root / record["source_name"]
    session_root.mkdir(parents=True, exist_ok=False)
    np.savez_compressed(
        session_root / "representations.npz",
        representation=channels,
        s1=s1,
        s2=s2,
        d_s2_dt=d_s2,
        archived_pvi_hp=archived_hp,
        archived_pvi_lp=archived_lp,
        physical_delta_voltage=physical_voltage.astype(np.float32),
    )
    gif_path = session_root / "archived_pvi_s1_s2_ds2_three_beats.gif"
    _render_gif(
        gif_path,
        [archived_hp, archived_lp, s1, s2, d_s2],
        ["Archived PVI HP", "Archived PVI LP", "GCNM S1", "GCNM S2", "dS2/dt"],
        source_name=record["source_name"],
    )
    report = {
        "schema": "pvi-gcnm-coordinate-direct-real-pilot-v1",
        "subject": record["subject"],
        "session": record["session"],
        "source_name": record["source_name"],
        "source_hdf5": str(source_path.resolve()),
        "source_hdf5_sha256": sha256_file(source_path),
        "mask05_index": mask_index,
        "mask_period_bounds_zero_based": [start, stop],
        "frames": 250,
        "voltage_contract": {
            "saved_pvi": "-I * (R_HP + R_LP)",
            "model_input": "negative of saved-PVI voltage after first-window-frame reference",
            "stim_current_a": STIM_CURRENT_A,
        },
        "bp_representation": {
            "channels": list(CHANNEL_NAMES),
            "shape": list(channels.shape),
            "derivative": "centered temporal difference of S2, zero-padded boundaries",
        },
        "diagnostics": diagnostics,
        "passes_non_degenerate_gate": nondegenerate,
        "archived_pvi_is_not_ground_truth": True,
        "behavioral_comparisons": {
            "s1_vs_archived_hp": comparison_metrics(s1, archived_hp),
            "s2_vs_archived_hp": comparison_metrics(s2, archived_hp),
            "d_s2_vs_d_archived_lp": comparison_metrics(
                d_s2, centered_temporal_difference(archived_lp)
            ),
        },
        "representation_npz": str((session_root / "representations.npz").resolve()),
        "gif": str(gif_path.resolve()),
    }
    (session_root / "report.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    return report


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--registry", type=Path, default=root / "data/registries/main_b045_v1.json"
    )
    parser.add_argument(
        "--config", type=Path, default=root / "configs/rings_b045/US120.yaml"
    )
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--model-name", default="coordinate_direct")
    parser.add_argument("--subject", default="subject006")
    parser.add_argument(
        "--sessions", nargs="+", default=["baseline", "valsalva", "pressor"]
    )
    parser.add_argument("--mask-position", choices=["first", "middle", "last"], default="middle")
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    if args.output_root.exists():
        raise FileExistsError(f"immutable subject pilot exists: {args.output_root}")
    registry = json.loads(args.registry.read_text(encoding="utf-8"))
    registry_records = registry.get("records", registry.get("sessions", []))
    records = [
        record
        for record in registry_records
        if record["subject"] == args.subject and record["session"] in args.sessions
    ]
    records.sort(key=lambda record: args.sessions.index(record["session"]))
    if len(records) != len(args.sessions):
        raise ValueError("registry does not contain every requested subject/session")
    args.output_root.mkdir(parents=True, exist_ok=False)
    reconstructor = CoordinateReconstructor(
        args.config, args.checkpoint_dir, args.model_name
    )
    reports = [
        reconstruct_session(
            record, reconstructor, args.output_root, mask_position=args.mask_position
        )
        for record in records
    ]
    summary = {
        "schema": "pvi-gcnm-coordinate-direct-subject-pilot-summary-v1",
        "subject": args.subject,
        "sessions": [report["session"] for report in reports],
        "all_sessions_pass_non_degenerate_gate": all(
            report["passes_non_degenerate_gate"] for report in reports
        ),
        "reports": reports,
    }
    (args.output_root / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
