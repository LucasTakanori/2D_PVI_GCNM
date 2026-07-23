#!/usr/bin/env python3
"""Render the required three-consecutive-beat GCNM visual inspection gates."""

from __future__ import annotations

import argparse
import json
from io import BytesIO
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from gcnm_pvi.hp_lp_signals import component_reference, resistance_to_voltage
from gcnm_pvi.mesh_registry import sha256_file
from gcnm_pvi.pvi_splits import stable_sample_id
from gcnm_pvi.representations import (
    CoordinateReconstructor,
    GlobalVoltageVesselSlotReconstructor,
    rasterize_frames,
)


FRAMES_PER_BEAT = 50
GIF_BEATS = 3


def three_consecutive_beat_indices(
    anatomy_id: np.ndarray,
    beat_id: np.ndarray,
    sample_index: np.ndarray,
    *,
    anatomy: int | None = None,
    first_beat: int = 0,
) -> np.ndarray:
    """Select exactly 150 chronologically ordered frames from one anatomy."""

    anatomies = np.asarray(anatomy_id)
    beats = np.asarray(beat_id)
    samples = np.asarray(sample_index)
    candidates = np.unique(anatomies)
    if anatomy is not None:
        candidates = np.asarray([anatomy])
    for candidate in candidates:
        selected_anatomy = np.flatnonzero(anatomies == candidate)
        available = np.unique(beats[selected_anatomy])
        if len(available) < first_beat + GIF_BEATS:
            continue
        chosen = available[first_beat : first_beat + GIF_BEATS]
        indices = []
        for beat in chosen:
            beat_indices = np.flatnonzero((anatomies == candidate) & (beats == beat))
            beat_indices = beat_indices[np.argsort(samples[beat_indices])]
            if len(beat_indices) != FRAMES_PER_BEAT or not np.array_equal(
                samples[beat_indices], np.arange(FRAMES_PER_BEAT)
            ):
                raise ValueError(
                    f"anatomy {candidate}, beat {beat} is not one complete 50-sample beat"
                )
            indices.extend(beat_indices.tolist())
        return np.asarray(indices, dtype=np.int64)
    raise ValueError("no anatomy contains the requested three consecutive beats")


def _reconstructor(
    family: str,
    config: Path,
    checkpoint_dir: Path,
    model_name: str,
    device: str | None,
):
    if family == "coordinate":
        return CoordinateReconstructor(config, checkpoint_dir, model_name, device=device)
    if family == "global_voltage_slots":
        return GlobalVoltageVesselSlotReconstructor(
            config,
            checkpoint_dir / f"{model_name}_localizer.pt",
            checkpoint_dir / f"{model_name}_refiner.pt",
            device=device,
        )
    raise ValueError(f"unsupported visualized family {family}")


def _infer(reconstructor, voltage: np.ndarray, template: np.ndarray | None = None):
    if getattr(reconstructor, "requires_beat_context", False):
        if template is None:
            raise ValueError("vessel-slot visualization requires stored beat templates")
        return reconstructor.reconstruct(voltage, template)
    return reconstructor.reconstruct(voltage)


def _images_from_elements(reconstructor, values: np.ndarray) -> np.ndarray:
    images = rasterize_frames(reconstructor.mappings, np.asarray(values))
    return np.moveaxis(images, 2, 0)


def _shared_limit(panels: np.ndarray) -> float:
    finite = np.abs(np.asarray(panels)[np.isfinite(panels)])
    if not len(finite):
        raise ValueError("visual panels contain no finite values")
    return max(float(np.quantile(finite, 0.995)), 1e-8)


def _comparison(reference: np.ndarray, estimate: np.ndarray) -> dict[str, float]:
    finite = np.isfinite(reference) & np.isfinite(estimate)
    truth = reference[finite].astype(np.float64)
    prediction = estimate[finite].astype(np.float64)
    truth_rms = max(float(np.sqrt(np.mean(truth * truth))), 1e-12)
    correlation = float(np.corrcoef(truth, prediction)[0, 1])
    return {
        "nrmse": float(np.sqrt(np.mean((prediction - truth) ** 2)) / truth_rms),
        "correlation": correlation,
    }


def _render(
    panels: np.ndarray,
    titles: list[str],
    *,
    output: Path,
    heading: str,
    limit: float,
    frame_ms: int,
    voltage_rms: np.ndarray,
    residual_1: np.ndarray,
    residual_2: np.ndarray,
) -> None:
    frames: list[Image.Image] = []
    for index in range(panels.shape[1]):
        figure, axes = plt.subplots(
            1, len(titles), figsize=(3.25 * len(titles), 3.7), dpi=105, squeeze=False
        )
        shown = None
        for panel, (axis, title) in enumerate(zip(axes[0], titles)):
            shown = axis.imshow(
                panels[panel, index],
                cmap="RdBu_r",
                vmin=-limit,
                vmax=limit,
                origin="upper",
            )
            axis.set_title(title, fontsize=10)
            axis.axis("off")
        beat = index // FRAMES_PER_BEAT + 1
        sample = index % FRAMES_PER_BEAT
        figure.suptitle(f"{heading} | beat {beat}/3, sample {sample + 1}/50", fontsize=11)
        figure.text(
            0.5,
            0.035,
            (
                f"component ΔV RMS={voltage_rms[index]:.3e} V  |  "
                f"S1 forward residual={residual_1[index]:.3e} V  |  "
                f"S2 forward residual={residual_2[index]:.3e} V"
            ),
            ha="center",
            fontsize=8,
        )
        figure.subplots_adjust(left=0.02, right=0.98, top=0.82, bottom=0.18, wspace=0.05)
        if shown is not None:
            bar = figure.colorbar(
                shown, ax=list(axes[0]), orientation="horizontal", fraction=0.06, pad=0.08
            )
            bar.set_label("Referenced conductivity change (S/m)", fontsize=8)
        buffer = BytesIO()
        figure.savefig(buffer, format="png", facecolor="white")
        plt.close(figure)
        buffer.seek(0)
        frames.append(Image.open(buffer).convert("P", palette=Image.Palette.ADAPTIVE))
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        raise FileExistsError(f"immutable GIF already exists: {output}")
    frames[0].save(
        output,
        save_all=True,
        append_images=frames[1:],
        duration=frame_ms,
        loop=0,
        optimize=False,
    )


def synthetic_gif(
    *,
    dataset: Path,
    family: str,
    component: str,
    config: Path,
    checkpoint_dir: Path,
    model_name: str,
    output: Path,
    anatomy: int | None,
    first_beat: int,
    frame_ms: int,
    device: str | None,
) -> dict:
    with np.load(dataset) as source:
        required = (
            "sigma",
            "newton",
            "V",
            "V_template",
            "anatomy_id",
            "beat_id",
            "sample_index",
        )
        missing = [key for key in required if key not in source]
        if missing:
            raise KeyError(f"{dataset} lacks synthetic GIF fields {missing}")
        indices = three_consecutive_beat_indices(
            source["anatomy_id"],
            source["beat_id"],
            source["sample_index"],
            anatomy=anatomy,
            first_beat=first_beat,
        )
        truth = np.asarray(source["sigma"][indices])
        newton = np.asarray(source["newton"][indices])
        voltage = np.asarray(source["V"][indices])
        template = np.asarray(source["V_template"][indices])
        anatomy_value = int(source["anatomy_id"][indices[0]])
        beat_values = [int(value) for value in np.unique(source["beat_id"][indices])]
        sample_values = [int(value) for value in source["sample_index"][indices]]
    reconstructor = _reconstructor(
        family, config, checkpoint_dir, model_name, device
    )
    stage_1, stage_2, residuals = _infer(reconstructor, voltage, template)
    panels = np.stack(
        [
            _images_from_elements(reconstructor, truth),
            _images_from_elements(reconstructor, newton),
            _images_from_elements(reconstructor, stage_1),
            _images_from_elements(reconstructor, stage_2),
        ]
    )
    limit = _shared_limit(panels)
    voltage_rms = np.sqrt(np.mean(voltage * voltage, axis=1))
    _render(
        panels,
        ["Clean truth", "Production Newton", "GCNM S1", "GCNM S2"],
        output=output,
        heading=f"Synthetic exact holdout | {family} | {component.upper()}",
        limit=limit,
        frame_ms=frame_ms,
        voltage_rms=voltage_rms,
        residual_1=np.asarray(residuals["stage_1_forward_voltage_rms"]),
        residual_2=np.asarray(residuals["stage_2_forward_voltage_rms"]),
    )
    return {
        "schema": "pvi-gcnm-three-beat-gif-v1",
        "kind": "synthetic_exact_holdout",
        "family": family,
        "component": component,
        "dataset": str(dataset.resolve()),
        "dataset_sha256": sha256_file(dataset),
        "checkpoint_dir": str(checkpoint_dir.resolve()),
        "model_name": model_name,
        "anatomy_id": anatomy_value,
        "beat_ids": beat_values,
        "sample_indices": sample_values,
        "array_indices": indices.tolist(),
        "frames": len(indices),
        "shared_color_limit_s_m": limit,
        "metrics_against_truth": {
            "newton": _comparison(panels[0], panels[1]),
            "stage_1": _comparison(panels[0], panels[2]),
            "stage_2": _comparison(panels[0], panels[3]),
        },
        "forward_residual_rms": {
            "stage_1_mean": float(
                np.mean(residuals["stage_1_forward_voltage_rms"])
            ),
            "stage_2_mean": float(
                np.mean(residuals["stage_2_forward_voltage_rms"])
            ),
        },
        "gif": str(output.resolve()),
    }


def _rank_one_template(voltage: np.ndarray) -> np.ndarray:
    centered = voltage - np.mean(voltage, axis=0, keepdims=True)
    if not np.any(np.abs(centered) > 1e-15):
        return np.zeros(voltage.shape[1], dtype=np.float64)
    u, singular, vh = np.linalg.svd(centered, full_matrices=False)
    spatial = vh[0].copy()
    temporal = singular[0] * u[:, 0].copy()
    anchor = int(np.argmax(np.abs(spatial)))
    if spatial[anchor] < 0:
        spatial *= -1
        temporal *= -1
    return temporal[int(np.argmax(temporal))] * spatial


def _real_templates(voltage: np.ndarray) -> np.ndarray:
    output = np.empty_like(voltage)
    for start in range(0, len(voltage), FRAMES_PER_BEAT):
        output[start : start + FRAMES_PER_BEAT] = _rank_one_template(
            voltage[start : start + FRAMES_PER_BEAT]
        )
    return output


def _read_exported_sample(
    shards: Path, sample_id: str, columns: list[str]
):
    """Read one sample without materializing complete fixed-size image columns."""

    import pyarrow.parquet as pq

    matches = []
    for path in sorted(Path(shards).glob("*.parquet")):
        parquet = pq.ParquetFile(path)
        for row_group in range(parquet.metadata.num_row_groups):
            identifiers = parquet.read_row_group(row_group, columns=["sample_id"])[
                "sample_id"
            ].to_pylist()
            offsets = [index for index, value in enumerate(identifiers) if value == sample_id]
            if not offsets:
                continue
            table = parquet.read_row_group(
                row_group, columns=["sample_id", *columns]
            )
            matches.extend(table.slice(offset, 1) for offset in offsets)
    if len(matches) != 1:
        raise ValueError(f"expected one exported row for sample_id {sample_id}; found {len(matches)}")
    return matches[0]


def real_gif(
    *,
    hdf5: Path,
    parquet_root: Path,
    family: str,
    component: str,
    output: Path,
    mask_row: int,
    first_beat: int,
    frame_ms: int,
) -> dict:
    """Render the literal exported Parquet stages beside archived Newton."""

    key = "pviHP" if component == "hp" else "pviLP"
    with h5py.File(hdf5, "r") as handle:
        masks = np.asarray(handle["masks/mask05"], dtype=np.int64)
        masks[:, 0] -= 1
        if mask_row < 0 or mask_row >= len(masks):
            raise IndexError("mask row is outside the archived mask05 table")
        start, stop = (int(value) for value in masks[mask_row])
        if stop - start != 5 or first_beat < 0 or first_beat + GIF_BEATS > 5:
            raise ValueError("real GIF requires three beats within one five-beat window")
        frames = np.arange(start * 50, stop * 50)
        resistance = np.asarray(handle[f"data/{key}/resistance"][:, frames]).T
        archived = np.asarray(handle[f"data/{key}/img"][:, :, frames], dtype=np.float32)
        subject_raw = np.asarray(handle["metadata/subject"])[0, 0]
        session_raw = np.asarray(handle["metadata/session"])[0, 0]
        subject = subject_raw.decode() if isinstance(subject_raw, bytes) else str(subject_raw)
        session = session_raw.decode() if isinstance(session_raw, bytes) else str(session_raw)
    voltage = component_reference(resistance_to_voltage(resistance), axis=0)
    archived = archived - archived[:, :, :1]
    source_name = f"{subject}_{session}"
    sample_id = stable_sample_id(source_name, "mask05", start, stop)
    manifest_path = parquet_root / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("representation_family") != family:
        raise ValueError(
            f"Parquet family {manifest.get('representation_family')!r} != {family!r}"
        )
    table = _read_exported_sample(
        parquet_root / "shards",
        sample_id,
        [f"{component}_s1", f"{component}_s2"],
    )

    def exported_image(column: str) -> np.ndarray:
        values = table[column][0].values.to_numpy(zero_copy_only=False)
        return np.asarray(values, dtype=np.float32).reshape(1, 40, 40, 250)[0]

    stage_1 = exported_image(f"{component}_s1")
    stage_2 = exported_image(f"{component}_s2")
    source_report = next(
        (
            report
            for report in manifest.get("source_sessions", [])
            if Path(report["source_hdf5"]).resolve() == hdf5.resolve()
        ),
        None,
    )
    if source_report is None:
        raise ValueError(f"Parquet manifest has no source report for {hdf5}")
    component_report = source_report["components"][component]
    selected = np.arange(first_beat * 50, (first_beat + GIF_BEATS) * 50)
    panels = np.stack(
        [
            np.moveaxis(archived[:, :, selected], 2, 0),
            np.moveaxis(stage_1[:, :, selected], 2, 0),
            np.moveaxis(stage_2[:, :, selected], 2, 0),
        ]
    )
    limit = _shared_limit(panels)
    voltage_rms = np.sqrt(np.mean(voltage[selected] * voltage[selected], axis=1))
    residual_1 = np.full(
        len(selected), component_report["stage_1_forward_voltage_rms"], dtype=float
    )
    residual_2 = np.full(
        len(selected), component_report["stage_2_forward_voltage_rms"], dtype=float
    )
    _render(
        panels,
        [f"Archived Newton {component.upper()} reference", "GCNM S1", "GCNM S2"],
        output=output,
        heading=f"{subject} {session} | {family} | {component.upper()}",
        limit=limit,
        frame_ms=frame_ms,
        voltage_rms=voltage_rms,
        residual_1=residual_1,
        residual_2=residual_2,
    )
    return {
        "schema": "pvi-gcnm-three-beat-gif-v1",
        "kind": "real_archived_newton_reference",
        "family": family,
        "component": component,
        "subject": subject,
        "session": session,
        "source_hdf5": str(hdf5.resolve()),
        "source_hdf5_sha256": source_report["source_hdf5_sha256"],
        "parquet_root": str(parquet_root.resolve()),
        "parquet_manifest_sha256": sha256_file(manifest_path),
        "sample_id": sample_id,
        "mask05_row": mask_row,
        "mask_period_bounds_zero_based": [start, stop],
        "selected_window_frame_indices": selected.tolist(),
        "source_frame_indices": frames[selected].tolist(),
        "frames": len(selected),
        "shared_color_limit_s_m": limit,
        "agreement_with_archived_newton_reference": {
            "stage_1": _comparison(panels[0], panels[1]),
            "stage_2": _comparison(panels[0], panels[2]),
        },
        "forward_residual_rms": {
            "stage_1_mean": float(
                np.mean(residual_1)
            ),
            "stage_2_mean": float(
                np.mean(residual_2)
            ),
        },
        "gif": str(output.resolve()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kind", choices=["synthetic", "real"], required=True)
    parser.add_argument(
        "--family", choices=["coordinate", "global_voltage_slots"], required=True
    )
    parser.add_argument("--component", choices=["full", "hp", "lp"], required=True)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--checkpoint-dir", type=Path)
    parser.add_argument("--model-name")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--dataset", type=Path)
    parser.add_argument("--hdf5", type=Path)
    parser.add_argument("--parquet-root", type=Path)
    parser.add_argument("--anatomy-id", type=int)
    parser.add_argument("--mask-row", type=int, default=0)
    parser.add_argument("--first-beat", type=int, default=0)
    parser.add_argument("--frame-ms", type=int, default=90)
    parser.add_argument("--device")
    args = parser.parse_args()
    if args.kind == "synthetic":
        required = {
            "--dataset": args.dataset,
            "--config": args.config,
            "--checkpoint-dir": args.checkpoint_dir,
            "--model-name": args.model_name,
        }
        missing = [name for name, value in required.items() if value is None]
        if missing:
            raise ValueError(f"synthetic GIF requires {', '.join(missing)}")
        report = synthetic_gif(
            dataset=args.dataset,
            family=args.family,
            component=args.component,
            config=args.config,
            checkpoint_dir=args.checkpoint_dir,
            model_name=args.model_name,
            output=args.output,
            anatomy=args.anatomy_id,
            first_beat=args.first_beat,
            frame_ms=args.frame_ms,
            device=args.device,
        )
    else:
        if args.hdf5 is None or args.parquet_root is None:
            raise ValueError("real GIF requires --hdf5 and --parquet-root")
        report = real_gif(
            hdf5=args.hdf5,
            parquet_root=args.parquet_root,
            family=args.family,
            component=args.component,
            output=args.output,
            mask_row=args.mask_row,
            first_beat=args.first_beat,
            frame_ms=args.frame_ms,
        )
    sidecar = args.output.with_suffix(".json")
    if sidecar.exists():
        raise FileExistsError(f"immutable GIF sidecar already exists: {sidecar}")
    sidecar.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"gif": str(args.output), "sidecar": str(sidecar)}, indent=2))


if __name__ == "__main__":
    main()
