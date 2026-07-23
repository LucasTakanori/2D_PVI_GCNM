#!/usr/bin/env python3
"""Generate accepted default-finger beats for generalized vessel-slot GCNMs.

Unlike the earlier independent-phase pack, every augmented finger contributes
a complete voltage beat template and four stratified supervised phases.  The
simulator's original two arteries and muscle diffusion are retained; no random
nonvascular blobs are added.  Training and validation remain linearized with
fine/coarse anti-inverse-crime meshes, while the small temporal test pack uses
full nonlinear FEM solves.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from gcnm_pvi.anatomical_phantoms import domain_transform
from gcnm_pvi.config import GcnmConfig
from gcnm_pvi.gcnm_differential import production_reconstruction_matrix
from gcnm_pvi.generate_finger_simulator_dataset import (
    _build_jacobian_bank,
    _ellipse_from_boundary,
    _fill_outside,
    _load_simulator,
    _local_to_normalized_mesh,
    _rotation,
    _sha256,
)
from gcnm_pvi.runtime import build_runtime


def _rank_one_template(voltage: np.ndarray) -> np.ndarray:
    values = np.asarray(voltage, dtype=np.float64)
    centered = values - np.mean(values, axis=0, keepdims=True)
    if not np.any(np.abs(centered) > 1e-15):
        return np.zeros(values.shape[1], dtype=np.float64)
    u, singular, vh = np.linalg.svd(centered, full_matrices=False)
    rank_one = singular[0] * u[:, 0:1] @ vh[0:1]
    rms = np.sqrt(np.mean(rank_one**2, axis=1))
    return rank_one[int(np.argmax(rms))]


def _selected_phases(
    waveform: np.ndarray,
    phase_grid: np.ndarray,
    count: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Select peak, natural negative trough, near-zero, and intermediates."""

    if count <= 0 or count >= len(phase_grid):
        return phase_grid.copy()
    values = np.asarray(waveform, dtype=np.float64)[phase_grid]
    chosen = [
        int(np.argmax(values)),
        int(np.argmin(values)),
        int(np.argmin(np.abs(values))),
    ]
    chosen = list(dict.fromkeys(chosen))
    remaining = [index for index in range(len(phase_grid)) if index not in chosen]
    while len(chosen) < count:
        pick = int(rng.choice(remaining))
        chosen.append(pick)
        remaining.remove(pick)
    return np.sort(phase_grid[np.asarray(chosen[:count], dtype=int)])


def _randomize_beat_voltage(
    clean: np.ndarray,
    rng: np.random.Generator,
    *,
    white_noise_rel: float,
    correlated_noise_rel: float,
    noise_floor: float,
    channel_gain_sd: float,
    current_gain_sd: float,
) -> np.ndarray:
    """Apply beat-coherent gains and smooth measurement noise."""

    clean = np.asarray(clean, dtype=np.float64)
    phases, channels = clean.shape
    channel_gain = np.exp(rng.normal(0.0, channel_gain_sd, channels))
    current_gain = float(np.exp(rng.normal(0.0, current_gain_sd)))
    output = current_gain * clean * channel_gain[None, :]
    phase_rms = np.sqrt(np.mean(clean**2, axis=1))
    white_sd = float(noise_floor) + float(white_noise_rel) * phase_rms
    output += rng.normal(size=output.shape) * white_sd[:, None]

    spatial = rng.normal(size=channels)
    spatial /= max(float(np.linalg.norm(spatial)), 1e-12)
    anchors = rng.normal(size=5)
    anchors[-1] = anchors[0]
    temporal = np.interp(
        np.linspace(0.0, 1.0, phases, endpoint=False),
        np.linspace(0.0, 1.0, len(anchors)),
        anchors,
    )
    temporal -= np.mean(temporal)
    peak = max(float(np.max(phase_rms)), float(noise_floor))
    output += (
        float(correlated_noise_rel)
        * peak
        * np.sqrt(channels)
        * temporal[:, None]
        * spatial[None, :]
    )
    return output


def _nonlinear_voltage_beat(
    physics,
    baseline: np.ndarray,
    dynamic: np.ndarray,
    rng: np.random.Generator,
    *,
    contact_static_sd: float,
    contact_drift_sd: float,
) -> np.ndarray:
    electrodes = physics.mesh.elecs
    impedance = np.asarray([float(item.impedance) for item in electrodes])
    static = np.exp(rng.normal(0.0, contact_static_sd, len(electrodes)))
    try:
        for electrode, value in zip(electrodes, impedance * static):
            electrode.impedance = float(value)
        reference = np.asarray(physics.solve(baseline), dtype=np.float64)
        output = []
        for sigma in dynamic:
            drift = np.exp(rng.normal(0.0, contact_drift_sd, len(electrodes)))
            for electrode, value in zip(
                electrodes, impedance * static * drift
            ):
                electrode.impedance = float(value)
            output.append(np.real(physics.solve(sigma) - reference))
    finally:
        for electrode, value in zip(electrodes, impedance):
            electrode.impedance = float(value)
    return np.stack(output)


def _shifted_vessel_record(
    model,
    waveform: np.ndarray,
    phase_index: int,
    ring_mesh,
    domain_center: np.ndarray,
    domain_radius: float,
    phase_shift,
    physical_offset_mm: np.ndarray,
) -> tuple[list[dict[str, float]], list[float]]:
    angles = np.linspace(0.0, 2.0 * np.pi, 256, endpoint=False)
    local_offset = _rotation(
        np.asarray(physical_offset_mm, dtype=np.float64)[None, :],
        float(model.rotation_deg),
    )[0]
    vessels, amplitudes = [], []
    for artery in model.arteries:
        aligned = np.column_stack(
            (
                artery.radius_x_mm * artery.lumen_fraction * np.cos(angles),
                artery.radius_y_mm * artery.lumen_fraction * np.sin(angles),
            )
        )
        local = _rotation(aligned, -float(artery.rotation_deg))
        local += np.array([artery.center_x_mm, artery.center_y_mm])
        normalized = _local_to_normalized_mesh(
            local + local_offset[None, :],
            model,
            ring_mesh,
            domain_center,
            domain_radius,
        )
        vessels.append(_ellipse_from_boundary(normalized))
        shifted = phase_shift(waveform, float(artery.phase_delay_fraction))
        amplitudes.append(
            float(
                artery.peak_delta_s_m
                * artery.waveform_scale
                * shifted[phase_index]
            )
        )
    if len(vessels) != 2:
        raise ValueError("the generalized vessel models require two arteries")
    return vessels, amplitudes


def _generate_split(
    *,
    beats: int,
    samples_per_beat: int,
    seed: int,
    mode: str,
    baseline_model,
    baseline_waveform,
    augmentation: dict,
    noise: dict,
    simulator: dict,
    inverse_mesh,
    forward_mesh,
    domain_center: np.ndarray,
    domain_radius: float,
    physics,
    inverse_matrix: np.ndarray,
    phase_stride: int,
    jacobian_bank_size: int,
    finger_position_mm: float,
) -> tuple[dict[str, np.ndarray], list[dict]]:
    rng = np.random.default_rng(seed)
    spec = simulator["AugmentationSpec"](
        samples=beats, seed=seed, **augmentation
    )
    spec.validate()
    bank = (
        _build_jacobian_bank(
            count=jacobian_bank_size,
            rng=rng,
            spec=spec,
            baseline_model=baseline_model,
            simulator=simulator,
            inverse_mesh=inverse_mesh,
            forward_mesh=forward_mesh,
            physics=physics,
            contact_static_sd=noise["contact_static_sd"],
        )
        if mode == "linearized"
        else []
    )
    names = (
        "sigma",
        "sigma_baseline",
        "sigma_dynamic",
        "V",
        "V_clean",
        "V_template",
        "tissue_labels",
        "phase_index",
        "waveform_amplitude",
        "beat_id",
    )
    arrays: dict[str, list] = {name: [] for name in names}
    records: list[dict] = []
    for beat in range(beats):
        model = simulator["augment_model"](baseline_model, spec, rng)
        waveform_spec = simulator["augment_waveform"](
            baseline_waveform, spec, rng
        )
        offset = rng.uniform(
            -float(finger_position_mm), float(finger_position_mm), size=2
        )
        inverse_points = (
            simulator["mesh_points_mm"](inverse_mesh, model) - offset[None, :]
        )
        forward_points = (
            simulator["mesh_points_mm"](forward_mesh, model) - offset[None, :]
        )
        inverse_result = simulator["simulate_points"](
            inverse_points, model, waveform_spec
        )
        forward_result = simulator["simulate_points"](
            forward_points, model, waveform_spec
        )
        waveform = np.asarray(inverse_result.waveform, dtype=np.float64)
        if not np.allclose(waveform, forward_result.waveform):
            raise RuntimeError("inverse and forward simulator waveforms differ")
        phase_grid = np.arange(0, len(waveform), phase_stride, dtype=int)
        selected = _selected_phases(
            waveform, phase_grid, samples_per_beat, rng
        )
        skin = float(model.conductivities.skin_s_m)
        baseline_inv, delta_inv = _fill_outside(inverse_result, skin)
        baseline_fwd, delta_fwd = _fill_outside(forward_result, skin)
        if mode == "linearized":
            distances = [
                float(np.mean((entry["baseline_inv"] - baseline_inv) ** 2))
                for entry in bank
            ]
            bank_index = int(np.argmin(distances))
            clean_beat = (
                bank[bank_index]["jacobian"] @ delta_fwd[phase_grid].T
            ).T
        else:
            bank_index = None
            clean_beat = _nonlinear_voltage_beat(
                physics,
                baseline_fwd,
                baseline_fwd[None, :] + delta_fwd[phase_grid],
                rng,
                contact_static_sd=noise["contact_static_sd"],
                contact_drift_sd=noise["contact_drift_sd"],
            )
        noisy_beat = _randomize_beat_voltage(
            clean_beat,
            rng,
            white_noise_rel=noise["white_noise_rel"],
            correlated_noise_rel=noise["correlated_noise_rel"],
            noise_floor=noise["noise_floor"],
            channel_gain_sd=noise["channel_gain_sd"],
            current_gain_sd=noise["current_gain_sd"],
        )
        template = _rank_one_template(noisy_beat)
        phase_lookup = {int(phase): i for i, phase in enumerate(phase_grid)}
        for phase in selected:
            context = phase_lookup[int(phase)]
            geometry, amplitudes = _shifted_vessel_record(
                model,
                waveform,
                int(phase),
                inverse_mesh,
                domain_center,
                domain_radius,
                simulator["phase_shift"],
                offset,
            )
            truth = delta_inv[int(phase)]
            arrays["sigma"].append(truth)
            arrays["sigma_baseline"].append(baseline_inv)
            arrays["sigma_dynamic"].append(baseline_inv + truth)
            arrays["V"].append(noisy_beat[context])
            arrays["V_clean"].append(clean_beat[context])
            arrays["V_template"].append(template)
            arrays["tissue_labels"].append(inverse_result.tissue_labels)
            arrays["phase_index"].append(int(phase))
            arrays["waveform_amplitude"].append(float(waveform[int(phase)]))
            arrays["beat_id"].append(beat)
            records.append(
                {
                    "rotation": 0.0,
                    "vessels": geometry,
                    "vessel_delta": amplitudes,
                    "beat_id": beat,
                    "finger_offset_in_ring_mm": offset.tolist(),
                    "simulator_finger_model": model.to_dict(),
                    "waveform": {
                        "kind": waveform_spec.kind,
                        "frames": len(waveform),
                        "duration_s": float(waveform_spec.duration_s),
                        "values": waveform.tolist(),
                        "selected_phase": int(phase),
                        "selected_amplitude": float(waveform[int(phase)]),
                    },
                    "simulation_mode": mode,
                    "jacobian_bank_index": bank_index,
                }
            )
        print(
            f"generated {mode} beat {beat + 1}/{beats} "
            f"({len(selected)} supervised phases)",
            flush=True,
        )
    output = {
        name: np.stack(values).astype(np.float32)
        for name, values in arrays.items()
        if name not in {"phase_index", "beat_id"}
    }
    output["phase_index"] = np.asarray(arrays["phase_index"], dtype=np.int16)
    output["beat_id"] = np.asarray(arrays["beat_id"], dtype=np.int32)
    output["delta_sigma"] = output["sigma"].copy()
    output["resting_absolute_conductivity"] = output["sigma_baseline"].copy()
    output["absolute_conductivity"] = output["sigma_dynamic"].copy()
    output["newton"] = -(
        inverse_matrix @ output["V"].astype(np.float64).T
    ).T.astype(np.float32)
    return output, records


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--simulator-root",
        type=Path,
        default=root.parent / "Finger-Conductivity-Simulator",
    )
    parser.add_argument("--finger-config", type=Path, default=None)
    parser.add_argument(
        "--config",
        type=Path,
        default=root / "configs" / "subject006_anatomical_gcnm.yaml",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=root / "data" / "finger_generalized_beats_US120_v2",
    )
    parser.add_argument("--train-beats", type=int, default=1000)
    parser.add_argument("--validation-beats", type=int, default=200)
    parser.add_argument("--test-beats", type=int, default=8)
    parser.add_argument("--train-phases-per-beat", type=int, default=4)
    parser.add_argument("--validation-phases-per-beat", type=int, default=4)
    parser.add_argument("--frames", type=int, default=50)
    parser.add_argument("--phase-stride", type=int, default=5)
    parser.add_argument("--seed", type=int, default=20260718)
    parser.add_argument("--jacobian-bank-size", type=int, default=32)
    parser.add_argument("--finger-size-variation", type=float, default=0.15)
    parser.add_argument("--finger-rotation-deg", type=float, default=20.0)
    parser.add_argument("--finger-position-mm", type=float, default=1.5)
    parser.add_argument("--artery-size-variation", type=float, default=0.30)
    parser.add_argument("--artery-position-mm", type=float, default=1.8)
    parser.add_argument("--artery-rotation-deg", type=float, default=25.0)
    parser.add_argument("--conductivity-variation", type=float, default=0.20)
    parser.add_argument("--diffusion-variation", type=float, default=0.30)
    parser.add_argument("--waveform-shape-variation", type=float, default=0.20)
    parser.add_argument("--duration-variation", type=float, default=0.15)
    parser.add_argument("--white-noise-rel", type=float, default=0.02)
    parser.add_argument("--correlated-noise-rel", type=float, default=0.015)
    parser.add_argument("--noise-floor", type=float, default=2e-7)
    parser.add_argument("--contact-static-sd", type=float, default=0.08)
    parser.add_argument("--contact-drift-sd", type=float, default=0.001)
    parser.add_argument("--channel-gain-sd", type=float, default=0.015)
    parser.add_argument("--current-gain-sd", type=float, default=0.02)
    parser.add_argument("--allow-overwrite", action="store_true")
    args = parser.parse_args()

    outputs = [
        path
        for split in ("train", "validation", "test")
        for path in (
            args.out_dir / f"{split}.npz",
            args.out_dir / f"{split}_anatomy.json",
        )
    ]
    if not args.allow_overwrite and any(path.exists() for path in outputs):
        raise FileExistsError("dataset already exists; choose a new output directory")

    simulator = _load_simulator(args.simulator_root.resolve())
    finger_config = (
        args.finger_config
        or args.simulator_root / "configs" / "default_finger.json"
    )
    baseline_model = simulator["FingerModel"].from_dict(
        json.loads(finger_config.read_text(encoding="utf-8"))
    )
    baseline_waveform = simulator["WaveformSpec"](
        kind="heartbeat", frames=args.frames, duration_s=1.0
    )
    cfg = GcnmConfig.from_yaml(args.config)
    runtime = build_runtime(cfg, include_forward=True)
    inverse_matrix = production_reconstruction_matrix(
        runtime["physics_inv"],
        hyper_pvi=cfg.hyper_pvi,
        regularizer=runtime["mappings"].laplace,
    )
    inverse_mesh = simulator["load_ring_mesh"](
        Path(cfg.mesh_inv_h5), "generalized_inverse"
    )
    forward_mesh = simulator["load_ring_mesh"](
        Path(cfg.mesh_fwd_h5), "generalized_forward"
    )
    center, radius = domain_transform(runtime["mesh_inv"])
    augmentation = {
        "finger_size_fraction": args.finger_size_variation,
        "finger_rotation_deg": args.finger_rotation_deg,
        "artery_size_fraction": args.artery_size_variation,
        "artery_position_mm": args.artery_position_mm,
        "artery_rotation_deg": args.artery_rotation_deg,
        "conductivity_fraction": args.conductivity_variation,
        "diffusion_fraction": args.diffusion_variation,
        "waveform_shape_fraction": args.waveform_shape_variation,
        "duration_fraction": args.duration_variation,
    }
    noise = {
        "white_noise_rel": args.white_noise_rel,
        "correlated_noise_rel": args.correlated_noise_rel,
        "noise_floor": args.noise_floor,
        "contact_static_sd": args.contact_static_sd,
        "contact_drift_sd": args.contact_drift_sd,
        "channel_gain_sd": args.channel_gain_sd,
        "current_gain_sd": args.current_gain_sd,
    }
    splits = {
        "train": (args.train_beats, args.train_phases_per_beat, "linearized"),
        "validation": (
            args.validation_beats,
            args.validation_phases_per_beat,
            "linearized",
        ),
        "test": (args.test_beats, 0, "nonlinear"),
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    counts = {}
    for offset, (split, (beats, phases, mode)) in enumerate(splits.items()):
        arrays, records = _generate_split(
            beats=beats,
            samples_per_beat=phases,
            seed=args.seed + offset,
            mode=mode,
            baseline_model=baseline_model,
            baseline_waveform=baseline_waveform,
            augmentation=augmentation,
            noise=noise,
            simulator=simulator,
            inverse_mesh=inverse_mesh,
            forward_mesh=forward_mesh,
            domain_center=center,
            domain_radius=radius,
            physics=runtime["physics_fwd"],
            inverse_matrix=inverse_matrix,
            phase_stride=args.phase_stride,
            jacobian_bank_size=args.jacobian_bank_size,
            finger_position_mm=args.finger_position_mm,
        )
        np.savez_compressed(args.out_dir / f"{split}.npz", **arrays)
        (args.out_dir / f"{split}_anatomy.json").write_text(
            json.dumps(records, indent=2), encoding="utf-8"
        )
        counts[split] = {"beats": beats, "samples": len(arrays["sigma"])}
    metadata = {
        "schema": "finger-generalized-signed-beats-v2",
        "target": "clean signed differential conductivity",
        "source_model": "accepted default finger with two arteries and muscle diffusion",
        "random_nonvascular_fields": False,
        "counts": counts,
        "phase_stride": args.phase_stride,
        "split_policy": "all supervised phases from one augmented anatomy stay in one split",
        "training_physics": "linearized fine-mesh anatomical Jacobian bank",
        "test_physics": "exact nonlinear two-solve beat",
        "inverse_baseline": "homogeneous 0.7 S/m during GCNM training and inference",
        "jacobian_bank_size": args.jacobian_bank_size,
        "augmentation": augmentation,
        "finger_position_mm": args.finger_position_mm,
        "noise": noise,
        "finger_config": str(finger_config.resolve()),
        "finger_config_sha256": _sha256(finger_config),
        "config": str(args.config.resolve()),
        "mesh": "subject006 US120; 5320 forward / 1330 inverse elements",
        "seed": args.seed,
    }
    (args.out_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
