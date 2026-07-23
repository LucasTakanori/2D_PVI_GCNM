#!/usr/bin/env python3
"""Generate signed, beat-grouped, multi-subject finger data for one ring mesh.

Each augmented anatomy contributes a complete beat context while a
configurable subset of ordered samples is stored for training.  The target is clean
differential conductivity.  Measurements include coherent beat noise,
contact variability, occasional channel artifacts, and a configurable mix of
linearized and exact nonlinear forward solves.
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


def _signed_waveform_spec(simulator: dict, points: np.ndarray, model, spec):
    probe = simulator["simulate_points"](points[:1], model, spec)
    values = np.asarray(probe.waveform, dtype=np.float64)
    values = values - float(np.mean(values))
    values = values / max(float(np.max(np.abs(values))), 1e-12)
    return simulator["WaveformSpec"](
        kind="custom",
        frames=len(values),
        duration_s=float(spec.duration_s),
        custom_values=values.tolist(),
        normalize="none",
    )


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
    """Map artery geometry back to the ring after whole-finger translation."""

    angles = np.linspace(0.0, 2.0 * np.pi, 256, endpoint=False)
    # _local_to_normalized_mesh rotates local points into ring coordinates.
    # Express the physical ring offset in local coordinates before that call.
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
        local_boundary = _rotation(aligned, -float(artery.rotation_deg))
        local_boundary += np.array([artery.center_x_mm, artery.center_y_mm])
        normalized = _local_to_normalized_mesh(
            local_boundary + local_offset[None, :],
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
        raise ValueError("core-guided training requires exactly two arteries")
    return vessels, amplitudes


def _selected_samples(
    waveform: np.ndarray,
    phase_grid: np.ndarray,
    count: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Select ordered beat samples across peak, trough, zero, and intermediates."""

    if count <= 0 or count >= len(phase_grid):
        return phase_grid.copy()
    values = np.asarray(waveform)[phase_grid]
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


def _selected_phases(
    waveform: np.ndarray,
    phase_grid: np.ndarray,
    count: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Backward-compatible alias for pre-``samples per beat`` callers.

    ``sample`` is the public terminology now, but older core-guided utilities
    imported this helper by its former name.  Keeping a small wrapper avoids a
    needless behavioral fork and lets the existing tests verify the same
    selection rule.
    """

    return _selected_samples(waveform, phase_grid, count, rng)


def _rank_one_template(voltage: np.ndarray) -> np.ndarray:
    """Denoise one beat and return its strongest rank-one voltage frame."""

    values = np.asarray(voltage, dtype=np.float64)
    centered = values - np.mean(values, axis=0, keepdims=True)
    if not np.any(np.abs(centered) > 1e-15):
        return np.zeros(values.shape[1], dtype=np.float64)
    u, singular, vh = np.linalg.svd(centered, full_matrices=False)
    spatial = vh[0].copy()
    temporal = singular[0] * u[:, 0].copy()
    # SVD signs and equal-magnitude systolic/diastolic extrema are otherwise
    # ambiguous. Canonicalize the spatial mode, then select its strongest
    # positive temporal coefficient so training and HDF export agree exactly.
    anchor = int(np.argmax(np.abs(spatial)))
    if spatial[anchor] < 0:
        spatial *= -1.0
        temporal *= -1.0
    return temporal[int(np.argmax(temporal))] * spatial


def _randomize_beat_voltage(
    clean: np.ndarray,
    rng: np.random.Generator,
    *,
    white_noise_rel: float,
    correlated_noise_rel: float,
    noise_floor: float,
    channel_gain_sd: float,
    current_gain_sd: float,
    artifact_probability: float,
    artifact_scale: float,
) -> np.ndarray:
    """Apply coherent hardware variation plus phase-dependent measurement noise."""

    clean = np.asarray(clean, dtype=np.float64)
    phases, channels = clean.shape
    channel_gain = np.exp(rng.normal(0.0, channel_gain_sd, channels))
    current_gain = float(np.exp(rng.normal(0.0, current_gain_sd)))
    output = current_gain * clean * channel_gain[None, :]
    phase_rms = np.sqrt(np.mean(clean**2, axis=1))
    peak_rms = max(float(np.max(phase_rms)), float(noise_floor))
    white_sd = float(noise_floor) + float(white_noise_rel) * phase_rms
    output += rng.normal(size=output.shape) * white_sd[:, None]

    spatial_mode = rng.normal(size=channels)
    spatial_mode /= max(float(np.linalg.norm(spatial_mode)), 1e-12)
    anchors = rng.normal(size=5)
    anchors[-1] = anchors[0]
    temporal = np.interp(
        np.linspace(0.0, 1.0, phases, endpoint=False),
        np.linspace(0.0, 1.0, len(anchors)),
        anchors,
    )
    temporal -= np.mean(temporal)
    output += (
        float(correlated_noise_rel)
        * peak_rms
        * np.sqrt(channels)
        * temporal[:, None]
        * spatial_mode[None, :]
    )

    if rng.random() < float(artifact_probability):
        phase = int(rng.integers(0, phases))
        count = int(rng.integers(1, min(4, channels) + 1))
        selected = rng.choice(channels, size=count, replace=False)
        output[phase, selected] += rng.normal(
            0.0, float(artifact_scale) * peak_rms, count
        )
    return output


def _nonvascular_delta(
    inverse_points: np.ndarray,
    forward_points: np.ndarray,
    inverse_labels: np.ndarray,
    forward_labels: np.ndarray,
    waveform: np.ndarray,
    rng: np.random.Generator,
    *,
    probability: float,
    maximum_amplitude: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Occasionally add a weak, legitimate smooth muscle conductivity field."""

    inverse = np.zeros((len(waveform), len(inverse_points)), dtype=np.float64)
    forward = np.zeros((len(waveform), len(forward_points)), dtype=np.float64)
    muscle_indices = np.flatnonzero(inverse_labels == 3)
    if not len(muscle_indices) or rng.random() >= float(probability):
        return inverse, forward
    blobs = int(rng.integers(1, 3))
    temporal = np.roll(waveform, int(rng.integers(-3, 4)))
    for _ in range(blobs):
        center = inverse_points[int(rng.choice(muscle_indices))]
        width_x = float(rng.uniform(1.4, 3.5))
        width_y = float(rng.uniform(1.4, 3.5))
        angle = float(rng.uniform(-np.pi, np.pi))
        cosine, sine = np.cos(angle), np.sin(angle)
        amplitude = float(
            rng.choice((-1.0, 1.0))
            * rng.uniform(0.25, 1.0)
            * maximum_amplitude
        )
        for points, labels, output in (
            (inverse_points, inverse_labels, inverse),
            (forward_points, forward_labels, forward),
        ):
            offset = points - center[None, :]
            local_x = cosine * offset[:, 0] + sine * offset[:, 1]
            local_y = -sine * offset[:, 0] + cosine * offset[:, 1]
            field = amplitude * np.exp(
                -0.5 * ((local_x / width_x) ** 2 + (local_y / width_y) ** 2)
            )
            field[labels != 3] = 0.0
            output += temporal[:, None] * field[None, :]
    return inverse, forward


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
            for electrode, value in zip(electrodes, impedance * static * drift):
                electrode.impedance = float(value)
            output.append(np.real(physics.solve(sigma) - reference))
    finally:
        for electrode, value in zip(electrodes, impedance):
            electrode.impedance = float(value)
    return np.stack(output)


def _generate_split(
    *,
    beats: int,
    samples_per_beat: int,
    seed: int,
    baseline_model,
    baseline_waveform,
    augmentation: dict,
    simulator: dict,
    inverse_mesh,
    forward_mesh,
    domain_center: np.ndarray,
    domain_radius: float,
    physics,
    inverse_matrix: np.ndarray,
    phase_stride: int,
    nonlinear_fraction: float,
    jacobian_bank_size: int,
    noise: dict,
    nonvascular_probability: float,
    nonvascular_amplitude: float,
    finger_position_mm: float,
) -> tuple[dict[str, np.ndarray], list[dict]]:
    rng = np.random.default_rng(seed)
    spec = simulator["AugmentationSpec"](samples=beats, seed=seed, **augmentation)
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
        if nonlinear_fraction < 1.0
        else []
    )
    arrays: dict[str, list] = {
        key: []
        for key in (
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
            "subject_id",
        )
    }
    records: list[dict] = []
    for beat in range(beats):
        model = simulator["augment_model"](baseline_model, spec, rng)
        waveform_spec = simulator["augment_waveform"](baseline_waveform, spec, rng)
        physical_offset = rng.uniform(
            -float(finger_position_mm),
            float(finger_position_mm),
            size=2,
        )
        inverse_points = (
            simulator["mesh_points_mm"](inverse_mesh, model)
            - physical_offset[None, :]
        )
        forward_points = (
            simulator["mesh_points_mm"](forward_mesh, model)
            - physical_offset[None, :]
        )
        signed_spec = _signed_waveform_spec(
            simulator, inverse_points, model, waveform_spec
        )
        inverse_result = simulator["simulate_points"](
            inverse_points, model, signed_spec
        )
        forward_result = simulator["simulate_points"](
            forward_points, model, signed_spec
        )
        waveform = np.asarray(inverse_result.waveform, dtype=np.float64)
        phase_grid = np.arange(0, len(waveform), phase_stride, dtype=int)
        if len(phase_grid) < 4:
            raise ValueError("beat context requires at least four sampled phases")
        skin = float(model.conductivities.skin_s_m)
        baseline_inv, delta_inv = _fill_outside(inverse_result, skin)
        baseline_fwd, delta_fwd = _fill_outside(forward_result, skin)
        extra_inv, extra_fwd = _nonvascular_delta(
            inverse_points,
            forward_points,
            inverse_result.tissue_labels,
            forward_result.tissue_labels,
            waveform,
            rng,
            probability=nonvascular_probability,
            maximum_amplitude=nonvascular_amplitude,
        )
        delta_inv = delta_inv + extra_inv
        delta_fwd = delta_fwd + extra_fwd

        exact = rng.random() < nonlinear_fraction
        if exact:
            clean_beat = _nonlinear_voltage_beat(
                physics,
                baseline_fwd,
                baseline_fwd[None, :] + delta_fwd[phase_grid],
                rng,
                contact_static_sd=noise["contact_static_sd"],
                contact_drift_sd=noise["contact_drift_sd"],
            )
            bank_index = None
            mode = "nonlinear"
        else:
            distances = [
                float(np.mean((entry["baseline_inv"] - baseline_inv) ** 2))
                for entry in bank
            ]
            bank_index = int(np.argmin(distances))
            clean_beat = (
                bank[bank_index]["jacobian"] @ delta_fwd[phase_grid].T
            ).T
            mode = "linearized"
        noisy_beat = _randomize_beat_voltage(
            clean_beat,
            rng,
            white_noise_rel=noise["white_noise_rel"],
            correlated_noise_rel=noise["correlated_noise_rel"],
            noise_floor=noise["noise_floor"],
            channel_gain_sd=noise["channel_gain_sd"],
            current_gain_sd=noise["current_gain_sd"],
            artifact_probability=noise["artifact_probability"],
            artifact_scale=noise["artifact_scale"],
        )
        template = _rank_one_template(noisy_beat)
        selected = _selected_samples(
            waveform, phase_grid, samples_per_beat, rng
        )
        phase_to_context = {int(value): i for i, value in enumerate(phase_grid)}
        for phase in selected:
            context_index = phase_to_context[int(phase)]
            vessel_geometry, vessel_delta = _shifted_vessel_record(
                model,
                waveform,
                int(phase),
                inverse_mesh,
                domain_center,
                domain_radius,
                simulator["phase_shift"],
                physical_offset,
            )
            truth = delta_inv[int(phase)]
            arrays["sigma"].append(truth)
            arrays["sigma_baseline"].append(baseline_inv)
            arrays["sigma_dynamic"].append(baseline_inv + truth)
            arrays["V"].append(noisy_beat[context_index])
            arrays["V_clean"].append(clean_beat[context_index])
            arrays["V_template"].append(template)
            arrays["tissue_labels"].append(inverse_result.tissue_labels)
            arrays["phase_index"].append(int(phase))
            arrays["waveform_amplitude"].append(float(waveform[int(phase)]))
            arrays["beat_id"].append(beat)
            arrays["subject_id"].append(beat)
            records.append(
                {
                    "rotation": 0.0,
                    "vessels": vessel_geometry,
                    "vessel_delta": vessel_delta,
                    "beat_id": beat,
                    "subject_id": beat,
                    "finger_offset_in_ring_mm": physical_offset.tolist(),
                    "simulator_finger_model": model.to_dict(),
                    "waveform": {
                        "kind": "signed high-pass heartbeat",
                        "frames": len(waveform),
                        "duration_s": float(signed_spec.duration_s),
                        "values": waveform.tolist(),
                        "selected_phase": int(phase),
                        "selected_amplitude": float(waveform[int(phase)]),
                    },
                    "simulation_mode": mode,
                    "jacobian_bank_index": bank_index,
                    "has_nonvascular_change": bool(np.any(extra_inv)),
                }
            )
        print(
            f"generated beat {beat + 1}/{beats} ({mode}, {len(selected)} samples)",
            flush=True,
        )
    output = {
        key: np.stack(value).astype(np.float32)
        for key, value in arrays.items()
        if key not in {"phase_index", "beat_id", "subject_id"}
    }
    output["phase_index"] = np.asarray(arrays["phase_index"], dtype=np.int16)
    # sample_index is the preferred public name. phase_index is retained so
    # existing model/evaluation code can read this schema without translation.
    output["sample_index"] = output["phase_index"].copy()
    output["beat_id"] = np.asarray(arrays["beat_id"], dtype=np.int32)
    output["subject_id"] = np.asarray(arrays["subject_id"], dtype=np.int32)
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
        default=root / "data" / "finger_multisubject_beats_US120_v1",
    )
    parser.add_argument("--train-beats", type=int, default=1000)
    parser.add_argument("--validation-beats", type=int, default=200)
    parser.add_argument("--test-beats", type=int, default=32)
    parser.add_argument(
        "--train-samples-per-beat",
        "--train-phases-per-beat",
        dest="train_samples_per_beat",
        type=int,
        default=4,
        help="ordered samples stored from each training beat",
    )
    parser.add_argument(
        "--validation-samples-per-beat",
        "--validation-phases-per-beat",
        dest="validation_samples_per_beat",
        type=int,
        default=4,
        help="ordered samples stored from each validation beat",
    )
    parser.add_argument("--frames", type=int, default=50)
    parser.add_argument(
        "--sample-stride",
        "--phase-stride",
        dest="sample_stride",
        type=int,
        default=5,
        help="stride on the generated beat before sample selection; use 1 for all 50 samples",
    )
    parser.add_argument("--seed", type=int, default=20260716)
    parser.add_argument("--jacobian-bank-size", type=int, default=16)
    parser.add_argument("--train-nonlinear-fraction", type=float, default=0.10)
    parser.add_argument("--validation-nonlinear-fraction", type=float, default=0.25)
    parser.add_argument("--finger-size-variation", type=float, default=0.15)
    parser.add_argument("--finger-rotation-deg", type=float, default=20.0)
    parser.add_argument("--finger-position-mm", type=float, default=1.0)
    parser.add_argument("--artery-size-variation", type=float, default=0.35)
    parser.add_argument("--artery-position-mm", type=float, default=1.8)
    parser.add_argument("--artery-rotation-deg", type=float, default=30.0)
    parser.add_argument("--conductivity-variation", type=float, default=0.20)
    parser.add_argument("--diffusion-variation", type=float, default=0.75)
    parser.add_argument("--waveform-shape-variation", type=float, default=0.20)
    parser.add_argument("--duration-variation", type=float, default=0.15)
    parser.add_argument("--white-noise-rel", type=float, default=0.04)
    parser.add_argument("--correlated-noise-rel", type=float, default=0.04)
    parser.add_argument("--noise-floor", type=float, default=1e-6)
    parser.add_argument("--contact-static-sd", type=float, default=0.08)
    parser.add_argument("--contact-drift-sd", type=float, default=0.002)
    parser.add_argument("--channel-gain-sd", type=float, default=0.015)
    parser.add_argument("--current-gain-sd", type=float, default=0.03)
    parser.add_argument("--artifact-probability", type=float, default=0.03)
    parser.add_argument("--artifact-scale", type=float, default=0.25)
    parser.add_argument("--nonvascular-probability", type=float, default=0.40)
    parser.add_argument("--nonvascular-amplitude", type=float, default=0.006)
    parser.add_argument("--allow-overwrite", action="store_true")
    args = parser.parse_args()

    if not 0.0 <= args.train_nonlinear_fraction <= 1.0:
        raise ValueError("train nonlinear fraction must be in [0, 1]")
    if not 0.0 <= args.validation_nonlinear_fraction <= 1.0:
        raise ValueError("validation nonlinear fraction must be in [0, 1]")
    if args.frames <= 0 or args.sample_stride <= 0:
        raise ValueError("frames and sample stride must be positive")
    available_samples = len(range(0, args.frames, args.sample_stride))
    for split, value in (
        ("train", args.train_samples_per_beat),
        ("validation", args.validation_samples_per_beat),
    ):
        if value <= 0 or value > available_samples:
            raise ValueError(
                f"{split} samples per beat must be in [1, {available_samples}]"
            )
    outputs = [args.out_dir / f"{split}.npz" for split in ("train", "validation", "test")]
    outputs.extend(
        args.out_dir / f"{split}_anatomy.json"
        for split in ("train", "validation", "test")
    )
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
        Path(cfg.mesh_inv_h5), "core_guided_inverse"
    )
    forward_mesh = simulator["load_ring_mesh"](
        Path(cfg.mesh_fwd_h5), "core_guided_forward"
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
        "artifact_probability": args.artifact_probability,
        "artifact_scale": args.artifact_scale,
    }
    split_specs = {
        "train": (
            args.train_beats,
            args.train_samples_per_beat,
            args.train_nonlinear_fraction,
        ),
        "validation": (
            args.validation_beats,
            args.validation_samples_per_beat,
            args.validation_nonlinear_fraction,
        ),
        "test": (args.test_beats, 0, 1.0),
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    counts = {}
    for offset, (split, (beats, samples_per_beat, nonlinear_fraction)) in enumerate(
        split_specs.items()
    ):
        values, records = _generate_split(
            beats=beats,
            samples_per_beat=samples_per_beat,
            seed=args.seed + offset,
            baseline_model=baseline_model,
            baseline_waveform=baseline_waveform,
            augmentation=augmentation,
            simulator=simulator,
            inverse_mesh=inverse_mesh,
            forward_mesh=forward_mesh,
            domain_center=center,
            domain_radius=radius,
            physics=runtime["physics_fwd"],
            inverse_matrix=inverse_matrix,
            phase_stride=args.sample_stride,
            nonlinear_fraction=nonlinear_fraction,
            jacobian_bank_size=args.jacobian_bank_size,
            noise=noise,
            nonvascular_probability=args.nonvascular_probability,
            nonvascular_amplitude=args.nonvascular_amplitude,
            finger_position_mm=args.finger_position_mm,
        )
        np.savez_compressed(args.out_dir / f"{split}.npz", **values)
        (args.out_dir / f"{split}_anatomy.json").write_text(
            json.dumps(records, indent=2), encoding="utf-8"
        )
        counts[split] = {
            "beats": beats,
            "samples_per_beat": int(len(values["sigma"]) // max(beats, 1)),
            "samples": len(values["sigma"]),
        }
    metadata = {
        "schema": "finger-multisubject-signed-beats-v2",
        "target": "clean signed differential conductivity",
        "ring_model_policy": "one model per fixed ring mesh",
        "subject_split_policy": "one augmented anatomy belongs to one split only",
        "counts": counts,
        "beat_resolution": {
            "generated_samples_per_beat": args.frames,
            "sample_stride": args.sample_stride,
            "stored_samples_per_beat": {
                key: value[1] if value[1] > 0 else available_samples
                for key, value in split_specs.items()
            },
            "ordering": "chronological sample index within each beat",
            "compatibility_alias": "phase_index duplicates sample_index",
        },
        "augmentation": augmentation,
        "whole_finger_position_mm": args.finger_position_mm,
        "noise": noise,
        "nonlinear_fraction": {
            key: value[2] for key, value in split_specs.items()
        },
        "nonvascular_change": {
            "probability": args.nonvascular_probability,
            "maximum_amplitude_s_m": args.nonvascular_amplitude,
            "support": "muscle only",
        },
        "anti_inverse_crime": "5320-element forward / 1330-element inverse",
        "inverse_baseline_used_by_training": "homogeneous 0.7 S/m; anatomical baselines are forward-data domain randomization",
        "finger_config": str(finger_config.resolve()),
        "finger_config_sha256": _sha256(finger_config),
        "config": str(args.config.resolve()),
        "config_sha256": _sha256(args.config),
        "mesh_inverse": str(Path(cfg.mesh_inv_h5).resolve()),
        "mesh_inverse_sha256": _sha256(Path(cfg.mesh_inv_h5)),
        "mesh_forward": str(Path(cfg.mesh_fwd_h5).resolve()),
        "mesh_forward_sha256": _sha256(Path(cfg.mesh_fwd_h5)),
        "mapping_40": str(Path(cfg.mappings_h5).resolve()),
        "mapping_40_sha256": _sha256(Path(cfg.mappings_h5)),
        "seed": args.seed,
    }
    (args.out_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
