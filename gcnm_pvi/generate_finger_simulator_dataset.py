#!/usr/bin/env python3
"""Convert the default finger simulator into anatomical PVI-GCNM data.

One augmented finger and heartbeat are drawn per sample.  A non-zero cardiac
phase is selected and the identical anatomy is rasterized on the inverse and
forward meshes.  Training/validation use a fine-mesh anatomical Jacobian bank;
the independent test pack uses two full FEM solves.  The clean simulator delta
conductivity remains the supervised target.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import sys

import numpy as np

from gcnm_pvi.anatomical_phantoms import domain_transform
from gcnm_pvi.config import GcnmConfig
from gcnm_pvi.gcnm_differential import production_reconstruction_matrix
from gcnm_pvi.generate_anatomical_dataset import (
    _randomize_delta_voltage,
    _simulate_voltage_pair,
)
from gcnm_pvi.runtime import build_runtime


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_simulator(simulator_root: Path):
    source = simulator_root / "src"
    if not source.is_dir():
        raise FileNotFoundError(f"finger simulator source not found: {source}")
    sys.path.insert(0, str(source))
    from finger_sim.augmentation import (  # noqa: PLC0415
        AugmentationSpec,
        augment_model,
        augment_waveform,
    )
    from finger_sim.mesh import load_ring_mesh, mesh_points_mm  # noqa: PLC0415
    from finger_sim.models import FingerModel, WaveformSpec  # noqa: PLC0415
    from finger_sim.simulation import simulate_points  # noqa: PLC0415
    from finger_sim.waveforms import phase_shift  # noqa: PLC0415

    return {
        "AugmentationSpec": AugmentationSpec,
        "augment_model": augment_model,
        "augment_waveform": augment_waveform,
        "load_ring_mesh": load_ring_mesh,
        "mesh_points_mm": mesh_points_mm,
        "FingerModel": FingerModel,
        "WaveformSpec": WaveformSpec,
        "simulate_points": simulate_points,
        "phase_shift": phase_shift,
    }


def _rotation(points: np.ndarray, degrees: float) -> np.ndarray:
    angle = np.deg2rad(degrees)
    cosine, sine = np.cos(angle), np.sin(angle)
    matrix = np.array([[cosine, sine], [-sine, cosine]])
    return np.asarray(points, dtype=np.float64) @ matrix.T


def _local_to_normalized_mesh(
    local_points_mm: np.ndarray,
    model,
    ring_mesh,
    domain_center: np.ndarray,
    domain_radius: float,
) -> np.ndarray:
    """Map finger-local millimetres back to GCNM normalized coordinates."""
    physical_mm = _rotation(local_points_mm, -float(model.rotation_deg))
    mesh_center = np.mean(ring_mesh.nodes, axis=0)
    centered_nodes = ring_mesh.nodes - mesh_center
    x_scale = max(float(np.max(np.abs(centered_nodes[:, 0]))), 1e-12)
    y_scale = max(float(np.max(np.abs(centered_nodes[:, 1]))), 1e-12)
    mesh_points = np.empty_like(physical_mm)
    mesh_points[:, 0] = (
        physical_mm[:, 0] * (2.0 * x_scale) / float(model.width_mm)
        + mesh_center[0]
    )
    mesh_points[:, 1] = (
        physical_mm[:, 1] * (2.0 * y_scale) / float(model.height_mm)
        + mesh_center[1]
    )
    return (mesh_points - domain_center[None, :]) / float(domain_radius)


def _ellipse_from_boundary(points: np.ndarray) -> dict[str, float]:
    center = np.mean(points, axis=0)
    covariance = np.cov(points - center, rowvar=False, bias=True)
    values, vectors = np.linalg.eigh(covariance)
    order = np.argsort(values)[::-1]
    values = values[order]
    vectors = vectors[:, order]
    axes = np.sqrt(np.maximum(2.0 * values, 1e-12))
    direction = vectors[:, 0]
    return {
        "center_x": float(center[0]),
        "center_y": float(center[1]),
        "axis_a": float(axes[0]),
        "axis_b": float(axes[1]),
        "angle": float(math.atan2(direction[1], direction[0])),
    }


def _vessel_record(
    model,
    waveform: np.ndarray,
    phase_index: int,
    ring_mesh,
    domain_center: np.ndarray,
    domain_radius: float,
    phase_shift,
) -> tuple[list[dict[str, float]], list[float]]:
    angles = np.linspace(0.0, 2.0 * np.pi, 256, endpoint=False)
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
            local_boundary,
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
        raise ValueError("the requested vessel-slot experiments require two arteries")
    return vessels, amplitudes


def _fill_outside(result, skin_conductivity: float) -> tuple[np.ndarray, np.ndarray]:
    baseline = np.nan_to_num(
        np.asarray(result.sigma_baseline, dtype=np.float64),
        nan=float(skin_conductivity),
    )
    delta = np.nan_to_num(
        np.asarray(result.delta_sigma, dtype=np.float64),
        nan=0.0,
    )
    return baseline, delta


def _select_phase(waveform: np.ndarray, rng: np.random.Generator, minimum: float) -> int:
    positive = np.flatnonzero(waveform >= float(minimum))
    if not len(positive):
        raise ValueError("augmented heartbeat has no phase above the requested minimum")
    desired = float(rng.uniform(minimum, max(float(np.max(waveform)), minimum)))
    return int(positive[np.argmin(np.abs(waveform[positive] - desired))])


def _build_jacobian_bank(
    *,
    count: int,
    rng: np.random.Generator,
    spec,
    baseline_model,
    simulator: dict,
    inverse_mesh,
    forward_mesh,
    physics,
    contact_static_sd: float,
) -> list[dict[str, np.ndarray]]:
    electrodes = physics.mesh.elecs
    original_impedance = np.asarray(
        [float(electrode.impedance) for electrode in electrodes]
    )
    bank = []
    try:
        for index in range(count):
            model = simulator["augment_model"](baseline_model, spec, rng)
            inverse_result = simulator["simulate_points"](
                simulator["mesh_points_mm"](inverse_mesh, model),
                model,
                simulator["WaveformSpec"](),
            )
            forward_result = simulator["simulate_points"](
                simulator["mesh_points_mm"](forward_mesh, model),
                model,
                simulator["WaveformSpec"](),
            )
            skin = float(model.conductivities.skin_s_m)
            baseline_inv, _ = _fill_outside(inverse_result, skin)
            baseline_fwd, _ = _fill_outside(forward_result, skin)
            factors = np.exp(
                rng.normal(0.0, contact_static_sd, len(original_impedance))
            )
            for electrode, impedance in zip(
                electrodes, original_impedance * factors
            ):
                electrode.impedance = float(impedance)
            forward = physics._forward(baseline_fwd)
            jacobian = np.asarray(physics.jacobian_from_forward(forward), dtype=np.float64)
            bank.append({"baseline_inv": baseline_inv, "jacobian": jacobian})
            print(f"built simulator fine-mesh Jacobian {index + 1}/{count}", flush=True)
    finally:
        for electrode, impedance in zip(electrodes, original_impedance):
            electrode.impedance = float(impedance)
    return bank


def _generate_split(
    *,
    count: int,
    seed: int,
    baseline_model,
    baseline_waveform,
    augmentation_kwargs: dict,
    simulator: dict,
    inverse_mesh,
    forward_mesh,
    domain_center: np.ndarray,
    domain_radius: float,
    physics,
    inverse_matrix: np.ndarray,
    minimum_waveform: float,
    noise: dict[str, float],
    simulation_mode: str,
    jacobian_bank_size: int,
) -> tuple[dict[str, np.ndarray], list[dict]]:
    rng = np.random.default_rng(seed)
    spec = simulator["AugmentationSpec"](samples=count, seed=seed, **augmentation_kwargs)
    spec.validate()
    jacobian_bank = (
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
        if simulation_mode == "linearized"
        else []
    )
    truth, baselines, dynamic, voltages, clean_voltages = [], [], [], [], []
    labels, points, phases, amplitudes, records = [], [], [], [], []
    for index in range(count):
        model = simulator["augment_model"](baseline_model, spec, rng)
        waveform_spec = simulator["augment_waveform"](baseline_waveform, spec, rng)
        inverse_points = simulator["mesh_points_mm"](inverse_mesh, model)
        forward_points = simulator["mesh_points_mm"](forward_mesh, model)
        inverse_result = simulator["simulate_points"](
            inverse_points, model, waveform_spec
        )
        forward_result = simulator["simulate_points"](
            forward_points, model, waveform_spec
        )
        if not np.allclose(inverse_result.waveform, forward_result.waveform):
            raise RuntimeError("inverse and forward simulator waveforms differ")
        phase = _select_phase(inverse_result.waveform, rng, minimum_waveform)
        skin = float(model.conductivities.skin_s_m)
        baseline_inv, delta_inv_all = _fill_outside(inverse_result, skin)
        baseline_fwd, delta_fwd_all = _fill_outside(forward_result, skin)
        delta_inv = delta_inv_all[phase]
        delta_fwd = delta_fwd_all[phase]
        bank_index = None
        if simulation_mode == "linearized":
            distances = [
                float(np.mean((entry["baseline_inv"] - baseline_inv) ** 2))
                for entry in jacobian_bank
            ]
            bank_index = int(np.argmin(distances))
            delta_voltage_clean = jacobian_bank[bank_index]["jacobian"] @ delta_fwd
        else:
            sigma_dynamic_fwd = baseline_fwd + delta_fwd
            v0, v1 = _simulate_voltage_pair(
                physics,
                baseline_fwd,
                sigma_dynamic_fwd,
                rng,
                contact_static_sd=noise["contact_static_sd"],
                contact_drift_sd=noise["contact_drift_sd"],
            )
            delta_voltage_clean = np.real(v1 - v0)
        delta_voltage = _randomize_delta_voltage(
            delta_voltage_clean,
            rng,
            white_noise_rel=noise["white_noise_rel"],
            correlated_noise_rel=noise["correlated_noise_rel"],
            channel_gain_sd=noise["channel_gain_sd"],
            current_gain_sd=noise["current_gain_sd"],
        )
        vessel_geometry, vessel_delta = _vessel_record(
            model,
            inverse_result.waveform,
            phase,
            inverse_mesh,
            domain_center,
            domain_radius,
            simulator["phase_shift"],
        )
        truth.append(delta_inv)
        baselines.append(baseline_inv)
        dynamic.append(baseline_inv + delta_inv)
        voltages.append(delta_voltage)
        clean_voltages.append(delta_voltage_clean)
        labels.append(inverse_result.tissue_labels)
        points.append(inverse_points)
        phases.append(phase)
        amplitudes.append(float(inverse_result.waveform[phase]))
        records.append(
            {
                "rotation": 0.0,
                "vessels": vessel_geometry,
                "vessel_delta": vessel_delta,
                "simulator_finger_model": model.to_dict(),
                "waveform": {
                    "kind": waveform_spec.kind,
                    "frames": waveform_spec.frames,
                    "duration_s": waveform_spec.duration_s,
                    "values": inverse_result.waveform.tolist(),
                    "selected_phase": phase,
                    "selected_amplitude": float(inverse_result.waveform[phase]),
                },
                "simulation_mode": simulation_mode,
                "jacobian_bank_index": bank_index,
            }
        )
        print(f"generated simulator/PVI sample {index + 1}/{count}", flush=True)
    voltage_array = np.stack(voltages)
    newton = -(inverse_matrix @ voltage_array.T).T
    truth_array = np.stack(truth)
    baseline_array = np.stack(baselines)
    dynamic_array = np.stack(dynamic)
    if not np.allclose(dynamic_array - baseline_array, truth_array, atol=1e-7):
        raise RuntimeError("absolute minus baseline conductivity does not equal delta")
    return {
        "sigma": truth_array.astype(np.float32),
        "delta_sigma": truth_array.astype(np.float32),
        "sigma_baseline": baseline_array.astype(np.float32),
        "resting_absolute_conductivity": baseline_array.astype(np.float32),
        "sigma_dynamic": dynamic_array.astype(np.float32),
        "absolute_conductivity": dynamic_array.astype(np.float32),
        "V": voltage_array.astype(np.float32),
        "V_clean": np.stack(clean_voltages).astype(np.float32),
        "newton": newton.astype(np.float32),
        "tissue_labels": np.stack(labels).astype(np.uint8),
        "points_mm": np.stack(points).astype(np.float32),
        "phase_index": np.asarray(phases, dtype=np.int16),
        "waveform_amplitude": np.asarray(amplitudes, dtype=np.float32),
    }, records


def _write_real_pack(source_path: Path, output_path: Path, baseline: np.ndarray) -> None:
    with np.load(source_path) as source:
        arrays = {key: np.asarray(source[key]) for key in source.files}
    sample_count = len(arrays["V"])
    arrays["sigma_baseline"] = np.broadcast_to(
        baseline[None, :], (sample_count, len(baseline))
    ).copy().astype(np.float32)
    arrays["baseline_metadata_json"] = np.asarray(
        json.dumps(
            {
                "source": "unaugmented default finger simulator model",
                "role": "fixed anatomical prior, not subject-006 ground truth",
            }
        )
    )
    np.savez_compressed(output_path, **arrays)


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
        "--out-dir", type=Path, default=root / "data" / "finger_default_anatomical_exact"
    )
    parser.add_argument("--train", type=int, default=512)
    parser.add_argument("--validation", type=int, default=128)
    parser.add_argument("--test", type=int, default=32)
    parser.add_argument("--seed", type=int, default=20260716)
    parser.add_argument("--frames", type=int, default=50)
    parser.add_argument("--minimum-waveform-amplitude", type=float, default=0.05)
    parser.add_argument(
        "--training-mode", choices=["linearized", "nonlinear"], default="linearized"
    )
    parser.add_argument(
        "--test-mode", choices=["linearized", "nonlinear"], default="nonlinear"
    )
    parser.add_argument("--jacobian-bank-size", type=int, default=8)
    parser.add_argument("--finger-size-variation", type=float, default=0.08)
    parser.add_argument("--finger-rotation-deg", type=float, default=10.0)
    parser.add_argument("--artery-size-variation", type=float, default=0.20)
    parser.add_argument("--artery-position-mm", type=float, default=1.0)
    parser.add_argument("--artery-rotation-deg", type=float, default=15.0)
    parser.add_argument("--conductivity-variation", type=float, default=0.10)
    parser.add_argument("--diffusion-variation", type=float, default=0.15)
    parser.add_argument("--waveform-shape-variation", type=float, default=0.12)
    parser.add_argument("--duration-variation", type=float, default=0.08)
    parser.add_argument("--white-noise-rel", type=float, default=0.03)
    parser.add_argument("--correlated-noise-rel", type=float, default=0.02)
    parser.add_argument("--contact-static-sd", type=float, default=0.15)
    parser.add_argument("--contact-drift-sd", type=float, default=0.003)
    parser.add_argument("--channel-gain-sd", type=float, default=0.02)
    parser.add_argument("--current-gain-sd", type=float, default=0.03)
    parser.add_argument(
        "--real-source",
        type=Path,
        default=root / "data" / "subject006_gcnm_hdf" / "test.npz",
    )
    parser.add_argument(
        "--skip-real-pack",
        action="store_true",
        help="Do not create the legacy subject006 real-data pack (used for non-US120 ring training).",
    )
    parser.add_argument("--allow-overwrite", action="store_true")
    args = parser.parse_args()

    simulator = _load_simulator(args.simulator_root.resolve())
    finger_config = args.finger_config or args.simulator_root / "configs" / "default_finger.json"
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
    ring_name = args.config.stem
    inverse_mesh = simulator["load_ring_mesh"](
        Path(cfg.mesh_inv_h5), f"{ring_name}_inv"
    )
    forward_mesh = simulator["load_ring_mesh"](
        Path(cfg.mesh_fwd_h5), f"{ring_name}_fwd"
    )
    domain_center, domain_radius = domain_transform(runtime["mesh_inv"])
    default_result = simulator["simulate_points"](
        simulator["mesh_points_mm"](inverse_mesh, baseline_model),
        baseline_model,
        baseline_waveform,
    )
    default_baseline, _ = _fill_outside(
        default_result, float(baseline_model.conductivities.skin_s_m)
    )
    default_tissue_labels = np.asarray(default_result.tissue_labels, dtype=np.uint8)

    outputs = [args.out_dir / f"{name}.npz" for name in ("train", "validation", "test")]
    if not args.skip_real_pack:
        outputs.append(args.out_dir / "subject006_test_default_finger_baseline.npz")
    outputs.append(args.out_dir / "default_finger_prior.npz")
    if not args.allow_overwrite and any(path.exists() for path in outputs):
        raise FileExistsError("finger simulator dataset already exists; choose a new out-dir")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.out_dir / "default_finger_prior.npz",
        sigma_baseline=default_baseline.astype(np.float32),
        tissue_labels=default_tissue_labels,
    )
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
        "contact_static_sd": args.contact_static_sd,
        "contact_drift_sd": args.contact_drift_sd,
        "channel_gain_sd": args.channel_gain_sd,
        "current_gain_sd": args.current_gain_sd,
    }
    counts = {"train": args.train, "validation": args.validation, "test": args.test}
    seeds = {"train": args.seed, "validation": args.seed + 1, "test": args.seed + 2}
    for split, count in counts.items():
        simulation_mode = args.test_mode if split == "test" else args.training_mode
        arrays, records = _generate_split(
            count=count,
            seed=seeds[split],
            baseline_model=baseline_model,
            baseline_waveform=baseline_waveform,
            augmentation_kwargs=augmentation,
            simulator=simulator,
            inverse_mesh=inverse_mesh,
            forward_mesh=forward_mesh,
            domain_center=domain_center,
            domain_radius=domain_radius,
            physics=runtime["physics_fwd"],
            inverse_matrix=inverse_matrix,
            minimum_waveform=args.minimum_waveform_amplitude,
            noise=noise,
            simulation_mode=simulation_mode,
            jacobian_bank_size=args.jacobian_bank_size,
        )
        np.savez_compressed(args.out_dir / f"{split}.npz", **arrays)
        (args.out_dir / f"{split}_anatomy.json").write_text(
            json.dumps(records, indent=2), encoding="utf-8"
        )
    if not args.skip_real_pack:
        _write_real_pack(
            args.real_source,
            args.out_dir / "subject006_test_default_finger_baseline.npz",
            default_baseline,
        )
    metadata = {
        "schema": "finger-simulator-exact-pvi-v1",
        "target": "clean differential conductivity from the finger simulator",
        "pvi_images_used_as_labels": False,
        "simulation": {
            "train_validation": args.training_mode,
            "test": args.test_mode,
            "jacobian_bank_size": args.jacobian_bank_size,
            "anti_inverse_crime": (
                f"{len(runtime['mesh_fwd'].elements)}-element forward mesh; "
                f"{len(runtime['mesh_inv'].elements)}-element inverse target"
            ),
        },
        "simulator_root": str(args.simulator_root.resolve()),
        "finger_config": str(finger_config.resolve()),
        "finger_config_sha256": _sha256(finger_config),
        "default_finger_model": baseline_model.to_dict(),
        "waveform": {"kind": "heartbeat", "frames": args.frames, "duration_s": 1.0},
        "phase_sampling": f"amplitude-uniform, waveform >= {args.minimum_waveform_amplitude}",
        "counts": counts,
        "seeds": seeds,
        "augmentation": augmentation,
        "noise": noise,
        "mesh": {
            "ring": ring_name,
            "forward_path": str(Path(cfg.mesh_fwd_h5).resolve()),
            "inverse_path": str(Path(cfg.mesh_inv_h5).resolve()),
            "forward_elements": int(len(runtime["mesh_fwd"].elements)),
            "inverse_elements": int(len(runtime["mesh_inv"].elements)),
        },
        "outside_mesh_policy": "simulator outside points filled with skin baseline and zero delta",
        "real_baseline_pack": {
            "path": (
                str((args.out_dir / "subject006_test_default_finger_baseline.npz").resolve())
                if not args.skip_real_pack
                else None
            ),
            "warning": "default finger anatomical prior; not subject-006 ground truth",
        },
        "ring_prior": {
            "path": str((args.out_dir / "default_finger_prior.npz").resolve()),
            "arrays": ["sigma_baseline", "tissue_labels"],
        },
    }
    (args.out_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
