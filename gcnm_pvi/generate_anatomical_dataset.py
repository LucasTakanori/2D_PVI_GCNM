#!/usr/bin/env python3
"""Generate clean-truth US120 vascular EIT datasets with domain randomization."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from gcnm_pvi.anatomical_phantoms import (
    domain_transform,
    normalized_centroids,
    rasterize_anatomy,
    sample_anatomy,
)
from gcnm_pvi.config import GcnmConfig
from gcnm_pvi.gcnm_differential import production_reconstruction_matrix
from gcnm_pvi.runtime import build_runtime


def _simulate_voltage_pair(
    physics,
    sigma_baseline: np.ndarray,
    sigma_dynamic: np.ndarray,
    rng: np.random.Generator,
    *,
    contact_static_sd: float,
    contact_drift_sd: float,
) -> tuple[np.ndarray, np.ndarray]:
    electrodes = physics.mesh.elecs
    base_impedance = np.array([float(electrode.impedance) for electrode in electrodes])
    static_factor = np.exp(rng.normal(0.0, contact_static_sd, len(electrodes)))
    dynamic_factor = np.exp(rng.normal(0.0, contact_drift_sd, len(electrodes)))
    try:
        for electrode, impedance in zip(electrodes, base_impedance * static_factor):
            electrode.impedance = float(impedance)
        voltage_baseline = physics.solve(sigma_baseline)
        for electrode, impedance in zip(
            electrodes,
            base_impedance * static_factor * dynamic_factor,
        ):
            electrode.impedance = float(impedance)
        voltage_dynamic = physics.solve(sigma_dynamic)
    finally:
        for electrode, impedance in zip(electrodes, base_impedance):
            electrode.impedance = float(impedance)
    return voltage_baseline, voltage_dynamic


def _randomize_delta_voltage(
    clean_delta_voltage: np.ndarray,
    rng: np.random.Generator,
    *,
    white_noise_rel: float,
    correlated_noise_rel: float,
    channel_gain_sd: float,
    current_gain_sd: float,
) -> np.ndarray:
    voltage = np.asarray(clean_delta_voltage, dtype=np.float64)
    rms = max(float(np.sqrt(np.mean(voltage**2))), 1e-12)
    channel_gain = np.exp(rng.normal(0.0, channel_gain_sd, voltage.shape[0]))
    current_gain = float(np.exp(rng.normal(0.0, current_gain_sd)))
    white = rng.normal(0.0, white_noise_rel * rms, voltage.shape)
    mode = rng.normal(size=voltage.shape)
    mode /= max(np.linalg.norm(mode), 1e-12)
    correlated = mode * rng.normal(0.0, correlated_noise_rel * rms * np.sqrt(len(mode)))
    return current_gain * channel_gain * voltage + white + correlated


def _build_linearized_bank(
    runtime: dict,
    inv_points: np.ndarray,
    fwd_points: np.ndarray,
    rng: np.random.Generator,
    size: int,
    *,
    contact_static_sd: float,
    vessel_count: int | None,
) -> list[dict[str, np.ndarray]]:
    """Build fine-mesh anatomical Jacobians for fast anti-inverse-crime data."""
    physics = runtime["physics_fwd"]
    electrodes = physics.mesh.elecs
    base_impedance = np.array([float(electrode.impedance) for electrode in electrodes])
    bank = []
    try:
        for index in range(size):
            anatomy = sample_anatomy(rng, vessel_count=vessel_count)
            sigma0_inv, _sigma1_inv, _delta_inv = rasterize_anatomy(inv_points, anatomy)
            sigma0_fwd, _sigma1_fwd, _delta_fwd = rasterize_anatomy(fwd_points, anatomy)
            factors = np.exp(rng.normal(0.0, contact_static_sd, len(electrodes)))
            for electrode, impedance in zip(electrodes, base_impedance * factors):
                electrode.impedance = float(impedance)
            forward = physics._forward(sigma0_fwd)
            jacobian = np.asarray(forward.compute_jacobian(), dtype=np.float64)
            bank.append(
                {
                    "sigma_baseline": sigma0_inv,
                    "jacobian": jacobian,
                }
            )
            print(f"built fine-mesh Jacobian {index + 1}/{size}", flush=True)
    finally:
        for electrode, impedance in zip(electrodes, base_impedance):
            electrode.impedance = float(impedance)
    return bank


def generate_split_linearized(
    runtime: dict,
    count: int,
    seed: int,
    *,
    white_noise_rel: float,
    correlated_noise_rel: float,
    contact_static_sd: float,
    channel_gain_sd: float,
    current_gain_sd: float,
    inverse_matrix: np.ndarray,
    jacobian_bank_size: int,
    vessel_count: int | None,
) -> tuple[dict[str, np.ndarray], list[dict]]:
    """Generate many differential examples from fine-mesh anatomical Jacobians."""
    rng = np.random.default_rng(seed)
    center, radius = domain_transform(runtime["mesh_inv"])
    inv_points = normalized_centroids(runtime["mesh_inv"], center, radius)
    fwd_points = normalized_centroids(runtime["mesh_fwd"], center, radius)
    bank = _build_linearized_bank(
        runtime,
        inv_points,
        fwd_points,
        rng,
        jacobian_bank_size,
        contact_static_sd=contact_static_sd,
        vessel_count=vessel_count,
    )
    truth, baseline, voltage_clean, voltage_noisy, parameters = [], [], [], [], []
    for index in range(count):
        for _attempt in range(100):
            anatomy = sample_anatomy(rng, vessel_count=vessel_count)
            _sigma0_inv, _sigma1_inv, delta_inv = rasterize_anatomy(inv_points, anatomy)
            _sigma0_fwd, _sigma1_fwd, delta_fwd = rasterize_anatomy(fwd_points, anatomy)
            if np.count_nonzero(delta_inv) >= 6:
                break
        else:
            raise RuntimeError("could not place a vessel on at least six inverse elements")
        bank_index = int(rng.integers(0, len(bank)))
        dv_clean = bank[bank_index]["jacobian"] @ delta_fwd
        dv_noisy = _randomize_delta_voltage(
            dv_clean,
            rng,
            white_noise_rel=white_noise_rel,
            correlated_noise_rel=correlated_noise_rel,
            channel_gain_sd=channel_gain_sd,
            current_gain_sd=current_gain_sd,
        )
        truth.append(delta_inv)
        baseline.append(bank[bank_index]["sigma_baseline"])
        voltage_clean.append(dv_clean)
        voltage_noisy.append(dv_noisy)
        record = anatomy.to_dict()
        record["jacobian_bank_index"] = bank_index
        parameters.append(record)
    voltage_noisy_array = np.stack(voltage_noisy)
    newton = -(inverse_matrix @ voltage_noisy_array.T).T
    return {
        "sigma": np.stack(truth).astype(np.float32),
        "sigma_baseline": np.stack(baseline).astype(np.float32),
        "V": voltage_noisy_array.astype(np.float32),
        "V_clean": np.stack(voltage_clean).astype(np.float32),
        "newton": newton.astype(np.float32),
    }, parameters


def generate_split(
    runtime: dict,
    cfg: GcnmConfig,
    count: int,
    seed: int,
    *,
    white_noise_rel: float,
    correlated_noise_rel: float,
    contact_static_sd: float,
    contact_drift_sd: float,
    channel_gain_sd: float,
    current_gain_sd: float,
    inverse_matrix: np.ndarray,
    vessel_count: int | None,
) -> tuple[dict[str, np.ndarray], list[dict]]:
    rng = np.random.default_rng(seed)
    center, radius = domain_transform(runtime["mesh_inv"])
    inv_points = normalized_centroids(runtime["mesh_inv"], center, radius)
    fwd_points = normalized_centroids(runtime["mesh_fwd"], center, radius)

    truth: list[np.ndarray] = []
    baseline: list[np.ndarray] = []
    voltage_clean: list[np.ndarray] = []
    voltage_noisy: list[np.ndarray] = []
    parameters: list[dict] = []

    for index in range(count):
        for _attempt in range(100):
            anatomy = sample_anatomy(rng, vessel_count=vessel_count)
            sigma0_inv, _sigma1_inv, delta_inv = rasterize_anatomy(inv_points, anatomy)
            if np.count_nonzero(delta_inv) >= 6:
                break
        else:
            raise RuntimeError("could not place a vessel on at least six inverse elements")
        sigma0_fwd, sigma1_fwd, _delta_fwd = rasterize_anatomy(fwd_points, anatomy)
        v0, v1 = _simulate_voltage_pair(
            runtime["physics_fwd"],
            sigma0_fwd,
            sigma1_fwd,
            rng,
            contact_static_sd=contact_static_sd,
            contact_drift_sd=contact_drift_sd,
        )
        dv_clean = np.real(v1 - v0)
        dv_noisy = _randomize_delta_voltage(
            dv_clean,
            rng,
            white_noise_rel=white_noise_rel,
            correlated_noise_rel=correlated_noise_rel,
            channel_gain_sd=channel_gain_sd,
            current_gain_sd=current_gain_sd,
        )
        truth.append(delta_inv)
        baseline.append(sigma0_inv)
        voltage_clean.append(dv_clean)
        voltage_noisy.append(dv_noisy)
        parameters.append(anatomy.to_dict())
        print(f"generated {index + 1}/{count}", flush=True)

    voltage_noisy_array = np.stack(voltage_noisy)
    # The MATLAB display path returns ``-D1 @ dV``.  The FEM Jacobian in the
    # Python solver is the actual derivative dV/dsigma, so supervised physical
    # conductivity training uses the Gauss--Newton direction ``+D1 @ dV``.
    newton = -(inverse_matrix @ voltage_noisy_array.T).T
    return {
        "sigma": np.stack(truth).astype(np.float32),
        "sigma_baseline": np.stack(baseline).astype(np.float32),
        "V": voltage_noisy_array.astype(np.float32),
        "V_clean": np.stack(voltage_clean).astype(np.float32),
        "newton": newton.astype(np.float32),
    }, parameters


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=root / "configs" / "subject006_pvi08_production.yaml")
    parser.add_argument("--out-dir", type=Path, default=root / "data" / "subject006_anatomical")
    parser.add_argument("--train", type=int, default=64)
    parser.add_argument("--validation", type=int, default=16)
    parser.add_argument("--test", type=int, default=16)
    parser.add_argument("--seed", type=int, default=20260712)
    parser.add_argument("--white-noise-rel", type=float, default=0.03)
    parser.add_argument("--correlated-noise-rel", type=float, default=0.02)
    parser.add_argument("--contact-static-sd", type=float, default=0.15)
    parser.add_argument("--contact-drift-sd", type=float, default=0.003)
    parser.add_argument("--channel-gain-sd", type=float, default=0.02)
    parser.add_argument("--current-gain-sd", type=float, default=0.03)
    parser.add_argument(
        "--simulation-mode",
        choices=["linearized", "nonlinear"],
        default="linearized",
        help="Fast fine-Jacobian training data or exact two-solve nonlinear data",
    )
    parser.add_argument("--jacobian-bank-size", type=int, default=3)
    parser.add_argument(
        "--vessel-count",
        choices=["mixed", "1", "2"],
        default="mixed",
        help="Force every anatomy to contain one or two vessels, or preserve the mixed distribution",
    )
    args = parser.parse_args()

    vessel_count = None if args.vessel_count == "mixed" else int(args.vessel_count)

    cfg = GcnmConfig.from_yaml(args.config)
    runtime = build_runtime(cfg, include_forward=True)
    inverse_matrix = production_reconstruction_matrix(
        runtime["physics_inv"],
        hyper_pvi=cfg.hyper_pvi,
        regularizer=runtime["mappings"].laplace,
    )
    args.out_dir.mkdir(parents=True, exist_ok=True)
    counts = {"train": args.train, "validation": args.validation, "test": args.test}
    seeds = {"train": args.seed, "validation": args.seed + 1, "test": args.seed + 2}
    parameter_files = {}
    for split, count in counts.items():
        if args.simulation_mode == "linearized":
            arrays, parameters = generate_split_linearized(
                runtime,
                count,
                seeds[split],
                white_noise_rel=args.white_noise_rel,
                correlated_noise_rel=args.correlated_noise_rel,
                contact_static_sd=args.contact_static_sd,
                channel_gain_sd=args.channel_gain_sd,
                current_gain_sd=args.current_gain_sd,
                inverse_matrix=inverse_matrix,
                jacobian_bank_size=args.jacobian_bank_size,
                vessel_count=vessel_count,
            )
        else:
            arrays, parameters = generate_split(
                runtime,
                cfg,
                count,
                seeds[split],
                white_noise_rel=args.white_noise_rel,
                correlated_noise_rel=args.correlated_noise_rel,
                contact_static_sd=args.contact_static_sd,
                contact_drift_sd=args.contact_drift_sd,
                channel_gain_sd=args.channel_gain_sd,
                current_gain_sd=args.current_gain_sd,
                inverse_matrix=inverse_matrix,
                vessel_count=vessel_count,
            )
        np.savez_compressed(args.out_dir / f"{split}.npz", **arrays)
        parameter_path = args.out_dir / f"{split}_anatomy.json"
        parameter_path.write_text(json.dumps(parameters, indent=2), encoding="utf-8")
        parameter_files[split] = str(parameter_path.resolve())

    metadata = {
        "target": "clean FEM element conductivity change (not PVI reconstruction)",
        "mesh": "subject006 US120",
        "counts": counts,
        "seeds": seeds,
        "simulation_mode": args.simulation_mode,
        "jacobian_bank_size": args.jacobian_bank_size,
        "vessel_count": args.vessel_count,
        "noise": {
            "white_noise_rel": args.white_noise_rel,
            "correlated_noise_rel": args.correlated_noise_rel,
            "contact_static_sd": args.contact_static_sd,
            "contact_drift_sd": args.contact_drift_sd,
            "channel_gain_sd": args.channel_gain_sd,
            "current_gain_sd": args.current_gain_sd,
        },
        "arrays": {
            "sigma": "clean inverse-mesh vascular delta conductivity",
            "sigma_baseline": "randomized static skin/fat/muscle/bone/blood conductivity",
            "V": "noisy differential 32-channel voltage",
            "V_clean": "clean differential voltage before acquisition perturbations",
            "newton": "noisy physical Gauss-Newton conductivity feature (+D1 dV); never used as target",
        },
        "anatomy_parameter_files": parameter_files,
    }
    (args.out_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
