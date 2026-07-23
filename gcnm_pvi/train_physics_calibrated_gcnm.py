#!/usr/bin/env python3
"""Train a two-stage core-guided GCNM with real voltage self-supervision."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.loader import DataLoader

from gcnm_pvi.anatomical_phantoms import element_positions
from gcnm_pvi.config import GcnmConfig
from gcnm_pvi.core_guided_physics import (
    FixedLinearOperator,
    build_fixed_linear_operator,
    normalized_context_direction,
)
from gcnm_pvi.generate_multisubject_beat_dataset import _rank_one_template
from gcnm_pvi.physics_calibrated_model import (
    PhysicsCalibratedCoreStage,
    calibrate_shape_batch,
)
from gcnm_pvi.physics_calibrated_physics import (
    StagePhysicsBatch,
    nonlinear_stage_physics,
    stage_physics_summary,
)
from gcnm_pvi.runtime import build_runtime
from gcnm_pvi.train_core_guided_gcnm import (
    _anatomy_targets,
    _auxiliary_losses,
    _correlation_loss,
    _graphs,
    _load_arrays,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _trial_templates(
    measured: np.ndarray,
    period_id: np.ndarray,
    trial_id: np.ndarray,
) -> np.ndarray:
    """Build one label-free voltage-direction template per acquisition trial."""

    period_templates = {}
    for period in np.unique(period_id):
        selected = np.flatnonzero(period_id == period)
        period_templates[int(period)] = _rank_one_template(measured[selected])
    templates = np.zeros_like(measured, dtype=np.float64)
    for trial in np.unique(trial_id):
        selected = np.flatnonzero(trial_id == trial)
        periods = np.unique(period_id[selected])
        stack = np.stack([period_templates[int(period)] for period in periods])
        norms = np.linalg.norm(stack, axis=1)
        normalized = stack / np.maximum(norms[:, None], 1e-12)
        _u, _s, vh = np.linalg.svd(normalized, full_matrices=False)
        templates[selected] = vh[0] * max(float(np.median(norms)), 1e-12)
    return templates


def _even_indices(samples: int, limit: int | None) -> np.ndarray:
    if limit is None or limit >= samples:
        return np.arange(samples)
    if limit <= 0:
        raise ValueError("sample limit must be positive")
    return np.unique(np.linspace(0, samples - 1, limit, dtype=int))


def _load_real_voltage(path: Path, limit: int | None) -> dict[str, np.ndarray]:
    with np.load(path) as source:
        required = ("V", "period_id", "trial_index")
        missing = [key for key in required if key not in source]
        if missing:
            raise KeyError(f"{path} is missing {missing}")
        saved = np.asarray(source["V"], dtype=np.float64)
        period = np.asarray(source["period_id"])
        trial = np.asarray(source["trial_index"])
    # The archived PVI pack uses the MATLAB display sign.  Physical inversion
    # follows the opposite sign, as in every existing real evaluator.
    measured = -saved
    templates = _trial_templates(measured, period, trial)
    selected = _even_indices(len(measured), limit)
    return {
        "V": measured[selected],
        "V_template": templates[selected],
        "period_id": period[selected],
        "trial_index": trial[selected],
        "source_indices": selected,
    }


def _match_synthetic_amplitudes(
    arrays: dict[str, np.ndarray],
    real_voltage: np.ndarray,
    *,
    seed: int,
    minimum_factor: float,
    maximum_factor: float,
) -> tuple[dict[str, np.ndarray], np.ndarray]:
    """Match synthetic voltage RMS to the observed real training distribution."""

    result = {key: np.asarray(value).copy() for key, value in arrays.items()}
    synthetic_rms = np.sqrt(np.mean(np.asarray(result["V"], dtype=float) ** 2, axis=1))
    real_rms = np.sqrt(np.mean(np.asarray(real_voltage, dtype=float) ** 2, axis=1))
    rng = np.random.default_rng(seed)
    target_rms = rng.choice(real_rms, size=len(synthetic_rms), replace=True)
    factors = np.clip(
        target_rms / np.maximum(synthetic_rms, 1e-12),
        float(minimum_factor),
        float(maximum_factor),
    )
    for key in ("sigma", "V", "V_clean", "V_template", "newton"):
        if key in result:
            shape = (len(factors),) + (1,) * (result[key].ndim - 1)
            result[key] = result[key] * factors.reshape(shape)
    return result, factors


def _fixed_stage_physics(
    fixed: FixedLinearOperator,
    current: np.ndarray,
    measured: np.ndarray,
) -> StagePhysicsBatch:
    current = np.asarray(current, dtype=np.float64)
    measured = np.asarray(measured, dtype=np.float64)
    target = measured - current @ fixed.jacobian.T
    directions = target @ fixed.inverse_jacobian.T
    samples = len(current)
    return StagePhysicsBatch(
        directions=directions.astype(np.float32),
        jacobians=np.broadcast_to(
            fixed.jacobian.astype(np.float32)[None, :, :],
            (samples,) + fixed.jacobian.shape,
        ),
        voltage_targets=target.astype(np.float32),
        baseline_voltages=np.broadcast_to(
            fixed.baseline_voltage.astype(np.float32)[None, :],
            measured.shape,
        ),
        current_voltage_residual_rms=np.sqrt(np.mean(target**2, axis=1)),
        step_rms=np.sqrt(np.mean(directions**2, axis=1)),
        clipped_elements=0,
    )


def _stage_graphs(
    *,
    truth: np.ndarray,
    current: np.ndarray,
    physics: StagePhysicsBatch,
    context: np.ndarray,
    measured: np.ndarray,
    tissue_labels: np.ndarray | None,
    vessel_parameters: np.ndarray | None,
    positions: np.ndarray,
    edge_index,
    conductivity_scale: float,
    voltage_scale: float,
    supervised: bool,
):
    graphs = _graphs(
        truth,
        current,
        physics.directions,
        context,
        measured,
        tissue_labels,
        vessel_parameters,
        positions,
        edge_index,
        conductivity_scale=conductivity_scale,
        voltage_scale=voltage_scale,
    )
    for index, graph in enumerate(graphs):
        graph.jacobian = torch.tensor(
            np.asarray(physics.jacobians[index])[None, :, :],
            dtype=torch.float32,
        )
        graph.voltage_target = torch.tensor(
            physics.voltage_targets[index][None, :], dtype=torch.float32
        )
        graph.current_physical = torch.tensor(
            current[index][None, :], dtype=torch.float32
        )
        graph.is_supervised = torch.tensor([supervised], dtype=torch.bool)
    return graphs


def _calibrated_loss(
    model,
    batch,
    *,
    conductivity_scale: float,
    maximum_amplitude: float,
    weights: dict[str, float],
    supervised: bool,
) -> tuple[torch.Tensor, dict[str, float]]:
    raw, core_logits, attention = model(batch, return_aux=True)
    calibrated = calibrate_shape_batch(
        raw,
        batch,
        conductivity_scale=conductivity_scale,
        maximum_amplitude=maximum_amplitude,
    )
    prediction = calibrated.prediction / float(conductivity_scale)
    voltage_loss = torch.mean(calibrated.relative_mse)
    correction = torch.mean(
        (calibrated.correction / float(conductivity_scale)) ** 2
    )
    if not supervised:
        total = voltage_loss + float(weights["real_correction"]) * correction
        return total, {
            "voltage": float(voltage_loss.detach()),
            "correction": float(correction.detach()),
            "amplitude": float(torch.mean(calibrated.amplitude).detach()),
            "residual_ratio": float(
                torch.mean(
                    calibrated.residual_rms
                    / calibrated.target_rms.clamp_min(1e-8)
                ).detach()
            ),
        }

    core = batch.core_target > 0.5
    muscle = batch.muscle_target > 0.5
    region_weight = (
        1.0
        + float(weights["core_map"]) * core.to(prediction.dtype)
        + float(weights["muscle_map"]) * muscle.to(prediction.dtype)
    )
    map_loss = torch.mean(
        region_weight
        * F.smooth_l1_loss(prediction, batch.y, beta=0.05, reduction="none")
    )
    artifact_mask = (batch.outside_target > 0.5) & (torch.abs(batch.y) < 1e-5)
    artifact = prediction.new_tensor(0.0)
    if torch.any(artifact_mask):
        errors = prediction[artifact_mask] ** 2
        count = max(1, int(math.ceil(0.05 * errors.numel())))
        artifact = torch.mean(torch.topk(errors, count).values)
    auxiliary = _auxiliary_losses(
        prediction, core_logits, attention, batch
    )
    correlation = _correlation_loss(prediction, batch)
    total = (
        map_loss
        + float(weights["synthetic_voltage"]) * voltage_loss
        + float(weights["artifact"]) * artifact
        + float(weights["core_segmentation"]) * auxiliary["core_segmentation"]
        + float(weights["slot_localization"]) * auxiliary["slot_localization"]
        + float(weights["centroid"]) * auxiliary["centroid"]
        + float(weights["attention_overlap"]) * auxiliary["attention_overlap"]
        + float(weights["core_amplitude"]) * auxiliary["core_amplitude"]
        + float(weights["mass"]) * auxiliary["mass"]
        + float(weights["correlation"]) * correlation
        + float(weights["correction"]) * correction
    )
    return total, {
        "map": float(map_loss.detach()),
        "voltage": float(voltage_loss.detach()),
        "artifact": float(artifact.detach()),
        "correlation": float(correlation.detach()),
        "correction": float(correction.detach()),
        "amplitude": float(torch.mean(calibrated.amplitude).detach()),
        **{key: float(value.detach()) for key, value in auxiliary.items()},
    }


def _mean_loader_loss(
    model,
    loader,
    *,
    conductivity_scale: float,
    maximum_amplitude: float,
    weights,
    supervised: bool,
) -> tuple[float, dict[str, float]]:
    device = next(model.parameters()).device
    total, graphs = 0.0, 0
    component_total: dict[str, float] = {}
    for batch in loader:
        batch = batch.to(device)
        loss, components = _calibrated_loss(
            model,
            batch,
            conductivity_scale=conductivity_scale,
            maximum_amplitude=maximum_amplitude,
            weights=weights,
            supervised=supervised,
        )
        count = int(batch.num_graphs)
        total += float(loss.detach()) * count
        graphs += count
        for key, value in components.items():
            component_total[key] = component_total.get(key, 0.0) + value * count
    return total / max(graphs, 1), {
        key: value / max(graphs, 1) for key, value in component_total.items()
    }


def _fit_mixed(
    model,
    synthetic_train,
    real_train,
    synthetic_validation,
    real_validation,
    cfg,
    *,
    epochs: int,
    conductivity_scale: float,
    maximum_amplitude: float,
    real_voltage_weight: float,
    weights: dict[str, float],
    label: str,
):
    synthetic_loader = DataLoader(
        synthetic_train, batch_size=cfg.batch_size, shuffle=True
    )
    real_loader = DataLoader(real_train, batch_size=cfg.batch_size, shuffle=True)
    synthetic_validation_loader = DataLoader(
        synthetic_validation, batch_size=cfg.batch_size, shuffle=False
    )
    real_validation_loader = DataLoader(
        real_validation, batch_size=cfg.batch_size, shuffle=False
    )
    optimizer = torch.optim.Adam(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=cfg.learning_rate,
    )
    device = next(model.parameters()).device
    best_state = copy.deepcopy(model.state_dict())
    best_loss, best_epoch, patience = float("inf"), -1, int(cfg.patience)
    history = []
    for epoch in range(int(epochs)):
        model.train(True)
        real_iterator = iter(real_loader)
        train_total, steps = 0.0, 0
        for synthetic_batch in synthetic_loader:
            try:
                real_batch = next(real_iterator)
            except StopIteration:
                real_iterator = iter(real_loader)
                real_batch = next(real_iterator)
            synthetic_batch = synthetic_batch.to(device)
            real_batch = real_batch.to(device)
            optimizer.zero_grad()
            synthetic_loss, _ = _calibrated_loss(
                model,
                synthetic_batch,
                conductivity_scale=conductivity_scale,
                maximum_amplitude=maximum_amplitude,
                weights=weights,
                supervised=True,
            )
            real_loss, _ = _calibrated_loss(
                model,
                real_batch,
                conductivity_scale=conductivity_scale,
                maximum_amplitude=maximum_amplitude,
                weights=weights,
                supervised=False,
            )
            loss = synthetic_loss + float(real_voltage_weight) * real_loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad], max_norm=5.0
            )
            optimizer.step()
            train_total += float(loss.detach())
            steps += 1

        model.train(False)
        with torch.no_grad():
            synthetic_loss, synthetic_components = _mean_loader_loss(
                model,
                synthetic_validation_loader,
                conductivity_scale=conductivity_scale,
                maximum_amplitude=maximum_amplitude,
                weights=weights,
                supervised=True,
            )
            real_loss, real_components = _mean_loader_loss(
                model,
                real_validation_loader,
                conductivity_scale=conductivity_scale,
                maximum_amplitude=maximum_amplitude,
                weights=weights,
                supervised=False,
            )
        validation_loss = synthetic_loss + float(real_voltage_weight) * real_loss
        history.append(
            {
                "epoch": epoch,
                "train_combined_loss": train_total / max(steps, 1),
                "validation_combined_loss": validation_loss,
                "validation_synthetic_loss": synthetic_loss,
                "validation_real_voltage_loss": real_loss,
                "validation_synthetic_components": synthetic_components,
                "validation_real_components": real_components,
            }
        )
        print(
            f"{label} epoch {epoch:03d} train={train_total / max(steps, 1):.6f} "
            f"validation={validation_loss:.6f} synthetic={synthetic_loss:.6f} "
            f"real_voltage={real_loss:.6f}",
            flush=True,
        )
        if validation_loss <= best_loss:
            best_loss, best_epoch = validation_loss, epoch
            best_state = copy.deepcopy(model.state_dict())
            patience = int(cfg.patience)
        else:
            patience -= 1
            if patience <= 0:
                break
    model.load_state_dict(best_state)
    return model, history, best_epoch, best_loss


def _predict_calibrated(
    model,
    graphs,
    *,
    conductivity_scale: float,
    maximum_amplitude: float,
    batch_size: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    loader = DataLoader(graphs, batch_size=batch_size, shuffle=False)
    device = next(model.parameters()).device
    maps, amplitudes, ratios, centers, probabilities = [], [], [], [], []
    model.eval()
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            raw, core_logits, attention = model(batch, return_aux=True)
            calibrated = calibrate_shape_batch(
                raw,
                batch,
                conductivity_scale=conductivity_scale,
                maximum_amplitude=maximum_amplitude,
            )
            maps.append(calibrated.prediction.cpu().numpy())
            amplitudes.append(calibrated.amplitude.cpu().numpy())
            ratios.append(
                (
                    calibrated.residual_rms
                    / calibrated.target_rms.clamp_min(1e-8)
                ).cpu().numpy()
            )
            probabilities.append(torch.sigmoid(core_logits).cpu().numpy())
            coordinates = batch.x[:, 3:5]
            for start, stop in zip(batch.ptr[:-1], batch.ptr[1:]):
                local_attention = attention[start:stop]
                local_coordinates = coordinates[start:stop]
                centers.append(
                    torch.stack(
                        [
                            torch.sum(
                                local_attention[:, slot : slot + 1]
                                * local_coordinates,
                                dim=0,
                            )
                            for slot in range(2)
                        ]
                    ).cpu().numpy()
                )
    nodes = graphs[0].num_nodes
    return (
        np.concatenate(maps).reshape(-1, nodes),
        np.concatenate(amplitudes),
        np.concatenate(ratios),
        np.stack(centers),
        np.concatenate(probabilities).reshape(-1, nodes),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--validation", type=Path, required=True)
    parser.add_argument("--train-anatomy", type=Path, required=True)
    parser.add_argument("--validation-anatomy", type=Path, required=True)
    parser.add_argument("--real-train", type=Path, required=True)
    parser.add_argument("--real-validation", type=Path, required=True)
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--models-dir", type=Path, required=True)
    parser.add_argument("--results-dir", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--max-train", type=int, default=None)
    parser.add_argument("--max-validation", type=int, default=None)
    parser.add_argument("--max-real-train", type=int, default=None)
    parser.add_argument("--max-real-validation", type=int, default=None)
    parser.add_argument("--baseline-conductivity", type=float, default=0.7)
    parser.add_argument("--maximum-amplitude", type=float, default=0.25)
    parser.add_argument("--amplitude-factor-min", type=float, default=0.25)
    parser.add_argument("--amplitude-factor-max", type=float, default=6.0)
    parser.add_argument("--real-voltage-weight", type=float, default=0.25)
    parser.add_argument("--synthetic-voltage-weight", type=float, default=0.05)
    parser.add_argument("--core-map-weight", type=float, default=4.0)
    parser.add_argument("--muscle-map-weight", type=float, default=1.0)
    parser.add_argument("--artifact-weight", type=float, default=0.25)
    parser.add_argument("--core-segmentation-weight", type=float, default=0.05)
    parser.add_argument("--slot-localization-weight", type=float, default=0.05)
    parser.add_argument("--centroid-weight", type=float, default=0.20)
    parser.add_argument("--attention-overlap-weight", type=float, default=0.02)
    parser.add_argument("--core-amplitude-weight", type=float, default=0.20)
    parser.add_argument("--mass-weight", type=float, default=0.02)
    parser.add_argument("--correlation-weight", type=float, default=0.05)
    parser.add_argument("--real-correction-weight", type=float, default=0.001)
    parser.add_argument("--stage2-correction-weight", type=float, default=0.02)
    parser.add_argument("--stage1-limit", type=float, default=1.0)
    parser.add_argument("--stage2-limit", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--allow-overwrite", action="store_true")
    args = parser.parse_args()

    args.models_dir.mkdir(parents=True, exist_ok=True)
    args.results_dir.mkdir(parents=True, exist_ok=True)
    stage1_path = args.models_dir / f"{args.model_name}_stage1.pt"
    stage2_path = args.models_dir / f"{args.model_name}_stage2.pt"
    report_path = args.results_dir / f"{args.model_name}_training_report.json"
    if not args.allow_overwrite and any(
        path.exists() for path in (stage1_path, stage2_path, report_path)
    ):
        raise FileExistsError("experiment output exists; choose a new name")

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    cfg = GcnmConfig.from_yaml(args.config)
    runtime = build_runtime(cfg, include_forward=False)
    synthetic_train_original = _load_arrays(args.train, args.max_train)
    synthetic_validation = _load_arrays(args.validation, args.max_validation)
    real_train = _load_real_voltage(args.real_train, args.max_real_train)
    real_validation = _load_real_voltage(
        args.real_validation, args.max_real_validation
    )
    magnitude = np.abs(
        np.asarray(synthetic_train_original["sigma"], dtype=np.float64)
    )
    nonzero = magnitude[magnitude > 1e-8]
    conductivity_scale = max(float(np.percentile(nonzero, 99.5)), 1e-6)
    voltage_scale = max(
        float(np.percentile(np.abs(synthetic_train_original["V"]), 99.5)),
        1e-12,
    )
    synthetic_train, amplitude_factors = _match_synthetic_amplitudes(
        synthetic_train_original,
        real_train["V"],
        seed=args.seed + 41,
        minimum_factor=args.amplitude_factor_min,
        maximum_factor=args.amplitude_factor_max,
    )
    train_parameters = _anatomy_targets(
        args.train_anatomy,
        conductivity_scale,
        args.max_train,
        include_diffusion=False,
    )
    validation_parameters = _anatomy_targets(
        args.validation_anatomy,
        conductivity_scale,
        args.max_validation,
        include_diffusion=False,
    )
    positions = element_positions(runtime["mesh_inv"]).astype(np.float32)
    elements = synthetic_train["sigma"].shape[1]
    baseline_vector = np.full(elements, args.baseline_conductivity, dtype=float)
    fixed = build_fixed_linear_operator(
        runtime["physics_inv"],
        baseline_vector,
        regularizer=runtime["mappings"].laplace,
        hyper_pvi=cfg.hyper_pvi,
        lambda_lm=cfg.lambda_lm,
    )

    def stage1_graphs(arrays, parameters, supervised):
        samples = len(arrays["V"])
        zero = np.zeros((samples, elements), dtype=np.float64)
        physics = _fixed_stage_physics(fixed, zero, arrays["V"])
        context = normalized_context_direction(
            fixed.directions(arrays["V_template"])
        )
        truth = arrays["sigma"] if supervised else zero
        return _stage_graphs(
            truth=truth,
            current=zero,
            physics=physics,
            context=context,
            measured=arrays["V"],
            tissue_labels=arrays.get("tissue_labels"),
            vessel_parameters=parameters,
            positions=positions,
            edge_index=runtime["edge_index"],
            conductivity_scale=conductivity_scale,
            voltage_scale=voltage_scale,
            supervised=supervised,
        ), context

    synthetic_graphs_1, synthetic_context_train = stage1_graphs(
        synthetic_train, train_parameters, True
    )
    synthetic_validation_graphs_1, synthetic_context_validation = stage1_graphs(
        synthetic_validation, validation_parameters, True
    )
    real_graphs_1, real_context_train = stage1_graphs(real_train, None, False)
    real_validation_graphs_1, real_context_validation = stage1_graphs(
        real_validation, None, False
    )
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    weights = {
        "core_map": args.core_map_weight,
        "muscle_map": args.muscle_map_weight,
        "artifact": args.artifact_weight,
        "core_segmentation": args.core_segmentation_weight,
        "slot_localization": args.slot_localization_weight,
        "centroid": args.centroid_weight,
        "attention_overlap": args.attention_overlap_weight,
        "core_amplitude": args.core_amplitude_weight,
        "mass": args.mass_weight,
        "correlation": args.correlation_weight,
        "synthetic_voltage": args.synthetic_voltage_weight,
        "real_correction": args.real_correction_weight,
        "correction": 0.0,
    }
    started = time.time()
    stage1 = PhysicsCalibratedCoreStage(
        correction_limit=args.stage1_limit
    ).to(device)
    stage1, history_1, best_epoch_1, best_loss_1 = _fit_mixed(
        stage1,
        synthetic_graphs_1,
        real_graphs_1,
        synthetic_validation_graphs_1,
        real_validation_graphs_1,
        cfg,
        epochs=args.epochs,
        conductivity_scale=conductivity_scale,
        maximum_amplitude=args.maximum_amplitude,
        real_voltage_weight=args.real_voltage_weight,
        weights=weights,
        label="stage 1 calibrated core map",
    )
    synthetic_current, synthetic_amp_1, synthetic_ratio_1, _, _ = (
        _predict_calibrated(
            stage1,
            synthetic_graphs_1,
            conductivity_scale=conductivity_scale,
            maximum_amplitude=args.maximum_amplitude,
            batch_size=cfg.batch_size,
        )
    )
    synthetic_validation_current, synthetic_validation_amp_1, _, _, _ = (
        _predict_calibrated(
            stage1,
            synthetic_validation_graphs_1,
            conductivity_scale=conductivity_scale,
            maximum_amplitude=args.maximum_amplitude,
            batch_size=cfg.batch_size,
        )
    )
    real_current, real_amp_1, real_ratio_1, _, _ = _predict_calibrated(
        stage1,
        real_graphs_1,
        conductivity_scale=conductivity_scale,
        maximum_amplitude=args.maximum_amplitude,
        batch_size=cfg.batch_size,
    )
    real_validation_current, real_validation_amp_1, real_validation_ratio_1, _, _ = (
        _predict_calibrated(
            stage1,
            real_validation_graphs_1,
            conductivity_scale=conductivity_scale,
            maximum_amplitude=args.maximum_amplitude,
            batch_size=cfg.batch_size,
        )
    )
    del synthetic_graphs_1, synthetic_validation_graphs_1
    del real_graphs_1, real_validation_graphs_1
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    def stage2_graphs(arrays, current, context, parameters, supervised, label):
        baseline = np.full_like(current, args.baseline_conductivity)
        baseline_voltages = np.broadcast_to(
            fixed.baseline_voltage[None, :], arrays["V"].shape
        )
        physics = nonlinear_stage_physics(
            runtime["physics_inv"],
            baseline,
            current,
            arrays["V"],
            regularizer=runtime["mappings"].laplace,
            hyper_pvi=cfg.hyper_pvi,
            lambda_lm=cfg.lambda_lm,
            step_size=cfg.lm_step_size,
            baseline_voltages=baseline_voltages,
            progress_label=label,
        )
        truth = arrays["sigma"] if supervised else np.zeros_like(current)
        graphs = _stage_graphs(
            truth=truth,
            current=current,
            physics=physics,
            context=context,
            measured=arrays["V"],
            tissue_labels=arrays.get("tissue_labels"),
            vessel_parameters=parameters,
            positions=positions,
            edge_index=runtime["edge_index"],
            conductivity_scale=conductivity_scale,
            voltage_scale=voltage_scale,
            supervised=supervised,
        )
        return graphs, stage_physics_summary(physics)

    synthetic_graphs_2, synthetic_physics_2 = stage2_graphs(
        synthetic_train,
        synthetic_current,
        synthetic_context_train,
        train_parameters,
        True,
        "calibrated stage 2 synthetic train",
    )
    synthetic_validation_graphs_2, synthetic_validation_physics_2 = stage2_graphs(
        synthetic_validation,
        synthetic_validation_current,
        synthetic_context_validation,
        validation_parameters,
        True,
        "calibrated stage 2 synthetic validation",
    )
    real_graphs_2, real_physics_2 = stage2_graphs(
        real_train,
        real_current,
        real_context_train,
        None,
        False,
        "calibrated stage 2 real train",
    )
    real_validation_graphs_2, real_validation_physics_2 = stage2_graphs(
        real_validation,
        real_validation_current,
        real_context_validation,
        None,
        False,
        "calibrated stage 2 real validation",
    )
    stage2 = PhysicsCalibratedCoreStage(
        correction_limit=args.stage2_limit
    )
    stage2.copy_context_from(stage1.cpu())
    stage2.freeze_context()
    stage2.to(device)
    stage2_weights = dict(weights)
    stage2_weights["correction"] = args.stage2_correction_weight
    stage2, history_2, best_epoch_2, best_loss_2 = _fit_mixed(
        stage2,
        synthetic_graphs_2,
        real_graphs_2,
        synthetic_validation_graphs_2,
        real_validation_graphs_2,
        cfg,
        epochs=args.epochs,
        conductivity_scale=conductivity_scale,
        maximum_amplitude=args.maximum_amplitude,
        real_voltage_weight=args.real_voltage_weight,
        weights=stage2_weights,
        label="stage 2 calibrated residual map",
    )
    _, synthetic_amp_2, synthetic_ratio_2, _, _ = _predict_calibrated(
        stage2,
        synthetic_graphs_2,
        conductivity_scale=conductivity_scale,
        maximum_amplitude=args.maximum_amplitude,
        batch_size=cfg.batch_size,
    )
    _, real_amp_2, real_ratio_2, _, _ = _predict_calibrated(
        stage2,
        real_graphs_2,
        conductivity_scale=conductivity_scale,
        maximum_amplitude=args.maximum_amplitude,
        batch_size=cfg.batch_size,
    )

    contract = {
        "model_name": args.model_name,
        "architecture": "physics_calibrated_core_residual_two_stage_v1",
        "measurements": int(synthetic_train["V"].shape[1]),
        "elements": int(elements),
        "node_features": 6,
        "conductivity_scale": conductivity_scale,
        "voltage_scale": voltage_scale,
        "baseline_mode": "homogeneous",
        "baseline_conductivity": args.baseline_conductivity,
        "stage1_limit": args.stage1_limit,
        "stage2_limit": args.stage2_limit,
        "maximum_amplitude_s_m": args.maximum_amplitude,
        "amplitude_calibration": (
            "unit-RMS direction-anchored shape with differentiable linear "
            "least-squares voltage amplitude"
        ),
        "real_supervision": (
            "differential voltage residual only; no PVI images or real "
            "conductivity labels"
        ),
        "heldout_policy": "trials 10-11 are evaluation-only and never loaded",
        "config_sha256": _sha256(args.config),
        "mesh_inverse_sha256": _sha256(Path(cfg.mesh_inv_h5)),
        "mappings_sha256": _sha256(Path(cfg.mappings_h5)),
        "synthetic_train_sha256": _sha256(args.train),
        "synthetic_validation_sha256": _sha256(args.validation),
        "real_train_sha256": _sha256(args.real_train),
        "real_validation_sha256": _sha256(args.real_validation),
    }
    torch.save(
        {
            "model_kind": "physics_calibrated_core_stage1",
            "state_dict": stage1.cpu().state_dict(),
            "contract": contract,
        },
        stage1_path,
    )
    torch.save(
        {
            "model_kind": "physics_calibrated_core_stage2",
            "state_dict": stage2.cpu().state_dict(),
            "contract": contract,
        },
        stage2_path,
    )
    report = {
        "method": "two-stage physics-calibrated core-guided residual GCNM",
        "target": "clean synthetic delta conductivity plus unlabeled real voltage",
        "pvi_images_used_as_labels": False,
        "contract": contract,
        "synthetic_train_samples": len(synthetic_train["V"]),
        "synthetic_validation_samples": len(synthetic_validation["V"]),
        "real_train_samples": len(real_train["V"]),
        "real_validation_samples": len(real_validation["V"]),
        "real_train_trials": sorted(
            int(value) for value in np.unique(real_train["trial_index"])
        ),
        "real_validation_trials": sorted(
            int(value) for value in np.unique(real_validation["trial_index"])
        ),
        "synthetic_amplitude_matching": {
            "minimum_factor": args.amplitude_factor_min,
            "maximum_factor": args.amplitude_factor_max,
            "mean_factor": float(np.mean(amplitude_factors)),
            "median_factor": float(np.median(amplitude_factors)),
            "fraction_at_minimum": float(
                np.mean(amplitude_factors == args.amplitude_factor_min)
            ),
            "fraction_at_maximum": float(
                np.mean(amplitude_factors == args.amplitude_factor_max)
            ),
        },
        "loss_stage_1": weights,
        "loss_stage_2": stage2_weights,
        "real_voltage_weight": args.real_voltage_weight,
        "stage_1": {
            "best_epoch": best_epoch_1,
            "best_validation_loss": best_loss_1,
            "synthetic_amplitude_mean_s_m": float(np.mean(synthetic_amp_1)),
            "synthetic_linear_residual_ratio_mean": float(
                np.mean(synthetic_ratio_1)
            ),
            "real_amplitude_mean_s_m": float(np.mean(real_amp_1)),
            "real_linear_residual_ratio_mean": float(np.mean(real_ratio_1)),
            "real_validation_amplitude_mean_s_m": float(
                np.mean(real_validation_amp_1)
            ),
            "real_validation_linear_residual_ratio_mean": float(
                np.mean(real_validation_ratio_1)
            ),
            "history": history_1,
        },
        "stage_2": {
            "best_epoch": best_epoch_2,
            "best_validation_loss": best_loss_2,
            "synthetic_correction_amplitude_mean_s_m": float(
                np.mean(synthetic_amp_2)
            ),
            "synthetic_linear_residual_ratio_mean": float(
                np.mean(synthetic_ratio_2)
            ),
            "real_correction_amplitude_mean_s_m": float(np.mean(real_amp_2)),
            "real_linear_residual_ratio_mean": float(np.mean(real_ratio_2)),
            "physics": {
                "synthetic_train": synthetic_physics_2,
                "synthetic_validation": synthetic_validation_physics_2,
                "real_train": real_physics_2,
                "real_validation": real_validation_physics_2,
            },
            "history": history_2,
        },
        "elapsed_seconds": time.time() - started,
    }
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
