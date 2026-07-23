#!/usr/bin/env python3
"""Train the two-stage, beat-context, core-guided dense PVI-GCNM."""

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
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

from gcnm_pvi.anatomical_phantoms import element_positions
from gcnm_pvi.config import GcnmConfig
from gcnm_pvi.core_guided_model import CoreGuidedGCNMStage
from gcnm_pvi.direction_anchored_model import DirectionAnchoredCoreStage
from gcnm_pvi.core_guided_physics import (
    build_fixed_linear_operator,
    linear_amplitude_calibration,
    normalized_context_direction,
)
from gcnm_pvi.iterative_physics import dataset_lm_directions, diagnostics_summary
from gcnm_pvi.runtime import build_runtime
from gcnm_pvi.train_voltage_vessel_gcnm import _anatomy_targets


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_arrays(path: Path, limit: int | None = None) -> dict[str, np.ndarray]:
    required = ("sigma", "V", "tissue_labels")
    with np.load(path) as source:
        missing = [key for key in required if key not in source]
        if missing:
            raise KeyError(f"{path} is missing {missing}")
        arrays = {key: np.asarray(source[key]) for key in source.files}
    # The original default-finger dataset stores one independently sampled
    # phase per anatomy, so it predates the multi-phase ``V_template`` field.
    # In that case the measured phase is the only label-free localization
    # context available.  Its absolute normalized LM direction is still
    # sign-invariant, and evaluation uses the same fallback for archives with
    # no beat/period identifiers.
    if "V_template" not in arrays:
        arrays["V_template"] = np.asarray(arrays["V"]).copy()
    if limit is not None:
        arrays = {
            key: value[:limit] if value.ndim and len(value) >= limit else value
            for key, value in arrays.items()
        }
    return arrays


def _vessel_targets(
    parameters: np.ndarray | None,
    positions: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    nodes = len(positions)
    if parameters is None:
        return (
            np.zeros((nodes, 1), dtype=np.float32),
            np.zeros((nodes, 2), dtype=np.float32),
            np.zeros((1, 2, 2), dtype=np.float32),
        )
    soft, hard = [], []
    xy = positions[:, :2]
    for vessel in parameters:
        dx = xy[:, 0] - float(vessel[0])
        dy = xy[:, 1] - float(vessel[1])
        cosine, sine = np.cos(float(vessel[4])), np.sin(float(vessel[4]))
        local_x = cosine * dx + sine * dy
        local_y = -sine * dx + cosine * dy
        radius_squared = (
            (local_x / max(float(vessel[2]), 1e-5)) ** 2
            + (local_y / max(float(vessel[3]), 1e-5)) ** 2
        )
        hard.append(radius_squared <= 1.0)
        target = np.exp(-0.5 * radius_squared / 1.25**2)
        target[radius_squared > 9.0] = 0.0
        soft.append(target)
    hard_array = np.stack(hard, axis=1)
    soft_array = np.stack(soft, axis=1).astype(np.float32)
    centers = np.asarray(parameters[:, :2], dtype=np.float32)[None, :, :]
    return np.any(hard_array, axis=1, keepdims=True).astype(np.float32), soft_array, centers


def _graphs(
    truth: np.ndarray,
    current: np.ndarray,
    direction: np.ndarray,
    context: np.ndarray,
    voltage: np.ndarray,
    tissue_labels: np.ndarray | None,
    vessel_parameters: np.ndarray | None,
    positions: np.ndarray,
    edge_index,
    *,
    conductivity_scale: float,
    voltage_scale: float,
) -> list[Data]:
    graphs = []
    for index in range(len(truth)):
        core, vessel_target, centers = _vessel_targets(
            None if vessel_parameters is None else vessel_parameters[index],
            positions,
        )
        labels = (
            np.zeros(len(positions), dtype=np.uint8)
            if tissue_labels is None
            else np.asarray(tissue_labels[index], dtype=np.uint8)
        )
        muscle = (labels == 3)[:, None]
        outside = (~muscle & ~(core > 0.5)).astype(np.float32)
        features = np.column_stack(
            (
                current[index] / conductivity_scale,
                direction[index] / conductivity_scale,
                context[index],
                positions,
            )
        )
        graphs.append(
            Data(
                x=torch.tensor(features, dtype=torch.float32),
                edge_index=edge_index,
                y=torch.tensor(
                    truth[index, :, None] / conductivity_scale,
                    dtype=torch.float32,
                ),
                voltage=torch.tensor(
                    voltage[index][None, :] / voltage_scale,
                    dtype=torch.float32,
                ),
                core_target=torch.tensor(core, dtype=torch.float32),
                vessel_target=torch.tensor(vessel_target, dtype=torch.float32),
                vessel_centers=torch.tensor(centers, dtype=torch.float32),
                muscle_target=torch.tensor(muscle, dtype=torch.float32),
                outside_target=torch.tensor(outside, dtype=torch.float32),
            )
        )
    return graphs


def _correlation_loss(prediction: torch.Tensor, batch) -> torch.Tensor:
    values = []
    for start, stop in zip(batch.ptr[:-1], batch.ptr[1:]):
        pred = prediction[start:stop, 0]
        truth = batch.y[start:stop, 0]
        pred = pred - torch.mean(pred)
        truth = truth - torch.mean(truth)
        truth_norm = torch.linalg.vector_norm(truth)
        if truth_norm < 1e-5:
            values.append(pred.new_tensor(0.0))
        else:
            denominator = (
                torch.linalg.vector_norm(pred) * truth_norm
            ).clamp_min(1e-8)
            values.append(1.0 - torch.sum(pred * truth) / denominator)
    return torch.stack(values).mean()


def _auxiliary_losses(
    prediction: torch.Tensor,
    core_logits: torch.Tensor,
    attention: torch.Tensor,
    batch,
) -> dict[str, torch.Tensor]:
    core_losses, slot_losses, centroid_losses = [], [], []
    overlap_losses, amplitude_losses, mass_losses = [], [], []
    coordinates = batch.x[:, 3:5]
    centers = batch.vessel_centers.reshape(-1, 2, 2)
    for graph_index, (start, stop) in enumerate(zip(batch.ptr[:-1], batch.ptr[1:])):
        pred = prediction[start:stop, 0]
        truth = batch.y[start:stop, 0]
        core_truth = batch.core_target[start:stop, 0]
        logits = core_logits[start:stop, 0]
        positive_weight = torch.clamp(
            (len(core_truth) - torch.sum(core_truth))
            / torch.sum(core_truth).clamp_min(1.0),
            max=50.0,
        )
        bce = F.binary_cross_entropy_with_logits(
            logits,
            core_truth,
            pos_weight=positive_weight,
        )
        probability = torch.sigmoid(logits)
        dice = 1.0 - (2.0 * torch.sum(probability * core_truth) + 1e-6) / (
            torch.sum(probability) + torch.sum(core_truth) + 1e-6
        )
        core_losses.append(bce + dice)

        local_attention = attention[start:stop]
        targets = batch.vessel_target[start:stop]
        target_probability = targets / targets.sum(dim=0, keepdim=True).clamp_min(1e-8)
        log_attention = torch.log(local_attention.clamp_min(1e-12))
        direct = -torch.sum(target_probability * log_attention) / 2.0
        swapped = -torch.sum(target_probability * log_attention.flip(dims=(1,))) / 2.0
        slot_losses.append(
            torch.minimum(direct, swapped) / max(math.log(stop - start), 1.0)
        )

        predicted_centers = torch.stack(
            [
                torch.sum(
                    local_attention[:, slot : slot + 1]
                    * coordinates[start:stop],
                    dim=0,
                )
                for slot in range(2)
            ]
        )
        direct_center = torch.mean((predicted_centers - centers[graph_index]) ** 2)
        swapped_center = torch.mean(
            (predicted_centers.flip(dims=(0,)) - centers[graph_index]) ** 2
        )
        centroid_losses.append(torch.minimum(direct_center, swapped_center))
        overlap_losses.append(
            torch.sum(local_attention[:, 0] * local_attention[:, 1])
            / (
                torch.linalg.vector_norm(local_attention[:, 0])
                * torch.linalg.vector_norm(local_attention[:, 1])
                + 1e-8
            )
        )

        for slot in range(2):
            weights = targets[:, slot]
            denominator = torch.sum(weights).clamp_min(1e-8)
            predicted_mean = torch.sum(weights * pred) / denominator
            truth_mean = torch.sum(weights * truth) / denominator
            amplitude_losses.append((predicted_mean - truth_mean) ** 2)
        truth_mass = torch.sum(torch.abs(truth))
        if truth_mass > 1e-4:
            ratio = torch.clamp(
                torch.sum(torch.abs(pred)) / truth_mass,
                min=0.0,
                max=5.0,
            )
            mass_losses.append(
                F.smooth_l1_loss(
                    ratio,
                    ratio.new_tensor(1.0),
                    beta=0.5,
                )
            )
    zero = prediction.new_tensor(0.0)
    return {
        "core_segmentation": torch.stack(core_losses).mean() if core_losses else zero,
        "slot_localization": torch.stack(slot_losses).mean() if slot_losses else zero,
        "centroid": torch.stack(centroid_losses).mean() if centroid_losses else zero,
        "attention_overlap": torch.stack(overlap_losses).mean() if overlap_losses else zero,
        "core_amplitude": torch.stack(amplitude_losses).mean() if amplitude_losses else zero,
        "mass": torch.stack(mass_losses).mean() if mass_losses else zero,
    }


def _loss(model, batch, *, weights: dict[str, float]) -> tuple[torch.Tensor, dict[str, float]]:
    prediction, core_logits, attention = model(batch, return_aux=True)
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
    auxiliary = _auxiliary_losses(prediction, core_logits, attention, batch)
    correlation = _correlation_loss(prediction, batch)
    correction = torch.mean((prediction - batch.x[:, 0:1]) ** 2)
    total = (
        map_loss
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
    components = {
        "map": float(map_loss.detach()),
        "artifact": float(artifact.detach()),
        "correlation": float(correlation.detach()),
        "correction": float(correction.detach()),
        **{key: float(value.detach()) for key, value in auxiliary.items()},
    }
    return total, components


def _epoch(model, loader, weights, optimizer=None) -> tuple[float, dict[str, float]]:
    training = optimizer is not None
    model.train(training)
    device = next(model.parameters()).device
    total, graphs = 0.0, 0
    component_total: dict[str, float] = {}
    for batch in loader:
        batch = batch.to(device)
        if training:
            optimizer.zero_grad()
        loss, components = _loss(model, batch, weights=weights)
        if training:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                [parameter for parameter in model.parameters() if parameter.requires_grad],
                max_norm=5.0,
            )
            optimizer.step()
        count = int(batch.num_graphs)
        total += float(loss.detach()) * count
        graphs += count
        for key, value in components.items():
            component_total[key] = component_total.get(key, 0.0) + value * count
    return total / max(graphs, 1), {
        key: value / max(graphs, 1) for key, value in component_total.items()
    }


def _fit(
    model,
    train_graphs,
    validation_graphs,
    cfg,
    *,
    epochs: int,
    weights: dict[str, float],
    label: str,
) -> tuple[CoreGuidedGCNMStage, list[dict], int, float]:
    train_loader = DataLoader(train_graphs, batch_size=cfg.batch_size, shuffle=True)
    validation_loader = DataLoader(
        validation_graphs, batch_size=cfg.batch_size, shuffle=False
    )
    optimizer = torch.optim.Adam(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=cfg.learning_rate,
    )
    best_state = copy.deepcopy(model.state_dict())
    best_loss, best_epoch, patience = float("inf"), -1, int(cfg.patience)
    history = []
    for epoch in range(int(epochs)):
        train_loss, train_components = _epoch(
            model, train_loader, weights, optimizer
        )
        with torch.no_grad():
            validation_loss, validation_components = _epoch(
                model, validation_loader, weights
            )
        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "validation_loss": validation_loss,
                "train_components": train_components,
                "validation_components": validation_components,
            }
        )
        print(
            f"{label} epoch {epoch:03d} train={train_loss:.6f} "
            f"validation={validation_loss:.6f}",
            flush=True,
        )
        if validation_loss <= best_loss:
            best_loss = validation_loss
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            patience = int(cfg.patience)
        else:
            patience -= 1
            if patience <= 0:
                break
    model.load_state_dict(best_state)
    return model, history, best_epoch, best_loss


def _predict(
    model,
    graphs,
    scale: float,
    *,
    batch_size: int = 16,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    loader = DataLoader(graphs, batch_size=batch_size, shuffle=False)
    device = next(model.parameters()).device
    maps, centers, core_probabilities = [], [], []
    model.eval()
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            prediction, core_logits, attention = model(batch, return_aux=True)
            maps.append(prediction.cpu().numpy())
            core_probabilities.append(torch.sigmoid(core_logits).cpu().numpy())
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
    node_count = graphs[0].num_nodes
    map_array = np.concatenate(maps).reshape(-1, node_count) * float(scale)
    core_array = np.concatenate(core_probabilities).reshape(-1, node_count)
    return map_array, np.stack(centers), core_array


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--validation", type=Path, required=True)
    parser.add_argument("--train-anatomy", type=Path, required=True)
    parser.add_argument("--validation-anatomy", type=Path, required=True)
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--models-dir", type=Path, required=True)
    parser.add_argument("--results-dir", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--max-train", type=int, default=None)
    parser.add_argument("--max-validation", type=int, default=None)
    parser.add_argument("--baseline-conductivity", type=float, default=0.7)
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
    parser.add_argument("--stage2-correction-weight", type=float, default=0.02)
    parser.add_argument("--stage1-limit", type=float, default=2.0)
    parser.add_argument("--stage2-limit", type=float, default=0.5)
    parser.add_argument("--maximum-amplitude-scale", type=float, default=2.0)
    parser.add_argument(
        "--architecture",
        choices=["core_guided", "direction_anchored"],
        default="core_guided",
        help="network output parameterization; all training protocol settings stay fixed",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--allow-overwrite", action="store_true")
    args = parser.parse_args()

    args.models_dir.mkdir(parents=True, exist_ok=True)
    args.results_dir.mkdir(parents=True, exist_ok=True)
    localizer_path = args.models_dir / f"{args.model_name}_stage1.pt"
    refiner_path = args.models_dir / f"{args.model_name}_stage2.pt"
    report_path = args.results_dir / f"{args.model_name}_training_report.json"
    if not args.allow_overwrite and any(
        path.exists() for path in (localizer_path, refiner_path, report_path)
    ):
        raise FileExistsError("experiment output exists; choose a new model name")

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    cfg = GcnmConfig.from_yaml(args.config)
    runtime = build_runtime(cfg, include_forward=False)
    train = _load_arrays(args.train, args.max_train)
    validation = _load_arrays(args.validation, args.max_validation)
    magnitude = np.abs(np.asarray(train["sigma"], dtype=np.float64))
    nonzero = magnitude[magnitude > 1e-8]
    conductivity_scale = max(float(np.percentile(nonzero, 99.5)), 1e-6)
    voltage_scale = max(
        float(np.percentile(np.abs(train["V"]), 99.5)), 1e-12
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
    elements = train["sigma"].shape[1]
    baseline_vector = np.full(elements, args.baseline_conductivity, dtype=np.float64)
    fixed = build_fixed_linear_operator(
        runtime["physics_inv"],
        baseline_vector,
        regularizer=runtime["mappings"].laplace,
        hyper_pvi=cfg.hyper_pvi,
        lambda_lm=cfg.lambda_lm,
    )
    zero_train = np.zeros_like(train["sigma"], dtype=np.float64)
    zero_validation = np.zeros_like(validation["sigma"], dtype=np.float64)
    direction_train_1 = fixed.directions(train["V"])
    direction_validation_1 = fixed.directions(validation["V"])
    context_train = normalized_context_direction(
        fixed.directions(train["V_template"])
    )
    context_validation = normalized_context_direction(
        fixed.directions(validation["V_template"])
    )
    train_graphs_1 = _graphs(
        train["sigma"], zero_train, direction_train_1, context_train, train["V"],
        train["tissue_labels"], train_parameters, positions, runtime["edge_index"],
        conductivity_scale=conductivity_scale, voltage_scale=voltage_scale,
    )
    validation_graphs_1 = _graphs(
        validation["sigma"], zero_validation, direction_validation_1,
        context_validation, validation["V"], validation["tissue_labels"],
        validation_parameters, positions, runtime["edge_index"],
        conductivity_scale=conductivity_scale, voltage_scale=voltage_scale,
    )
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    common_weights = {
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
        "correction": 0.0,
    }
    start = time.time()
    if args.architecture == "direction_anchored":
        stage1 = DirectionAnchoredCoreStage(
            accumulate_current=False,
            correction_limit=args.stage1_limit,
        ).to(device)
    else:
        stage1 = CoreGuidedGCNMStage(
            residual=False, correction_limit=args.stage1_limit
        ).to(device)
    stage1, history_1, best_epoch_1, best_loss_1 = _fit(
        stage1,
        train_graphs_1,
        validation_graphs_1,
        cfg,
        epochs=args.epochs,
        weights=common_weights,
        label="stage 1 core-guided map",
    )
    stage1_train_raw, _, _ = _predict(
        stage1, train_graphs_1, conductivity_scale, batch_size=cfg.batch_size
    )
    stage1_validation_raw, _, _ = _predict(
        stage1, validation_graphs_1, conductivity_scale, batch_size=cfg.batch_size
    )
    stage1_train, train_scale, _ = linear_amplitude_calibration(
        stage1_train_raw,
        train["V"],
        fixed.jacobian,
        maximum_scale=args.maximum_amplitude_scale,
    )
    stage1_validation, validation_scale, _ = linear_amplitude_calibration(
        stage1_validation_raw,
        validation["V"],
        fixed.jacobian,
        maximum_scale=args.maximum_amplitude_scale,
    )
    baseline_train = np.full_like(stage1_train, args.baseline_conductivity)
    baseline_validation = np.full_like(
        stage1_validation, args.baseline_conductivity
    )
    baseline_voltage_train = np.broadcast_to(
        fixed.baseline_voltage[None, :], train["V"].shape
    )
    baseline_voltage_validation = np.broadcast_to(
        fixed.baseline_voltage[None, :], validation["V"].shape
    )
    direction_train_2, diagnostics_train_2, _ = dataset_lm_directions(
        runtime["physics_inv"], baseline_train, stage1_train, train["V"],
        regularizer=runtime["mappings"].laplace, hyper_pvi=cfg.hyper_pvi,
        lambda_lm=cfg.lambda_lm, step_size=cfg.lm_step_size,
        baseline_voltages=baseline_voltage_train,
        progress_label="core-guided stage 2 train",
    )
    direction_validation_2, diagnostics_validation_2, _ = dataset_lm_directions(
        runtime["physics_inv"], baseline_validation, stage1_validation,
        validation["V"], regularizer=runtime["mappings"].laplace,
        hyper_pvi=cfg.hyper_pvi, lambda_lm=cfg.lambda_lm,
        step_size=cfg.lm_step_size,
        baseline_voltages=baseline_voltage_validation,
        progress_label="core-guided stage 2 validation",
    )
    train_graphs_2 = _graphs(
        train["sigma"], stage1_train, direction_train_2, context_train, train["V"],
        train["tissue_labels"], train_parameters, positions, runtime["edge_index"],
        conductivity_scale=conductivity_scale, voltage_scale=voltage_scale,
    )
    validation_graphs_2 = _graphs(
        validation["sigma"], stage1_validation, direction_validation_2,
        context_validation, validation["V"], validation["tissue_labels"],
        validation_parameters, positions, runtime["edge_index"],
        conductivity_scale=conductivity_scale, voltage_scale=voltage_scale,
    )
    if args.architecture == "direction_anchored":
        stage2 = DirectionAnchoredCoreStage(
            accumulate_current=True,
            correction_limit=args.stage2_limit,
        )
    else:
        stage2 = CoreGuidedGCNMStage(
            residual=True, correction_limit=args.stage2_limit
        )
    stage2.copy_context_from(stage1.cpu())
    stage2.freeze_context()
    stage2.to(device)
    stage2_weights = dict(common_weights)
    stage2_weights["correction"] = args.stage2_correction_weight
    stage2, history_2, best_epoch_2, best_loss_2 = _fit(
        stage2,
        train_graphs_2,
        validation_graphs_2,
        cfg,
        epochs=args.epochs,
        weights=stage2_weights,
        label="stage 2 bounded physics refiner",
    )

    architecture_id = (
        "direction_anchored_core_residual_two_stage_v1"
        if args.architecture == "direction_anchored"
        else "core_guided_dense_two_stage_v1"
    )
    contract = {
        "model_name": args.model_name,
        "architecture": architecture_id,
        "measurements": int(train["V"].shape[1]),
        "elements": int(elements),
        "node_features": 6,
        "conductivity_scale": conductivity_scale,
        "voltage_scale": voltage_scale,
        "baseline_mode": "homogeneous",
        "baseline_conductivity": args.baseline_conductivity,
        "stage1_limit": args.stage1_limit,
        "stage2_limit": args.stage2_limit,
        "maximum_amplitude_scale": args.maximum_amplitude_scale,
        "beat_context": (
            "absolute normalized fixed-LM direction; saved rank-one beat "
            "template when present, same-frame voltage for legacy single-phase data"
        ),
        "stage2_context_frozen": True,
        "config_sha256": _sha256(args.config),
        "mesh_inverse_sha256": _sha256(Path(cfg.mesh_inv_h5)),
        "mappings_sha256": _sha256(Path(cfg.mappings_h5)),
        "train_dataset_sha256": _sha256(args.train),
        "validation_dataset_sha256": _sha256(args.validation),
    }
    torch.save(
        {
            "model_kind": f"{args.architecture}_stage1",
            "state_dict": stage1.cpu().state_dict(),
            "contract": contract,
        },
        localizer_path,
    )
    torch.save(
        {
            "model_kind": f"{args.architecture}_stage2",
            "state_dict": stage2.cpu().state_dict(),
            "contract": contract,
        },
        refiner_path,
    )
    report = {
        "method": (
            "two-stage direction-anchored core-guided residual GCNM"
            if args.architecture == "direction_anchored"
            else "two-stage beat-context core-guided dense GCNM"
        ),
        "target": "clean signed synthetic differential conductivity",
        "pvi_images_used_as_labels": False,
        "contract": contract,
        "loss_stage_1": common_weights,
        "loss_stage_2": stage2_weights,
        "train_samples": len(train["sigma"]),
        "validation_samples": len(validation["sigma"]),
        "stage_1": {
            "best_epoch": best_epoch_1,
            "best_validation_loss": best_loss_1,
            "amplitude_scale_train": {
                "mean": float(np.mean(train_scale)),
                "median": float(np.median(train_scale)),
            },
            "amplitude_scale_validation": {
                "mean": float(np.mean(validation_scale)),
                "median": float(np.median(validation_scale)),
            },
            "history": history_1,
        },
        "stage_2": {
            "best_epoch": best_epoch_2,
            "best_validation_loss": best_loss_2,
            "physics": {
                "train": diagnostics_summary(diagnostics_train_2),
                "validation": diagnostics_summary(diagnostics_validation_2),
            },
            "history": history_2,
        },
        "elapsed_seconds": time.time() - start,
    }
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
