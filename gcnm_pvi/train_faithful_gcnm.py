#!/usr/bin/env python3
"""Train faithful PVI-GCNM stages with newly recomputed physics per stage."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import torch
from torch_geometric.loader import DataLoader

from gcnm_pvi.anatomical_phantoms import element_positions
from gcnm_pvi.config import GcnmConfig
from gcnm_pvi.coordinate_runtime import (
    build_coordinate_model,
    make_coordinate_graphs,
    predict_coordinate_graphs,
)
from gcnm_pvi.dual_mesh_physics import ProjectedFineMeshPhysics
from gcnm_pvi.iterative_physics import (
    FixedZeroCurrentLMSolver,
    LowRankRegularizedSolver,
    diagnostics_summary,
    parallel_dataset_absolute_lm_directions,
    parallel_dataset_lm_directions,
)
from gcnm_pvi.runtime import build_runtime


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load(
    path: Path,
    limit: int | None = None,
    *,
    require_resting: bool = False,
) -> dict[str, np.ndarray]:
    source = np.load(path)
    required = ("sigma", "sigma_baseline", "V") + (
        ("sigma_resting",) if require_resting else ()
    )
    missing = [key for key in required if key not in source]
    if missing:
        raise ValueError(f"{path} is missing required arrays: {missing}")
    values = {key: np.asarray(source[key], dtype=np.float64) for key in required}
    if limit is not None:
        values = {key: value[:limit] for key, value in values.items()}
    return values


# Compatibility aliases for historical imports and saved experiment scripts.
# New code imports these helpers from ``coordinate_runtime`` directly.
_make_dataset = make_coordinate_graphs
_model = build_coordinate_model


def _soft_support_dice_loss(prediction: torch.Tensor, batch) -> torch.Tensor:
    """Differentiable vessel-support Dice, averaged per graph.

    The adaptive threshold is derived only from each clean training target.
    This loss is never used as evidence on real subject data.
    """
    losses = []
    boundaries = batch.ptr if hasattr(batch, "ptr") else [0, len(prediction)]
    for start, stop in zip(boundaries[:-1], boundaries[1:]):
        pred = prediction[start:stop, 0]
        target = batch.y[start:stop, 0]
        if hasattr(batch, "reference"):
            reference = batch.reference[start:stop, 0]
            pred = pred - reference
            target = target - reference
        support = torch.abs(target) > 1e-8
        if not torch.any(support):
            continue
        reference = torch.median(torch.abs(target[support])).detach().clamp_min(1e-6)
        probability = torch.sigmoid(
            (torch.abs(pred) - 0.25 * reference) / (0.10 * reference + 1e-6)
        )
        truth = support.to(probability.dtype)
        intersection = torch.sum(probability * truth)
        dice = (2.0 * intersection + 1e-6) / (
            torch.sum(probability) + torch.sum(truth) + 1e-6
        )
        losses.append(1.0 - dice)
    if not losses:
        return prediction.new_tensor(0.0)
    return torch.stack(losses).mean()


def _loss(
    model,
    batch,
    background_weight: float,
    dice_weight: float = 0.0,
    hard_background_weight: float = 0.0,
    hard_background_fraction: float = 0.05,
    balanced_support: bool = False,
    correlation_weight: float = 0.0,
    amplitude_weight: float = 0.0,
    relative_amplitude_weight: float = 0.0,
    temporal_delta_weight: float = 0.0,
    temporal_correlation_weight: float = 0.0,
    temporal_amplitude_weight: float = 0.0,
    temporal_group_size: int = 0,
) -> torch.Tensor:
    prediction = model(batch)
    if balanced_support:
        graph_losses = []
        boundaries = batch.ptr if hasattr(batch, "ptr") else [0, len(prediction)]
        for start, stop in zip(boundaries[:-1], boundaries[1:]):
            pred = prediction[start:stop, 0]
            target = batch.y[start:stop, 0]
            support_target = target
            if hasattr(batch, "reference"):
                support_target = support_target - batch.reference[start:stop, 0]
            support = torch.abs(support_target) > 1e-8
            terms = []
            if torch.any(support):
                terms.append(torch.mean((pred[support] - target[support]) ** 2))
            if torch.any(~support):
                terms.append(torch.mean((pred[~support] - target[~support]) ** 2))
            graph_losses.append(torch.stack(terms).mean())
        loss = torch.stack(graph_losses).mean()
    else:
        loss = torch.mean(batch.weights * (prediction - batch.y) ** 2)
    if background_weight > 0:
        mask = batch.background.bool()
        if torch.any(mask):
            background_error = (prediction[mask] - batch.y[mask]) ** 2
            loss = loss + float(background_weight) * torch.mean(background_error)
            if hard_background_weight > 0:
                count = max(
                    1,
                    int(np.ceil(float(hard_background_fraction) * background_error.numel())),
                )
                hardest = torch.topk(background_error, k=count, largest=True).values
                loss = loss + float(hard_background_weight) * torch.mean(hardest)
    if dice_weight > 0:
        loss = loss + float(dice_weight) * _soft_support_dice_loss(prediction, batch)
    if correlation_weight > 0 or amplitude_weight > 0 or relative_amplitude_weight > 0:
        correlation_losses = []
        amplitude_losses = []
        relative_amplitude_losses = []
        boundaries = batch.ptr if hasattr(batch, "ptr") else [0, len(prediction)]
        for start, stop in zip(boundaries[:-1], boundaries[1:]):
            pred = prediction[start:stop, 0]
            target = batch.y[start:stop, 0]
            if hasattr(batch, "reference"):
                reference = batch.reference[start:stop, 0]
                pred = pred - reference
                target = target - reference
            pred_centered = pred - torch.mean(pred)
            target_centered = target - torch.mean(target)
            denominator = torch.linalg.vector_norm(pred_centered) * torch.linalg.vector_norm(
                target_centered
            )
            correlation = torch.sum(pred_centered * target_centered) / denominator.clamp_min(
                1e-8
            )
            correlation_losses.append(1.0 - correlation)
            amplitude_losses.append(
                (
                    torch.sqrt(torch.mean(pred * pred) + 1e-8)
                    - torch.sqrt(torch.mean(target * target) + 1e-8)
                )
                ** 2
            )
            pred_rms = torch.sqrt(torch.mean(pred * pred) + 1e-12)
            target_rms = torch.sqrt(torch.mean(target * target) + 1e-12)
            relative_amplitude_losses.append(
                torch.log((pred_rms + 1e-6) / (target_rms + 1e-6)) ** 2
            )
        if correlation_weight > 0:
            loss = loss + float(correlation_weight) * torch.stack(correlation_losses).mean()
        if amplitude_weight > 0:
            loss = loss + float(amplitude_weight) * torch.stack(amplitude_losses).mean()
        if relative_amplitude_weight > 0:
            loss = loss + float(relative_amplitude_weight) * torch.stack(
                relative_amplitude_losses
            ).mean()
    if temporal_delta_weight > 0 or temporal_correlation_weight > 0 or temporal_amplitude_weight > 0:
        graphs = int(batch.num_graphs)
        if temporal_group_size <= 1 or graphs % temporal_group_size:
            raise ValueError(
                "temporal loss requires complete, fixed-size ordered graph groups"
            )
        elements = prediction.numel() // graphs
        predicted_graphs = prediction[:, 0].reshape(graphs, elements)
        target_graphs = batch.y[:, 0].reshape(graphs, elements)
        temporal_losses = []
        temporal_correlations = []
        temporal_amplitudes = []
        for start in range(0, graphs, temporal_group_size):
            stop = start + temporal_group_size
            pred = predicted_graphs[start:stop] - predicted_graphs[start : start + 1]
            target = target_graphs[start:stop] - target_graphs[start : start + 1]
            target_power = torch.mean(target * target).clamp_min(1e-12)
            temporal_losses.append(torch.mean((pred - target) ** 2) / target_power)
            pred_centered = pred - torch.mean(pred)
            target_centered = target - torch.mean(target)
            correlation = torch.sum(pred_centered * target_centered) / (
                torch.linalg.vector_norm(pred_centered)
                * torch.linalg.vector_norm(target_centered)
            ).clamp_min(1e-8)
            temporal_correlations.append(1.0 - correlation)
            pred_rms = torch.sqrt(torch.mean(pred * pred) + 1e-12)
            target_rms = torch.sqrt(target_power)
            temporal_amplitudes.append(
                torch.log((pred_rms + 1e-6) / (target_rms + 1e-6)) ** 2
            )
        if temporal_delta_weight > 0:
            loss = loss + float(temporal_delta_weight) * torch.stack(temporal_losses).mean()
        if temporal_correlation_weight > 0:
            loss = loss + float(temporal_correlation_weight) * torch.stack(
                temporal_correlations
            ).mean()
        if temporal_amplitude_weight > 0:
            loss = loss + float(temporal_amplitude_weight) * torch.stack(
                temporal_amplitudes
            ).mean()
    return loss


def _run_epoch(
    model,
    loader,
    background_weight: float,
    dice_weight: float = 0.0,
    hard_background_weight: float = 0.0,
    hard_background_fraction: float = 0.05,
    balanced_support: bool = False,
    correlation_weight: float = 0.0,
    amplitude_weight: float = 0.0,
    relative_amplitude_weight: float = 0.0,
    temporal_delta_weight: float = 0.0,
    temporal_correlation_weight: float = 0.0,
    temporal_amplitude_weight: float = 0.0,
    temporal_group_size: int = 0,
    optimizer=None,
) -> float:
    training = optimizer is not None
    model.train(training)
    total, graphs = 0.0, 0
    device = next(model.parameters()).device
    for batch in loader:
        batch = batch.to(device)
        if training:
            optimizer.zero_grad()
        loss = _loss(
            model,
            batch,
            background_weight,
            dice_weight,
            hard_background_weight,
            hard_background_fraction,
            balanced_support,
            correlation_weight,
            amplitude_weight,
            relative_amplitude_weight,
            temporal_delta_weight,
            temporal_correlation_weight,
            temporal_amplitude_weight,
            temporal_group_size,
        )
        if training:
            loss.backward()
            optimizer.step()
        total += float(loss.detach()) * batch.num_graphs
        graphs += batch.num_graphs
    return total / max(graphs, 1)


def _checkpoint_score(
    model,
    loader,
    *,
    mode: str,
    validation_loss: float,
    background_coefficient: float,
    dice_coefficient: float,
) -> tuple[float, dict[str, float]]:
    if mode == "loss":
        return validation_loss, {"validation_loss": validation_loss}
    device = next(model.parameters()).device
    predictions, targets, backgrounds = [], [], []
    dice_values: list[float] = []
    model.eval()
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            prediction_batch = model(batch).cpu().numpy().ravel()
            target_batch = batch.y.cpu().numpy().ravel()
            background_batch = batch.background.cpu().numpy().ravel().astype(bool)
            predictions.append(prediction_batch)
            targets.append(target_batch)
            backgrounds.append(background_batch)
            boundaries = batch.ptr.cpu().numpy()
            for start, stop in zip(boundaries[:-1], boundaries[1:]):
                support = np.flatnonzero(~background_batch[start:stop])
                if not len(support):
                    continue
                local_prediction = prediction_batch[start:stop]
                selected = np.argpartition(np.abs(local_prediction), -len(support))[
                    -len(support) :
                ]
                dice_values.append(
                    len(np.intersect1d(support, selected)) / len(support)
                )
    prediction = np.concatenate(predictions)
    target = np.concatenate(targets)
    background = np.concatenate(backgrounds)
    rmse = float(np.sqrt(np.mean((prediction - target) ** 2)))
    truth_rms = max(float(np.sqrt(np.mean(target**2))), 1e-12)
    nrmse = rmse / truth_rms
    bg_rms = float(
        np.sqrt(np.mean((prediction[background] - target[background]) ** 2))
    )
    dice = float(np.mean(dice_values)) if dice_values else 0.0
    score = nrmse + background_coefficient * bg_rms - dice_coefficient * dice
    return score, {
        "validation_loss": validation_loss,
        "nrmse": nrmse,
        "background_rms_normalized": bg_rms,
        "dice_global": dice,
        "composite_score": score,
    }


_predict = predict_coordinate_graphs


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=root / "configs" / "subject006_anatomical_gcnm.yaml",
    )
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--validation", type=Path, required=True)
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--models-dir", type=Path, required=True)
    parser.add_argument("--results-dir", type=Path, required=True)
    parser.add_argument(
        "--output-mode",
        choices=["direct", "positive_direct", "proposal_residual", "shallow_residual"],
        default="direct",
    )
    parser.add_argument("--use-coordinates", action="store_true")
    parser.add_argument(
        "--use-voltage-mlp",
        action="store_true",
        help="Encode all boundary voltages with an MLP and broadcast them to graph nodes",
    )
    parser.add_argument("--temporal-delta-weight", type=float, default=0.0)
    parser.add_argument("--temporal-correlation-weight", type=float, default=0.0)
    parser.add_argument("--temporal-amplitude-weight", type=float, default=0.0)
    parser.add_argument("--temporal-group-size", type=int, default=0)
    parser.add_argument("--voltage-latent", type=int, default=64)
    parser.add_argument("--positive-weight", type=float, default=0.0)
    parser.add_argument("--background-weight", type=float, default=0.0)
    parser.add_argument("--dice-weight", type=float, default=0.0)
    parser.add_argument("--hard-background-weight", type=float, default=0.0)
    parser.add_argument("--hard-background-fraction", type=float, default=0.05)
    parser.add_argument("--balanced-support-loss", action="store_true")
    parser.add_argument("--correlation-weight", type=float, default=0.0)
    parser.add_argument("--amplitude-weight", type=float, default=0.0)
    parser.add_argument(
        "--relative-amplitude-weight",
        type=float,
        default=0.0,
        help="log-RMS ratio loss on the dynamic field; use only with background control",
    )
    parser.add_argument("--checkpoint-mode", choices=["loss", "composite"], default="loss")
    parser.add_argument("--checkpoint-background-coefficient", type=float, default=1.0)
    parser.add_argument("--checkpoint-dice-coefficient", type=float, default=0.25)
    parser.add_argument("--iterations", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--patience", type=int, default=None)
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="override config batch size; large frame batches improve GPU occupancy",
    )
    parser.add_argument(
        "--loader-workers",
        type=int,
        default=0,
        help="PyG loader workers (physics parallelism is controlled separately)",
    )
    parser.add_argument(
        "--validation-is-train-copy",
        action="store_true",
        help=(
            "reuse deterministic physics features when the immutable validation "
            "archive is byte-for-byte the same overfit pack as training"
        ),
    )
    parser.add_argument("--max-train", type=int, default=None)
    parser.add_argument("--max-validation", type=int, default=None)
    parser.add_argument("--minimum-conductivity", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument(
        "--baseline-mode", choices=["saved", "homogeneous"], default="homogeneous"
    )
    parser.add_argument(
        "--physics-contract",
        choices=["differential", "absolute"],
        default="differential",
        help=(
            "differential uses F(sigma_b+delta)-F(sigma_b); absolute uses "
            "F(sigma)-V_absolute and predicts absolute conductivity"
        ),
    )
    parser.add_argument(
        "--physics-mesh-mode",
        choices=["coarse", "projected_fine"],
        default="coarse",
        help=(
            "coarse evaluates F and J directly on the inverse mesh; "
            "projected_fine evaluates F_f(P sigma_c) and uses J_f P while "
            "retaining coarse conductivity updates and graph nodes"
        ),
    )
    parser.add_argument("--baseline-conductivity", type=float, default=0.7)
    parser.add_argument("--allow-overwrite", action="store_true")
    args = parser.parse_args()

    cfg = GcnmConfig.from_yaml(args.config)
    seed = cfg.seed if args.seed is None else int(args.seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    runtime = build_runtime(
        cfg, include_forward=args.physics_mesh_mode == "projected_fine"
    )
    if args.physics_mesh_mode == "projected_fine":
        if runtime["mappings"].c2f is None:
            raise ValueError(
                "projected-fine training requires a coarse-to-fine mapping"
            )
        stage_physics = ProjectedFineMeshPhysics(
            runtime["physics_fwd"], runtime["mappings"].c2f
        )
    else:
        stage_physics = runtime["physics_inv"]
    train = _load(
        args.train,
        args.max_train,
        require_resting=args.physics_contract == "absolute",
    )
    validation = _load(
        args.validation,
        args.max_validation,
        require_resting=args.physics_contract == "absolute",
    )
    if args.baseline_mode == "homogeneous":
        train["sigma_baseline"] = np.full_like(
            train["sigma_baseline"], args.baseline_conductivity
        )
        validation["sigma_baseline"] = np.full_like(
            validation["sigma_baseline"], args.baseline_conductivity
        )
    positions = element_positions(runtime["mesh_inv"]).astype(np.float32)
    scale = max(float(np.percentile(np.abs(train["sigma"]), 99.5)), 1e-6)
    voltage_scale = max(float(np.percentile(np.abs(train["V"]), 99.5)), 1e-12)
    iterations = args.iterations or cfg.iterations
    epochs = args.epochs or cfg.max_epochs
    in_channels = 5 if args.use_coordinates else 2
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if args.physics_contract == "absolute":
        current_train = train["sigma_baseline"].copy()
        current_validation = validation["sigma_baseline"].copy()
    else:
        current_train = np.zeros_like(train["sigma"])
        current_validation = np.zeros_like(validation["sigma"])
    baseline_train = None
    baseline_validation = None
    if args.validation_is_train_copy:
        for key in ("sigma", "sigma_baseline", "V"):
            if not np.array_equal(train[key], validation[key]):
                raise ValueError(
                    f"--validation-is-train-copy is invalid because {key} differs"
                )
    low_rank_solver = LowRankRegularizedSolver(
        runtime["mappings"].laplace,
        runtime["mappings"].num_elements,
        hyper_pvi=cfg.hyper_pvi,
        lambda_lm=cfg.lambda_lm,
    )
    fixed_first_stage = (
        FixedZeroCurrentLMSolver(
            stage_physics,
            np.full(runtime["mappings"].num_elements, args.baseline_conductivity),
            regularizer=runtime["mappings"].laplace,
            hyper_pvi=cfg.hyper_pvi,
            lambda_lm=cfg.lambda_lm,
            step_size=cfg.lm_step_size,
        )
        if args.baseline_mode == "homogeneous"
        else None
    )
    existing_checkpoints = list(args.models_dir.glob(f"{args.model_name}_*.pt"))
    existing_report = args.results_dir / f"{args.model_name}_training_report.json"
    if not args.allow_overwrite and (existing_checkpoints or existing_report.exists()):
        raise FileExistsError(
            f"experiment {args.model_name} already has outputs; use a new EXPERIMENT "
            "name or pass --allow-overwrite explicitly"
        )
    args.models_dir.mkdir(parents=True, exist_ok=True)
    args.results_dir.mkdir(parents=True, exist_ok=True)
    artifact_hashes = {
        "config_sha256": _sha256(args.config),
        "mesh_forward_sha256": _sha256(Path(cfg.mesh_fwd_h5)),
        "mesh_inverse_sha256": _sha256(Path(cfg.mesh_inv_h5)),
        "mappings_sha256": _sha256(Path(cfg.mappings_h5)),
        "train_dataset_sha256": _sha256(args.train),
        "validation_dataset_sha256": _sha256(args.validation),
    }

    history: list[list[dict]] = []
    physics_reports: list[dict] = []
    start_time = time.time()
    for iteration in range(iterations):
        if args.physics_contract == "absolute":
            current_train = np.maximum(
                current_train, float(args.minimum_conductivity)
            )
            current_validation = np.maximum(
                current_validation, float(args.minimum_conductivity)
            )
        else:
            current_train = (
                np.maximum(
                    train["sigma_baseline"] + current_train,
                    float(args.minimum_conductivity),
                )
                - train["sigma_baseline"]
            )
            current_validation = (
                np.maximum(
                    validation["sigma_baseline"] + current_validation,
                    float(args.minimum_conductivity),
                )
                - validation["sigma_baseline"]
            )
        if iteration == 0 and fixed_first_stage is not None:
            if args.physics_contract == "absolute":
                direction_train, diag_train, baseline_train = (
                    fixed_first_stage.solve_many_absolute(train["V"])
                )
                if args.validation_is_train_copy:
                    direction_validation = direction_train.copy()
                    diag_validation = copy.deepcopy(diag_train)
                    baseline_validation = baseline_train.copy()
                else:
                    direction_validation, diag_validation, baseline_validation = (
                        fixed_first_stage.solve_many_absolute(validation["V"])
                    )
            else:
                direction_train, diag_train, baseline_train = fixed_first_stage.solve_many(
                    train["V"]
                )
                if args.validation_is_train_copy:
                    direction_validation = direction_train.copy()
                    diag_validation = copy.deepcopy(diag_train)
                    baseline_validation = baseline_train.copy()
                else:
                    direction_validation, diag_validation, baseline_validation = (
                        fixed_first_stage.solve_many(validation["V"])
                    )
        elif args.physics_contract == "absolute":
            direction_train, diag_train, baseline_train = (
                parallel_dataset_absolute_lm_directions(
                    stage_physics,
                    current_train,
                    train["V"],
                    regularizer=runtime["mappings"].laplace,
                    hyper_pvi=cfg.hyper_pvi,
                    lambda_lm=cfg.lambda_lm,
                    step_size=cfg.lm_step_size,
                    minimum_conductivity=args.minimum_conductivity,
                    progress_label=f"stage {iteration + 1} train absolute",
                    system_solver=low_rank_solver,
                )
            )
            if args.validation_is_train_copy:
                direction_validation = direction_train.copy()
                diag_validation = copy.deepcopy(diag_train)
                baseline_validation = baseline_train.copy()
            else:
                direction_validation, diag_validation, baseline_validation = (
                    parallel_dataset_absolute_lm_directions(
                        stage_physics,
                    current_validation,
                    validation["V"],
                    regularizer=runtime["mappings"].laplace,
                    hyper_pvi=cfg.hyper_pvi,
                    lambda_lm=cfg.lambda_lm,
                    step_size=cfg.lm_step_size,
                    minimum_conductivity=args.minimum_conductivity,
                    progress_label=f"stage {iteration + 1} validation absolute",
                    system_solver=low_rank_solver,
                )
            )
        else:
            direction_train, diag_train, baseline_train = parallel_dataset_lm_directions(
                stage_physics,
                train["sigma_baseline"],
                current_train,
                train["V"],
                regularizer=runtime["mappings"].laplace,
                hyper_pvi=cfg.hyper_pvi,
                lambda_lm=cfg.lambda_lm,
                step_size=cfg.lm_step_size,
                minimum_conductivity=args.minimum_conductivity,
                baseline_voltages=baseline_train,
                progress_label=f"stage {iteration + 1} train",
                system_solver=low_rank_solver,
            )
            if args.validation_is_train_copy:
                direction_validation = direction_train.copy()
                diag_validation = copy.deepcopy(diag_train)
                baseline_validation = baseline_train.copy()
            else:
                direction_validation, diag_validation, baseline_validation = (
                    parallel_dataset_lm_directions(
                        stage_physics,
                    validation["sigma_baseline"],
                    current_validation,
                    validation["V"],
                    regularizer=runtime["mappings"].laplace,
                    hyper_pvi=cfg.hyper_pvi,
                    lambda_lm=cfg.lambda_lm,
                    step_size=cfg.lm_step_size,
                    minimum_conductivity=args.minimum_conductivity,
                    baseline_voltages=baseline_validation,
                    progress_label=f"stage {iteration + 1} validation",
                    system_solver=low_rank_solver,
                )
            )
        physics_reports.append(
            {
                "stage": iteration + 1,
                "train": diagnostics_summary(diag_train),
                "validation": diagnostics_summary(diag_validation),
            }
        )
        train_set = _make_dataset(
            train["sigma"], current_train, direction_train, positions, runtime["edge_index"],
            scale=scale, use_coordinates=args.use_coordinates,
            positive_weight=args.positive_weight,
            voltage=train["V"] if args.use_voltage_mlp else None,
            voltage_scale=voltage_scale,
            reference=(
                train["sigma_resting"]
                if args.physics_contract == "absolute"
                else None
            ),
        )
        validation_set = _make_dataset(
            validation["sigma"], current_validation, direction_validation,
            positions, runtime["edge_index"], scale=scale,
            use_coordinates=args.use_coordinates, positive_weight=args.positive_weight,
            voltage=validation["V"] if args.use_voltage_mlp else None,
            voltage_scale=voltage_scale,
            reference=(
                validation["sigma_resting"]
                if args.physics_contract == "absolute"
                else None
            ),
        )
        temporal_enabled = any(
            value > 0
            for value in (
                args.temporal_delta_weight,
                args.temporal_correlation_weight,
                args.temporal_amplitude_weight,
            )
        )
        if temporal_enabled:
            if args.temporal_group_size <= 1:
                raise ValueError("temporal losses require --temporal-group-size")
            if len(train_set) % args.temporal_group_size or len(validation_set) % args.temporal_group_size:
                raise ValueError("datasets must contain complete ordered temporal groups")
            loader_batch_size = args.temporal_group_size
            train_shuffle = False
        else:
            loader_batch_size = args.batch_size or cfg.batch_size
            train_shuffle = True
        train_loader = DataLoader(
            train_set,
            batch_size=loader_batch_size,
            shuffle=train_shuffle,
            num_workers=args.loader_workers,
            pin_memory=device.type == "cuda" and args.loader_workers > 0,
        )
        validation_loader = DataLoader(
            validation_set,
            batch_size=loader_batch_size,
            shuffle=False,
            num_workers=args.loader_workers,
            pin_memory=device.type == "cuda" and args.loader_workers > 0,
        )
        model = _model(
            args.output_mode,
            cfg.channels,
            in_channels,
            use_voltage_mlp=args.use_voltage_mlp,
            measurements=int(train["V"].shape[1]),
            voltage_latent=args.voltage_latent,
        ).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=cfg.learning_rate)
        best_state = copy.deepcopy(model.state_dict())
        best_score = float("inf")
        best_epoch = 0
        patience_limit = cfg.patience if args.patience is None else int(args.patience)
        patience = patience_limit
        iteration_history: list[dict] = []
        for epoch in range(epochs):
            train_loss = _run_epoch(
                model,
                train_loader,
                args.background_weight,
                args.dice_weight,
                args.hard_background_weight,
                args.hard_background_fraction,
                args.balanced_support_loss,
                args.correlation_weight,
                args.amplitude_weight,
                args.relative_amplitude_weight,
                args.temporal_delta_weight,
                args.temporal_correlation_weight,
                args.temporal_amplitude_weight,
                args.temporal_group_size,
                optimizer,
            )
            with torch.no_grad():
                validation_loss = _run_epoch(
                    model,
                    validation_loader,
                    args.background_weight,
                    args.dice_weight,
                    args.hard_background_weight,
                    args.hard_background_fraction,
                    args.balanced_support_loss,
                    args.correlation_weight,
                    args.amplitude_weight,
                    args.relative_amplitude_weight,
                    args.temporal_delta_weight,
                    args.temporal_correlation_weight,
                    args.temporal_amplitude_weight,
                    args.temporal_group_size,
                )
            score, score_metrics = _checkpoint_score(
                model,
                validation_loader,
                mode=args.checkpoint_mode,
                validation_loss=validation_loss,
                background_coefficient=args.checkpoint_background_coefficient,
                dice_coefficient=args.checkpoint_dice_coefficient,
            )
            record = {
                "epoch": epoch,
                "train_loss": train_loss,
                **score_metrics,
            }
            iteration_history.append(record)
            print(
                f"stage {iteration + 1} epoch {epoch}: train={train_loss:.6e} "
                f"validation={validation_loss:.6e} selection={score:.6e}",
                flush=True,
            )
            if score < best_score:
                best_score = score
                best_epoch = epoch
                best_state = copy.deepcopy(model.state_dict())
                patience = patience_limit
            else:
                patience -= 1
                if patience <= 0:
                    break
        model.load_state_dict(best_state)
        current_train = _predict(model, train_set, scale)
        current_validation = _predict(model, validation_set, scale)
        feature_order = ["current", "per_stage_lm_direction"]
        if args.use_coordinates:
            feature_order.extend(["x", "y", "radius"])
        if args.use_voltage_mlp:
            feature_order.append("global_voltage_mlp")
        checkpoint = {
            "state_dict": model.cpu().state_dict(),
            "channels": cfg.channels,
            "in_channels": in_channels,
            "scale": scale,
            "iteration": iteration,
            "feature_order": feature_order,
            "use_coordinates": args.use_coordinates,
            "use_voltage_mlp": args.use_voltage_mlp,
            "voltage_scale": voltage_scale,
            "voltage_latent": args.voltage_latent,
            "output_mode": args.output_mode,
            "loss_contract": {
                "positive_weight": args.positive_weight,
                "background_weight": args.background_weight,
                "dice_weight": args.dice_weight,
                "hard_background_weight": args.hard_background_weight,
                "hard_background_fraction": args.hard_background_fraction,
                "balanced_support_loss": args.balanced_support_loss,
                "correlation_weight": args.correlation_weight,
                "amplitude_weight": args.amplitude_weight,
                "relative_amplitude_weight": args.relative_amplitude_weight,
                "dynamic_reference": (
                    "sigma_resting" if args.physics_contract == "absolute" else None
                ),
                "temporal_delta_weight": args.temporal_delta_weight,
                "temporal_correlation_weight": args.temporal_correlation_weight,
                "temporal_amplitude_weight": args.temporal_amplitude_weight,
                "temporal_group_size": args.temporal_group_size,
                "dice_supervision": "clean synthetic targets only",
            },
            "physics": (
                f"{args.physics_mesh_mode} nonlinear absolute F/J/LM "
                "recomputed at every stage"
                if args.physics_contract == "absolute"
                else f"{args.physics_mesh_mode} nonlinear differential F/J/LM "
                "recomputed at every stage"
            ),
            "best_epoch": best_epoch,
            "best_selection_score": best_score,
            "physics_contract": {
                "baseline_mode": args.baseline_mode,
                "measurement_contract": args.physics_contract,
                "physics_mesh_mode": args.physics_mesh_mode,
                "forward_model": (
                    "F_f(P sigma_c) with chain-rule Jacobian J_f P"
                    if args.physics_mesh_mode == "projected_fine"
                    else "F_c(sigma_c) with directly computed coarse Jacobian J_c"
                ),
                "baseline_conductivity": args.baseline_conductivity,
                "hyper_pvi": cfg.hyper_pvi,
                "lambda_lm": cfg.lambda_lm,
                "lm_step_size": cfg.lm_step_size,
                "minimum_conductivity": args.minimum_conductivity,
                "num_elements": int(train["sigma"].shape[1]),
                "num_measurements": int(train["V"].shape[1]),
                "regularizer": "projected_laplacian",
                "voltage_weighting": "none",
                **artifact_hashes,
            },
        }
        checkpoint_path = args.models_dir / f"{args.model_name}_{iteration}.pt"
        torch.save(checkpoint, checkpoint_path)
        history.append(iteration_history)
        print(f"saved {checkpoint_path}", flush=True)

    report = {
        "method": f"faithful {args.physics_contract} PVI-GCNM",
        "target": (
            "true synthetic absolute conductivity"
            if args.physics_contract == "absolute"
            else "clean synthetic vascular delta conductivity"
        ),
        "pvi_images_used_as_labels": False,
        "per_stage_physics_recomputed": True,
        "model_name": args.model_name,
        "train_samples": int(len(train["sigma"])),
        "validation_samples": int(len(validation["sigma"])),
        "iterations": iterations,
        "epochs_requested": epochs,
        "patience": cfg.patience if args.patience is None else int(args.patience),
        "scale": scale,
        "output_mode": args.output_mode,
        "use_coordinates": args.use_coordinates,
        "use_voltage_mlp": args.use_voltage_mlp,
        "voltage_scale": voltage_scale,
        "voltage_latent": args.voltage_latent,
        "batch_size": int(args.batch_size or cfg.batch_size),
        "loader_workers": int(args.loader_workers),
        "validation_is_train_copy": bool(args.validation_is_train_copy),
        "positive_weight": args.positive_weight,
        "background_weight": args.background_weight,
        "dice_weight": args.dice_weight,
        "hard_background_weight": args.hard_background_weight,
        "hard_background_fraction": args.hard_background_fraction,
        "balanced_support_loss": args.balanced_support_loss,
        "correlation_weight": args.correlation_weight,
        "amplitude_weight": args.amplitude_weight,
        "relative_amplitude_weight": args.relative_amplitude_weight,
        "temporal_delta_weight": args.temporal_delta_weight,
        "temporal_correlation_weight": args.temporal_correlation_weight,
        "temporal_amplitude_weight": args.temporal_amplitude_weight,
        "temporal_group_size": args.temporal_group_size,
        "checkpoint_mode": args.checkpoint_mode,
        "random_seed": seed,
        "baseline_mode": args.baseline_mode,
        "physics_contract": args.physics_contract,
        "physics_mesh_mode": args.physics_mesh_mode,
        "baseline_conductivity": args.baseline_conductivity,
        "physics_stages": physics_reports,
        "artifact_hashes": artifact_hashes,
        "history": history,
        "elapsed_seconds": time.time() - start_time,
    }
    output = args.results_dir / f"{args.model_name}_training_report.json"
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
