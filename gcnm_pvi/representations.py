"""Inference adapters that expose stage-1/stage-2 mesh representations."""

from __future__ import annotations

import copy
import os
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import torch

from gcnm_pvi.anatomical_phantoms import element_positions
from gcnm_pvi.config import GcnmConfig
from gcnm_pvi.iterative_physics import (
    FixedZeroCurrentLMSolver,
    LowRankRegularizedSolver,
    dataset_lm_directions,
    dataset_voltage_residual_rms,
)
from gcnm_pvi.runtime import build_runtime
from gcnm_pvi.mesh_registry import sha256_file


def _validate_physics_hashes(contract: dict, cfg: GcnmConfig, config_path: Path) -> None:
    expected = {
        "config_sha256": sha256_file(config_path),
        "mesh_inverse_sha256": sha256_file(Path(cfg.mesh_inv_h5)),
        "mappings_sha256": sha256_file(Path(cfg.mappings_h5)),
    }
    for key, value in expected.items():
        if contract.get(key) != value:
            raise ValueError(f"checkpoint {key} does not match selected ring configuration")


def _predict_batched(
    model,
    graphs,
    scale: float,
    *,
    return_parameters: bool = False,
    batch_size: int | None = None,
):
    """Run graph inference in real GPU batches instead of one frame at a time."""
    from torch_geometric.loader import DataLoader

    if batch_size is None:
        batch_size = int(os.environ.get("GCNM_INFERENCE_BATCH_SIZE", "64"))
    if batch_size <= 0:
        raise ValueError("GCNM inference batch size must be positive")
    device = next(model.parameters()).device
    predictions, parameters = [], []
    loader = DataLoader(graphs, batch_size=batch_size, shuffle=False)
    model.eval()
    with torch.inference_mode():
        for batch in loader:
            batch = batch.to(device)
            output = model(batch, return_parameters=True) if return_parameters else model(batch)
            conductivity = output[0] if return_parameters else output
            predictions.append(
                conductivity.reshape(batch.num_graphs, -1).cpu().numpy() * float(scale)
            )
            if return_parameters:
                parameters.append(output[1].cpu().numpy())
    values = np.concatenate(predictions, axis=0)
    if return_parameters:
        return values, np.concatenate(parameters, axis=0)
    return values


def _stage_2_residual_indices(count: int) -> tuple[np.ndarray, int]:
    """Select deterministic frames for validation-only stage-2 residuals.

    Reconstruction is always performed for every frame.  The optional stride
    only reduces the extra forward FEM calls used to report the final-stage
    voltage residual; it cannot change either stage output.
    """

    stride = int(os.environ.get("GCNM_STAGE2_RESIDUAL_STRIDE", "1"))
    if stride <= 0:
        raise ValueError("GCNM stage-2 residual stride must be positive")
    indices = np.arange(0, count, stride, dtype=np.int64)
    if count and (len(indices) == 0 or indices[-1] != count - 1):
        indices = np.append(indices, count - 1)
    return indices, stride


class _ParallelPhysics:
    """Distribute independent frame physics over private mutable mesh objects."""

    def __init__(self, physics, workers: int | None = None) -> None:
        requested = int(os.environ.get("GCNM_PHYSICS_WORKERS", workers or 16))
        self.workers = max(1, requested)
        self.physics = [physics, *(copy.deepcopy(physics) for _ in range(self.workers - 1))]
        self.executor = ThreadPoolExecutor(max_workers=self.workers)

    def _slices(self, count: int) -> list[np.ndarray]:
        return [part for part in np.array_split(np.arange(count), min(count, self.workers)) if len(part)]

    def lm_directions(
        self,
        baseline: np.ndarray,
        current: np.ndarray,
        measured: np.ndarray,
        *,
        regularizer,
        hyper_pvi: float,
        lambda_lm: float,
        step_size: float,
        minimum_conductivity: float,
        baseline_voltages: np.ndarray,
        system_solver: LowRankRegularizedSolver | None,
    ):
        parts = self._slices(len(measured))
        futures = []
        for lane, indices in enumerate(parts):
            futures.append(
                self.executor.submit(
                    dataset_lm_directions,
                    self.physics[lane],
                    baseline[indices],
                    current[indices],
                    measured[indices],
                    regularizer=regularizer,
                    hyper_pvi=hyper_pvi,
                    lambda_lm=lambda_lm,
                    step_size=step_size,
                    minimum_conductivity=minimum_conductivity,
                    baseline_voltages=baseline_voltages[indices],
                    system_solver=system_solver,
                )
            )
        results = [future.result() for future in futures]
        return (
            np.concatenate([item[0] for item in results]),
            [diagnostic for item in results for diagnostic in item[1]],
            np.concatenate([item[2] for item in results]),
        )

    def residual_rms(
        self,
        baseline: np.ndarray,
        current: np.ndarray,
        measured: np.ndarray,
        *,
        minimum_conductivity: float,
        baseline_voltages: np.ndarray,
    ):
        parts = self._slices(len(measured))
        futures = []
        for lane, indices in enumerate(parts):
            futures.append(
                self.executor.submit(
                    dataset_voltage_residual_rms,
                    self.physics[lane],
                    baseline[indices],
                    current[indices],
                    measured[indices],
                    minimum_conductivity=minimum_conductivity,
                    baseline_voltages=baseline_voltages[indices],
                )
            )
        results = [future.result() for future in futures]
        return (
            np.concatenate([item[0] for item in results]),
            np.concatenate([item[1] for item in results]),
            sum(item[2] for item in results),
        )


class CoordinateReconstructor:
    def __init__(self, config: Path, checkpoint_dir: Path, model_name: str, device=None) -> None:
        from gcnm_pvi.train_faithful_gcnm import _model

        self.config_path = Path(config)
        self.cfg = GcnmConfig.from_yaml(config)
        self.runtime = build_runtime(self.cfg, include_forward=False)
        self.device = torch.device(device or ("cuda:0" if torch.cuda.is_available() else "cpu"))
        self.positions = element_positions(self.runtime["mesh_inv"]).astype(np.float32)
        self.checkpoints = []
        self.models = []
        for stage in range(2):
            checkpoint = torch.load(
                Path(checkpoint_dir) / f"{model_name}_{stage}.pt",
                map_location=self.device,
                weights_only=False,
            )
            if not checkpoint.get("use_coordinates", False):
                raise ValueError("coordinate family checkpoint does not enable coordinates")
            contract = checkpoint["physics_contract"]
            _validate_physics_hashes(contract, self.cfg, self.config_path)
            if contract.get("baseline_mode") != "homogeneous" or not np.isclose(
                float(contract.get("baseline_conductivity", np.nan)), 0.7
            ):
                raise ValueError("coordinate checkpoint does not use the 0.7 S/m homogeneous baseline")
            model = _model(
                checkpoint["output_mode"],
                checkpoint["channels"],
                int(checkpoint["in_channels"]),
                use_voltage_mlp=bool(checkpoint.get("use_voltage_mlp", False)),
                measurements=int(contract.get("num_measurements", 32)),
                voltage_latent=int(checkpoint.get("voltage_latent", 64)),
            ).to(self.device)
            model.load_state_dict(checkpoint["state_dict"])
            model.eval()
            self.checkpoints.append(checkpoint)
            self.models.append(model)
        self.baseline = np.full(self.mappings.num_elements, 0.7, dtype=np.float64)
        self.fixed_stage = FixedZeroCurrentLMSolver(
            self.runtime["physics_inv"],
            self.baseline,
            regularizer=self.mappings.laplace,
            hyper_pvi=self.cfg.hyper_pvi,
            lambda_lm=self.cfg.lambda_lm,
            step_size=self.cfg.lm_step_size,
        )
        self.nonlinear_solver = LowRankRegularizedSolver(
            self.mappings.laplace,
            self.mappings.num_elements,
            hyper_pvi=self.cfg.hyper_pvi,
            lambda_lm=self.cfg.lambda_lm,
        )
        self.parallel_physics = _ParallelPhysics(self.runtime["physics_inv"])

    @property
    def mappings(self):
        return self.runtime["mappings"]

    def reconstruct(self, voltage: np.ndarray) -> tuple[np.ndarray, np.ndarray, dict]:
        from gcnm_pvi.train_faithful_gcnm import _make_dataset

        measured = np.asarray(voltage, dtype=np.float64)
        elements = self.mappings.num_elements
        truth = np.zeros((len(measured), elements), dtype=np.float64)
        baseline = np.broadcast_to(self.baseline[None, :], truth.shape)
        zero = np.zeros_like(truth)
        direction_1, _, baseline_voltages = self.fixed_stage.solve_many(measured)
        checkpoint_1, checkpoint_2 = self.checkpoints
        graphs_1 = _make_dataset(
            truth, zero, direction_1, self.positions, self.runtime["edge_index"],
            scale=float(checkpoint_1["scale"]), use_coordinates=True, positive_weight=0.0,
            voltage=measured if checkpoint_1.get("use_voltage_mlp", False) else None,
            voltage_scale=float(checkpoint_1.get("voltage_scale", 1.0)),
        )
        stage_1 = _predict_batched(self.models[0], graphs_1, float(checkpoint_1["scale"]))
        direction_2, diagnostics_2, _ = self.parallel_physics.lm_directions(
            baseline, stage_1, measured,
            regularizer=self.mappings.laplace,
            hyper_pvi=self.cfg.hyper_pvi,
            lambda_lm=self.cfg.lambda_lm,
            step_size=self.cfg.lm_step_size,
            minimum_conductivity=float(
                checkpoint_2["physics_contract"].get("minimum_conductivity", 1e-4)
            ),
            baseline_voltages=baseline_voltages,
            system_solver=self.nonlinear_solver,
        )
        graphs_2 = _make_dataset(
            truth, stage_1, direction_2, self.positions, self.runtime["edge_index"],
            scale=float(checkpoint_2["scale"]), use_coordinates=True, positive_weight=0.0,
            voltage=measured if checkpoint_2.get("use_voltage_mlp", False) else None,
            voltage_scale=float(checkpoint_2.get("voltage_scale", 1.0)),
        )
        stage_2 = _predict_batched(self.models[1], graphs_2, float(checkpoint_2["scale"]))
        compute_residuals = os.environ.get(
            "GCNM_COMPUTE_STAGE2_RESIDUALS", "1"
        ).lower() not in {"0", "false", "no"}
        if compute_residuals:
            residual_indices, residual_stride = _stage_2_residual_indices(len(measured))
            residual_2, _, _ = self.parallel_physics.residual_rms(
                baseline[residual_indices],
                stage_2[residual_indices],
                measured[residual_indices],
                minimum_conductivity=float(
                    checkpoint_2["physics_contract"].get(
                        "minimum_conductivity", 1e-4
                    )
                ),
                baseline_voltages=baseline_voltages[residual_indices],
            )
        else:
            residual_2 = np.empty(0, dtype=np.float64)
            residual_stride = 0
        return stage_1, stage_2, {
            "stage_1_forward_voltage_rms": np.asarray(
                [item.voltage_residual_rms for item in diagnostics_2]
            ),
            "stage_2_forward_voltage_rms": np.asarray(residual_2),
            "stage_2_forward_voltage_rms_stride": residual_stride,
        }

    def reconstruct_reference(self, voltage: np.ndarray) -> tuple[np.ndarray, np.ndarray, dict]:
        """Original dense, frame-wise implementation used for parity tests."""
        from gcnm_pvi.train_faithful_gcnm import _make_dataset, _predict

        measured = np.asarray(voltage, dtype=np.float64)
        elements = self.mappings.num_elements
        truth = np.zeros((len(measured), elements), dtype=np.float64)
        baseline = np.full_like(truth, 0.7)
        current = np.zeros_like(truth)
        stages, residuals = [], []
        baseline_voltages = None
        for checkpoint, model in zip(self.checkpoints, self.models):
            direction, _, baseline_voltages = dataset_lm_directions(
                self.runtime["physics_inv"], baseline, current, measured,
                regularizer=self.mappings.laplace,
                hyper_pvi=self.cfg.hyper_pvi, lambda_lm=self.cfg.lambda_lm,
                step_size=self.cfg.lm_step_size,
                minimum_conductivity=float(checkpoint["physics_contract"].get("minimum_conductivity", 1e-4)),
                baseline_voltages=baseline_voltages,
            )
            graphs = _make_dataset(
                truth, current, direction, self.positions, self.runtime["edge_index"],
                scale=float(checkpoint["scale"]), use_coordinates=True, positive_weight=0.0,
                voltage=measured if checkpoint.get("use_voltage_mlp", False) else None,
                voltage_scale=float(checkpoint.get("voltage_scale", 1.0)),
            )
            current = _predict(model, graphs, float(checkpoint["scale"]))
            stage_residual, baseline_voltages, _ = dataset_voltage_residual_rms(
                self.runtime["physics_inv"], baseline, current, measured,
                minimum_conductivity=float(checkpoint["physics_contract"].get("minimum_conductivity", 1e-4)),
                baseline_voltages=baseline_voltages,
            )
            stages.append(current.copy())
            residuals.append(stage_residual)
        return stages[0], stages[1], {
            "stage_1_forward_voltage_rms": np.asarray(residuals[0]),
            "stage_2_forward_voltage_rms": np.asarray(residuals[1]),
        }


class GlobalVoltageVesselSlotReconstructor:
    """Signed two-stage graph reconstruction with whole-beat voltage context.

    The returned arrays are the literal learned stages.  The voltage-residual
    comparison is reported as a diagnostic and never replaces ``stage_2`` with
    ``stage_1`` in the exported ``s2`` column.
    """

    def __init__(
        self,
        config: Path,
        localizer_checkpoint: Path,
        refiner_checkpoint: Path,
        device=None,
    ) -> None:
        from gcnm_pvi.generalized_vessel_model import (
            BeatMeanPoolVesselLocalizer,
            BeatVesselParameterRefiner,
        )

        self.config_path = Path(config)
        self.cfg = GcnmConfig.from_yaml(config)
        self.runtime = build_runtime(self.cfg, include_forward=False)
        self.device = torch.device(device or ("cuda:0" if torch.cuda.is_available() else "cpu"))
        local = torch.load(localizer_checkpoint, map_location="cpu", weights_only=False)
        refine = torch.load(refiner_checkpoint, map_location="cpu", weights_only=False)
        if local["contract"] != refine["contract"]:
            raise ValueError("global vessel-slot localizer/refiner contracts differ")
        self.contract = local["contract"]
        _validate_physics_hashes(self.contract, self.cfg, self.config_path)
        architecture = self.contract.get("architecture")
        if architecture != "beat_voltage_slots":
            raise ValueError(
                f"checkpoint architecture is {architecture}, not beat_voltage_slots"
            )
        if not self.contract.get("beat_context_required", False):
            raise ValueError("global vessel-slot checkpoint does not require beat context")
        if not self.contract.get("signed_amplitude", False):
            raise ValueError("global vessel-slot checkpoint does not support signed amplitude")
        if self.contract.get("baseline_mode") != "homogeneous" or not np.isclose(
            float(self.contract.get("baseline_conductivity", np.nan)), 0.7
        ):
            raise ValueError("global vessel-slot checkpoint does not use homogeneous 0.7 S/m")
        minimum = float(self.contract.get("minimum_vessel_axis", 0.05))
        maximum = float(self.contract.get("maximum_vessel_axis", 0.23))
        self.localizer = BeatMeanPoolVesselLocalizer(
            minimum_axis=minimum, maximum_axis=maximum
        ).to(self.device)
        self.refiner = BeatVesselParameterRefiner(
            diffusion=False,
            minimum_axis=minimum,
            maximum_axis=maximum,
        ).to(self.device)
        self.localizer.load_state_dict(local["state_dict"])
        self.refiner.load_state_dict(refine["state_dict"])
        self.localizer.eval()
        self.refiner.eval()
        self.baseline = np.full(
            self.runtime["mappings"].num_elements, 0.7, dtype=np.float64
        )
        self.positions = element_positions(self.runtime["mesh_inv"]).astype(np.float32)
        self.fixed_stage = FixedZeroCurrentLMSolver(
            self.runtime["physics_inv"],
            self.baseline,
            regularizer=self.mappings.laplace,
            hyper_pvi=self.cfg.hyper_pvi,
            lambda_lm=self.cfg.lambda_lm,
            step_size=self.cfg.lm_step_size,
        )
        self.nonlinear_solver = LowRankRegularizedSolver(
            self.mappings.laplace,
            self.mappings.num_elements,
            hyper_pvi=self.cfg.hyper_pvi,
            lambda_lm=self.cfg.lambda_lm,
        )
        self.parallel_physics = _ParallelPhysics(self.runtime["physics_inv"])
        self.requires_beat_context = True

    @property
    def mappings(self):
        return self.runtime["mappings"]

    def reconstruct(
        self,
        voltage: np.ndarray,
        voltage_template: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray, dict]:
        from gcnm_pvi.train_voltage_vessel_gcnm import _graphs

        total_started = time.perf_counter()
        measured = np.asarray(voltage, dtype=np.float64)
        if voltage_template is None:
            raise ValueError("global vessel-slot inference requires one beat template per frame")
        templates = np.asarray(voltage_template, dtype=np.float64)
        if templates.shape != measured.shape:
            raise ValueError(
                f"beat templates have shape {templates.shape}, expected {measured.shape}"
            )
        count, elements = len(measured), self.mappings.num_elements
        baseline = np.broadcast_to(self.baseline[None, :], (count, elements))
        zero = np.zeros((count, elements), dtype=np.float64)
        dummy = np.zeros((count, 2, 6), dtype=np.float32)
        started = time.perf_counter()
        direction_1, _, baseline_voltages = self.fixed_stage.solve_many(measured)
        context_direction, _, _ = self.fixed_stage.solve_many(templates)
        fixed_seconds = time.perf_counter() - started
        started = time.perf_counter()
        stage_1_graphs = _graphs(
            zero,
            zero,
            direction_1,
            measured,
            dummy,
            self.positions,
            self.runtime["edge_index"],
            conductivity_scale=float(self.contract["conductivity_scale"]),
            voltage_scale=float(self.contract["voltage_scale"]),
            voltage_template=templates,
            context_direction=context_direction,
            voltage_clean=measured,
            voltage_input_mode="beat_normalized",
            voltage_rms_reference=float(self.contract["voltage_rms_reference"]),
        )
        stage_1_graph_seconds = time.perf_counter() - started
        started = time.perf_counter()
        stage_1, stage_1_parameters = _predict_batched(
            self.localizer,
            stage_1_graphs,
            float(self.contract["conductivity_scale"]),
            return_parameters=True,
        )
        stage_1_model_seconds = time.perf_counter() - started
        started = time.perf_counter()
        direction_2, diagnostics_2, _ = self.parallel_physics.lm_directions(
            baseline,
            stage_1,
            measured,
            regularizer=self.mappings.laplace,
            hyper_pvi=self.cfg.hyper_pvi,
            lambda_lm=self.cfg.lambda_lm,
            step_size=self.cfg.lm_step_size,
            minimum_conductivity=1e-4,
            baseline_voltages=baseline_voltages,
            system_solver=self.nonlinear_solver,
        )
        nonlinear_physics_seconds = time.perf_counter() - started
        started = time.perf_counter()
        stage_2_graphs = _graphs(
            zero,
            stage_1,
            direction_2,
            measured,
            dummy,
            self.positions,
            self.runtime["edge_index"],
            conductivity_scale=float(self.contract["conductivity_scale"]),
            voltage_scale=float(self.contract["voltage_scale"]),
            initial_parameters=stage_1_parameters,
            voltage_template=templates,
            context_direction=context_direction,
            voltage_clean=measured,
            voltage_input_mode="beat_normalized",
            voltage_rms_reference=float(self.contract["voltage_rms_reference"]),
        )
        stage_2_graph_seconds = time.perf_counter() - started
        started = time.perf_counter()
        stage_2, _stage_2_parameters = _predict_batched(
            self.refiner,
            stage_2_graphs,
            float(self.contract["conductivity_scale"]),
            return_parameters=True,
        )
        stage_2_model_seconds = time.perf_counter() - started
        residual_1 = np.asarray(
            [item.voltage_residual_rms for item in diagnostics_2], dtype=np.float64
        )
        compute_residuals = os.environ.get(
            "GCNM_COMPUTE_STAGE2_RESIDUALS", "1"
        ).lower() not in {"0", "false", "no"}
        if compute_residuals:
            started = time.perf_counter()
            residual_indices, residual_stride = _stage_2_residual_indices(count)
            residual_2, _, _ = self.parallel_physics.residual_rms(
                baseline[residual_indices],
                stage_2[residual_indices],
                measured[residual_indices],
                minimum_conductivity=1e-4,
                baseline_voltages=baseline_voltages[residual_indices],
            )
            lower_fraction = float(
                np.mean(residual_2 <= residual_1[residual_indices])
            )
            validation_seconds = time.perf_counter() - started
        else:
            residual_2 = np.empty(0, dtype=np.float64)
            residual_stride = 0
            lower_fraction = None
            validation_seconds = 0.0
        return stage_1, stage_2, {
            "stage_1_forward_voltage_rms": residual_1,
            "stage_2_forward_voltage_rms": np.asarray(residual_2),
            "stage_2_forward_voltage_rms_stride": residual_stride,
            "stage_2_lower_residual_fraction": lower_fraction,
            "timings_seconds": {
                "fixed_stage_features": fixed_seconds,
                "stage_1_graph_build": stage_1_graph_seconds,
                "stage_1_model_cuda": stage_1_model_seconds,
                "nonlinear_fem_jacobian_lm": nonlinear_physics_seconds,
                "stage_2_graph_build": stage_2_graph_seconds,
                "stage_2_model_cuda": stage_2_model_seconds,
                "post_stage_2_validation": validation_seconds,
                "total": time.perf_counter() - total_started,
            },
        }


class DiffusionSlotReconstructor:
    def __init__(
        self,
        config: Path,
        localizer_checkpoint: Path,
        refiner_checkpoint: Path,
        anatomical_prior: Path,
        device=None,
    ) -> None:
        from gcnm_pvi.voltage_vessel_model import (
            SpatialAttentionVesselDiffusionLocalizer,
            SpatialVesselDiffusionRefiner,
        )

        self.config_path = Path(config)
        self.cfg = GcnmConfig.from_yaml(config)
        self.runtime = build_runtime(self.cfg, include_forward=False)
        self.device = torch.device(device or ("cuda:0" if torch.cuda.is_available() else "cpu"))
        local = torch.load(localizer_checkpoint, map_location="cpu", weights_only=False)
        refine = torch.load(refiner_checkpoint, map_location="cpu", weights_only=False)
        if local["contract"] != refine["contract"]:
            raise ValueError("diffusion localizer/refiner contracts differ")
        self.contract = local["contract"]
        _validate_physics_hashes(self.contract, self.cfg, self.config_path)
        if self.contract.get("architecture") != "diffusion_slots":
            raise ValueError("checkpoint architecture is not diffusion_slots")
        minimum = float(self.contract.get("minimum_vessel_axis", 0.025))
        maximum = float(self.contract.get("maximum_vessel_axis", 0.23))
        self.localizer = SpatialAttentionVesselDiffusionLocalizer(
            minimum_axis=minimum, maximum_axis=maximum
        ).to(self.device)
        self.refiner = SpatialVesselDiffusionRefiner(
            minimum_axis=minimum, maximum_axis=maximum
        ).to(self.device)
        self.localizer.load_state_dict(local["state_dict"])
        self.refiner.load_state_dict(refine["state_dict"])
        self.localizer.eval()
        self.refiner.eval()
        with np.load(anatomical_prior) as prior:
            baseline = np.asarray(prior["sigma_baseline"], dtype=np.float64)
            labels = np.asarray(prior["tissue_labels"], dtype=np.uint8) if "tissue_labels" in prior else None
        self.baseline = baseline[0] if baseline.ndim == 2 else baseline
        self.tissue_labels = (
            labels[0] if labels is not None and labels.ndim == 2 else labels
        )
        if self.tissue_labels is None:
            self.tissue_labels = np.zeros_like(self.baseline, dtype=np.uint8)
            self.tissue_labels[np.isclose(self.baseline, 0.352, rtol=0.05, atol=0.005)] = 3
        if len(self.baseline) != self.runtime["mappings"].num_elements:
            raise ValueError("anatomical prior element count does not match ring mesh")
        self.positions = element_positions(self.runtime["mesh_inv"]).astype(np.float32)
        self.fixed_stage = FixedZeroCurrentLMSolver(
            self.runtime["physics_inv"],
            self.baseline,
            regularizer=self.mappings.laplace,
            hyper_pvi=self.cfg.hyper_pvi,
            lambda_lm=self.cfg.lambda_lm,
            step_size=self.cfg.lm_step_size,
        )
        self.nonlinear_solver = LowRankRegularizedSolver(
            self.mappings.laplace,
            self.mappings.num_elements,
            hyper_pvi=self.cfg.hyper_pvi,
            lambda_lm=self.cfg.lambda_lm,
        )
        self.parallel_physics = _ParallelPhysics(self.runtime["physics_inv"])

    @property
    def mappings(self):
        return self.runtime["mappings"]

    def reconstruct(self, voltage: np.ndarray) -> tuple[np.ndarray, np.ndarray, dict]:
        from gcnm_pvi.train_voltage_vessel_gcnm import _graphs

        measured = np.asarray(voltage, dtype=np.float64)
        count, elements = len(measured), self.mappings.num_elements
        baseline = np.broadcast_to(self.baseline[None, :], (count, elements)).copy()
        labels = np.broadcast_to(self.tissue_labels[None, :], (count, elements))
        zero = np.zeros_like(baseline)
        dummy = np.zeros((count, 2, 8), dtype=np.float32)
        direction_1, _, baseline_voltages = self.fixed_stage.solve_many(measured)
        graphs_1 = _graphs(
            zero, zero, direction_1, measured, dummy, self.positions, self.runtime["edge_index"],
            conductivity_scale=float(self.contract["conductivity_scale"]),
            voltage_scale=float(self.contract["voltage_scale"]), tissue_labels=labels,
            voltage_clean=measured,
        )
        stage_1, parameters = _predict_batched(
            self.localizer, graphs_1, float(self.contract["conductivity_scale"]),
            return_parameters=True,
        )
        direction_2, diagnostics_2, _ = self.parallel_physics.lm_directions(
            baseline, stage_1, measured,
            regularizer=self.mappings.laplace, hyper_pvi=self.cfg.hyper_pvi,
            lambda_lm=self.cfg.lambda_lm, step_size=self.cfg.lm_step_size,
            minimum_conductivity=1e-4,
            baseline_voltages=baseline_voltages,
            system_solver=None,
        )
        graphs_2 = _graphs(
            zero, stage_1, direction_2, measured, dummy, self.positions, self.runtime["edge_index"],
            conductivity_scale=float(self.contract["conductivity_scale"]),
            voltage_scale=float(self.contract["voltage_scale"]), initial_parameters=parameters,
            tissue_labels=labels, voltage_clean=measured,
        )
        stage_2, _ = _predict_batched(
            self.refiner, graphs_2, float(self.contract["conductivity_scale"]),
            return_parameters=True,
        )
        residual_2, _, _ = self.parallel_physics.residual_rms(
            baseline, stage_2, measured,
            minimum_conductivity=1e-4,
            baseline_voltages=baseline_voltages,
        )
        return stage_1, stage_2, {
            "stage_1_forward_voltage_rms": np.asarray(
                [item.voltage_residual_rms for item in diagnostics_2]
            ),
            "stage_2_forward_voltage_rms": np.asarray(residual_2),
        }

    def reconstruct_reference(self, voltage: np.ndarray) -> tuple[np.ndarray, np.ndarray, dict]:
        """Original dense, frame-wise implementation used for parity tests."""
        from gcnm_pvi.train_voltage_vessel_gcnm import _graphs, _predict_with_parameters

        measured = np.asarray(voltage, dtype=np.float64)
        count, elements = len(measured), self.mappings.num_elements
        baseline = np.broadcast_to(self.baseline[None, :], (count, elements)).copy()
        labels = np.broadcast_to(self.tissue_labels[None, :], (count, elements))
        zero = np.zeros_like(baseline)
        dummy = np.zeros((count, 2, 8), dtype=np.float32)
        direction_1, _, baseline_voltages = dataset_lm_directions(
            self.runtime["physics_inv"], baseline, zero, measured,
            regularizer=self.mappings.laplace, hyper_pvi=self.cfg.hyper_pvi,
            lambda_lm=self.cfg.lambda_lm, step_size=self.cfg.lm_step_size,
        )
        graphs_1 = _graphs(
            zero, zero, direction_1, measured, dummy, self.positions,
            self.runtime["edge_index"],
            conductivity_scale=float(self.contract["conductivity_scale"]),
            voltage_scale=float(self.contract["voltage_scale"]),
            tissue_labels=labels, voltage_clean=measured,
        )
        stage_1, parameters = _predict_with_parameters(
            self.localizer, graphs_1, float(self.contract["conductivity_scale"])
        )
        residual_1, _, _ = dataset_voltage_residual_rms(
            self.runtime["physics_inv"], baseline, stage_1, measured,
            baseline_voltages=baseline_voltages,
        )
        direction_2, _, _ = dataset_lm_directions(
            self.runtime["physics_inv"], baseline, stage_1, measured,
            regularizer=self.mappings.laplace, hyper_pvi=self.cfg.hyper_pvi,
            lambda_lm=self.cfg.lambda_lm, step_size=self.cfg.lm_step_size,
            baseline_voltages=baseline_voltages,
        )
        graphs_2 = _graphs(
            zero, stage_1, direction_2, measured, dummy, self.positions,
            self.runtime["edge_index"],
            conductivity_scale=float(self.contract["conductivity_scale"]),
            voltage_scale=float(self.contract["voltage_scale"]),
            initial_parameters=parameters, tissue_labels=labels, voltage_clean=measured,
        )
        stage_2, _ = _predict_with_parameters(
            self.refiner, graphs_2, float(self.contract["conductivity_scale"])
        )
        residual_2, _, _ = dataset_voltage_residual_rms(
            self.runtime["physics_inv"], baseline, stage_2, measured,
            baseline_voltages=baseline_voltages,
        )
        return stage_1, stage_2, {
            "stage_1_forward_voltage_rms": np.asarray(residual_1),
            "stage_2_forward_voltage_rms": np.asarray(residual_2),
        }


def rasterize_frames(mappings, elements: np.ndarray) -> np.ndarray:
    """Map ``(frames,elements)`` to pvi_ml's ``(40,40,frames)`` layout."""
    flat = mappings.elem_to_image(np.asarray(elements).T)
    return flat.reshape(40, 40, flat.shape[1], order="F").astype(np.float32)
