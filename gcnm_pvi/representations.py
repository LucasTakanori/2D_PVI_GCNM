"""Inference adapters that expose stage-1/stage-2 mesh representations."""

from __future__ import annotations

import copy
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import torch

from gcnm_pvi.anatomical_phantoms import element_positions
from gcnm_pvi.config import GcnmConfig
from gcnm_pvi.coordinate_runtime import (
    build_coordinate_model,
    make_coordinate_graphs,
    predict_coordinate_graphs,
)
from gcnm_pvi.iterative_physics import (
    FixedZeroCurrentLMSolver,
    LowRankRegularizedSolver,
    dataset_lm_directions,
    dataset_voltage_residual_rms,
)
from gcnm_pvi.runtime import build_runtime
from gcnm_pvi.mesh_registry import sha256_file
from gcnm_pvi.dual_mesh_physics import ProjectedFineMeshPhysics
from gcnm_pvi.vessel_runtime import (
    make_vessel_graphs,
    predict_vessel_graphs_with_parameters,
)


def _validate_physics_hashes(
    contract: dict,
    cfg: GcnmConfig,
    config_path: Path,
    *,
    allow_config_hash_mismatch: bool = False,
    effective_physics_mesh_mode: str | None = None,
    allow_physics_mode_mismatch: bool = False,
) -> None:
    expected = {
        "config_sha256": sha256_file(config_path),
        "mesh_inverse_sha256": sha256_file(Path(cfg.mesh_inv_h5)),
        "mappings_sha256": sha256_file(Path(cfg.mappings_h5)),
    }
    trained_mode = str(contract.get("physics_mesh_mode", "coarse"))
    effective_mode = effective_physics_mesh_mode or trained_mode
    deliberate_legacy_override = (
        allow_physics_mode_mismatch
        and trained_mode != effective_mode
        and trained_mode == "coarse"
    )
    if effective_mode == "projected_fine" and not deliberate_legacy_override:
        expected["mesh_forward_sha256"] = sha256_file(Path(cfg.mesh_fwd_h5))
    for key, value in expected.items():
        if key == "config_sha256" and allow_config_hash_mismatch:
            continue
        if contract.get(key) != value:
            raise ValueError(f"checkpoint {key} does not match selected ring configuration")
    if allow_config_hash_mismatch:
        critical = {
            "hyper_pvi": float(cfg.hyper_pvi),
            "lambda_lm": float(cfg.lambda_lm),
        }
        for key in ("hyper_pvi", "lambda_lm"):
            if not np.isclose(float(contract.get(key, np.nan)), critical[key]):
                raise ValueError(
                    f"checkpoint {key} differs from the current ring configuration"
                )
        if cfg.connectivity != "node":
            raise ValueError("coordinate checkpoints require node-sharing graph connectivity")


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


def _stage_2_residual_indices(
    count: int, stride: int | None = None
) -> tuple[np.ndarray, int]:
    """Select deterministic frames for validation-only stage-2 residuals.

    Reconstruction is always performed for every frame.  The optional stride
    only reduces the extra forward FEM calls used to report the final-stage
    voltage residual; it cannot change either stage output.
    """

    if stride is None:
        stride = int(os.environ.get("GCNM_STAGE2_RESIDUAL_STRIDE", "1"))
    else:
        stride = int(stride)
    if stride <= 0:
        raise ValueError("GCNM stage-2 residual stride must be positive")
    indices = np.arange(0, count, stride, dtype=np.int64)
    if count and (len(indices) == 0 or indices[-1] != count - 1):
        indices = np.append(indices, count - 1)
    return indices, stride


def _available_cpu_workers() -> int:
    """Return the process CPU allowance, honoring Slurm when it is narrower."""

    candidates: list[int] = []
    try:
        candidates.append(len(os.sched_getaffinity(0)))
    except (AttributeError, OSError):
        candidates.append(os.cpu_count() or 1)
    slurm_cpus = os.environ.get("SLURM_CPUS_PER_TASK")
    if slurm_cpus is not None:
        try:
            value = int(slurm_cpus)
        except ValueError as exc:
            raise ValueError("SLURM_CPUS_PER_TASK must be a positive integer") from exc
        if value <= 0:
            raise ValueError("SLURM_CPUS_PER_TASK must be a positive integer")
        candidates.append(value)
    return max(1, min(candidates))


def _positive_integer(value: int, label: str) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must be a positive integer") from exc
    if parsed <= 0:
        raise ValueError(f"{label} must be a positive integer")
    return parsed


def _resolve_physics_mesh_mode(
    checkpoints: list[dict],
    requested: str | None,
    *,
    allow_mismatch: bool = False,
) -> str:
    """Infer physics mode from both stages and reject semantic mismatches."""

    checkpoint_modes = [
        str(item["physics_contract"].get("physics_mesh_mode", "coarse"))
        for item in checkpoints
    ]
    if checkpoint_modes[0] != checkpoint_modes[1]:
        raise ValueError(
            "stage checkpoints disagree on physics_mesh_mode: "
            f"{checkpoint_modes[0]!r} versus {checkpoint_modes[1]!r}"
        )
    trained = checkpoint_modes[0]
    if trained not in {"coarse", "projected_fine"}:
        raise ValueError(
            "checkpoint physics_mesh_mode must be 'coarse' or 'projected_fine'"
        )
    if requested is None:
        return trained
    if requested not in {"coarse", "projected_fine"}:
        raise ValueError("physics_mesh_mode must be 'coarse' or 'projected_fine'")
    if requested != trained and not allow_mismatch:
        raise ValueError(
            f"checkpoint was trained with {trained!r} physics, "
            f"but {requested!r} was requested"
        )
    return requested


class _ParallelPhysics:
    """Distribute independent frame physics over private mutable mesh objects."""

    def __init__(self, physics, workers: int | None = None) -> None:
        if workers is None:
            requested = _positive_integer(
                os.environ.get(
                    "GCNM_PHYSICS_WORKERS", min(16, _available_cpu_workers())
                ),
                "GCNM physics workers",
            )
        else:
            requested = _positive_integer(workers, "GCNM physics workers")
        # A Slurm allocation (or the process CPU affinity outside Slurm) is a
        # hard safety boundary.  A larger request is intentionally capped.
        self.workers = min(requested, _available_cpu_workers())
        self.physics = [physics, *(copy.deepcopy(physics) for _ in range(self.workers - 1))]
        self.executor = ThreadPoolExecutor(
            max_workers=self.workers, thread_name_prefix="gcnm-physics"
        )
        self._operation_lock = threading.Lock()
        self._closed = False

    def active_workers(self, count: int, workers: int | None = None) -> int:
        requested = self.workers
        if workers is not None:
            requested = _positive_integer(workers, "physics_workers")
            if requested > self.workers:
                raise ValueError(
                    f"physics_workers={requested} exceeds the resident pool size "
                    f"of {self.workers}; set physics_workers when constructing "
                    "CoordinateReconstructor"
                )
        return min(max(int(count), 1), requested)

    def _slices(
        self, count: int, workers: int | None = None
    ) -> list[np.ndarray]:
        lanes = self.active_workers(count, workers)
        return [
            part
            for part in np.array_split(np.arange(count), lanes)
            if len(part)
        ]

    def close(self) -> None:
        """Release resident worker threads after all current physics completes."""

        with self._operation_lock:
            if not self._closed:
                self.executor.shutdown(wait=True)
                self._closed = True

    def _ensure_open(self) -> None:
        if self._closed:
            raise RuntimeError("the GCNM physics worker pool is closed")

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
        workers: int | None = None,
    ):
        with self._operation_lock:
            self._ensure_open()
            parts = self._slices(len(measured), workers)
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
        workers: int | None = None,
    ):
        with self._operation_lock:
            self._ensure_open()
            parts = self._slices(len(measured), workers)
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
    def __init__(
        self,
        config: Path,
        checkpoint_dir: Path,
        model_name: str,
        device=None,
        *,
        physics_mesh_mode: str | None = None,
        allow_config_hash_mismatch: bool = False,
        allow_physics_mode_mismatch: bool = False,
        physics_workers: int | None = None,
        forward_backend: str = "dense",
    ) -> None:
        self.config_path = Path(config)
        self.cfg = GcnmConfig.from_yaml(config)
        self.device = torch.device(device or ("cuda:0" if torch.cuda.is_available() else "cpu"))
        loaded_checkpoints = [
            torch.load(
                Path(checkpoint_dir) / f"{model_name}_{stage}.pt",
                map_location=self.device,
                weights_only=False,
            )
            for stage in range(2)
        ]
        physics_mesh_mode = _resolve_physics_mesh_mode(
            loaded_checkpoints,
            physics_mesh_mode,
            allow_mismatch=allow_physics_mode_mismatch,
        )
        self.checkpoint_physics_mesh_mode = str(
            loaded_checkpoints[0]["physics_contract"].get(
                "physics_mesh_mode", "coarse"
            )
        )
        self.physics_mode_override = (
            physics_mesh_mode != self.checkpoint_physics_mesh_mode
        )
        self.physics_mesh_mode = physics_mesh_mode
        self.forward_backend = forward_backend
        self.runtime = build_runtime(
            self.cfg,
            include_forward=physics_mesh_mode == "projected_fine",
            forward_backend=forward_backend,
        )
        self.positions = element_positions(self.runtime["mesh_inv"]).astype(np.float32)
        self.checkpoints = []
        self.models = []
        for checkpoint in loaded_checkpoints:
            if not checkpoint.get("use_coordinates", False):
                raise ValueError("coordinate family checkpoint does not enable coordinates")
            contract = checkpoint["physics_contract"]
            _validate_physics_hashes(
                contract,
                self.cfg,
                self.config_path,
                allow_config_hash_mismatch=allow_config_hash_mismatch,
                effective_physics_mesh_mode=physics_mesh_mode,
                allow_physics_mode_mismatch=allow_physics_mode_mismatch,
            )
            if contract.get("baseline_mode") != "homogeneous" or not np.isclose(
                float(contract.get("baseline_conductivity", np.nan)), 0.7
            ):
                raise ValueError("coordinate checkpoint does not use the 0.7 S/m homogeneous baseline")
            model = build_coordinate_model(
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
        if physics_mesh_mode == "projected_fine":
            if self.mappings.c2f is None:
                raise ValueError(
                    "projected-fine reconstruction requires a coarse-to-fine mapping"
                )
            stage_physics = ProjectedFineMeshPhysics(
                self.runtime["physics_fwd"], self.mappings.c2f
            )
        else:
            stage_physics = self.runtime["physics_inv"]
        self.stage_physics = stage_physics
        self.fixed_stage = FixedZeroCurrentLMSolver(
            stage_physics,
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
        self.parallel_physics = _ParallelPhysics(stage_physics, workers=physics_workers)
        # The FEM objects are mutable.  Serialize calls on one resident
        # reconstructor while still parallelizing independent frames inside a
        # call across its private physics lanes.
        self._reconstruction_lock = threading.Lock()

    @property
    def mappings(self):
        return self.runtime["mappings"]

    def close(self) -> None:
        """Release the resident physics worker pool."""

        with self._reconstruction_lock:
            self.parallel_physics.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()

    def _validate_voltage_batch(self, voltage: np.ndarray) -> np.ndarray:
        measured = np.asarray(voltage, dtype=np.float64)
        if measured.ndim != 2:
            raise ValueError(
                "voltage batch must have shape (frames, measurements)"
            )
        if measured.shape[0] == 0:
            raise ValueError("voltage batch must contain at least one frame")
        expected = [
            int(item["physics_contract"].get("num_measurements", 32))
            for item in self.checkpoints
        ]
        if expected[0] != expected[1]:
            raise ValueError("stage checkpoints disagree on the measurement count")
        if measured.shape[1] != expected[0]:
            raise ValueError(
                f"voltage batch has {measured.shape[1]} measurements per frame; "
                f"checkpoint requires {expected[0]}"
            )
        if not np.all(np.isfinite(measured)):
            raise ValueError("voltage batch contains non-finite values")
        return measured

    def reconstruct(
        self, voltage: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, dict]:
        """Reconstruct a frame batch using environment-configured controls.

        This remains the backward-compatible entrypoint.  Online callers that
        need explicit controls should use :meth:`reconstruct_batch`.
        """

        return self.reconstruct_batch(voltage)

    def reconstruct_batch(
        self,
        voltage: np.ndarray,
        *,
        model_batch_size: int | None = None,
        physics_workers: int | None = None,
        compute_stage2_residuals: bool | None = None,
        stage2_residual_stride: int | None = None,
    ) -> tuple[np.ndarray, np.ndarray, dict]:
        """Reconstruct an ordered batch, including a one-second 50-frame batch.

        Models, graph topology, mappings, and FEM objects stay resident on this
        instance.  Learned graph stages are evaluated in real PyG batches and
        nonlinear frame physics uses at most ``physics_workers`` private lanes.
        The returned arrays and diagnostic indices retain input frame order.
        """

        measured = self._validate_voltage_batch(voltage)
        if model_batch_size is None:
            model_batch_size = _positive_integer(
                os.environ.get("GCNM_INFERENCE_BATCH_SIZE", "64"),
                "model_batch_size",
            )
        else:
            model_batch_size = _positive_integer(
                model_batch_size, "model_batch_size"
            )
        active_physics_workers = self.parallel_physics.active_workers(
            len(measured), physics_workers
        )
        if compute_stage2_residuals is None:
            compute_stage2_residuals = os.environ.get(
                "GCNM_COMPUTE_STAGE2_RESIDUALS", "1"
            ).lower() not in {"0", "false", "no"}
        if not isinstance(compute_stage2_residuals, (bool, np.bool_)):
            raise TypeError("compute_stage2_residuals must be a boolean")
        if stage2_residual_stride is not None:
            stage2_residual_stride = _positive_integer(
                stage2_residual_stride, "stage2_residual_stride"
            )

        with self._reconstruction_lock:
            return self._reconstruct_batch_unlocked(
                measured,
                model_batch_size=model_batch_size,
                physics_workers=physics_workers,
                active_physics_workers=active_physics_workers,
                compute_stage2_residuals=bool(compute_stage2_residuals),
                stage2_residual_stride=stage2_residual_stride,
            )

    def _reconstruct_batch_unlocked(
        self,
        measured: np.ndarray,
        *,
        model_batch_size: int,
        physics_workers: int | None,
        active_physics_workers: int,
        compute_stage2_residuals: bool,
        stage2_residual_stride: int | None,
    ) -> tuple[np.ndarray, np.ndarray, dict]:
        total_started = time.perf_counter()
        elements = self.mappings.num_elements
        truth = np.zeros((len(measured), elements), dtype=np.float64)
        baseline = np.broadcast_to(self.baseline[None, :], truth.shape)
        zero = np.zeros_like(truth)
        started = time.perf_counter()
        direction_1, _, baseline_voltages = self.fixed_stage.solve_many(measured)
        fixed_stage_seconds = time.perf_counter() - started
        checkpoint_1, checkpoint_2 = self.checkpoints
        started = time.perf_counter()
        graphs_1 = make_coordinate_graphs(
            truth, zero, direction_1, self.positions, self.runtime["edge_index"],
            scale=float(checkpoint_1["scale"]), use_coordinates=True, positive_weight=0.0,
            voltage=measured if checkpoint_1.get("use_voltage_mlp", False) else None,
            voltage_scale=float(checkpoint_1.get("voltage_scale", 1.0)),
        )
        stage_1_graph_seconds = time.perf_counter() - started
        started = time.perf_counter()
        stage_1 = _predict_batched(
            self.models[0], graphs_1, float(checkpoint_1["scale"]),
            batch_size=model_batch_size,
        )
        stage_1_model_seconds = time.perf_counter() - started
        started = time.perf_counter()
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
            workers=physics_workers,
        )
        nonlinear_physics_seconds = time.perf_counter() - started
        started = time.perf_counter()
        graphs_2 = make_coordinate_graphs(
            truth, stage_1, direction_2, self.positions, self.runtime["edge_index"],
            scale=float(checkpoint_2["scale"]), use_coordinates=True, positive_weight=0.0,
            voltage=measured if checkpoint_2.get("use_voltage_mlp", False) else None,
            voltage_scale=float(checkpoint_2.get("voltage_scale", 1.0)),
        )
        stage_2_graph_seconds = time.perf_counter() - started
        started = time.perf_counter()
        stage_2 = _predict_batched(
            self.models[1], graphs_2, float(checkpoint_2["scale"]),
            batch_size=model_batch_size,
        )
        stage_2_model_seconds = time.perf_counter() - started
        if compute_stage2_residuals:
            started = time.perf_counter()
            residual_indices, residual_stride = _stage_2_residual_indices(
                len(measured), stage2_residual_stride
            )
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
                workers=physics_workers,
            )
            residual_seconds = time.perf_counter() - started
        else:
            residual_2 = np.empty(0, dtype=np.float64)
            residual_indices = np.empty(0, dtype=np.int64)
            residual_stride = 0
            residual_seconds = 0.0
        frame_indices = np.arange(len(measured), dtype=np.int64)
        return stage_1, stage_2, {
            "stage_1_forward_voltage_rms": np.asarray(
                [item.voltage_residual_rms for item in diagnostics_2]
            ),
            "stage_2_forward_voltage_rms": np.asarray(residual_2),
            "stage_2_forward_voltage_rms_stride": residual_stride,
            "stage_1_forward_voltage_rms_indices": frame_indices,
            "stage_2_forward_voltage_rms_indices": residual_indices,
            "frame_indices": frame_indices,
            "batch": {
                "frames": len(measured),
                "measurements": measured.shape[1],
                "model_batch_size": model_batch_size,
                "physics_workers": active_physics_workers,
            },
            "timings_seconds": {
                "fixed_stage_physics": fixed_stage_seconds,
                "stage_1_graph_build": stage_1_graph_seconds,
                "stage_1_model": stage_1_model_seconds,
                "nonlinear_stage_2_physics": nonlinear_physics_seconds,
                "stage_2_graph_build": stage_2_graph_seconds,
                "stage_2_model": stage_2_model_seconds,
                "stage_2_residual": residual_seconds,
                "total": time.perf_counter() - total_started,
            },
        }

    def reconstruct_reference(self, voltage: np.ndarray) -> tuple[np.ndarray, np.ndarray, dict]:
        """Original dense, frame-wise implementation used for parity tests."""
        measured = np.asarray(voltage, dtype=np.float64)
        elements = self.mappings.num_elements
        truth = np.zeros((len(measured), elements), dtype=np.float64)
        baseline = np.full_like(truth, 0.7)
        current = np.zeros_like(truth)
        stages, residuals = [], []
        baseline_voltages = None
        for checkpoint, model in zip(self.checkpoints, self.models):
            direction, _, baseline_voltages = dataset_lm_directions(
                self.stage_physics, baseline, current, measured,
                regularizer=self.mappings.laplace,
                hyper_pvi=self.cfg.hyper_pvi, lambda_lm=self.cfg.lambda_lm,
                step_size=self.cfg.lm_step_size,
                minimum_conductivity=float(checkpoint["physics_contract"].get("minimum_conductivity", 1e-4)),
                baseline_voltages=baseline_voltages,
            )
            graphs = make_coordinate_graphs(
                truth, current, direction, self.positions, self.runtime["edge_index"],
                scale=float(checkpoint["scale"]), use_coordinates=True, positive_weight=0.0,
                voltage=measured if checkpoint.get("use_voltage_mlp", False) else None,
                voltage_scale=float(checkpoint.get("voltage_scale", 1.0)),
            )
            current = predict_coordinate_graphs(
                model, graphs, float(checkpoint["scale"])
            )
            stage_residual, baseline_voltages, _ = dataset_voltage_residual_rms(
                self.stage_physics, baseline, current, measured,
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
        stage_1_graphs = make_vessel_graphs(
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
        stage_2_graphs = make_vessel_graphs(
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
        measured = np.asarray(voltage, dtype=np.float64)
        count, elements = len(measured), self.mappings.num_elements
        baseline = np.broadcast_to(self.baseline[None, :], (count, elements)).copy()
        labels = np.broadcast_to(self.tissue_labels[None, :], (count, elements))
        zero = np.zeros_like(baseline)
        dummy = np.zeros((count, 2, 8), dtype=np.float32)
        direction_1, _, baseline_voltages = self.fixed_stage.solve_many(measured)
        graphs_1 = make_vessel_graphs(
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
        graphs_2 = make_vessel_graphs(
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
        graphs_1 = make_vessel_graphs(
            zero, zero, direction_1, measured, dummy, self.positions,
            self.runtime["edge_index"],
            conductivity_scale=float(self.contract["conductivity_scale"]),
            voltage_scale=float(self.contract["voltage_scale"]),
            tissue_labels=labels, voltage_clean=measured,
        )
        stage_1, parameters = predict_vessel_graphs_with_parameters(
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
        graphs_2 = make_vessel_graphs(
            zero, stage_1, direction_2, measured, dummy, self.positions,
            self.runtime["edge_index"],
            conductivity_scale=float(self.contract["conductivity_scale"]),
            voltage_scale=float(self.contract["voltage_scale"]),
            initial_parameters=parameters, tissue_labels=labels, voltage_clean=measured,
        )
        stage_2, _ = predict_vessel_graphs_with_parameters(
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
