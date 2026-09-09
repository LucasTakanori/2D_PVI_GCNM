"""Per-stage nonlinear differential physics for faithful PVI-GCNM.

The original GCNM recomputes the forward solution, Jacobian, residual, and
Levenberg--Marquardt direction after every learned stage.  This module provides
that operation for the PVI complete-electrode forward model and also exposes an
iterative-LM-only control using exactly the same physics.
"""

from __future__ import annotations

import copy
import multiprocessing as mp
import os
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

import numpy as np
from scipy import linalg, sparse


_PROCESS_LM_STATE = None
_PROCESS_RESIDUAL_STATE = None


def _process_lm_lane(indices: np.ndarray):
    """Fork-worker entry point using copy-on-write shared input arrays."""

    if _PROCESS_LM_STATE is None:
        raise RuntimeError("process LM worker was started without inherited state")
    (
        physics,
        baselines,
        currents,
        measured,
        regularizer,
        hyper_pvi,
        lambda_lm,
        step_size,
        minimum_conductivity,
        cached,
        system_solver,
    ) = _PROCESS_LM_STATE
    return dataset_lm_directions(
        physics,
        baselines[indices],
        currents[indices],
        measured[indices],
        regularizer=regularizer,
        hyper_pvi=hyper_pvi,
        lambda_lm=lambda_lm,
        step_size=step_size,
        minimum_conductivity=minimum_conductivity,
        baseline_voltages=None if cached is None else cached[indices],
        system_solver=system_solver,
    )


def _process_residual_lane(indices: np.ndarray):
    """Fork-worker entry point for nonlinear voltage residual evaluation."""

    if _PROCESS_RESIDUAL_STATE is None:
        raise RuntimeError("process residual worker was started without inherited state")
    physics, baselines, currents, measured, minimum_conductivity, cached = (
        _PROCESS_RESIDUAL_STATE
    )
    return dataset_voltage_residual_rms(
        physics,
        baselines[indices],
        currents[indices],
        measured[indices],
        minimum_conductivity=minimum_conductivity,
        baseline_voltages=None if cached is None else cached[indices],
    )


@dataclass(frozen=True)
class StepDiagnostics:
    """Diagnostics evaluated before applying one LM direction."""

    voltage_residual_rms: float
    step_rms: float
    clipped_elements: int


class LowRankRegularizedSolver:
    """Solve ``(J.T J + H) p = -J.T r`` using a 32-channel update.

    ``H`` is fixed for a ring and stage protocol, while ``J`` changes for each
    nonlinear frame.  The projected Laplacian has the constant vector as its
    one-dimensional nullspace; adding and then removing that projector through
    Woodbury keeps the solve algebraically equivalent to the dense system.
    """

    def __init__(
        self,
        regularizer: np.ndarray | sparse.spmatrix | None,
        elements: int,
        *,
        hyper_pvi: float,
        lambda_lm: float,
    ) -> None:
        reg = _regularizer_array(regularizer, elements)
        matrix = np.zeros((elements, elements), dtype=np.float64)
        if reg is not None and hyper_pvi > 0:
            matrix += float(hyper_pvi) ** 2 * reg
        if lambda_lm > 0:
            matrix += float(lambda_lm) * np.eye(elements)

        eigenvalues, eigenvectors = linalg.eigh(matrix, check_finite=False)
        tolerance = max(float(np.max(np.abs(eigenvalues))) * 1e-12, 1e-15)
        if np.any(eigenvalues < -tolerance):
            raise ValueError("fixed LM regularizer is not positive semidefinite")
        positive = eigenvalues > tolerance
        self.positive_vectors = eigenvectors[:, positive]
        self.positive_values = eigenvalues[positive]
        self.positive_inverse = 1.0 / self.positive_values
        self.null_vectors = eigenvectors[:, ~positive]
        if self.null_vectors.shape[1] > 1:
            raise ValueError(
                "low-rank LM solver supports at most one regularizer null mode"
            )

    def _positive_solve(
        self, projected_jacobian: np.ndarray, rhs: np.ndarray
    ) -> np.ndarray:
        """Apply ``(D + J.T J)^-1`` in the positive eigen-subspace.

        The SVD form avoids subtracting two very large Woodbury terms when the
        scaled Laplacian eigenvalues are small.
        """

        rhs_2d = np.asarray(rhs, dtype=np.float64)
        was_vector = rhs_2d.ndim == 1
        if was_vector:
            rhs_2d = rhs_2d[:, None]
        inverse_sqrt = np.sqrt(self.positive_inverse)
        whitened_jacobian = projected_jacobian * inverse_sqrt[None, :]
        if not np.all(np.isfinite(whitened_jacobian)):
            raise FloatingPointError(
                "non-finite whitened Jacobian in low-rank LM solve"
            )
        try:
            _u, singular, vh = linalg.svd(
                whitened_jacobian,
                full_matrices=False,
                check_finite=False,
                lapack_driver="gesdd",
            )
        except linalg.LinAlgError:
            # LAPACK's divide-and-conquer driver can rarely fail to converge
            # on a finite, strongly scaled real frame.  The QR-based driver is
            # slower but more robust and computes the same SVD.  Normal frames
            # never enter this branch, preserving the optimized pilot path.
            _u, singular, vh = linalg.svd(
                whitened_jacobian,
                full_matrices=False,
                check_finite=False,
                lapack_driver="gesvd",
            )
        def apply_inverse(values: np.ndarray) -> np.ndarray:
            whitened_rhs = inverse_sqrt[:, None] * values
            coefficients = vh @ whitened_rhs
            # Split row-space and orthogonal-space components explicitly.
            orthogonal = whitened_rhs - vh.T @ coefficients
            retained = coefficients / (1.0 + singular[:, None] ** 2)
            return inverse_sqrt[:, None] * (orthogonal + vh.T @ retained)

        result = apply_inverse(rhs_2d)
        # Mixed scales (h^2 L versus J.T J) make the closed-form application
        # sensitive to roundoff. A few cheap refinement steps restore the
        # residual to the dense equation without another factorization.
        for _ in range(6):
            residual = rhs_2d - (
                self.positive_values[:, None] * result
                + projected_jacobian.T @ (projected_jacobian @ result)
            )
            result = result + apply_inverse(residual)
        return result[:, 0] if was_vector else result

    def solve(self, jacobian: np.ndarray, residual: np.ndarray) -> np.ndarray:
        jacobian = np.asarray(jacobian, dtype=np.float64)
        residual = np.asarray(residual, dtype=np.float64).ravel()
        positive_jacobian = jacobian @ self.positive_vectors
        positive_rhs = -(positive_jacobian.T @ residual)
        if self.null_vectors.shape[1] == 0:
            positive_solution = self._positive_solve(
                positive_jacobian, positive_rhs
            )
            return self.positive_vectors @ positive_solution

        null_jacobian = jacobian @ self.null_vectors
        null_rhs = -(null_jacobian.T @ residual)
        coupling = positive_jacobian.T @ null_jacobian
        solved = self._positive_solve(
            positive_jacobian,
            np.column_stack((positive_rhs, coupling)),
        )
        positive_base = solved[:, 0]
        positive_coupling = solved[:, 1:]
        null_block = null_jacobian.T @ null_jacobian
        schur = null_block - coupling.T @ positive_coupling
        null_solution = linalg.solve(
            schur,
            null_rhs - coupling.T @ positive_base,
            assume_a="sym",
            check_finite=False,
        )
        positive_solution = positive_base - positive_coupling @ null_solution
        return (
            self.positive_vectors @ positive_solution
            + self.null_vectors @ null_solution
        )


class FixedZeroCurrentLMSolver:
    """Cache the exact first-stage map for a fixed baseline and zero state."""

    def __init__(
        self,
        physics,
        baseline: np.ndarray,
        *,
        regularizer: np.ndarray | sparse.spmatrix | None,
        hyper_pvi: float,
        lambda_lm: float,
        step_size: float = 1.0,
    ) -> None:
        self.baseline = np.asarray(baseline, dtype=np.float64).ravel()
        self.baseline_voltage, jacobian = physics.forward_and_jacobian(self.baseline)
        jacobian = np.asarray(jacobian, dtype=np.float64)
        system = jacobian.T @ jacobian
        reg = _regularizer_array(regularizer, len(self.baseline))
        if reg is not None and hyper_pvi > 0:
            system += float(hyper_pvi) ** 2 * reg
        if lambda_lm > 0:
            system += float(lambda_lm) * np.eye(len(self.baseline))
        self.voltage_to_direction = float(step_size) * linalg.solve(
            system,
            jacobian.T,
            assume_a="sym",
            check_finite=False,
        )

    def solve_many(
        self, measured: np.ndarray
    ) -> tuple[np.ndarray, list[StepDiagnostics], np.ndarray]:
        measured = np.asarray(measured, dtype=np.float64)
        directions = measured @ self.voltage_to_direction.T
        diagnostics = [
            StepDiagnostics(
                voltage_residual_rms=float(np.sqrt(np.mean(frame**2))),
                step_rms=float(np.sqrt(np.mean(direction**2))),
                clipped_elements=0,
            )
            for frame, direction in zip(measured, directions)
        ]
        baselines = np.broadcast_to(
            np.asarray(self.baseline_voltage)[None, :],
            (len(measured), len(self.baseline_voltage)),
        ).copy()
        return directions, diagnostics, baselines

    def solve_many_absolute(
        self, measured: np.ndarray
    ) -> tuple[np.ndarray, list[StepDiagnostics], np.ndarray]:
        """First absolute-LM direction from the shared homogeneous state."""

        measured = np.asarray(measured, dtype=np.float64)
        residual = np.asarray(self.baseline_voltage)[None, :] - measured
        directions = -residual @ self.voltage_to_direction.T
        diagnostics = [
            StepDiagnostics(
                voltage_residual_rms=float(np.sqrt(np.mean(frame**2))),
                step_rms=float(np.sqrt(np.mean(direction**2))),
                clipped_elements=0,
            )
            for frame, direction in zip(residual, directions)
        ]
        baselines = np.broadcast_to(
            np.asarray(self.baseline_voltage)[None, :], measured.shape
        ).copy()
        return directions, diagnostics, baselines


def _regularizer_array(
    regularizer: np.ndarray | sparse.spmatrix | None,
    elements: int,
) -> np.ndarray | None:
    if regularizer is None:
        return None
    matrix = regularizer.toarray() if sparse.issparse(regularizer) else np.asarray(regularizer)
    matrix = np.asarray(matrix, dtype=np.float64)
    if matrix.shape != (elements, elements):
        raise ValueError(
            f"regularizer shape {matrix.shape} does not match {(elements, elements)}"
        )
    return matrix


def differential_lm_step(
    physics,
    sigma_baseline: np.ndarray,
    delta_current: np.ndarray,
    delta_voltage_measured: np.ndarray,
    *,
    regularizer: np.ndarray | sparse.spmatrix | None,
    hyper_pvi: float,
    lambda_lm: float,
    step_size: float = 1.0,
    minimum_conductivity: float = 1e-4,
    baseline_voltage: np.ndarray | None = None,
    system_solver: LowRankRegularizedSolver | None = None,
) -> tuple[np.ndarray, StepDiagnostics, np.ndarray]:
    """Recompute one nonlinear differential LM direction.

    For the current differential estimate ``delta_current`` this evaluates

    ``F(sigma_b + delta_current) - F(sigma_b)`` and
    ``J(sigma_b + delta_current)``

    before solving

    ``(J.T J + hyper_pvi**2 R + lambda_lm I) p = -J.T r``.

    The returned vector is the direction ``p``, not the accumulated state.
    ``baseline_voltage`` can be supplied when the baseline forward solution is
    cached across stages.
    """

    baseline = np.asarray(sigma_baseline, dtype=np.float64).ravel()
    current = np.asarray(delta_current, dtype=np.float64).ravel()
    measured = np.asarray(delta_voltage_measured, dtype=np.float64).ravel()
    if baseline.shape != current.shape:
        raise ValueError(
            f"baseline shape {baseline.shape} does not match current {current.shape}"
        )
    absolute_unclipped = baseline + current
    absolute = np.maximum(absolute_unclipped, float(minimum_conductivity))
    clipped = int(np.count_nonzero(absolute != absolute_unclipped))

    if baseline_voltage is None:
        baseline_voltage = np.asarray(physics.solve(baseline), dtype=np.float64).ravel()
    else:
        baseline_voltage = np.asarray(baseline_voltage, dtype=np.float64).ravel()
    voltage_current, jacobian = physics.forward_and_jacobian(absolute)
    voltage_current = np.asarray(voltage_current, dtype=np.float64).ravel()
    jacobian = np.asarray(jacobian, dtype=np.float64)
    if jacobian.shape != (measured.size, current.size):
        raise ValueError(
            f"Jacobian shape {jacobian.shape} does not match "
            f"{(measured.size, current.size)}"
        )

    predicted = voltage_current - baseline_voltage
    residual = predicted - measured
    if system_solver is None:
        system = jacobian.T @ jacobian
        reg = _regularizer_array(regularizer, current.size)
        if reg is not None and hyper_pvi > 0:
            system = system + float(hyper_pvi) ** 2 * reg
        if lambda_lm > 0:
            system = system + float(lambda_lm) * np.eye(current.size)
        rhs = -(jacobian.T @ residual)
        try:
            direction = linalg.solve(
                system,
                rhs,
                assume_a="sym",
                check_finite=False,
            )
        except linalg.LinAlgError:
            direction = linalg.lstsq(system, rhs, check_finite=False)[0]
    else:
        direction = system_solver.solve(jacobian, residual)
    direction = float(step_size) * np.asarray(direction, dtype=np.float64)
    diagnostics = StepDiagnostics(
        voltage_residual_rms=float(np.sqrt(np.mean(residual**2))),
        step_rms=float(np.sqrt(np.mean(direction**2))),
        clipped_elements=clipped,
    )
    return direction, diagnostics, baseline_voltage


def absolute_lm_step(
    physics,
    sigma_current: np.ndarray,
    voltage_measured: np.ndarray,
    *,
    regularizer: np.ndarray | sparse.spmatrix | None,
    hyper_pvi: float,
    lambda_lm: float,
    step_size: float = 1.0,
    minimum_conductivity: float = 1e-4,
    system_solver: LowRankRegularizedSolver | None = None,
) -> tuple[np.ndarray, StepDiagnostics, np.ndarray]:
    """Compute one absolute-voltage LM direction.

    Unlike :func:`differential_lm_step`, this compares the forward solve
    directly with the measured absolute electrode voltage:

    ``r = F(sigma_current) - V_absolute``.
    """

    current_unclipped = np.asarray(sigma_current, dtype=np.float64).ravel()
    current = np.maximum(current_unclipped, float(minimum_conductivity))
    clipped = int(np.count_nonzero(current != current_unclipped))
    measured = np.asarray(voltage_measured, dtype=np.float64).ravel()
    predicted, jacobian = physics.forward_and_jacobian(current)
    predicted = np.asarray(predicted, dtype=np.float64).ravel()
    jacobian = np.asarray(jacobian, dtype=np.float64)
    if jacobian.shape != (measured.size, current.size):
        raise ValueError(
            f"Jacobian shape {jacobian.shape} does not match "
            f"{(measured.size, current.size)}"
        )
    residual = predicted - measured
    if system_solver is None:
        system = jacobian.T @ jacobian
        reg = _regularizer_array(regularizer, current.size)
        if reg is not None and hyper_pvi > 0:
            system = system + float(hyper_pvi) ** 2 * reg
        if lambda_lm > 0:
            system = system + float(lambda_lm) * np.eye(current.size)
        rhs = -(jacobian.T @ residual)
        try:
            direction = linalg.solve(
                system, rhs, assume_a="sym", check_finite=False
            )
        except linalg.LinAlgError:
            direction = linalg.lstsq(system, rhs, check_finite=False)[0]
    else:
        direction = system_solver.solve(jacobian, residual)
    direction = float(step_size) * np.asarray(direction, dtype=np.float64)
    diagnostics = StepDiagnostics(
        voltage_residual_rms=float(np.sqrt(np.mean(residual**2))),
        step_rms=float(np.sqrt(np.mean(direction**2))),
        clipped_elements=clipped,
    )
    return direction, diagnostics, predicted


def dataset_absolute_lm_directions(
    physics,
    sigma_current: np.ndarray,
    voltage_measured: np.ndarray,
    *,
    regularizer: np.ndarray | sparse.spmatrix | None,
    hyper_pvi: float,
    lambda_lm: float,
    step_size: float = 1.0,
    minimum_conductivity: float = 1e-4,
    progress_label: str | None = None,
    system_solver: LowRankRegularizedSolver | None = None,
) -> tuple[np.ndarray, list[StepDiagnostics], np.ndarray]:
    """Compute absolute-LM directions for independent frames in order."""

    currents = np.asarray(sigma_current, dtype=np.float64)
    voltages = np.asarray(voltage_measured, dtype=np.float64)
    if len(currents) != len(voltages):
        raise ValueError("current conductivity and voltage counts must match")
    directions, diagnostics, predictions = [], [], []
    for index, (current, measured) in enumerate(zip(currents, voltages)):
        direction, diagnostic, predicted = absolute_lm_step(
            physics,
            current,
            measured,
            regularizer=regularizer,
            hyper_pvi=hyper_pvi,
            lambda_lm=lambda_lm,
            step_size=step_size,
            minimum_conductivity=minimum_conductivity,
            system_solver=system_solver,
        )
        directions.append(direction)
        diagnostics.append(diagnostic)
        predictions.append(predicted)
        if progress_label and (index + 1 == len(currents) or (index + 1) % 25 == 0):
            print(
                f"{progress_label}: physics {index + 1}/{len(currents)}",
                flush=True,
            )
    return np.stack(directions), diagnostics, np.stack(predictions)


def parallel_dataset_absolute_lm_directions(
    physics,
    sigma_current: np.ndarray,
    voltage_measured: np.ndarray,
    *,
    regularizer: np.ndarray | sparse.spmatrix | None,
    hyper_pvi: float,
    lambda_lm: float,
    step_size: float = 1.0,
    minimum_conductivity: float = 1e-4,
    progress_label: str | None = None,
    system_solver: LowRankRegularizedSolver | None = None,
    workers: int | None = None,
) -> tuple[np.ndarray, list[StepDiagnostics], np.ndarray]:
    """Thread independent absolute-LM frames with private FEM runtimes."""

    requested = int(os.environ.get("GCNM_PHYSICS_WORKERS", workers or 1))
    count = len(sigma_current)
    lanes = min(max(requested, 1), count)
    if lanes == 1:
        return dataset_absolute_lm_directions(
            physics,
            sigma_current,
            voltage_measured,
            regularizer=regularizer,
            hyper_pvi=hyper_pvi,
            lambda_lm=lambda_lm,
            step_size=step_size,
            minimum_conductivity=minimum_conductivity,
            progress_label=progress_label,
            system_solver=system_solver,
        )
    parts = [part for part in np.array_split(np.arange(count), lanes) if len(part)]
    physics_lanes = [physics, *(copy.deepcopy(physics) for _ in range(len(parts) - 1))]
    solver_lanes = (
        [None] * len(parts)
        if system_solver is None
        else [system_solver, *(copy.deepcopy(system_solver) for _ in range(len(parts) - 1))]
    )

    def run(lane: int, indices: np.ndarray):
        return dataset_absolute_lm_directions(
            physics_lanes[lane],
            np.asarray(sigma_current)[indices],
            np.asarray(voltage_measured)[indices],
            regularizer=regularizer,
            hyper_pvi=hyper_pvi,
            lambda_lm=lambda_lm,
            step_size=step_size,
            minimum_conductivity=minimum_conductivity,
            progress_label=None,
            system_solver=solver_lanes[lane],
        )

    with ThreadPoolExecutor(max_workers=len(parts)) as executor:
        futures = [
            executor.submit(run, lane, indices)
            for lane, indices in enumerate(parts)
        ]
        results = [future.result() for future in futures]
    if progress_label:
        print(f"{progress_label}: physics {count}/{count} across {len(parts)} lanes", flush=True)
    return (
        np.concatenate([result[0] for result in results], axis=0),
        [item for result in results for item in result[1]],
        np.concatenate([result[2] for result in results], axis=0),
    )


def dataset_lm_directions(
    physics,
    sigma_baseline: np.ndarray,
    delta_current: np.ndarray,
    delta_voltage_measured: np.ndarray,
    *,
    regularizer: np.ndarray | sparse.spmatrix | None,
    hyper_pvi: float,
    lambda_lm: float,
    step_size: float = 1.0,
    minimum_conductivity: float = 1e-4,
    baseline_voltages: np.ndarray | None = None,
    progress_label: str | None = None,
    system_solver: LowRankRegularizedSolver | None = None,
) -> tuple[np.ndarray, list[StepDiagnostics], np.ndarray]:
    """Compute faithful LM features for every sample in a dataset."""

    baselines = np.asarray(sigma_baseline, dtype=np.float64)
    currents = np.asarray(delta_current, dtype=np.float64)
    voltages = np.asarray(delta_voltage_measured, dtype=np.float64)
    if not (len(baselines) == len(currents) == len(voltages)):
        raise ValueError("baseline, current, and voltage sample counts must match")
    cached = None if baseline_voltages is None else np.asarray(baseline_voltages)
    if cached is not None and len(cached) != len(baselines):
        raise ValueError("baseline voltage cache length does not match dataset")

    directions: list[np.ndarray] = []
    diagnostics: list[StepDiagnostics] = []
    baseline_output: list[np.ndarray] = []
    for index in range(len(baselines)):
        direction, diagnostic, base_voltage = differential_lm_step(
            physics,
            baselines[index],
            currents[index],
            voltages[index],
            regularizer=regularizer,
            hyper_pvi=hyper_pvi,
            lambda_lm=lambda_lm,
            step_size=step_size,
            minimum_conductivity=minimum_conductivity,
            baseline_voltage=None if cached is None else cached[index],
            system_solver=system_solver,
        )
        directions.append(direction)
        diagnostics.append(diagnostic)
        baseline_output.append(base_voltage)
        if progress_label and (index + 1 == len(baselines) or (index + 1) % 25 == 0):
            print(
                f"{progress_label}: physics {index + 1}/{len(baselines)}",
                flush=True,
            )
    return np.stack(directions), diagnostics, np.stack(baseline_output)


def parallel_dataset_lm_directions(
    physics,
    sigma_baseline: np.ndarray,
    delta_current: np.ndarray,
    delta_voltage_measured: np.ndarray,
    *,
    regularizer: np.ndarray | sparse.spmatrix | None,
    hyper_pvi: float,
    lambda_lm: float,
    step_size: float = 1.0,
    minimum_conductivity: float = 1e-4,
    baseline_voltages: np.ndarray | None = None,
    progress_label: str | None = None,
    system_solver: LowRankRegularizedSolver | None = None,
    workers: int | None = None,
) -> tuple[np.ndarray, list[StepDiagnostics], np.ndarray]:
    """Compute independent frame directions concurrently without reordering.

    Each lane owns a private mutable forward-model object and low-rank solver.
    This is numerically the same per-frame operation as
    :func:`dataset_lm_directions`; only independent frames are scheduled in
    parallel.
    """

    requested = int(os.environ.get("GCNM_PHYSICS_WORKERS", workers or 1))
    count = len(sigma_baseline)
    lanes = min(max(requested, 1), count)
    if lanes == 1:
        return dataset_lm_directions(
            physics,
            sigma_baseline,
            delta_current,
            delta_voltage_measured,
            regularizer=regularizer,
            hyper_pvi=hyper_pvi,
            lambda_lm=lambda_lm,
            step_size=step_size,
            minimum_conductivity=minimum_conductivity,
            baseline_voltages=baseline_voltages,
            progress_label=progress_label,
            system_solver=system_solver,
        )
    parts = [part for part in np.array_split(np.arange(count), lanes) if len(part)]
    executor_mode = os.environ.get("GCNM_PHYSICS_EXECUTOR", "thread").lower()
    if executor_mode not in {"thread", "process"}:
        raise ValueError("GCNM_PHYSICS_EXECUTOR must be 'thread' or 'process'")
    cached = None if baseline_voltages is None else np.asarray(baseline_voltages)
    if executor_mode == "process":
        if "fork" not in mp.get_all_start_methods():
            raise RuntimeError(
                "process physics requires the POSIX fork start method"
            )
        global _PROCESS_LM_STATE
        _PROCESS_LM_STATE = (
            physics,
            np.asarray(sigma_baseline),
            np.asarray(delta_current),
            np.asarray(delta_voltage_measured),
            regularizer,
            float(hyper_pvi),
            float(lambda_lm),
            float(step_size),
            float(minimum_conductivity),
            cached,
            system_solver,
        )
        try:
            with mp.get_context("fork").Pool(processes=len(parts)) as pool:
                results = pool.map(_process_lm_lane, parts)
        finally:
            _PROCESS_LM_STATE = None
        if progress_label:
            print(
                f"{progress_label}: physics {count}/{count} across "
                f"{len(parts)} processes",
                flush=True,
            )
        return (
            np.concatenate([result[0] for result in results], axis=0),
            [item for result in results for item in result[1]],
            np.concatenate([result[2] for result in results], axis=0),
        )
    physics_lanes = [physics, *(copy.deepcopy(physics) for _ in range(len(parts) - 1))]
    solver_lanes = (
        [None] * len(parts)
        if system_solver is None
        else [
            system_solver,
            *(copy.deepcopy(system_solver) for _ in range(len(parts) - 1)),
        ]
    )
    def run(lane: int, indices: np.ndarray):
        return dataset_lm_directions(
            physics_lanes[lane],
            np.asarray(sigma_baseline)[indices],
            np.asarray(delta_current)[indices],
            np.asarray(delta_voltage_measured)[indices],
            regularizer=regularizer,
            hyper_pvi=hyper_pvi,
            lambda_lm=lambda_lm,
            step_size=step_size,
            minimum_conductivity=minimum_conductivity,
            baseline_voltages=None if cached is None else cached[indices],
            progress_label=None,
            system_solver=solver_lanes[lane],
        )

    with ThreadPoolExecutor(max_workers=len(parts)) as executor:
        futures = [
            executor.submit(run, lane, indices)
            for lane, indices in enumerate(parts)
        ]
        results = [future.result() for future in futures]
    if progress_label:
        print(f"{progress_label}: physics {count}/{count} across {len(parts)} lanes", flush=True)
    return (
        np.concatenate([result[0] for result in results], axis=0),
        [item for result in results for item in result[1]],
        np.concatenate([result[2] for result in results], axis=0),
    )


def iterative_lm_reconstruction(
    physics,
    sigma_baseline: np.ndarray,
    delta_voltage_measured: np.ndarray,
    *,
    iterations: int,
    regularizer: np.ndarray | sparse.spmatrix | None,
    hyper_pvi: float,
    lambda_lm: float,
    step_size: float = 1.0,
    minimum_conductivity: float = 1e-4,
    progress_label: str = "iterative LM",
) -> tuple[np.ndarray, list[list[StepDiagnostics]]]:
    """Run the physics-only control and return every accumulated LM stage."""

    baselines = np.asarray(sigma_baseline, dtype=np.float64)
    measured = np.asarray(delta_voltage_measured, dtype=np.float64)
    current = np.zeros_like(baselines)
    baseline_voltages = None
    stages: list[np.ndarray] = []
    all_diagnostics: list[list[StepDiagnostics]] = []
    for iteration in range(int(iterations)):
        current = (
            np.maximum(baselines + current, float(minimum_conductivity)) - baselines
        )
        directions, diagnostics, baseline_voltages = parallel_dataset_lm_directions(
            physics,
            baselines,
            current,
            measured,
            regularizer=regularizer,
            hyper_pvi=hyper_pvi,
            lambda_lm=lambda_lm,
            step_size=step_size,
            minimum_conductivity=minimum_conductivity,
            baseline_voltages=baseline_voltages,
            progress_label=f"{progress_label} stage {iteration + 1}",
        )
        current = current + directions
        stages.append(current.copy())
        all_diagnostics.append(diagnostics)
    return np.stack(stages), all_diagnostics


def dataset_voltage_residual_rms(
    physics,
    sigma_baseline: np.ndarray,
    delta_current: np.ndarray,
    delta_voltage_measured: np.ndarray,
    *,
    minimum_conductivity: float = 1e-4,
    baseline_voltages: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, int]:
    """Evaluate nonlinear differential voltage residuals without a Jacobian."""

    baselines = np.asarray(sigma_baseline, dtype=np.float64)
    currents = np.asarray(delta_current, dtype=np.float64)
    measured = np.asarray(delta_voltage_measured, dtype=np.float64)
    cached = None if baseline_voltages is None else np.asarray(baseline_voltages)
    residuals, baseline_output = [], []
    clipped_total = 0
    for index in range(len(baselines)):
        baseline_voltage = (
            np.asarray(physics.solve(baselines[index]), dtype=np.float64).ravel()
            if cached is None
            else cached[index]
        )
        absolute_unclipped = baselines[index] + currents[index]
        absolute = np.maximum(absolute_unclipped, float(minimum_conductivity))
        clipped_total += int(np.count_nonzero(absolute != absolute_unclipped))
        predicted = np.asarray(physics.solve(absolute), dtype=np.float64).ravel() - baseline_voltage
        residual = predicted - measured[index]
        residuals.append(float(np.sqrt(np.mean(residual**2))))
        baseline_output.append(baseline_voltage)
    return np.asarray(residuals), np.stack(baseline_output), clipped_total


def parallel_dataset_voltage_residual_rms(
    physics,
    sigma_baseline: np.ndarray,
    delta_current: np.ndarray,
    delta_voltage_measured: np.ndarray,
    *,
    minimum_conductivity: float = 1e-4,
    baseline_voltages: np.ndarray | None = None,
    workers: int | None = None,
) -> tuple[np.ndarray, np.ndarray, int]:
    """Evaluate independent forward residuals concurrently and in order."""

    requested = int(os.environ.get("GCNM_PHYSICS_WORKERS", workers or 1))
    count = len(sigma_baseline)
    lanes = min(max(requested, 1), count)
    if lanes == 1:
        return dataset_voltage_residual_rms(
            physics,
            sigma_baseline,
            delta_current,
            delta_voltage_measured,
            minimum_conductivity=minimum_conductivity,
            baseline_voltages=baseline_voltages,
        )
    parts = [part for part in np.array_split(np.arange(count), lanes) if len(part)]
    executor_mode = os.environ.get("GCNM_PHYSICS_EXECUTOR", "thread").lower()
    if executor_mode not in {"thread", "process"}:
        raise ValueError("GCNM_PHYSICS_EXECUTOR must be 'thread' or 'process'")
    cached = None if baseline_voltages is None else np.asarray(baseline_voltages)
    if executor_mode == "process":
        if "fork" not in mp.get_all_start_methods():
            raise RuntimeError(
                "process physics requires the POSIX fork start method"
            )
        global _PROCESS_RESIDUAL_STATE
        _PROCESS_RESIDUAL_STATE = (
            physics,
            np.asarray(sigma_baseline),
            np.asarray(delta_current),
            np.asarray(delta_voltage_measured),
            float(minimum_conductivity),
            cached,
        )
        try:
            with mp.get_context("fork").Pool(processes=len(parts)) as pool:
                results = pool.map(_process_residual_lane, parts)
        finally:
            _PROCESS_RESIDUAL_STATE = None
        return (
            np.concatenate([result[0] for result in results]),
            np.concatenate([result[1] for result in results], axis=0),
            sum(result[2] for result in results),
        )
    physics_lanes = [physics, *(copy.deepcopy(physics) for _ in range(len(parts) - 1))]

    def run(lane: int, indices: np.ndarray):
        return dataset_voltage_residual_rms(
            physics_lanes[lane],
            np.asarray(sigma_baseline)[indices],
            np.asarray(delta_current)[indices],
            np.asarray(delta_voltage_measured)[indices],
            minimum_conductivity=minimum_conductivity,
            baseline_voltages=None if cached is None else cached[indices],
        )

    with ThreadPoolExecutor(max_workers=len(parts)) as executor:
        futures = [
            executor.submit(run, lane, indices)
            for lane, indices in enumerate(parts)
        ]
        results = [future.result() for future in futures]
    return (
        np.concatenate([result[0] for result in results]),
        np.concatenate([result[1] for result in results], axis=0),
        int(sum(result[2] for result in results)),
    )


def diagnostics_summary(values: list[StepDiagnostics]) -> dict[str, float | int]:
    """Summarize per-sample diagnostics for JSON reports."""

    return {
        "samples": len(values),
        "voltage_residual_rms_mean": float(
            np.mean([value.voltage_residual_rms for value in values])
        ),
        "voltage_residual_rms_max": float(
            np.max([value.voltage_residual_rms for value in values])
        ),
        "step_rms_mean": float(np.mean([value.step_rms for value in values])),
        "clipped_elements_total": int(sum(value.clipped_elements for value in values)),
    }
