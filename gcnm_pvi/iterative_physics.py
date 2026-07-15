"""Per-stage nonlinear differential physics for faithful PVI-GCNM.

The original GCNM recomputes the forward solution, Jacobian, residual, and
Levenberg--Marquardt direction after every learned stage.  This module provides
that operation for the PVI complete-electrode forward model and also exposes an
iterative-LM-only control using exactly the same physics.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import linalg, sparse


@dataclass(frozen=True)
class StepDiagnostics:
    """Diagnostics evaluated before applying one LM direction."""

    voltage_residual_rms: float
    step_rms: float
    clipped_elements: int


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
    direction = float(step_size) * np.asarray(direction, dtype=np.float64)
    diagnostics = StepDiagnostics(
        voltage_residual_rms=float(np.sqrt(np.mean(residual**2))),
        step_rms=float(np.sqrt(np.mean(direction**2))),
        clipped_elements=clipped,
    )
    return direction, diagnostics, baseline_voltage


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
        directions, diagnostics, baseline_voltages = dataset_lm_directions(
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
