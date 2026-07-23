"""Stage physics for the two-stage voltage-calibrated GCNM."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import linalg, sparse


@dataclass(frozen=True)
class StagePhysicsBatch:
    """Physics tensors needed to train or evaluate one learned stage."""

    directions: np.ndarray
    jacobians: np.ndarray
    voltage_targets: np.ndarray
    baseline_voltages: np.ndarray
    current_voltage_residual_rms: np.ndarray
    step_rms: np.ndarray
    clipped_elements: int


def _regularizer_array(regularizer, elements: int) -> np.ndarray | None:
    if regularizer is None:
        return None
    value = regularizer.toarray() if sparse.issparse(regularizer) else regularizer
    value = np.asarray(value, dtype=np.float64)
    if value.shape != (elements, elements):
        raise ValueError(
            f"regularizer shape {value.shape} does not match {(elements, elements)}"
        )
    return value


def _solve_direction(
    jacobian: np.ndarray,
    target: np.ndarray,
    *,
    regularizer: np.ndarray | None,
    hyper_pvi: float,
    lambda_lm: float,
    step_size: float,
) -> np.ndarray:
    system = jacobian.T @ jacobian
    if regularizer is not None and hyper_pvi > 0:
        system = system + float(hyper_pvi) ** 2 * regularizer
    if lambda_lm > 0:
        system = system + float(lambda_lm) * np.eye(system.shape[0])
    rhs = jacobian.T @ target
    try:
        direction = linalg.solve(
            system,
            rhs,
            assume_a="sym",
            check_finite=False,
        )
    except linalg.LinAlgError:
        direction = linalg.lstsq(system, rhs, check_finite=False)[0]
    return float(step_size) * np.asarray(direction, dtype=np.float64)


def nonlinear_stage_physics(
    physics,
    baselines: np.ndarray,
    currents: np.ndarray,
    measured: np.ndarray,
    *,
    regularizer,
    hyper_pvi: float,
    lambda_lm: float,
    step_size: float = 1.0,
    minimum_conductivity: float = 1e-4,
    baseline_voltages: np.ndarray | None = None,
    progress_label: str | None = None,
) -> StagePhysicsBatch:
    """Recompute nonlinear voltage, Jacobian, and LM direction per sample.

    ``voltage_targets`` is the voltage still unexplained by ``currents``.  It is
    therefore the correct target for the analytic correction-amplitude solve.
    """

    baseline_array = np.asarray(baselines, dtype=np.float64)
    current_array = np.asarray(currents, dtype=np.float64)
    voltage_array = np.asarray(measured, dtype=np.float64)
    if not (
        len(baseline_array) == len(current_array) == len(voltage_array)
    ):
        raise ValueError("baseline, current, and voltage counts must match")
    elements = current_array.shape[1]
    reg = _regularizer_array(regularizer, elements)
    cached = None if baseline_voltages is None else np.asarray(
        baseline_voltages, dtype=np.float64
    )
    if cached is not None and len(cached) != len(current_array):
        raise ValueError("baseline voltage cache length does not match")

    directions, jacobians, targets, base_output = [], [], [], []
    residual_rms, step_rms = [], []
    clipped_total = 0
    for index in range(len(current_array)):
        baseline = baseline_array[index]
        current = current_array[index]
        absolute_unclipped = baseline + current
        absolute = np.maximum(absolute_unclipped, float(minimum_conductivity))
        clipped_total += int(np.count_nonzero(absolute != absolute_unclipped))
        baseline_voltage = (
            np.asarray(physics.solve(baseline), dtype=np.float64).ravel()
            if cached is None
            else cached[index]
        )
        voltage_current, jacobian = physics.forward_and_jacobian(absolute)
        voltage_current = np.asarray(voltage_current, dtype=np.float64).ravel()
        jacobian = np.asarray(jacobian, dtype=np.float64)
        predicted = voltage_current - baseline_voltage
        target = voltage_array[index] - predicted
        direction = _solve_direction(
            jacobian,
            target,
            regularizer=reg,
            hyper_pvi=hyper_pvi,
            lambda_lm=lambda_lm,
            step_size=step_size,
        )
        directions.append(direction.astype(np.float32))
        jacobians.append(jacobian.astype(np.float32))
        targets.append(target.astype(np.float32))
        base_output.append(np.asarray(baseline_voltage, dtype=np.float32))
        residual_rms.append(float(np.sqrt(np.mean(target**2))))
        step_rms.append(float(np.sqrt(np.mean(direction**2))))
        if progress_label and (
            index + 1 == len(current_array) or (index + 1) % 25 == 0
        ):
            print(
                f"{progress_label}: physics {index + 1}/{len(current_array)}",
                flush=True,
            )
    return StagePhysicsBatch(
        directions=np.stack(directions),
        jacobians=np.stack(jacobians),
        voltage_targets=np.stack(targets),
        baseline_voltages=np.stack(base_output),
        current_voltage_residual_rms=np.asarray(residual_rms),
        step_rms=np.asarray(step_rms),
        clipped_elements=clipped_total,
    )


def stage_physics_summary(values: StagePhysicsBatch) -> dict[str, float | int]:
    """Compact JSON summary for a stage-physics batch."""

    return {
        "samples": int(len(values.directions)),
        "voltage_target_rms_mean": float(
            np.mean(values.current_voltage_residual_rms)
        ),
        "voltage_target_rms_maximum": float(
            np.max(values.current_voltage_residual_rms)
        ),
        "lm_direction_rms_mean": float(np.mean(values.step_rms)),
        "clipped_elements_total": int(values.clipped_elements),
    }
