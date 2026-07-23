"""Fixed-baseline physics helpers for the core-guided PVI-GCNM."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import linalg, sparse


@dataclass(frozen=True)
class FixedLinearOperator:
    """Linear LM operator at one homogeneous ring baseline."""

    baseline_voltage: np.ndarray
    jacobian: np.ndarray
    inverse_jacobian: np.ndarray

    def directions(
        self,
        measured: np.ndarray,
        current: np.ndarray | None = None,
    ) -> np.ndarray:
        voltage = np.asarray(measured, dtype=np.float64)
        if current is None:
            residual_target = voltage
        else:
            state = np.asarray(current, dtype=np.float64)
            residual_target = voltage - state @ self.jacobian.T
        return residual_target @ self.inverse_jacobian.T


def build_fixed_linear_operator(
    physics,
    baseline: np.ndarray,
    *,
    regularizer,
    hyper_pvi: float,
    lambda_lm: float,
) -> FixedLinearOperator:
    """Build ``(J^T J + regularization)^-1 J^T`` once per mesh."""

    baseline = np.asarray(baseline, dtype=np.float64).ravel()
    voltage, jacobian = physics.forward_and_jacobian(baseline)
    jacobian = np.asarray(jacobian, dtype=np.float64)
    system = jacobian.T @ jacobian
    if regularizer is not None and hyper_pvi > 0:
        matrix = (
            regularizer.toarray()
            if sparse.issparse(regularizer)
            else np.asarray(regularizer)
        )
        system = system + float(hyper_pvi) ** 2 * np.asarray(
            matrix, dtype=np.float64
        )
    if lambda_lm > 0:
        system = system + float(lambda_lm) * np.eye(system.shape[0])
    try:
        inverse_jacobian = linalg.solve(
            system,
            jacobian.T,
            assume_a="sym",
            check_finite=False,
        )
    except linalg.LinAlgError:
        inverse_jacobian = linalg.lstsq(
            system, jacobian.T, check_finite=False
        )[0]
    return FixedLinearOperator(
        baseline_voltage=np.asarray(voltage, dtype=np.float64).ravel(),
        jacobian=jacobian,
        inverse_jacobian=np.asarray(inverse_jacobian, dtype=np.float64),
    )


def normalized_context_direction(direction: np.ndarray) -> np.ndarray:
    """Return a sign-invariant, per-beat physics localization feature."""

    magnitude = np.abs(np.asarray(direction, dtype=np.float64))
    scale = np.percentile(magnitude, 95.0, axis=1, keepdims=True)
    normalized = magnitude / np.maximum(scale, 1e-12)
    return np.clip(normalized, 0.0, 5.0)


def linear_amplitude_calibration(
    maps: np.ndarray,
    measured: np.ndarray,
    jacobian: np.ndarray,
    *,
    maximum_scale: float = 2.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Fit one non-negative conductivity scale per sample in voltage space."""

    images = np.asarray(maps, dtype=np.float64)
    voltage = np.asarray(measured, dtype=np.float64)
    predicted_voltage = images @ np.asarray(jacobian, dtype=np.float64).T
    numerator = np.sum(predicted_voltage * voltage, axis=1)
    denominator = np.sum(predicted_voltage**2, axis=1)
    scale = np.divide(
        numerator,
        denominator,
        out=np.zeros_like(numerator),
        where=denominator > 1e-24,
    )
    scale = np.clip(scale, 0.0, float(maximum_scale))
    calibrated = images * scale[:, None]
    residual = np.sqrt(
        np.mean((scale[:, None] * predicted_voltage - voltage) ** 2, axis=1)
    )
    return calibrated, scale, residual
