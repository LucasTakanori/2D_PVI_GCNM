"""Shared temporal contract for synthetic and archived PVI HP/LP signals.

The production MATLAB pipeline first applies a third-order, zero-phase 5 Hz
Butterworth filter to each real voltage channel.  It then separates every PVI
field with ``movmean(..., 100)``.  This module keeps the Python implementation
in one place so dataset generation, cohort audits, and real-data export cannot
silently use different boundary or reference rules.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.signal import butter, filtfilt


PRODUCTION_BUTTERWORTH_ORDER = 3
PRODUCTION_CUTOFF_HZ = 5.0
PRODUCTION_MOVMEAN_WINDOW = 100
PRODUCTION_CURRENT_A = -0.01


def _normalize_axis(axis: int, ndim: int) -> int:
    axis = int(axis)
    if axis < 0:
        axis += ndim
    if axis < 0 or axis >= ndim:
        raise np.AxisError(axis, ndim=ndim)
    return axis


def resistance_to_voltage(
    resistance: np.ndarray, current_a: float = PRODUCTION_CURRENT_A
) -> np.ndarray:
    """Apply the production sign convention ``delta V = -I * delta R``."""

    return -float(current_a) * np.asarray(resistance, dtype=np.float64)


def matlab_movmean(
    values: np.ndarray,
    window: int = PRODUCTION_MOVMEAN_WINDOW,
    *,
    axis: int = -1,
) -> np.ndarray:
    """Match MATLAB ``movmean(A, window, dim)`` including shortened ends.

    For an even window MATLAB centers the interval between the current and
    previous samples.  A window of 100 therefore uses up to 50 samples before
    the current index, the current sample, and 49 samples after it.  Endpoints
    use only the samples that exist (MATLAB's default ``Endpoints='shrink'``).
    """

    array = np.asarray(values)
    if window <= 0:
        raise ValueError("moving-mean window must be positive")
    if array.ndim == 0:
        raise ValueError("moving mean requires at least one dimension")
    axis = _normalize_axis(axis, array.ndim)
    length = array.shape[axis]
    if length == 0:
        return array.astype(np.result_type(array.dtype, np.float64), copy=True)

    work = np.moveaxis(array, axis, -1).astype(
        np.result_type(array.dtype, np.float64), copy=False
    )
    left = window // 2
    right = window - left - 1
    indices = np.arange(length, dtype=np.int64)
    starts = np.maximum(indices - left, 0)
    stops = np.minimum(indices + right + 1, length)

    prefix = np.concatenate(
        (np.zeros((*work.shape[:-1], 1), dtype=work.dtype), np.cumsum(work, axis=-1)),
        axis=-1,
    )
    totals = np.take(prefix, stops, axis=-1) - np.take(prefix, starts, axis=-1)
    counts = (stops - starts).astype(np.float64)
    result = totals / counts
    return np.moveaxis(result, -1, axis)


def butterworth_lowpass_5hz(
    values: np.ndarray,
    *,
    sampling_rate_hz: float,
    axis: int = -1,
    cutoff_hz: float = PRODUCTION_CUTOFF_HZ,
    order: int = PRODUCTION_BUTTERWORTH_ORDER,
) -> np.ndarray:
    """Apply the production Butterworth design with zero-phase filtering."""

    if sampling_rate_hz <= 0:
        raise ValueError("sampling rate must be positive")
    if cutoff_hz <= 0 or cutoff_hz >= sampling_rate_hz / 2:
        raise ValueError("cutoff must lie strictly between 0 and Nyquist")
    array = np.asarray(values, dtype=np.float64)
    axis = _normalize_axis(axis, array.ndim)
    minimum = 3 * (order + 1)
    if array.shape[axis] <= minimum:
        raise ValueError(
            f"zero-phase order-{order} filtering requires more than {minimum} samples"
        )
    b, a = butter(order, cutoff_hz / (sampling_rate_hz / 2.0), btype="low")
    return filtfilt(b, a, array, axis=axis)


def centered_difference(values: np.ndarray, *, axis: int = -1) -> np.ndarray:
    """Match ``BasePviLearner._compute_diff`` with zero-valued endpoints."""

    array = np.asarray(values)
    axis = _normalize_axis(axis, array.ndim)
    output = np.zeros_like(array)
    if array.shape[axis] < 3:
        return output
    middle = [slice(None)] * array.ndim
    previous = [slice(None)] * array.ndim
    following = [slice(None)] * array.ndim
    middle[axis] = slice(1, -1)
    previous[axis] = slice(None, -2)
    following[axis] = slice(2, None)
    output[tuple(middle)] = (
        array[tuple(following)] - array[tuple(previous)]
    ) / 2.0
    return output


def component_reference(
    values: np.ndarray, *, reference_index: int = 0, axis: int = 0
) -> np.ndarray:
    """Subtract one frozen frame from every frame of a component sequence."""

    array = np.asarray(values)
    axis = _normalize_axis(axis, array.ndim)
    if not -array.shape[axis] <= reference_index < array.shape[axis]:
        raise IndexError("component reference index is outside the sequence")
    reference = np.take(array, reference_index, axis=axis)
    return array - np.expand_dims(reference, axis=axis)


@dataclass(frozen=True)
class HpLpComponents:
    """Full, low-pass, and high-pass arrays before component referencing."""

    full: np.ndarray
    hp: np.ndarray
    lp: np.ndarray


def split_hp_lp(
    values: np.ndarray,
    *,
    window: int = PRODUCTION_MOVMEAN_WINDOW,
    axis: int = 0,
) -> HpLpComponents:
    """Separate an already acquisition-filtered continuous signal."""

    full = np.asarray(values, dtype=np.float64)
    lp = matlab_movmean(full, window=window, axis=axis)
    hp = full - lp
    return HpLpComponents(full=full, hp=hp, lp=lp)


def referenced_hp_lp(
    values: np.ndarray,
    *,
    window: int = PRODUCTION_MOVMEAN_WINDOW,
    time_axis: int = 0,
    reference_index: int = 0,
) -> HpLpComponents:
    """Split a signal and apply the same per-component reference rule.

    The returned invariant is exact up to floating-point roundoff:

    ``hp + lp == full - full[reference_index]``.
    """

    separated = split_hp_lp(values, window=window, axis=time_axis)
    return HpLpComponents(
        full=component_reference(
            separated.full, reference_index=reference_index, axis=time_axis
        ),
        hp=component_reference(
            separated.hp, reference_index=reference_index, axis=time_axis
        ),
        lp=component_reference(
            separated.lp, reference_index=reference_index, axis=time_axis
        ),
    )
