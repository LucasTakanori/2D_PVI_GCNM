"""Production-like preprocessing mirroring scionova_01 + scionova_03 (PVI path)."""

from __future__ import annotations

import numpy as np
from scipy.signal import butter, filtfilt


def movmean(x: np.ndarray, window: int, axis: int = -1) -> np.ndarray:
    """MATLAB ``movmean(X, window, axis)`` with centered shrink endpoints."""
    w = max(int(window), 1)
    raw = np.asarray(x)
    arr = np.asarray(raw, dtype=np.result_type(raw.dtype, np.float64))
    moved = np.moveaxis(arr, axis, -1)
    shape = moved.shape
    rows = moved.reshape(-1, shape[-1])
    n = rows.shape[1]
    left = w // 2
    right = w // 2 if w % 2 == 0 else w // 2 + 1
    idx = np.arange(n)
    starts = np.maximum(0, idx - left)
    stops = np.minimum(n, idx + right)
    prefix = np.concatenate(
        [np.zeros((rows.shape[0], 1), dtype=rows.dtype), np.cumsum(rows, axis=1)],
        axis=1,
    )
    out = (prefix[:, stops] - prefix[:, starts]) / (stops - starts)[None, :]
    out = out.reshape(shape)
    return np.moveaxis(out, -1, axis)


def _movmean_1d(x: np.ndarray, w: int) -> np.ndarray:
    return movmean(np.asarray(x), w, axis=0)


def trim_idx0(vmeas: np.ndarray, fs: float, detrend_window: int | None = None) -> np.ndarray:
    """Drop initial samples after movmean detrend (SIGNALOPERATIONS.doFFT idx0)."""
    vm = np.asarray(vmeas)
    if vm.ndim == 1:
        vm = vm[:, None]
    w = int(detrend_window or round(fs))
    idx0 = w + 1
    if vm.shape[1] <= idx0:
        return vm
    return vm[:, idx0 - 1 :]


def butter_lowpass_complex(
    vmeas: np.ndarray,
    fs: float,
    cutoff_hz: float = 5.0,
    order: int = 3,
) -> np.ndarray:
    """scionova_01: separate 5 Hz lowpass on real and imaginary parts."""
    vm = np.asarray(vmeas)
    if vm.ndim == 1:
        vm = vm[:, None]
    nyq = 0.5 * fs
    wn = min(cutoff_hz / nyq, 0.999)
    b, a = butter(order, wn, btype="low")
    padlen = 3 * max(len(a), len(b))
    if vm.shape[1] <= padlen:
        return vm.astype(np.complex128, copy=False)

    out = np.zeros_like(vm, dtype=np.complex128)
    for i in range(vm.shape[0]):
        out[i, :] = filtfilt(b, a, np.real(vm[i, :])) + 1j * filtfilt(b, a, np.imag(vm[i, :]))
    return out


def hp_lp_movmean(
    data: np.ndarray,
    window: int = 100,
    time_axis: int = -1,
) -> tuple[np.ndarray, np.ndarray]:
    """scionova_03: HP = X - movmean(X), LP = movmean(X)."""
    drift = movmean(data, window, axis=time_axis)
    return data - drift, drift


def resistance_reactance(
    vmeas: np.ndarray,
    stim_current: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Match scionova_03: R = -real(v)/I, X = -imag(v)/I."""
    vm = np.asarray(vmeas)
    i = float(stim_current)
    if i == 0:
        raise ValueError("stim_current must be non-zero")
    return -np.real(vm) / i, -np.imag(vm) / i


def process_vmeas_production(
    vmeas: np.ndarray,
    fs: float,
    stim_current: float,
    *,
    detrend_window: int | None = None,
    filter_cutoff_hz: float = 5.0,
    hp_lp_window: int = 100,
) -> dict[str, np.ndarray]:
    """Full production-like chain on complex vmeas (M, T). Returns pvi + pviHP + pviLP."""
    vm = trim_idx0(vmeas, fs, detrend_window=detrend_window)
    vm = butter_lowpass_complex(vm, fs, cutoff_hz=filter_cutoff_hz)

    r, x = resistance_reactance(vm, stim_current)
    pack = {"vmeas": vm, "resistance": r, "reactance": x}

    out_hp: dict[str, np.ndarray] = {}
    out_lp: dict[str, np.ndarray] = {}
    for key, arr in pack.items():
        hp, lp = hp_lp_movmean(arr, window=hp_lp_window, time_axis=1)
        out_hp[key] = hp
        out_lp[key] = lp

    return {
        "pvi": pack,
        "pviHP": out_hp,
        "pviLP": out_lp,
    }
