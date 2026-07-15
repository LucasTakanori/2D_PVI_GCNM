"""Signal preprocessing for PVI vmeas (filtering, HP/LP split)."""

from __future__ import annotations

import numpy as np
from scipy.signal import butter, filtfilt


def lowpass(vmeas: np.ndarray, fs: float, cutoff_hz: float, order: int = 3) -> np.ndarray:
    """Zero-phase lowpass along time (axis 1). Accepts complex input."""
    vm = np.asarray(vmeas)
    if vm.ndim == 1:
        vm = vm[:, None]
    nyq = 0.5 * fs
    wn = min(cutoff_hz / nyq, 0.999)
    b, a = butter(order, wn, btype="low")
    padlen = 3 * max(len(a), len(b))
    if vm.shape[1] <= padlen:
        return vm.copy()
    out = np.zeros_like(vm, dtype=np.complex128 if np.iscomplexobj(vm) else float)
    for i in range(vm.shape[0]):
        out[i, :] = filtfilt(b, a, vm[i, :])
    return out if vmeas.ndim > 1 else out[:, 0]


def hp_lp_split(vmeas: np.ndarray, fs: float, cutoff_hz: float = 5.0):
    """Return (high_pass, low_pass) like scionova HP/LP split (LP = movmean proxy)."""
    lp = lowpass(vmeas, fs, cutoff_hz)
    hp = vmeas - lp
    return hp, lp


def to_real_vmeas(vmeas: np.ndarray) -> np.ndarray:
    vm = np.asarray(vmeas)
    if np.iscomplexobj(vm):
        return np.real(vm)
    return vm.astype(np.float64)
