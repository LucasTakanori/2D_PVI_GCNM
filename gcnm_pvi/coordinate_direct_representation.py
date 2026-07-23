"""Three-channel BP representation derived from coordinate-direct GCNM stages."""

from __future__ import annotations

import numpy as np


CHANNEL_NAMES = ("s1", "s2", "d_s2_dt")


def centered_temporal_difference(values: np.ndarray) -> np.ndarray:
    """Match pvi_ml's centered difference and zero-padded time boundaries."""

    array = np.asarray(values)
    clean = np.nan_to_num(array, nan=0.0)
    output = np.zeros_like(clean)
    output[..., 1:-1] = (clean[..., 2:] - clean[..., :-2]) / 2.0
    if array.ndim >= 3:
        outside = ~np.isfinite(array[..., 0])
        output[outside, :] = np.nan
    return output


def compose_s1_s2_ds2(stage_1: np.ndarray, stage_2: np.ndarray) -> np.ndarray:
    """Return ``[S1, S2, dS2/dt]`` with channel-first image layout.

    Inputs must be rasterized as ``(height, width, frames)``. The derivative is
    not learned and is computed only after all ordered frames are reconstructed.
    """

    first = np.asarray(stage_1)
    second = np.asarray(stage_2)
    if first.shape != second.shape or first.ndim != 3:
        raise ValueError("S1/S2 must share (height, width, frames) shape")
    return np.stack((first, second, centered_temporal_difference(second)), axis=0)


def representation_diagnostics(channels: np.ndarray) -> dict:
    values = np.asarray(channels)
    if values.ndim != 4 or values.shape[0] != 3:
        raise ValueError("representation must have (3, height, width, frames) shape")
    reports = {}
    for index, name in enumerate(CHANNEL_NAMES):
        finite = values[index][np.isfinite(values[index])].astype(np.float64)
        reports[name] = {
            "finite_values": int(len(finite)),
            "rms": float(np.sqrt(np.mean(finite * finite))),
            "standard_deviation": float(np.std(finite)),
            "minimum": float(np.min(finite)),
            "maximum": float(np.max(finite)),
            "negative_fraction": float(np.mean(finite < 0)),
        }
    reports["shape"] = list(values.shape)
    reports["all_channels_nonzero"] = all(
        reports[name]["rms"] > 0 for name in CHANNEL_NAMES
    )
    return reports
