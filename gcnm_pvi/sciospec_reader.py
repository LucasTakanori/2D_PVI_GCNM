"""Offline ScioSpec .eit reader and vmeas extraction (Python port of SCIOSPEC.m)."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

import numpy as np

from gcnm_pvi.electrode_protocol import PviElecConfigs, default_elec_configs


@dataclass
class EitFrame:
    name: str
    idx: int
    stim_current: float
    stim_patterns: np.ndarray  # (n_stim, 2) 1-based electrode ids
    meas_channels: np.ndarray  # 1-based channel ids
    voltage_data: np.ndarray  # (n_stim, n_channels) complex at first frequency


@dataclass
class EitSession:
    frames: list[EitFrame] = field(default_factory=list)
    frame_rate: float | None = None


def _parse_int_list(line: str) -> np.ndarray:
    return np.array([int(x) for x in re.findall(r"\d+", line)], dtype=int)


def read_eit_frame(path: Path | str) -> EitFrame:
    path = Path(path)
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    num_headers = int(lines[0].strip())
    headers = lines[:num_headers]
    name = headers[2].strip()
    idx = int(name.split("_")[-1])
    stim_current = float(headers[8])
    num_freqs = int(headers[7])
    is_log = int(headers[6])

    if num_headers == 11:
        meas_channels = np.arange(1, 17)
    elif num_headers == 18:
        meas_channels = _parse_int_list(headers[-1])
    else:
        raise ValueError(f"Unsupported EIT header count {num_headers} in {path}")

    data_lines = lines[num_headers:]
    while data_lines and not data_lines[-1].strip():
        data_lines.pop()

    num_blocks = len(data_lines) // (num_freqs + 1)
    if num_blocks * (num_freqs + 1) != len(data_lines):
        raise ValueError(
            f"Unexpected EIT data layout in {path}: "
            f"{len(data_lines)} data lines, {num_freqs} freqs/block"
        )

    stim_patterns = np.zeros((num_blocks, 2), dtype=int)
    voltage_blocks = []
    line_idx = 0
    for b in range(num_blocks):
        stim_patterns[b, :] = _parse_int_list(data_lines[line_idx])
        line_idx += 1
        for _ in range(num_freqs):
            vals = np.array([float(x) for x in data_lines[line_idx].split()], dtype=float)
            voltage_blocks.append(
                vals[0::2] + 1j * vals[1::2]
            )  # interleaved real/imag per channel
            line_idx += 1

    # first frequency only (single-freq acquisitions)
    v0 = np.stack(voltage_blocks[0::num_freqs], axis=0)
    return EitFrame(
        name=name,
        idx=idx,
        stim_current=stim_current,
        stim_patterns=stim_patterns,
        meas_channels=meas_channels,
        voltage_data=v0,
    )


def load_eit_session(
    folder: Path | str,
    pattern: str = "*.eit",
    max_frames: int | None = None,
    stride: int = 1,
) -> EitSession:
    folder = Path(folder)
    files = sorted(folder.glob(pattern), key=lambda p: int(p.stem.split("_")[-1]))
    if not files:
        raise FileNotFoundError(f"No .eit files matching {pattern} in {folder}")
    if max_frames is not None:
        files = files[: max_frames * stride : stride]
    elif stride > 1:
        files = files[::stride]
    frames = [read_eit_frame(f) for f in files]
    return EitSession(frames=frames)


def _channel_idx_from_patterns(stim_patterns: np.ndarray) -> np.ndarray:
    items = stim_patterns.ravel()
    return np.sort(np.unique(items)) - 1  # 0-based electrode indices


def frame_to_vmeas(frame: EitFrame, elec_configs: PviElecConfigs) -> np.ndarray:
    """Complex differential vmeas vector (num_meas,).

    Keeping the imaginary component is required for production reactance export.
    Physics callers that solve the real-valued conductivity problem explicitly
    select the real component downstream.
    """
    M = elec_configs.potential_config.extractor
    num_stim = elec_configs.num_stim
    num_meas = elec_configs.num_meas_total
    num_meas_per_stim = elec_configs.num_meas_per_stim

    # Electrode voltages (8) live in the first eight independent channels.
    n_elecs = elec_configs.num_stim
    v_elecs = frame.voltage_data[:, :n_elecs].T  # (n_elecs, n_stim)
    n_stim_frame = v_elecs.shape[1]
    if n_stim_frame != num_stim:
        raise ValueError(
            f"Frame {frame.name}: expected {num_stim} stim blocks, got {n_stim_frame}"
        )

    vmeas = np.zeros(num_meas, dtype=np.complex128)
    for s in range(num_stim):
        rows = np.arange(num_meas_per_stim) + num_meas_per_stim * s
        vmeas[rows] = (M[s, :, :] @ v_elecs[:, s].reshape(-1, 1)).ravel()
    return vmeas


def session_to_vmeas(
    session: EitSession,
    elec_configs: PviElecConfigs | None = None,
) -> np.ndarray:
    """Stack frames -> (num_meas, T) complex vmeas."""
    if elec_configs is None:
        elec_configs = default_elec_configs()
    cols = [frame_to_vmeas(fr, elec_configs) for fr in session.frames]
    return np.stack(cols, axis=1)


def export_vmeas_npy(session_dir: Path | str, out_npy: Path | str, **cfg_kwargs) -> Path:
    session = load_eit_session(session_dir)
    cfg = default_elec_configs(**cfg_kwargs)
    vm = session_to_vmeas(session, cfg)
    out = Path(out_npy)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.save(out, vm)
    return out
