"""Stable constructors for electrode protocols used by the GCNM core."""

from __future__ import annotations

from gcnm_pvi.paths import ensure_pvi_solver_on_path

ensure_pvi_solver_on_path()
from pvi_configs import PviElecConfigs  # noqa: E402


def default_elec_configs(
    num_elecs: int = 8,
    stim_current: float = 1e-3,
    stim_pattern: tuple[int, int] = (0, 3),
    meas_pattern: tuple[int, int] = (0, 1),
) -> PviElecConfigs:
    """Build the canonical 8-electrode, 32-measurement GCNM protocol."""
    cfg = PviElecConfigs(
        num_elecs=num_elecs,
        stim_current=stim_current,
        stim_pattern=stim_pattern,
        meas_pattern=meas_pattern,
    )
    assert cfg.num_meas_total == 32, f"Expected 32 meas, got {cfg.num_meas_total}"
    return cfg
