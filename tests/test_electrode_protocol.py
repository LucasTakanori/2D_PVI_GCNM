"""Contract tests for the core electrode protocol boundary."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from gcnm_pvi.electrode_protocol import default_elec_configs


def test_default_protocol_preserves_exact_electrode_configuration() -> None:
    cfg = default_elec_configs()

    assert cfg.num_stim == 8
    assert cfg.num_meas_per_stim == 4
    assert cfg.num_meas_total == 32
    assert cfg.stim_config.current == 1e-3
    assert cfg.stim_config.pattern == (0, 3)
    assert cfg.meas_config.pattern == (0, 1)
    np.testing.assert_array_equal(
        cfg.stim_config.pairs,
        np.vstack((np.arange(8), np.roll(np.arange(8), -3))),
    )
    np.testing.assert_array_equal(
        cfg.meas_config.pairs,
        np.vstack((np.arange(8), np.roll(np.arange(8), -1))),
    )
    np.testing.assert_array_equal(
        cfg.meas_config.idx,
        np.asarray(
            [
                [1, 2, 0, 0, 0, 1, 2, 0],
                [4, 5, 3, 1, 1, 2, 3, 3],
                [5, 6, 6, 4, 2, 3, 4, 4],
                [6, 7, 7, 7, 5, 6, 7, 5],
            ]
        ),
    )
    np.testing.assert_array_equal(
        cfg.potential_config.extractor,
        cfg.meas_config.matrix[cfg.meas_config.idx.T],
    )


def test_protocol_rejects_non_32_measurement_configuration() -> None:
    with pytest.raises(AssertionError, match="Expected 32 meas"):
        default_elec_configs(num_elecs=6)


def test_core_smoke_test_imports_protocol_from_core_boundary() -> None:
    source = (
        Path(__file__).resolve().parents[1] / "gcnm_pvi" / "smoke_test.py"
    ).read_text(encoding="utf-8")
    assert "from gcnm_pvi.electrode_protocol import default_elec_configs" in source
    assert "from gcnm_pvi.sciospec_reader import" not in source
