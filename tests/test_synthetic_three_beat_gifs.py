from pathlib import Path

import numpy as np
import pytest

from gcnm_pvi.synthetic_three_beat_gifs import (
    _comparison,
    _shared_limit,
    three_consecutive_beat_indices,
)


def test_three_beat_selection_is_chronological_and_anatomy_local() -> None:
    anatomy = np.repeat([8, 9], 5 * 50)
    beats = np.concatenate(
        [np.repeat(np.arange(base, base + 5), 50) for base in (40, 45)]
    )
    samples = np.tile(np.arange(50), 10)

    indices = three_consecutive_beat_indices(
        anatomy, beats, samples, anatomy=9, first_beat=1
    )

    assert indices.dtype == np.int64
    assert len(indices) == 150
    assert np.all(anatomy[indices] == 9)
    np.testing.assert_array_equal(np.unique(beats[indices]), [46, 47, 48])
    np.testing.assert_array_equal(samples[indices], np.tile(np.arange(50), 3))


def test_three_beat_selection_rejects_incomplete_beat() -> None:
    anatomy = np.zeros(149, dtype=int)
    beats = np.repeat([0, 1, 2], [50, 50, 49])
    samples = np.concatenate((np.arange(50), np.arange(50), np.arange(49)))

    with pytest.raises(ValueError, match="complete 50-sample"):
        three_consecutive_beat_indices(anatomy, beats, samples)


def test_three_beat_selection_rejects_unavailable_window() -> None:
    anatomy = np.zeros(150, dtype=int)
    beats = np.repeat([0, 1, 2], 50)
    samples = np.tile(np.arange(50), 3)

    with pytest.raises(ValueError, match="requested three consecutive beats"):
        three_consecutive_beat_indices(
            anatomy, beats, samples, anatomy=1, first_beat=0
        )


def test_shared_limit_ignores_nan_mesh_exterior() -> None:
    panels = np.array([[[[np.nan, -1.0], [2.0, np.nan]]]])
    assert 1.9 < _shared_limit(panels) <= 2.0


def test_shared_limit_rejects_panels_without_finite_values() -> None:
    with pytest.raises(ValueError, match="no finite values"):
        _shared_limit(np.full((1, 1, 2, 2), np.nan))


def test_comparison_ignores_shared_nonfinite_pixels() -> None:
    metrics = _comparison(
        np.asarray([1.0, 2.0, np.nan]),
        np.asarray([1.0, 2.0, 99.0]),
    )
    assert metrics["nrmse"] == 0.0
    assert metrics["correlation"] == pytest.approx(1.0)


def test_retained_launchers_use_core_synthetic_module() -> None:
    root = Path(__file__).resolve().parents[1]
    launchers = (
        "launch_coordinate_direct_synthetic_gif.sh",
        "launch_synthetic_us120_gifs.sh",
        "launch_train_differential_generalization_winner.sh",
        "launch_coordinate_main_b045_synthetic_gifs.sh",
        "launch_train_us120_hp_lp.sh",
    )
    for launcher in launchers:
        source = (root / "slurm" / launcher).read_text(encoding="utf-8")
        assert "gcnm_pvi.synthetic_three_beat_gifs" in source
        assert "gcnm_pvi.three_beat_gifs" not in source
