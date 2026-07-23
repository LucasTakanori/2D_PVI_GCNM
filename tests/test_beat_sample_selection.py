import numpy as np

from gcnm_pvi.generate_multisubject_beat_dataset import _selected_samples


def test_all_50_samples_are_kept_in_chronological_order() -> None:
    waveform = np.sin(np.linspace(0.0, 2.0 * np.pi, 50, endpoint=False))
    sample_grid = np.arange(50)

    selected = _selected_samples(
        waveform, sample_grid, count=50, rng=np.random.default_rng(0)
    )

    np.testing.assert_array_equal(selected, sample_grid)


def test_subsampled_selection_is_unique_and_ordered() -> None:
    waveform = np.sin(np.linspace(0.0, 2.0 * np.pi, 50, endpoint=False))
    sample_grid = np.arange(0, 50, 5)

    selected = _selected_samples(
        waveform, sample_grid, count=4, rng=np.random.default_rng(0)
    )

    assert len(selected) == 4
    assert len(np.unique(selected)) == 4
    assert np.all(np.diff(selected) > 0)
