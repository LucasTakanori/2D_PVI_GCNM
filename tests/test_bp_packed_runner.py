from gcnm_pvi.bp_packed_runner import _complete, _postprocess_ready


def _row():
    return {
        "target": "gcnm-coordinate-3ch-crt-image-to-waveform",
        "subject": "subject001",
    }


def test_postprocess_ready_requires_native_training_outputs_but_no_statistics(tmp_path):
    row = _row()
    main = tmp_path / row["target"] / "main"
    for folder, suffix in (
        ("checkpoints", "checkpoints.pth"),
        ("history", "history.csv"),
        ("results", "results.csv"),
    ):
        path = main / folder / f"{row['subject']}_{suffix}"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("ready")
    assert _postprocess_ready(tmp_path, row)
    assert not _complete(tmp_path, row)

    statistics = main / "statistics" / f"{row['subject']}_statistics.json"
    statistics.parent.mkdir(parents=True)
    statistics.write_text("{}")
    assert not _postprocess_ready(tmp_path, row)
