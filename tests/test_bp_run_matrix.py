from gcnm_pvi.bp_run_matrix import (
    build_original_run_matrix,
    build_pilot_run_matrix,
    build_run_matrix,
    renumber_task_ids,
    write_manifests,
)


def test_mask05_pilot_matrix_has_one_unique_row_per_requested_run(tmp_path):
    subjects = ["subject006", "subject010"]
    rows = build_run_matrix(
        subjects, tmp_path / "coordinate", tmp_path / "global_voltage_slots"
    )

    assert len(rows) == 16
    assert [row["task_id"] for row in rows] == list(range(16))
    assert {row["mask_key"] for row in rows} == {"mask05"}
    assert len(
        {
            (row["subject"], row["family"], row["architecture"], row["output_mode"])
            for row in rows
        }
    ) == 16
    assert {row["target"] for row in rows} == {
        "gcnm-coordinate-3ch-crt-image-to-waveform",
        "gcnm-coordinate-3ch-crt-image-to-fiducials",
        "gcnm-coordinate-3ch-crs-image-to-waveform",
        "gcnm-coordinate-3ch-crs-image-to-fiducials",
        "gcnm-global_voltage_slots-3ch-crt-image-to-waveform",
        "gcnm-global_voltage_slots-3ch-crt-image-to-fiducials",
        "gcnm-global_voltage_slots-3ch-crs-image-to-waveform",
        "gcnm-global_voltage_slots-3ch-crs-image-to-fiducials",
    }


def test_full_two_family_matrix_is_728_but_selected_family_production_is_364(tmp_path):
    subjects = [f"subject{i:03d}" for i in range(1, 92)]
    both = build_run_matrix(subjects, tmp_path / "c", tmp_path / "v")
    selected = build_run_matrix(
        subjects,
        tmp_path / "c",
        tmp_path / "v",
        families=("coordinate",),
    )
    assert len(both) == 728
    assert len(selected) == 364


def test_split_matched_newton_pilot_has_eight_runs():
    rows = build_original_run_matrix(["subject006", "subject010"])
    assert len(rows) == 8
    assert len({(r["subject"], r["architecture"], r["output_mode"]) for r in rows}) == 8
    assert {row["mask_key"] for row in rows} == {"mask05"}


def test_unified_pilot_keeps_four_gpu_scheduler_ceiling(tmp_path):
    rows = build_pilot_run_matrix(
        ["subject006", "subject010"], tmp_path / "c", tmp_path / "v"
    )
    assert len(rows) == 24
    assert [row["task_id"] for row in rows] == list(range(24))
    assert sum(row["representation"] == "gcnm" for row in rows) == 16
    assert sum(row["representation"] == "original" for row in rows) == 8


def test_filtered_matrix_is_renumbered_for_slurm_array_indices(tmp_path):
    rows = build_run_matrix(
        ["subject006", "subject010"], tmp_path / "c", tmp_path / "v"
    )
    crt = renumber_task_ids([row for row in rows if row["architecture"] == "crt"])
    assert len(crt) == 8
    assert [row["task_id"] for row in crt] == list(range(8))


def test_tsv_manifest_uses_unix_newlines(tmp_path):
    rows = renumber_task_ids(
        build_run_matrix(["subject006"], tmp_path / "c", tmp_path / "v")[:1]
    )
    tsv = tmp_path / "runs.tsv"
    write_manifests(rows, tsv, tmp_path / "runs.json", representation="gcnm-3ch")
    assert b"\r" not in tsv.read_bytes()
