from pathlib import Path

from gcnm_pvi.mesh_registry import RING_IDS, normalize_ring_size, read_subject_rings


def test_ring_normalization_covers_production_range():
    assert normalize_ring_size(6) == "US060"
    assert normalize_ring_size(10.5) == "US105"
    assert normalize_ring_size("13") == "US130"
    assert len(RING_IDS) == 15


def test_workbook_audit_reader():
    workbook = Path(__file__).resolve().parents[1] / "notes_raw.xlsx"
    values = read_subject_rings(workbook)
    assert len(values) == 100
    assert values["subject006"] == 12
    assert values["subject001"] == 10.5
