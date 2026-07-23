import json

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from gcnm_pvi.export_pvi_parquet import _fixed_list, _schema
from gcnm_pvi.preflight_pvi_bp import EXPECTED_SHAPES, preflight
from gcnm_pvi.pvi_splits import stable_sample_id


def _representation(root, family, rows):
    shards = root / "shards"
    shards.mkdir(parents=True)
    count = len(rows)
    width = 40 * 40 * 250
    table = pa.Table.from_arrays(
        [
            _fixed_list(np.ones((count, width), dtype=np.float32) * 3, width),
            _fixed_list(np.ones((count, width), dtype=np.float32) * 2, width),
            _fixed_list(np.zeros((count, width), dtype=np.float32), width),
            _fixed_list(np.ones((count, width), dtype=np.float32), width),
            _fixed_list(np.zeros((count, 50), dtype=np.float32), 50),
            _fixed_list(np.zeros((count, 10), dtype=np.float32), 10),
            pa.array([row["subject"] for row in rows]),
            pa.array([row["session"] for row in rows]),
            pa.array([row["source_name"] for row in rows]),
            pa.array([row["sample_id"] for row in rows]),
            pa.array([row["source_order"] for row in rows], type=pa.int32()),
            pa.array([row["mask_start"] for row in rows], type=pa.int32()),
            pa.array([row["mask_stop"] for row in rows], type=pa.int32()),
            pa.array([row["num_periods"] for row in rows], type=pa.int32()),
        ],
        schema=_schema(),
    )
    pq.write_table(table, shards / "subject001_baseline.parquet", compression="zstd")
    manifest = {
        "representation_family": family,
        "mask_key": "mask05",
        "mesh_variant": "b045",
        "stage_columns": {
            "hp_s1": "HP stage 1",
            "hp_s2": "HP stage 2",
            "lp_s1": "LP stage 1",
            "lp_s2": "LP stage 2",
        },
        "source_signal": {
            "hp_resistance": "data/pviHP/resistance",
            "lp_resistance": "data/pviLP/resistance",
            "reference_rule": "subtract first frame independently within each mask05 window",
        },
        "tensor_shapes": EXPECTED_SHAPES,
        "row_count": count,
    }
    (root / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")


def _inputs(tmp_path):
    rows = []
    for start in (0, 10):
        rows.append(
            {
                "sample_id": stable_sample_id(
                    "subject001_baseline", "mask05", start, start + 5
                ),
                "subject": "subject001",
                "session": "baseline",
                "source_name": "subject001_baseline",
                "source_order": 0,
                "mask_start": start,
                "mask_stop": start + 5,
                "num_periods": 20,
            }
        )
    coordinate = tmp_path / "coordinate_v2"
    vessel = tmp_path / "global_voltage_slots_v2"
    _representation(coordinate, "coordinate", rows)
    _representation(vessel, "global_voltage_slots", rows)
    registry = tmp_path / "registry.json"
    registry.write_text(
        json.dumps({"records": [{"subject": "subject001", "exclusion_reason": None}]}),
        encoding="utf-8",
    )
    split = tmp_path / "splits.json"
    split.write_text(
        json.dumps(
            {
                "mask_key": "mask05",
                "assignments": {
                    rows[0]["sample_id"]: "train",
                    rows[1]["sample_id"]: "test",
                },
            }
        ),
        encoding="utf-8",
    )
    return coordinate, vessel, split, registry


def test_bp_preflight_accepts_exact_mask05_parity(tmp_path):
    coordinate, vessel, split, registry = _inputs(tmp_path)
    report = preflight(coordinate, vessel, split, registry, expected_subjects=1)
    assert report["status"] == "pass"
    assert report["subject_count"] == 1
    assert report["sample_count"] == 2


def test_bp_preflight_rejects_incomplete_representation(tmp_path):
    coordinate, vessel, split, registry = _inputs(tmp_path)
    (vessel / "_INCOMPLETE").write_text("export in progress\n", encoding="utf-8")
    with pytest.raises(ValueError, match="incomplete"):
        preflight(coordinate, vessel, split, registry, expected_subjects=1)
