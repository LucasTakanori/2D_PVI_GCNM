#!/usr/bin/env python3
"""Run complete row-level and cross-family validation for the US120 pilot."""

from __future__ import annotations

import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from gcnm_pvi.preflight_pvi_bp import preflight
from gcnm_pvi.validate_pvi_parquet import validate


def _validate(payload: tuple[str, str, int]):
    name, root, rows = payload
    return name, validate(Path(root), max_rows=rows)


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    roots = {
        "coordinate": root / "gcnm_parquet/us120_pilot_coordinate_hp_lp_v1",
        "global_voltage_slots": (
            root / "gcnm_parquet/us120_pilot_global_voltage_slots_hp_lp_v1"
        ),
    }
    expected_rows = 3382
    with ProcessPoolExecutor(max_workers=2) as executor:
        full = dict(
            executor.map(
                _validate,
                [(name, str(path), expected_rows) for name, path in roots.items()],
            )
        )
    parity = preflight(
        roots["coordinate"],
        roots["global_voltage_slots"],
        root / "data/splits/us120_pilot_subject006_subject010_mask05_v1.json",
        root / "data/registries/main_b045_v1.json",
        expected_subjects=2,
        subjects={"subject006", "subject010"},
    )
    report = {"status": "pass", "full_row_validation": full, "cross_family": parity}
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
