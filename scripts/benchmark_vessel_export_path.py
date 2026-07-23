#!/usr/bin/env python3
"""Time the exact HP/LP global-vessel export path on one real mask05 window."""

from __future__ import annotations

import json
import time
from pathlib import Path

import h5py
import numpy as np

from gcnm_pvi.export_pvi_parquet import _frame_beat_templates
from gcnm_pvi.hp_lp_signals import component_reference, resistance_to_voltage
from gcnm_pvi.representations import GlobalVoltageVesselSlotReconstructor


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    registry = json.loads(
        (root / "data/registries/main_b045_v1.json").read_text(encoding="utf-8")
    )
    record = next(
        item for item in registry["records"]
        if item["source_name"] == "subject006_baseline"
    )
    with h5py.File(record["source_hdf5"], "r") as handle:
        start, stop = (int(value) for value in np.asarray(handle["masks/mask05"])[0])
        start -= 1
        frame_indices = np.arange(start * 50, stop * 50)
        components = {
            name: resistance_to_voltage(
                np.asarray(handle[f"data/pvi{name.upper()}/resistance"])[:, frame_indices]
            ).T
            for name in ("hp", "lp")
        }

    output = {"source_name": record["source_name"], "frames_per_component": 250}
    for component, voltage in components.items():
        name = f"global_voltage_slots_{component}_b045_US120_seed0"
        model_dir = root / "models/hp_lp_us120_v1/global_voltage_slots" / component / "US120"
        started = time.perf_counter()
        reconstructor = GlobalVoltageVesselSlotReconstructor(
            Path(record["config"]),
            model_dir / f"{name}_localizer.pt",
            model_dir / f"{name}_refiner.pt",
            device="cuda:0",
        )
        initialization = time.perf_counter() - started
        referenced = component_reference(voltage, reference_index=0, axis=0)
        template = _frame_beat_templates(referenced)
        stage_1, stage_2, report = reconstructor.reconstruct(referenced, template)
        difference = stage_2 - stage_1
        output[component] = {
            "initialization_seconds": initialization,
            "timings_seconds": report["timings_seconds"],
            "stage_1_rms": float(np.sqrt(np.mean(stage_1 * stage_1))),
            "stage_2_rms": float(np.sqrt(np.mean(stage_2 * stage_2))),
            "stage_2_minus_stage_1_rms": float(np.sqrt(np.mean(difference * difference))),
            "stage_outputs_are_distinct": bool(np.any(stage_1 != stage_2)),
        }
    total_frames = 500
    inference_seconds = sum(
        output[name]["timings_seconds"]["total"] for name in ("hp", "lp")
    )
    output["aggregate_component_frames_per_second"] = total_frames / inference_seconds
    print(json.dumps(output, indent=2), flush=True)


if __name__ == "__main__":
    main()
