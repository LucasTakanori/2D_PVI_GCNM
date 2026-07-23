"""Create the selected 30-run mesh matrix and per-run hash manifests."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

from gcnm_pvi.config import GcnmConfig
from gcnm_pvi.mesh_registry import RING_IDS, sha256_file


PROTOCOLS = {
    "coordinate": {
        "source_experiment": "coordinate_beats1000x50_v1_seed0",
        "stages": 2,
        "output": "direct",
        "coordinates": ["x", "y", "radius"],
        "voltage_mlp": False,
        "baseline": {"mode": "homogeneous", "conductivity_s_m": 0.7},
        "loss": {"positive": 1.0, "background": 0.25},
        "checkpoint_selection": "composite",
        "synthetic_dataset": {
            "whole_beats": {"train": 800, "validation": 100, "test": 100},
            "samples_per_beat": 50,
            "samples": {"train": 40000, "validation": 5000, "test": 5000},
            "split_unit": "whole beat",
            "test_forward_model": "exact nonlinear",
        },
        "seed": 0,
    },
    "diffusion": {
        "source_experiment": "finger_default_diffusion_slots_corr010_mlp_seed0",
        "stages": 2,
        "architecture": "diffusion_slots",
        "voltage_mlp": True,
        "baseline": {"mode": "saved-ring-default-finger", "muscle_gate": True},
        "loss": {
            "positive": 1.0, "background": 0.25, "dice": 0.05,
            "slot": 0.5, "separation": 0.15, "attention": 0.05, "correlation": 0.10,
        },
        "samples": {"train": 512, "validation": 128},
        "seed": 0,
    },
    "global_voltage_slots": {
        "source_experiment": "global_voltage_slots_beats1000x50_v1_seed0",
        "stages": 2,
        "architecture": "beat_voltage_slots",
        "voltage_mlp": True,
        "graph_layers_per_stage": 2,
        "beat_context": "rank-one voltage template shared by all 50 beat samples",
        "stage_1_output": "two smooth signed ellipse slots",
        "stage_2_output": "bounded signed vessel-parameter refinement",
        "baseline": {"mode": "homogeneous", "conductivity_s_m": 0.7},
        "loss": {
            "positive": 1.0,
            "background": 0.25,
            "dice": 0.05,
            "slot": 0.2,
            "separation": 0.1,
            "attention": 0.05,
            "correlation": 0.0,
        },
        "synthetic_dataset": {
            "whole_beats": {"train": 800, "validation": 100, "test": 100},
            "samples_per_beat": 50,
            "samples": {"train": 40000, "validation": 5000, "test": 5000},
            "split_unit": "whole beat",
            "test_forward_model": "exact nonlinear",
        },
        "synthetic_generator": {
            "vessel_count": 2,
            "minimum_vessel_gap": 0.06,
            "train_validation_simulation_mode": "linearized",
            "test_simulation_mode": "exact nonlinear",
            "jacobian_bank_size": 8,
        },
        "seed": 0,
    },
}

SELECTED_FAMILIES = ("coordinate", "global_voltage_slots")


def matrix(root: Path) -> dict:
    runs = []
    for family in SELECTED_FAMILIES:
        for ring in RING_IDS:
            runs.append(
                {
                    "family": family,
                    "ring": ring,
                    "model_name": f"{family}_b045_{ring}_seed0",
                    "config": str((root / "configs/rings_b045" / f"{ring}.yaml").resolve()),
                    "dataset": str(
                        (root / "data/mesh_training/beats1000x50_v1" / ring).resolve()
                    ),
                    "models": str(
                        (
                            root
                            / "models/mesh_representations/beats1000x50_v1"
                            / family
                            / ring
                        ).resolve()
                    ),
                    "results": str(
                        (
                            root
                            / "data/mesh_training_results/beats1000x50_v1"
                            / family
                            / ring
                        ).resolve()
                    ),
                    "protocol": PROTOCOLS[family],
                }
            )
    return {
        "schema": "pvi-gcnm-mesh-training-matrix-v3",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "mesh_variant": "b045",
        "run_count": len(runs),
        "runs": runs,
    }


def record_run(family: str, ring: str, config: Path, model_dir: Path, output: Path) -> None:
    model_name = f"{family}_b045_{ring}_seed0"
    suffixes = (
        ["0.pt", "1.pt"]
        if family == "coordinate"
        else ["localizer.pt", "refiner.pt"]
    )
    checkpoints = [model_dir / f"{model_name}_{suffix}" for suffix in suffixes]
    # Component-specific pilots include ``_hp``/``_lp`` in the model name.
    # Preserve the historical name when present, otherwise discover the
    # uniquely matching checkpoint pair in this immutable model directory.
    if not all(path.is_file() for path in checkpoints):
        candidates = sorted(model_dir.glob("*.pt"))
        if family == "coordinate":
            checkpoints = [p for p in candidates if p.name.endswith(("_0.pt", "_1.pt"))]
        else:
            checkpoints = [p for p in candidates if p.name.endswith(("_localizer.pt", "_refiner.pt"))]
        if len(checkpoints) != 2:
            raise FileNotFoundError(checkpoints[0] if checkpoints else model_dir)
    for path in [config, *checkpoints]:
        if not path.is_file():
            raise FileNotFoundError(path)
    cfg = GcnmConfig.from_yaml(config)
    mesh_paths = {
        "forward": Path(cfg.mesh_fwd_h5),
        "inverse": Path(cfg.mesh_inv_h5),
        "mapping_40": Path(cfg.mappings_h5),
    }
    for path in mesh_paths.values():
        if not path.is_file():
            raise FileNotFoundError(path)
    value = {
        "schema": "pvi-gcnm-mesh-run-v3",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "family": family,
        "ring": ring,
        "model_name": model_name,
        "protocol": PROTOCOLS[family],
        "config": str(config.resolve()),
        "config_sha256": sha256_file(config),
        "mesh_files": {
            name: {"path": str(path.resolve()), "sha256": sha256_file(path)}
            for name, path in mesh_paths.items()
        },
        "checkpoints": {str(path.resolve()): sha256_file(path) for path in checkpoints},
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", type=Path, default=root / "data/manifests/mesh_training_v3.json"
    )
    parser.add_argument("--record", action="store_true")
    parser.add_argument("--family", choices=sorted(PROTOCOLS))
    parser.add_argument("--ring", choices=RING_IDS)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--model-dir", type=Path)
    args = parser.parse_args()
    if args.record:
        if not all((args.family, args.ring, args.config, args.model_dir)):
            parser.error("--record requires --family, --ring, --config and --model-dir")
        record_run(args.family, args.ring, args.config, args.model_dir, args.output)
    else:
        value = matrix(root)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
        print(f"wrote {value['run_count']} runs to {args.output}")


if __name__ == "__main__":
    main()
