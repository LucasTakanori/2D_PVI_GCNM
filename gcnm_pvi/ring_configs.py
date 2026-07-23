"""Generate immutable ring-specific YAML configurations from one protocol file."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import yaml

from gcnm_pvi.mesh_registry import RING_IDS


def write_ring_configs(template: Path, mesh_root: Path, output_root: Path) -> list[Path]:
    template = Path(template).resolve()
    raw = yaml.safe_load(template.read_text(encoding="utf-8"))
    solver = Path(raw["pvi_solver_root"])
    if not solver.is_absolute():
        solver = (template.parent / solver).resolve()
    output_root = Path(output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    mesh_root = Path(mesh_root).resolve()
    raw["pvi_solver_root"] = os.path.relpath(solver, output_root)
    outputs = []
    for ring in RING_IDS:
        base = mesh_root / ring
        config = dict(raw)
        config["mesh_fwd_h5"] = os.path.relpath(
            base / f"ring_{ring}_fwd.h5", output_root
        )
        config["mesh_inv_h5"] = os.path.relpath(
            base / f"ring_{ring}_inv.h5", output_root
        )
        config["mappings_h5"] = os.path.relpath(
            base / f"ring_{ring}_mappings_40.h5", output_root
        )
        config["model_name"] = f"gcnm_b045_{ring}"
        path = output_root / f"{ring}.yaml"
        path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
        outputs.append(path)
    return outputs


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--template", type=Path, default=root / "configs/subject006_anatomical_gcnm.yaml"
    )
    parser.add_argument(
        "--mesh-root", type=Path, default=root / "data/ring_meshes/collections/b045"
    )
    parser.add_argument("--output-root", type=Path, default=root / "configs/rings_b045")
    args = parser.parse_args()
    outputs = write_ring_configs(args.template, args.mesh_root, args.output_root)
    print(f"wrote {len(outputs)} ring configurations to {args.output_root}")


if __name__ == "__main__":
    main()
