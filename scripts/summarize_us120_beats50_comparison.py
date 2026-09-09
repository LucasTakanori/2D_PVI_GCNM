"""Summarize the paired US120 50-samples-per-beat evaluations.

The voltage evaluator already writes JSON reports containing exact-nonlinear
residuals and vessel-localization variation. This small post-processing step
keeps those reconstruction values together; it never recomputes or mutates
source data.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def _load(root: Path) -> dict:
    files = sorted(root.rglob("*.json"))
    if not files:
        raise FileNotFoundError(f"no JSON evaluation report under {root}")
    # Prefer the evaluator's top-level report if present.
    preferred = [p for p in files if p.name in {"report.json", "summary.json"}]
    return json.loads((preferred[0] if preferred else files[0]).read_text())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--new-root", type=Path, required=True)
    ap.add_argument("--baseline-root", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    result = {
        "resolution": {"samples_per_beat": 50, "frames_per_beat": 50},
        "new_diffusion_model": _load(args.new_root),
        "accepted_baseline_model": _load(args.baseline_root),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
