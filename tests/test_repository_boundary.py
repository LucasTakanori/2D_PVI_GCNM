"""Guards for the one-way GCNM-core to BP-pipeline repository boundary."""

from __future__ import annotations

import ast
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_core_package_never_imports_bp_pipeline() -> None:
    violations: list[str] = []
    for path in sorted((ROOT / "gcnm_pvi").glob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                names = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom):
                names = [node.module or ""]
            else:
                continue
            if any(name == "pvi_gcnm_bp" or name.startswith("pvi_gcnm_bp.") for name in names):
                violations.append(str(path.relative_to(ROOT)))
    assert not violations


def test_bp_only_modules_are_not_vendored_in_core() -> None:
    names = (
        "train_population_6ch_bp.py",
        "infer_population_6ch_bp.py",
        "pvi_parquet_dataset.py",
        "population_6ch_parquet.py",
        "pvi_splits.py",
        "bp_artifact_gifs.py",
    )
    assert not [name for name in names if (ROOT / "gcnm_pvi" / name).exists()]
