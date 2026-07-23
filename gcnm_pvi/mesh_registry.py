"""Build and validate the versioned subject/session to b045 mesh registry.

The workbook reader intentionally uses only the Python standard library so the
registry can be checked on login nodes before the scientific environment is
activated.  Mapping-shape validation uses h5py when available.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import xml.etree.ElementTree as ET
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


REGISTRY_VERSION = 1
RING_IDS = tuple(f"US{size:03d}" for size in range(60, 131, 5))
SESSION_RE = re.compile(
    r"^(subject\d{3})_(baseline|valsalva|pressor)_masked\.h5$", re.IGNORECASE
)
SESSION_ORDER = {"baseline": 0, "valsalva": 1, "pressor": 2}
_XML_NS = "{http://schemas.openxmlformats.org/spreadsheetml/2006/main}"


def sha256_file(path: Path, block_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(block_size), b""):
            digest.update(block)
    return digest.hexdigest()


def normalize_ring_size(value: Any) -> str:
    """Convert workbook sizes such as 12 or 10.5 to US120 or US105."""
    if value is None or str(value).strip() == "":
        raise ValueError("ring size is missing")
    try:
        tenths = int(round(float(value) * 10.0))
    except (TypeError, ValueError) as exc:
        raise ValueError(f"invalid ring size {value!r}") from exc
    ring = f"US{tenths:03d}"
    if ring not in RING_IDS:
        raise ValueError(f"ring size {value!r} normalizes to unsupported {ring}")
    return ring


def _column_index(cell_ref: str) -> int:
    letters = "".join(ch for ch in cell_ref if ch.isalpha())
    result = 0
    for ch in letters.upper():
        result = result * 26 + ord(ch) - ord("A") + 1
    return result - 1


def _xlsx_rows(path: Path, sheet_name: str) -> list[list[Any]]:
    """Read the small audit sheet without an openpyxl runtime dependency."""
    with zipfile.ZipFile(path) as archive:
        shared: list[str] = []
        if "xl/sharedStrings.xml" in archive.namelist():
            root = ET.fromstring(archive.read("xl/sharedStrings.xml"))
            for item in root.findall(f"{_XML_NS}si"):
                shared.append("".join(node.text or "" for node in item.iter(f"{_XML_NS}t")))

        workbook = ET.fromstring(archive.read("xl/workbook.xml"))
        relationships = ET.fromstring(archive.read("xl/_rels/workbook.xml.rels"))
        rel_targets = {
            rel.attrib["Id"]: rel.attrib["Target"]
            for rel in relationships
        }
        target = None
        rel_ns = "{http://schemas.openxmlformats.org/officeDocument/2006/relationships}id"
        for sheet in workbook.find(f"{_XML_NS}sheets") or []:
            if sheet.attrib.get("name") == sheet_name:
                target = rel_targets[sheet.attrib[rel_ns]]
                break
        if target is None:
            raise KeyError(f"workbook has no sheet {sheet_name!r}")
        target = target.lstrip("/")
        if not target.startswith("xl/"):
            target = f"xl/{target}"
        root = ET.fromstring(archive.read(target))
        output: list[list[Any]] = []
        for row in root.iter(f"{_XML_NS}row"):
            values: dict[int, Any] = {}
            for cell in row.findall(f"{_XML_NS}c"):
                index = _column_index(cell.attrib["r"])
                kind = cell.attrib.get("t")
                value_node = cell.find(f"{_XML_NS}v")
                if kind == "inlineStr":
                    value = "".join(n.text or "" for n in cell.iter(f"{_XML_NS}t"))
                elif value_node is None:
                    value = None
                elif kind == "s":
                    value = shared[int(value_node.text)]
                elif kind in {"str", "e"}:
                    value = value_node.text
                else:
                    number = float(value_node.text)
                    value = int(number) if number.is_integer() else number
                values[index] = value
            width = max(values, default=-1) + 1
            output.append([values.get(index) for index in range(width)])
    return output


def read_subject_rings(workbook: Path) -> dict[str, Any]:
    rows = _xlsx_rows(Path(workbook), "audit_raw")
    if not rows:
        raise ValueError("audit_raw is empty")
    headers = [str(value) if value is not None else "" for value in rows[0]]
    required = {"subject_id", "ring_size"}
    if not required.issubset(headers):
        raise KeyError(f"audit_raw must contain {sorted(required)}")
    result: dict[str, Any] = {}
    for row in rows[1:]:
        record = dict(zip(headers, row))
        subject = record.get("subject_id")
        if subject:
            result[str(subject).lower()] = record.get("ring_size")
    return result


def discover_main_sessions(data_root: Path) -> list[dict[str, str]]:
    main = Path(data_root) / "main"
    sessions = []
    for path in sorted(main.glob("*_masked.h5")):
        match = SESSION_RE.match(path.name)
        if match:
            sessions.append(
                {
                    "subject": match.group(1).lower(),
                    "session": match.group(2).lower(),
                    "source_name": path.name.removesuffix("_masked.h5"),
                    "source_hdf5": str(path.resolve()),
                }
            )
    if not sessions:
        raise FileNotFoundError(f"no main HDF5 sessions found under {main}")
    return sorted(
        sessions,
        key=lambda entry: (
            entry["subject"], SESSION_ORDER.get(entry["session"], 99), entry["source_name"]
        ),
    )


def _mapping_shape(path: Path) -> tuple[int, int] | None:
    try:
        import h5py  # type: ignore
        import numpy as np  # type: ignore
    except ImportError:
        return None
    with h5py.File(path, "r") as handle:
        shape = np.asarray(handle["mat2D/m2i/size"]).reshape(-1)
    if len(shape) != 2:
        raise ValueError(f"invalid m2i size metadata in {path}")
    return int(shape[0]), int(shape[1])


def mesh_assets(mesh_root: Path, config_root: Path, ring: str) -> dict[str, str]:
    base = Path(mesh_root) / ring
    return {
        "mesh_forward": str((base / f"ring_{ring}_fwd.h5").resolve()),
        "mesh_inverse": str((base / f"ring_{ring}_inv.h5").resolve()),
        "mapping_40": str((base / f"ring_{ring}_mappings_40.h5").resolve()),
        "mesh_manifest": str((base / "manifest.json").resolve()),
        "config": str((Path(config_root) / f"{ring}.yaml").resolve()),
    }


def build_registry(
    workbook: Path,
    data_root: Path,
    mesh_root: Path,
    config_root: Path,
    *,
    require_configs: bool = True,
) -> dict[str, Any]:
    rings = read_subject_rings(workbook)
    discovered = discover_main_sessions(data_root)
    records, exclusions = [], []
    subjects = sorted({entry["subject"] for entry in discovered})
    for subject in subjects:
        raw_ring = rings.get(subject)
        try:
            ring = normalize_ring_size(raw_ring)
            assets = mesh_assets(mesh_root, config_root, ring)
            required = ["mesh_forward", "mesh_inverse", "mapping_40", "mesh_manifest"]
            if require_configs:
                required.append("config")
            missing = [key for key in required if not Path(assets[key]).is_file()]
            if missing:
                raise FileNotFoundError(f"missing {', '.join(missing)}")
            mapping_shape = _mapping_shape(Path(assets["mapping_40"]))
            if mapping_shape is not None and mapping_shape[0] != 40 * 40:
                raise ValueError(f"m2i shape is {mapping_shape}, expected first dimension 1600")
        except (ValueError, FileNotFoundError) as exc:
            exclusions.append(
                {"subject": subject, "ring_size": raw_ring, "exclusion_reason": str(exc)}
            )
            continue
        for entry in discovered:
            if entry["subject"] == subject:
                records.append(
                    {
                        **entry,
                        "ring_size": raw_ring,
                        "ring": ring,
                        **assets,
                        "source_order": len(records),
                        "exclusion_reason": None,
                    }
                )

    registry = {
        "schema": "pvi-gcnm-mesh-registry-v1",
        "version": REGISTRY_VERSION,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "workbook": str(Path(workbook).resolve()),
        "workbook_sha256": sha256_file(Path(workbook)),
        "data_root": str(Path(data_root).resolve()),
        "mesh_variant": "b045",
        "rings": list(RING_IDS),
        "subject_count": len({record["subject"] for record in records}),
        "session_count": len(records),
        "records": records,
        "excluded_subjects": exclusions,
    }
    if exclusions:
        reasons = "; ".join(f"{x['subject']}: {x['exclusion_reason']}" for x in exclusions)
        raise ValueError(f"main subjects cannot be registered: {reasons}")
    return registry


def write_registry(registry: dict[str, Any], output: Path) -> None:
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(registry, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workbook", type=Path, default=root / "notes_raw.xlsx")
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument(
        "--mesh-root", type=Path, default=root / "data/ring_meshes/collections/b045"
    )
    parser.add_argument("--config-root", type=Path, default=root / "configs/rings_b045")
    parser.add_argument(
        "--output", type=Path, default=root / "data/registries/main_b045_v1.json"
    )
    parser.add_argument("--allow-missing-configs", action="store_true")
    args = parser.parse_args()
    registry = build_registry(
        args.workbook,
        args.data_root,
        args.mesh_root,
        args.config_root,
        require_configs=not args.allow_missing_configs,
    )
    write_registry(registry, args.output)
    print(
        f"wrote {registry['subject_count']} subjects / {registry['session_count']} sessions "
        f"to {args.output}"
    )


if __name__ == "__main__":
    main()
