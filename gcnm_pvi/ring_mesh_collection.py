"""Export subject-sized MATLAB ring mesh collections for the Python PVI runtime."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import h5py
import numpy as np
from scipy import sparse
from scipy.io import loadmat

from gcnm_pvi.fem2d_mesh2img import fem2d_mesh2img


def load_ring_collection(path: Path | str) -> dict:
    """Load the top-level ``collection`` struct from a MATLAB v5 MAT file."""
    data = loadmat(Path(path), simplify_cells=True, variable_names=["collection"])
    collection = data.get("collection")
    if not isinstance(collection, dict):
        raise ValueError(f"No MATLAB struct named 'collection' in {path}")
    return collection


def available_ring_ids(path: Path | str) -> list[str]:
    return sorted(load_ring_collection(path))


def _as_1d(value, dtype=None) -> np.ndarray:
    return np.asarray(value, dtype=dtype).reshape(-1)


def _write_string(group: h5py.Group, name: str, value: str) -> None:
    dtype = h5py.string_dtype(encoding="utf-8")
    group.create_dataset(name, data=np.asarray([[value]], dtype=dtype))


def write_mesh_hdf5(mesh: dict, path: Path | str, name: str) -> Path:
    """Write one MATLAB mesh struct in the layout consumed by load_mesh_hdf5."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "w") as f:
        metadata = f.create_group("metadata")
        _write_string(metadata, "version", "1.0")
        _write_string(metadata, "date", "exported by gcnm_pvi.ring_mesh_collection")

        # pvi_mesh2d.load_mesh_hdf5 transposes these MATLAB-oriented datasets.
        f.create_dataset("nodes", data=np.asarray(mesh["nodes"], dtype=np.float64).T)
        f.create_dataset("elems", data=np.asarray(mesh["elems"], dtype=np.int16).T)
        f.create_dataset("links", data=np.asarray(mesh["links"], dtype=np.int16).T)
        for key in ("nodes_idx_boundary", "elems_idx_boundary", "links_idx_boundary"):
            f.create_dataset(key, data=_as_1d(mesh[key], np.int16)[None, :])
        f.create_dataset("elems_data", data=_as_1d(mesh["elems_data"], np.float64)[None, :])

        elecs = mesh["elecs"]
        if isinstance(elecs, dict):
            elecs = [elecs]
        eg = f.create_group("elecs")
        eg.create_dataset("num_elecs", data=np.asarray([[len(elecs)]], dtype=np.int16))
        for idx, elec in enumerate(elecs, start=1):
            g = eg.create_group(f"elec_{idx}")
            g.create_dataset("impedance", data=np.asarray([[float(np.asarray(elec["impedance"]).item())]]))
            for key in ("nodes", "elems", "links"):
                g.create_dataset(key, data=_as_1d(elec[key], np.int16)[None, :])

        additional = f.create_group("additional")
        _write_string(additional, "name", name)
    return path


def _mapping_csr(matrix) -> sparse.csr_matrix:
    """Convert MATLAB mapping to CSR and compact all-NaN outside-image rows."""
    if sparse.issparse(matrix):
        mat = matrix.tocsr().astype(np.float64)
        dense_nan_rows = np.unique(mat.nonzero()[0][np.isnan(mat.data)]) if np.isnan(mat.data).any() else np.array([], dtype=int)
        if dense_nan_rows.size:
            mat.data[np.isnan(mat.data)] = 0.0
            mat.eliminate_zeros()
            marker = sparse.coo_matrix(
                (np.full(dense_nan_rows.size, np.nan), (dense_nan_rows, np.zeros(dense_nan_rows.size, dtype=int))),
                shape=mat.shape,
            ).tocsr()
            mat = mat + marker
        return mat

    arr = np.asarray(matrix, dtype=np.float64)
    nan_rows = np.all(np.isnan(arr), axis=1)
    arr = np.nan_to_num(arr, nan=0.0)
    mat = sparse.csr_matrix(arr)
    if nan_rows.any():
        rows = np.flatnonzero(nan_rows)
        marker = sparse.coo_matrix(
            (np.full(rows.size, np.nan), (rows, np.zeros(rows.size, dtype=int))),
            shape=mat.shape,
        ).tocsr()
        mat = mat + marker
    return mat


def _mesh_for_mapping(mesh: dict) -> SimpleNamespace:
    """Create the zero-based mesh view expected by fem2d_mesh2img."""
    return SimpleNamespace(
        nodes=np.asarray(mesh["nodes"], dtype=np.float64),
        elems=np.asarray(mesh["elems"], dtype=np.int64) - 1,
    )


def build_image_mapping(mesh: dict, img_size: int) -> sparse.csr_matrix:
    """Build m2i and mark pixels outside the FEM domain as NaN."""
    m2i = fem2d_mesh2img(_mesh_for_mapping(mesh), img_size=img_size).tocsr()
    outside = np.flatnonzero(np.asarray(m2i.sum(axis=1)).ravel() == 0)
    if outside.size:
        marker = sparse.coo_matrix(
            (np.full(outside.size, np.nan), (outside, np.zeros(outside.size, dtype=int))),
            shape=m2i.shape,
        ).tocsr()
        m2i = m2i + marker
    return m2i


def _write_sparse(group: h5py.Group, name: str, matrix) -> None:
    """Write MATLAB-export-compatible COO data.

    The existing Python loader swaps ``rows`` and ``cols`` to account for the
    MATLAB/HDF5 orientation, so the stored coordinates are intentionally swapped.
    """
    mat = sparse.coo_matrix(matrix)
    g = group.create_group(name)
    g.create_dataset("rows", data=mat.col.astype(np.int64) + 1)
    g.create_dataset("cols", data=mat.row.astype(np.int64) + 1)
    g.create_dataset("values", data=mat.data.astype(np.float64))
    g.create_dataset("size", data=np.asarray(mat.shape, dtype=np.int64).reshape(2, 1))


def write_mappings_hdf5(mappings: dict, m2i, path: Path | str) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "w") as f:
        metadata = f.create_group("metadata")
        _write_string(metadata, "version", "1.0")
        _write_string(metadata, "date", "exported by gcnm_pvi.ring_mesh_collection")
        g = f.create_group("mat2D")
        _write_sparse(g, "m2i", m2i)
        _write_sparse(g, "laplace", mappings["laplace"])
        _write_sparse(g, "c2f", mappings["c2f"])
    return path


def export_ring_mesh(
    collection_path: Path | str,
    ring_id: str,
    out_dir: Path | str,
    *,
    img_size: int = 40,
) -> dict[str, Path]:
    collection_path = Path(collection_path).resolve()
    collection = load_ring_collection(collection_path)
    ring_id = ring_id.upper()
    if ring_id not in collection:
        raise ValueError(f"Unknown ring ID {ring_id}; choose from {', '.join(sorted(collection))}")

    item = collection[ring_id]
    inv, fwd, mappings = item["inv"], item["fwd"], item["mappings"]
    source_m2i = mappings["m2i"]
    source_side = int(round(np.sqrt(source_m2i.shape[0])))
    if source_side == img_size:
        m2i = _mapping_csr(source_m2i)
        mapping_source = "collection"
    else:
        m2i = build_image_mapping(inv, img_size)
        mapping_source = "rebuilt"

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "mesh_inv_h5": out_dir / f"ring_{ring_id}_inv.h5",
        "mesh_fwd_h5": out_dir / f"ring_{ring_id}_fwd.h5",
        "mappings_h5": out_dir / f"ring_{ring_id}_mappings_{img_size}.h5",
    }
    write_mesh_hdf5(inv, paths["mesh_inv_h5"], f"ring_{ring_id}_inv")
    write_mesh_hdf5(fwd, paths["mesh_fwd_h5"], f"ring_{ring_id}_fwd")
    write_mappings_hdf5(mappings, m2i, paths["mappings_h5"])

    manifest = {
        "collection": str(collection_path),
        "ring_id": ring_id,
        "img_size": img_size,
        "mapping_source": mapping_source,
        "inverse_nodes": int(np.asarray(inv["nodes"]).shape[0]),
        "inverse_elements": int(np.asarray(inv["elems"]).shape[0]),
        "forward_nodes": int(np.asarray(fwd["nodes"]).shape[0]),
        "forward_elements": int(np.asarray(fwd["elems"]).shape[0]),
        **{key: str(value.resolve()) for key, value in paths.items()},
    }
    manifest_path = out_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    paths["manifest"] = manifest_path
    return paths


def main() -> None:
    p = argparse.ArgumentParser(description="Export a subject-sized PVI ring mesh collection entry")
    p.add_argument("--collection", type=Path, required=True)
    p.add_argument("--ring-id", required=True, help="For example US095")
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--img-size", type=int, default=40)
    p.add_argument("--list", action="store_true", help="List available IDs and exit")
    args = p.parse_args()
    if args.list:
        print("\n".join(available_ring_ids(args.collection)))
        return
    paths = export_ring_mesh(args.collection, args.ring_id, args.out_dir, img_size=args.img_size)
    for key, value in paths.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()
