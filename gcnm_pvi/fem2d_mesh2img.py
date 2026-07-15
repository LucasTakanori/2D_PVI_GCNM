"""Python port of fem2d_mesh2img.m — build m2i mapping (elem -> N×N pixels)."""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
from scipy.sparse import coo_matrix, csr_matrix

try:
    from shapely.geometry import Polygon

    _HAS_SHAPELY = True
except ImportError:
    _HAS_SHAPELY = False


def _triangle_vol(nodes: np.ndarray, elems_row: np.ndarray) -> float:
    pts = nodes[elems_row, :2]
    v1 = pts[1] - pts[0]
    v2 = pts[2] - pts[0]
    return abs(v1[0] * v2[1] - v1[1] * v2[0]) / 2.0


def _bbox(nodes: np.ndarray, elems_row: np.ndarray) -> tuple[float, float, float, float]:
    pts = nodes[elems_row, :2]
    return float(pts[:, 0].min()), float(pts[:, 0].max()), float(pts[:, 1].min()), float(pts[:, 1].max())


def _rect_overlap(b1, b2) -> bool:
    return b1[0] <= b2[1] and b2[0] <= b1[1] and b1[2] <= b2[3] and b2[2] <= b1[3]


def _clip_polygon_axis(poly: list[np.ndarray], axis: int, bound: float, keep_greater: bool) -> list[np.ndarray]:
    """Clip a polygon against one axis-aligned half-plane."""
    if not poly:
        return []

    def inside(p: np.ndarray) -> bool:
        return bool(p[axis] >= bound) if keep_greater else bool(p[axis] <= bound)

    out: list[np.ndarray] = []
    previous = poly[-1]
    previous_inside = inside(previous)
    for current in poly:
        current_inside = inside(current)
        if current_inside != previous_inside:
            delta = current - previous
            if abs(delta[axis]) > 1e-30:
                t = (bound - previous[axis]) / delta[axis]
                out.append(previous + t * delta)
        if current_inside:
            out.append(current)
        previous = current
        previous_inside = current_inside
    return out


def _triangle_rect_intersection_area(points: np.ndarray, rect) -> float:
    """Exact triangle/axis-aligned rectangle overlap without Shapely."""
    x0, x1, y0, y1 = rect
    poly = [np.asarray(p, dtype=np.float64) for p in points]
    poly = _clip_polygon_axis(poly, 0, x0, True)
    poly = _clip_polygon_axis(poly, 0, x1, False)
    poly = _clip_polygon_axis(poly, 1, y0, True)
    poly = _clip_polygon_axis(poly, 1, y1, False)
    if len(poly) < 3:
        return 0.0
    arr = np.asarray(poly)
    return float(
        0.5
        * abs(
            np.dot(arr[:, 0], np.roll(arr[:, 1], -1))
            - np.dot(arr[:, 1], np.roll(arr[:, 0], -1))
        )
    )


def _img_grid(n: int, xdata: np.ndarray, ydata: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    x_min, x_max = float(xdata.min()), float(xdata.max())
    y_min, y_max = float(ydata.min()), float(ydata.max())
    xvec = np.linspace(x_min, x_max, n + 1)
    yvec = np.linspace(y_max, y_min, n + 1)
    return xvec, yvec


def fem2d_mesh2img(mesh, img_size: int = 40) -> csr_matrix:
    """Return sparse m2i (N^2, K) mapping element sigma to image pixels."""
    nodes = np.asarray(mesh.nodes, dtype=np.float64)
    elems = np.asarray(mesh.elems, dtype=np.int64)
    n_e = elems.shape[0]
    n = int(img_size)

    xvec, yvec = _img_grid(n, nodes[:, 0], nodes[:, 1])
    tri_vols = np.array([_triangle_vol(nodes, elems[k]) for k in range(n_e)])
    tri_bboxes = [_bbox(nodes, elems[k]) for k in range(n_e)]

    rows, cols, vals = [], [], []
    t0 = time.time()
    for pi in range(n):
        for pj in range(n):
            # MATLAB ``img_make`` stores the y index fastest because its
            # arrays are column-major: p = y + x*N.  The mapping row order
            # must follow that convention because images are later reshaped
            # with ``order="F"``.
            p = pj * n + pi
            x0, x1 = xvec[pj], xvec[pj + 1]
            y0, y1 = yvec[pi + 1], yvec[pi]
            pix = Polygon([(x0, y0), (x1, y0), (x1, y1), (x0, y1)]) if _HAS_SHAPELY else None
            pb = (x0, x1, y0, y1)
            for k in range(n_e):
                if not _rect_overlap(pb, tri_bboxes[k]):
                    continue
                pts = nodes[elems[k, :3], :2]
                if _HAS_SHAPELY:
                    tri = Polygon(pts)
                    if not pix.intersects(tri):
                        continue
                    inter = pix.intersection(tri)
                    area = inter.area if not inter.is_empty else 0.0
                else:
                    area = _triangle_rect_intersection_area(pts, pb)
                if area > 0 and tri_vols[k] > 0:
                    rows.append(p)
                    cols.append(k)
                    vals.append(area / tri_vols[k])

    m = coo_matrix((vals, (rows, cols)), shape=(n * n, n_e)).tocsr()
    row_sums = np.asarray(m.sum(axis=1)).ravel()
    for i in range(m.shape[0]):
        if row_sums[i] > 0:
            m[i, :] /= row_sums[i]
    print(
        f"FEM2D: built m2i ({m.shape[0]} x {m.shape[1]}) "
        f"for img_size={n} in {time.time() - t0:.1f}s"
    )
    return m


def save_m2i_hdf5(
    m2i: csr_matrix,
    out_path: Path | str,
    laplace_src: Path | str | None = None,
) -> Path:
    """Write mat2D/m2i (+ optional laplace copy) in PVI HDF5 format."""
    import h5py

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.exists():
        out_path.unlink()

    def _write_sparse(g, name: str, mat: csr_matrix):
        mat = mat.tocoo()
        grp = g.create_group(name)
        grp.create_dataset("rows", data=mat.row + 1)
        grp.create_dataset("cols", data=mat.col + 1)
        grp.create_dataset("values", data=mat.data)
        grp.create_dataset("size", data=np.array(mat.shape, dtype=np.int64))

    with h5py.File(out_path, "w") as f:
        g = f.create_group("mat2D")
        _write_sparse(g, "m2i", m2i)
        if laplace_src and Path(laplace_src).is_file():
            with h5py.File(laplace_src, "r") as src:
                for key in ("laplace", "c2f"):
                    if f"mat2D/{key}" in src:
                        src_g = src[f"mat2D/{key}"]
                        dst = g.create_group(key)
                        for ds in src_g:
                            dst.create_dataset(ds, data=src_g[ds][()])
    return out_path
