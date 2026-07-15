# gcnm_phantoms.py
"""Synthetic phantoms for GCNM training on PVI meshes."""

import os

import numpy as np


def domain_circle(mesh):
    bnd = mesh.nodes[mesh.nodes_idx_boundary.astype(int)]
    center = bnd.mean(axis=0)
    radius = np.linalg.norm(bnd - center, axis=1).mean()
    return center, radius


def elem_centroids(mesh):
    return mesh.nodes[mesh.elems].mean(axis=1)


def _inside_ellipse(pts, center, a, b, theta):
    d = pts - center
    c, s = np.cos(theta), np.sin(theta)
    u = d @ np.array([c, s])
    v = d @ np.array([-s, c])
    return (u / a) ** 2 + (v / b) ** 2 <= 1.0


def _ellipse_inside_domain(center, a, b, theta, dom_center, dom_radius, margin, n_check=64):
    t = np.linspace(0, 2 * np.pi, n_check, endpoint=False)
    pts = np.stack([a * np.cos(t), b * np.sin(t)], axis=1)
    c, s = np.cos(theta), np.sin(theta)
    R = np.array([[c, -s], [s, c]])
    bd = pts @ R.T + center
    return np.all(np.linalg.norm(bd - dom_center, axis=1) < dom_radius * (1.0 - margin))


def generate_ellipses(n_inclusions, dom_center, dom_radius, rng, **kwargs):
    a_range = kwargs.get("a_range", (0.15, 0.30))
    b_range = kwargs.get("b_range", (0.15, 0.30))
    low_range = kwargs.get("low_range", (0.15, 0.25))
    high_range = kwargs.get("high_range", (0.65, 0.95))
    p_high = kwargs.get("p_high", 0.5)
    gap = kwargs.get("gap", 0.10)
    margin = kwargs.get("margin", 0.05)
    max_attempt = kwargs.get("max_attempt", 5000)

    R = dom_radius
    placed = []
    for _ in range(n_inclusions):
        a = rng.uniform(*a_range) * R
        b = rng.uniform(*b_range) * R
        a, b = max(a, b), min(a, b)
        r_eff = a
        for _ in range(max_attempt):
            rho = (R - r_eff) * np.sqrt(rng.random())
            phi = 2 * np.pi * rng.random()
            center = dom_center + rho * np.array([np.cos(phi), np.sin(phi)])
            theta = rng.uniform(0, 2 * np.pi)
            if not _ellipse_inside_domain(center, a, b, theta, dom_center, R, margin):
                continue
            if any(
                np.linalg.norm(center - e["center"]) < (r_eff + e["r_eff"] + gap * R)
                for e in placed
            ):
                continue
            perm = (
                rng.uniform(*high_range) if rng.random() < p_high else rng.uniform(*low_range)
            )
            placed.append(
                {
                    "center": center,
                    "a": a,
                    "b": b,
                    "theta": theta,
                    "perm": perm,
                    "r_eff": r_eff,
                }
            )
            break
    return placed


def rasterize(ellipses, background, centroids):
    sigma = np.full(centroids.shape[0], background, dtype=np.float64)
    for e in ellipses:
        mask = _inside_ellipse(centroids, e["center"], e["a"], e["b"], e["theta"])
        sigma[mask] = e["perm"]
    return sigma


def generate_perturbation_phantoms(
    mesh_inv,
    n_samples,
    rng,
    bkg=0.7,
    rel_amp_range=(0.02, 0.12),
    n_blobs_range=(1, 3),
):
    """Small conductivity changes around baseline (finger-like dynamic range)."""
    dom_center, dom_radius = domain_circle(mesh_inv)
    cen = elem_centroids(mesh_inv)
    sigma_list = []
    for _ in range(n_samples):
        sigma = np.full(cen.shape[0], bkg, dtype=np.float64)
        n_blobs = rng.integers(n_blobs_range[0], n_blobs_range[1] + 1)
        for _ in range(n_blobs):
            rel = rng.uniform(*rel_amp_range) * (1 if rng.random() < 0.5 else -1)
            ell = generate_ellipses(
                1,
                dom_center,
                dom_radius,
                rng,
                a_range=(0.08, 0.18),
                b_range=(0.08, 0.18),
                low_range=(bkg * (1 + rel), bkg * (1 + rel)),
                high_range=(bkg * (1 + rel), bkg * (1 + rel)),
                p_high=1.0,
            )
            if ell:
                sigma = rasterize(ell, bkg, cen)
        sigma_list.append(sigma)
    return np.stack(sigma_list)


def generate_dataset(
    physics_fwd,
    mesh_inv,
    n_samples,
    rng,
    mode="inclusion",
    n_inclusions_range=(1, 4),
    bkg_range=(0.40, 0.43),
    noise_scale=0.005,
    out_dir=None,
    **ellipse_kwargs,
):
    cen_fwd = elem_centroids(physics_fwd.mesh)
    cen_inv = elem_centroids(mesh_inv)

    if mode == "perturbation":
        sigma_inv_stack = generate_perturbation_phantoms(mesh_inv, n_samples, rng)
        v_list = []
        for i in range(n_samples):
            sigma_fwd = sigma_inv_stack[i].copy()
            V = physics_fwd.solve(sigma_fwd)
            V = V + noise_scale * np.mean(np.abs(V)) * rng.standard_normal(V.shape)
            v_list.append(V)
            if out_dir:
                os.makedirs(out_dir, exist_ok=True)
                np.savez_compressed(
                    os.path.join(out_dir, f"sample_{i:05d}.npz"),
                    sigma=sigma_inv_stack[i],
                    V=V,
                )
        return sigma_inv_stack, np.stack(v_list)

    dom_center, dom_radius = domain_circle(mesh_inv)
    sigma_list, v_list = [], []
    for i in range(n_samples):
        n_inc = rng.integers(n_inclusions_range[0], n_inclusions_range[1] + 1)
        bkg = rng.uniform(*bkg_range)
        ellipses = generate_ellipses(n_inc, dom_center, dom_radius, rng, **ellipse_kwargs)

        sigma_fwd = rasterize(ellipses, bkg, cen_fwd)
        sigma_inv = rasterize(ellipses, bkg, cen_inv)

        V = physics_fwd.solve(sigma_fwd)
        V = V + noise_scale * np.mean(np.abs(V)) * rng.standard_normal(V.shape)

        sigma_list.append(sigma_inv)
        v_list.append(V)

        if out_dir is not None:
            os.makedirs(out_dir, exist_ok=True)
            np.savez_compressed(
                os.path.join(out_dir, f"sample_{i:05d}.npz"),
                bkg=bkg,
                sigma=sigma_inv,
                V=V,
            )
        if (i + 1) % 25 == 0 or i == n_samples - 1:
            print(f"Sample {i + 1}/{n_samples} done.")

    return np.stack(sigma_list), np.stack(v_list)


def load_dataset(npz_dir):
    if os.path.isfile(npz_dir):
        data = np.load(npz_dir)
        return np.asarray(data["sigma"]), np.asarray(data["V"])
    files = sorted(f for f in os.listdir(npz_dir) if f.endswith(".npz"))
    sigma_list, v_list = [], []
    for f in files:
        d = np.load(os.path.join(npz_dir, f))
        sigma_list.append(d["sigma"])
        v_list.append(d["V"])
    return np.stack(sigma_list), np.stack(v_list)
