"""Matched coarse/fine anatomical phantoms for vascular differential EIT."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

from gcnm_pvi.gcnm_phantoms import elem_centroids


@dataclass
class Ellipse:
    center_x: float
    center_y: float
    axis_a: float
    axis_b: float
    angle: float


@dataclass
class AnatomyParameters:
    rotation: float
    skin_thickness: float
    fat_thickness: float
    sigma_skin: float
    sigma_fat: float
    sigma_muscle: float
    sigma_bone: float
    sigma_blood: float
    bone: Ellipse
    vessels: list[Ellipse]
    vessel_delta: list[float]

    def to_dict(self) -> dict:
        return asdict(self)


def domain_transform(mesh) -> tuple[np.ndarray, float]:
    """Return center and robust radius used for shared normalized coordinates."""
    boundary = np.asarray(mesh.nodes)[np.asarray(mesh.nodes_idx_boundary, dtype=int), :2]
    center = np.mean(boundary, axis=0)
    radius = float(np.median(np.linalg.norm(boundary - center, axis=1)))
    return center, radius


def normalized_centroids(mesh, center: np.ndarray, radius: float) -> np.ndarray:
    return (elem_centroids(mesh)[:, :2] - center[None, :]) / radius


def _rotate(points: np.ndarray, angle: float) -> np.ndarray:
    cosine, sine = np.cos(angle), np.sin(angle)
    rotation = np.array([[cosine, sine], [-sine, cosine]])
    return points @ rotation.T


def ellipse_mask(points: np.ndarray, ellipse: Ellipse) -> np.ndarray:
    local = _rotate(
        points - np.array([ellipse.center_x, ellipse.center_y]),
        ellipse.angle,
    )
    return (local[:, 0] / ellipse.axis_a) ** 2 + (
        local[:, 1] / ellipse.axis_b
    ) ** 2 <= 1.0


def _sample_ellipse_center(
    rng: np.random.Generator,
    *,
    radial_limit: float,
) -> tuple[float, float]:
    radial = radial_limit * np.sqrt(rng.random())
    angle = rng.uniform(-np.pi, np.pi)
    return radial * np.cos(angle), radial * np.sin(angle)


def sample_anatomy(rng: np.random.Generator) -> AnatomyParameters:
    """Draw one arm cross-section with a bone and one or two vessels."""
    rotation = float(rng.uniform(-np.pi, np.pi))
    skin_thickness = float(rng.uniform(0.035, 0.075))
    fat_thickness = float(rng.uniform(0.08, 0.20))

    bone_x, bone_y = _sample_ellipse_center(rng, radial_limit=0.25)
    bone = Ellipse(
        center_x=float(bone_x),
        center_y=float(bone_y),
        axis_a=float(rng.uniform(0.10, 0.18)),
        axis_b=float(rng.uniform(0.07, 0.14)),
        angle=float(rng.uniform(-np.pi, np.pi)),
    )

    vessels: list[Ellipse] = []
    vessel_delta: list[float] = []
    count = int(rng.integers(1, 3))
    for _ in range(count):
        for _attempt in range(200):
            x, y = _sample_ellipse_center(rng, radial_limit=0.55)
            distance_to_bone = np.hypot(x - bone_x, y - bone_y)
            if distance_to_bone > max(bone.axis_a, bone.axis_b) + 0.10:
                break
        vessels.append(
            Ellipse(
                center_x=float(x),
                center_y=float(y),
                axis_a=float(rng.uniform(0.09, 0.18)),
                axis_b=float(rng.uniform(0.07, 0.14)),
                angle=float(rng.uniform(-np.pi, np.pi)),
            )
        )
        vessel_delta.append(float(rng.uniform(0.025, 0.11)))

    # A random cardiac phase prevents the network from assuming a fixed peak.
    phase_scale = float(rng.uniform(0.25, 1.0))
    vessel_delta = [value * phase_scale for value in vessel_delta]
    return AnatomyParameters(
        rotation=rotation,
        skin_thickness=skin_thickness,
        fat_thickness=fat_thickness,
        sigma_skin=float(rng.uniform(0.16, 0.30)),
        sigma_fat=float(rng.uniform(0.055, 0.13)),
        sigma_muscle=float(rng.uniform(0.28, 0.48)),
        sigma_bone=float(rng.uniform(0.025, 0.075)),
        sigma_blood=float(rng.uniform(0.55, 0.85)),
        bone=bone,
        vessels=vessels,
        vessel_delta=vessel_delta,
    )


def rasterize_anatomy(
    normalized_points: np.ndarray,
    parameters: AnatomyParameters,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Rasterize baseline, dynamic, and clean delta conductivity fields."""
    points = _rotate(np.asarray(normalized_points), parameters.rotation)
    radius = np.linalg.norm(points, axis=1)
    sigma_baseline = np.full(len(points), parameters.sigma_muscle, dtype=np.float64)

    fat_start = 1.0 - parameters.skin_thickness - parameters.fat_thickness
    skin_start = 1.0 - parameters.skin_thickness
    sigma_baseline[radius >= fat_start] = parameters.sigma_fat
    sigma_baseline[radius >= skin_start] = parameters.sigma_skin
    sigma_baseline[ellipse_mask(points, parameters.bone)] = parameters.sigma_bone

    delta = np.zeros(len(points), dtype=np.float64)
    for vessel, amplitude in zip(parameters.vessels, parameters.vessel_delta):
        mask = ellipse_mask(points, vessel)
        sigma_baseline[mask] = parameters.sigma_blood
        delta[mask] += amplitude

    sigma_dynamic = sigma_baseline + delta
    return sigma_baseline, sigma_dynamic, delta


def element_positions(mesh) -> np.ndarray:
    """Return x/y/r positional features normalized on the inverse domain."""
    center, radius = domain_transform(mesh)
    xy = normalized_centroids(mesh, center, radius)
    radial = np.linalg.norm(xy, axis=1, keepdims=True)
    return np.concatenate([xy, radial], axis=1)
