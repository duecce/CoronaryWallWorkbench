from __future__ import annotations

from typing import Literal

import numpy as np
from scipy.interpolate import CubicSpline
from scipy.ndimage import map_coordinates

from .models import ParallelTransportFrames, WallSurface

WallName = Literal["inner", "outer"]


def _sample_mask(mask: np.ndarray, affine: np.ndarray, world_xyz: np.ndarray) -> np.ndarray:
    inv = np.linalg.inv(affine)
    flat = world_xyz.reshape(-1, 3)
    hom = np.c_[flat, np.ones(len(flat))]
    ijk = (inv @ hom.T).T[:, :3]
    values = map_coordinates(
        mask.astype(np.uint8),
        [ijk[:, 0], ijk[:, 1], ijk[:, 2]],
        order=0,
        mode="constant",
        cval=0,
    )
    return values.reshape(world_xyz.shape[:-1]) > 0


def inner_radii_from_lumen_mask(
    lumen_mask: np.ndarray,
    affine: np.ndarray,
    frames: ParallelTransportFrames,
    *,
    n_theta: int = 16,
    max_radius_mm: float = 5.0,
    radial_step_mm: float = 0.05,
) -> tuple[np.ndarray, np.ndarray]:
    theta = np.linspace(0.0, 2.0 * np.pi, n_theta, endpoint=False)
    radial = np.arange(0.0, max_radius_mm + radial_step_mm * 0.5, radial_step_mm)
    radii = np.zeros((len(frames.arc_length_mm), n_theta), dtype=float)

    for j, angle in enumerate(theta):
        direction = np.cos(angle) * frames.normal_u + np.sin(angle) * frames.normal_v
        world = (
            frames.centerline_xyz_mm[:, None, :]
            + radial[None, :, None] * direction[:, None, :]
        )
        values = _sample_mask(lumen_mask, affine, world)
        for i in range(len(frames.arc_length_mm)):
            row = values[i]
            if not row[0]:
                # Centerline is slightly outside the discrete mask. Find the first lumen hit.
                hits = np.flatnonzero(row)
                if len(hits) == 0:
                    radii[i, j] = radial_step_mm
                    continue
                start = hits[0]
            else:
                start = 0
            after = np.flatnonzero(~row[start:])
            if len(after) == 0:
                radii[i, j] = max_radius_mm
            else:
                idx = start + after[0]
                radii[i, j] = max(radial_step_mm, radial[max(idx - 1, 0)])
    return theta, radii


def initialize_wall_surface(
    lumen_mask: np.ndarray,
    affine: np.ndarray,
    frames: ParallelTransportFrames,
    *,
    n_theta: int = 16,
    outer_offset_mm: float = 0.75,
    min_separation_mm: float = 0.10,
) -> WallSurface:
    theta, inner = inner_radii_from_lumen_mask(
        lumen_mask,
        affine,
        frames,
        n_theta=n_theta,
    )
    outer = inner + float(outer_offset_mm)
    return WallSurface(
        s_mm=frames.arc_length_mm.copy(),
        theta_rad=theta,
        inner_radii_mm=inner,
        outer_radii_mm=outer,
        inner_anchors=np.zeros_like(inner, dtype=bool),
        outer_anchors=np.zeros_like(outer, dtype=bool),
        active_wall="outer",
        min_separation_mm=float(min_separation_mm),
    )


def set_active_wall(surface: WallSurface, wall: WallName) -> None:
    if wall not in {"inner", "outer"}:
        raise ValueError(f"Unknown wall: {wall}")
    surface.active_wall = wall


def _periodic_node_distance(indices: np.ndarray, center: int, n: int) -> np.ndarray:
    d = np.abs(indices - center)
    return np.minimum(d, n - d)


def edit_control_node(
    surface: WallSurface,
    *,
    wall: WallName,
    s_index: int,
    theta_index: int,
    delta_radius_mm: float,
    sigma_s_mm: float = 1.5,
    sigma_theta_nodes: float = 1.25,
    make_anchor: bool = True,
) -> None:
    if wall != surface.active_wall:
        raise ValueError(
            f"{wall} wall is not active. Activate it before editing; current wall is {surface.active_wall}."
        )
    target = surface.inner_radii_mm if wall == "inner" else surface.outer_radii_mm
    anchors = surface.inner_anchors if wall == "inner" else surface.outer_anchors
    if not 0 <= s_index < target.shape[0]:
        raise IndexError("s_index out of range")
    if not 0 <= theta_index < target.shape[1]:
        raise IndexError("theta_index out of range")

    ds = np.abs(surface.s_mm - surface.s_mm[s_index])
    ks = np.exp(-0.5 * (ds / sigma_s_mm) ** 2)
    theta_nodes = np.arange(target.shape[1])
    dtheta = _periodic_node_distance(theta_nodes, theta_index, target.shape[1])
    kt = np.exp(-0.5 * (dtheta / sigma_theta_nodes) ** 2)
    influence = ks[:, None] * kt[None, :]

    # Existing explicit anchors are fixed, except the node currently being edited.
    frozen = anchors.copy()
    frozen[s_index, theta_index] = False
    influence[frozen] = 0.0
    target += float(delta_radius_mm) * influence

    if wall == "inner":
        target[:] = np.maximum(target, 0.05)
        surface.outer_radii_mm[:] = np.maximum(
            surface.outer_radii_mm,
            target + surface.min_separation_mm,
        )
    else:
        target[:] = np.maximum(
            target,
            surface.inner_radii_mm + surface.min_separation_mm,
        )

    if make_anchor:
        anchors[s_index, theta_index] = True


def _periodic_spline(theta_nodes: np.ndarray, radii: np.ndarray, samples: int = 256) -> tuple[np.ndarray, np.ndarray]:
    x = np.r_[theta_nodes, 2.0 * np.pi]
    y = np.r_[radii, radii[0]]
    spline = CubicSpline(x, y, bc_type="periodic")
    theta_dense = np.linspace(0.0, 2.0 * np.pi, samples, endpoint=False)
    return theta_dense, spline(theta_dense)


def contour_plane_xy(surface: WallSurface, wall: WallName, s_index: int, samples: int = 256) -> np.ndarray:
    radii = surface.inner_radii_mm if wall == "inner" else surface.outer_radii_mm
    theta, r = _periodic_spline(surface.theta_rad, radii[s_index], samples=samples)
    return np.c_[r * np.cos(theta), r * np.sin(theta)]


def control_points_plane_xy(surface: WallSurface, wall: WallName, s_index: int) -> np.ndarray:
    radii = surface.inner_radii_mm if wall == "inner" else surface.outer_radii_mm
    r = radii[s_index]
    return np.c_[r * np.cos(surface.theta_rad), r * np.sin(surface.theta_rad)]


def surface_points_world(
    surface: WallSurface,
    frames: ParallelTransportFrames,
    wall: WallName,
) -> np.ndarray:
    radii = surface.inner_radii_mm if wall == "inner" else surface.outer_radii_mm
    directions = (
        np.cos(surface.theta_rad)[None, :, None] * frames.normal_u[:, None, :]
        + np.sin(surface.theta_rad)[None, :, None] * frames.normal_v[:, None, :]
    )
    return frames.centerline_xyz_mm[:, None, :] + radii[..., None] * directions
