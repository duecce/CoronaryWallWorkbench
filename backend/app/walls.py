from __future__ import annotations

from typing import Literal

import numpy as np
import SimpleITK as sitk
from scipy.interpolate import CubicSpline

from .geometry import generate_longitudinal_reformation
from .models import ParallelTransportFrames, WallSurface

WallName = Literal["inner", "outer"]


def inner_radii_from_lumen_mask(
    lumen_mask: sitk.Image,
    frames: ParallelTransportFrames,
    *,
    n_theta: int = 16,
    max_radius_mm: float = 5.0,
    radial_step_mm: float = 0.05,
) -> tuple[np.ndarray, np.ndarray]:
    theta = np.linspace(0.0, 2.0*np.pi, n_theta, endpoint=False)
    radii = np.zeros((len(frames.arc_length_mm), n_theta), dtype=float)
    for j, angle in enumerate(theta):
        strip, radial = generate_longitudinal_reformation(
            lumen_mask,
            frames,
            float(angle),
            radial_extent_mm=max_radius_mm,
            radial_spacing_mm=radial_step_mm,
            nearest=True,
        )
        positive = radial >= -1e-9
        rpos = radial[positive]
        values = strip[positive, :].T > 0.5
        for i, row in enumerate(values):
            if not row[0]:
                hits = np.flatnonzero(row)
                if len(hits) == 0:
                    radii[i, j] = radial_step_mm
                    continue
                start = int(hits[0])
            else:
                start = 0
            exits = np.flatnonzero(~row[start:])
            if len(exits) == 0:
                radii[i, j] = max_radius_mm
            else:
                k = start + int(exits[0])
                radii[i, j] = max(radial_step_mm, float(rpos[max(k-1, 0)]))
    return theta, radii


def initialize_wall_surface(
    lumen_mask: sitk.Image,
    frames: ParallelTransportFrames,
    *,
    n_theta: int = 16,
    outer_offset_mm: float = 0.75,
    min_separation_mm: float = 0.10,
) -> WallSurface:
    theta, inner = inner_radii_from_lumen_mask(lumen_mask, frames, n_theta=n_theta)
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
    d = np.abs(indices-center)
    return np.minimum(d, n-d)


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
        raise ValueError(f"{wall} wall is not active; current wall is {surface.active_wall}")
    target = surface.inner_radii_mm if wall == "inner" else surface.outer_radii_mm
    anchors = surface.inner_anchors if wall == "inner" else surface.outer_anchors
    if not 0 <= s_index < target.shape[0] or not 0 <= theta_index < target.shape[1]:
        raise IndexError("control-node index out of range")
    ds = np.abs(surface.s_mm-surface.s_mm[s_index])
    ks = np.exp(-0.5*(ds/sigma_s_mm)**2)
    idx = np.arange(target.shape[1])
    kt = np.exp(-0.5*(_periodic_node_distance(idx, theta_index, target.shape[1])/sigma_theta_nodes)**2)
    influence = ks[:,None]*kt[None,:]
    frozen = anchors.copy(); frozen[s_index, theta_index] = False; influence[frozen] = 0.0
    target += float(delta_radius_mm)*influence
    if wall == "inner":
        target[:] = np.maximum(target, 0.05)
        surface.outer_radii_mm[:] = np.maximum(surface.outer_radii_mm, target+surface.min_separation_mm)
    else:
        target[:] = np.maximum(target, surface.inner_radii_mm+surface.min_separation_mm)
    if make_anchor:
        anchors[s_index, theta_index] = True


def _periodic_spline(theta_nodes: np.ndarray, radii: np.ndarray, samples: int=256) -> tuple[np.ndarray,np.ndarray]:
    x = np.r_[theta_nodes, 2*np.pi]; y = np.r_[radii, radii[0]]
    spline = CubicSpline(x, y, bc_type="periodic")
    td = np.linspace(0, 2*np.pi, samples, endpoint=False)
    return td, spline(td)


def contour_plane_xy(surface: WallSurface, wall: WallName, s_index: int, samples: int=256) -> np.ndarray:
    radii = surface.inner_radii_mm if wall == "inner" else surface.outer_radii_mm
    theta, r = _periodic_spline(surface.theta_rad, radii[s_index], samples)
    return np.c_[r*np.cos(theta), r*np.sin(theta)]


def control_points_plane_xy(surface: WallSurface, wall: WallName, s_index: int) -> np.ndarray:
    radii = surface.inner_radii_mm if wall == "inner" else surface.outer_radii_mm
    r = radii[s_index]
    return np.c_[r*np.cos(surface.theta_rad), r*np.sin(surface.theta_rad)]


def longitudinal_wall_profiles(surface: WallSurface, wall: WallName, theta_index: int) -> dict[str, np.ndarray]:
    radii = surface.inner_radii_mm if wall == "inner" else surface.outer_radii_mm
    n = radii.shape[1]
    opposite = (theta_index + n//2) % n
    return {
        "positive_mm": radii[:, theta_index],
        "negative_mm": -radii[:, opposite],
        "positive_anchors": (surface.inner_anchors if wall == "inner" else surface.outer_anchors)[:, theta_index],
        "negative_anchors": (surface.inner_anchors if wall == "inner" else surface.outer_anchors)[:, opposite],
        "opposite_theta_index": np.asarray(opposite),
    }


def surface_points_world(surface: WallSurface, frames: ParallelTransportFrames, wall: WallName) -> np.ndarray:
    radii = surface.inner_radii_mm if wall == "inner" else surface.outer_radii_mm
    directions = np.cos(surface.theta_rad)[None,:,None]*frames.normal_u[:,None,:] + np.sin(surface.theta_rad)[None,:,None]*frames.normal_v[:,None,:]
    return frames.centerline_xyz_mm[:,None,:] + radii[...,None]*directions
