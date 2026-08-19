from __future__ import annotations

from typing import Literal

import numpy as np
import SimpleITK as sitk

from .geometry import generate_longitudinal_reformation
from .models import ParallelTransportFrames, WallSurface
from .surface_fit import (
    evaluate_fourier_contour,
    fit_fourier_bspline_surface,
    fourier_design,
)

WallName = Literal["inner", "outer"]

INNER_FOURIER_ORDER = 6
INNER_SPECTRAL_LAMBDA = 0.015
INNER_KNOT_SPACING_MM = 0.80
INNER_LONGITUDINAL_LAMBDA = 5.0

OUTER_FOURIER_ORDER = 8
OUTER_SPECTRAL_LAMBDA = 0.020
OUTER_KNOT_SPACING_MM = 1.00
OUTER_LONGITUDINAL_LAMBDA = 12.0
ANCHOR_WEIGHT = 45.0


def inner_radii_from_lumen_mask(
    lumen_mask: sitk.Image,
    frames: ParallelTransportFrames,
    *,
    n_theta: int = 96,
    max_radius_mm: float = 7.5,
    radial_step_mm: float = 0.05,
) -> tuple[np.ndarray, np.ndarray]:
    """Sample the lumen boundary in the common RMF (s, theta) coordinates."""

    theta = np.linspace(0.0, 2.0 * np.pi, n_theta, endpoint=False)
    radii = np.zeros((len(frames.arc_length_mm), n_theta), dtype=float)
    for theta_index, angle in enumerate(theta):
        strip, radial = generate_longitudinal_reformation(
            lumen_mask,
            frames,
            float(angle),
            radial_extent_mm=max_radius_mm,
            radial_spacing_mm=radial_step_mm,
            nearest=True,
        )
        positive = radial >= -1e-9
        positive_radial = radial[positive]
        # strip shape = [radial, s].
        values = strip[positive, :].T > 0.5
        for s_index, row in enumerate(values):
            if not row[0]:
                hits = np.flatnonzero(row)
                if len(hits) == 0:
                    radii[s_index, theta_index] = radial_step_mm
                    continue
                start = int(hits[0])
            else:
                start = 0
            exits = np.flatnonzero(~row[start:])
            if len(exits) == 0:
                radii[s_index, theta_index] = max_radius_mm
            else:
                index = start + int(exits[0])
                radii[s_index, theta_index] = max(
                    radial_step_mm,
                    float(positive_radial[max(index - 1, 0)]),
                )
    return theta, radii


def _fit_initial_inner(
    raw_inner: np.ndarray,
    theta: np.ndarray,
    s_mm: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    empty = np.zeros_like(raw_inner, dtype=bool)
    result = fit_fourier_bspline_surface(
        raw_inner,
        theta,
        s_mm,
        empty,
        np.full_like(raw_inner, np.nan),
        fourier_order=INNER_FOURIER_ORDER,
        spectral_lambda=INNER_SPECTRAL_LAMBDA,
        longitudinal_knot_spacing_mm=INNER_KNOT_SPACING_MM,
        longitudinal_lambda=INNER_LONGITUDINAL_LAMBDA,
        baseline_weight=1.0,
        anchor_weight=ANCHOR_WEIGHT,
        lower_bound_mm=0.05,
        containment_iterations=3,
    )
    return result.radii_mm, result.smooth_fourier_coefficients


def _fit_surface_in_place(surface: WallSurface, wall: WallName) -> None:
    if wall == "inner":
        result = fit_fourier_bspline_surface(
            surface.inner_reference_radii_mm,
            surface.theta_rad,
            surface.s_mm,
            surface.inner_anchors,
            surface.inner_anchor_values_mm,
            fourier_order=INNER_FOURIER_ORDER,
            spectral_lambda=INNER_SPECTRAL_LAMBDA,
            longitudinal_knot_spacing_mm=INNER_KNOT_SPACING_MM,
            longitudinal_lambda=INNER_LONGITUDINAL_LAMBDA,
            baseline_weight=0.05,
            anchor_weight=surface.anchor_weight,
            lower_bound_mm=0.05,
            upper_bound_mm=np.maximum(
                0.05,
                surface.outer_radii_mm - surface.min_separation_mm,
            ),
            containment_iterations=5,
        )
        surface.inner_radii_mm[:] = result.radii_mm
        surface.inner_fourier_coefficients = result.smooth_fourier_coefficients
    else:
        result = fit_fourier_bspline_surface(
            surface.outer_reference_radii_mm,
            surface.theta_rad,
            surface.s_mm,
            surface.outer_anchors,
            surface.outer_anchor_values_mm,
            fourier_order=surface.fourier_order,
            spectral_lambda=surface.spectral_lambda,
            longitudinal_knot_spacing_mm=surface.longitudinal_knot_spacing_mm,
            longitudinal_lambda=surface.longitudinal_lambda,
            baseline_weight=0.05,
            anchor_weight=surface.anchor_weight,
            lower_bound_mm=surface.inner_radii_mm + surface.min_separation_mm,
            containment_iterations=5,
        )
        surface.outer_radii_mm[:] = result.radii_mm
        surface.outer_fourier_coefficients = result.smooth_fourier_coefficients


def initialize_wall_surface(
    lumen_mask: sitk.Image,
    frames: ParallelTransportFrames,
    *,
    n_theta: int = 96,
    outer_offset_mm: float = 0.75,
    min_separation_mm: float = 0.10,
) -> WallSurface:
    theta, raw_inner = inner_radii_from_lumen_mask(
        lumen_mask,
        frames,
        n_theta=n_theta,
    )
    inner, inner_coefficients = _fit_initial_inner(
        raw_inner,
        theta,
        frames.arc_length_mm,
    )
    outer_reference = inner + float(outer_offset_mm)
    empty = np.zeros_like(inner, dtype=bool)
    outer_fit = fit_fourier_bspline_surface(
        outer_reference,
        theta,
        frames.arc_length_mm,
        empty,
        np.full_like(inner, np.nan),
        fourier_order=OUTER_FOURIER_ORDER,
        spectral_lambda=OUTER_SPECTRAL_LAMBDA,
        longitudinal_knot_spacing_mm=OUTER_KNOT_SPACING_MM,
        longitudinal_lambda=OUTER_LONGITUDINAL_LAMBDA,
        baseline_weight=1.0,
        anchor_weight=ANCHOR_WEIGHT,
        lower_bound_mm=inner + float(min_separation_mm),
        containment_iterations=3,
    )

    return WallSurface(
        s_mm=frames.arc_length_mm.copy(),
        theta_rad=theta,
        inner_radii_mm=inner.copy(),
        outer_radii_mm=outer_fit.radii_mm.copy(),
        inner_reference_radii_mm=inner.copy(),
        outer_reference_radii_mm=outer_reference.copy(),
        inner_anchors=np.zeros_like(inner, dtype=bool),
        outer_anchors=np.zeros_like(inner, dtype=bool),
        inner_anchor_values_mm=np.full_like(inner, np.nan),
        outer_anchor_values_mm=np.full_like(inner, np.nan),
        inner_fourier_coefficients=inner_coefficients.copy(),
        outer_fourier_coefficients=outer_fit.smooth_fourier_coefficients.copy(),
        active_wall="outer",
        min_separation_mm=float(min_separation_mm),
        fourier_order=OUTER_FOURIER_ORDER,
        spectral_lambda=OUTER_SPECTRAL_LAMBDA,
        longitudinal_knot_spacing_mm=OUTER_KNOT_SPACING_MM,
        longitudinal_lambda=OUTER_LONGITUDINAL_LAMBDA,
        anchor_weight=ANCHOR_WEIGHT,
    )


def set_active_wall(surface: WallSurface, wall: WallName) -> None:
    if wall not in {"inner", "outer"}:
        raise ValueError(f"Unknown wall: {wall}")
    surface.active_wall = wall


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
    """Apply a manual anchor and globally refit the Fourier-B-spline surface.

    ``sigma_*`` are kept in the API for backward compatibility.  Propagation is
    now produced by the same weighted angular anchor + longitudinal B-spline
    model used by the desktop reference tool, rather than by direct Gaussian
    deformation of the radius matrix.
    """

    del sigma_s_mm, sigma_theta_nodes
    if wall != surface.active_wall:
        raise ValueError(
            f"{wall} wall is not active; current wall is {surface.active_wall}"
        )
    target = surface.inner_radii_mm if wall == "inner" else surface.outer_radii_mm
    anchor_mask = surface.inner_anchors if wall == "inner" else surface.outer_anchors
    anchor_values = (
        surface.inner_anchor_values_mm
        if wall == "inner"
        else surface.outer_anchor_values_mm
    )
    if not 0 <= s_index < target.shape[0] or not 0 <= theta_index < target.shape[1]:
        raise IndexError("control-node index out of range")

    requested = float(target[s_index, theta_index] + delta_radius_mm)
    if wall == "inner":
        requested = float(
            np.clip(
                requested,
                0.05,
                max(0.05, surface.outer_radii_mm[s_index, theta_index] - surface.min_separation_mm),
            )
        )
    else:
        requested = max(
            requested,
            float(surface.inner_radii_mm[s_index, theta_index] + surface.min_separation_mm),
        )

    if make_anchor:
        anchor_mask[s_index, theta_index] = True
    anchor_values[s_index, theta_index] = requested
    _fit_surface_in_place(surface, wall)

    # If the inner wall changed, refit the outer wall too so its existing
    # anchors are preserved while enforcing containment against the new lumen.
    if wall == "inner":
        _fit_surface_in_place(surface, "outer")


def remove_control_anchor(
    surface: WallSurface,
    *,
    wall: WallName,
    s_index: int,
    theta_index: int,
) -> None:
    if wall != surface.active_wall:
        raise ValueError("Activate the wall before removing one of its anchors")
    mask = surface.inner_anchors if wall == "inner" else surface.outer_anchors
    values = (
        surface.inner_anchor_values_mm
        if wall == "inner"
        else surface.outer_anchor_values_mm
    )
    mask[s_index, theta_index] = False
    values[s_index, theta_index] = np.nan
    _fit_surface_in_place(surface, wall)
    if wall == "inner":
        _fit_surface_in_place(surface, "outer")


def contour_plane_xy(
    surface: WallSurface,
    wall: WallName,
    s_index: int,
    samples: int = 384,
) -> np.ndarray:
    coefficients = (
        surface.inner_fourier_coefficients
        if wall == "inner"
        else surface.outer_fourier_coefficients
    )
    theta_dense = np.linspace(0.0, 2.0 * np.pi, samples, endpoint=False)
    radius = evaluate_fourier_contour(coefficients[s_index], theta_dense)
    return np.c_[radius * np.cos(theta_dense), radius * np.sin(theta_dense)]


def control_points_plane_xy(
    surface: WallSurface,
    wall: WallName,
    s_index: int,
) -> np.ndarray:
    radii = surface.inner_radii_mm if wall == "inner" else surface.outer_radii_mm
    radius = radii[s_index]
    return np.c_[radius * np.cos(surface.theta_rad), radius * np.sin(surface.theta_rad)]


def longitudinal_wall_profiles(
    surface: WallSurface,
    wall: WallName,
    theta_index: int,
) -> dict[str, np.ndarray]:
    radii = surface.inner_radii_mm if wall == "inner" else surface.outer_radii_mm
    anchors = surface.inner_anchors if wall == "inner" else surface.outer_anchors
    n_theta = radii.shape[1]
    opposite = (theta_index + n_theta // 2) % n_theta
    return {
        "positive_mm": radii[:, theta_index],
        "negative_mm": -radii[:, opposite],
        "positive_anchors": anchors[:, theta_index],
        "negative_anchors": anchors[:, opposite],
        "opposite_theta_index": np.asarray(opposite),
    }


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
