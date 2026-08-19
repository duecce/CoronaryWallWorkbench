from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.interpolate import BSpline


@dataclass(slots=True)
class SurfaceFitResult:
    radii_mm: np.ndarray
    raw_fourier_coefficients: np.ndarray
    smooth_fourier_coefficients: np.ndarray
    bspline_control_coefficients: np.ndarray
    bspline_knots_mm: np.ndarray


def fourier_design(theta: np.ndarray, order: int) -> np.ndarray:
    theta = np.asarray(theta, dtype=float)
    columns = [np.ones_like(theta)]
    for harmonic in range(1, order + 1):
        columns.extend((np.cos(harmonic * theta), np.sin(harmonic * theta)))
    return np.column_stack(columns)


def spectral_diagonal(order: int) -> np.ndarray:
    values = [0.0]
    for harmonic in range(1, order + 1):
        values.extend((float(harmonic**4), float(harmonic**4)))
    return np.asarray(values, dtype=float)


def solve_weighted_fourier(
    radii_mm: np.ndarray,
    weights: np.ndarray,
    design: np.ndarray,
    spectral_lambda: float,
    order: int,
) -> np.ndarray:
    radii = np.asarray(radii_mm, dtype=float)
    w = np.clip(np.asarray(weights, dtype=float), 1e-6, None)
    lhs = design.T @ (w[:, None] * design)
    lhs += float(spectral_lambda) * np.diag(spectral_diagonal(order))
    lhs += np.eye(lhs.shape[0]) * 1e-8
    rhs = design.T @ (w * radii)
    return np.linalg.solve(lhs, rhs)


def clamped_bspline_basis(
    s_mm: np.ndarray,
    knot_spacing_mm: float,
    degree: int = 3,
) -> tuple[np.ndarray, np.ndarray]:
    s = np.asarray(s_mm, dtype=float)
    if s.ndim != 1 or len(s) < 2:
        raise ValueError("At least two longitudinal samples are required")
    start, stop = float(s[0]), float(s[-1])
    if stop <= start:
        return np.ones((len(s), 1), dtype=float), np.asarray([start, stop])

    internal = np.arange(start + knot_spacing_mm, stop, knot_spacing_mm)
    knots = np.r_[
        np.repeat(start, degree + 1),
        internal,
        np.repeat(stop, degree + 1),
    ]
    n_basis = len(knots) - degree - 1
    basis = np.empty((len(s), n_basis), dtype=float)
    for index in range(n_basis):
        control = np.zeros(n_basis, dtype=float)
        control[index] = 1.0
        basis[:, index] = BSpline(
            knots, control, degree, extrapolate=True
        )(s)
    return basis, knots


def smooth_fourier_coefficients(
    raw_coefficients: np.ndarray,
    slice_weights: np.ndarray,
    s_mm: np.ndarray,
    knot_spacing_mm: float,
    longitudinal_lambda: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Represent every Fourier coefficient by a cubic longitudinal B-spline."""

    raw = np.asarray(raw_coefficients, dtype=float)
    weights = np.clip(np.asarray(slice_weights, dtype=float), 1e-4, None)
    basis, knots = clamped_bspline_basis(s_mm, knot_spacing_mm, degree=3)
    n_control = basis.shape[1]
    difference = (
        np.diff(np.eye(n_control), n=2, axis=0)
        if n_control >= 3
        else np.zeros((0, n_control), dtype=float)
    )
    controls = np.empty((raw.shape[1], n_control), dtype=float)
    smooth = np.empty_like(raw)
    for column in range(raw.shape[1]):
        harmonic = 0 if column == 0 else (column + 1) // 2
        penalty_scale = float(longitudinal_lambda) * (
            1.0 + 0.08 * harmonic**2
        )
        lhs = basis.T @ (weights[:, None] * basis)
        if difference.size:
            lhs += penalty_scale * (difference.T @ difference)
        lhs += np.eye(n_control) * 1e-8
        rhs = basis.T @ (weights * raw[:, column])
        controls[column] = np.linalg.solve(lhs, rhs)
        smooth[:, column] = basis @ controls[column]
    return smooth, controls, knots


def _apply_manual_anchors(
    targets: np.ndarray,
    weights: np.ndarray,
    theta_rad: np.ndarray,
    anchor_mask: np.ndarray,
    anchor_values_mm: np.ndarray,
    *,
    anchor_weight: float,
) -> None:
    """Apply manual nodes using the desktop tool's angular anchor kernel."""

    angular_step = 2.0 * np.pi / len(theta_rad)
    for s_index, theta_index in np.argwhere(anchor_mask):
        radius = float(anchor_values_mm[s_index, theta_index])
        if not np.isfinite(radius):
            continue
        delta = np.angle(
            np.exp(1j * (theta_rad - float(theta_rad[theta_index])))
        )
        for index in np.flatnonzero(np.abs(delta) <= 2.5 * angular_step):
            angular_weight = np.exp(-0.5 * (delta[index] / angular_step) ** 2)
            targets[s_index, index] = radius
            weights[s_index, index] = max(
                weights[s_index, index],
                float(anchor_weight) * angular_weight,
            )


def fit_fourier_bspline_surface(
    reference_radii_mm: np.ndarray,
    theta_rad: np.ndarray,
    s_mm: np.ndarray,
    anchor_mask: np.ndarray,
    anchor_values_mm: np.ndarray,
    *,
    fourier_order: int,
    spectral_lambda: float,
    longitudinal_knot_spacing_mm: float,
    longitudinal_lambda: float,
    baseline_weight: float = 0.05,
    anchor_weight: float = 45.0,
    lower_bound_mm: np.ndarray | float | None = None,
    upper_bound_mm: np.ndarray | float | None = None,
    containment_iterations: int = 4,
) -> SurfaceFitResult:
    """Fit the shared Fourier circumferential / B-spline longitudinal surface."""

    reference = np.asarray(reference_radii_mm, dtype=float)
    theta = np.asarray(theta_rad, dtype=float)
    s = np.asarray(s_mm, dtype=float)
    anchors = np.asarray(anchor_mask, dtype=bool)
    anchor_values = np.asarray(anchor_values_mm, dtype=float)
    if reference.ndim != 2 or reference.shape != anchors.shape:
        raise ValueError("Invalid wall-surface shape")
    if anchor_values.shape != reference.shape:
        raise ValueError("Anchor values must match wall-surface shape")
    if reference.shape != (len(s), len(theta)):
        raise ValueError("Surface dimensions do not match s/theta coordinates")

    max_order = max(1, (len(theta) - 1) // 2)
    order = int(min(max(1, fourier_order), max_order))
    design = fourier_design(theta, order)

    candidates = reference.copy()
    base_weights = np.full_like(reference, float(baseline_weight), dtype=float)
    _apply_manual_anchors(
        candidates,
        base_weights,
        theta,
        anchors,
        anchor_values,
        anchor_weight=float(anchor_weight),
    )

    lower = (
        np.full_like(reference, -np.inf)
        if lower_bound_mm is None
        else np.broadcast_to(np.asarray(lower_bound_mm, dtype=float), reference.shape)
    )
    upper = (
        np.full_like(reference, np.inf)
        if upper_bound_mm is None
        else np.broadcast_to(np.asarray(upper_bound_mm, dtype=float), reference.shape)
    )

    targets = candidates.copy()
    fitting_weights = base_weights.copy()
    raw = np.empty((len(s), design.shape[1]), dtype=float)
    smooth = raw.copy()
    controls = np.empty((design.shape[1], 1), dtype=float)
    knots = np.asarray([float(s[0]), float(s[-1])])

    for iteration in range(max(1, int(containment_iterations))):
        for s_index in range(len(s)):
            raw[s_index] = solve_weighted_fourier(
                targets[s_index],
                fitting_weights[s_index],
                design,
                spectral_lambda,
                order,
            )
        slice_weights = np.clip(np.mean(fitting_weights, axis=1), 0.04, None)
        smooth, controls, knots = smooth_fourier_coefficients(
            raw,
            slice_weights,
            s,
            longitudinal_knot_spacing_mm,
            longitudinal_lambda,
        )
        radii = smooth @ design.T
        maximum_violation = float(
            max(
                np.max(np.clip(lower - radii, 0.0, None)),
                np.max(np.clip(radii - upper, 0.0, None)),
            )
        )
        if maximum_violation < 0.005:
            break
        if iteration + 1 < containment_iterations:
            low = radii < lower
            high = radii > upper
            violation = low | high
            targets = np.where(low, lower, np.where(high, upper, candidates))
            fitting_weights = base_weights + violation * (
                20.0 * (2.5**iteration)
            )

    radii = np.clip(radii, lower, upper)
    return SurfaceFitResult(
        radii_mm=np.asarray(radii, dtype=float),
        raw_fourier_coefficients=np.asarray(raw, dtype=float),
        smooth_fourier_coefficients=np.asarray(smooth, dtype=float),
        bspline_control_coefficients=np.asarray(controls, dtype=float),
        bspline_knots_mm=np.asarray(knots, dtype=float),
    )


def evaluate_fourier_contour(
    coefficients: np.ndarray,
    theta_dense: np.ndarray,
) -> np.ndarray:
    order = (len(coefficients) - 1) // 2
    return fourier_design(np.asarray(theta_dense, dtype=float), order) @ np.asarray(
        coefficients, dtype=float
    )
