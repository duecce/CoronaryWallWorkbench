import numpy as np
import pytest

from app.geometry import (
    cumulative_arc_length,
    resample_centerline,
    rotation_minimizing_frames,
)
from app.models import CoronaryPath, WallSurface
from app.walls import edit_control_node, remove_control_anchor, set_active_wall


def make_surface() -> WallSurface:
    s = np.linspace(0.0, 6.0, 25)
    theta = np.linspace(0.0, 2.0 * np.pi, 96, endpoint=False)
    inner = np.full((len(s), len(theta)), 1.5)
    outer = np.full((len(s), len(theta)), 2.2)
    inner_coeff = np.zeros((len(s), 13), dtype=float)
    inner_coeff[:, 0] = 1.5
    outer_coeff = np.zeros((len(s), 17), dtype=float)
    outer_coeff[:, 0] = 2.2
    return WallSurface(
        s_mm=s,
        theta_rad=theta,
        inner_radii_mm=inner.copy(),
        outer_radii_mm=outer.copy(),
        inner_reference_radii_mm=inner.copy(),
        outer_reference_radii_mm=outer.copy(),
        inner_anchors=np.zeros_like(inner, dtype=bool),
        outer_anchors=np.zeros_like(outer, dtype=bool),
        inner_anchor_values_mm=np.full_like(inner, np.nan),
        outer_anchor_values_mm=np.full_like(outer, np.nan),
        inner_fourier_coefficients=inner_coeff,
        outer_fourier_coefficients=outer_coeff,
    )


def test_arc_length():
    points = np.array([[0, 0, 0], [1, 0, 0], [1, 2, 0]], dtype=float)
    assert np.allclose(cumulative_arc_length(points), [0, 1, 3])


def test_centerline_resampling_is_strictly_uniform():
    points = np.array([[0, 0, 0], [0, 0, 1.1], [0, 0, 2.7]], dtype=float)
    path = CoronaryPath(
        path_id="test",
        coronary_name="RCA",
        node_ids=[0, 1, 2],
        centerline_xyz_mm=points,
        arc_length_mm=cumulative_arc_length(points),
        anatomical_labels=[None, None, None],
    )
    _, s = resample_centerline(path, 0.5)
    assert np.allclose(np.diff(s), np.diff(s)[0])
    assert np.diff(s)[0] <= 0.5 + 1e-12
    assert np.isclose(s[-1], 2.7)


def test_rmf_is_orthonormal_and_does_not_twist_on_planar_circle():
    angle = np.linspace(0.0, 1.5 * np.pi, 240)
    points = np.c_[5.0 * np.cos(angle), 5.0 * np.sin(angle), np.zeros_like(angle)]
    s = cumulative_arc_length(points)
    frames = rotation_minimizing_frames(points, s)

    assert np.allclose(np.sum(frames.tangent * frames.normal_u, axis=1), 0, atol=1e-6)
    assert np.allclose(np.sum(frames.tangent * frames.normal_v, axis=1), 0, atol=1e-6)
    assert np.allclose(np.linalg.norm(frames.normal_u, axis=1), 1, atol=1e-6)
    assert np.allclose(np.linalg.norm(frames.normal_v, axis=1), 1, atol=1e-6)
    # For a curve lying in the XY plane, the deterministic initial +Z normal
    # is a true rotation-minimizing vector and must remain essentially +Z.
    assert np.min(frames.normal_u[:, 2]) > 0.999
    assert np.max(np.abs(frames.normal_u[:, :2])) < 1e-5


def test_only_active_wall_can_be_edited():
    surface = make_surface()
    with pytest.raises(ValueError):
        edit_control_node(
            surface,
            wall="inner",
            s_index=12,
            theta_index=16,
            delta_radius_mm=0.2,
        )


def test_outer_anchor_refit_is_smooth_and_does_not_modify_inner():
    surface = make_surface()
    before_inner = surface.inner_radii_mm.copy()
    before_outer = surface.outer_radii_mm.copy()
    edit_control_node(
        surface,
        wall="outer",
        s_index=12,
        theta_index=16,
        delta_radius_mm=0.35,
    )
    assert np.array_equal(surface.inner_radii_mm, before_inner)
    assert surface.outer_anchors[12, 16]
    assert np.isfinite(surface.outer_anchor_values_mm[12, 16])
    assert surface.outer_radii_mm[12, 16] > before_outer[12, 16]
    # Fourier/B-spline refit must propagate progressively to nearby samples.
    local_change = abs(surface.outer_radii_mm[11, 16] - before_outer[11, 16])
    distant_change = abs(surface.outer_radii_mm[0, 16] - before_outer[0, 16])
    assert local_change > distant_change


def test_anchor_can_be_removed_and_surface_returns_toward_reference():
    surface = make_surface()
    edit_control_node(
        surface,
        wall="outer",
        s_index=12,
        theta_index=16,
        delta_radius_mm=0.35,
    )
    edited = float(surface.outer_radii_mm[12, 16])
    remove_control_anchor(surface, wall="outer", s_index=12, theta_index=16)
    assert not surface.outer_anchors[12, 16]
    assert np.isnan(surface.outer_anchor_values_mm[12, 16])
    restored = float(surface.outer_radii_mm[12, 16])
    assert abs(restored - 2.2) < abs(edited - 2.2)


def test_inner_outer_never_cross_and_inner_edit_preserves_outer_anchors():
    surface = make_surface()
    # First create an explicit outer anchor.
    edit_control_node(
        surface,
        wall="outer",
        s_index=12,
        theta_index=16,
        delta_radius_mm=0.2,
    )
    outer_anchor_value = surface.outer_anchor_values_mm[12, 16]
    set_active_wall(surface, "inner")
    edit_control_node(
        surface,
        wall="inner",
        s_index=12,
        theta_index=16,
        delta_radius_mm=5.0,
    )
    assert np.all(
        surface.outer_radii_mm
        >= surface.inner_radii_mm + surface.min_separation_mm - 1e-9
    )
    assert surface.outer_anchors[12, 16]
    assert surface.outer_anchor_values_mm[12, 16] == outer_anchor_value
