import numpy as np
import pytest

from app.geometry import cumulative_arc_length, parallel_transport_frames
from app.models import WallSurface
from app.walls import edit_control_node, set_active_wall


def make_surface() -> WallSurface:
    s = np.arange(0.0, 5.0, 0.5)
    theta = np.linspace(0.0, 2.0 * np.pi, 16, endpoint=False)
    inner = np.full((len(s), len(theta)), 1.5)
    outer = np.full((len(s), len(theta)), 2.2)
    return WallSurface(
        s_mm=s,
        theta_rad=theta,
        inner_radii_mm=inner,
        outer_radii_mm=outer,
        inner_anchors=np.zeros_like(inner, dtype=bool),
        outer_anchors=np.zeros_like(outer, dtype=bool),
    )


def test_arc_length():
    p = np.array([[0, 0, 0], [1, 0, 0], [1, 2, 0]], dtype=float)
    assert np.allclose(cumulative_arc_length(p), [0, 1, 3])


def test_parallel_transport_is_orthonormal():
    s = np.linspace(0, 8, 20)
    p = np.c_[s, np.sin(s / 3), np.cos(s / 4)]
    frames = parallel_transport_frames(p, cumulative_arc_length(p))
    assert np.allclose(np.sum(frames.tangent * frames.normal_u, axis=1), 0, atol=1e-6)
    assert np.allclose(np.sum(frames.tangent * frames.normal_v, axis=1), 0, atol=1e-6)
    assert np.allclose(np.linalg.norm(frames.normal_u, axis=1), 1, atol=1e-6)


def test_only_active_wall_can_be_edited():
    surface = make_surface()
    with pytest.raises(ValueError):
        edit_control_node(surface, wall="inner", s_index=3, theta_index=4, delta_radius_mm=0.2)


def test_outer_edit_does_not_modify_inner():
    surface = make_surface()
    before = surface.inner_radii_mm.copy()
    edit_control_node(surface, wall="outer", s_index=3, theta_index=4, delta_radius_mm=0.4)
    assert np.array_equal(surface.inner_radii_mm, before)


def test_inner_outer_never_cross():
    surface = make_surface()
    set_active_wall(surface, "inner")
    edit_control_node(surface, wall="inner", s_index=3, theta_index=4, delta_radius_mm=5.0)
    assert np.all(surface.outer_radii_mm >= surface.inner_radii_mm + surface.min_separation_mm - 1e-9)
