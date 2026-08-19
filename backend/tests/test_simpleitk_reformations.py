import numpy as np
import SimpleITK as sitk

from app.geometry import (
    generate_cross_section,
    generate_longitudinal_reformation,
    rotation_minimizing_frames,
)
from app.walls import inner_radii_from_lumen_mask


def make_synthetic_case():
    size = [96, 96, 96]
    spacing = [0.25, 0.25, 0.25]
    origin = [-12.0, -12.0, -12.0]

    zz, yy, xx = np.indices((size[2], size[1], size[0]))
    x = origin[0] + xx * spacing[0]
    y = origin[1] + yy * spacing[1]
    radius = np.sqrt(x**2 + y**2)
    lumen_np = (radius <= 2.0).astype(np.uint8)
    ccta_np = np.full_like(radius, -1000.0, dtype=np.float32)
    ccta_np[radius <= 2.0] = 450.0

    ccta = sitk.GetImageFromArray(ccta_np)
    lumen = sitk.GetImageFromArray(lumen_np)
    for image in (ccta, lumen):
        image.SetSpacing(spacing)
        image.SetOrigin(origin)
        image.SetDirection(np.eye(3).ravel().tolist())

    centerline = np.c_[np.zeros(41), np.zeros(41), np.linspace(-5.0, 5.0, 41)]
    s = np.linspace(0.0, 10.0, 41)
    frames = rotation_minimizing_frames(centerline, s)
    return ccta, lumen, frames


def test_default_cross_section_is_exactly_15_mm_square_and_centered():
    ccta, lumen, frames = make_synthetic_case()
    section = generate_cross_section(ccta, lumen, frames, 20)
    assert section["image"].shape == (151, 151)
    assert section["lumen_mask"].shape == (151, 151)
    assert section["lumen_mask"][75, 75]
    assert np.isclose(section["size_mm"], 15.0)
    assert np.isclose(section["spacing_mm"], 0.10)


def test_cross_section_custom_fov_remains_square_without_zoom_change():
    ccta, lumen, frames = make_synthetic_case()
    section = generate_cross_section(
        ccta, lumen, frames, 20, size_mm=8.0, spacing_mm=0.10
    )
    assert section["image"].shape == (81, 81)
    assert section["lumen_mask"].shape == (81, 81)
    assert section["lumen_mask"][40, 40]
    assert np.isclose((section["image"].shape[0] - 1) * section["spacing_mm"], 8.0)
    assert np.isclose((section["image"].shape[1] - 1) * section["spacing_mm"], 8.0)


def test_longitudinal_shape_centerline_signal_and_metric_sampling():
    ccta, _, frames = make_synthetic_case()
    image, radial = generate_longitudinal_reformation(
        ccta, frames, 0.0, radial_extent_mm=7.5, radial_spacing_mm=0.10
    )
    assert image.shape == (len(radial), len(frames.arc_length_mm))
    assert np.allclose(np.diff(radial), 0.10)
    assert np.allclose(np.diff(frames.arc_length_mm), np.diff(frames.arc_length_mm)[0])
    center_index = int(np.argmin(np.abs(radial)))
    assert np.median(image[center_index]) > 300.0


def test_inner_wall_matches_known_cylinder_radius():
    _, lumen, frames = make_synthetic_case()
    _, radii = inner_radii_from_lumen_mask(
        lumen, frames, n_theta=32, max_radius_mm=4.0, radial_step_mm=0.05
    )
    np.testing.assert_allclose(np.median(radii), 2.0, atol=0.15)
