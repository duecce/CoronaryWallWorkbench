import numpy as np
import SimpleITK as sitk

from app.geometry import generate_cross_section, generate_longitudinal_reformation, parallel_transport_frames
from app.walls import inner_radii_from_lumen_mask


def make_synthetic_case():
    size = [96, 96, 96]
    spacing = [0.25, 0.25, 0.25]
    origin = [-12.0, -12.0, -12.0]

    zz, yy, xx = np.indices((size[2], size[1], size[0]))
    x = origin[0] + xx * spacing[0]
    y = origin[1] + yy * spacing[1]
    z = origin[2] + zz * spacing[2]

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
    frames = parallel_transport_frames(centerline, s)
    return ccta, lumen, frames


def test_cross_section_is_centered_on_lumen():
    ccta, lumen, frames = make_synthetic_case()
    section = generate_cross_section(ccta, lumen, frames, 20, size_mm=8.0, spacing_mm=0.10)
    assert section["image"].shape == (81, 81)
    assert section["lumen_mask"].shape == (81, 81)
    assert section["lumen_mask"][40, 40]


def test_longitudinal_shape_and_centerline_signal():
    ccta, _, frames = make_synthetic_case()
    image, radial = generate_longitudinal_reformation(
        ccta, frames, 0.0, radial_extent_mm=4.0, radial_spacing_mm=0.10
    )
    assert image.shape == (len(radial), len(frames.arc_length_mm))
    center_idx = int(np.argmin(np.abs(radial)))
    assert np.median(image[center_idx]) > 300.0


def test_inner_wall_matches_known_cylinder_radius():
    _, lumen, frames = make_synthetic_case()
    _, radii = inner_radii_from_lumen_mask(
        lumen, frames, n_theta=16, max_radius_mm=4.0, radial_step_mm=0.05
    )
    assert np.median(radii) == np.testing.assert_allclose(np.median(radii), 2.0, atol=0.15)
