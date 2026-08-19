from __future__ import annotations

import numpy as np
import SimpleITK as sitk
from scipy.interpolate import interp1d

from .models import CoronaryPath, ParallelTransportFrames


def cumulative_arc_length(points: np.ndarray) -> np.ndarray:
    points = np.asarray(points, dtype=float)
    if len(points) < 2:
        raise ValueError("A path requires at least two points")
    ds = np.linalg.norm(np.diff(points, axis=0), axis=1)
    return np.r_[0.0, np.cumsum(ds)]


def resample_centerline(
    path: CoronaryPath,
    step_mm: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Resample a path at a strictly uniform longitudinal spacing.

    A uniform s-grid is essential for metric-correct sCPR rendering.  The
    requested step is an upper bound: the actual spacing is total/(N-1), so
    the final interval is never shorter than the preceding intervals.
    """

    if step_mm <= 0:
        raise ValueError("centerline step must be positive")
    s = np.asarray(path.arc_length_mm, dtype=float)
    total = float(s[-1])
    if total <= 0:
        raise ValueError("Path length must be positive")
    intervals = max(1, int(np.ceil(total / float(step_mm))))
    sample_s = np.linspace(0.0, total, intervals + 1, dtype=float)
    interpolator = interp1d(
        s,
        path.centerline_xyz_mm,
        axis=0,
        kind="linear",
        assume_sorted=True,
    )
    return np.asarray(interpolator(sample_s), dtype=float), sample_s


def _unit(vector: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    if norm < 1e-12:
        raise ValueError("Degenerate frame vector")
    return np.asarray(vector, dtype=float) / norm


def _initial_rmf_normal(tangent: np.ndarray) -> np.ndarray:
    """Choose a deterministic LPS-referenced initial frame axis.

    Prefer Superior (+Z), then Posterior (+Y), then Left (+X), selecting the
    axis whose projection onto the cross-sectional plane has the largest norm.
    This makes theta=0 reproducible between sessions and independent paths.
    """

    candidates = (
        np.array([0.0, 0.0, 1.0]),
        np.array([0.0, 1.0, 0.0]),
        np.array([1.0, 0.0, 0.0]),
    )
    projected: list[tuple[float, np.ndarray]] = []
    for axis in candidates:
        normal = axis - np.dot(axis, tangent) * tangent
        projected.append((float(np.linalg.norm(normal)), normal))
    _, best = max(projected, key=lambda item: item[0])
    return _unit(best)


def rotation_minimizing_frames(
    centerline_xyz_mm: np.ndarray,
    arc_length_mm: np.ndarray,
) -> ParallelTransportFrames:
    """Compute a discrete rotation-minimizing frame by double reflection.

    This is the Bishop/RMF construction used to avoid the twist and frame
    flips typical of Frenet frames on nearly straight coronary segments.
    """

    points = np.asarray(centerline_xyz_mm, dtype=float)
    s = np.asarray(arc_length_mm, dtype=float)
    if len(points) < 2 or points.shape != (len(s), 3):
        raise ValueError("Invalid centerline for RMF")

    tangent = np.gradient(points, s, axis=0, edge_order=1)
    tangent /= np.linalg.norm(tangent, axis=1, keepdims=True).clip(min=1e-12)

    normal_u = np.zeros_like(points)
    normal_v = np.zeros_like(points)
    normal_u[0] = _initial_rmf_normal(tangent[0])
    normal_v[0] = _unit(np.cross(tangent[0], normal_u[0]))

    for index in range(len(points) - 1):
        chord = points[index + 1] - points[index]
        chord_squared = float(np.dot(chord, chord))
        if chord_squared < 1e-14:
            transported = normal_u[index].copy()
        else:
            # First reflection: across the plane normal to the centerline chord.
            reflected_u = normal_u[index] - (
                2.0 * np.dot(chord, normal_u[index]) / chord_squared
            ) * chord
            reflected_tangent = tangent[index] - (
                2.0 * np.dot(chord, tangent[index]) / chord_squared
            ) * chord

            # Second reflection aligns the reflected tangent to t_(i+1) with
            # the least possible rotation of the transverse frame.
            correction = tangent[index + 1] - reflected_tangent
            correction_squared = float(np.dot(correction, correction))
            if correction_squared < 1e-14:
                transported = reflected_u
            else:
                transported = reflected_u - (
                    2.0 * np.dot(correction, reflected_u) / correction_squared
                ) * correction

        # Re-orthogonalize only to remove accumulated floating-point error.
        transported -= (
            np.dot(transported, tangent[index + 1]) * tangent[index + 1]
        )
        normal_u[index + 1] = _unit(transported)
        normal_v[index + 1] = _unit(
            np.cross(tangent[index + 1], normal_u[index + 1])
        )

    return ParallelTransportFrames(
        centerline_xyz_mm=points,
        arc_length_mm=s,
        tangent=tangent,
        normal_u=normal_u,
        normal_v=normal_v,
    )


# Backward-compatible alias.  The implementation is now explicitly an RMF.
def parallel_transport_frames(
    centerline_xyz_mm: np.ndarray,
    arc_length_mm: np.ndarray,
) -> ParallelTransportFrames:
    return rotation_minimizing_frames(centerline_xyz_mm, arc_length_mm)


def prepare_frames(
    path: CoronaryPath,
    step_mm: float = 0.5,
) -> ParallelTransportFrames:
    centerline, s = resample_centerline(path, step_mm)
    return rotation_minimizing_frames(centerline, s)


def _cross_reference(
    frames: ParallelTransportFrames,
    s_index: int,
    size_mm: float,
    spacing_mm: float,
) -> sitk.Image:
    """Build a square metric reference grid centered exactly on the centerline."""

    n = int(round(float(size_mm) / float(spacing_mm))) + 1
    actual_extent = float(n - 1) * float(spacing_mm)
    center = frames.centerline_xyz_mm[s_index]
    u = frames.normal_u[s_index]
    v = frames.normal_v[s_index]
    t = frames.tangent[s_index]
    origin = center - 0.5 * actual_extent * u - 0.5 * actual_extent * v

    reference = sitk.Image([n, n, 1], sitk.sitkFloat32)
    reference.SetSpacing([float(spacing_mm), float(spacing_mm), 1.0])
    reference.SetOrigin(tuple(float(value) for value in origin))
    reference.SetDirection(
        tuple(np.column_stack([u, v, t]).ravel(order="C"))
    )
    return reference


def generate_cross_section(
    ccta: sitk.Image,
    lumen_mask: sitk.Image,
    frames: ParallelTransportFrames,
    s_index: int,
    *,
    size_mm: float = 15.0,
    spacing_mm: float = 0.10,
) -> dict:
    reference = _cross_reference(frames, s_index, size_mm, spacing_mm)
    ct = sitk.Resample(
        ccta,
        reference,
        sitk.Transform(3, sitk.sitkIdentity),
        sitk.sitkLinear,
        -1024.0,
        sitk.sitkFloat32,
    )
    mask = sitk.Resample(
        lumen_mask,
        reference,
        sitk.Transform(3, sitk.sitkIdentity),
        sitk.sitkNearestNeighbor,
        0,
        sitk.sitkUInt8,
    )
    array = sitk.GetArrayViewFromImage(ct)[0].copy()
    mask_array = (sitk.GetArrayViewFromImage(mask)[0] > 0).copy()
    physical_extent = float(reference.GetSpacing()[0] * (reference.GetSize()[0] - 1))
    return {
        "image": array,
        "lumen_mask": mask_array,
        "size_mm": physical_extent,
        "spacing_mm": float(spacing_mm),
    }


def generate_longitudinal_reformation(
    image: sitk.Image,
    frames: ParallelTransportFrames,
    theta_rad: float,
    *,
    radial_extent_mm: float = 7.5,
    radial_spacing_mm: float = 0.10,
    nearest: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate a metric-preserving curved longitudinal reformation.

    Output array convention is [radial, longitudinal].  The physical output
    spacing is explicitly (ds, dr), so no geometric zoom is introduced by the
    resampler.  The web viewer must preserve this same metric aspect ratio.
    """

    radial = np.arange(
        -float(radial_extent_mm),
        float(radial_extent_mm) + 0.5 * float(radial_spacing_mm),
        float(radial_spacing_mm),
    )
    n_s = len(frames.arc_length_mm)
    n_r = len(radial)
    ds = (
        float(frames.arc_length_mm[1] - frames.arc_length_mm[0])
        if n_s > 1
        else 1.0
    )
    if n_s > 2 and not np.allclose(
        np.diff(frames.arc_length_mm), ds, rtol=1e-6, atol=1e-8
    ):
        raise ValueError("sCPR requires uniformly sampled RMF frames")

    direction = (
        np.cos(theta_rad) * frames.normal_u
        + np.sin(theta_rad) * frames.normal_v
    )
    target = (
        frames.centerline_xyz_mm[:, None, :]
        + radial[None, :, None] * direction[:, None, :]
    )

    reference = sitk.Image([n_s, n_r, 1], sitk.sitkFloat32)
    reference.SetSpacing([ds, float(radial_spacing_mm), 1.0])
    reference.SetOrigin([0.0, -float(radial_extent_mm), 0.0])
    reference.SetDirection(np.eye(3).ravel().tolist())

    grid_s = np.arange(n_s, dtype=float) * ds
    output_physical = np.zeros_like(target)
    output_physical[..., 0] = grid_s[:, None]
    output_physical[..., 1] = radial[None, :]
    displacement = target - output_physical

    # SimpleITK array layout: z, y(radial), x(longitudinal), vector-components.
    field_array = np.transpose(displacement, (1, 0, 2))[None, ...]
    field = sitk.GetImageFromArray(field_array.astype(np.float64), isVector=True)
    field.CopyInformation(reference)
    transform = sitk.DisplacementFieldTransform(field)

    interpolator = sitk.sitkNearestNeighbor if nearest else sitk.sitkLinear
    default_value = 0.0 if nearest else -1024.0
    resampled = sitk.Resample(
        image,
        reference,
        transform,
        interpolator,
        default_value,
        sitk.sitkFloat32,
    )
    return sitk.GetArrayViewFromImage(resampled)[0].copy(), radial
