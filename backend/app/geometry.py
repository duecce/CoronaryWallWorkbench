from __future__ import annotations

import numpy as np
from scipy.interpolate import interp1d
from scipy.ndimage import map_coordinates

from .models import CoronaryPath, ParallelTransportFrames


def cumulative_arc_length(points: np.ndarray) -> np.ndarray:
    points = np.asarray(points, dtype=float)
    if len(points) < 2:
        raise ValueError("A path requires at least two points")
    ds = np.linalg.norm(np.diff(points, axis=0), axis=1)
    return np.r_[0.0, np.cumsum(ds)]


def resample_centerline(path: CoronaryPath, step_mm: float) -> tuple[np.ndarray, np.ndarray]:
    s = path.arc_length_mm
    total = float(s[-1])
    if total <= 0:
        raise ValueError("Path length must be positive")
    sample_s = np.arange(0.0, total, step_mm)
    if not np.isclose(sample_s[-1], total):
        sample_s = np.r_[sample_s, total]
    f = interp1d(s, path.centerline_xyz_mm, axis=0, kind="linear")
    return np.asarray(f(sample_s), dtype=float), sample_s


def _rodrigues(v: np.ndarray, axis: np.ndarray, angle: float) -> np.ndarray:
    axis = axis / np.linalg.norm(axis)
    return (
        v * np.cos(angle)
        + np.cross(axis, v) * np.sin(angle)
        + axis * np.dot(axis, v) * (1.0 - np.cos(angle))
    )


def parallel_transport_frames(centerline_xyz_mm: np.ndarray, arc_length_mm: np.ndarray) -> ParallelTransportFrames:
    p = np.asarray(centerline_xyz_mm, dtype=float)
    tangent = np.gradient(p, arc_length_mm, axis=0, edge_order=1)
    tangent /= np.linalg.norm(tangent, axis=1, keepdims=True).clip(min=1e-12)

    t0 = tangent[0]
    basis_candidates = np.eye(3)
    seed = basis_candidates[np.argmin(np.abs(basis_candidates @ t0))]
    u0 = seed - np.dot(seed, t0) * t0
    u0 /= np.linalg.norm(u0)
    v0 = np.cross(t0, u0)
    v0 /= np.linalg.norm(v0)

    u = np.zeros_like(p)
    v = np.zeros_like(p)
    u[0], v[0] = u0, v0
    for i in range(1, len(p)):
        a, b = tangent[i - 1], tangent[i]
        axis = np.cross(a, b)
        axis_norm = np.linalg.norm(axis)
        if axis_norm < 1e-10:
            ui = u[i - 1]
        else:
            angle = float(np.arctan2(axis_norm, np.clip(np.dot(a, b), -1.0, 1.0)))
            ui = _rodrigues(u[i - 1], axis / axis_norm, angle)
        ui -= np.dot(ui, b) * b
        ui /= np.linalg.norm(ui).clip(min=1e-12)
        vi = np.cross(b, ui)
        vi /= np.linalg.norm(vi).clip(min=1e-12)
        u[i], v[i] = ui, vi

    return ParallelTransportFrames(
        centerline_xyz_mm=p,
        arc_length_mm=np.asarray(arc_length_mm, dtype=float),
        tangent=tangent,
        normal_u=u,
        normal_v=v,
    )


def prepare_frames(path: CoronaryPath, step_mm: float = 0.5) -> ParallelTransportFrames:
    centerline, s = resample_centerline(path, step_mm)
    return parallel_transport_frames(centerline, s)


def _plane_world_coordinates(
    frames: ParallelTransportFrames,
    s_index: int,
    *,
    size_mm: float,
    spacing_mm: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    half = size_mm / 2.0
    axis = np.arange(-half, half + spacing_mm * 0.5, spacing_mm)
    xx, yy = np.meshgrid(axis, axis, indexing="xy")
    center = frames.centerline_xyz_mm[s_index]
    u = frames.normal_u[s_index]
    v = frames.normal_v[s_index]
    world = center[None, None, :] + xx[..., None] * u + yy[..., None] * v
    return world, xx, yy


def _sample_volume(volume: np.ndarray, affine: np.ndarray, world_xyz: np.ndarray, order: int) -> np.ndarray:
    inv = np.linalg.inv(affine)
    flat = world_xyz.reshape(-1, 3)
    hom = np.c_[flat, np.ones(len(flat))]
    ijk = (inv @ hom.T).T[:, :3]
    sampled = map_coordinates(
        np.asarray(volume),
        [ijk[:, 0], ijk[:, 1], ijk[:, 2]],
        order=order,
        mode="constant",
        cval=0.0,
        prefilter=order > 1,
    )
    return sampled.reshape(world_xyz.shape[:2])


def generate_cross_section(
    ccta: np.ndarray,
    lumen_mask: np.ndarray,
    affine: np.ndarray,
    frames: ParallelTransportFrames,
    s_index: int,
    *,
    size_mm: float = 10.0,
    spacing_mm: float = 0.10,
) -> dict:
    world, _, _ = _plane_world_coordinates(
        frames, s_index, size_mm=size_mm, spacing_mm=spacing_mm
    )
    image = _sample_volume(ccta, affine, world, order=1)
    mask = _sample_volume(lumen_mask.astype(np.uint8), affine, world, order=0) > 0
    return {
        "image": image,
        "lumen_mask": mask,
        "world_xyz": world,
        "size_mm": float(size_mm),
        "spacing_mm": float(spacing_mm),
    }


def generate_longitudinal_reformation(
    volume: np.ndarray,
    affine: np.ndarray,
    frames: ParallelTransportFrames,
    theta_rad: float,
    *,
    radial_extent_mm: float = 5.0,
    radial_spacing_mm: float = 0.10,
    order: int = 1,
) -> tuple[np.ndarray, np.ndarray]:
    radial = np.arange(-radial_extent_mm, radial_extent_mm + radial_spacing_mm * 0.5, radial_spacing_mm)
    direction = (
        np.cos(theta_rad) * frames.normal_u
        + np.sin(theta_rad) * frames.normal_v
    )
    world = frames.centerline_xyz_mm[:, None, :] + radial[None, :, None] * direction[:, None, :]
    sampled = _sample_volume(volume, affine, world, order=order)
    return sampled, radial
