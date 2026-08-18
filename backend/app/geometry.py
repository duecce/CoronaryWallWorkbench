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


def resample_centerline(path: CoronaryPath, step_mm: float) -> tuple[np.ndarray, np.ndarray]:
    s = path.arc_length_mm
    total = float(s[-1])
    sample_s = np.arange(0.0, total, step_mm)
    if len(sample_s) == 0 or not np.isclose(sample_s[-1], total):
        sample_s = np.r_[sample_s, total]
    f = interp1d(s, path.centerline_xyz_mm, axis=0, kind="linear")
    return np.asarray(f(sample_s), dtype=float), sample_s


def _rodrigues(v: np.ndarray, axis: np.ndarray, angle: float) -> np.ndarray:
    axis = axis / np.linalg.norm(axis)
    return v*np.cos(angle) + np.cross(axis, v)*np.sin(angle) + axis*np.dot(axis, v)*(1-np.cos(angle))


def parallel_transport_frames(centerline_xyz_mm: np.ndarray, arc_length_mm: np.ndarray) -> ParallelTransportFrames:
    p = np.asarray(centerline_xyz_mm, dtype=float)
    tangent = np.gradient(p, arc_length_mm, axis=0, edge_order=1)
    tangent /= np.linalg.norm(tangent, axis=1, keepdims=True).clip(min=1e-12)
    t0 = tangent[0]
    seed = np.eye(3)[np.argmin(np.abs(np.eye(3) @ t0))]
    u0 = seed - np.dot(seed, t0)*t0
    u0 /= np.linalg.norm(u0)
    v0 = np.cross(t0, u0); v0 /= np.linalg.norm(v0)
    u = np.zeros_like(p); v = np.zeros_like(p); u[0] = u0; v[0] = v0
    for i in range(1, len(p)):
        a, b = tangent[i-1], tangent[i]
        axis = np.cross(a, b); n = np.linalg.norm(axis)
        ui = u[i-1] if n < 1e-10 else _rodrigues(u[i-1], axis/n, float(np.arctan2(n, np.clip(np.dot(a,b),-1,1))))
        ui -= np.dot(ui, b)*b; ui /= np.linalg.norm(ui).clip(min=1e-12)
        vi = np.cross(b, ui); vi /= np.linalg.norm(vi).clip(min=1e-12)
        u[i], v[i] = ui, vi
    return ParallelTransportFrames(p, np.asarray(arc_length_mm), tangent, u, v)


def prepare_frames(path: CoronaryPath, step_mm: float = 0.5) -> ParallelTransportFrames:
    centerline, s = resample_centerline(path, step_mm)
    return parallel_transport_frames(centerline, s)


def _cross_reference(frames: ParallelTransportFrames, s_index: int, size_mm: float, spacing_mm: float) -> sitk.Image:
    n = int(round(size_mm / spacing_mm)) + 1
    center = frames.centerline_xyz_mm[s_index]
    u, v, t = frames.normal_u[s_index], frames.normal_v[s_index], frames.tangent[s_index]
    origin = center - (size_mm/2.0)*u - (size_mm/2.0)*v
    ref = sitk.Image([n, n, 1], sitk.sitkFloat32)
    ref.SetSpacing([spacing_mm, spacing_mm, 1.0])
    ref.SetOrigin(tuple(origin))
    ref.SetDirection(tuple(np.column_stack([u, v, t]).ravel(order="C")))
    return ref


def generate_cross_section(ccta: sitk.Image, lumen_mask: sitk.Image, frames: ParallelTransportFrames, s_index: int, *, size_mm: float=10.0, spacing_mm: float=0.10) -> dict:
    ref = _cross_reference(frames, s_index, size_mm, spacing_mm)
    ct = sitk.Resample(ccta, ref, sitk.Transform(3, sitk.sitkIdentity), sitk.sitkLinear, -1024.0, sitk.sitkFloat32)
    mask = sitk.Resample(lumen_mask, ref, sitk.Transform(3, sitk.sitkIdentity), sitk.sitkNearestNeighbor, 0, sitk.sitkUInt8)
    return {
        "image": sitk.GetArrayViewFromImage(ct)[0].copy(),
        "lumen_mask": (sitk.GetArrayViewFromImage(mask)[0] > 0).copy(),
        "size_mm": float(size_mm),
        "spacing_mm": float(spacing_mm),
    }


def generate_longitudinal_reformation(image: sitk.Image, frames: ParallelTransportFrames, theta_rad: float, *, radial_extent_mm: float=5.0, radial_spacing_mm: float=0.10, nearest: bool=False) -> tuple[np.ndarray, np.ndarray]:
    radial = np.arange(-radial_extent_mm, radial_extent_mm + 0.5*radial_spacing_mm, radial_spacing_mm)
    ns, nr = len(frames.arc_length_mm), len(radial)
    ds = float(np.median(np.diff(frames.arc_length_mm))) if ns > 1 else 1.0
    direction = np.cos(theta_rad)*frames.normal_u + np.sin(theta_rad)*frames.normal_v
    target = frames.centerline_xyz_mm[:,None,:] + radial[None,:,None]*direction[:,None,:]

    ref = sitk.Image([ns, nr, 1], sitk.sitkFloat32)
    ref.SetSpacing([ds, radial_spacing_mm, 1.0]); ref.SetOrigin([0.0, -radial_extent_mm, 0.0]); ref.SetDirection(np.eye(3).ravel().tolist())
    out_phys = np.zeros_like(target)
    out_phys[...,0] = frames.arc_length_mm[:,None]
    out_phys[...,1] = radial[None,:]
    displacement = target - out_phys
    field_arr = np.transpose(displacement, (1,0,2))[None,...]
    field = sitk.GetImageFromArray(field_arr.astype(np.float64), isVector=True)
    field.CopyInformation(ref)
    transform = sitk.DisplacementFieldTransform(field)
    interp = sitk.sitkNearestNeighbor if nearest else sitk.sitkLinear
    resampled = sitk.Resample(image, ref, transform, interp, 0.0, sitk.sitkFloat32)
    array = sitk.GetArrayViewFromImage(resampled)[0].copy().T
    return array, radial
