from __future__ import annotations

from pathlib import Path
from typing import Literal
import re
import shutil

import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from .geometry import generate_cross_section, generate_longitudinal_reformation
from .state import registry
from .walls import (
    contour_plane_xy,
    control_points_plane_xy,
    edit_control_node,
    longitudinal_wall_profiles,
    remove_control_anchor,
    set_active_wall,
)

app = FastAPI(title="CoronaryWallWorkbench API", version="0.4.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
WallName = Literal["inner", "outer"]
DATA_ROOT = Path.home() / ".coronarywallworkbench" / "cases"


class PreparePathRequest(BaseModel):
    centerline_step_mm: float = Field(default=0.5, gt=0)
    cross_section_size_mm: float = Field(default=15.0, gt=0)
    cross_section_spacing_mm: float = Field(default=0.10, gt=0)
    n_theta: int = Field(default=96, ge=32, le=192)
    outer_offset_mm: float = Field(default=0.75, gt=0)
    longitudinal_radial_extent_mm: float = Field(default=7.5, gt=0)
    longitudinal_radial_spacing_mm: float = Field(default=0.10, gt=0)


class ActiveWallRequest(BaseModel):
    wall: WallName


class WallEditRequest(BaseModel):
    wall: WallName
    s_index: int = Field(ge=0)
    theta_index: int = Field(ge=0)
    delta_radius_mm: float
    # Retained for compatibility with older clients. Propagation is now
    # Fourier circumferential + B-spline longitudinal, as in the desktop tool.
    sigma_s_mm: float = Field(default=1.5, gt=0)
    sigma_theta_nodes: float = Field(default=1.25, gt=0)
    make_anchor: bool = True


class RemoveAnchorRequest(BaseModel):
    wall: WallName
    s_index: int = Field(ge=0)
    theta_index: int = Field(ge=0)


def _prepared(case_id: str, path_id: str):
    try:
        loaded = registry.get(case_id)
        return loaded, loaded.prepared_paths[path_id]
    except KeyError as exc:
        raise HTTPException(404, str(exc)) from exc


def _case_payload(loaded) -> dict:
    qa = loaded.case.qa
    return {
        "case_id": loaded.case.case_id,
        "qa": {
            "passed": qa.passed,
            "same_shape": qa.same_shape,
            "geometry_max_abs_diff_mm": qa.geometry_max_abs_diff_mm,
            "centerline_inside_volume_fraction": qa.centerline_inside_volume_fraction,
            "centerline_inside_lumen_fraction": qa.centerline_inside_lumen_fraction,
            "median_centerline_to_lumen_mm": qa.median_centerline_to_lumen_mm,
            "p95_centerline_to_lumen_mm": qa.p95_centerline_to_lumen_mm,
            "warnings": qa.warnings,
        },
        "paths": [
            {
                "path_id": path.path_id,
                "coronary_name": path.coronary_name,
                "length_mm": float(path.arc_length_mm[-1]),
                "labels": sorted({label for label in path.anatomical_labels if label}),
            }
            for path in loaded.paths.values()
        ],
    }


def _control_stride(n_theta: int) -> int:
    # 96 internal angles -> 24 visible control nodes (15-degree spacing).
    return max(1, int(round(n_theta / 24)))


def _surface_payload(prepared, s_index: int) -> dict:
    surface = prepared.surface
    return {
        "s_index": s_index,
        "s_mm": float(surface.s_mm[s_index]),
        "sample_count": len(surface.s_mm),
        "active_wall": surface.active_wall,
        "theta_rad": surface.theta_rad.tolist(),
        "control_stride": _control_stride(len(surface.theta_rad)),
        "frame_type": "rotation-minimizing-double-reflection",
        "inner": {
            "radii_mm": surface.inner_radii_mm[s_index].tolist(),
            "control_points_mm": control_points_plane_xy(surface, "inner", s_index).tolist(),
            "contour_mm": contour_plane_xy(surface, "inner", s_index).tolist(),
            "anchors": surface.inner_anchors[s_index].tolist(),
            "editable": surface.active_wall == "inner",
        },
        "outer": {
            "radii_mm": surface.outer_radii_mm[s_index].tolist(),
            "control_points_mm": control_points_plane_xy(surface, "outer", s_index).tolist(),
            "contour_mm": contour_plane_xy(surface, "outer", s_index).tolist(),
            "anchors": surface.outer_anchors[s_index].tolist(),
            "editable": surface.active_wall == "outer",
        },
    }


def _save_upload(upload: UploadFile, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as output:
        shutil.copyfileobj(upload.file, output)


def _safe_case_id(value: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9._-]+", "_", value.strip())
    if not safe or safe in {".", ".."}:
        raise HTTPException(400, "Invalid case ID")
    return safe


def _safe_filename(filename: str | None, fallback: str) -> str:
    return Path(filename or fallback).name


@app.get("/health")
def health():
    return {"status": "ok", "version": "0.4.0"}


@app.post("/api/cases/upload")
def upload_case(
    case_id: str = Form(...),
    require_alignment: bool = Form(True),
    ccta: UploadFile = File(...),
    lumen_mask: UploadFile = File(...),
    coronary_xml: UploadFile = File(...),
) -> dict:
    # Loading and graph/centerline extraction intentionally remain unchanged.
    case_id = _safe_case_id(case_id)
    case_dir = DATA_ROOT / case_id
    case_dir.mkdir(parents=True, exist_ok=True)
    ccta_path = case_dir / _safe_filename(ccta.filename, "ccta.nii.gz")
    lumen_path = case_dir / _safe_filename(lumen_mask.filename, "lumen.nii.gz")
    xml_path = case_dir / _safe_filename(coronary_xml.filename, "coronary.xml")
    _save_upload(ccta, ccta_path)
    _save_upload(lumen_mask, lumen_path)
    _save_upload(coronary_xml, xml_path)
    try:
        loaded = registry.load_case(
            case_id=case_id,
            ccta_path=str(ccta_path),
            lumen_mask_path=str(lumen_path),
            xml_path=str(xml_path),
            require_alignment=require_alignment,
            case_dir=str(case_dir / "annotations"),
        )
    except Exception as exc:
        raise HTTPException(400, str(exc)) from exc
    return _case_payload(loaded)


@app.post("/api/cases/{case_id}/paths/{path_id}/prepare")
def prepare_path(case_id: str, path_id: str, request: PreparePathRequest) -> dict:
    try:
        prepared = registry.prepare_path(case_id, path_id, **request.model_dump())
    except Exception as exc:
        raise HTTPException(400, str(exc)) from exc
    spacing = (
        float(prepared.frames.arc_length_mm[1] - prepared.frames.arc_length_mm[0])
        if len(prepared.frames.arc_length_mm) > 1
        else 0.0
    )
    return {
        "path_id": path_id,
        "sample_count": len(prepared.frames.arc_length_mm),
        "length_mm": float(prepared.frames.arc_length_mm[-1]),
        "centerline_spacing_mm": spacing,
        "n_theta": len(prepared.surface.theta_rad),
        "control_stride": _control_stride(len(prepared.surface.theta_rad)),
        "active_wall": prepared.surface.active_wall,
        "frame_type": "rotation-minimizing-double-reflection",
        "cross_section_fov_mm": prepared.cross_section_size_mm,
    }


@app.get("/api/cases/{case_id}/paths/{path_id}/sections/{s_index}")
def get_section(case_id: str, path_id: str, s_index: int) -> dict:
    loaded, prepared = _prepared(case_id, path_id)
    if not 0 <= s_index < len(prepared.frames.arc_length_mm):
        raise HTTPException(404, "s_index out of range")
    if s_index not in prepared.cross_section_cache:
        prepared.cross_section_cache[s_index] = generate_cross_section(
            loaded.case.ccta.image,
            loaded.case.lumen_mask.image,
            prepared.frames,
            s_index,
            size_mm=prepared.cross_section_size_mm,
            spacing_mm=prepared.cross_section_spacing_mm,
        )
    section = prepared.cross_section_cache[s_index]
    payload = _surface_payload(prepared, s_index)
    payload.update(
        {
            "image_hu": section["image"].tolist(),
            "lumen_mask": section["lumen_mask"].astype(np.uint8).tolist(),
            "size_mm": section["size_mm"],
            "spacing_mm": section["spacing_mm"],
        }
    )
    return payload


@app.get("/api/cases/{case_id}/paths/{path_id}/longitudinal/{theta_index}")
def get_longitudinal(case_id: str, path_id: str, theta_index: int) -> dict:
    loaded, prepared = _prepared(case_id, path_id)
    surface = prepared.surface
    if not 0 <= theta_index < len(surface.theta_rad):
        raise HTTPException(404, "theta_index out of range")
    if theta_index not in prepared.longitudinal_cache:
        prepared.longitudinal_cache[theta_index] = generate_longitudinal_reformation(
            loaded.case.ccta.image,
            prepared.frames,
            float(surface.theta_rad[theta_index]),
            radial_extent_mm=prepared.longitudinal_radial_extent_mm,
            radial_spacing_mm=prepared.longitudinal_radial_spacing_mm,
        )
    image, radial = prepared.longitudinal_cache[theta_index]
    inner = longitudinal_wall_profiles(surface, "inner", theta_index)
    outer = longitudinal_wall_profiles(surface, "outer", theta_index)
    longitudinal_spacing = (
        float(surface.s_mm[1] - surface.s_mm[0]) if len(surface.s_mm) > 1 else 1.0
    )
    return {
        "theta_index": theta_index,
        "theta_rad": float(surface.theta_rad[theta_index]),
        "s_mm": surface.s_mm.tolist(),
        "radial_mm": radial.tolist(),
        "image_hu": image.tolist(),
        "active_wall": surface.active_wall,
        "longitudinal_spacing_mm": longitudinal_spacing,
        "radial_spacing_mm": prepared.longitudinal_radial_spacing_mm,
        "metric_equal": True,
        "inner": {
            key: (value.tolist() if hasattr(value, "tolist") else int(value))
            for key, value in inner.items()
        },
        "outer": {
            key: (value.tolist() if hasattr(value, "tolist") else int(value))
            for key, value in outer.items()
        },
    }


@app.post("/api/cases/{case_id}/paths/{path_id}/active-wall")
def active_wall(case_id: str, path_id: str, request: ActiveWallRequest):
    _, prepared = _prepared(case_id, path_id)
    set_active_wall(prepared.surface, request.wall)
    return {"active_wall": prepared.surface.active_wall}


@app.post("/api/cases/{case_id}/paths/{path_id}/edit")
def edit_wall(case_id: str, path_id: str, request: WallEditRequest):
    _, prepared = _prepared(case_id, path_id)
    try:
        edit_control_node(prepared.surface, **request.model_dump())
        saved = registry.save_annotation(case_id, path_id)
    except Exception as exc:
        raise HTTPException(400, str(exc)) from exc
    return {"saved_to": str(saved), **_surface_payload(prepared, request.s_index)}


@app.post("/api/cases/{case_id}/paths/{path_id}/remove-anchor")
def remove_anchor(case_id: str, path_id: str, request: RemoveAnchorRequest):
    _, prepared = _prepared(case_id, path_id)
    try:
        remove_control_anchor(prepared.surface, **request.model_dump())
        saved = registry.save_annotation(case_id, path_id)
    except Exception as exc:
        raise HTTPException(400, str(exc)) from exc
    return {"saved_to": str(saved), **_surface_payload(prepared, request.s_index)}


@app.post("/api/cases/{case_id}/paths/{path_id}/save")
def save(case_id: str, path_id: str):
    return {"saved_to": str(registry.save_annotation(case_id, path_id))}
