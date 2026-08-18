from __future__ import annotations

from typing import Literal

import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from .geometry import generate_cross_section
from .state import registry
from .walls import (
    contour_plane_xy,
    control_points_plane_xy,
    edit_control_node,
    set_active_wall,
)

app = FastAPI(
    title="CoronaryWallWorkbench API",
    version="0.2.0",
    description="API for coronary case loading, cross-sectional reformation, and assisted wall annotation.",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

WallName = Literal["inner", "outer"]


class LoadCaseRequest(BaseModel):
    case_id: str
    ccta_path: str
    lumen_mask_path: str
    xml_path: str
    require_alignment: bool = True


class PreparePathRequest(BaseModel):
    centerline_step_mm: float = Field(default=0.5, gt=0.0)
    cross_section_size_mm: float = Field(default=10.0, gt=0.0)
    cross_section_spacing_mm: float = Field(default=0.10, gt=0.0)
    n_theta: int = Field(default=16, ge=8, le=64)
    outer_offset_mm: float = Field(default=0.75, gt=0.0)


class ActiveWallRequest(BaseModel):
    wall: WallName


class WallEditRequest(BaseModel):
    wall: WallName
    s_index: int = Field(ge=0)
    theta_index: int = Field(ge=0)
    delta_radius_mm: float
    sigma_s_mm: float = Field(default=1.5, gt=0.0)
    sigma_theta_nodes: float = Field(default=1.25, gt=0.0)
    make_anchor: bool = True


def _prepared(case_id: str, path_id: str):
    try:
        loaded = registry.get(case_id)
        return loaded, loaded.prepared_paths[path_id]
    except (KeyError, IndexError) as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


def _surface_payload(prepared, s_index: int) -> dict:
    surface = prepared.surface
    return {
        "s_index": s_index,
        "s_mm": float(surface.s_mm[s_index]),
        "active_wall": surface.active_wall,
        "theta_rad": surface.theta_rad.tolist(),
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


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/api/cases/load")
def load_case(request: LoadCaseRequest) -> dict:
    try:
        loaded = registry.load_case(**request.model_dump())
    except (ValueError, RuntimeError, OSError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    qa = loaded.case.qa
    return {
        "case_id": loaded.case.case_id,
        "coordinate_system": loaded.case.graph.coordinate_system,
        "qa": {
            "passed": qa.passed,
            "same_shape": qa.same_shape,
            "affine_max_abs_diff_mm": qa.affine_max_abs_diff_mm,
            "centerline_inside_volume_fraction": qa.centerline_inside_volume_fraction,
            "centerline_inside_lumen_fraction": qa.centerline_inside_lumen_fraction,
            "median_centerline_to_lumen_mm": qa.median_centerline_to_lumen_mm,
            "p95_centerline_to_lumen_mm": qa.p95_centerline_to_lumen_mm,
            "warnings": qa.warnings,
        },
        "paths": [
            {
                "path_id": p.path_id,
                "coronary_name": p.coronary_name,
                "length_mm": float(p.arc_length_mm[-1]),
                "node_count": len(p.node_ids),
                "labels": sorted({label for label in p.anatomical_labels if label}),
            }
            for p in loaded.paths.values()
        ],
    }


@app.post("/api/cases/{case_id}/paths/{path_id}/prepare")
def prepare_path(case_id: str, path_id: str, request: PreparePathRequest) -> dict:
    try:
        prepared = registry.prepare_path(case_id, path_id, **request.model_dump())
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    return {
        "path_id": path_id,
        "sample_count": len(prepared.frames.arc_length_mm),
        "length_mm": float(prepared.frames.arc_length_mm[-1]),
        "n_theta": len(prepared.surface.theta_rad),
        "active_wall": prepared.surface.active_wall,
        "cross_section_size_mm": prepared.cross_section_size_mm,
        "cross_section_spacing_mm": prepared.cross_section_spacing_mm,
    }


@app.get("/api/cases/{case_id}/paths/{path_id}/sections/{s_index}")
def get_cross_section(case_id: str, path_id: str, s_index: int) -> dict:
    loaded, prepared = _prepared(case_id, path_id)
    if not 0 <= s_index < len(prepared.frames.arc_length_mm):
        raise HTTPException(status_code=404, detail="s_index out of range")

    section = generate_cross_section(
        loaded.case.ccta.data,
        loaded.case.lumen_mask.data,
        loaded.case.ccta.affine,
        prepared.frames,
        s_index,
        size_mm=prepared.cross_section_size_mm,
        spacing_mm=prepared.cross_section_spacing_mm,
    )
    payload = _surface_payload(prepared, s_index)
    payload.update(
        {
            "image_hu": np.asarray(section["image"]).tolist(),
            "lumen_mask": np.asarray(section["lumen_mask"], dtype=np.uint8).tolist(),
            "size_mm": section["size_mm"],
            "spacing_mm": section["spacing_mm"],
        }
    )
    return payload


@app.post("/api/cases/{case_id}/paths/{path_id}/active-wall")
def select_active_wall(case_id: str, path_id: str, request: ActiveWallRequest) -> dict:
    _, prepared = _prepared(case_id, path_id)
    set_active_wall(prepared.surface, request.wall)
    return {"active_wall": prepared.surface.active_wall}


@app.post("/api/cases/{case_id}/paths/{path_id}/edit")
def edit_wall(case_id: str, path_id: str, request: WallEditRequest) -> dict:
    _, prepared = _prepared(case_id, path_id)
    try:
        edit_control_node(prepared.surface, **request.model_dump())
    except (ValueError, IndexError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return _surface_payload(prepared, request.s_index)
