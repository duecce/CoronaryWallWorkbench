from __future__ import annotations

from pathlib import Path
from typing import Literal
import shutil

import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from .geometry import generate_cross_section, generate_longitudinal_reformation
from .state import registry
from .walls import contour_plane_xy, control_points_plane_xy, edit_control_node, longitudinal_wall_profiles, set_active_wall

app = FastAPI(title="CoronaryWallWorkbench API", version="0.3.0")
app.add_middleware(CORSMiddleware, allow_origins=["http://localhost:5173","http://127.0.0.1:5173"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])
WallName = Literal["inner","outer"]
DATA_ROOT = Path.home()/".coronarywallworkbench"/"cases"


class PreparePathRequest(BaseModel):
    centerline_step_mm: float = Field(default=0.5, gt=0)
    cross_section_size_mm: float = Field(default=10.0, gt=0)
    cross_section_spacing_mm: float = Field(default=0.10, gt=0)
    n_theta: int = Field(default=16, ge=8, le=64)
    outer_offset_mm: float = Field(default=0.75, gt=0)
    longitudinal_radial_extent_mm: float = Field(default=5.0, gt=0)
    longitudinal_radial_spacing_mm: float = Field(default=0.10, gt=0)

class ActiveWallRequest(BaseModel): wall: WallName
class WallEditRequest(BaseModel):
    wall: WallName
    s_index: int = Field(ge=0)
    theta_index: int = Field(ge=0)
    delta_radius_mm: float
    sigma_s_mm: float = Field(default=1.5, gt=0)
    sigma_theta_nodes: float = Field(default=1.25, gt=0)
    make_anchor: bool = True


def _prepared(case_id: str, path_id: str):
    try:
        loaded = registry.get(case_id)
        return loaded, loaded.prepared_paths[path_id]
    except KeyError as exc:
        raise HTTPException(404, str(exc)) from exc


def _case_payload(loaded) -> dict:
    q = loaded.case.qa
    return {"case_id":loaded.case.case_id,"qa":{"passed":q.passed,"same_shape":q.same_shape,"geometry_max_abs_diff_mm":q.geometry_max_abs_diff_mm,"centerline_inside_volume_fraction":q.centerline_inside_volume_fraction,"centerline_inside_lumen_fraction":q.centerline_inside_lumen_fraction,"median_centerline_to_lumen_mm":q.median_centerline_to_lumen_mm,"p95_centerline_to_lumen_mm":q.p95_centerline_to_lumen_mm,"warnings":q.warnings},"paths":[{"path_id":p.path_id,"coronary_name":p.coronary_name,"length_mm":float(p.arc_length_mm[-1]),"labels":sorted({x for x in p.anatomical_labels if x})} for p in loaded.paths.values()]}


def _surface_payload(prepared, s_index: int) -> dict:
    s=prepared.surface
    return {"s_index":s_index,"s_mm":float(s.s_mm[s_index]),"sample_count":len(s.s_mm),"active_wall":s.active_wall,"theta_rad":s.theta_rad.tolist(),"inner":{"radii_mm":s.inner_radii_mm[s_index].tolist(),"control_points_mm":control_points_plane_xy(s,"inner",s_index).tolist(),"contour_mm":contour_plane_xy(s,"inner",s_index).tolist(),"anchors":s.inner_anchors[s_index].tolist(),"editable":s.active_wall=="inner"},"outer":{"radii_mm":s.outer_radii_mm[s_index].tolist(),"control_points_mm":control_points_plane_xy(s,"outer",s_index).tolist(),"contour_mm":contour_plane_xy(s,"outer",s_index).tolist(),"anchors":s.outer_anchors[s_index].tolist(),"editable":s.active_wall=="outer"}}


def _save_upload(upload: UploadFile, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as f: shutil.copyfileobj(upload.file, f)


@app.get("/health")
def health(): return {"status":"ok"}

@app.post("/api/cases/upload")
def upload_case(case_id: str=Form(...), require_alignment: bool=Form(True), ccta: UploadFile=File(...), lumen_mask: UploadFile=File(...), coronary_xml: UploadFile=File(...)) -> dict:
    case_dir=DATA_ROOT/case_id; case_dir.mkdir(parents=True, exist_ok=True)
    ccta_p=case_dir/(ccta.filename or "ccta.nii.gz"); lumen_p=case_dir/(lumen_mask.filename or "lumen.nii.gz"); xml_p=case_dir/(coronary_xml.filename or "coronary.xml")
    _save_upload(ccta,ccta_p); _save_upload(lumen_mask,lumen_p); _save_upload(coronary_xml,xml_p)
    try: loaded=registry.load_case(case_id=case_id,ccta_path=str(ccta_p),lumen_mask_path=str(lumen_p),xml_path=str(xml_p),require_alignment=require_alignment,case_dir=str(case_dir/"annotations"))
    except Exception as exc: raise HTTPException(400,str(exc)) from exc
    return _case_payload(loaded)

@app.post("/api/cases/{case_id}/paths/{path_id}/prepare")
def prepare_path(case_id:str,path_id:str,request:PreparePathRequest)->dict:
    try: p=registry.prepare_path(case_id,path_id,**request.model_dump())
    except Exception as exc: raise HTTPException(400,str(exc)) from exc
    return {"path_id":path_id,"sample_count":len(p.frames.arc_length_mm),"length_mm":float(p.frames.arc_length_mm[-1]),"n_theta":len(p.surface.theta_rad),"active_wall":p.surface.active_wall}

@app.get("/api/cases/{case_id}/paths/{path_id}/sections/{s_index}")
def get_section(case_id:str,path_id:str,s_index:int)->dict:
    loaded,p=_prepared(case_id,path_id)
    if not 0<=s_index<len(p.frames.arc_length_mm): raise HTTPException(404,"s_index out of range")
    x=generate_cross_section(loaded.case.ccta.image,loaded.case.lumen_mask.image,p.frames,s_index,size_mm=p.cross_section_size_mm,spacing_mm=p.cross_section_spacing_mm)
    payload=_surface_payload(p,s_index); payload.update({"image_hu":x["image"].tolist(),"lumen_mask":x["lumen_mask"].astype(np.uint8).tolist(),"size_mm":x["size_mm"],"spacing_mm":x["spacing_mm"]}); return payload

@app.get("/api/cases/{case_id}/paths/{path_id}/longitudinal/{theta_index}")
def get_longitudinal(case_id:str,path_id:str,theta_index:int)->dict:
    loaded,p=_prepared(case_id,path_id); s=p.surface
    if not 0<=theta_index<len(s.theta_rad): raise HTTPException(404,"theta_index out of range")
    image,radial=generate_longitudinal_reformation(loaded.case.ccta.image,p.frames,float(s.theta_rad[theta_index]),radial_extent_mm=p.longitudinal_radial_extent_mm,radial_spacing_mm=p.longitudinal_radial_spacing_mm)
    ip=longitudinal_wall_profiles(s,"inner",theta_index); op=longitudinal_wall_profiles(s,"outer",theta_index)
    return {"theta_index":theta_index,"theta_rad":float(s.theta_rad[theta_index]),"s_mm":s.s_mm.tolist(),"radial_mm":radial.tolist(),"image_hu":image.tolist(),"active_wall":s.active_wall,"inner":{k:(v.tolist() if hasattr(v,"tolist") else int(v)) for k,v in ip.items()},"outer":{k:(v.tolist() if hasattr(v,"tolist") else int(v)) for k,v in op.items()}}

@app.post("/api/cases/{case_id}/paths/{path_id}/active-wall")
def active_wall(case_id:str,path_id:str,request:ActiveWallRequest):
    _,p=_prepared(case_id,path_id); set_active_wall(p.surface,request.wall); return {"active_wall":p.surface.active_wall}

@app.post("/api/cases/{case_id}/paths/{path_id}/edit")
def edit_wall(case_id:str,path_id:str,request:WallEditRequest):
    _,p=_prepared(case_id,path_id)
    try: edit_control_node(p.surface,**request.model_dump()); saved=registry.save_annotation(case_id,path_id)
    except Exception as exc: raise HTTPException(400,str(exc)) from exc
    return {"saved_to":str(saved),**_surface_payload(p,request.s_index)}

@app.post("/api/cases/{case_id}/paths/{path_id}/save")
def save(case_id:str,path_id:str): return {"saved_to":str(registry.save_annotation(case_id,path_id))}
