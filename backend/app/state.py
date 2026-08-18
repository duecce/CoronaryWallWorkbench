from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import json
import numpy as np

from .case_loader import load_case
from .geometry import prepare_frames
from .models import PatientCase, PreparedPath
from .paths import reconstruct_leaf_paths
from .walls import initialize_wall_surface


@dataclass(slots=True)
class LoadedCase:
    case: PatientCase
    paths: dict
    prepared_paths: dict[str, PreparedPath] = field(default_factory=dict)


class CaseRegistry:
    def __init__(self) -> None:
        self._cases: dict[str, LoadedCase] = {}

    def load_case(self, **kwargs) -> LoadedCase:
        case = load_case(**kwargs)
        loaded = LoadedCase(case=case, paths=reconstruct_leaf_paths(case))
        self._cases[case.case_id] = loaded
        return loaded

    def get(self, case_id: str) -> LoadedCase:
        if case_id not in self._cases:
            raise KeyError(f"Unknown case_id: {case_id}")
        return self._cases[case_id]

    def annotation_path(self, case_id: str, path_id: str) -> Path:
        loaded = self.get(case_id)
        safe = "".join(c if c.isalnum() or c in "-_" else "_" for c in path_id)
        return loaded.case.case_dir / f"{safe}_walls.npz"

    def prepare_path(self, case_id: str, path_id: str, *, centerline_step_mm: float=0.5, cross_section_size_mm: float=10.0, cross_section_spacing_mm: float=0.10, n_theta: int=16, outer_offset_mm: float=0.75, longitudinal_radial_extent_mm: float=5.0, longitudinal_radial_spacing_mm: float=0.10) -> PreparedPath:
        loaded = self.get(case_id)
        if path_id not in loaded.paths:
            raise KeyError(f"Unknown path_id: {path_id}")
        path = loaded.paths[path_id]
        frames = prepare_frames(path, step_mm=centerline_step_mm)
        surface = initialize_wall_surface(loaded.case.lumen_mask.image, frames, n_theta=n_theta, outer_offset_mm=outer_offset_mm)
        prepared = PreparedPath(path, frames, surface, cross_section_size_mm, cross_section_spacing_mm, longitudinal_radial_extent_mm, longitudinal_radial_spacing_mm)
        saved = self.annotation_path(case_id, path_id)
        if saved.exists():
            d = np.load(saved, allow_pickle=False)
            if d["inner_radii_mm"].shape == surface.inner_radii_mm.shape and np.allclose(d["s_mm"], surface.s_mm) and np.allclose(d["theta_rad"], surface.theta_rad):
                surface.inner_radii_mm[:] = d["inner_radii_mm"]
                surface.outer_radii_mm[:] = d["outer_radii_mm"]
                surface.inner_anchors[:] = d["inner_anchors"].astype(bool)
                surface.outer_anchors[:] = d["outer_anchors"].astype(bool)
        loaded.prepared_paths[path_id] = prepared
        return prepared

    def save_annotation(self, case_id: str, path_id: str) -> Path:
        loaded = self.get(case_id)
        prepared = loaded.prepared_paths[path_id]
        p = self.annotation_path(case_id, path_id)
        p.parent.mkdir(parents=True, exist_ok=True)
        s = prepared.surface
        np.savez_compressed(p, s_mm=s.s_mm, theta_rad=s.theta_rad, inner_radii_mm=s.inner_radii_mm, outer_radii_mm=s.outer_radii_mm, inner_anchors=s.inner_anchors.astype(np.uint8), outer_anchors=s.outer_anchors.astype(np.uint8))
        meta = {"case_id": case_id, "path_id": path_id, "coordinate_system": "LPS", "min_separation_mm": s.min_separation_mm}
        p.with_suffix(".json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
        return p


registry = CaseRegistry()
