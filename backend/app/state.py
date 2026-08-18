from __future__ import annotations

from dataclasses import dataclass, field

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

    def prepare_path(
        self,
        case_id: str,
        path_id: str,
        *,
        centerline_step_mm: float = 0.5,
        cross_section_size_mm: float = 10.0,
        cross_section_spacing_mm: float = 0.10,
        n_theta: int = 16,
        outer_offset_mm: float = 0.75,
    ) -> PreparedPath:
        loaded = self.get(case_id)
        if path_id not in loaded.paths:
            raise KeyError(f"Unknown path_id: {path_id}")
        path = loaded.paths[path_id]
        frames = prepare_frames(path, step_mm=centerline_step_mm)
        surface = initialize_wall_surface(
            loaded.case.lumen_mask.data,
            loaded.case.lumen_mask.affine,
            frames,
            n_theta=n_theta,
            outer_offset_mm=outer_offset_mm,
        )
        prepared = PreparedPath(
            path=path,
            frames=frames,
            surface=surface,
            cross_section_size_mm=cross_section_size_mm,
            cross_section_spacing_mm=cross_section_spacing_mm,
        )
        loaded.prepared_paths[path_id] = prepared
        return prepared


registry = CaseRegistry()
