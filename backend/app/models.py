from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import numpy as np


@dataclass(slots=True)
class VolumeData:
    path: Path
    data: np.ndarray
    affine: np.ndarray
    shape: tuple[int, int, int]


@dataclass(slots=True)
class GraphNode:
    node_id: int
    xyz_mm: np.ndarray
    parent_id: int | None
    degree: int
    is_root: bool
    radius_mm: float | None = None
    name: str | None = None


@dataclass(slots=True)
class CoronarySubgraph:
    coronary_name: str
    nodes: dict[int, GraphNode]
    labels_by_node: dict[int, str] = field(default_factory=dict)


@dataclass(slots=True)
class CoronaryGraph:
    coordinate_system: str
    graphs: dict[str, CoronarySubgraph]


@dataclass(slots=True)
class SpatialQA:
    passed: bool
    same_shape: bool
    affine_max_abs_diff_mm: float
    centerline_inside_volume_fraction: float
    centerline_inside_lumen_fraction: float
    median_centerline_to_lumen_mm: float
    p95_centerline_to_lumen_mm: float
    warnings: list[str] = field(default_factory=list)


@dataclass(slots=True)
class CoronaryPath:
    path_id: str
    coronary_name: str
    node_ids: list[int]
    centerline_xyz_mm: np.ndarray
    arc_length_mm: np.ndarray
    anatomical_labels: list[str | None]


@dataclass(slots=True)
class PatientCase:
    case_id: str
    ccta: VolumeData
    lumen_mask: VolumeData
    graph: CoronaryGraph
    qa: SpatialQA


@dataclass(slots=True)
class ParallelTransportFrames:
    centerline_xyz_mm: np.ndarray
    arc_length_mm: np.ndarray
    tangent: np.ndarray
    normal_u: np.ndarray
    normal_v: np.ndarray


@dataclass(slots=True)
class WallSurface:
    s_mm: np.ndarray
    theta_rad: np.ndarray
    inner_radii_mm: np.ndarray
    outer_radii_mm: np.ndarray
    inner_anchors: np.ndarray
    outer_anchors: np.ndarray
    active_wall: Literal["inner", "outer"] = "outer"
    min_separation_mm: float = 0.10


@dataclass(slots=True)
class PreparedPath:
    path: CoronaryPath
    frames: ParallelTransportFrames
    surface: WallSurface
    cross_section_size_mm: float
    cross_section_spacing_mm: float
