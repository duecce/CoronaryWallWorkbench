from __future__ import annotations

from pathlib import Path
import xml.etree.ElementTree as ET

import nibabel as nib
import numpy as np
from scipy.ndimage import distance_transform_edt

from .models import (
    CoronaryGraph,
    CoronarySubgraph,
    GraphNode,
    PatientCase,
    SpatialQA,
    VolumeData,
)


def _load_nifti(path: str, *, binary: bool = False) -> VolumeData:
    p = Path(path)
    image = nib.load(str(p))
    data = np.asanyarray(image.dataobj)
    if data.ndim != 3:
        raise ValueError(f"Expected 3D NIfTI, got shape {data.shape} for {p.name}")
    if binary:
        data = data > 0
    return VolumeData(
        path=p,
        data=np.asarray(data),
        affine=np.asarray(image.affine, dtype=float),
        shape=tuple(int(v) for v in data.shape),
    )


def _parse_graph(xml_path: str) -> CoronaryGraph:
    root = ET.parse(xml_path).getroot()
    reconstruction = root.find(".//Reconstruction")
    if reconstruction is None:
        raise ValueError("XML does not contain a Reconstruction element")
    coordinate_system = reconstruction.attrib.get("coordinateSystem", "UNKNOWN").upper()

    graphs: dict[str, CoronarySubgraph] = {}
    for graph_el in reconstruction.findall("Graph"):
        coronary_name = graph_el.attrib.get("coronaryName", graph_el.attrib.get("id", "unknown"))
        nodes: dict[int, GraphNode] = {}
        for node_el in graph_el.findall("Node"):
            node_id = int(node_el.attrib["id"])
            parent = node_el.attrib.get("parentId")
            nodes[node_id] = GraphNode(
                node_id=node_id,
                xyz_mm=np.array(
                    [float(node_el.attrib["x"]), float(node_el.attrib["y"]), float(node_el.attrib["z"])],
                    dtype=float,
                ),
                parent_id=int(parent) if parent is not None else None,
                degree=int(node_el.attrib.get("degree", "0")),
                is_root=node_el.attrib.get("isRoot", "0") == "1",
                radius_mm=float(node_el.attrib["radiusMm"]) if "radiusMm" in node_el.attrib else None,
                name=node_el.attrib.get("name"),
            )

        labels_by_node: dict[int, str] = {}
        labeling = graph_el.find("Labeling")
        if labeling is not None:
            for label_el in list(labeling):
                label = label_el.attrib.get("markupLabel", label_el.tag)
                for ref in label_el.findall("NodeRef"):
                    labels_by_node[int(ref.attrib["id"])] = label

        graphs[coronary_name] = CoronarySubgraph(
            coronary_name=coronary_name,
            nodes=nodes,
            labels_by_node=labels_by_node,
        )

    if not graphs:
        raise ValueError("No coronary Graph elements found in XML")
    return CoronaryGraph(coordinate_system=coordinate_system, graphs=graphs)


def _to_nifti_world(points_xyz_mm: np.ndarray, coordinate_system: str) -> np.ndarray:
    points = np.asarray(points_xyz_mm, dtype=float).copy()
    if coordinate_system == "LPS":
        points[:, 0] *= -1.0
        points[:, 1] *= -1.0
    elif coordinate_system not in {"RAS", "UNKNOWN"}:
        raise ValueError(f"Unsupported XML coordinate system: {coordinate_system}")
    return points


def _sample_mask_at_world(mask: VolumeData, points_world: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    inv = np.linalg.inv(mask.affine)
    hom = np.c_[points_world, np.ones(len(points_world))]
    ijk_float = (inv @ hom.T).T[:, :3]
    ijk = np.rint(ijk_float).astype(int)
    inside = np.all((ijk >= 0) & (ijk < np.array(mask.shape)), axis=1)
    values = np.zeros(len(points_world), dtype=bool)
    valid = ijk[inside]
    values[inside] = mask.data[valid[:, 0], valid[:, 1], valid[:, 2]] > 0
    return inside, values


def _spatial_qa(ccta: VolumeData, lumen: VolumeData, graph: CoronaryGraph) -> SpatialQA:
    same_shape = ccta.shape == lumen.shape
    affine_diff = float(np.max(np.abs(ccta.affine - lumen.affine)))
    warnings: list[str] = []
    if not same_shape:
        warnings.append("CCTA and lumen-mask shapes differ")
    if affine_diff > 1e-3:
        warnings.append(f"CCTA/lumen affine mismatch: max abs diff {affine_diff:.4f}")

    xml_points = np.vstack([node.xyz_mm for g in graph.graphs.values() for node in g.nodes.values()])
    world_points = _to_nifti_world(xml_points, graph.coordinate_system)
    inside, in_lumen = _sample_mask_at_world(lumen, world_points)
    inside_fraction = float(inside.mean()) if len(inside) else 0.0
    lumen_fraction = float(in_lumen[inside].mean()) if np.any(inside) else 0.0
    if inside_fraction < 0.98:
        warnings.append(f"Only {inside_fraction:.1%} of centerline nodes fall inside the image volume")

    spacing = np.sqrt(np.sum(lumen.affine[:3, :3] ** 2, axis=0))
    distance_mm = distance_transform_edt(~(lumen.data > 0), sampling=spacing)
    inv = np.linalg.inv(lumen.affine)
    hom = np.c_[world_points, np.ones(len(world_points))]
    ijk = np.rint((inv @ hom.T).T[:, :3]).astype(int)
    valid = np.all((ijk >= 0) & (ijk < np.array(lumen.shape)), axis=1)
    distances = np.full(len(ijk), np.nan, dtype=float)
    v = ijk[valid]
    distances[valid] = distance_mm[v[:, 0], v[:, 1], v[:, 2]]
    finite = distances[np.isfinite(distances)]
    median_distance = float(np.median(finite)) if len(finite) else float("inf")
    p95_distance = float(np.percentile(finite, 95)) if len(finite) else float("inf")
    if p95_distance > 2.0:
        warnings.append(f"Centerline-to-lumen p95 distance is {p95_distance:.2f} mm")

    passed = same_shape and affine_diff <= 1e-3 and inside_fraction >= 0.98 and p95_distance <= 2.0
    return SpatialQA(
        passed=passed,
        same_shape=same_shape,
        affine_max_abs_diff_mm=affine_diff,
        centerline_inside_volume_fraction=inside_fraction,
        centerline_inside_lumen_fraction=lumen_fraction,
        median_centerline_to_lumen_mm=median_distance,
        p95_centerline_to_lumen_mm=p95_distance,
        warnings=warnings,
    )


def load_case(
    case_id: str,
    ccta_path: str,
    lumen_mask_path: str,
    xml_path: str,
    require_alignment: bool = True,
) -> PatientCase:
    ccta = _load_nifti(ccta_path)
    lumen = _load_nifti(lumen_mask_path, binary=True)
    graph = _parse_graph(xml_path)
    qa = _spatial_qa(ccta, lumen, graph)
    if require_alignment and not qa.passed:
        raise ValueError("Case failed spatial QA: " + "; ".join(qa.warnings))
    return PatientCase(case_id=case_id, ccta=ccta, lumen_mask=lumen, graph=graph, qa=qa)


def graph_points_nifti_world(case: PatientCase, coronary_name: str) -> dict[int, np.ndarray]:
    graph = case.graph.graphs[coronary_name]
    ids = list(graph.nodes)
    xyz = np.vstack([graph.nodes[node_id].xyz_mm for node_id in ids])
    converted = _to_nifti_world(xyz, case.graph.coordinate_system)
    return {node_id: converted[i] for i, node_id in enumerate(ids)}
