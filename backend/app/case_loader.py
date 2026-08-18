from __future__ import annotations

from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np
import SimpleITK as sitk

from .models import (
    CoronaryGraph,
    CoronarySubgraph,
    GraphNode,
    PatientCase,
    SpatialQA,
    VolumeData,
)


def _load_nifti(path: str, *, binary: bool = False) -> VolumeData:
    p = Path(path).expanduser().resolve()
    if not p.exists():
        raise FileNotFoundError(str(p))
    image = sitk.ReadImage(str(p))
    if image.GetDimension() != 3:
        raise ValueError(f"Expected a 3D NIfTI image, got dimension {image.GetDimension()} for {p.name}")
    if binary:
        image = sitk.Cast(image > 0, sitk.sitkUInt8)
    return VolumeData(
        path=p,
        image=image,
        shape=tuple(int(v) for v in image.GetSize()),
        spacing_mm=tuple(float(v) for v in image.GetSpacing()),
        origin_mm=tuple(float(v) for v in image.GetOrigin()),
        direction=tuple(float(v) for v in image.GetDirection()),
    )


def _parse_graph(xml_path: str) -> CoronaryGraph:
    root = ET.parse(xml_path).getroot()
    reconstruction = root.find(".//Reconstruction")
    if reconstruction is None:
        raise ValueError("XML does not contain a Reconstruction element")
    coordinate_system = reconstruction.attrib.get("coordinateSystem", "LPS").upper()

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


def _to_lps(points_xyz_mm: np.ndarray, coordinate_system: str) -> np.ndarray:
    points = np.asarray(points_xyz_mm, dtype=float).copy()
    if coordinate_system == "RAS":
        points[:, 0] *= -1.0
        points[:, 1] *= -1.0
    elif coordinate_system not in {"LPS", "UNKNOWN"}:
        raise ValueError(f"Unsupported XML coordinate system: {coordinate_system}")
    return points


def _same_geometry(a: sitk.Image, b: sitk.Image) -> tuple[bool, float]:
    size_ok = tuple(a.GetSize()) == tuple(b.GetSize())
    values_a = np.r_[a.GetSpacing(), a.GetOrigin(), a.GetDirection()]
    values_b = np.r_[b.GetSpacing(), b.GetOrigin(), b.GetDirection()]
    diff = float(np.max(np.abs(values_a - values_b)))
    return size_ok, diff


def _physical_point_inside(image: sitk.Image, point: np.ndarray) -> tuple[bool, tuple[int, int, int] | None]:
    try:
        idx = image.TransformPhysicalPointToIndex(tuple(float(v) for v in point))
    except RuntimeError:
        return False, None
    inside = all(0 <= idx[d] < image.GetSize()[d] for d in range(3))
    return inside, tuple(int(v) for v in idx) if inside else None


def _sample_points(image: sitk.Image, points_lps: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    inside = np.zeros(len(points_lps), dtype=bool)
    values = np.zeros(len(points_lps), dtype=float)
    for i, point in enumerate(points_lps):
        ok, idx = _physical_point_inside(image, point)
        inside[i] = ok
        if ok and idx is not None:
            values[i] = float(image[idx])
    return inside, values


def _spatial_qa(ccta: VolumeData, lumen: VolumeData, graph: CoronaryGraph) -> SpatialQA:
    same_shape, geometry_diff = _same_geometry(ccta.image, lumen.image)
    warnings: list[str] = []
    if not same_shape:
        warnings.append("CCTA and lumen-mask sizes differ")
    if geometry_diff > 1e-4:
        warnings.append(f"CCTA/lumen physical-geometry mismatch: max abs diff {geometry_diff:.6f}")

    xml_points = np.vstack([node.xyz_mm for g in graph.graphs.values() for node in g.nodes.values()])
    lps_points = _to_lps(xml_points, graph.coordinate_system)
    inside, lumen_values = _sample_points(lumen.image, lps_points)
    inside_fraction = float(inside.mean()) if len(inside) else 0.0
    lumen_fraction = float((lumen_values[inside] > 0).mean()) if np.any(inside) else 0.0
    if inside_fraction < 0.98:
        warnings.append(f"Only {inside_fraction:.1%} of centerline nodes fall inside the image volume")

    distance = sitk.SignedMaurerDistanceMap(
        sitk.Cast(lumen.image > 0, sitk.sitkUInt8),
        insideIsPositive=False,
        squaredDistance=False,
        useImageSpacing=True,
    )
    valid_distances: list[float] = []
    for point in lps_points:
        ok, idx = _physical_point_inside(distance, point)
        if ok and idx is not None:
            valid_distances.append(max(0.0, float(distance[idx])))
    finite = np.asarray(valid_distances, dtype=float)
    median_distance = float(np.median(finite)) if len(finite) else float("inf")
    p95_distance = float(np.percentile(finite, 95)) if len(finite) else float("inf")
    if p95_distance > 2.0:
        warnings.append(f"Centerline-to-lumen p95 distance is {p95_distance:.2f} mm")

    passed = same_shape and geometry_diff <= 1e-4 and inside_fraction >= 0.98 and p95_distance <= 2.0
    return SpatialQA(
        passed=passed,
        same_shape=same_shape,
        geometry_max_abs_diff_mm=geometry_diff,
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
    case_dir: str | None = None,
) -> PatientCase:
    ccta = _load_nifti(ccta_path)
    lumen = _load_nifti(lumen_mask_path, binary=True)
    graph = _parse_graph(xml_path)
    qa = _spatial_qa(ccta, lumen, graph)
    if require_alignment and not qa.passed:
        raise ValueError("Case failed spatial QA: " + "; ".join(qa.warnings))
    output_dir = Path(case_dir).expanduser().resolve() if case_dir else Path(ccta_path).expanduser().resolve().parent / f"{case_id}_coronarywall"
    output_dir.mkdir(parents=True, exist_ok=True)
    return PatientCase(
        case_id=case_id,
        ccta=ccta,
        lumen_mask=lumen,
        graph=graph,
        qa=qa,
        case_dir=output_dir,
    )


def graph_points_lps(case: PatientCase, coronary_name: str) -> dict[int, np.ndarray]:
    graph = case.graph.graphs[coronary_name]
    ids = list(graph.nodes)
    xyz = np.vstack([graph.nodes[node_id].xyz_mm for node_id in ids])
    converted = _to_lps(xyz, case.graph.coordinate_system)
    return {node_id: converted[i] for i, node_id in enumerate(ids)}
