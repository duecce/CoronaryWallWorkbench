from __future__ import annotations

import numpy as np

from .case_loader import graph_points_lps
from .geometry import cumulative_arc_length
from .models import CoronaryPath, PatientCase


def reconstruct_leaf_paths(case: PatientCase) -> dict[str, CoronaryPath]:
    paths: dict[str, CoronaryPath] = {}
    for coronary_name, graph in case.graph.graphs.items():
        children: dict[int, list[int]] = {node_id: [] for node_id in graph.nodes}
        roots: list[int] = []
        for node in graph.nodes.values():
            if node.parent_id is None:
                roots.append(node.node_id)
            elif node.parent_id in children:
                children[node.parent_id].append(node.node_id)
        if not roots:
            roots = [n.node_id for n in graph.nodes.values() if n.is_root]
        if not roots:
            raise ValueError(f"No root node found for {coronary_name}")

        world = graph_points_lps(case, coronary_name)
        leaves = [node_id for node_id, child_ids in children.items() if not child_ids]
        for leaf in leaves:
            chain = [leaf]
            current = leaf
            seen = {leaf}
            while graph.nodes[current].parent_id is not None:
                current = graph.nodes[current].parent_id  # type: ignore[assignment]
                if current in seen or current not in graph.nodes:
                    raise ValueError(f"Invalid/cyclic parent chain in {coronary_name}")
                seen.add(current)
                chain.append(current)
            chain.reverse()
            xyz = np.vstack([world[node_id] for node_id in chain])
            labels = [graph.labels_by_node.get(node_id) for node_id in chain]
            terminal = next((x for x in reversed(labels) if x), f"leaf_{leaf}")
            path_id = f"{coronary_name}_{terminal}_{leaf}"
            paths[path_id] = CoronaryPath(
                path_id=path_id,
                coronary_name=coronary_name,
                node_ids=chain,
                centerline_xyz_mm=xyz,
                arc_length_mm=cumulative_arc_length(xyz),
                anatomical_labels=labels,
            )
    return paths
