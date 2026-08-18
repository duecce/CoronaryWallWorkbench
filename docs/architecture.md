# CoronaryWallWorkbench architecture

## Design principle

The platform separates immutable source data from editable coronary-wall annotations.

### Immutable case inputs

- CCTA volume in NIfTI format.
- Validated coronary lumen mask in NIfTI format.
- Coronary graph XML containing centerline geometry, topology, and optional anatomical labeling.

### Derived geometry

For each coronary path, the backend derives:

- arc-length parameterized centerline `C(s)`;
- a stable local orthonormal frame propagated along the path;
- cross-sectional sampling planes;
- longitudinal reformations;
- lumen-derived inner-wall radial representation `r_inner(s, theta)`.

### Editable geometry

The outer wall is stored as a parametric radial surface `r_outer(s, theta)` rather than as a primary voxel mask.

Control nodes are equi-angular in each cross-section and topologically connected to corresponding nodes in neighboring longitudinal positions. Interactive deformation propagates to neighboring nodes through a distance-dependent kernel, while explicit anchors constrain the solution.

## Coordinate handling

The XML graph is expected to encode physical coordinates, commonly in LPS millimetres. NIfTI files provide voxel-to-world transforms through their affine matrices.

The loader must never assume direct equivalence between XML coordinates and voxel indices. A case is accepted for annotation only after spatial consistency checks between:

1. CCTA affine and dimensions;
2. lumen-mask affine and dimensions;
3. XML centerline physical coordinates.

QA should include geometric checks and a visual overlay of the centerline and lumen mask on the CCTA.

## Core domain objects

```text
PatientCase
├── ccta
├── lumen_mask
├── coronary_graph
└── paths[]

CoronaryGraph
├── graphs[LCA, RCA, ...]
├── nodes
├── edges
├── ostia
└── anatomical_labels

CoronaryPath
├── path_id
├── node_ids
├── centerline_xyz
├── arc_length_mm
├── local_frames
├── anatomical_labels
├── inner_surface
└── outer_surface

WallSurface
├── s_mm[]
├── theta_rad[]
├── radii_mm[][]
├── anchors
└── annotation_metadata
```

## Annotation invariants

The geometry engine must enforce:

- `r_outer(s, theta) >= r_inner(s, theta) + epsilon`;
- periodic continuity around `theta`;
- longitudinal correspondence of angular nodes;
- smooth local propagation during editing;
- preservation of explicit anchors;
- reproducible conversion from parametric surface to physical 3D coordinates.

## Downstream analyses

The common curvilinear coordinate system is intentionally reusable for:

- total and component plaque volume;
- plaque burden and eccentricity;
- stenosis and minimum lumen area;
- vessel remodeling;
- calcification analysis;
- PCAT segmentation;
- FAI and radial attenuation profiles;
- longitudinal and circumferential coronary maps.

These analyses are downstream modules and are not part of the first outer-wall annotation MVP.
