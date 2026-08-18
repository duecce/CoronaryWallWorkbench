# CoronaryWallWorkbench

CoronaryWallWorkbench is a research-oriented platform for assisted coronary wall annotation and quantitative coronary analysis from CCTA.

The first development target is a fast, reproducible **outer-wall annotation workflow** built around an already available and validated coronary lumen segmentation. The platform is intended to become the geometric foundation for downstream plaque, stenosis, remodeling, PCAT/FAI, and other coronary analyses.

## Core inputs

A case is defined by three aligned inputs:

- **CCTA image** (`.nii` / `.nii.gz`) containing the CT intensities in HU.
- **Coronary lumen mask** (`.nii` / `.nii.gz`) used as the fixed inner-wall reference.
- **Coronary graph** (`.xml`) containing centerline nodes, graph topology, physical coordinates, local radii, and optional anatomical `Labeling` information.

All spatial data are handled in physical coordinates. Input loading must explicitly validate the consistency of the CCTA affine, lumen-mask affine, and XML coordinate system before annotation begins.

## Geometric model

For every selected coronary path, the centerline is parameterized by arc length `s`. A stable local orthonormal frame is propagated along the centerline and used to define cross-sectional planes and straightened/curved reformations.

The wall is represented in curvilinear coordinates by a radial function:

```text
r(s, theta)
```

with equi-angular control nodes in each cross-section. Nodes are connected circumferentially and longitudinally. Interactive edits propagate smoothly to neighboring nodes with distance-dependent influence, while preserving explicit user anchors.

The lumen-derived inner wall is treated as immutable during outer-wall annotation. The outer wall is constrained to remain outside the inner wall.

## Initial workflow

```text
CCTA NIfTI
    +
Lumen mask NIfTI
    +
Coronary graph XML
        |
        v
Input validation / alignment QA
        |
        v
Coronary graph + path reconstruction
        |
        v
Parallel-transport frames
        |
        v
Cross-sectional and longitudinal views
        |
        v
Outer-wall initialization
        |
        v
Spline-based interactive correction
        |
        v
Versioned outer-wall surface
        |
        +--> NIfTI mask / mesh export
        +--> plaque analysis
        +--> PCAT / FAI analysis
        +--> stenosis / remodeling analysis
```

## MVP scope

The first milestone will implement:

1. case loading from CCTA NIfTI, lumen NIfTI, and coronary XML;
2. spatial consistency checks;
3. XML graph parsing and optional anatomical labeling;
4. reconstruction of coronary paths;
5. parallel-transport local frames;
6. cross-sectional CCTA/lumen resampling;
7. synchronized longitudinal views;
8. equi-angular outer-wall control nodes and periodic splines;
9. local 2D and longitudinal deformation with neighboring-node propagation;
10. anchor points/slices, undo/redo, and annotation versioning;
11. export of the parametric surface and derived volumetric masks.

## Repository layout

```text
backend/    Python / FastAPI services and geometry processing
frontend/   React + TypeScript annotation interface
docs/       architecture and data-contract documentation
```

## Status

Early development / research prototype. The software is not a medical device and is not intended for clinical use.
