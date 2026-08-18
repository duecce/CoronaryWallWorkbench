# CoronaryWallWorkbench

CoronaryWallWorkbench is a research-oriented platform for assisted coronary wall annotation and quantitative coronary analysis from CCTA.

The first development target is a fast, reproducible **inner/outer coronary-wall annotation workflow** built around an already available and validated coronary lumen segmentation. The platform is intended to become the geometric foundation for downstream plaque, stenosis, remodeling, PCAT/FAI, and other coronary analyses.

## Core inputs

A case is defined by three aligned inputs:

- **CCTA image** (`.nii` / `.nii.gz`) containing CT intensities in HU.
- **Coronary lumen mask** (`.nii` / `.nii.gz`) used to initialize the inner wall.
- **Coronary graph** (`.xml`) containing centerline nodes, graph topology, physical coordinates, local radii, and optional anatomical `Labeling` information.

All spatial data are handled in physical coordinates. The loader checks CCTA/mask dimensions and affines, maps XML LPS coordinates into the NIfTI RAS world coordinate convention, and verifies the centerline against the image volume and lumen mask.

## Geometric model

For every selected coronary path, the centerline is parameterized by arc length `s`. A parallel-transport orthonormal frame is propagated along the centerline and used to define cross-sectional planes and longitudinal reformations.

Both vessel boundaries are represented in curvilinear coordinates:

```text
r_inner(s, theta)
r_outer(s, theta)
```

with equi-angular control nodes in each cross-section. The initial inner wall is sampled from the validated lumen mask. The initial outer wall is generated as an offset from the inner wall.

Control nodes are connected circumferentially and longitudinally. Interactive edits propagate smoothly to neighboring nodes with a Gaussian distance-dependent influence kernel. Explicitly edited nodes become anchors and are preserved by subsequent nearby edits.

### Single-active-wall editing

Only **one wall is editable at a time**. The inactive boundary remains visible as a reference contour but its control nodes are hidden and the backend rejects edits addressed to it. This avoids accidental edits where lumen and outer-wall boundaries are close.

A hard geometric invariant is enforced after every edit:

```text
r_outer(s, theta) >= r_inner(s, theta) + min_separation
```

The current default minimum separation is 0.10 mm.

## Implemented workflow

```text
CCTA NIfTI
    +
Lumen mask NIfTI
    +
Coronary graph XML
        |
        v
CaseLoader + spatial QA
        |
        v
Root-to-leaf coronary path reconstruction
        |
        v
Centerline resampling + parallel-transport frames
        |
        v
Cross-sectional CCTA / lumen resampling
        |
        v
Inner-wall radial initialization from lumen mask
        |
        v
Outer-wall offset initialization
        |
        v
Periodic spline wall editor
        |
        v
Local circumferential + longitudinal deformation
```

## Backend API

Run from `backend/`:

```bash
python -m pip install -e .
uvicorn app.main:app --reload
```

Main endpoints:

```text
POST /api/cases/load
POST /api/cases/{case_id}/paths/{path_id}/prepare
GET  /api/cases/{case_id}/paths/{path_id}/sections/{s_index}
POST /api/cases/{case_id}/paths/{path_id}/active-wall
POST /api/cases/{case_id}/paths/{path_id}/edit
```

The current case registry is deliberately in-memory for the MVP. Persistent database-backed case and annotation storage will replace it later.

## Frontend

Run from `frontend/`:

```bash
npm install
npm run dev
```

The current editor provides:

- cross-sectional CCTA display;
- simultaneous inner/outer contour visualization;
- explicit `Edit inner wall` / `Edit outer wall` mode switching;
- control nodes only on the active contour;
- radial drag editing;
- automatic local deformation of neighboring circumferential and longitudinal nodes;
- anchor visualization;
- section-by-section navigation.

## Repository layout

```text
backend/
  app/
    case_loader.py   NIfTI/XML loading and spatial QA
    geometry.py      centerline frames and reformations
    paths.py         coronary graph path reconstruction
    walls.py         radial wall model and editing
    state.py         MVP in-memory case registry
    main.py          FastAPI endpoints
  tests/

frontend/
  src/
    main.tsx         cross-sectional wall editor
    styles.css

docs/
```

## Next development steps

1. synchronized longitudinal/sCPR editor using the same `r(s, theta)` control grid;
2. undo/redo and persistent annotation versioning;
3. branch/path selector driven by XML `Labeling`;
4. export of inner/outer surfaces as NIfTI and VTP;
5. collaborative user/reviewer workflow;
6. downstream plaque and PCAT/FAI modules.

## Status

Early development / research prototype. The software is not a medical device and is not intended for clinical use.
