# CoronaryWallWorkbench

Research platform for assisted coronary inner/outer-wall annotation and subsequent quantitative CCTA analysis.

## Current workflow

The application supports:

- browser upload of a **CCTA NIfTI**, **validated lumen-mask NIfTI**, and **coronary graph XML**;
- SimpleITK-based image I/O and physical-space processing;
- LPS coordinates throughout the backend;
- spatial QA between CCTA, lumen mask, and XML centerlines;
- root-to-leaf coronary path reconstruction and XML anatomical labels;
- **rotation-minimizing frames (RMF)** computed by the double-reflection method;
- strictly uniform centerline sampling for metric-correct reformations;
- orthogonal axial sCPR cross-sections with a fixed **15 x 15 mm FOV**;
- curved longitudinal sCPR views with the same physical scale on both axes;
- lumen-derived inner-wall initialization;
- offset-based initial outer-wall proposal;
- **Fourier circumferential + cubic B-spline longitudinal wall surfaces**, matching the management strategy of the desktop reference tool;
- 96 internal equi-angular samples with a sparse visible control-node subset in the web UI;
- weighted manual anchors with global surface refitting;
- explicit anchor removal and refitting;
- single-active-wall editing (`inner` or `outer`);
- synchronized cross-sectional and longitudinal editors;
- navigation along the vessel from the longitudinal view;
- hard non-crossing constraint between inner and outer wall;
- cached cross-sectional/longitudinal reformations for responsive editing;
- automatic annotation persistence after every edit.

The primary geometric representation remains:

```text
r_inner(s, theta)
r_outer(s, theta)
```

where `s` is arc length along the coronary path and `theta` is defined by the RMF. The UI edits sparse anchors, while the final boundary is a regularized Fourier-B-spline surface rather than a directly deformed radius matrix.

## Input data

The loading workflow has deliberately not changed. For each case select:

1. `CCTA.nii` or `CCTA.nii.gz`
2. `lumen_mask.nii` or `lumen_mask.nii.gz`
3. coronary graph `.xml`

When the XML contains `Labeling`, anatomical labels are included in the coronary-path selector.

SimpleITK exposes NIfTI geometry in physical LPS coordinates. XML coordinates declared as LPS are therefore used directly; XML coordinates declared as RAS are converted to LPS during loading.

## Geometry guarantees

### Rotation-minimizing frame

The selected coronary path is first resampled to a strictly uniform `s` grid. A deterministic double-reflection RMF is then propagated along the centerline. The initial transverse direction is anchored to the LPS axes (prefer +Z, then +Y, then +X after projection), making the angular convention reproducible across sessions.

This avoids the twisting and frame flips associated with Frenet frames in low-curvature coronary segments.

### Axial sCPR

Default cross-section geometry:

```text
FOV:            15.0 x 15.0 mm
spacing:         0.10 x 0.10 mm
matrix:          151 x 151
```

The output reference image is square in physical space and centered exactly on the centerline. No slice-dependent zoom is applied.

### Longitudinal sCPR

The curved longitudinal image is generated in the same RMF coordinate system. The backend preserves physical sampling (`ds`, `dr`) and the frontend renders the canvas with identical pixels/mm on the longitudinal and radial axes.

The image is never stretched to fill its panel. If necessary, the web panel scrolls instead. Therefore one millimetre in `s` is displayed with the same pixel distance as one millimetre radially.

## Wall-surface model

The web implementation follows the desktop `CCTA_OuterWall_FS` management strategy.

### Inner wall

The validated lumen mask is sampled into `r_inner(s, theta)` and regularized with:

```text
angular samples:                 96
Fourier order:                    6
spectral lambda:              0.015
longitudinal knot spacing:      0.80 mm
longitudinal lambda:             5.0
```

### Outer wall

The initial outer reference is currently initialized from the regularized inner wall plus the configured offset. Manual editing/refitting uses:

```text
angular samples:                 96
Fourier order:                    8
spectral lambda:              0.020
longitudinal knot spacing:      1.00 mm
longitudinal lambda:            12.0
manual anchor weight:           45.0
```

These values match the corresponding defaults in the supplied desktop reference implementation.

### Manual anchors

A web control-point drag does not directly deform neighboring radius samples. Instead it creates or updates a manual anchor. The anchor is injected into the angular candidates with the same local Gaussian angular weighting used by the desktop tool; Fourier coefficients are fitted per cross-section and then represented longitudinally by cubic B-splines.

This gives progressive neighboring displacement while preserving a globally smooth surface.

In the UI:

- drag a visible control node to create/update an anchor;
- magenta markers indicate explicit anchors;
- right-click a magenta anchor to remove it and refit the surface;
- only the active wall exposes editable controls.

If the inner wall is changed, the outer wall is refitted with its own existing anchors preserved while enforcing containment.

## Ubuntu installation

Ubuntu 22.04/24.04 or equivalent is recommended. Conda is also supported and is a good choice for the Python backend.

### Clone

```bash
git clone https://github.com/duecce/CoronaryWallWorkbench.git
cd CoronaryWallWorkbench
```

Because the repository is private, use an authenticated GitHub method.

### Backend with Conda

```bash
conda create -n coronarywall python=3.11 -y
conda activate coronarywall
cd backend
python -m pip install --upgrade pip
pip install -e ".[dev]"
pytest -vv
uvicorn app.main:app --reload --host 127.0.0.1 --port 8000
```

### Frontend

In another terminal:

```bash
cd CoronaryWallWorkbench/frontend
npm install
npm run dev
```

Open the Vite URL, normally:

```text
http://127.0.0.1:5173
```

If the repository is stored on an exFAT/NTFS external drive and `npm install` fails while creating symlinks in `node_modules/.bin`, keep the repository or at least `node_modules` on a native Linux filesystem such as ext4.

## How to use

1. Enter the case ID.
2. Select CCTA NIfTI, lumen-mask NIfTI and coronary XML.
3. Click **Load case**.
4. Check that spatial QA passes.
5. Select the desired coronary path.
6. Click **Prepare path**.
7. Use **Edit inner wall** or **Edit outer wall**.
8. Navigate with the section slider or by clicking the longitudinal view.
9. Select the longitudinal circumferential angle.
10. Drag a control node/active wall to add an anchor.
11. Right-click an existing magenta anchor to remove it.

All successful edits are autosaved.

## Autosave

Annotations are stored under:

```text
~/.coronarywallworkbench/cases/<CASE_ID>/annotations/
```

For each path:

```text
<path>_walls.npz
<path>_walls.json
```

The NPZ now stores:

- `s_mm`, `theta_rad`;
- inner/outer fitted radii;
- inner/outer baseline/reference radii;
- anchor masks and exact anchor radii;
- inner/outer Fourier coefficient fields.

The JSON records the RMF and fitting parameters used for that annotation.

## Default geometry parameters

```text
centerline requested step:               <= 0.50 mm, made strictly uniform
RMF:                                      double-reflection rotation minimizing
axial sCPR FOV:                           15.0 x 15.0 mm
axial sCPR pixel spacing:                  0.10 mm
internal angular samples:                 96
visible web control spacing:              ~15 degrees
outer-wall initial offset:                 0.75 mm
minimum inner/outer separation:            0.10 mm
longitudinal radial extent:               +/- 7.5 mm
longitudinal radial spacing:               0.10 mm
longitudinal rendering aspect:             metric 1:1
```

## Backend API

```text
POST /api/cases/upload
POST /api/cases/{case_id}/paths/{path_id}/prepare
GET  /api/cases/{case_id}/paths/{path_id}/sections/{s_index}
GET  /api/cases/{case_id}/paths/{path_id}/longitudinal/{theta_index}
POST /api/cases/{case_id}/paths/{path_id}/active-wall
POST /api/cases/{case_id}/paths/{path_id}/edit
POST /api/cases/{case_id}/paths/{path_id}/remove-anchor
POST /api/cases/{case_id}/paths/{path_id}/save
```

Interactive documentation:

```text
http://127.0.0.1:8000/docs
```

## Validation priorities

Before using the annotations as a definitive reference dataset, validate on real cases:

- CCTA/XML/lumen alignment;
- RMF orientation continuity and absence of visible twisting;
- axial cross-section centering and fixed 15 mm FOV;
- metric equality of longitudinal axes;
- correspondence of the reconstructed inner surface to the validated lumen mask;
- behavior at bifurcations/ostia;
- sensitivity of the outer wall to Fourier/B-spline parameters and anchor density.

Still planned:

- formal undo/redo history;
- multi-user authentication and reviewer/consensus workflow;
- full volumetric NIfTI and VTP surface export;
- downstream plaque, stenosis, remodeling and PCAT/FAI modules.

The software is an early research prototype, not a medical device.
