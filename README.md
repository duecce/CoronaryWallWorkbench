# CoronaryWallWorkbench

Research platform for assisted coronary inner/outer-wall annotation and subsequent quantitative CCTA analysis.

## Current workflow

The application supports:

- browser upload of a **CCTA NIfTI**, **validated lumen-mask NIfTI**, and **coronary graph XML**;
- SimpleITK-based image I/O and physical-space processing;
- LPS coordinates throughout the backend;
- spatial QA between CCTA, lumen mask, and XML centerlines;
- root-to-leaf coronary path reconstruction and XML anatomical labels;
- parallel-transport frames along the selected path;
- orthogonal cross-sectional CCTA reformations;
- curved longitudinal/sCPR reformations at selectable circumferential angles;
- lumen-derived inner-wall initialization;
- offset-based initial outer-wall proposal;
- equi-angular periodic-spline control nodes;
- single-active-wall editing (`inner` or `outer`);
- local circumferential and longitudinal propagation of edits;
- synchronized cross-sectional and longitudinal editors;
- navigation along the vessel from the longitudinal view;
- hard non-crossing constraint between inner and outer wall;
- automatic annotation persistence after every edit.

The primary geometric representation is:

```text
r_inner(s, theta)
r_outer(s, theta)
```

where `s` is arc length along the coronary path and `theta` is the angular coordinate defined by the parallel-transport frame.

## Input data

For each case select:

1. `CCTA.nii` or `CCTA.nii.gz`
2. `lumen_mask.nii` or `lumen_mask.nii.gz`
3. coronary graph `.xml`

When the XML contains `Labeling`, anatomical labels are included in the coronary-path selector.

SimpleITK exposes NIfTI geometry in physical LPS coordinates. XML coordinates declared as LPS are therefore used directly; XML coordinates declared as RAS are converted to LPS during loading.

## Ubuntu installation

Ubuntu 22.04/24.04 or equivalent is recommended.

### 1. Install system prerequisites

```bash
sudo apt update
sudo apt install -y git python3 python3-venv python3-pip nodejs npm
```

Check the versions:

```bash
python3 --version
node --version
npm --version
```

The backend requires Python >=3.11. The frontend should be run with a modern Node.js LTS release; if the Ubuntu repository provides an older Node version, install a current LTS release before running `npm install`.

### 2. Clone the repository

```bash
git clone https://github.com/duecce/CoronaryWallWorkbench.git
cd CoronaryWallWorkbench
```

Because the repository is private, use your normal authenticated GitHub method (SSH or HTTPS token).

### 3. Install the backend

```bash
cd backend
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -e .
```

This installs FastAPI, SimpleITK, NumPy, SciPy, Pydantic and multipart upload support.

Start the backend:

```bash
uvicorn app.main:app --reload --host 127.0.0.1 --port 8000
```

Keep this terminal open.

Optional sanity check from another terminal:

```bash
curl http://127.0.0.1:8000/health
```

Expected response:

```json
{"status":"ok"}
```

### 4. Install and start the frontend

In another terminal:

```bash
cd CoronaryWallWorkbench/frontend
npm install
npm run dev
```

Open the URL printed by Vite, normally:

```text
http://127.0.0.1:5173
```

## How to use the application

### Load a case

At the top of the page:

1. enter a case ID, for example `ACTA20`;
2. select the CCTA NIfTI;
3. select the lumen-mask NIfTI;
4. select the coronary XML;
5. click **Load case**.

The backend performs spatial QA before annotation. By default a case that fails strict alignment checks is rejected rather than silently accepted.

### Select and prepare a coronary path

After loading, select one of the reconstructed coronary paths. If XML `Labeling` is present, the anatomical labels are displayed in the selector.

Click **Prepare path**.

The backend then:

1. resamples the centerline at 0.5 mm;
2. constructs parallel-transport frames;
3. initializes the inner wall from the validated lumen mask;
4. initializes the outer wall 0.75 mm outside it;
5. restores an existing saved annotation when its geometry is compatible.

### Edit the walls

Use:

- **Edit inner wall** to expose only the inner-wall control nodes;
- **Edit outer wall** to expose only the outer-wall control nodes.

The inactive wall remains visible but cannot be edited.

#### Cross-sectional view

Drag a control point radially. The displacement is propagated to nearby angular and longitudinal nodes with a Gaussian kernel. The explicitly edited node becomes an anchor.

#### Longitudinal view

Choose the desired circumferential angle from the angle selector.

The longitudinal view displays both sides of the vessel:

- the selected `theta` direction on one side;
- `theta + pi` on the opposite side.

You can:

- click the longitudinal image to move the synchronized cross-section;
- drag the active wall directly in the longitudinal view;
- switch angular direction to inspect and correct other circumferential sectors.

Every longitudinal edit updates the same `r(s, theta)` surface used by the cross-sectional view.

### Autosave

Every successful edit is automatically written under:

```text
~/.coronarywallworkbench/cases/<CASE_ID>/annotations/
```

For each prepared path:

```text
<path>_walls.npz
<path>_walls.json
```

The NPZ stores `s`, `theta`, inner/outer radii and anchor masks. Reloading the same case/path restores the annotation when the stored sampling geometry matches.

## Default geometry parameters

```text
centerline sampling:              0.50 mm
cross-section FOV:               10.0 mm
cross-section pixel spacing:      0.10 mm
angular control nodes:           16
outer-wall initial offset:        0.75 mm
minimum wall separation:          0.10 mm
longitudinal radial extent:       +/- 5.0 mm
longitudinal radial spacing:      0.10 mm
edit propagation sigma (s):       1.5 mm
edit propagation sigma (theta):   1.25 nodes
```

These are MVP defaults and should be refined experimentally for the final annotation protocol.

## Backend API

```text
POST /api/cases/upload
POST /api/cases/{case_id}/paths/{path_id}/prepare
GET  /api/cases/{case_id}/paths/{path_id}/sections/{s_index}
GET  /api/cases/{case_id}/paths/{path_id}/longitudinal/{theta_index}
POST /api/cases/{case_id}/paths/{path_id}/active-wall
POST /api/cases/{case_id}/paths/{path_id}/edit
POST /api/cases/{case_id}/paths/{path_id}/save
```

Interactive FastAPI documentation is available while the backend is running at:

```text
http://127.0.0.1:8000/docs
```

## Current limitations requiring validation

This is an early research prototype. Before treating the resulting annotations as a definitive reference dataset, validate at least:

- CCTA/XML alignment visually on real cases;
- curved SimpleITK longitudinal resampling against known anatomical landmarks;
- agreement between the reconstructed radial inner wall and the original validated lumen mask;
- behavior around bifurcations and severe centerline/mask disagreement;
- the chosen control-node density and deformation-kernel widths.

Still planned:

- undo/redo and formal version history;
- multi-user authentication and reviewer/consensus workflow;
- export to full volumetric inner/outer-wall, plaque and PCAT masks;
- VTP/mesh export;
- downstream plaque, stenosis, remodeling and PCAT/FAI modules.

The software is not a medical device and is not intended for clinical use.
