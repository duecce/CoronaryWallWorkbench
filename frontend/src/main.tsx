import React, { useEffect, useRef, useState } from "react";
import { createRoot } from "react-dom/client";
import "./styles.css";

type WallName = "inner" | "outer";
type Point = [number, number];
type PathInfo = {
  path_id: string;
  coronary_name: string;
  length_mm: number;
  labels: string[];
};
type Wall = {
  radii_mm: number[];
  control_points_mm: Point[];
  contour_mm: Point[];
  anchors: boolean[];
  editable: boolean;
};
type Section = {
  s_index: number;
  s_mm: number;
  sample_count: number;
  active_wall: WallName;
  theta_rad: number[];
  control_stride: number;
  frame_type: string;
  inner: Wall;
  outer: Wall;
  image_hu: number[][];
  size_mm: number;
  spacing_mm: number;
};
type Profile = {
  positive_mm: number[];
  negative_mm: number[];
  positive_anchors: boolean[];
  negative_anchors: boolean[];
  opposite_theta_index: number;
};
type Longitudinal = {
  theta_index: number;
  theta_rad: number;
  s_mm: number[];
  radial_mm: number[];
  image_hu: number[][];
  active_wall: WallName;
  longitudinal_spacing_mm: number;
  radial_spacing_mm: number;
  metric_equal: boolean;
  inner: Profile;
  outer: Profile;
};

const API = "http://127.0.0.1:8000";
const CROSS_CANVAS_PX = 660;
const LONG_PIXELS_PER_MM = 9;

function gray(hu: number): number {
  // Coronary CCTA display window: approximately WL 300 / WW 1100.
  const low = -250;
  const high = 850;
  return Math.max(0, Math.min(255, ((hu - low) / (high - low)) * 255));
}

function drawImage(
  ctx: CanvasRenderingContext2D,
  data: number[][],
  width: number,
  height: number,
  flipY = true,
) {
  const imageData = ctx.createImageData(width, height);
  const sourceHeight = data.length;
  const sourceWidth = data[0]?.length || 1;
  for (let y = 0; y < height; y++) {
    const sourceY0 = Math.min(
      sourceHeight - 1,
      Math.floor((y / Math.max(1, height - 1)) * Math.max(0, sourceHeight - 1)),
    );
    const sourceY = flipY ? sourceHeight - 1 - sourceY0 : sourceY0;
    for (let x = 0; x < width; x++) {
      const sourceX = Math.min(
        sourceWidth - 1,
        Math.floor((x / Math.max(1, width - 1)) * Math.max(0, sourceWidth - 1)),
      );
      const value = gray(data[sourceY]?.[sourceX] ?? -1024);
      const index = 4 * (y * width + x);
      imageData.data[index] = value;
      imageData.data[index + 1] = value;
      imageData.data[index + 2] = value;
      imageData.data[index + 3] = 255;
    }
  }
  ctx.putImageData(imageData, 0, 0);
}

function visibleControlIndices(data: Section): number[] {
  const stride = Math.max(1, data.control_stride || 1);
  const result: number[] = [];
  for (let index = 0; index < data.theta_rad.length; index += stride) result.push(index);
  return result;
}

function CrossCanvas({
  data,
  onEdit,
  onRemove,
}: {
  data: Section;
  onEdit: (theta: number, delta: number) => void;
  onRemove: (theta: number) => void;
}) {
  const ref = useRef<HTMLCanvasElement>(null);
  const [drag, setDrag] = useState<number | null>(null);
  const active = data[data.active_wall];
  const controls = visibleControlIndices(data);
  const width = CROSS_CANVAS_PX;

  // Canvas x/y use the same physical scale. Positive v is displayed upward,
  // matching matplotlib origin="lower" in the reference desktop tool.
  const toPx = (point: Point): Point => [
    width / 2 + (point[0] / data.size_mm) * width,
    width / 2 - (point[1] / data.size_mm) * width,
  ];

  useEffect(() => {
    const canvas = ref.current;
    const ctx = canvas?.getContext("2d");
    if (!ctx || !canvas) return;
    drawImage(ctx, data.image_hu, width, width, true);

    const drawContour = (points: Point[], color: string, lineWidth: number) => {
      if (!points.length) return;
      ctx.beginPath();
      let q = toPx(points[0]);
      ctx.moveTo(q[0], q[1]);
      points.slice(1).forEach((point) => {
        q = toPx(point);
        ctx.lineTo(q[0], q[1]);
      });
      ctx.closePath();
      ctx.strokeStyle = color;
      ctx.lineWidth = lineWidth;
      ctx.stroke();
    };

    drawContour(
      data.outer.contour_mm,
      data.active_wall === "outer" ? "#ffd166" : "#79683f",
      data.active_wall === "outer" ? 3 : 1.3,
    );
    drawContour(
      data.inner.contour_mm,
      data.active_wall === "inner" ? "#00d5ff" : "#3a6874",
      data.active_wall === "inner" ? 3 : 1.3,
    );

    controls.forEach((index) => {
      const point = active.control_points_mm[index];
      const q = toPx(point);
      ctx.beginPath();
      ctx.arc(q[0], q[1], active.anchors[index] ? 6 : 4, 0, Math.PI * 2);
      ctx.fillStyle = data.active_wall === "inner" ? "#00d5ff" : "#ffd166";
      ctx.fill();
      if (active.anchors[index]) {
        ctx.strokeStyle = "#ff4fd8";
        ctx.lineWidth = 2;
        ctx.stroke();
      }
    });

    ctx.strokeStyle = "rgba(255,255,255,.22)";
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(width / 2, 0);
    ctx.lineTo(width / 2, width);
    ctx.moveTo(0, width / 2);
    ctx.lineTo(width, width / 2);
    ctx.stroke();
  }, [data]);

  const pointer = (event: React.PointerEvent<HTMLCanvasElement>): Point => {
    const rect = event.currentTarget.getBoundingClientRect();
    return [
      ((event.clientX - rect.left) * width) / rect.width,
      ((event.clientY - rect.top) * width) / rect.height,
    ];
  };

  const nearestControl = (point: Point, anchorsOnly = false): number | null => {
    let best: number | null = null;
    let distance = 16;
    controls.forEach((index) => {
      if (anchorsOnly && !active.anchors[index]) return;
      const q = toPx(active.control_points_mm[index]);
      const candidate = Math.hypot(q[0] - point[0], q[1] - point[1]);
      if (candidate < distance) {
        distance = candidate;
        best = index;
      }
    });
    return best;
  };

  const down = (event: React.PointerEvent<HTMLCanvasElement>) => {
    const index = nearestControl(pointer(event));
    if (index !== null) {
      setDrag(index);
      event.currentTarget.setPointerCapture(event.pointerId);
    }
  };

  const up = (event: React.PointerEvent<HTMLCanvasElement>) => {
    if (drag === null) return;
    const point = pointer(event);
    const radiusPx = Math.hypot(point[0] - width / 2, point[1] - width / 2);
    const radiusMm = (radiusPx / width) * data.size_mm;
    onEdit(drag, radiusMm - active.radii_mm[drag]);
    setDrag(null);
  };

  const context = (event: React.MouseEvent<HTMLCanvasElement>) => {
    event.preventDefault();
    const rect = event.currentTarget.getBoundingClientRect();
    const point: Point = [
      ((event.clientX - rect.left) * width) / rect.width,
      ((event.clientY - rect.top) * width) / rect.height,
    ];
    const index = nearestControl(point, true);
    if (index !== null) onRemove(index);
  };

  return (
    <canvas
      className="crossCanvas"
      ref={ref}
      width={width}
      height={width}
      onPointerDown={down}
      onPointerUp={up}
      onContextMenu={context}
    />
  );
}

function LongCanvas({
  data,
  sIndex,
  onNavigate,
  onEdit,
  onRemove,
}: {
  data: Longitudinal;
  sIndex: number;
  onNavigate: (index: number) => void;
  onEdit: (s: number, theta: number, delta: number) => void;
  onRemove: (s: number, theta: number) => void;
}) {
  const ref = useRef<HTMLCanvasElement>(null);
  const drag = useRef<{ s: number; side: "positive" | "negative" } | null>(null);
  const active = data[data.active_wall];
  const sMin = data.s_mm[0];
  const sMax = data.s_mm[data.s_mm.length - 1];
  const radialMin = data.radial_mm[0];
  const radialMax = data.radial_mm[data.radial_mm.length - 1];
  const sSpan = Math.max(data.longitudinal_spacing_mm, sMax - sMin);
  const radialSpan = radialMax - radialMin;

  // Critical: identical pixels/mm on both axes. Never stretch the sCPR to the
  // panel dimensions; the surrounding container scrolls instead.
  const width = Math.max(2, Math.round(sSpan * LONG_PIXELS_PER_MM) + 1);
  const height = Math.max(2, Math.round(radialSpan * LONG_PIXELS_PER_MM) + 1);
  const x = (index: number) =>
    ((data.s_mm[index] - sMin) / Math.max(1e-9, sSpan)) * (width - 1);
  const y = (radius: number) =>
    (height - 1) -
    ((radius - radialMin) / Math.max(1e-9, radialSpan)) * (height - 1);

  useEffect(() => {
    const canvas = ref.current;
    const ctx = canvas?.getContext("2d");
    if (!ctx || !canvas) return;
    drawImage(ctx, data.image_hu, width, height, true);

    const curve = (values: number[], color: string, lineWidth: number) => {
      ctx.beginPath();
      values.forEach((radius, index) => {
        if (index === 0) ctx.moveTo(x(index), y(radius));
        else ctx.lineTo(x(index), y(radius));
      });
      ctx.strokeStyle = color;
      ctx.lineWidth = lineWidth;
      ctx.stroke();
    };

    curve(data.outer.positive_mm, "#ffd166", data.active_wall === "outer" ? 2.6 : 1.2);
    curve(data.outer.negative_mm, "#ffd166", data.active_wall === "outer" ? 2.6 : 1.2);
    curve(data.inner.positive_mm, "#00d5ff", data.active_wall === "inner" ? 2.6 : 1.2);
    curve(data.inner.negative_mm, "#00d5ff", data.active_wall === "inner" ? 2.6 : 1.2);

    ctx.strokeStyle = "rgba(255,255,255,.3)";
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(0, y(0));
    ctx.lineTo(width, y(0));
    ctx.stroke();

    ctx.strokeStyle = "#ff4f81";
    ctx.beginPath();
    ctx.moveTo(x(sIndex), 0);
    ctx.lineTo(x(sIndex), height);
    ctx.stroke();

    active.positive_mm.forEach((radius, index) => {
      if (!active.positive_anchors[index]) return;
      ctx.beginPath();
      ctx.arc(x(index), y(radius), 3.5, 0, Math.PI * 2);
      ctx.fillStyle = "#ff4fd8";
      ctx.fill();
    });
    active.negative_mm.forEach((radius, index) => {
      if (!active.negative_anchors[index]) return;
      ctx.beginPath();
      ctx.arc(x(index), y(radius), 3.5, 0, Math.PI * 2);
      ctx.fillStyle = "#ff4fd8";
      ctx.fill();
    });
  }, [data, sIndex, width, height]);

  const point = (event: React.PointerEvent<HTMLCanvasElement>): Point => {
    const rect = event.currentTarget.getBoundingClientRect();
    return [
      ((event.clientX - rect.left) * width) / rect.width,
      ((event.clientY - rect.top) * height) / rect.height,
    ];
  };

  const sFromX = (px: number): number => {
    const positionMm = sMin + (px / Math.max(1, width - 1)) * sSpan;
    let best = 0;
    let bestDistance = Infinity;
    data.s_mm.forEach((value, index) => {
      const distance = Math.abs(value - positionMm);
      if (distance < bestDistance) {
        best = index;
        bestDistance = distance;
      }
    });
    return best;
  };

  const down = (event: React.PointerEvent<HTMLCanvasElement>) => {
    const [px, py] = point(event);
    const s = sFromX(px);
    const positiveDistance = Math.abs(py - y(active.positive_mm[s]));
    const negativeDistance = Math.abs(py - y(active.negative_mm[s]));
    if (Math.min(positiveDistance, negativeDistance) < 14) {
      drag.current = {
        s,
        side: positiveDistance < negativeDistance ? "positive" : "negative",
      };
      event.currentTarget.setPointerCapture(event.pointerId);
    } else {
      onNavigate(s);
    }
  };

  const up = (event: React.PointerEvent<HTMLCanvasElement>) => {
    if (!drag.current) return;
    const [, py] = point(event);
    const newRadius =
      radialMin + ((height - 1 - py) / Math.max(1, height - 1)) * radialSpan;
    const { s, side } = drag.current;
    const theta = side === "positive" ? data.theta_index : active.opposite_theta_index;
    const oldRadius = side === "positive" ? active.positive_mm[s] : -active.negative_mm[s];
    const targetRadius = side === "positive" ? newRadius : -newRadius;
    onEdit(s, theta, targetRadius - oldRadius);
    drag.current = null;
  };

  const context = (event: React.MouseEvent<HTMLCanvasElement>) => {
    event.preventDefault();
    const rect = event.currentTarget.getBoundingClientRect();
    const px = ((event.clientX - rect.left) * width) / rect.width;
    const py = ((event.clientY - rect.top) * height) / rect.height;
    const s = sFromX(px);
    const positiveDistance = Math.abs(py - y(active.positive_mm[s]));
    const negativeDistance = Math.abs(py - y(active.negative_mm[s]));
    const positiveAnchor = active.positive_anchors[s];
    const negativeAnchor = active.negative_anchors[s];
    if (positiveAnchor && positiveDistance < 14) onRemove(s, data.theta_index);
    else if (negativeAnchor && negativeDistance < 14)
      onRemove(s, active.opposite_theta_index);
  };

  return (
    <canvas
      className="longCanvas"
      ref={ref}
      width={width}
      height={height}
      style={{ width: `${width}px`, height: `${height}px` }}
      onPointerDown={down}
      onPointerUp={up}
      onContextMenu={context}
    />
  );
}

function App() {
  const [caseId, setCaseId] = useState("case001");
  const [files, setFiles] = useState<{ ccta?: File; lumen?: File; xml?: File }>({});
  const [paths, setPaths] = useState<PathInfo[]>([]);
  const [pathId, setPathId] = useState("");
  const [section, setSection] = useState<Section | null>(null);
  const [longitudinal, setLongitudinal] = useState<Longitudinal | null>(null);
  const [sIndex, setSIndex] = useState(0);
  const [theta, setTheta] = useState(0);
  const [status, setStatus] = useState("Load a case to begin");

  const refresh = async (si = sIndex, ti = theta) => {
    if (!pathId) return;
    const [sectionResponse, longitudinalResponse] = await Promise.all([
      fetch(`${API}/api/cases/${caseId}/paths/${pathId}/sections/${si}`),
      fetch(`${API}/api/cases/${caseId}/paths/${pathId}/longitudinal/${ti}`),
    ]);
    if (!sectionResponse.ok || !longitudinalResponse.ok) {
      setStatus(await (!sectionResponse.ok ? sectionResponse : longitudinalResponse).text());
      return;
    }
    setSection(await sectionResponse.json());
    setLongitudinal(await longitudinalResponse.json());
    setSIndex(si);
    setTheta(ti);
  };

  const upload = async () => {
    if (!files.ccta || !files.lumen || !files.xml) {
      setStatus("Select CCTA, lumen mask and XML");
      return;
    }
    const form = new FormData();
    form.append("case_id", caseId);
    form.append("require_alignment", "true");
    form.append("ccta", files.ccta);
    form.append("lumen_mask", files.lumen);
    form.append("coronary_xml", files.xml);
    setStatus("Loading case and running spatial QA...");
    const response = await fetch(`${API}/api/cases/upload`, { method: "POST", body: form });
    if (!response.ok) {
      setStatus(await response.text());
      return;
    }
    const data = await response.json();
    setPaths(data.paths);
    setPathId(data.paths[0]?.path_id || "");
    setStatus(data.qa.passed ? "Spatial QA passed" : "Loaded with QA warnings");
  };

  const prepare = async () => {
    if (!pathId) return;
    setStatus("Preparing uniform centerline, RMF and Fourier–B-spline walls...");
    const response = await fetch(`${API}/api/cases/${caseId}/paths/${pathId}/prepare`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        centerline_step_mm: 0.5,
        cross_section_size_mm: 15.0,
        cross_section_spacing_mm: 0.1,
        n_theta: 96,
        outer_offset_mm: 0.75,
        longitudinal_radial_extent_mm: 7.5,
        longitudinal_radial_spacing_mm: 0.1,
      }),
    });
    if (!response.ok) {
      setStatus(await response.text());
      return;
    }
    setSIndex(0);
    setTheta(0);
    await refresh(0, 0);
    setStatus("Ready — RMF / metric 1:1 / autosave enabled");
  };

  const chooseWall = async (wall: WallName) => {
    await fetch(`${API}/api/cases/${caseId}/paths/${pathId}/active-wall`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ wall }),
    });
    await refresh();
  };

  const edit = async (si: number, ti: number, delta: number) => {
    const wall = section?.active_wall || "outer";
    setStatus("Refitting Fourier–B-spline surface...");
    const response = await fetch(`${API}/api/cases/${caseId}/paths/${pathId}/edit`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        wall,
        s_index: si,
        theta_index: ti,
        delta_radius_mm: delta,
        sigma_s_mm: 1.5,
        sigma_theta_nodes: 1.25,
        make_anchor: true,
      }),
    });
    if (!response.ok) {
      setStatus(await response.text());
      return;
    }
    await refresh(si, theta);
    setStatus("Anchor fitted and saved");
  };

  const removeAnchor = async (si: number, ti: number) => {
    const wall = section?.active_wall || "outer";
    const response = await fetch(
      `${API}/api/cases/${caseId}/paths/${pathId}/remove-anchor`,
      {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ wall, s_index: si, theta_index: ti }),
      },
    );
    if (!response.ok) {
      setStatus(await response.text());
      return;
    }
    await refresh(si, theta);
    setStatus("Anchor removed and surface refitted");
  };

  const angleOptions = section
    ? section.theta_rad
        .map((value, index) => ({ value, index }))
        .filter(({ index }) => index % Math.max(1, section.control_stride) === 0)
    : [];

  return (
    <main>
      <header>
        <div>
          <h1>CoronaryWallWorkbench</h1>
          <p>RMF-guided Fourier–B-spline inner/outer wall annotation</p>
        </div>
        {section && (
          <div className="wall-switch">
            <button
              className={section.active_wall === "inner" ? "active inner" : ""}
              onClick={() => chooseWall("inner")}
            >
              Edit inner wall
            </button>
            <button
              className={section.active_wall === "outer" ? "active outer" : ""}
              onClick={() => chooseWall("outer")}
            >
              Edit outer wall
            </button>
          </div>
        )}
      </header>

      <section className="loader">
        <input value={caseId} onChange={(event) => setCaseId(event.target.value)} placeholder="Case ID" />
        <label>
          CCTA
          <input type="file" accept=".nii,.gz" onChange={(event) => setFiles({ ...files, ccta: event.target.files?.[0] })} />
        </label>
        <label>
          Lumen
          <input type="file" accept=".nii,.gz" onChange={(event) => setFiles({ ...files, lumen: event.target.files?.[0] })} />
        </label>
        <label>
          XML
          <input type="file" accept=".xml" onChange={(event) => setFiles({ ...files, xml: event.target.files?.[0] })} />
        </label>
        <button onClick={upload}>Load case</button>
        {paths.length > 0 && (
          <>
            <select value={pathId} onChange={(event) => setPathId(event.target.value)}>
              {paths.map((path) => (
                <option key={path.path_id} value={path.path_id}>
                  {path.coronary_name} — {path.labels.join(" / ") || path.path_id} ({path.length_mm.toFixed(1)} mm)
                </option>
              ))}
            </select>
            <button onClick={prepare}>Prepare path</button>
          </>
        )}
        <span className="status">{status}</span>
      </section>

      {section && longitudinal && (
        <>
          <section className="toolbar">
            <label>
              Section
              <input
                type="range"
                min="0"
                max={section.sample_count - 1}
                value={sIndex}
                onChange={(event) => refresh(Number(event.target.value), theta)}
              />
              <b>{section.s_mm.toFixed(1)} mm</b>
            </label>
            <label>
              Longitudinal angle
              <select value={theta} onChange={(event) => refresh(sIndex, Number(event.target.value))}>
                {angleOptions.map(({ value, index }) => (
                  <option key={index} value={index}>
                    {(value * 180 / Math.PI).toFixed(0)}°
                  </option>
                ))}
              </select>
            </label>
            <span className="metric-badge">FOV axial 15×15 mm · RMF · axes 1:1</span>
          </section>

          <section className="workspace">
            <div className="panel cross-panel">
              <h2>Axial sCPR cross-section — fixed 15×15 mm</h2>
              <CrossCanvas
                data={section}
                onEdit={(thetaIndex, delta) => edit(sIndex, thetaIndex, delta)}
                onRemove={(thetaIndex) => removeAnchor(sIndex, thetaIndex)}
              />
              <p>Drag a visible control node to create/update an anchor. Right-click a magenta anchor to remove it.</p>
            </div>
            <div className="panel longitudinal">
              <h2>Longitudinal sCPR — metric scale 1:1</h2>
              <div className="metric-scroll">
                <LongCanvas
                  data={longitudinal}
                  sIndex={sIndex}
                  onNavigate={(index) => refresh(index, theta)}
                  onEdit={edit}
                  onRemove={removeAnchor}
                />
              </div>
              <p>
                The browser never stretches this image: one millimetre along s and one millimetre radially use the same number of pixels. Click to move the axial plane; drag the active wall to add an anchor; right-click an anchor to remove it.
              </p>
            </div>
          </section>
        </>
      )}
    </main>
  );
}

createRoot(document.getElementById("root")!).render(<App />);
