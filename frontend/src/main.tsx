import React, { useEffect, useMemo, useRef, useState } from "react";
import { createRoot } from "react-dom/client";
import "./styles.css";

type WallName = "inner" | "outer";
type Point = [number, number];

type WallPayload = {
  radii_mm: number[];
  control_points_mm: Point[];
  contour_mm: Point[];
  anchors: boolean[];
  editable: boolean;
};

type SectionPayload = {
  s_index: number;
  s_mm: number;
  active_wall: WallName;
  theta_rad: number[];
  inner: WallPayload;
  outer: WallPayload;
  image_hu: number[][];
  lumen_mask: number[][];
  size_mm: number;
  spacing_mm: number;
};

const API = "http://127.0.0.1:8000";

function mmToCanvas(p: Point, sizeMm: number, width: number): Point {
  return [width / 2 + (p[0] / sizeMm) * width, width / 2 + (p[1] / sizeMm) * width];
}

function AnnotationCanvas({ data, caseId, pathId, onChanged }: { data: SectionPayload; caseId: string; pathId: string; onChanged: (p: SectionPayload) => void }) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [dragNode, setDragNode] = useState<number | null>(null);
  const width = 640;

  const active = data[data.active_wall];

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;
    const h = data.image_hu.length;
    const w = data.image_hu[0]?.length ?? 0;
    const pixels = ctx.createImageData(width, width);
    for (let y = 0; y < width; y++) {
      for (let x = 0; x < width; x++) {
        const iy = Math.min(h - 1, Math.floor((y / width) * h));
        const ix = Math.min(w - 1, Math.floor((x / width) * w));
        const hu = data.image_hu[iy]?.[ix] ?? -1000;
        const value = Math.max(0, Math.min(255, ((hu + 200) / 1000) * 255));
        const idx = (y * width + x) * 4;
        pixels.data[idx] = value;
        pixels.data[idx + 1] = value;
        pixels.data[idx + 2] = value;
        pixels.data[idx + 3] = 255;
      }
    }
    ctx.putImageData(pixels, 0, 0);

    const drawContour = (points: Point[], stroke: string, lineWidth: number) => {
      if (!points.length) return;
      ctx.beginPath();
      const p0 = mmToCanvas(points[0], data.size_mm, width);
      ctx.moveTo(p0[0], p0[1]);
      for (const p of points.slice(1)) {
        const q = mmToCanvas(p, data.size_mm, width);
        ctx.lineTo(q[0], q[1]);
      }
      ctx.closePath();
      ctx.strokeStyle = stroke;
      ctx.lineWidth = lineWidth;
      ctx.stroke();
    };

    drawContour(data.outer.contour_mm, data.active_wall === "outer" ? "#ffb020" : "#876f3f", data.active_wall === "outer" ? 3 : 1.5);
    drawContour(data.inner.contour_mm, data.active_wall === "inner" ? "#4dc7ff" : "#456878", data.active_wall === "inner" ? 3 : 1.5);

    for (const [idx, p] of active.control_points_mm.entries()) {
      const q = mmToCanvas(p, data.size_mm, width);
      ctx.beginPath();
      ctx.arc(q[0], q[1], active.anchors[idx] ? 6 : 4, 0, Math.PI * 2);
      ctx.fillStyle = data.active_wall === "inner" ? "#4dc7ff" : "#ffb020";
      ctx.fill();
      if (active.anchors[idx]) {
        ctx.strokeStyle = "white";
        ctx.lineWidth = 1;
        ctx.stroke();
      }
    }
  }, [data, active]);

  const closestNode = (x: number, y: number): number | null => {
    let best: number | null = null;
    let bestD = 14;
    active.control_points_mm.forEach((p, i) => {
      const q = mmToCanvas(p, data.size_mm, width);
      const d = Math.hypot(q[0] - x, q[1] - y);
      if (d < bestD) { best = i; bestD = d; }
    });
    return best;
  };

  const pointer = (e: React.PointerEvent<HTMLCanvasElement>): Point => {
    const r = e.currentTarget.getBoundingClientRect();
    return [(e.clientX - r.left) * width / r.width, (e.clientY - r.top) * width / r.height];
  };

  const onPointerDown = (e: React.PointerEvent<HTMLCanvasElement>) => {
    const [x, y] = pointer(e);
    const node = closestNode(x, y);
    if (node !== null) {
      setDragNode(node);
      e.currentTarget.setPointerCapture(e.pointerId);
    }
  };

  const onPointerUp = async (e: React.PointerEvent<HTMLCanvasElement>) => {
    if (dragNode === null) return;
    const [x, y] = pointer(e);
    const center = width / 2;
    const newRadiusPx = Math.hypot(x - center, y - center);
    const newRadiusMm = (newRadiusPx / width) * data.size_mm;
    const oldRadiusMm = active.radii_mm[dragNode];
    const response = await fetch(`${API}/api/cases/${caseId}/paths/${pathId}/edit`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        wall: data.active_wall,
        s_index: data.s_index,
        theta_index: dragNode,
        delta_radius_mm: newRadiusMm - oldRadiusMm,
        sigma_s_mm: 1.5,
        sigma_theta_nodes: 1.25,
        make_anchor: true,
      }),
    });
    if (response.ok) onChanged(await response.json());
    setDragNode(null);
  };

  return <canvas ref={canvasRef} width={width} height={width} onPointerDown={onPointerDown} onPointerUp={onPointerUp} />;
}

function App() {
  const params = useMemo(() => new URLSearchParams(window.location.search), []);
  const [caseId, setCaseId] = useState(params.get("case") ?? "");
  const [pathId, setPathId] = useState(params.get("path") ?? "");
  const [sIndex, setSIndex] = useState(Number(params.get("s") ?? 0));
  const [section, setSection] = useState<SectionPayload | null>(null);
  const [error, setError] = useState("");

  const loadSection = async (index = sIndex) => {
    if (!caseId || !pathId) return;
    const r = await fetch(`${API}/api/cases/${caseId}/paths/${pathId}/sections/${index}`);
    if (!r.ok) { setError(await r.text()); return; }
    setError("");
    setSection(await r.json());
  };

  const chooseWall = async (wall: WallName) => {
    const r = await fetch(`${API}/api/cases/${caseId}/paths/${pathId}/active-wall`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ wall }),
    });
    if (r.ok) await loadSection();
  };

  return <main>
    <header>
      <div>
        <h1>CoronaryWallWorkbench</h1>
        <p>Cross-sectional assisted wall editor</p>
      </div>
      {section && <div className="wall-switch">
        <button className={section.active_wall === "inner" ? "active inner" : ""} onClick={() => chooseWall("inner")}>Edit inner wall</button>
        <button className={section.active_wall === "outer" ? "active outer" : ""} onClick={() => chooseWall("outer")}>Edit outer wall</button>
      </div>}
    </header>

    <section className="toolbar">
      <label>Case <input value={caseId} onChange={e => setCaseId(e.target.value)} /></label>
      <label>Path <input value={pathId} onChange={e => setPathId(e.target.value)} /></label>
      <label>Section <input type="number" min="0" value={sIndex} onChange={e => setSIndex(Number(e.target.value))} /></label>
      <button onClick={() => loadSection()}>Load section</button>
    </section>

    {error && <pre className="error">{error}</pre>}
    {section && <section className="workspace">
      <div className="viewer">
        <AnnotationCanvas data={section} caseId={caseId} pathId={pathId} onChanged={(p) => setSection({ ...section, ...p })} />
      </div>
      <aside>
        <h2>{section.active_wall === "inner" ? "Inner wall" : "Outer wall"}</h2>
        <p>Only the selected wall exposes editable control nodes. The inactive wall remains visible as a reference contour.</p>
        <dl>
          <dt>Position</dt><dd>{section.s_mm.toFixed(2)} mm</dd>
          <dt>Control nodes</dt><dd>{section.theta_rad.length}</dd>
          <dt>Spacing</dt><dd>{section.spacing_mm.toFixed(2)} mm</dd>
        </dl>
        <div className="nav">
          <button disabled={sIndex <= 0} onClick={() => { const n = Math.max(0, sIndex - 1); setSIndex(n); loadSection(n); }}>Previous</button>
          <button onClick={() => { const n = sIndex + 1; setSIndex(n); loadSection(n); }}>Next</button>
        </div>
      </aside>
    </section>}
  </main>;
}

createRoot(document.getElementById("root")!).render(<React.StrictMode><App /></React.StrictMode>);
