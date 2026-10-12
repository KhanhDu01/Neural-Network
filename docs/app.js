"use strict";

/* ------------------------------------------------------------------ *
 * Quick, Draw! neural network playground
 *
 * Runs the exact CNN trained in main/TrainCNN.py, in the browser:
 *   Conv3x3 -> ReLU -> MaxPool2
 *   Conv3x3 -> ReLU -> MaxPool2
 *   Flatten -> Dense -> ReLU -> Dense -> Softmax
 *
 * Weights are loaded from model.bin (raw float32) using the offsets in
 * model.json, which main/export_web.py wrote out.
 * ------------------------------------------------------------------ */

const IMG = 28;
const LOGICAL_H = 260;

const state = {
  meta: null,
  tensors: {},
  img: new Float32Array(IMG * IMG), // 0..1, channel-major (1 channel)
  acts: null,
  strokes: [],       // [[{x,y}, ...], ...] in 0..255 drawing space
  current: null,
  hasDrawing: false,
  lastInfer: 0,
  pulse: 0,
};

/* ============================ model loading ========================= */

async function loadModel() {
  const opts = { cache: "no-store" };
  const meta = await (await fetch("model.json", opts)).json();
  const buf = await (await fetch("model.bin", opts)).arrayBuffer();
  const all = new Float32Array(buf);

  for (const [name, info] of Object.entries(meta.tensors)) {
    state.tensors[name] = all.subarray(info.offset, info.offset + info.length);
  }
  state.meta = meta;

  let h = 2166136261 >>> 0;
  const bytes = new Uint8Array(buf);
  for (let i = 0; i < bytes.length; i++) h = Math.imul(h ^ bytes[i], 16777619) >>> 0;
  const fp = h.toString(16).padStart(8, "0");
  console.log(`model.bin: ${bytes.length} bytes, fnv1a=${fp}`);
  state.modelFingerprint = fp;
}

/* ============================ CNN forward =========================== */

function conv2d(input, C, H, W, weight, bias, F, k, pad) {
  const Ho = H + 2 * pad - k + 1;
  const Wo = W + 2 * pad - k + 1;
  const out = new Float32Array(F * Ho * Wo);
  const kk = k * k;

  for (let f = 0; f < F; f++) {
    const fOut = f * Ho * Wo;
    for (let oy = 0; oy < Ho; oy++) {
      for (let ox = 0; ox < Wo; ox++) {
        let sum = bias[f];
        for (let c = 0; c < C; c++) {
          const cIn = c * H * W;
          const cW = c * kk;
          for (let ki = 0; ki < k; ki++) {
            const iy = oy + ki - pad;
            if (iy < 0 || iy >= H) continue;
            for (let kj = 0; kj < k; kj++) {
              const ix = ox + kj - pad;
              if (ix < 0 || ix >= W) continue;
              sum += input[cIn + iy * W + ix] * weight[(cW + ki * k + kj) * F + f];
            }
          }
        }
        out[fOut + oy * Wo + ox] = sum;
      }
    }
  }
  return { data: out, C: F, H: Ho, W: Wo };
}

function relu_inplace(t) {
  const d = t.data;
  for (let i = 0; i < d.length; i++) if (d[i] < 0) d[i] = 0;
  return t;
}

function maxpool2(t) {
  const { C, H, W } = t;
  const Ho = H >> 1;
  const Wo = W >> 1;
  const out = new Float32Array(C * Ho * Wo);
  for (let c = 0; c < C; c++) {
    const base = c * H * W;
    const obase = c * Ho * Wo;
    for (let oy = 0; oy < Ho; oy++) {
      for (let ox = 0; ox < Wo; ox++) {
        const a = base + (2 * oy) * W + 2 * ox;
        const b = a + 1;
        const cc = a + W;
        const dd = cc + 1;
        let m = t.data[a];
        if (t.data[b] > m) m = t.data[b];
        if (t.data[cc] > m) m = t.data[cc];
        if (t.data[dd] > m) m = t.data[dd];
        out[obase + oy * Wo + ox] = m;
      }
    }
  }
  return { data: out, C, H: Ho, W: Wo };
}

function dense(input, nIn, nOut, weight, bias) {
  const out = new Float32Array(nOut);
  for (let j = 0; j < nOut; j++) {
    let sum = bias[j];
    for (let i = 0; i < nIn; i++) sum += input[i] * weight[i * nOut + j];
    out[j] = sum;
  }
  return out;
}

function softmax(logits) {
  let max = -Infinity;
  for (let i = 0; i < logits.length; i++) if (logits[i] > max) max = logits[i];
  let sum = 0;
  const out = new Float32Array(logits.length);
  for (let i = 0; i < logits.length; i++) {
    const e = Math.exp(logits[i] - max);
    out[i] = e;
    sum += e;
  }
  for (let i = 0; i < logits.length; i++) out[i] /= sum;
  return out;
}

function forward(input) {
  const T = state.tensors;
  const { conv1, conv2, hidden, classes } = state.meta;

  const x = { data: input, C: 1, H: IMG, W: IMG };

  const c1 = relu_inplace(conv2d(x.data, 1, IMG, IMG, T.conv1_w, T.conv1_b, conv1.out, conv1.k, conv1.pad));
  const p1 = maxpool2(c1);

  const c2 = relu_inplace(conv2d(p1.data, conv2.in, p1.H, p1.W, T.conv2_w, T.conv2_b, conv2.out, conv2.k, conv2.pad));
  const p2 = maxpool2(c2);

  const flat = p2.data; // (C,H,W) contiguous == numpy reshape(N,-1)
  const d1v = dense(flat, p2.C * p2.H * p2.W, hidden, T.dense1_w, T.dense1_b);
  const d1 = { data: d1v };
  const relu1 = new Float32Array(d1v.length);
  for (let i = 0; i < d1v.length; i++) relu1[i] = d1v[i] > 0 ? d1v[i] : 0;

  const logits = dense(relu1, hidden, classes, T.dense2_w, T.dense2_b);
  const probs = softmax(logits);

  return { x, c1, p1, c2, p2, flat, d1: relu1, probs };
}

/* ============================= drawing ============================== */

const pad = document.getElementById("pad");
const pctx = pad.getContext("2d");
pctx.imageSmoothingEnabled = true;

function clearPad() {
  pctx.fillStyle = "#000";
  pctx.fillRect(0, 0, pad.width, pad.height);
}

function redrawStrokes(brush, color) {
  clearPad();
  pctx.strokeStyle = color;
  pctx.fillStyle = color;
  pctx.lineWidth = brush;
  pctx.lineCap = "round";
  pctx.lineJoin = "round";
  for (const stroke of state.strokes) {
    if (stroke.length === 1) {
      pctx.beginPath();
      pctx.arc(stroke[0].x, stroke[0].y, brush / 2, 0, Math.PI * 2);
      pctx.fill();
      continue;
    }
    pctx.beginPath();
    pctx.moveTo(stroke[0].x, stroke[0].y);
    for (let i = 1; i < stroke.length; i++) pctx.lineTo(stroke[i].x, stroke[i].y);
    pctx.stroke();
  }
}

const brushInput = document.getElementById("brush");

function padPos(e) {
  const r = pad.getBoundingClientRect();
  return {
    x: Math.min(pad.width, Math.max(0, (e.clientX - r.left) * (pad.width / r.width))),
    y: Math.min(pad.height, Math.max(0, (e.clientY - r.top) * (pad.height / r.height))),
  };
}

let drawing = false;

pad.addEventListener("pointerdown", (e) => {
  e.preventDefault();
  pad.setPointerCapture(e.pointerId);
  drawing = true;
  state.current = [padPos(e)];
  state.strokes.push(state.current);
  redrawStrokes(Number(brushInput.value), "#fff");
});

pad.addEventListener("pointermove", (e) => {
  if (!drawing) return;
  state.current.push(padPos(e));
  redrawStrokes(Number(brushInput.value), "#fff");
});

const VALID_END = () => {
  if (!drawing) return;
  drawing = false;
  state.hasDrawing = true;
  clearTimeout(VALID_END._t);
  VALID_END._t = setTimeout(predict, 60);
};

pad.addEventListener("pointerup", VALID_END);
pad.addEventListener("pointerleave", (e) => { if (drawing) VALID_END(); });
pad.addEventListener("pointercancel", VALID_END);

document.getElementById("clearBtn").addEventListener("click", () => {
  state.strokes = [];
  state.hasDrawing = false;
  state.acts = null;
  clearPad();
  renderPredictions([], 0);
  renderStaticDiagram();
});

/* ======================= image preprocessing ======================== */
/*
 * The model was trained on the Quick, Draw! dataset, whose strokes are
 * normalized so the drawing is anchored at the origin and scaled until its
 * longest side fills the 256px canvas. User drawings are NOT normalized, so we
 * reproduce that here first: without it a hand-drawn doodle scores ~3% instead
 * of ~65%.
 *
 * Then (mirroring main/Layer.py load_drawing): render the strokes on a 256x256
 * canvas with a width-5 white brush and downsample to 28x28. The model input is
 * the resulting grayscale array divided by 255.
 */
function normalizeStrokes() {
  let minX = Infinity, minY = Infinity, maxX = -Infinity, maxY = -Infinity;
  for (const stroke of state.strokes) {
    for (const p of stroke) {
      if (p.x < minX) minX = p.x;
      if (p.x > maxX) maxX = p.x;
      if (p.y < minY) minY = p.y;
      if (p.y > maxY) maxY = p.y;
    }
  }
  if (!isFinite(minX)) return [];
  const w = maxX - minX;
  const h = maxY - minY;
  const scale = 255 / Math.max(w, h, 1e-6);
  return state.strokes.map((stroke) =>
    stroke.map((p) => ({ x: (p.x - minX) * scale, y: (p.y - minY) * scale }))
  );
}

function preprocess() {
  const strokes = normalizeStrokes();
  const c256 = document.createElement("canvas");
  c256.width = c256.height = 256;
  const g = c256.getContext("2d");
  g.fillStyle = "#000";
  g.fillRect(0, 0, 256, 256);
  g.strokeStyle = "#fff";
  g.fillStyle = "#fff";
  g.lineWidth = 5;
  g.lineCap = "round";
  g.lineJoin = "round";

  for (const stroke of strokes) {
    if (stroke.length === 1) {
      g.beginPath();
      g.arc(stroke[0].x, stroke[0].y, 2.5, 0, Math.PI * 2);
      g.fill();
      continue;
    }
    g.beginPath();
    g.moveTo(stroke[0].x, stroke[0].y);
    for (let i = 1; i < stroke.length; i++) g.lineTo(stroke[i].x, stroke[i].y);
    g.stroke();
  }

  const c28 = document.createElement("canvas");
  c28.width = c28.height = 28;
  const h = c28.getContext("2d");
  h.imageSmoothingEnabled = true;
  h.imageSmoothingQuality = "high";
  h.drawImage(c256, 0, 0, 28, 28);

  const data = h.getImageData(0, 0, 28, 28).data;
  const out = state.img;
  for (let i = 0; i < 784; i++) out[i] = data[i * 4] / 255;
  return out;
}

/* ============================ inference ============================= */

function predict() {
  if (!state.meta || !state.hasDrawing) return;
  const t0 = performance.now();
  const input = preprocess();
  const acts = forward(input);
  const dt = performance.now() - t0;

  state.acts = acts;
  state.lastInfer = t0;
  state.pulse = 0;

  const top = topK(acts.probs, 3);
  renderPredictions(top, dt);
  renderStaticDiagram();
}

function topK(probs, k) {
  const idx = [];
  for (let i = 0; i < probs.length; i++) idx.push(i);
  idx.sort((a, b) => probs[b] - probs[a]);
  return idx.slice(0, k).map((i) => ({ i, p: probs[i] }));
}

function renderPredictions(top, dt) {
  const list = document.getElementById("predictions");
  if (!top.length) {
    list.innerHTML = '<li class="placeholder">Start drawing to see predictions&hellip;</li>';
  } else {
    const cats = state.meta.categories;
    list.innerHTML = top
      .map((t, r) => {
        const pct = (t.p * 100).toFixed(1);
        return `<li>
          <div class="pred-row">
            <span class="pred-name"><span class="pred-rank">#${r + 1}</span>${escapeHtml(cats[t.i])}</span>
            <span class="pred-pct">${pct}%</span>
          </div>
          <div class="bar"><span style="width:${Math.max(2, t.p * 100)}%"></span></div>
        </li>`;
      })
      .join("");
  }
  document.getElementById("inferTime").textContent = dt
    ? `${dt.toFixed(0)} ms`
    : "";
}

function escapeHtml(s) {
  return s.replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));
}

/* ======================== network diagram ========================== */

const diagram = document.getElementById("diagram");
const dctx = diagram.getContext("2d");
let base = document.createElement("canvas"); // static layer (maps)
let bctx = base.getContext("2d");
let dpr = Math.min(window.devicePixelRatio || 1, 2);
let cssW = 0;

const COLUMNS = [
  { key: "input", label: "Input", sub: "28\u00d728", w: 0.07 },
  { key: "c1", label: "Conv 3\u00d73", sub: "32\u00d728\u00d728", w: 0.13 },
  { key: "p1", label: "MaxPool", sub: "32\u00d714\u00d714", w: 0.10 },
  { key: "c2", label: "Conv 3\u00d73", sub: "64\u00d714\u00d714", w: 0.13 },
  { key: "p2", label: "MaxPool", sub: "64\u00d77\u00d77", w: 0.095 },
  { key: "flat", label: "Flatten", sub: "3136", w: 0.055 },
  { key: "d1", label: "Dense + ReLU", sub: "256", w: 0.15 },
  { key: "out", label: "Output", sub: "345", w: 0.16 },
];

function layout() {
  const gap = Math.max(8, cssW * 0.012);
  const totalW = cssW - 24; // side padding
  let sum = 0;
  for (const c of COLUMNS) sum += c.w;
  const free = totalW - gap * (COLUMNS.length - 1);
  let x = 12;
  for (const c of COLUMNS) {
    c.px = x;
    c.pw = (c.w / sum) * free;
    c.cx = c.px + c.pw / 2;
    x += c.pw + gap;
  }
}

function resizeDiagram() {
  const wrap = diagram.parentElement;
  cssW = Math.max(680, wrap.clientWidth);
  diagram.style.width = cssW + "px";
  diagram.style.height = LOGICAL_H + "px";
  for (const c of [diagram, base]) {
    c.width = Math.floor(cssW * dpr);
    c.height = Math.floor(LOGICAL_H * dpr);
  }
  dctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  bctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  layout();
  renderStaticDiagram();
}

function heatColor(v) {
  v = Math.max(0, Math.min(1, v));
  // dark navy -> blue -> cyan -> near-white
  const stops = [
    [0.0, 8, 14, 24],
    [0.35, 34, 88, 168],
    [0.7, 60, 190, 210],
    [1.0, 235, 248, 255],
  ];
  for (let i = 1; i < stops.length; i++) {
    if (v <= stops[i][0]) {
      const t = (v - stops[i - 1][0]) / (stops[i][0] - stops[i - 1][0]);
      return [
        Math.round(stops[i - 1][1] + t * (stops[i][1] - stops[i - 1][1])),
        Math.round(stops[i - 1][2] + t * (stops[i][2] - stops[i - 1][2])),
        Math.round(stops[i - 1][3] + t * (stops[i][3] - stops[i - 1][3])),
      ];
    }
  }
  return stops[stops.length - 1].slice(1);
}

// pick up to n channel indices, preferring the most active maps
function pickMaps(t, n) {
  const { C, H, W } = t;
  const scores = new Float32Array(C);
  for (let c = 0; c < C; c++) {
    const b = c * H * W;
    let m = 0;
    for (let i = 0; i < H * W; i++) if (t.data[b + i] > m) m = t.data[b + i];
    scores[c] = m;
  }
  const idx = [];
  for (let c = 0; c < C; c++) idx.push(c);
  idx.sort((a, b) => scores[b] - scores[a]);
  return idx.slice(0, Math.min(n, C));
}

function drawMap(ctx, t, ch, x, y, size, mode) {
  const { H, W } = t;
  const b = ch * H * W;
  const cw = size / W;
  const chh = size / H;
  let max = 1e-6;
  if (mode === "heat") {
    for (let i = 0; i < H * W; i++) if (t.data[b + i] > max) max = t.data[b + i];
  }
  for (let py = 0; py < H; py++) {
    for (let px = 0; px < W; px++) {
      let v = t.data[b + py * W + px];
      let col;
      if (mode === "gray") {
        const g = Math.round(Math.min(1, v) * 255);
        col = `rgb(${g},${g},${g})`;
      } else {
        const rgb = heatColor(v / max);
        col = `rgb(${rgb[0]},${rgb[1]},${rgb[2]})`;
      }
      ctx.fillStyle = col;
      ctx.fillRect(x + px * cw, y + py * chh, Math.ceil(cw), Math.ceil(chh));
    }
  }
}

function drawGrid(ctx, t, x, y, w, h, cols, rows, mode) {
  const maps = pickMaps(t, cols * rows);
  const cellW = w / cols;
  const cellH = h / rows;
  const size = Math.min(cellW, cellH) - 2;
  const padX = (cellW - size) / 2;
  const padY = (cellH - size) / 2;
  maps.forEach((ch, i) => {
    const gx = x + (i % cols) * cellW + padX;
    const gy = y + Math.floor(i / cols) * cellH + padY;
    drawMap(ctx, t, ch, gx, gy, size, mode);
  });
}

function renderStaticDiagram() {
  bctx.clearRect(0, 0, cssW, LOGICAL_H);
  const a = state.acts;
  const areaTop = 46;
  const areaH = LOGICAL_H - areaTop - 34;
  const areaW = (c) => c.pw;

  for (const c of COLUMNS) {
    // title + subtitle
    bctx.fillStyle = "#cdd9e5";
    bctx.font = "600 12px 'Segoe UI', system-ui, sans-serif";
    bctx.textAlign = "center";
    bctx.fillText(c.label, c.cx, 22);
    bctx.fillStyle = "#71808f";
    bctx.font = "10px 'Segoe UI', system-ui, sans-serif";
    bctx.fillText(c.sub, c.cx, 35);

    if (!a) {
      bctx.fillStyle = "#1b2430";
      roundRect(bctx, c.px, areaTop, c.pw, areaH, 8);
      bctx.fill();
      continue;
    }

    const w = areaW(c);
    const h = areaH;

    if (c.key === "input") {
      bctx.fillStyle = "#05070a";
      roundRect(bctx, c.px, areaTop, w, h, 8);
      bctx.fill();
      const size = Math.min(w - 10, h - 10);
      drawMap(bctx, a.x, 0, c.px + (w - size) / 2, areaTop + (h - size) / 2, size, "gray");
    } else if (c.key === "c1") {
      drawGrid(bctx, a.c1, c.px, areaTop, w, h, 4, 3, "heat");
    } else if (c.key === "p1") {
      drawGrid(bctx, a.p1, c.px, areaTop, w, h, 4, 3, "heat");
    } else if (c.key === "c2") {
      drawGrid(bctx, a.c2, c.px, areaTop, w, h, 4, 3, "heat");
    } else if (c.key === "p2") {
      drawGrid(bctx, a.p2, c.px, areaTop, w, h, 4, 3, "heat");
    } else if (c.key === "flat") {
      // vertical heat strip of a sampled subset of the 3136 activations
      const samples = 64;
      const n = a.flat.length;
      let max = 1e-6;
      for (let i = 0; i < n; i++) if (a.flat[i] > max) max = a.flat[i];
      const stripW = Math.min(w * 0.7, 26);
      const sx = c.px + (w - stripW) / 2;
      const sh = h / samples;
      for (let i = 0; i < samples; i++) {
        const v = a.flat[Math.floor((i / samples) * n)] / max;
        const rgb = heatColor(v);
        bctx.fillStyle = `rgb(${rgb[0]},${rgb[1]},${rgb[2]})`;
        bctx.fillRect(sx, areaTop + i * sh, stripW, Math.ceil(sh));
      }
    } else if (c.key === "d1") {
      drawDots(bctx, a.d1, c.px, areaTop, w, h, 16, 16);
    } else if (c.key === "out") {
      drawOutput(bctx, c.px, areaTop, w, h);
    }
  }
}

function drawDots(ctx, vec, x, y, w, h, cols, rows) {
  let max = 1e-6;
  for (let i = 0; i < vec.length; i++) if (vec[i] > max) max = vec[i];
  const cw = w / cols;
  const chh = h / rows;
  const r = Math.min(cw, chh) * 0.36;
  for (let i = 0; i < cols * rows && i < vec.length; i++) {
    const cx = x + (i % cols) * cw + cw / 2;
    const cy = y + Math.floor(i / cols) * chh + chh / 2;
    const rgb = heatColor(vec[i] / max);
    ctx.fillStyle = `rgb(${rgb[0]},${rgb[1]},${rgb[2]})`;
    ctx.beginPath();
    ctx.arc(cx, cy, r, 0, Math.PI * 2);
    ctx.fill();
  }
}

function drawOutput(ctx, x, y, w, h) {
  const cats = state.meta.categories;
  const top = topK(state.acts.probs, 3);
  const rowH = Math.min(46, h / 3);
  ctx.textAlign = "left";
  top.forEach((t, i) => {
    const ry = y + i * (rowH + 6) + 4;
    const bw = w - 8;
    const fill = Math.max(0.04, t.p) * bw;
    ctx.fillStyle = "#0b1219";
    roundRect(ctx, x + 4, ry, bw, rowH, 6);
    ctx.fill();
    const grad = ctx.createLinearGradient(x + 4, 0, x + 4 + bw, 0);
    grad.addColorStop(0, i === 0 ? "#3fb950" : "#4cc2ff");
    grad.addColorStop(1, "#7c5cff");
    ctx.fillStyle = grad;
    roundRect(ctx, x + 4, ry, fill, rowH, 6);
    ctx.fill();
    ctx.fillStyle = "#e6edf3";
    ctx.font = "600 11px 'Segoe UI', system-ui, sans-serif";
    ctx.textAlign = "left";
    ctx.fillText(`${i + 1}. ${cats[t.i]}`, x + 12, ry + rowH / 2 + 4);
    ctx.textAlign = "right";
    ctx.font = "10px 'Segoe UI', system-ui, sans-serif";
    ctx.fillText(`${(t.p * 100).toFixed(1)}%`, x + bw - 8, ry + rowH / 2 + 4);
  });
  ctx.textAlign = "left";
}

function roundRect(ctx, x, y, w, h, r) {
  ctx.beginPath();
  ctx.moveTo(x + r, y);
  ctx.arcTo(x + w, y, x + w, y + h, r);
  ctx.arcTo(x + w, y + h, x, y + h, r);
  ctx.arcTo(x, y + h, x, y, r);
  ctx.arcTo(x, y, x + w, y, r);
  ctx.closePath();
}

/* ======================= animated overlay ========================== */

function animate(now) {
  requestAnimationFrame(animate);
  dctx.clearRect(0, 0, cssW, LOGICAL_H);
  dctx.drawImage(base, 0, 0, cssW, LOGICAL_H);

  // flowing signal pulses along the pipeline
  const cy = 46 + (LOGICAL_H - 46 - 34) / 2;
  const t = now / 1000;
  dctx.lineWidth = 1.5;
  for (let i = 0; i < COLUMNS.length - 1; i++) {
    const a = COLUMNS[i];
    const b = COLUMNS[i + 1];
    const x0 = a.px + a.pw;
    const x1 = b.px;
    dctx.strokeStyle = "rgba(76,194,255,0.18)";
    dctx.beginPath();
    dctx.moveTo(x0, cy);
    dctx.lineTo(x1, cy);
    dctx.stroke();

    // travelling dots
    const nDots = 3;
    for (let d = 0; d < nDots; d++) {
      const phase = ((t * 0.6 + i * 0.18 + d / nDots) % 1);
      const px = x0 + (x1 - x0) * phase;
      const alpha = 0.25 + 0.75 * Math.sin(phase * Math.PI);
      dctx.fillStyle = `rgba(124,92,255,${alpha})`;
      dctx.beginPath();
      dctx.arc(px, cy, 2.6, 0, Math.PI * 2);
      dctx.fill();
    }
  }
}

/* ============================== boot =============================== */

async function boot() {
  clearPad();
  try {
    await loadModel();
    document.getElementById("modelStatus").textContent =
      `${state.meta.categories.length} classes \u00b7 ${(state.meta.totalFloats * 4 / 1048576).toFixed(1)} MB \u00b7 fp:${state.modelFingerprint}`;
  } catch (err) {
    document.getElementById("modelStatus").textContent =
      "Model failed to load (serve over http, not file://)";
    console.error(err);
    return;
  }
  resizeDiagram();
  renderPredictions([], 0);
  requestAnimationFrame(animate);
}

window.addEventListener("resize", () => {
  clearTimeout(window._rt);
  window._rt = setTimeout(resizeDiagram, 120);
});

boot();
