// feature-maps concept page -- what a CNN "sees" at each depth: the per-channel
// activations (feature maps) of a small fixed CNN, showing edges -> textures ->
// parts. Layer 1 = real oriented Sobel edge filters; layer 2/3 = combine the
// maps below (3x3 conv + ReLU, stride 2) so they grow abstract + downsample.
// Drag (paint) on the input to draw a shape; hover a map to inspect; the
// forward pass reveals layer by layer. cur built only by the transport compute.
import { mount } from '../framework/layout.js';
import { seededRandn } from '../framework/tensor.js';

const INK = '#111', BLUE = '#1f6feb', GREEN = '#2ca02c', GREY = '#9aa4ad';
const N = 16, PX = 3.2;                       // input size, display px per feature-map cell
const EDGE = [                                // Layer-1 oriented edge kernels (Sobel-like)
  [-1, -2, -1, 0, 0, 0, 1, 2, 1],             // horizontal
  [-1, 0, 1, -2, 0, 2, -1, 0, 1],             // vertical
  [0, 1, 2, -1, 0, 1, -2, -1, 0],             // diag /
  [2, 1, 0, 1, 0, -1, 0, -1, -2],             // diag \
];
const L1LBL = ['│ horiz', '─ vert', '╱ diag', '╲ diag'];

let cur = null;
let inRect = null, mapRects = [];   // captured in draw
let painting = false;

function shapeImg(shape, seed) {
  const X = new Float32Array(N * N), c = (N - 1) / 2;
  const rnd = seededRandn(seed, [N, N], { std: 1 }).data;
  for (let y = 0; y < N; y++) for (let x = 0; x < N; x++) {
    const dx = x - c, dy = y - c, r = Math.hypot(dx, dy); let v = 0;
    if (shape === 'circle') v = r < 5.5 ? 1 : 0;
    else if (shape === 'square') v = (Math.abs(dx) < 4.5 && Math.abs(dy) < 4.5) ? 1 : 0;
    else if (shape === 'cross') v = (Math.abs(dx) < 1.6 || Math.abs(dy) < 1.6) ? 1 : 0;
    else if (shape === 'diag') v = (((x + y) % 5) < 2) ? 1 : 0;          // stripes
    else v = rnd[y * N + x] > 0.6 ? 1 : 0;                               // 'random'
    X[y * N + x] = v;
  }
  return X;
}

function conv(maps, Cin, H, W, filt, k, pad, stride, act) {
  const oH = Math.floor((H + 2 * pad - k) / stride) + 1, oW = Math.floor((W + 2 * pad - k) / stride) + 1;
  const out = new Float32Array(oH * oW);
  for (let oy = 0; oy < oH; oy++) for (let ox = 0; ox < oW; ox++) {
    let s = 0;
    for (let c = 0; c < Cin; c++) for (let i = 0; i < k; i++) for (let j = 0; j < k; j++) {
      const iy = oy * stride + i - pad, ix = ox * stride + j - pad;
      if (iy >= 0 && iy < H && ix >= 0 && ix < W) s += filt[(i * k + j) * Cin + c] * maps[c].d[iy * W + ix];
    }
    out[oy * oW + ox] = act === 'abs' ? Math.abs(s) : Math.max(0, s);   // ReLU or |.|
  }
  return { d: out, H: oH, W: oW };
}

function buildData(st) {
  const seed = st.seed | 0;
  cur = {
    X: shapeImg(st.shape, seed), W2: Array.from({ length: 6 }, (_, c) => seededRandn(seed + 10 + c, 3 * 3 * 4, { std: 0.5 })),
    W3: Array.from({ length: 6 }, (_, c) => seededRandn(seed + 30 + c, 3 * 3 * 6, { std: 0.5 })),
  };
  return [{ s: 0, label: 'input image' }, { s: 1, label: 'Layer 1 — edges (Sobel)' }, { s: 2, label: 'Layer 2 — textures' }, { s: 3, label: 'Layer 3 — parts' }];
}

function forward() {
  const X = { d: cur.X, H: N, W: N };
  const L1 = EDGE.map((k) => conv([X], 1, N, N, k, 3, 1, 1, 'abs'));               // [N×N] ×4, edges
  const L2 = cur.W2.map((f) => conv(L1, 4, N, N, f, 3, 1, 2, 'relu'));             // [N/2] ×6, textures
  const h2 = L2[0].H;
  const L3 = cur.W3.map((f) => conv(L2, 6, h2, h2, f, 3, 1, 2, 'relu'));           // [N/4] ×6, parts
  return { L1, L2, L3 };
}

const mmax = (a) => { let m = 1e-6; for (let i = 0; i < a.length; i++) if (a[i] > m) m = a[i]; return m; };

mount({
  mount: 'body',
  title: 'feature-maps — edges → textures → parts',
  blurb: 'What a CNN sees at each depth. A small fixed conv stack runs over the input: Layer 1 uses real oriented edge filters (Sobel) → four maps that light up along edges; Layer 2 combines those edge maps (3×3 conv + ReLU, stride 2) → texture/corner maps at half resolution; Layer 3 combines again → coarse part/blob maps at quarter resolution. Each layer is built from the one below, downsamples, and grows more abstract — early neurons fire on edges, deep neurons on whole parts. Drag (paint) on the input to draw a shape and watch the maps respond; pick a preset; hover a map to see what it detects; step the forward pass.',
  prefer: 'canvas2d',
  aspect: '2 / 1',
  autoplay: true,
  controls: (c, page) => {
    c.select('shape', { label: 'input shape', options: ['circle', 'square', 'cross', 'diag', 'random'], value: 'circle', rebuild: true });
    c.slider('seed', { label: 'seed (filters)', min: 0, max: 99, step: 1, value: 3, rebuild: true });
    c.transport({ compute: () => buildData(page.state), speed: 1.1, loop: true });
  },
  onPointer: (page, ev) => {
    if (!cur || !inRect) return;
    const inPx = inRect.w / N;
    const cellOf = (x, y) => { const c = Math.floor((x - inRect.x) / inPx), r = Math.floor((y - inRect.y) / inPx); return (r >= 0 && r < N && c >= 0 && c < N && x >= inRect.x && y >= inRect.y) ? { r, c } : null; };
    if (ev.type === 'down') { painting = !!cellOf(ev.x, ev.y); if (painting) { const h = cellOf(ev.x, ev.y); cur.X[h.r * N + h.c] = 1; page.redraw(); } }
    else if (ev.type === 'up' || ev.type === 'leave') painting = false;
    else if (ev.type === 'move' && painting && page.pointer.down) { const h = cellOf(ev.x, ev.y); if (h) { cur.X[h.r * N + h.c] = 1; page.redraw(); } }
  },
  draw: (page) => {
    const r = page.renderer, ctx = page.ctx, st = page.state;
    if (!cur) return;
    r.clear('#ffffff');
    const F = forward(), sp = page.step(), reveal = sp ? sp.s : 3;
    mapRects = [];
    const pad = 16;

    // draw one feature map (non-negative) at (x,y), px/cell, normalized to its own max
    const drawMap = (m, x, y, px, col, label, info) => {
      const mx = mmax(m.d);
      for (let yy = 0; yy < m.H; yy++) for (let xx = 0; xx < m.W; xx++) {
        const t = Math.min(1, m.d[yy * m.W + xx] / mx);
        ctx.fillStyle = `rgb(${Math.round(255 - t * col[0])},${Math.round(255 - t * col[1])},${Math.round(255 - t * col[2])})`;
        ctx.fillRect(x + xx * px, y + yy * px, px, px);
      }
      ctx.strokeStyle = '#dfe3e6'; ctx.lineWidth = 1; ctx.strokeRect(x + 0.5, y + 0.5, m.W * px, m.H * px);
      mapRects.push({ x, y, w: m.W * px, h: m.H * px, label, info });
      return m.W * px;
    };

    // input (paintable)
    const inX = pad + 8, inY = 56, inPx = 7;
    inRect = { x: inX, y: inY, w: N * inPx, h: N * inPx };
    r.label('input — drag to paint', inX, inY - 12, { color: INK, font: '11px ui-monospace, monospace' });
    drawMap({ d: cur.X, H: N, W: N }, inX, inY, inPx, [200, 200, 200], 'input', 'the image (drag to paint)');

    // layer rows: L1 edges, L2 textures, L3 parts
    const rowX = inX + N * inPx + 56;
    const rows = [
      { maps: F.L1, lbl: 'Layer 1 — edges', col: [120, 60, 230], px: 4.2, names: L1LBL, show: reveal >= 1 },   // purple
      { maps: F.L2, lbl: 'Layer 2 — textures', col: [210, 105, 30], px: 5.0, names: null, show: reveal >= 2 }, // orange
      { maps: F.L3, lbl: 'Layer 3 — parts', col: [31, 111, 235], px: 7.0, names: null, show: reveal >= 3 },    // blue
    ];
    // stack the three layer rows down the right side
    let yy = inY - 6;
    for (const row of rows) {
      r.label(row.lbl, rowX, yy + 8, { color: row.show ? `rgb(${row.col[0]},${row.col[1]},${row.col[2]})` : GREY, font: '11px ui-monospace, monospace' });
      let cx = rowX;
      for (let c = 0; c < row.maps.length; c++) {
        ctx.save(); if (!row.show) ctx.globalAlpha = 0.18;
        const w = drawMap(row.maps[c], cx, yy + 14, row.px, row.col, `${row.lbl} · ch ${c}`, row.names ? row.names[c] : (row.lbl.includes('texture') ? 'edge combination (corner/pattern)' : 'part/blob detector'));
        if (row.names) { ctx.fillStyle = row.show ? '#586069' : GREY; ctx.font = '8px ui-monospace, monospace'; ctx.textAlign = 'center'; ctx.fillText(row.names[c], cx + w / 2, yy + 14 + N * row.px + 9); }
        ctx.restore();
        cx += w + 14;
      }
      yy += Math.max(N * row.px, 40) + 30;
    }

    // arrow input -> layers
    ctx.save(); ctx.strokeStyle = GREY; ctx.lineWidth = 1.4; ctx.beginPath(); ctx.moveTo(inX + N * inPx + 6, inY + N * inPx / 2); ctx.lineTo(rowX - 8, inY + 6); ctx.stroke(); ctx.restore();

    // hover
    if (page.pointer.over && !painting) {
      const p = page.pointer;
      for (const mr of mapRects) if (p.x >= mr.x && p.x <= mr.x + mr.w && p.y >= mr.y && p.y <= mr.y + mr.h) { page.setTip(`${mr.label}\n${mr.info}`); break; }
    }

    let o = `feature maps: a CNN's per-channel activations get deeper, smaller, and more abstract.  Layer 1 = real Sobel edges → Layer 2 = edge combinations (textures) → Layer 3 = parts.    tier:${r.name}\n`;
    o += sp ? `${sp.label}.` : `edges → textures → parts. Drag the input to paint a shape and watch each layer respond; deeper layers downsample (½, ¼) and abstract.`;
    page.setReadout(o);
  },
}).then((page) => {
  window.__fmPage = page;
  const q = new URLSearchParams(location.search);
  const t = page.controls._transport;
  let reb = false;
  if (q.has('shape')) { page.controls.set('shape', q.get('shape')); reb = true; }
  if (q.has('seed')) { page.controls.set('seed', parseInt(q.get('seed'), 10)); reb = true; }
  if (reb && t) t.rebuild();
  // ?paint=r,c,...  sets input pixels to 1 (headless stand-in for painting)
  if (q.has('paint') && cur) { const ns = q.get('paint').split(';'); for (const pr of ns) { const [rr, cc] = pr.split(',').map(Number); if (rr * N + cc < cur.X.length) cur.X[rr * N + cc] = 1; } }
  if (q.has('step') || q.has('paint') || q.has('hover') || reb) { if (t) t.pause(); }
  if (q.has('step') && t) t.seek(parseInt(q.get('step'), 10));
  if (q.has('hover')) { const [hx, hy] = q.get('hover').split(',').map(Number); page.pointer.x = hx; page.pointer.y = hy; page.pointer.over = true; }
  if (q.get('play') === '1' && t) t.play();
  page.redraw();
});
