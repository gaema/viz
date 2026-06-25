// pooling concept page -- max / average pooling and downsampling. A k×k window
// steps over the n×n input with stride s (s=k tiles without overlap); each
// window -> one output cell: out = max(window) (argmax highlighted) or
// mean(window). Output size = floor((n-k)/s)+1, so a 2×2/stride-2 pool halves
// each dimension. Drag an input cell to change it; hover to inspect. cur is
// built ONLY by the transport compute (no draw-side rebuild).
import { mount } from '../framework/layout.js';
import { seededRandn } from '../framework/tensor.js';

const INK = '#111', BLUE = '#1f6feb', RED = '#d1242f', GREEN = '#2ca02c', GREY = '#9aa4ad';
const divcol = (v, dom) => { const t = Math.max(-1, Math.min(1, v / (dom || 1))), m = Math.abs(t); return t >= 0 ? `rgb(255,${Math.round(255 - m * 150)},${Math.round(255 - m * 165)})` : `rgb(${Math.round(255 - m * 165)},${Math.round(255 - m * 120)},255)`; };
const maxAbs = (a) => { let m = 1e-9; for (let i = 0; i < a.length; i++) if (Math.abs(a[i]) > m) m = Math.abs(a[i]); return m; };

let cur = null;
let inRect = null, outRect = null, ic = 26, oc = 30;   // captured in draw
let grab = null;                                       // {r,c} input cell while dragging

function buildData(st) {
  const n = st.n | 0, seed = st.seed | 0;
  cur = { X: seededRandn(seed, [n, n], { std: 1 }).data, n };
  const k = st.k | 0, s = st.stride | 0;
  const out = Math.max(1, Math.floor((n - k) / s) + 1);
  const steps = [];
  for (let oy = 0; oy < out; oy++) for (let ox = 0; ox < out; ox++) steps.push({ oy, ox, label: `window (${oy},${ox}) → ${st.pool}` });
  return steps;
}

function pool(st) {
  const { X, n } = cur, k = st.k | 0, s = st.stride | 0, type = st.pool;
  const out = Math.max(1, Math.floor((n - k) / s) + 1);
  const O = new Float32Array(out * out), arg = new Int32Array(out * out);
  for (let oy = 0; oy < out; oy++) for (let ox = 0; ox < out; ox++) {
    let mx = -Infinity, mi = -1, sum = 0;
    for (let i = 0; i < k; i++) for (let j = 0; j < k; j++) { const idx = (oy * s + i) * n + (ox * s + j), v = X[idx]; sum += v; if (v > mx) { mx = v; mi = idx; } }
    O[oy * out + ox] = type === 'avg' ? sum / (k * k) : mx; arg[oy * out + ox] = mi;
  }
  return { O, arg, out, k, s };
}

mount({
  mount: 'body',
  title: 'pooling — max / avg, downsampling',
  blurb: 'Pooling is the cheap, parameter-free downsampler in a CNN: a k×k window steps over the input (stride s, usually s=k so windows tile without overlap) and reduces each window to one number — its max (max pooling keeps the strongest activation, the argmax cell highlighted) or its mean (avg pooling smooths). Output size = ⌊(n−k)/s⌋+1, so a 2×2/stride-2 pool halves each spatial dimension. Drag an input cell to change it (for max-pool the output only moves if you change the window’s max); hover to inspect; toggle max↔avg to compare.',
  prefer: 'canvas2d',
  aspect: '2 / 1',
  autoplay: true,
  challenges: [
    { goal: 'Pool harder — shrink the output to 2×2 or smaller.', hint: 'raise the stride s (or shrink the input n).', check: (api) => ({ solved: (api.probe.out ?? 9) <= 2, detail: `output ${api.probe.out ?? '?'}×${api.probe.out ?? '?'} (need ≤ 2×2)` }) },
    { goal: 'Switch to AVERAGE pooling (smooth, not the max).', hint: 'set the pooling control to "avg".', check: (api) => ({ solved: api.state.pool === 'avg', detail: `pooling = ${api.state.pool}` }) },
  ],
  controls: (c, page) => {
    c.stepper('n', { label: 'input size (n×n)', min: 4, max: 8, value: 6 });
    c.stepper('k', { label: 'window (k)', min: 2, max: 3, value: 2 });
    c.stepper('stride', { label: 'stride s', min: 1, max: 3, value: 2 });
    c.select('pool', { label: 'pooling', options: ['max', 'avg'], value: 'max' });
    c.slider('seed', { label: 'seed', min: 0, max: 99, step: 1, value: 7, rebuild: true });
    c.transport({ compute: () => buildData(page.state), speed: 2, loop: true });
  },
  onPointer: (page, ev) => {
    if (!cur || !inRect) return;
    const n = cur.n;
    const cellOf = (x, y) => { const c = Math.floor((x - inRect.x) / ic), r = Math.floor((y - inRect.y) / ic); return (r >= 0 && r < n && c >= 0 && c < n && x >= inRect.x && y >= inRect.y) ? { r, c } : null; };
    if (ev.type === 'down') grab = cellOf(ev.x, ev.y);
    else if (ev.type === 'up' || ev.type === 'leave') grab = null;
    else if (ev.type === 'move' && grab && page.pointer.down) { const i = grab.r * n + grab.c; cur.X[i] = Math.max(-3, Math.min(3, cur.X[i] - ev.dy * 0.02)); page.redraw(); }
  },
  draw: (page) => {
    const r = page.renderer, ctx = page.ctx, st = page.state;
    if (!cur) return;
    const { X, n } = cur;
    r.clear('#ffffff');
    const Pl = pool(st), { O, arg, out, k, s } = Pl, type = st.pool;
    page.probe = { out, n: cur.n, pool: type };
    const sp = page.step(), oy = sp ? sp.oy : out - 1, ox = sp ? sp.ox : out - 1, ostep = sp ? oy * out + ox : out * out - 1;
    const xdom = Math.max(maxAbs(X), 0.5), odom = Math.max(maxAbs(O), 0.5);

    const pad = 16, topY = 64;
    ic = Math.max(18, Math.min(40, Math.min((page.W * 0.34) / n, (page.H * 0.56) / n)));
    inRect = { x: pad + 24, y: topY, w: n * ic, h: n * ic };
    r.label(`input  ${n}×${n}  — drag a cell ↕`, inRect.x, topY - 12, { color: INK, font: '11px ui-monospace, monospace' });

    // input grid
    ctx.save(); ctx.font = `${Math.max(9, ic * 0.34)}px ui-monospace, monospace`; ctx.textAlign = 'center'; ctx.textBaseline = 'middle';
    for (let rr = 0; rr < n; rr++) for (let cc = 0; cc < n; cc++) { const x = inRect.x + cc * ic, y = inRect.y + rr * ic, v = X[rr * n + cc]; ctx.fillStyle = divcol(v, xdom); ctx.fillRect(x, y, ic - 1, ic - 1); ctx.fillStyle = '#33383d'; ctx.fillText(v.toFixed(1), x + ic / 2, y + ic / 2); }
    // faint tiling guides (every stride, when non-overlap)
    if (s === k) { ctx.strokeStyle = 'rgba(0,0,0,0.18)'; ctx.lineWidth = 1; for (let g = 0; g <= out; g++) { const p2 = g * s * ic; ctx.beginPath(); ctx.moveTo(inRect.x + p2, inRect.y); ctx.lineTo(inRect.x + p2, inRect.y + Math.min(out * s, n) * ic); ctx.moveTo(inRect.x, inRect.y + p2); ctx.lineTo(inRect.x + Math.min(out * s, n) * ic, inRect.y + p2); ctx.stroke(); } }
    // current window
    const wx = inRect.x + ox * s * ic, wy = inRect.y + oy * s * ic;
    ctx.strokeStyle = BLUE; ctx.lineWidth = 2.6; ctx.strokeRect(wx + 1, wy + 1, k * ic - 2, k * ic - 2);
    // for max-pool, mark the argmax cell
    if (type === 'max') { const mi = arg[oy * out + ox], mr = Math.floor(mi / n), mc = mi % n; ctx.strokeStyle = RED; ctx.lineWidth = 2.4; ctx.strokeRect(inRect.x + mc * ic + 2, inRect.y + mr * ic + 2, ic - 4, ic - 4); ctx.fillStyle = RED; ctx.font = '8px ui-monospace, monospace'; ctx.textAlign = 'left'; ctx.textBaseline = 'top'; ctx.fillText('max', inRect.x + mc * ic + 2, inRect.y + mr * ic + 1); }
    ctx.restore();

    // output grid (downsampled)
    oc = Math.max(22, Math.min(48, Math.min((page.W * 0.24) / out, (page.H * 0.5) / out)));
    const outX = inRect.x + n * ic + 90, outY = topY + (n * ic - out * oc) / 2;
    outRect = { x: outX, y: outY, w: out * oc, h: out * oc };
    r.label(`output  ${out}×${out}  (${type})`, outX, outY - 12, { color: GREEN, font: '11px ui-monospace, monospace' });
    ctx.save(); ctx.font = `${Math.max(9, oc * 0.32)}px ui-monospace, monospace`; ctx.textAlign = 'center'; ctx.textBaseline = 'middle';
    for (let yy = 0; yy < out; yy++) for (let xx = 0; xx < out; xx++) { const x = outX + xx * oc, y = outY + yy * oc, idx = yy * out + xx, done = idx <= ostep; ctx.fillStyle = done ? divcol(O[idx], odom) : '#f4f5f6'; ctx.fillRect(x, y, oc - 1, oc - 1); ctx.strokeStyle = '#e0e3e6'; ctx.strokeRect(x + 0.5, y + 0.5, oc - 1, oc - 1); if (done) { ctx.fillStyle = '#33383d'; ctx.fillText(O[idx].toFixed(1), x + oc / 2, y + oc / 2); } if (yy === oy && xx === ox) { ctx.strokeStyle = BLUE; ctx.lineWidth = 2.6; ctx.strokeRect(x + 1, y + 1, oc - 2, oc - 2); } }
    ctx.restore();
    // arrow window -> output
    ctx.save(); ctx.strokeStyle = GREY; ctx.lineWidth = 1.4; ctx.beginPath(); ctx.moveTo(wx + k * ic, wy + k * ic / 2); ctx.lineTo(outX + ox * oc + oc / 2, outY + oy * oc); ctx.stroke(); ctx.fillStyle = GREY; ctx.font = '10px ui-monospace, monospace'; ctx.textAlign = 'center'; ctx.fillText(type === 'max' ? 'max' : 'avg', (wx + k * ic + outX) / 2, wy - 4); ctx.restore();

    // readout for current window
    const wvals = []; for (let i = 0; i < k; i++) for (let j = 0; j < k; j++) wvals.push(X[(oy * s + i) * n + (ox * s + j)]);
    const res = O[oy * out + ox];
    r.label(type === 'max'
      ? `out[${oy},${ox}] = max(${wvals.map((v) => v.toFixed(1)).join(', ')}) = ${res.toFixed(1)}  (the strongest activation; the rest are dropped)`
      : `out[${oy},${ox}] = mean(${wvals.map((v) => v.toFixed(1)).join(', ')}) = ${res.toFixed(2)}  (the window averaged)`,
      inRect.x, inRect.y + n * ic + 24, { color: INK, font: '12px ui-monospace, monospace' });

    // hover
    if (page.pointer.over && !grab) {
      const pt = page.pointer;
      const icc = Math.floor((pt.x - inRect.x) / ic), irr = Math.floor((pt.y - inRect.y) / ic);
      if (irr >= 0 && irr < n && icc >= 0 && icc < n && pt.x >= inRect.x && pt.y >= inRect.y) page.setTip(`X[${irr},${icc}] = ${X[irr * n + icc].toFixed(3)}\ndrag ↕ to change`);
      else { const oxx = Math.floor((pt.x - outX) / oc), oyy = Math.floor((pt.y - outY) / oc); if (oyy >= 0 && oyy < out && oxx >= 0 && oxx < out && pt.x >= outX && pt.y >= outY) page.setTip(`out[${oyy},${oxx}] = ${O[oyy * out + oxx].toFixed(3)}\n${type} of its ${k}×${k} window`); }
    }

    let o = `pooling (${type}): a ${k}×${k} window strides by ${s} over the ${n}×${n} input → ${out}×${out} output (downsampling, no weights).   out = ⌊(${n}−${k})/${s}⌋+1 = ${out}.    tier:${r.name}\n`;
    o += sp ? `window (${oy},${ox}): ${type === 'max' ? 'max keeps the strongest cell (red), drops the rest' : 'avg = mean of the window'} = ${res.toFixed(2)}.`
      : `${type} pooling halves/2 the map per 2×2 stride-2; max → translation tolerance, avg → smoothing.`;
    page.setReadout(o);
  },
}).then((page) => {
  window.__poolPage = page;
  const q = new URLSearchParams(location.search);
  const t = page.controls._transport;
  let reb = false;
  for (const [qk, ck] of [['n', 'n'], ['k', 'k'], ['stride', 'stride']]) if (q.has(qk)) { page.controls.set(ck, parseInt(q.get(qk), 10)); reb = true; }
  if (q.has('pool')) page.controls.set('pool', q.get('pool'));
  if (reb && t) t.rebuild();
  if (q.has('drag')) { const [rr, cc, v] = q.get('drag').split(',').map(Number); if (cur && rr * cur.n + cc < cur.X.length) cur.X[rr * cur.n + cc] = v; }
  if (q.has('step') || q.has('drag') || q.has('hover') || reb) { if (t) t.pause(); }
  if (q.has('step') && t) t.seek(parseInt(q.get('step'), 10));
  if (q.has('hover')) { const [hx, hy] = q.get('hover').split(',').map(Number); page.pointer.x = hx; page.pointer.y = hy; page.pointer.over = true; }
  if (q.get('play') === '1' && t) t.play();
  page.redraw();
});
