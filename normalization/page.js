// normalization concept page -- RMSNorm vs LayerNorm (per-row) + pre/post-norm.
// Uses the verified framework: layout.mount() + controls + a per-row Transport.
//
// Interactive per the framework contract (plan/framework.md): drag any input
// cell vertically to change x[i] and watch that row's μ / σ (or RMS) and every
// normalized output recompute live (the row visibly re-centers + re-scales);
// hover an input cell for x[i], or a normalized cell for its full derivation
// ((x[i]-μ)/√(σ²+ε)·γ+β, or x[i]/rms·γ); the per-row sweep auto-plays + loops.
import { mount } from '../framework/layout.js';
import { ramps, cellAt } from '../framework/render.js';
import { seededRandn } from '../framework/tensor.js';

const INK = '#111', BLUE = '#1f6feb', GREEN = '#2ca02c';
const maxAbs = (a) => { let m = 1e-9; for (let i = 0; i < a.length; i++) if (Math.abs(a[i]) > m) m = Math.abs(a[i]); return m; };

// Shared between compute() (builds the per-row step list), draw() (renders +
// captures the rects), and onPointer() (hit-tests + edits). layout rebuilds the
// transport (=> compute()) before the matching draw, so `cur` is fresh. Drag
// edits mutate cur.X.data in place; resync() just refreshes the row count of the
// transport (the per-row stats + output are recomputed from cur.X every draw).
let cur = null;
let Irect = null, Orect = null;   // input + output heatmap rects, captured in draw for hit-testing
let grab = null;                  // {r, c} of the input cell being dragged

function buildData(st) {
  const N = st.N, D = st.D, seed = st.seed | 0;
  cur = { X: seededRandn(seed, [N, D]), N, D };
  return Array.from({ length: N }, (_, i) => ({ i, label: `token ${i}: normalize its ${D} features using stats from that row only` }));
}

// Recompute the transport's per-row step list from cur (after a dim change or a
// drag edit). The stats + normalized output are derived from cur.X live in
// draw(), so a drag survives + the row re-normalizes immediately.
function resync(page) {
  const t = page.controls._transport;
  if (!t || !cur) return;
  t.steps = Array.from({ length: cur.N }, (_, i) => ({ i, label: `token ${i}: normalize its ${cur.D} features using stats from that row only` }));
  t.scrub.max = Math.max(0, t.steps.length - 1);
  if (t.index > t.steps.length - 1) t.index = t.steps.length - 1;
  t._sync();
}

// Per-row LayerNorm / RMSNorm stats + normalized output, computed live from X.
function computeNorm(X, N, D, isLN) {
  const eps = isLN ? 1e-5 : 1e-6;
  const mu = new Float32Array(N), sd = new Float32Array(N), rms = new Float32Array(N);
  const Y = { data: new Float32Array(N * D), rows: N, cols: D };
  for (let i = 0; i < N; i++) {
    const row = X.data.subarray(i * D, i * D + D);
    if (isLN) {
      let m = 0; for (let j = 0; j < D; j++) m += row[j]; m /= D; mu[i] = m;
      let v = 0; for (let j = 0; j < D; j++) { const d = row[j] - m; v += d * d; } v /= D; sd[i] = Math.sqrt(v + eps);
      for (let j = 0; j < D; j++) Y.data[i * D + j] = (row[j] - m) / sd[i];
    } else {
      let ms = 0; for (let j = 0; j < D; j++) ms += row[j] * row[j]; ms /= D; rms[i] = Math.sqrt(ms + eps);
      for (let j = 0; j < D; j++) Y.data[i * D + j] = row[j] / rms[i];
    }
  }
  return { mu, sd, rms, Y, eps };
}

function tableHeat(r, M, rect, hiRow, label, dom) {
  const ctx = r.ctx, cell = rect.h / M.rows;
  r.heatmap(M, { rows: M.rows, cols: M.cols, rect, ramp: ramps.diverging, domain: [-dom, dom] });
  r.grid({ stroke: 'rgba(0,0,0,0.10)' });
  r.label(label, rect.x, rect.y - 7, { color: '#586069', font: '11px ui-monospace, monospace' });
  for (let i = 0; i < M.rows; i++) r.label('t' + i, rect.x - 22, rect.y + i * cell + cell / 2 + 3, { color: i === hiRow ? BLUE : '#9aa4ad', font: '10px ui-monospace, monospace' });
  if (hiRow >= 0) { ctx.save(); ctx.strokeStyle = INK; ctx.lineWidth = 2.5; ctx.strokeRect(rect.x - 1, rect.y + hiRow * cell - 1, M.cols * cell + 2, cell + 2); ctx.restore(); }
}

function statCol(r, vals, x, y, cell, label, hiRow, dom, ramp) {
  const ctx = r.ctx, N = vals.length;
  r.heatmap(vals, { rows: N, cols: 1, rect: { x, y, w: cell, h: N * cell }, ramp, domain: dom });
  r.grid({ stroke: 'rgba(0,0,0,0.10)' });
  r.label(label, x, y - 7, { color: '#586069', font: '10px ui-monospace, monospace' });
  ctx.save(); ctx.font = '9px ui-monospace, monospace'; ctx.textAlign = 'center'; ctx.textBaseline = 'middle';
  for (let i = 0; i < N; i++) { ctx.fillStyle = '#1a1d21'; ctx.fillText(vals[i].toFixed(2), x + cell / 2, y + i * cell + cell / 2); }
  if (hiRow >= 0) { ctx.strokeStyle = INK; ctx.lineWidth = 2; ctx.strokeRect(x - 1, y + hiRow * cell - 1, cell + 2, cell + 2); }
  ctx.restore();
}

function flowBox(ctx, x, y, w, h, label, col, hi) {
  ctx.save();
  ctx.fillStyle = hi ? 'rgba(31,111,235,0.16)' : '#f3f4f7'; ctx.strokeStyle = hi ? BLUE : (col || '#c4ccd3'); ctx.lineWidth = hi ? 2.5 : 1.5;
  ctx.fillRect(x, y, w, h); ctx.strokeRect(x, y, w, h);
  ctx.fillStyle = hi ? BLUE : '#3a4047'; ctx.font = '11px ui-monospace, monospace'; ctx.textAlign = 'center'; ctx.textBaseline = 'middle';
  ctx.fillText(label, x + w / 2, y + h / 2);
  ctx.restore();
}

mount({
  mount: 'body',
  title: 'normalization — RMSNorm vs LayerNorm',
  blurb: 'Each token (row) is normalized using stats from its own features. Drag any input cell vertically to change x[i] and watch that row\'s μ / σ (or RMS) and every normalized output re-center + re-scale live; hover an input cell for its value, or a normalized cell for its full derivation. LayerNorm subtracts the mean and divides by std; RMSNorm just divides by RMS (no mean-subtract, no bias). Let it play (or scrub) to sweep rows; toggle norm type and pre/post placement.',
  prefer: 'canvas2d',
  aspect: '2 / 1',
  autoplay: true,
  controls: (c, page) => {
    c.stepper('N', { label: 'tokens (N)', min: 3, max: 6, value: 4 });
    c.stepper('D', { label: 'features (D)', min: 4, max: 8, value: 6 });
    c.select('norm', { label: 'norm', value: 'rmsnorm', options: [{ value: 'rmsnorm', label: 'RMSNorm' }, { value: 'layernorm', label: 'LayerNorm' }] });
    c.toggle('prenorm', { label: 'pre-norm (else post)', value: true });
    c.transport({ compute: () => buildData(page.state), speed: 1.5, loop: true });
  },
  // Direct manipulation: grab an input cell, drag vertically to change x[i].
  // The row's μ / σ (or RMS) and every normalized output recompute under the
  // cursor (resync only refreshes the per-row transport axis).
  onPointer: (page, ev) => {
    if (!cur) return;
    const { X, N, D } = cur;
    if (ev.type === 'down') {
      grab = null;
      const h = Irect && cellAt(Irect, N, D, ev.x, ev.y);
      if (h) grab = { r: h.r, c: h.c };
    } else if (ev.type === 'up' || ev.type === 'leave') {
      grab = null;
    } else if (ev.type === 'move' && grab && page.pointer.down) {
      const idx = grab.r * D + grab.c;
      X.data[idx] = Math.max(-3, Math.min(3, X.data[idx] - ev.dy * 0.02));  // drag up = larger
      resync(page);
    }
  },
  draw: (page) => {
    const r = page.renderer, ctx = page.ctx, st = page.state;
    if (!cur) return;
    const { X, N, D } = cur;
    r.clear('#ffffff');
    const isLN = st.norm === 'layernorm';
    const { mu, sd, rms, Y, eps } = computeNorm(X, N, D, isLN);
    const s = page.step(), ri = s ? s.i : N - 1;

    const pad = 16, topY = 66;
    const cell = Math.max(13, Math.min(26, (page.W - 2 * pad - 220) / (2 * D + 3)));
    const Ix = pad + 30;
    Irect = { x: Ix, y: topY, w: D * cell, h: N * cell };
    const dX = maxAbs(X.data), dY = maxAbs(Y.data);
    tableHeat(r, X, Irect, ri, `input x [${N}×${D}]`, dX);

    let sx = Ix + D * cell + 22;
    if (isLN) { statCol(r, mu, sx, topY, cell, 'μ', ri, [-maxAbs(mu), maxAbs(mu)], ramps.diverging); sx += cell + 12; statCol(r, sd, sx, topY, cell, 'σ', ri, [0, maxAbs(sd)], ramps.sequential); sx += cell + 12; }
    else { statCol(r, rms, sx, topY, cell, 'RMS', ri, [0, maxAbs(rms)], ramps.sequential); sx += cell + 12; }

    r.label('→', sx + 2, topY + N * cell / 2 + 4, { color: '#9aa4ad', font: '16px ui-monospace, monospace' });
    Orect = { x: sx + 26, y: topY, w: D * cell, h: N * cell };
    tableHeat(r, Y, Orect, ri, isLN ? 'output (centered + scaled)' : 'output (unit RMS)', dY);

    // pre / post-norm placement diagram
    const dy = topY + N * cell + 52, bh = 30, bw = 78, gap = 16;
    r.label(st.prenorm ? 'pre-norm:  out = x + sublayer(norm(x))' : 'post-norm:  out = norm(x + sublayer(x))', pad, dy - 10, { color: INK, font: '12px ui-monospace, monospace' });
    const seq = st.prenorm ? ['x', 'Norm', 'Sublayer', '⊕ (+x)', 'out'] : ['x', 'Sublayer', '⊕ (+x)', 'Norm', 'out'];
    let bx = pad + 6;
    const centers = [];
    for (let k = 0; k < seq.length; k++) {
      const hi = seq[k] === 'Norm';
      flowBox(ctx, bx, dy, bw, bh, seq[k], hi ? BLUE : '#c4ccd3', hi);
      centers.push(bx + bw / 2);
      if (k < seq.length - 1) { ctx.save(); ctx.strokeStyle = '#9aa4ad'; ctx.beginPath(); ctx.moveTo(bx + bw, dy + bh / 2); ctx.lineTo(bx + bw + gap, dy + bh / 2); ctx.stroke(); ctx.restore(); }
      bx += bw + gap;
    }
    // residual arc: x -> the ⊕ box
    const addIdx = seq.findIndex((b) => b.startsWith('⊕'));
    ctx.save(); ctx.strokeStyle = GREEN; ctx.lineWidth = 1.5; ctx.beginPath(); ctx.moveTo(centers[0], dy); ctx.bezierCurveTo(centers[0], dy - 24, centers[addIdx], dy - 24, centers[addIdx], dy); ctx.stroke();
    ctx.fillStyle = GREEN; ctx.font = '9px ui-monospace, monospace'; ctx.textAlign = 'center'; ctx.fillText('residual', (centers[0] + centers[addIdx]) / 2, dy - 27); ctx.restore();

    // Hover-to-inspect: input cell -> x[i]; normalized cell -> full derivation.
    // γ=1, β=0 in this synthetic view (no learned affine), so the affine terms
    // are shown explicitly but evaluate to identity.
    if (page.pointer.over && !grab) {
      const p = page.pointer;
      const ih = Irect && cellAt(Irect, N, D, p.x, p.y);
      const oh = Orect && cellAt(Orect, N, D, p.x, p.y);
      let tip = null;
      if (ih) {
        const x = X.data[ih.r * D + ih.c];
        const st2 = isLN ? `row μ=${mu[ih.r].toFixed(3)}, σ=${sd[ih.r].toFixed(3)}` : `row RMS=${rms[ih.r].toFixed(3)}`;
        tip = `x[t${ih.r},${ih.c}] = ${x.toFixed(3)}\n${st2}\ndrag ↕ to change`;
      } else if (oh) {
        const x = X.data[oh.r * D + oh.c], y = Y.data[oh.r * D + oh.c];
        if (isLN) tip = `y[t${oh.r},${oh.c}] = (x−μ)/√(σ²+ε)·γ+β\n= (${x.toFixed(2)}−${mu[oh.r].toFixed(2)})/${sd[oh.r].toFixed(3)}·1+0\n= ${y.toFixed(3)}   (μ=${mu[oh.r].toFixed(3)}, σ=${sd[oh.r].toFixed(3)})`;
        else tip = `y[t${oh.r},${oh.c}] = x/rms·γ\n= ${x.toFixed(2)}/${rms[oh.r].toFixed(3)}·1\n= ${y.toFixed(3)}   (rms=${rms[oh.r].toFixed(3)})`;
      }
      if (tip) page.setTip(tip);
    }

    let o = `${isLN ? 'LayerNorm: y = (x−μ)/√(σ²+ε)·γ+β — subtract mean, divide by std (center + scale)' : 'RMSNorm: y = x/√(mean(x²)+ε)·w — divide by RMS only (no mean-subtract, no bias; Llama)'}    tier:${r.name}\n`;
    o += s ? `token ${ri}: ${isLN ? `μ=${mu[ri].toFixed(3)}, σ=${sd[ri].toFixed(3)}` : `RMS=${rms[ri].toFixed(3)}`} → normalize that row    (drag an input cell ↕ to watch the row re-normalize)` : '(drag an input cell ↕ to re-normalize · press ▶ or scrub rows; toggle RMSNorm/LayerNorm and pre/post-norm)';
    page.setReadout(o);
  },
}).then((page) => {
  window.__normPage = page;
  const q = new URLSearchParams(location.search);
  const t = page.controls._transport;
  if (q.has('norm')) { page.controls.set('norm', q.get('norm')); }
  if (q.has('prenorm')) { page.controls.set('prenorm', q.get('prenorm') !== '0'); }
  // ?drag=r,c,val sets input cell x[r,c] to a value (headless stand-in for a
  // vertical drag, since --screenshot has no pointer). e.g. ?drag=2,2.5 sets
  // x[row 2, col 0]; ?drag=1,3,2.5 sets x[row 1, col 3]. resync re-normalizes.
  if (q.has('drag') && cur) {
    const parts = q.get('drag').split(',').map(Number);
    const [rr, cc, v] = parts.length >= 3 ? parts : [parts[0], 0, parts[1]];
    if (rr >= 0 && rr < cur.N && cc >= 0 && cc < cur.D) { cur.X.data[rr * cur.D + cc] = v; resync(page); }
  }
  // ?hover=x,y fakes the cursor position (headless stand-in for a real hover)
  // so the tooltip path is verifiable.
  if (q.has('hover')) {
    const [hx, hy] = q.get('hover').split(',').map(Number);
    page.pointer.x = hx; page.pointer.y = hy; page.pointer.over = true;
  }
  // Deterministic frame for capture: pause the transport for any of these hooks.
  if (q.has('step') || q.has('drag') || q.has('hover')) { if (t) t.pause(); }
  if (q.has('step') && t) t.seek(parseInt(q.get('step'), 10));
  if (q.get('play') === '1' && t) t.play();
  page.redraw();
});
