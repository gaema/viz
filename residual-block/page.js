// residual-block concept page -- the ResNet skip connection y = F(x) + x, and
// WHY it eases very deep training. Two linked views:
//   (1) one block: x -> F(x)=W2·relu(W1·x) -> add the skip -> y = F(x) + x.
//       The F branch is scaled to a gain g relative to x, so at init (g small)
//       F(x)≈0 and the block ≈ identity (y≈x) -- a small refinement on top of x.
//   (2) depth chart: the gradient magnitude reaching layer 0 as it propagates
//       back through L stacked blocks. Per block the local factor is g (no skip)
//       vs 1+g (skip), so grad_0 = g^L vanishes without the skip while the
//       identity path keeps (1+g)^L ~ O(1) with it. ∂y/∂x = I + ∂F/∂x: the "+I"
//       is the gradient highway. Drag x; tune depth L, gain g, toggle the skip.
import { mount } from '../framework/layout.js';
import { seededRandn } from '../framework/tensor.js';

const INK = '#111', BLUE = '#1f6feb', ORANGE = '#d2691e', GREEN = '#2ca02c', RED = '#d1242f', GREY = '#9aa4ad', PURPLE = '#8250df';
const N = 8, EPS = 1e-9;
const relu = (a) => Float32Array.from(a, (v) => (v > 0 ? v : 0));
const norm = (a) => { let s = 0; for (let i = 0; i < a.length; i++) s += a[i] * a[i]; return Math.sqrt(s); };
const matvec = (W, x) => { const out = new Float32Array(N); for (let i = 0; i < N; i++) { let s = 0; for (let j = 0; j < N; j++) s += W[i * N + j] * x[j]; out[i] = s; } return out; };

let cur = null, bsig = '';
let xRect = null, dragI = -1;

function build(seed) {
  const x = seededRandn(seed, N, { std: 0.7 });
  for (let i = 0; i < N; i++) x[i] = Math.max(-1, Math.min(2, x[i] + 0.5));
  const W1 = seededRandn(seed + 1, [N, N], { std: 0.5 }).data, W2 = seededRandn(seed + 2, [N, N], { std: 0.5 }).data;
  return { x, W1, W2 };
}

// vertical bar-chart of a vector, from a zero baseline; returns the rect.
function drawVec(r, ctx, x0, y0, w, h, vals, lo, hi, color, label) {
  const bw = w / vals.length, zy = y0 + h * (hi / (hi - lo));
  ctx.save();
  ctx.strokeStyle = '#e6e8ea'; ctx.lineWidth = 1; ctx.strokeRect(x0, y0, w, h);
  ctx.strokeStyle = '#c4c9ce'; ctx.beginPath(); ctx.moveTo(x0, zy); ctx.lineTo(x0 + w, zy); ctx.stroke();
  for (let i = 0; i < vals.length; i++) {
    const v = Math.max(lo, Math.min(hi, vals[i])), bh = Math.abs(v) / (hi - lo) * h;
    ctx.fillStyle = color; ctx.fillRect(x0 + i * bw + 1, v >= 0 ? zy - bh : zy, bw - 2, bh);
  }
  ctx.restore();
  if (label) r.label(label, x0, y0 - 6, { color: INK, font: '10px ui-monospace, monospace' });
  return { x: x0, y: y0, w, h, n: vals.length, lo, hi };
}

mount({
  mount: 'body',
  title: 'residual-block — the skip connection y = F(x) + x',
  blurb: 'A residual (ResNet) block computes y = F(x) + x: a small learned function F runs in parallel with an identity shortcut that copies x straight to the output. The point is deep training. At initialization F is near zero, so the block ≈ identity — signal passes through untouched. The real win is the gradient: ∂y/∂x = I + ∂F/∂x, an identity term plus F\'s. Stack L blocks and that "+I" gives a gradient HIGHWAY straight back to layer 0 — the magnitude stays ~(1+g)^L ≈ O(1). Remove the skip and the same per-block factor g compounds to g^L, which for g<1 vanishes (no gradient reaches the early layers → they never learn). Drag the input x to refeed the block; tune the depth L and the F-branch gain g; toggle the skip to watch the gradient curve crash. The gradient pulse animates back through the stack.',
  prefer: 'canvas2d',
  aspect: '2 / 1',
  animate: true,
  compare: { key: 'skip', a: true, b: false, labelA: 'skip ON — gradient flows', labelB: 'skip OFF — gradient vanishes' },
  challenges: [
    { goal: 'WITHOUT the skip, make the gradient vanish: reaching layer 0 below 1e-6.', hint: 'turn the skip OFF, lower the gain g, and raise the depth L — g^L collapses.', check: (api) => ({ solved: !api.state.skip && (api.probe.gNo ?? 1) < 1e-6, detail: api.state.skip ? 'skip is ON — turn it off first' : `no-skip grad = ${(api.probe.gNo ?? 1).toExponential(1)} (need < 1e-6)` }) },
    { goal: 'WITH the skip ON, keep the gradient healthy (≥ 0.5) at depth ≥ 30.', hint: 'skip ON keeps (1+g)^L ~ O(1); a small gain g stays near 1 even when L is large.', check: (api) => ({ solved: api.state.skip && (api.state.L | 0) >= 30 && (api.probe.gSk ?? 0) >= 0.5, detail: api.state.skip ? `skip grad = ${(api.probe.gSk ?? 0).toFixed(2)}, L=${api.state.L | 0}` : 'turn the skip ON' }) },
  ],
  controls: (c, page) => {
    c.slider('L', { label: 'depth (blocks)', min: 2, max: 40, step: 1, value: 24 });
    c.slider('gain', { label: 'F-branch gain g', min: 0, max: 0.6, step: 0.01, value: 0.06 });
    c.toggle('skip', { label: 'skip connection', value: true });
    c.slider('seed', { label: 'seed', min: 0, max: 99, step: 1, value: 7, rebuild: true });
  },
  onPointer: (page, ev) => {
    if (!cur || !xRect) return;
    const hit = (x, y) => { const c = Math.floor((x - xRect.x) / (xRect.w / N)); return (c >= 0 && c < N && x >= xRect.x && x <= xRect.x + xRect.w && y >= xRect.y - 8 && y <= xRect.y + xRect.h + 8) ? c : -1; };
    const setv = (y) => Math.max(xRect.lo, Math.min(xRect.hi, xRect.hi - (y - xRect.y) / xRect.h * (xRect.hi - xRect.lo)));
    if (ev.type === 'down') { dragI = hit(ev.x, ev.y); if (dragI >= 0) { cur.x[dragI] = setv(ev.y); page.redraw(); } }
    else if (ev.type === 'up' || ev.type === 'leave') dragI = -1;
    else if (ev.type === 'move' && dragI >= 0 && page.pointer.down) { cur.x[dragI] = setv(ev.y); page.redraw(); }
  },
  draw: (page) => {
    const r = page.renderer, ctx = page.ctx, st = page.state;
    if (st.seed + '' !== bsig) { cur = build(st.seed | 0); bsig = st.seed + ''; }
    r.clear('#ffffff');
    const g = st.gain, L = st.L | 0, skip = st.skip, x = cur.x;
    // one block, F scaled to gain g relative to x
    const h = relu(matvec(cur.W1, x)), Fraw = matvec(cur.W2, h);
    const nx = norm(x), nf = norm(Fraw) + EPS, s = g * nx / nf;
    const F = Float32Array.from(Fraw, (v) => v * s);
    const y = Float32Array.from(F, (v, i) => v + (skip ? x[i] : 0));

    const W = page.W, pad = 16, lo = -1.4, hi = 2.4, vh = 92, vw = 78, ty = 56;
    // --- single block (top) ---
    r.label('one residual block', pad, 30, { color: INK, font: '12px ui-monospace, monospace' });
    xRect = drawVec(r, ctx, pad, ty, vw, vh, x, lo, hi, BLUE, 'input x   (drag ↕)');
    const fbX = pad + vw + 34;
    ctx.save(); ctx.fillStyle = 'rgba(130,80,223,0.08)'; ctx.fillRect(fbX, ty + 8, 86, vh - 16); ctx.strokeStyle = PURPLE; ctx.lineWidth = 1.3; ctx.strokeRect(fbX, ty + 8, 86, vh - 16);
    ctx.fillStyle = PURPLE; ctx.font = '10px ui-monospace, monospace'; ctx.textAlign = 'center';
    ctx.fillText('F branch', fbX + 43, ty + 26); ctx.fillStyle = '#444'; ctx.fillText('W₂·relu(W₁x)', fbX + 43, ty + 44); ctx.fillText(`gain g=${g.toFixed(2)}`, fbX + 43, ty + 62); ctx.restore();
    const fxX = fbX + 86 + 34;
    drawVec(r, ctx, fxX, ty, vw, vh, F, lo, hi, ORANGE, 'F(x)');
    const plusX = fxX + vw + 22;
    ctx.save(); ctx.strokeStyle = INK; ctx.lineWidth = 1.4; ctx.beginPath(); ctx.arc(plusX, ty + vh / 2, 11, 0, 7); ctx.stroke(); ctx.fillStyle = INK; ctx.font = '15px ui-monospace, monospace'; ctx.textAlign = 'center'; ctx.fillText('+', plusX, ty + vh / 2 + 5); ctx.restore();
    const yX = plusX + 26;
    drawVec(r, ctx, yX, ty, vw, vh, y, lo, hi, GREEN, skip ? 'y = F(x) + x' : 'y = F(x)  (no skip)');
    // flow arrows + the skip arc
    ctx.save(); ctx.strokeStyle = GREY; ctx.lineWidth = 1.3;
    const ay = ty + vh / 2;
    ctx.beginPath(); ctx.moveTo(pad + vw + 3, ay); ctx.lineTo(fbX - 3, ay); ctx.moveTo(fbX + 86 + 3, ay); ctx.lineTo(fxX - 3, ay); ctx.moveTo(fxX + vw + 3, ay); ctx.lineTo(plusX - 12, ay); ctx.moveTo(plusX + 12, ay); ctx.lineTo(yX - 3, ay); ctx.stroke();
    if (skip) { // identity highway from x over F to the +
      ctx.strokeStyle = GREEN; ctx.lineWidth = 1.8; ctx.setLineDash([4, 3]); ctx.beginPath();
      ctx.moveTo(pad + vw / 2, ty - 2); ctx.bezierCurveTo(pad + vw / 2, ty - 30, plusX, ty - 30, plusX, ty + vh / 2 - 12); ctx.stroke(); ctx.setLineDash([]);
      ctx.fillStyle = GREEN; ctx.font = '9px ui-monospace, monospace'; ctx.textAlign = 'center'; ctx.fillText('identity skip (copies x)', (pad + plusX) / 2, ty - 34);
    }
    ctx.restore();
    // gradient identity note
    r.label(`backward:  ∂y/∂x = I + ∂F/∂x   — the "+I" is a gradient highway that bypasses F`, pad, ty + vh + 22, { color: skip ? GREEN : '#586069', font: '10px ui-monospace, monospace' });
    r.label(g < 0.12 ? 'g small → F(x)≈0 → block ≈ identity (y≈x): a small refinement on top of the signal.' : 'larger g → F(x) contributes more; without normalization deep skips can also blow up.', pad, ty + vh + 38, { color: '#586069', font: '10px ui-monospace, monospace' });

    // --- depth / gradient chart (bottom) ---
    const cx = pad, cTop = ty + vh + 54, cw = W - 2 * pad - 6, chH = Math.max(58, page.H - cTop - 42);
    r.label('gradient magnitude as it flows back through the stack (log scale)', cx, cTop - 8, { color: INK, font: '11px ui-monospace, monospace' });
    const eLo = -22, eHi = 2.5, X = (d) => cx + d / Math.max(1, L) * cw, Y = (e) => cTop + chH - (Math.max(eLo, Math.min(eHi, e)) - eLo) / (eHi - eLo) * chH;
    ctx.save();
    // axes + 1.0 line
    ctx.strokeStyle = '#eceef0'; ctx.lineWidth = 1; ctx.strokeRect(cx, cTop, cw, chH);
    ctx.strokeStyle = 'rgba(150,160,170,0.6)'; ctx.setLineDash([4, 4]); ctx.beginPath(); ctx.moveTo(cx, Y(0)); ctx.lineTo(cx + cw, Y(0)); ctx.stroke(); ctx.setLineDash([]);
    r.label('|grad| = 1', cx + cw - 64, Y(0) - 5, { color: '#8a939b', font: '9px ui-monospace, monospace' });
    // curves
    const noskip = (d) => d * Math.log10(g + EPS), withskip = (d) => d * Math.log10(1 + g);
    const drawCurve = (fn, col, wid) => { ctx.strokeStyle = col; ctx.lineWidth = wid; ctx.beginPath(); for (let d = 0; d <= L; d++) { const px = X(d), py = Y(fn(d)); if (d === 0) ctx.moveTo(px, py); else ctx.lineTo(px, py); } ctx.stroke(); };
    drawCurve(noskip, skip ? 'rgba(209,36,47,0.45)' : RED, skip ? 1.5 : 2.5);
    drawCurve(withskip, skip ? GREEN : 'rgba(44,160,44,0.45)', skip ? 2.5 : 1.5);
    // animated backprop pulse traveling output(left)->input(right)
    const pd = ((page.t || 0) % 3) / 3 * L, active = skip ? withskip : noskip;
    ctx.fillStyle = skip ? GREEN : RED; ctx.beginPath(); ctx.arc(X(pd), Y(active(pd)), 3.5, 0, 7); ctx.fill();
    ctx.restore();
    // endpoint annotations
    const gNo = Math.pow(g, L), gSk = Math.pow(1 + g, L);
    page.probe = { gNo, gSk };
    r.label('output (layer L)', cx + 2, cTop + chH + 14, { color: '#586069', font: '9px ui-monospace, monospace' });
    r.label('→ input (layer 0)', cx + cw - 92, cTop + chH + 14, { color: '#586069', font: '9px ui-monospace, monospace' });
    r.label(`reaching layer 0 after ${L} blocks:`, cx + 2, cTop + chH + 30, { color: INK, font: '10px ui-monospace, monospace' });
    r.label(`skip: ${gSk >= 0.01 ? gSk.toFixed(2) : gSk.toExponential(1)} (gradient flows)`, cx + 210, cTop + chH + 30, { color: GREEN, font: '10px ui-monospace, monospace' });
    r.label(`no-skip: ${gNo >= 1e-4 ? gNo.toFixed(4) : gNo.toExponential(1)} ${gNo < 1e-3 ? '(vanished)' : ''}`, cx + 430, cTop + chH + 30, { color: RED, font: '10px ui-monospace, monospace' });

    // hover
    if (page.pointer.over && dragI < 0) {
      const p = page.pointer;
      if (xRect && p.x >= xRect.x && p.x <= xRect.x + xRect.w && p.y >= xRect.y - 8 && p.y <= xRect.y + xRect.h + 8) {
        const i = Math.floor((p.x - xRect.x) / (xRect.w / N)); if (i >= 0 && i < N) page.setTip(`x[${i}] = ${x[i].toFixed(2)}\nF(x)[${i}]=${F[i].toFixed(2)}  →  y[${i}]=${y[i].toFixed(2)}${skip ? ` = ${F[i].toFixed(2)}+${x[i].toFixed(2)}` : ''}\ndrag ↕ to change`);
      } else if (p.x >= cx && p.x <= cx + cw && p.y >= cTop && p.y <= cTop + chH) {
        const d = Math.round((p.x - cx) / cw * L); page.setTip(`after ${d} block(s):\nskip:   (1+g)^${d} = ${Math.pow(1 + g, d).toExponential(1)}\nno-skip: g^${d} = ${Math.pow(g, d).toExponential(1)}`);
      }
    }

    let o = `residual block: y = F(x) + x.  F branch gain g=${g.toFixed(2)} (|F|≈g·|x|), depth L=${L}, skip ${skip ? 'ON' : 'OFF'}.   tier:${r.name}\n`;
    o += skip
      ? `with the skip, the gradient reaching layer 0 is (1+g)^${L} ≈ ${gSk >= 0.01 ? gSk.toFixed(2) : gSk.toExponential(1)} — the identity path keeps it O(1) no matter how deep. At init g≈0 so y≈x and (1+0)^L=1: perfect signal + gradient flow through the whole stack.`
      : `WITHOUT the skip the per-block factor g=${g.toFixed(2)} compounds: gradient at layer 0 = g^${L} ≈ ${gNo.toExponential(1)} — vanished. The early layers get almost no gradient and never learn. Toggle the skip back on to restore the highway.`;
    page.setReadout(o);
  },
}).then((page) => {
  window.__rbPage = page;
  const q = new URLSearchParams(location.search);
  if (q.has('L')) page.controls.set('L', +q.get('L'));
  if (q.has('gain')) page.controls.set('gain', +q.get('gain'));
  if (q.has('skip')) page.controls.set('skip', q.get('skip') === '1' || q.get('skip') === 'true');
  if (q.has('drag') && cur) for (const pr of q.get('drag').split(';')) { const [i, v] = pr.split(',').map(Number); if (i >= 0 && i < N) cur.x[i] = v; }
  if (q.has('hover')) { const [hx, hy] = q.get('hover').split(',').map(Number); page.pointer.x = hx; page.pointer.y = hy; page.pointer.over = true; }
  page.redraw();
});
