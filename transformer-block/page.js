// transformer-block concept page -- one full pre-norm block end to end:
// x -> RMSNorm -> Attention -> (+) -> RMSNorm -> MLP -> (+) -> out, drawn as a
// residual spine with two sublayer branches that read it (via norm) and add
// back. Every tensor is really computed (Q/K/V + causal softmax + SwiGLU).
// Interactive per the framework contract: drag any x cell to steer the input
// (the whole block recomputes), hover any cell for its value + stage, step the
// six stages (auto-plays + loops).
import { mount } from '../framework/layout.js';
import { ramps, cellAt } from '../framework/render.js';
import { matmul, transpose, softmax, rmsnorm, sigmoid, seededRandn } from '../framework/tensor.js';

const INK = '#111', BLUE = '#1f6feb', GREEN = '#2ca02c', PURPLE = '#8957e5', GREY = '#8a939b';
const M = (r, c) => ({ data: new Float32Array(r * c), rows: r, cols: c });
const maxAbs = (a) => { let m = 1e-9; for (let i = 0; i < a.length; i++) if (Math.abs(a[i]) > m) m = Math.abs(a[i]); return m; };

const STAGES = [
  { key: 'norm1', tag: 'attn', label: 'RMSNorm(x) — read the stream' },
  { key: 'attn', tag: 'attn', label: 'a = Attention(RMSNorm(x))' },
  { key: 'add1', tag: 'spine', label: 'h1 = x + a — add back (residual ⊕)' },
  { key: 'norm2', tag: 'mlp', label: 'RMSNorm(h1) — read the stream' },
  { key: 'mlp', tag: 'mlp', label: 'm = MLP(RMSNorm(h1)) — SwiGLU' },
  { key: 'add2', tag: 'spine', label: 'out = h1 + m — add back (residual ⊕)' },
];

let cur = null;
let xRect = null, aRect = null, h1Rect = null, mRect = null, outRect = null;  // captured in draw
let grab = null;  // {r,c} while dragging an x cell

function rmsRows(X) {                       // RMSNorm each row -> new mat
  const out = M(X.rows, X.cols);
  for (let i = 0; i < X.rows; i++) out.data.set(rmsnorm(X.data.subarray(i * X.cols, (i + 1) * X.cols)), i * X.cols);
  return out;
}
function addMat(A, B) { const o = M(A.rows, A.cols); for (let i = 0; i < o.data.length; i++) o.data[i] = A.data[i] + B.data[i]; return o; }

function attn(X, W) {                        // single-head causal self-attention -> [N×D]
  const N = X.rows, d = X.cols, scale = 1 / Math.sqrt(d);
  const Q = matmul(X, W.Wq), K = matmul(X, W.Wk), V = matmul(X, W.Wv);
  const S = matmul(Q, transpose(K));
  const P = M(N, N);
  for (let i = 0; i < N; i++) {
    const row = new Float32Array(N);
    for (let j = 0; j < N; j++) row[j] = j <= i ? S.data[i * N + j] * scale : -Infinity;  // causal
    P.data.set(softmax(row), i * N);
  }
  return matmul(matmul(P, V), W.Wo);
}
function mlp(X, W) {                          // SwiGLU FFN -> [N×D]
  const G = matmul(X, W.Wg), U = matmul(X, W.Wu), H = M(G.rows, G.cols);
  for (let i = 0; i < H.data.length; i++) { const g = G.data[i]; H.data[i] = g * sigmoid(g) * U.data[i]; }
  return matmul(H, W.Wd);
}

function buildData(st) {
  const N = st.N | 0, D = st.D | 0, F = 2 * D, seed = st.seed | 0, sw = 1 / Math.sqrt(D);
  const x = seededRandn(seed, [N, D], { std: 1 });
  const W = {
    Wq: seededRandn(seed + 1, [D, D], { std: sw }), Wk: seededRandn(seed + 2, [D, D], { std: sw }),
    Wv: seededRandn(seed + 3, [D, D], { std: sw }), Wo: seededRandn(seed + 4, [D, D], { std: sw }),
    Wg: seededRandn(seed + 5, [D, F], { std: sw }), Wu: seededRandn(seed + 6, [D, F], { std: sw }),
    Wd: seededRandn(seed + 7, [F, D], { std: 1 / Math.sqrt(F) }),
  };
  cur = { x, W, N, D };
  return STAGES.map((s, i) => ({ i, ...s }));
}

// Recompute the whole block from the (possibly steered) cur.x.
function block() {
  const { x, W } = cur;
  const n1 = rmsRows(x), a = attn(n1, W), h1 = addMat(x, a);
  const n2 = rmsRows(h1), m = mlp(n2, W), out = addMat(h1, m);
  return { n1, a, h1, n2, m, out };
}

function fbox(ctx, x, y, w, h, label, color, active) {
  ctx.save();
  ctx.fillStyle = active ? color : '#f3f4f7'; ctx.globalAlpha = active ? 0.16 : 1; ctx.fillRect(x, y, w, h); ctx.globalAlpha = 1;
  ctx.strokeStyle = active ? color : '#c4ccd3'; ctx.lineWidth = active ? 2.5 : 1.4; ctx.strokeRect(x, y, w, h);
  ctx.fillStyle = active ? color : '#3a4047'; ctx.font = '10px ui-monospace, monospace'; ctx.textAlign = 'center'; ctx.textBaseline = 'middle';
  for (const [k, ln] of label.split('\n').entries()) ctx.fillText(ln, x + w / 2, y + h / 2 + (k - (label.split('\n').length - 1) / 2) * 12);
  ctx.restore();
}
function arrow(ctx, x1, y1, x2, y2, color, dim) {
  ctx.save(); ctx.strokeStyle = color; ctx.fillStyle = color; ctx.lineWidth = 1.6; ctx.globalAlpha = dim ? 0.25 : 1;
  ctx.beginPath(); ctx.moveTo(x1, y1); ctx.lineTo(x2, y2); ctx.stroke();
  const ang = Math.atan2(y2 - y1, x2 - x1);
  ctx.beginPath(); ctx.moveTo(x2, y2);
  ctx.lineTo(x2 - 7 * Math.cos(ang - 0.4), y2 - 7 * Math.sin(ang - 0.4));
  ctx.lineTo(x2 - 7 * Math.cos(ang + 0.4), y2 - 7 * Math.sin(ang + 0.4)); ctx.closePath(); ctx.fill();
  ctx.restore();
}
function oplus(ctx, x, y, active) {
  ctx.save(); ctx.strokeStyle = active ? GREEN : '#9aa4ad'; ctx.fillStyle = '#fff'; ctx.lineWidth = active ? 2.5 : 1.5;
  ctx.beginPath(); ctx.arc(x, y, 11, 0, 7); ctx.fill(); ctx.stroke();
  ctx.beginPath(); ctx.moveTo(x - 6, y); ctx.lineTo(x + 6, y); ctx.moveTo(x, y - 6); ctx.lineTo(x, y + 6); ctx.stroke();
  ctx.restore();
}

mount({
  mount: 'body',
  title: 'transformer-block — norm → attn → ⊕ → norm → mlp → ⊕',
  blurb: 'One full pre-norm transformer block end to end. A token sequence x flows along the residual spine; two sublayers branch off — Attention then MLP — each reading the stream through RMSNorm and adding its result back (never overwriting). Every tensor is really computed (Q/K/V + causal softmax + SwiGLU). Drag any x cell to steer the input and watch the whole block recompute; hover any cell for its value; step the six stages (or let it play).',
  prefer: 'webgl2',
  aspect: '2 / 1',
  autoplay: true,
  controls: (c, page) => {
    c.stepper('N', { label: 'tokens (N)', min: 3, max: 6, value: 4 });
    c.stepper('D', { label: 'features (D)', min: 4, max: 8, value: 6 });
    c.slider('seed', { label: 'seed', min: 0, max: 99, step: 1, value: 7, rebuild: true });
    c.transport({ compute: () => buildData(page.state), speed: 1.2, loop: true });
  },
  onPointer: (page, ev) => {
    if (!cur) return;
    const D = cur.D, N = cur.N;
    if (ev.type === 'down') { grab = xRect && cellAt(xRect, N, D, ev.x, ev.y); }
    else if (ev.type === 'up' || ev.type === 'leave') { grab = null; }
    else if (ev.type === 'move' && grab && page.pointer.down) {
      const i = grab.r * D + grab.c;
      cur.x.data[i] = Math.max(-3, Math.min(3, cur.x.data[i] - ev.dy * 0.02));
      page.redraw();
    }
  },
  draw: (page) => {
    const r = page.renderer, ctx = page.ctx, st = page.state;
    if (!cur) return;
    const { N, D } = cur;
    r.clear('#ffffff');
    const B = block();
    const s = page.step(), stage = s ? s.i : STAGES.length - 1;   // reveal up to this stage

    // shared diverging domain across the stream + branch tensors
    let dom = 1e-9; for (const T of [cur.x, B.a, B.h1, B.m, B.out]) dom = Math.max(dom, maxAbs(T.data));

    const pad = 16;
    const cs = Math.max(11, Math.min(22, Math.min((page.W * 0.13) / D, (page.H * 0.30) / N)));
    const hw = D * cs, hh = N * cs;
    const ySpine = page.H - pad - hh - 26;
    const yBranch = 42;
    const sX = pad + hw / 2 + 6, sOut = page.W - pad - hw / 2 - 6, sH1 = (sX + sOut) / 2;
    const ox1 = (sX + sH1) / 2, ox2 = (sH1 + sOut) / 2;
    const yMid = ySpine + hh / 2;

    const drawHeat = (T, cx, top, faded) => {
      const rect = { x: cx - hw / 2, y: top, w: hw, h: hh };
      ctx.save(); if (faded) ctx.globalAlpha = 0.18;
      r.heatmap(T, { rows: T.rows, cols: T.cols, rect, ramp: ramps.diverging, domain: [-dom, dom] });
      r.grid({ stroke: 'rgba(0,0,0,0.10)' });
      ctx.restore();
      return rect;
    };

    // ---- residual spine (the identity highway) ----
    ctx.save(); ctx.strokeStyle = '#c4ccd3'; ctx.lineWidth = 3;
    ctx.beginPath(); ctx.moveTo(sX, yMid); ctx.lineTo(sOut, yMid); ctx.stroke(); ctx.restore();
    r.label('residual stream (identity)', sX - hw / 2, ySpine - 8, { color: GREY, font: '10px ui-monospace, monospace' });

    // ---- spine nodes: x, h1, out ----
    xRect = drawHeat(cur.x, sX, ySpine, false);
    h1Rect = drawHeat(B.h1, sH1, ySpine, stage < 2);
    outRect = drawHeat(B.out, sOut, ySpine, stage < 5);
    r.label('x [N×D]', sX - hw / 2, ySpine + hh + 14, { color: INK, font: '11px ui-monospace, monospace' });
    r.label('h1 = x+a', sH1 - hw / 2, ySpine + hh + 14, { color: stage < 2 ? GREY : INK, font: '11px ui-monospace, monospace' });
    r.label('out = h1+m', sOut - hw / 2, ySpine + hh + 14, { color: stage < 5 ? GREY : INK, font: '11px ui-monospace, monospace' });
    ctx.save(); ctx.fillStyle = GREEN; ctx.font = '10px ui-monospace, monospace'; ctx.textAlign = 'center'; ctx.textBaseline = 'top';
    ctx.fillText('drag ↕ to steer', sX, ySpine + hh + 26); ctx.restore();

    // ---- the two branches (attn, mlp) ----
    const branch = (cx, normLabel, subLabel, color, T, tagActive, revealed, op_x, opActive) => {
      const bw = 64, bh = 30, gap = 12, groupW = bw + gap + hw, gx = cx - groupW / 2;
      const aRectB = drawHeat(T, gx + bw + gap + hw / 2, yBranch, !revealed);
      fbox(ctx, gx, yBranch + hh / 2 - bh / 2, bw, bh, normLabel + '\n' + subLabel, color, tagActive);
      arrow(ctx, gx + bw, yBranch + hh / 2, gx + bw + gap + hw / 2 - hw / 2 - 4, yBranch + hh / 2, color, !revealed);  // box -> heatmap
      // read: spine (left of op) up into the box
      arrow(ctx, op_x - 26, yMid - 6, gx + bw / 2, yBranch + hh / 2 + bh / 2 + 4, color, !tagActive && !revealed);
      // add: heatmap down to the ⊕
      arrow(ctx, aRectB.x + hw / 2, yBranch + hh + 4, op_x, yMid - 12, color, !revealed);
      oplus(ctx, op_x, yMid, opActive);
      return aRectB;
    };
    aRect = branch(ox1, 'RMSNorm', 'Attention', BLUE, B.a, STAGES[stage].tag === 'attn', stage >= 1, ox1, stage >= 2);
    mRect = branch(ox2, 'RMSNorm', 'MLP·SwiGLU', PURPLE, B.m, STAGES[stage].tag === 'mlp', stage >= 4, ox2, stage >= 5);
    r.label('a = Attn(norm(x))', aRect.x, yBranch - 6, { color: stage >= 1 ? BLUE : GREY, font: '10px ui-monospace, monospace' });
    r.label('m = MLP(norm(h1))', mRect.x, yBranch - 6, { color: stage >= 4 ? PURPLE : GREY, font: '10px ui-monospace, monospace' });

    // ---- hover-to-inspect ----
    if (page.pointer.over && !grab) {
      const p = page.pointer;
      const tens = [['x', cur.x, xRect, 'input'], ['a', B.a, aRect, 'Attention(RMSNorm(x))'],
        ['h1', B.h1, h1Rect, 'x + a (residual)'], ['m', B.m, mRect, 'MLP(RMSNorm(h1))'], ['out', B.out, outRect, 'h1 + m (residual)']];
      for (const [name, T, rect, note] of tens) {
        const hit = rect && cellAt(rect, N, D, p.x, p.y);
        if (hit) { page.setTip(`${name}[t${hit.r}, d${hit.c}] = ${T.data[hit.r * D + hit.c].toFixed(3)}\n${note}${name === 'x' ? '\ndrag ↕ to steer' : ''}`); break; }
      }
    }

    let o = `pre-norm block: stream ← stream + sublayer(RMSNorm(stream)), twice (attention, then MLP).    tier:${r.name}\n`;
    o += s ? `stage ${stage + 1}/6 — ${s.label}` : '(press ▶ or scrub the 6 stages · drag an x cell to steer the block)';
    page.setReadout(o);
  },
}).then((page) => {
  window.__tbPage = page;
  const q = new URLSearchParams(location.search);
  const t = page.controls._transport;
  // ?drag=r,c,val sets x[r,c] (headless stand-in for a vertical drag on x).
  if (q.has('drag')) {
    const [r, c, v] = q.get('drag').split(',').map(Number);
    if (cur && cur.x && r * cur.D + c < cur.x.data.length) cur.x.data[r * cur.D + c] = v;
  }
  if (q.has('hover')) {
    const [hx, hy] = q.get('hover').split(',').map(Number);
    page.pointer.x = hx; page.pointer.y = hy; page.pointer.over = true;
  }
  if (q.has('step') || q.has('drag') || q.has('hover')) { if (t) t.pause(); }
  if (q.has('step') && t) t.seek(parseInt(q.get('step'), 10));
  if (q.get('play') === '1' && t) t.play();
  page.redraw();
});
