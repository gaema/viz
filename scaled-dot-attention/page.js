// scaled-dot-attention concept page -- the single-head pipeline
// QKᵀ/√d -> mask -> softmax -> ·V, stage by stage. Uses the verified framework.
//
// Interactive per the framework contract (plan/framework.md): the QKᵀ → scale →
// softmax → ·V pipeline auto-plays + loops; hover a score / weight / output cell
// for its derivation; drag a Q-row cell vertically to change a query component
// and watch that row of scores, the softmax weights, and the output recompute
// live (the attention "aha" -- the query is the operand you steer).
import { mount } from '../framework/layout.js';
import { ramps, cellAt } from '../framework/render.js';
import { dot, softmax, seededRandn } from '../framework/tensor.js';

const INK = '#111', BLUE = '#1f6feb', ORANGE = '#d2691e';
const maxAbs = (a) => { let m = 1e-9; for (let i = 0; i < a.length; i++) { const x = Math.abs(a[i]); if (x > m) m = x; } return m; };
const row = (M, i) => M.data.subarray(i * M.cols, i * M.cols + M.cols);

// Shared between buildData(), draw(), and onPointer(). recompute() rebuilds the
// derived matrices (scores -> scaled -> weights -> output) from cur.Q/K/V, so a
// drag edit of a Q cell flows all the way through. Drag edits mutate cur.Q in
// place; recompute() refreshes everything downstream.
let cur = null;
// Hit-test rects captured in draw(): the Q input strip, the N×N hero matrix
// (scores when softstage=false, weights when softstage=true), and the output.
let qRect = null, matRect = null, matSoft = false, outRect = null;
let grab = null;   // {i, c} while dragging a Q[i,c] component

const STAGES = [
  { key: 'qkt',     label: 'scores = Q·Kᵀ — each cell [i,j] = qᵢ · kⱼ' },
  { key: 'scale',   label: 'scale by 1/√d — keeps the softmax from saturating' },
  { key: 'mask',    label: 'causal mask — query i sees only keys j ≤ i (upper triangle → −∞)' },
  { key: 'softmax', label: 'softmax over each row — attention weights (each visible row sums to 1)' },
  { key: 'output',  label: 'output = weights · V — each row = Σⱼ wᵢⱼ · vⱼ' },
];

// Recompute scores / scaled / weights / output from cur.Q, cur.K, cur.V in place.
// Called by buildData() (fresh) and by resync() after a Q-cell drag edit.
function recompute() {
  const { Q, K, V, N, d } = cur, sq = Math.sqrt(d);
  const raw = cur.raw, scaled = cur.scaled, weights = cur.weights, output = cur.output;
  for (let i = 0; i < N; i++) for (let j = 0; j < N; j++) {
    const s = dot(row(Q, i), row(K, j));
    raw.data[i * N + j] = s; scaled.data[i * N + j] = s / sq;
  }
  for (let i = 0; i < N; i++) {
    const r = new Float32Array(N);
    for (let j = 0; j < N; j++) r[j] = j <= i ? scaled.data[i * N + j] : -Infinity;
    const w = softmax(r);
    for (let j = 0; j < N; j++) weights.data[i * N + j] = w[j];
  }
  for (let i = 0; i < N; i++) for (let c = 0; c < d; c++) {
    let acc = 0;
    for (let j = 0; j <= i; j++) acc += weights.data[i * N + j] * V.data[j * d + c];
    output.data[i * d + c] = acc;
  }
}

function buildData(st) {
  const N = st.N, d = st.d, seed = st.seed | 0;
  const Q = seededRandn(seed, [N, d]), K = seededRandn(seed + 1, [N, d]), V = seededRandn(seed + 2, [N, d]);
  cur = {
    Q, K, V, N, d,
    raw: { data: new Float32Array(N * N), rows: N, cols: N },
    scaled: { data: new Float32Array(N * N), rows: N, cols: N },
    weights: { data: new Float32Array(N * N), rows: N, cols: N },
    output: { data: new Float32Array(N * d), rows: N, cols: d },
  };
  recompute();
  return STAGES.map((s) => ({ ...s }));
}

// Recompute the derived matrices after a Q-cell drag, keeping the transport's
// 5-stage axis valid (the stage list itself never changes -- only the numbers).
function resync(page) {
  recompute();
  const t = page.controls._transport;
  if (t) t._sync();
}

// small input strip [rows x cols] heatmap with a label
function strip(r, M, rect, label, col) {
  r.heatmap(M, { rows: M.rows, cols: M.cols, rect, ramp: ramps.diverging, domain: [-maxAbs(M.data), maxAbs(M.data)] });
  r.grid({ stroke: 'rgba(0,0,0,0.10)' });
  r.label(label, rect.x, rect.y - 5, { color: col || '#586069', font: '11px ui-monospace, monospace' });
}

mount({
  mount: 'body',
  title: 'scaled-dot-attention — softmax(QKᵀ/√d + mask)·V',
  blurb: 'The full single-head attention pipeline over a sequence (prefill). It auto-plays the five stages: QKᵀ → scale → causal mask → softmax → ·V. Drag a Q (query) cell vertically to change a query component and watch its row of scores, the softmax weights, and the output recompute live; hover a score / weight / output cell for its derivation. The N×N matrix is the star; the companion kv-cache page shows the decode (one-query) view.',
  prefer: 'webgl2',
  aspect: '2 / 1',
  autoplay: true,
  controls: (c, page) => {
    c.stepper('N', { label: 'tokens (N)', min: 3, max: 8, value: 6 });
    c.stepper('d', { label: 'head_dim', min: 3, max: 8, value: 4 });
    c.slider('seed', { label: 'seed', min: 0, max: 99, step: 1, value: 2, rebuild: true });
    c.transport({ compute: () => buildData(page.state), speed: 2, loop: true });
  },
  // Direct manipulation: grab a Q-row component and drag vertically to change it.
  // The query is the attention operand -- editing it re-steers what the row attends to.
  onPointer: (page, ev) => {
    if (!cur) return;
    const { N, d } = cur;
    if (ev.type === 'down') {
      grab = null;
      const qh = qRect && cellAt(qRect, N, d, ev.x, ev.y);
      if (qh) grab = { i: qh.r, c: qh.c };
    } else if (ev.type === 'up' || ev.type === 'leave') {
      grab = null;
    } else if (ev.type === 'move' && grab && page.pointer.down) {
      const idx = grab.i * d + grab.c;
      cur.Q.data[idx] = Math.max(-3, Math.min(3, cur.Q.data[idx] - ev.dy * 0.02));  // drag up = larger
      resync(page);
    }
  },
  draw: (page) => {
    const r = page.renderer, ctx = page.ctx, st = page.state;
    if (!cur) return;
    const { Q, K, V, raw, scaled, weights, output, N, d } = cur;
    r.clear('#ffffff');
    const s = page.step();
    const stage = s ? s.key : 'output';
    const si = STAGES.findIndex((x) => x.key === stage);

    // --- input strips Q, K, V (top) ---
    const pad = 14, topY = 56;
    const cellS = Math.max(8, Math.min(16, (page.W * 0.32) / (3 * d + 4)));
    let sx = pad + 14;
    qRect = null;
    for (const [M, nm, cl] of [[Q, 'Q', BLUE], [K, 'K', INK], [V, 'V', ORANGE]]) {
      const rect = { x: sx, y: topY, w: d * cellS, h: N * cellS };
      strip(r, M, rect, `${nm} [${N}×${d}]`, cl);
      if (nm === 'Q') qRect = rect;   // the draggable operand strip
      sx += d * cellS + 26;
    }
    // mark Q as draggable
    if (qRect) r.label('↕ drag Q', qRect.x, qRect.y + N * cellS + 12, { color: BLUE, font: '9px ui-monospace, monospace' });

    // --- the N×N matrix (hero), content per stage ---
    let mat = raw, ramp = ramps.diverging, dom = maxAbs(raw.data), softstage = false;
    if (stage === 'scale' || stage === 'mask') { mat = scaled; dom = maxAbs(scaled.data); }
    if (stage === 'softmax' || stage === 'output') { mat = weights; ramp = ramps.sequential; dom = maxAbs(weights.data); softstage = true; }

    const mTop = topY + N * cellS + 38;
    const cellC = Math.max(14, Math.min(40, Math.min((page.W * 0.42) / N, (page.H - mTop - 40) / N)));
    const mX = pad + 34, mRect = { x: mX, y: mTop, w: N * cellC, h: N * cellC };
    matRect = mRect; matSoft = softstage;   // capture for hit-testing
    r.label(`stage ${si + 1}/5 — ${s ? s.label : STAGES[4].label}`, pad, mTop - 12, { color: INK, font: '12px ui-monospace, monospace' });

    r.heatmap(mat, { rows: N, cols: N, rect: mRect, ramp, domain: softstage ? [0, dom] : [-dom, dom] });
    r.grid({ stroke: 'rgba(0,0,0,0.12)' });
    // row/col index labels
    for (let i = 0; i < N; i++) {
      r.label('q' + i, mX - 22, mTop + i * cellC + cellC * 0.62, { color: '#9aa4ad', font: '10px ui-monospace, monospace' });
      r.label('k' + i, mX + i * cellC + cellC * 0.28, mTop - 1, { color: '#9aa4ad', font: '10px ui-monospace, monospace' });
    }
    // per-cell values (small N)
    if (cellC >= 26) for (let i = 0; i < N; i++) for (let j = 0; j < N; j++) {
      if ((stage === 'mask' || softstage) && j > i) continue;     // masked cells unlabeled
      const v = mat.data[i * N + j];
      r.layout = { x: mRect.x, y: mRect.y, w: mRect.w, h: mRect.h, rows: N, cols: N, cellW: cellC, cellH: cellC };
      r.cell(i, j, { stroke: false, label: softstage ? v.toFixed(2) : v.toFixed(1), labelColor: (softstage ? v > dom * 0.6 : Math.abs(v) > dom * 0.6) ? '#fff' : '#333', font: '9px ui-monospace, monospace' });
    }
    // causal mask treatment on the upper triangle (j>i)
    if (si >= 2) {
      ctx.save();
      for (let i = 0; i < N; i++) for (let j = i + 1; j < N; j++) {
        ctx.fillStyle = softstage ? 'rgba(40,44,52,0.16)' : 'rgba(40,44,52,0.55)';
        ctx.fillRect(mRect.x + j * cellC, mRect.y + i * cellC, cellC, cellC);
      }
      ctx.restore();
    }

    // --- output [N×d] (final stage) ---
    const oX = mX + N * cellC + 46;
    outRect = null;
    if (stage === 'output') {
      const orect = { x: oX, y: mTop, w: d * cellC, h: N * cellC };
      strip(r, output, orect, `output [${N}×${d}]`, ORANGE);
      outRect = orect;
    } else {
      r.label('output', oX, mTop - 12, { color: '#c4ccd3', font: '11px ui-monospace, monospace' });
      ctx.save(); ctx.strokeStyle = '#e3e6ea'; ctx.setLineDash([4, 4]); ctx.strokeRect(oX, mTop, d * cellC, N * cellC); ctx.restore();
    }

    // --- hover-to-inspect: score / weight / output cell -> its derivation ---
    if (page.pointer.over && !grab) {
      const p = page.pointer;
      const sq = Math.sqrt(d);
      let tip = null;
      const qh = qRect && cellAt(qRect, N, d, p.x, p.y);
      const mh = matRect && cellAt(matRect, N, N, p.x, p.y);
      const oh = outRect && cellAt(outRect, N, d, p.x, p.y);
      if (qh) {
        tip = `Q[${qh.r},${qh.c}] = ${Q.data[qh.r * d + qh.c].toFixed(3)}\ndrag ↕ to change the query`;
      } else if (mh && mh.c <= mh.r) {          // visible (unmasked) cell
        const i = mh.r, j = mh.c;
        if (matSoft) {
          // softmax weight cell: its share of the row's attention.
          tip = `w[${i},${j}] = softmax row ${i} over keys 0..${i}\n= ${weights.data[i * N + j].toFixed(3)}  (this key's share)`;
        } else {
          // score cell: q_i·k_j / √d.
          const raws = raw.data[i * N + j];
          tip = `score[${i},${j}] = q${i}·k${j}/√d\n= ${raws.toFixed(3)} / ${sq.toFixed(3)} = ${scaled.data[i * N + j].toFixed(3)}`;
        }
      } else if (oh) {
        // output cell: Σ_j w[i,j]·v[j,c].
        const i = oh.r, c = oh.c;
        const terms = [];
        for (let j = 0; j <= i; j++) terms.push(`${weights.data[i * N + j].toFixed(2)}·${V.data[j * d + c].toFixed(2)}`);
        const shown = terms.length <= 4 ? terms.join(' + ') : terms.slice(0, 3).join(' + ') + ' + … (' + (i + 1) + ' terms)';
        tip = `out[${i},${c}] = Σⱼ w[${i},j]·v[j,${c}]\n= ${shown}\n= ${output.data[i * d + c].toFixed(3)}`;
      }
      if (tip) page.setTip(tip);
    }

    let o = `Attention(Q,K,V) = softmax(QKᵀ/√d + mask)·V    N=${N}  d=${d}  1/√d=${(1 / Math.sqrt(d)).toFixed(3)}    tier:${r.name}\n`;
    o += s ? `stage ${si + 1}/5 — ${s.label}` : '(drag a Q cell to edit · press ▶ or scrub through the 5 stages: QKᵀ → scale → mask → softmax → ·V)';
    page.setReadout(o);
  },
}).then((page) => {
  window.__sdaPage = page;
  const q = new URLSearchParams(location.search);
  const t = page.controls._transport;
  // ?drag=i,c,val sets Q[i,c] to a value (headless stand-in for a vertical drag,
  // since --screenshot has no pointer). e.g. ?drag=0,2,1.5 -> Q[0,2] = 1.5.
  if (q.has('drag') && cur) {
    const [i, c, v] = q.get('drag').split(',');
    cur.Q.data[(+i) * cur.d + (+c)] = +v; resync(page);
  }
  // ?hover=x,y fakes the cursor position (headless stand-in for a real hover)
  // so the tooltip path is verifiable.
  if (q.has('hover')) {
    const [hx, hy] = q.get('hover').split(',').map(Number);
    page.pointer.x = hx; page.pointer.y = hy; page.pointer.over = true;
  }
  // Deterministic frame for capture: pause the transport for any of these hooks.
  if (q.has('step') || q.has('drag') || q.has('hover')) { if (t) t.pause(); }
  // Park on a deterministic stage so the captured frame is stable: ?drag lands
  // on the final output stage (drag flows Q -> scores -> weights -> output);
  // ?hover lands on the scale stage (a live score matrix to inspect). An
  // explicit ?step overrides both.
  if (!q.has('step') && t) {
    if (q.has('drag')) t.seek(4);            // output stage
    else if (q.has('hover')) t.seek(1);      // scale stage (score matrix visible)
  }
  if (q.has('step') && t) t.seek(parseInt(q.get('step'), 10));
  if (q.get('play') === '1' && t) t.play();
  page.redraw();
});
