// flash-attention concept page -- tiled online softmax; the N×N matrix is
// never materialized. Uses the verified framework: layout.mount() + controls +
// a per-tile Transport that runs the online-softmax algorithm.
//
// Interactive per the framework contract (plan/framework.md): this page IS a
// streaming animation, so it AUTO-PLAYS + LOOPS -- the lit tile sweeps across
// K/V and the running max m / sum l / output accumulator O update on load and
// on every loop. Hover a score tile cell for its value + the running stats at
// that step (m, l, rescale=exp(m_old−m_new)); hover an O / output accumulator
// cell for its partial value. DRAG the ◂▸ tile handle horizontally to scrub the
// streaming step by hand -- hand-control the sweep across the K/V tiles.
import { mount } from '../framework/layout.js';
import { ramps, cellAt } from '../framework/render.js';
import { seededRandn } from '../framework/tensor.js';

const INK = '#111', BLUE = '#1f6feb', ORANGE = '#d2691e';
const BQ = 4, D = 4;                       // query-block rows, head_dim (fixed for clarity)
const rgb = (c) => `rgba(${c[0]},${c[1]},${c[2]},1)`;
const maxOf = (a) => { let m = 1e-9; for (let i = 0; i < a.length; i++) if (Math.abs(a[i]) > m) m = Math.abs(a[i]); return m; };

let cur = null;
// Hit-test rects captured in draw() for hover + drag. rScore is the [BQ×N]
// score grid; rM/rL/rRescale are the per-row stat columns; rO is the [BQ×D]
// accumulator; rNorm is the final [BQ×D] output (only on the last tile).
let rScore = null, rM = null, rL = null, rRescale = null, rO = null, rNorm = null;
let scoreCell = 0;                         // px size of one score cell (for hit-test geometry)
let rHandle = null;                        // grabbable tile-sweep handle rect
let dragging = false;                      // true while scrubbing the tile by drag

function buildData(st) {
  const N = st.N, Bk = st.Bk, seed = st.seed | 0, sq = Math.sqrt(D);
  const Q = seededRandn(seed, [BQ, D]), K = seededRandn(seed + 1, [N, D]), V = seededRandn(seed + 2, [N, D]);
  const Sfull = new Float32Array(BQ * N);
  for (let i = 0; i < BQ; i++) for (let kk = 0; kk < N; kk++) { let s = 0; for (let c = 0; c < D; c++) s += Q.data[i * D + c] * K.data[kk * D + c]; Sfull[i * N + kk] = s / sq; }
  const nT = Math.ceil(N / Bk);
  const m = new Float32Array(BQ).fill(-Infinity), l = new Float32Array(BQ), O = new Float32Array(BQ * D);
  const steps = [];
  for (let j = 0; j < nT; j++) {
    const lo = j * Bk, hi = Math.min((j + 1) * Bk, N), tw = hi - lo;
    const P = new Float32Array(BQ * tw), rescale = new Float32Array(BQ), mold = Float32Array.from(m);
    for (let i = 0; i < BQ; i++) {
      let mloc = -Infinity; for (let kk = lo; kk < hi; kk++) mloc = Math.max(mloc, Sfull[i * N + kk]);
      const mnew = Math.max(m[i], mloc);
      const rs = isFinite(m[i]) ? Math.exp(m[i] - mnew) : 0;
      rescale[i] = isFinite(m[i]) ? rs : 1;
      let sumP = 0; const add = new Float32Array(D);
      for (let t = 0; t < tw; t++) { const kk = lo + t, p = Math.exp(Sfull[i * N + kk] - mnew); P[i * tw + t] = p; sumP += p; for (let c = 0; c < D; c++) add[c] += p * V.data[kk * D + c]; }
      l[i] = rs * l[i] + sumP;
      for (let c = 0; c < D; c++) O[i * D + c] = rs * O[i * D + c] + add[c];
      m[i] = mnew;
    }
    const isLast = j === nT - 1;
    let norm = null;
    if (isLast) { norm = new Float32Array(BQ * D); for (let i = 0; i < BQ; i++) for (let c = 0; c < D; c++) norm[i * D + c] = O[i * D + c] / l[i]; }
    steps.push({ j, lo, hi, tw, P: Float32Array.from(P), m: Float32Array.from(m), mold, l: Float32Array.from(l), O: Float32Array.from(O), rescale: Float32Array.from(rescale), norm, isLast, nT, Bk });
  }
  cur = { Sfull, N, nT, Bk };
  return steps;
}

mount({
  mount: 'body',
  title: 'flash-attention — tiled online softmax',
  blurb: 'Exact attention without materializing the N×N matrix: stream K/V in tiles, keep a running max m, sum l, and output O, rescaling prior contributions when the max grows. Only one tile lives in memory at a time → O(N), not O(N²). It auto-plays: the lit tile sweeps across the keys while m, l, and O update. Hover a score tile cell for its value + the running stats at that step; hover an O/output cell for its partial value; DRAG the ◂▸ handle to scrub the tile sweep by hand.',
  prefer: 'webgl2',   // T3 page; this fleet's chromium WebGPU hangs. render.js now falls back after a 1.5s timeout, but we prefer webgl2 directly to skip that delay
  aspect: '2 / 1',
  challenges: [
    { goal: 'Stream all the way through — reach the LAST tile (final output).', hint: 'drag the ◂▸ handle to the right end (or let it play to the last tile).', check: (api) => ({ solved: (api.probe.j ?? -1) === (api.probe.nT ?? 0) - 1, detail: `tile ${(api.probe.j ?? 0) + 1} / ${api.probe.nT ?? 0}` }) },
    { goal: 'Minimize on-chip memory — use the smallest tile size (2).', hint: 'set the tile-size control to 2 (more, smaller tiles → O(tile) memory).', check: (api) => ({ solved: (api.state.Bk | 0) === 2, detail: `tile size = ${api.state.Bk}` }) },
  ],
  autoplay: true,
  controls: (c, page) => {
    c.stepper('N', { label: 'keys (N)', min: 8, max: 16, value: 12 });
    c.stepper('Bk', { label: 'tile size', min: 2, max: 4, value: 3 });
    c.slider('seed', { label: 'seed', min: 0, max: 99, step: 1, value: 3, rebuild: true });
    c.transport({ compute: () => buildData(page.state), speed: 1.2, loop: true });
  },
  // Direct manipulation: grab the ◂▸ tile handle (or click anywhere on the
  // score grid) and drag horizontally to scrub which tile is being streamed.
  // The key x position maps onto the tile index -> the step transport.
  onPointer: (page, ev) => {
    const t = page.controls._transport;
    if (!t || !cur) return;
    const { N, nT, Bk } = cur;
    // Map a pointer x within the score grid's column span to a tile index 0..nT-1.
    const tileAtX = (x) => {
      if (!rScore || !scoreCell) return 0;
      const kk = Math.floor((x - rScore.x) / scoreCell);
      return Math.max(0, Math.min(nT - 1, Math.floor(kk / Bk)));
    };
    const onHandle = (x, y) => rHandle && x >= rHandle.x && x <= rHandle.x + rHandle.w && y >= rHandle.y && y <= rHandle.y + rHandle.h;
    const onGrid = (x, y) => rScore && x >= rScore.x && x <= rScore.x + rScore.w && y >= rScore.y && y <= rScore.y + rScore.h;
    if (ev.type === 'down') {
      if (onHandle(ev.x, ev.y) || onGrid(ev.x, ev.y)) {
        dragging = true;
        t.pause();
        t.seek(tileAtX(ev.x));
      }
    } else if (ev.type === 'up' || ev.type === 'leave') {
      dragging = false;
    } else if (ev.type === 'move' && dragging && page.pointer.down) {
      t.seek(tileAtX(ev.x));
    }
  },
  draw: (page) => {
    const r = page.renderer, ctx = page.ctx, st = page.state;
    if (!cur) return;
    const { Sfull, N, nT, Bk } = cur;
    r.clear('#ffffff');
    const s = page.step();
    const j = s ? s.j : nT - 1, lo = s ? s.lo : (nT - 1) * Bk, hi = s ? s.hi : N, tw = hi - lo;
    page.probe = { j, nT };

    const pad = 16, topY = 70;
    const cellM = Math.max(12, Math.min(30, (page.W - 2 * pad - 230) / (N + 3 + 2 * D)));
    const mx = pad + 30, my = topY;
    const maxP = s ? maxOf(s.P) : 1;
    scoreCell = cellM;
    rScore = { x: mx, y: my, w: N * cellM, h: BQ * cellM };

    // --- score matrix [BQ × N]: ghosted, only the current tile lit ---
    r.label('scores [Bq×N] — ghosted; only the current tile exists in memory', pad, topY - 14, { color: INK, font: '12px ui-monospace, monospace' });
    for (let i = 0; i < BQ; i++) for (let kk = 0; kk < N; kk++) {
      const x = mx + kk * cellM, y = my + i * cellM;
      if (kk >= lo && kk < hi && s) { const p = s.P[i * tw + (kk - lo)]; ctx.fillStyle = rgb(ramps.sequential(p / maxP)); }
      else if (kk < lo) ctx.fillStyle = '#e7eaee';       // processed + discarded
      else ctx.fillStyle = '#f6f7f9';                    // future (not yet computed)
      ctx.fillRect(x, y, cellM - 1, cellM - 1);
    }
    ctx.save();
    ctx.strokeStyle = '#c4ccd3'; ctx.lineWidth = 1;
    for (let t = 0; t <= nT; t++) { const x = mx + Math.min(t * Bk, N) * cellM; ctx.beginPath(); ctx.moveTo(x, my); ctx.lineTo(x, my + BQ * cellM); ctx.stroke(); }
    ctx.strokeStyle = BLUE; ctx.lineWidth = 2.5; ctx.strokeRect(mx + lo * cellM, my - 1, tw * cellM, BQ * cellM + 2);
    ctx.fillStyle = BLUE; ctx.font = '10px ui-monospace, monospace'; ctx.textAlign = 'center'; ctx.fillText(`tile ${j + 1}/${nT}`, mx + lo * cellM + tw * cellM / 2, my + BQ * cellM + 13);
    if (hi < N) { ctx.fillStyle = '#9aa4ad'; ctx.fillText('not materialized →', mx + (hi + (N - hi) / 2) * cellM, my + BQ * cellM + 13); }
    ctx.restore();

    // --- drag handle: a grabbable bar at the right edge of the lit tile,
    // spanning the score grid. Drag it (or the grid) to scrub the tile sweep. ---
    const hX = mx + hi * cellM;
    rHandle = { x: hX - 5, y: my, w: 10, h: BQ * cellM };
    ctx.save();
    ctx.strokeStyle = dragging ? ORANGE : 'rgba(31,111,235,0.9)';
    ctx.lineWidth = dragging ? 3 : 2;
    ctx.beginPath(); ctx.moveTo(hX, my - 5); ctx.lineTo(hX, my + BQ * cellM + 5); ctx.stroke();
    ctx.fillStyle = ctx.strokeStyle;
    ctx.font = '10px ui-monospace, monospace'; ctx.textAlign = 'center';
    ctx.fillText('◂▸', hX, my - 9);
    ctx.restore();

    // --- running state panels: m, l, rescale, O, output ---
    let gx = mx + N * cellM + 24;
    rM = rL = rRescale = rO = rNorm = null;
    const col = (vals, label, ramp, dom, showVal, cols) => {
      const cc = cols || 1, w = cc * cellM;
      const rect = { x: gx, y: my, w, h: BQ * cellM };
      r.heatmap(vals, { rows: BQ, cols: cc, rect, ramp, domain: dom });
      r.grid({ stroke: 'rgba(0,0,0,0.10)' });
      r.label(label, gx, my - 5, { color: '#586069', font: '10px ui-monospace, monospace' });
      if (showVal) { ctx.save(); ctx.font = '9px ui-monospace, monospace'; ctx.textAlign = 'center'; ctx.textBaseline = 'middle'; ctx.fillStyle = '#1a1d21'; for (let i = 0; i < BQ; i++) ctx.fillText(vals[i].toFixed(2), gx + cellM / 2, my + i * cellM + cellM / 2); ctx.restore(); }
      gx += w + 14;
      return rect;
    };
    if (s) {
      rM = col(s.m, 'm (max)', ramps.diverging, [-maxOf(s.m), maxOf(s.m)], true);
      rL = col(s.l, 'l (sum)', ramps.sequential, [0, maxOf(s.l)], true);
      rRescale = col(s.rescale, 'rescale', ramps.sequential, [0, 1], true);
      rO = col(s.O, 'O (accum)', ramps.diverging, [-maxOf(s.O), maxOf(s.O)], false, D);
      if (s.isLast && s.norm) { ctx.save(); ctx.fillStyle = BLUE; ctx.font = '11px ui-monospace, monospace'; ctx.textAlign = 'center'; ctx.fillText('= O/l →', gx - 7, my + BQ * cellM / 2); ctx.restore(); gx += 6; rNorm = col(s.norm, 'output', ramps.diverging, [-maxOf(s.norm), maxOf(s.norm)], false, D); }
    }

    // --- hover-to-inspect: a score tile cell -> value + running stats at this
    // step; an O / output accumulator cell -> its partial value ---
    if (s && page.pointer.over && !dragging) {
      const p = page.pointer;
      let tip = null;
      const sh = cellAt(rScore, BQ, N, p.x, p.y);
      const oh = rO && cellAt(rO, BQ, D, p.x, p.y);
      const nh = rNorm && cellAt(rNorm, BQ, D, p.x, p.y);
      if (sh) {
        const i = sh.r, kk = sh.c;
        const sval = Sfull[i * N + kk].toFixed(3);
        const rs = s.rescale[i].toFixed(3), mnew = s.m[i].toFixed(3), li = s.l[i].toFixed(3);
        if (kk >= lo && kk < hi) {
          const pv = s.P[i * tw + (kk - lo)].toFixed(3);
          tip = `tile (qi=${i}, kj=${kk})  IN current tile ${j + 1}/${nT}\n`
              + `score S=${sval};  p=exp(S−m)=${pv}\n`
              + `running max m=${mnew}, sum l=${li}\n`
              + `rescale prev acc by exp(m_old−m_new)=${rs}`;
        } else if (kk < lo) {
          tip = `tile (qi=${i}, kj=${kk})  ALREADY streamed (discarded)\n`
              + `score S=${sval} was folded into m=${mnew}, l=${li}\n`
              + `(its contribution survives only in the running O accumulator)`;
        } else {
          const tj = Math.floor(kk / Bk);
          tip = `tile (qi=${i}, kj=${kk})  NOT materialized yet\n`
              + `streamed at tile ${tj + 1}/${nT};  current m=${mnew}, l=${li}`;
        }
      } else if (oh) {
        const i = oh.r, c = oh.c;
        tip = `O accumulator [qi=${i}, dim ${c}] = ${s.O[i * D + c].toFixed(3)}\n`
            + `partial Σ exp(S−m)·V over tiles 1..${j + 1};  l=${s.l[i].toFixed(3)}\n`
            + `final output = O/l${s.isLast ? ` = ${(s.O[i * D + c] / s.l[i]).toFixed(3)}` : ' (after the last tile)'}`;
      } else if (nh) {
        const i = nh.r, c = nh.c;
        tip = `output [qi=${i}, dim ${c}] = O/l = ${s.norm[i * D + c].toFixed(3)}\n`
            + `= ${s.O[i * D + c].toFixed(3)} / ${s.l[i].toFixed(3)}  (normalized at the last tile)`;
      }
      if (tip) page.setTip(tip);
    }

    let o = `FlashAttention — online softmax over K/V tiles (Bk=${Bk}); the N×N matrix is never materialized → O(N) memory, not O(N²).    tier:${r.name}\n`;
    o += s ? `tile ${j + 1}/${nT} keys [${lo}:${hi}):  m←max(m, rowmax S);  rescale=exp(m_old−m_new);  l←rescale·l+Σe^{S−m};  O←rescale·O+Σe^{S−m}·V${s.isLast ? ';  output = O / l' : ''}`
           : '(plays on load; hover a tile cell for its running stats, or DRAG the ◂▸ handle to scrub the sweep across the K/V tiles)';
    page.setReadout(o);
  },
}).then((page) => {
  window.__faPage = page;
  const q = new URLSearchParams(location.search);
  const t = page.controls._transport;
  // ?pos=N / ?step=N set the streaming tile/step (headless stand-in for a drag
  // of the ◂▸ handle, since --screenshot has no pointer). Both names accepted;
  // N is the 0-based tile index (0 = first K/V tile).
  const posKey = q.has('pos') ? 'pos' : (q.has('step') ? 'step' : null);
  if (posKey && t) t.seek(parseInt(q.get(posKey), 10));
  // ?hover=x,y fakes the cursor position (headless stand-in for a real hover)
  // so the running-stats tooltip path is verifiable.
  if (q.has('hover')) {
    const [hx, hy] = q.get('hover').split(',').map(Number);
    page.pointer.x = hx; page.pointer.y = hy; page.pointer.over = true;
  }
  // ?hover with no explicit step/pos parks on a deterministic mid-sweep tile so
  // there IS a running-stats step under the cursor (the score grid + stat panels
  // only exist when a tile is current; index −1 draws nothing to inspect).
  if (q.has('hover') && !posKey && t && t.steps.length) t.seek(Math.min(2, t.steps.length - 1));
  // Deterministic frame for capture: pause the transport for any of these hooks.
  if (q.has('step') || q.has('pos') || q.has('hover')) { if (t) t.pause(); }
  if (q.get('play') === '1' && t) t.play();
  page.redraw();
});
