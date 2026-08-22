// hyper-connections concept page -- the residual highway widened to n parallel
// streams with learned connection weights.
//
// One residual stream means every block reads the same vector and adds into the
// same vector. Hyper-connections keep n of them side by side and let the network
// LEARN, per block: which streams to read from (the read weights), which streams
// to write into (the write weights), and how the streams mix with each other
// between blocks (the width connection). The classic residual is the special
// case n = 1 with every weight pinned to 1 -- and this page carries that case as
// live arithmetic, so the equivalence is a number on screen, not a claim.
//
// Source: Zhu et al., "Hyper-Connections", https://arxiv.org/abs/2409.19606
//
// The neighbouring pages own the pieces this one does not re-teach: the single
// highway itself is `residual-stream`, and what happens INSIDE a block is
// `transformer-block`. This page is about the wiring between blocks.
//
// Interactive per the shared render framework's contract: transport steps block
// by block (autoplay + loop); DIRECT MANIPULATION -- drag any connection handle
// vertically to change that weight and watch the block's contribution and every
// level downstream recompute; drag the stream-count slider to widen or narrow
// the highway; a preset switch walks classic residual -> learned -> a collapsed
// configuration; hover any handle, stream node or bar for its value and what it
// does.
import { mount } from '../framework/layout.js';
import { seededRandn, seededRand } from '../framework/tensor.js';
import { T, alphaOf, mixColor } from '../framework/theme.js';

const PRESETS = [
  { value: 'classic', label: 'classic residual (n=1)' },
  { value: 'wide-id', label: 'widened, identity weights' },
  { value: 'learned', label: 'learned (streams diverge)' },
  { value: 'strong', label: 'strong-write (post-norm side)' },
  { value: 'collapsed', label: 'collapsed (late blocks silent)' },
  { value: 'custom', label: 'custom (dragged)' },
];

const rmsOf = (v) => { let s = 0; for (let i = 0; i < v.length; i++) s += v[i] * v[i]; return Math.sqrt(s / v.length); };
const clampW = (v) => Math.max(-2, Math.min(2, v));
const f3 = (v) => (Math.abs(v) < 1e-12 ? '0.000' : v.toFixed(3));

// Shared between buildData() (seeds the weights + the sublayer matrices),
// draw() (runs the forward pass, renders, and captures the handle rects) and
// onPointer() (hit-tests + edits a weight). The transport rebuild runs
// buildData() before the matching draw, so `cur` is fresh. Drag edits mutate
// cur.A / cur.Bw / cur.M in place and the whole stack is recomputed from them
// every draw, so one weight change propagates to every level below it at once.
let cur = null;
let handles = null;      // [{x, y, rad, kind:'A'|'B'|'M', b, i, j}]
let nodeRects = null;    // [{x, y, w, h, d, m}] stream nodes, for hover
let barRects = null;     // [{x, y, w, h, b}] per-block contribution bars
let grab = null;         // the handle being dragged

// ---------------------------------------------------------------------------
// The block itself. Deliberately simple and deliberately SHARED: the classic
// single-stream reference path below calls this exact function with this exact
// matrix, so any difference between the two readouts comes from the WIRING and
// from nothing else. Pre-norm read, SiLU, one dense projection -- a stand-in
// for attention or an MLP, whose internals are the `transformer-block` page.
function sublayer(x, Wb, D) {
  let ms = 0; for (let j = 0; j < D; j++) ms += x[j] * x[j];
  const inv = 1 / Math.sqrt(ms / D + 1e-6);
  const g = new Float32Array(D);
  for (let j = 0; j < D; j++) { const u = x[j] * inv; g[j] = u / (1 + Math.exp(-u)); }
  const y = new Float32Array(D);
  for (let i = 0; i < D; i++) { let s = 0; for (let j = 0; j < D; j++) s += Wb[i * D + j] * g[j]; y[i] = s * 0.6; }
  return y;
}

function eye(n, diag) { return Array.from({ length: n }, (_, k) => Array.from({ length: n }, (_, m) => (k === m ? diag : 0))); }

// Connection weights per preset. Three groups per block b:
//   A[b][m]     read weight  -- how much of stream m the block reads
//   Bw[b][k]    write weight -- how much of the block output lands in stream k
//   M[b][k][m]  width connection -- how much of stream m carries into stream k
// Classic residual is n=1 with A=[1], Bw=[1], M=[[1]].
function presetWeights(preset, n, L, seed) {
  // NOTE: a one-element shape returns the raw Float32Array, not a {data} matrix.
  const noise = seededRand(seed * 31 + 7, [L * n * (n + 2)]);
  let p = 0; const nx = () => noise[(p++) % noise.length];
  const A = [], Bw = [], M = [];
  for (let b = 0; b < L; b++) {
    let a, w, m;
    if (preset === 'classic' || preset === 'wide-id') {
      // Read the average of the streams, write the whole output into every
      // stream, carry each stream straight through. With n=1 that IS the plain
      // residual; with n>1 the streams stay exact copies of each other, which
      // is the point -- widening alone changes nothing until the weights move.
      a = new Array(n).fill(1 / n); w = new Array(n).fill(1); m = eye(n, 1);
    } else if (preset === 'strong') {
      // The identity path is damped (width diagonal < 1) while the write stays
      // full strength, so each block's own output keeps a large share of the
      // stream all the way down -- the post-norm side of the tradeoff: strong
      // late blocks, a weaker carried-through signal.
      a = new Array(n).fill(1 / n); w = new Array(n).fill(1); m = eye(n, 0.72);
    } else if (preset === 'collapsed') {
      // The write weights decay with depth while the identity path stays at 1,
      // so the stream grows and later blocks add a vanishing share of it --
      // representation collapse, drawn.
      const f = Math.pow(0.42, b + 1);
      a = new Array(n).fill(1 / n); w = new Array(n).fill(f); m = eye(n, 1);
    } else {
      // 'learned' / 'custom' base: each block leans on a different stream to
      // read from and a different one to write into, and the streams exchange a
      // little content between blocks. This is what a trained connection matrix
      // is free to do and a single highway structurally cannot.
      const rd = b % n, wr = (b + 1) % n;
      a = Array.from({ length: n }, (_, k) => +( (k === rd ? 0.72 : 0.10) + 0.18 * nx() ).toFixed(3));
      w = Array.from({ length: n }, (_, k) => +( (k === wr ? 0.95 : 0.22) + 0.30 * nx() ).toFixed(3));
      // Off-diagonal only to the NEXT stream, and no wrap-around: a k = n-1 → 0
      // edge would span the whole width of the picture and cross every lane.
      m = Array.from({ length: n }, (_, k) => Array.from({ length: n }, (_, mm) => (
        k === mm ? +(0.86 + 0.12 * nx()).toFixed(3)
          : (mm === k + 1 ? +(0.10 + 0.16 * nx()).toFixed(3) : 0)
      )));
    }
    A.push(a); Bw.push(w); M.push(m);
  }
  return { A, Bw, M };
}

function buildData(st, page) {
  let n = Math.max(1, Math.min(5, st.n | 0));
  if (st.preset === 'classic' && n !== 1) { n = 1; if (page) page.controls.set('n', 1, { silent: true }); }
  const L = Math.max(2, Math.min(6, st.L | 0));
  const D = Math.max(4, Math.min(10, st.D | 0));
  const seed = st.seed | 0;
  const { A, Bw, M } = presetWeights(st.preset, n, L, seed);
  const W = [];
  for (let b = 0; b < L; b++) {
    const R = seededRandn(seed + 101 * (b + 1), [D, D], { std: 1 / Math.sqrt(D) });
    W.push(R.data);
  }
  const x0 = seededRandn(seed + 5, [D], { std: 1 });   // vector shape -> raw Float32Array
  cur = { n, L, D, seed, A, Bw, M, W, x0 };
  return Array.from({ length: L }, (_, b) => ({ b, label: `block ${b}: read a mixture of the ${n} stream${n > 1 ? 's' : ''} → transform → write back → mix streams` }));
}

// Keep the transport axis in sync with L after a rebuildless edit.
function resync(page) {
  const t = page.controls._transport;
  if (!t || !cur) return;
  t.steps = Array.from({ length: cur.L }, (_, b) => ({ b, label: `block ${b}: read a mixture of the ${cur.n} stream${cur.n > 1 ? 's' : ''} → transform → write back → mix streams` }));
  t.scrub.max = Math.max(0, t.steps.length - 1);
  if (t.index > t.steps.length - 1) t.index = t.steps.length - 1;
  t._sync();
}

// ---------------------------------------------------------------------------
// The whole stack, run for real, twice: once through the n hyper-connected
// streams and once through a plain single residual stream using the SAME
// sublayer matrices. Everything the page reports is read off this.
function forward(c) {
  const { n, L, D, A, Bw, M, W, x0 } = c;

  // Streams start as n copies of the embedding; the readout averages them, so
  // for n = 1 both ends of the widening are the identity.
  let Hs = Array.from({ length: n }, () => Float32Array.from(x0));
  const levels = [Hs.map((h) => Float32Array.from(h))];
  const readouts = [], blocks = [];
  const reduce = (S) => { const o = new Float32Array(D); for (let m = 0; m < n; m++) for (let j = 0; j < D; j++) o[j] += S[m][j]; for (let j = 0; j < D; j++) o[j] /= n; return o; };
  readouts.push(reduce(Hs));

  for (let b = 0; b < L; b++) {
    const x = new Float32Array(D);
    for (let m = 0; m < n; m++) { const a = A[b][m], h = Hs[m]; for (let j = 0; j < D; j++) x[j] += a * h[j]; }
    const y = sublayer(x, W[b], D);
    const Hn = Array.from({ length: n }, () => new Float32Array(D));
    for (let k = 0; k < n; k++) {
      const out = Hn[k], bw = Bw[b][k];
      for (let j = 0; j < D; j++) out[j] = bw * y[j];
      for (let m = 0; m < n; m++) { const w = M[b][k][m]; if (!w) continue; const h = Hs[m]; for (let j = 0; j < D; j++) out[j] += w * h[j]; }
    }
    Hs = Hn;
    levels.push(Hs.map((h) => Float32Array.from(h)));
    const ro = reduce(Hs); readouts.push(ro);
    // What this block actually put into the readout: the mean write weight
    // times its output. c = that magnitude as a share of the readout's.
    let mw = 0; for (let k = 0; k < n; k++) mw += Bw[b][k]; mw /= n;
    const add = Float32Array.from(y, (v) => mw * v);
    blocks.push({ x, y, add, rIn: rmsOf(x), rY: rmsOf(y), c: rmsOf(add) / (rmsOf(ro) + 1e-9) });
  }

  // Plain single-stream residual reference, same sublayer, same matrices.
  let r = Float32Array.from(x0);
  const plain = [Float32Array.from(r)];
  for (let b = 0; b < L; b++) { const y = sublayer(r, W[b], D); for (let j = 0; j < D; j++) r[j] = r[j] + y[j]; plain.push(Float32Array.from(r)); }

  const hcOut = readouts[L], plOut = plain[L];
  let diff = 0; for (let j = 0; j < D; j++) diff = Math.max(diff, Math.abs(hcOut[j] - plOut[j]));

  const cs = blocks.map((x2) => x2.c);
  const s1 = cs.reduce((a, v) => a + v, 0), s2 = cs.reduce((a, v) => a + v * v, 0);
  const eff = s2 > 1e-12 ? (s1 * s1) / s2 : 0;

  return { levels, readouts, plain, blocks, diff, eff, hcOut, plOut };
}

// ---------------------------------------------------------------------------
function edge(ctx, x0, y0, x1, y1, v, hot) {
  const a = Math.min(1, Math.abs(v));
  ctx.save();
  ctx.strokeStyle = alphaOf(v < 0 ? T.bad : T.accent, (hot ? 0.40 : 0.14) + 0.55 * a);
  ctx.lineWidth = (hot ? 0.8 : 0.5) + 3.4 * a;
  ctx.beginPath(); ctx.moveTo(x0, y0);
  ctx.bezierCurveTo(x0, (y0 + y1) / 2, x1, (y0 + y1) / 2, x1, y1);
  ctx.stroke(); ctx.restore();
}

function handle(ctx, h, hot, v) {
  ctx.save();
  ctx.fillStyle = hot ? (v < 0 ? T.bad : T.accent) : alphaOf(v < 0 ? T.bad : T.accent, 0.5);
  ctx.strokeStyle = T.n0; ctx.lineWidth = 1.5;
  ctx.beginPath(); ctx.arc(h.x, h.y, h.rad, 0, Math.PI * 2); ctx.fill(); ctx.stroke();
  if (hot) { ctx.fillStyle = T.n0; ctx.font = '8px ui-monospace, monospace'; ctx.textAlign = 'center'; ctx.textBaseline = 'middle'; ctx.fillText(Math.abs(v) >= 1 ? v.toFixed(1) : v.toFixed(2).replace(/^(-?)0/, '$1'), h.x, h.y + 0.5); }
  ctx.restore();
}

function weightAt(h) {
  if (!cur || !h) return 0;
  if (h.kind === 'A') return cur.A[h.b][h.i];
  if (h.kind === 'B') return cur.Bw[h.b][h.i];
  return cur.M[h.b][h.i][h.j];
}
function setWeight(h, v) {
  if (!cur || !h) return;
  if (h.kind === 'A') cur.A[h.b][h.i] = v;
  else if (h.kind === 'B') cur.Bw[h.b][h.i] = v;
  else cur.M[h.b][h.i][h.j] = v;
}
function hitHandle(x, y) {
  if (!handles) return null;
  let best = null, bd = 1e9;
  for (const h of handles) { const d = Math.hypot(x - h.x, y - h.y); if (d < h.rad + 5 && d < bd) { bd = d; best = h; } }
  return best;
}
const KIND_WORD = {
  A: 'read weight (depth connection, in)',
  B: 'write weight (depth connection, out)',
  M: 'width connection (stream → stream)',
};

mount({
  mount: 'body',
  title: 'hyper-connections — the residual highway, widened',
  blurb: 'A transformer has ONE residual stream: every block reads it, adds its output, writes it back. Hyper-connections keep n streams side by side and learn, per block, which streams to read from, which to write into, and how the streams mix in between — so the network chooses its own position between the pre-norm and post-norm extremes instead of the architect fixing one. Drag any connection handle ↕ to change that weight and watch the block\'s contribution and every level below it recompute; drag the stream count to widen the highway; step block by block. Set the preset to "classic residual (n=1)" and the readout must match the plain residual EXACTLY — the panel on the right reports max|Δ| so you can check it.',
  prefer: 'canvas2d',
  aspect: '2 / 1',
  autoplay: true,
  controls: (c, page) => {
    c.select('preset', { label: 'preset', value: 'learned', options: PRESETS, rebuild: true });
    c.slider('n', { label: 'streams (n) — drag', min: 1, max: 5, step: 1, value: 4, rebuild: true });
    c.stepper('L', { label: 'blocks (L)', min: 2, max: 6, value: 4 });
    c.stepper('D', { label: 'features (D)', min: 4, max: 10, value: 8 });
    c.slider('seed', { label: 'seed', min: 0, max: 99, step: 1, value: 7, rebuild: true });
    c.transport({ compute: () => buildData(page.state, page), speed: 1.4, loop: true });
  },

  // Direct manipulation: grab a connection handle and drag vertically. The
  // weight moves, the block's contribution moves with it, and every level below
  // is re-run from the edited weight -- that is the whole mechanism in one
  // gesture. Editing a weight puts the preset into 'custom' so a rebuild does
  // not silently discard the edit.
  onPointer: (page, ev) => {
    if (!cur) return;
    if (ev.type === 'down') {
      grab = hitHandle(ev.x, ev.y);
      if (grab && page.state.preset !== 'custom') page.controls.set('preset', 'custom', { silent: true });
    } else if (ev.type === 'up' || ev.type === 'leave') {
      grab = null;
    } else if (ev.type === 'move' && grab && page.pointer.down) {
      setWeight(grab, clampW(+(weightAt(grab) - ev.dy * 0.012).toFixed(4)));   // drag up = larger
      resync(page);
    }
  },

  draw: (page) => {
    const r = page.renderer, ctx = page.ctx, st = page.state;
    if (!cur) return;
    r.clear(T.n0);
    const { n, L, D } = cur;
    const F = forward(cur);
    const s = page.step(), cb = s ? s.b : -1;

    const pad = 14, topY = 74;
    const laneW = Math.max(30, Math.min(60, (page.W * 0.30) / n));
    const PX = pad + 24;                       // plain-residual reference lane
    const LX = pad + 66;                       // first hyper stream lane
    const bw = 60, bh = 20;
    const BX = LX + n * laneW + 26;
    // The right panel never starts before 330: at n = 1 the lane block is narrow
    // enough that the two column headings would otherwise overlap.
    const RX = Math.min(page.W - 176, Math.max(BX + bw + 30, 330));
    // The bottom reserve is what keeps the identity-check card on screen: the
    // lane diagram must finish high enough that the card below it is visible
    // without scrolling, which is the whole point of putting the check there.
    const rowH = Math.max(30, Math.min(58, (page.H - topY - 112) / (L + 1)));
    const levelY = (d) => topY + d * rowH;
    const laneX = (m) => LX + m * laneW + laneW / 2;
    const nodeW = Math.min(laneW * 0.66, 42);

    const rmsAll = [];
    for (let d = 0; d <= L; d++) for (let m = 0; m < n; m++) rmsAll.push(rmsOf(F.levels[d][m]));
    for (let d = 0; d <= L; d++) rmsAll.push(rmsOf(F.plain[d]));
    const rMax = Math.max(1e-6, ...rmsAll);

    handles = []; nodeRects = []; barRects = [];

    // ---- headings -----------------------------------------------------------
    // Fall back to a shorter heading rather than let it run under the right
    // panel's own heading, which sits on the same baseline.
    ctx.save(); ctx.font = '12px ui-monospace, monospace';
    let head = `${n} parallel residual stream${n > 1 ? 's' : ''} through ${L} blocks`;
    if (pad + ctx.measureText(head).width > RX - 10) head = `${n} stream${n > 1 ? 's' : ''} × ${L} blocks`;
    ctx.restore();
    r.label(head, pad, topY - 30, { color: T.n14, font: '12px ui-monospace, monospace' });
    r.label('plain', PX, topY - 14, { color: T.n10, font: '9px ui-monospace, monospace', align: 'center' });
    for (let m = 0; m < n; m++) r.label('s' + m, laneX(m), topY - 14, { color: T.n11, font: '9px ui-monospace, monospace', align: 'center' });

    // ---- plain single-stream reference lane ---------------------------------
    ctx.save();
    ctx.strokeStyle = alphaOf(T.n9, 0.5); ctx.lineWidth = 1; ctx.setLineDash([3, 3]);
    ctx.beginPath(); ctx.moveTo(PX, levelY(0)); ctx.lineTo(PX, levelY(L)); ctx.stroke();
    ctx.restore();
    for (let d = 0; d <= L; d++) {
      const rr = rmsOf(F.plain[d]);
      const rect = { x: PX - nodeW / 2, y: levelY(d) - 7, w: nodeW, h: 14 };
      ctx.save();
      ctx.fillStyle = mixColor(T.n2, T.n9, Math.min(1, rr / rMax));
      ctx.strokeStyle = alphaOf(T.n11, 0.6); ctx.lineWidth = 1;
      ctx.fillRect(rect.x, rect.y, rect.w, rect.h); ctx.strokeRect(rect.x, rect.y, rect.w, rect.h);
      ctx.fillStyle = T.n13; ctx.font = '8px ui-monospace, monospace'; ctx.textAlign = 'center'; ctx.textBaseline = 'middle';
      ctx.fillText(rr.toFixed(2), rect.x + rect.w / 2, rect.y + 7);
      ctx.restore();
      nodeRects.push({ ...rect, d, m: -1 });
    }

    // ---- stream lanes, block by block ---------------------------------------
    for (let m = 0; m < n; m++) {
      ctx.save(); ctx.strokeStyle = alphaOf(T.n8, 0.45); ctx.lineWidth = 1;
      ctx.beginPath(); ctx.moveTo(laneX(m), levelY(0)); ctx.lineTo(laneX(m), levelY(L)); ctx.stroke(); ctx.restore();
    }

    for (let b = 0; b < L; b++) {
      const cy = levelY(b) + rowH / 2, hot = b === cb;
      // width connections m -> k
      for (let k = 0; k < n; k++) for (let m = 0; m < n; m++) {
        const v = cur.M[b][k][m]; if (Math.abs(v) < 0.005) continue;
        edge(ctx, laneX(m), levelY(b) + 7, laneX(k), levelY(b + 1) - 7, v, hot);
        if (hot) handles.push({ x: (laneX(m) + laneX(k)) / 2, y: cy + rowH * 0.16, rad: 6, kind: 'M', b, i: k, j: m });
      }
      // read edges: stream m -> block
      for (let m = 0; m < n; m++) {
        const v = cur.A[b][m]; if (Math.abs(v) < 0.005 && !hot) continue;
        edge(ctx, laneX(m), levelY(b) + 7, BX, cy, v, hot);
        handles.push({ x: (laneX(m) + BX) / 2, y: cy - rowH * 0.14, rad: hot ? 6 : 3.6, kind: 'A', b, i: m, j: -1 });
      }
      // write edges: block -> stream k
      for (let k = 0; k < n; k++) {
        const v = cur.Bw[b][k]; if (Math.abs(v) < 0.005 && !hot) continue;
        edge(ctx, BX + bw, cy, laneX(k), levelY(b + 1) - 7, v, hot);
        handles.push({ x: (BX + bw + laneX(k)) / 2, y: cy + rowH * 0.14, rad: hot ? 6 : 3.6, kind: 'B', b, i: k, j: -1 });
      }
      // the block box
      ctx.save();
      ctx.fillStyle = hot ? alphaOf(T.accent, 0.18) : T.n2;
      ctx.strokeStyle = hot ? T.accent : T.n7; ctx.lineWidth = hot ? 2.2 : 1.2;
      ctx.fillRect(BX, cy - bh / 2, bw, bh); ctx.strokeRect(BX, cy - bh / 2, bw, bh);
      ctx.fillStyle = hot ? T.accent : T.n12; ctx.font = '10px ui-monospace, monospace';
      ctx.textAlign = 'center'; ctx.textBaseline = 'middle';
      ctx.fillText('block ' + b, BX + bw / 2, cy);
      ctx.restore();
    }

    // stream nodes on top of the edges
    for (let d = 0; d <= L; d++) for (let m = 0; m < n; m++) {
      const rr = rmsOf(F.levels[d][m]);
      const rect = { x: laneX(m) - nodeW / 2, y: levelY(d) - 7, w: nodeW, h: 14 };
      ctx.save();
      ctx.fillStyle = mixColor(T.n1, T.teal, Math.min(1, rr / rMax));
      ctx.strokeStyle = alphaOf(T.n12, 0.55); ctx.lineWidth = 1;
      ctx.fillRect(rect.x, rect.y, rect.w, rect.h); ctx.strokeRect(rect.x, rect.y, rect.w, rect.h);
      ctx.fillStyle = T.n13; ctx.font = '8px ui-monospace, monospace'; ctx.textAlign = 'center'; ctx.textBaseline = 'middle';
      ctx.fillText(rr.toFixed(2), rect.x + rect.w / 2, rect.y + 7);
      ctx.restore();
      nodeRects.push({ ...rect, d, m });
    }
    // handles drawn last so they stay grabbable-looking
    for (const h of handles) handle(ctx, h, h.b === cb, weightAt(h));

    r.label('readout = mean of the streams', PX - 10, levelY(L) + 26, { color: T.n11, font: '9px ui-monospace, monospace' });

    // ---- right panel: contribution per block, effective depth, identity check
    const panelW = Math.max(150, page.W - RX - pad);
    r.label('contribution per block', RX, topY - 30, { color: T.n14, font: '11px ui-monospace, monospace' });
    r.label('c = ‖what the block wrote‖ / ‖readout‖', RX, topY - 17, { color: T.n10, font: '9px ui-monospace, monospace' });
    const cMax = Math.max(0.02, ...F.blocks.map((x2) => x2.c));
    for (let b = 0; b < L; b++) {
      const cy = levelY(b) + rowH / 2, bhh = Math.min(15, rowH * 0.34);
      const rect = { x: RX, y: cy - bhh / 2, w: panelW - 44, h: bhh };
      ctx.save();
      ctx.strokeStyle = T.n5; ctx.lineWidth = 1; ctx.strokeRect(rect.x, rect.y, rect.w, rect.h);
      ctx.fillStyle = b === cb ? T.accent : alphaOf(T.accent, 0.45);
      ctx.fillRect(rect.x, rect.y, rect.w * Math.min(1, F.blocks[b].c / cMax), rect.h);
      ctx.fillStyle = T.n13; ctx.font = '9px ui-monospace, monospace'; ctx.textAlign = 'left'; ctx.textBaseline = 'middle';
      ctx.fillText(F.blocks[b].c.toFixed(3), rect.x + rect.w + 6, cy);
      ctx.restore();
      barRects.push({ ...rect, b });
    }

    const fy = levelY(L) + 22;
    r.label(`effective depth  (Σc)²/Σc²  =  ${F.eff.toFixed(2)} of ${L}`, RX, fy, { color: T.n14, font: '10px ui-monospace, monospace' });

    // The identity check, always live: the plain single-stream residual next to
    // the hyper-connection readout, and the largest gap between them.
    const exact = F.diff === 0;
    ctx.save();
    ctx.fillStyle = exact ? alphaOf(T.ok, 0.14) : alphaOf(T.warn, 0.12);
    ctx.strokeStyle = exact ? T.ok : T.warn; ctx.lineWidth = 1.2;
    ctx.fillRect(RX, fy + 8, panelW - 8, 44); ctx.strokeRect(RX, fy + 8, panelW - 8, 44);
    ctx.restore();
    r.label(`plain residual ‖·‖ = ${rmsOf(F.plOut).toFixed(4)}`, RX + 7, fy + 22, { color: T.n12, font: '9px ui-monospace, monospace' });
    r.label(`hyper readout  ‖·‖ = ${rmsOf(F.hcOut).toFixed(4)}`, RX + 7, fy + 34, { color: T.n12, font: '9px ui-monospace, monospace' });
    r.label(`max|Δ| = ${F.diff === 0 ? '0 (exact)' : F.diff.toExponential(3)}`, RX + 7, fy + 46, { color: exact ? T.okDeep : T.warnDeep, font: '9px ui-monospace, monospace' });

    // ---- hover --------------------------------------------------------------
    if (page.pointer.over && !grab) {
      const p = page.pointer;
      let tip = null;
      const hh = hitHandle(p.x, p.y);
      if (hh) {
        const v = weightAt(hh);
        if (hh.kind === 'A') tip = `A[block ${hh.b}][stream ${hh.i}] = ${f3(v)}\n${KIND_WORD.A} — how much of stream ${hh.i} goes into block ${hh.b}'s input.\n0 = this block ignores that stream.\ndrag ↕ to change`;
        else if (hh.kind === 'B') tip = `B[block ${hh.b}][stream ${hh.i}] = ${f3(v)}\n${KIND_WORD.B} — how much of block ${hh.b}'s output is added to stream ${hh.i}.\n0 = the block writes nothing there.\ndrag ↕ to change`;
        else tip = `M[block ${hh.b}][stream ${hh.i}][stream ${hh.j}] = ${f3(v)}\n${KIND_WORD.M} — how much of stream ${hh.j} carries into stream ${hh.i} across this block.\n1 on the diagonal = the classic identity skip; off-diagonal = the streams exchange content.\ndrag ↕ to change`;
      } else {
        const nh = nodeRects.find((q) => p.x >= q.x && p.x <= q.x + q.w && p.y >= q.y && p.y <= q.y + q.h);
        if (nh) {
          if (nh.m < 0) tip = `plain residual, level ${nh.d}\n‖·‖rms = ${rmsOf(F.plain[nh.d]).toFixed(4)}\nthe single-highway reference: one stream, all weights pinned to 1`;
          else tip = `stream ${nh.m}, level ${nh.d}\n‖·‖rms = ${rmsOf(F.levels[nh.d][nh.m]).toFixed(4)}\n${nh.d === 0 ? 'level 0: every stream starts as a copy of the embedding' : `after block ${nh.d - 1} wrote into it and the width connection mixed the streams`}`;
        } else {
          const br = barRects.find((q) => p.x >= q.x && p.x <= q.x + q.w && p.y >= q.y && p.y <= q.y + q.h);
          if (br) {
            const B = F.blocks[br.b];
            tip = `block ${br.b} contribution c = ${B.c.toFixed(4)}\n‖written‖ = ${rmsOf(B.add).toFixed(4)}, ‖readout after‖ = ${rmsOf(F.readouts[br.b + 1]).toFixed(4)}\nsmall c late in the stack = that block barely changes the answer`;
          }
        }
      }
      if (tip) page.setTip(tip);
    }

    page.probe = { diff: F.diff, eff: F.eff, cLast: F.blocks[L - 1].c, n, preset: st.preset, rms: F.readouts.map(rmsOf) };

    const cList = F.blocks.map((x2, i) => `c${i}=${x2.c.toFixed(3)}`).join('  ');
    let o = `hyper-connections: ${n} stream${n > 1 ? 's' : ''} × ${L} blocks, D=${D}, preset "${st.preset}".  Each block reads Σ A[m]·s[m], transforms, writes B[k]·y into stream k, then M mixes the streams.    tier:${r.name}\n`;
    o += `${cList}   effective depth (Σc)²/Σc² = ${F.eff.toFixed(2)} of ${L}\n`;
    o += `n=1 identity check — plain residual out[0]=${F.plOut[0].toFixed(6)}, hyper-connection out[0]=${F.hcOut[0].toFixed(6)}, max|Δ| over all ${D} features = ${F.diff === 0 ? '0 (exact)' : F.diff.toExponential(3)}${n === 1 && st.preset !== 'custom' ? '  ← the classic residual is the n=1 unit-weight case' : ''}`;
    if (cb >= 0) o += `\nblock ${cb}: ‖read‖=${F.blocks[cb].rIn.toFixed(3)} → ‖output‖=${F.blocks[cb].rY.toFixed(3)} → contribution c=${F.blocks[cb].c.toFixed(3)}   (drag a handle ↕ to change a weight; every level below recomputes)`;
    page.setReadout(o);
  },

  challenges: [
    {
      goal: 'Reduce the widened highway to the classic residual: get max|Δ| to exactly 0.',
      hint: 'One stream, and every weight on 1 — that is what "the residual is the n=1 special case" means.',
      check: (api) => ({ solved: api.probe.diff === 0, detail: `max|Δ| = ${api.probe.diff == null ? '?' : (api.probe.diff === 0 ? '0' : api.probe.diff.toExponential(2))}` }),
    },
    {
      goal: 'Silence the last block: drive its contribution c below 0.02 while keeping n ≥ 2.',
      hint: 'Its write weights are what decide how much of its output reaches the readout.',
      check: (api) => ({ solved: api.probe.n >= 2 && api.probe.cLast < 0.02, detail: `c(last) = ${api.probe.cLast == null ? '?' : api.probe.cLast.toFixed(4)}, n = ${api.probe.n}` }),
    },
    {
      goal: 'Spread the work evenly: get the effective depth above 3.5 with L = 4 blocks.',
      hint: 'Effective depth peaks when every block contributes the same share — the collapsed preset is the opposite of this.',
      check: (api) => ({ solved: api.probe.eff > 3.5, detail: `effective depth = ${api.probe.eff == null ? '?' : api.probe.eff.toFixed(2)}` }),
    },
  ],
}).then((page) => {
  window.__hyperPage = page;
  const q = new URLSearchParams(location.search);
  const t = page.controls._transport;
  if (q.has('preset')) page.controls.set('preset', q.get('preset'), { rebuild: true });
  if (q.has('n')) page.controls.set('n', Math.max(1, Math.min(5, +q.get('n') | 0)), { rebuild: true });
  if (q.has('L')) page.controls.set('L', Math.max(2, Math.min(6, +q.get('L') | 0)), { rebuild: true });
  if (q.has('D')) page.controls.set('D', Math.max(4, Math.min(10, +q.get('D') | 0)), { rebuild: true });
  if (q.has('seed')) page.controls.set('seed', +q.get('seed') | 0, { rebuild: true });
  if (t) t.rebuildIfDirty();
  // ?w=KIND,b,i[,j],value edits ONE connection weight -- the headless stand-in
  // for a vertical drag, since a screenshot has no pointer.
  //   ?w=A,1,2,0.5   read weight of block 1 from stream 2
  //   ?w=B,1,2,0     write weight of block 1 into stream 2
  //   ?w=M,1,0,2,0.3 width connection of block 1, stream 2 -> stream 0
  if (q.has('w') && cur) {
    for (const spec of q.getAll('w')) {
      const p = spec.split(',');
      const kind = (p[0] || '').toUpperCase();
      const b = +p[1] | 0;
      if (!(b >= 0 && b < cur.L)) continue;
      if (kind === 'M') { const i = +p[2] | 0, j = +p[3] | 0, v = +p[4]; if (i < cur.n && j < cur.n && Number.isFinite(v)) cur.M[b][i][j] = clampW(v); }
      else if (kind === 'A' || kind === 'B') {
        const i = +p[2] | 0, v = +p[3];
        if (i < cur.n && Number.isFinite(v)) { if (kind === 'A') cur.A[b][i] = clampW(v); else cur.Bw[b][i] = clampW(v); }
      }
    }
    page.controls.set('preset', 'custom', { silent: true });
    resync(page);
  }
  // ?hover=x,y fakes the cursor (headless stand-in for a real hover) so the
  // tooltip path is verifiable from a screenshot.
  if (q.has('hover')) {
    const [hx, hy] = q.get('hover').split(',').map(Number);
    page.pointer.x = hx; page.pointer.y = hy; page.pointer.over = true;
  }
  // Deterministic frame for capture: pause the transport for any of these hooks.
  if (q.has('step') || q.has('w') || q.has('hover')) { if (t) t.pause(); }
  if (q.has('step') && t) t.seek(parseInt(q.get('step'), 10));
  if (q.get('play') === '1' && t) t.play();
  page.redraw();
});
