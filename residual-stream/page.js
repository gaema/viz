// residual-stream concept page -- the residual highway, traced through depth
// per token. Uses the verified framework: layout.mount() + controls + a
// per-token Transport.
//
// Interactive per the framework contract (plan/framework.md): hover a depth×D
// cell for `depth "<block>" dim d = value`, an rms-column cell for the running
// magnitude (and how it grew from the previous depth), or a read+add flow box
// for what that step does. DIRECT MANIPULATION: drag a cell of the depth×D
// heatmap to edit that block's CONTRIBUTION (row 0 = the embedding, rows 1-4 =
// that block's delta written into the stream) and watch the running sum from
// that depth DOWNWARD recompute live + the ‖rms‖ column update -- one block's
// write propagating down the highway. The per-token transport auto-plays + loops.
import { mount } from '../framework/layout.js';
import { ramps, cellAt } from '../framework/render.js';
import { seededRandn } from '../framework/tensor.js';

const INK = '#111', BLUE = '#1f6feb', GREEN = '#2ca02c';
const DEPTHS = ['embed', '+ attn 0', '+ mlp 0', '+ attn 1', '+ mlp 1', 'final norm'];
const maxAbs = (a) => { let m = 1e-9; for (let i = 0; i < a.length; i++) if (Math.abs(a[i]) > m) m = Math.abs(a[i]); return m; };
const rmsOf = (v) => { let s = 0; for (let i = 0; i < v.length; i++) s += v[i] * v[i]; return Math.sqrt(s / v.length); };
const rmsNorm = (v) => { const r = rmsOf(v) + 1e-6; return Float32Array.from(v, (x) => x / r); };

// Shared between buildData() (seeds embed + the 4 deltas), draw() (renders +
// captures the rects), and onPointer() (hit-tests + edits). The transport
// rebuild runs compute() before the matching draw, so `cur` is fresh. Drag
// edits mutate cur.embed/cur.deltas in place; the trace is recomputed from them
// every draw, so the manipulation propagates down the highway immediately and
// resync() just refreshes the per-token transport axis.
let cur = null;
let Trect = null;          // depth×D heatmap rect, captured in draw for hit-testing
let magRects = null;       // per-depth ‖rms‖ column cell rects
let flowRects = null;      // read+add flow-diagram box rects [{x,y,w,h,label}]
let grab = null;           // {d, c} of the depth×D cell being dragged (d=depth row, c=dim)

function buildData(st) {
  const N = st.N, D = st.D, seed = st.seed | 0;
  const embed = seededRandn(seed, [N, D]);
  const deltas = [1, 2, 3, 4].map((k) => seededRandn(seed + k, [N, D], { std: 0.7 }));  // attn0, mlp0, attn1, mlp1
  cur = { embed, deltas, N, D };
  return Array.from({ length: N }, (_, t) => ({ t, label: `token ${t}: its residual stream from embedding through the blocks to the final norm` }));
}

// Recompute the per-token transport axis from cur (after a dim change or a drag
// edit). The trace itself is rebuilt from cur live in draw(), so a drag edit
// survives + the highway re-runs immediately; this only keeps the axis length
// in sync with N.
function resync(page) {
  const t = page.controls._transport;
  if (!t || !cur) return;
  t.steps = Array.from({ length: cur.N }, (_, i) => ({ t: i, label: `token ${i}: its residual stream from embedding through the blocks to the final norm` }));
  t.scrub.max = Math.max(0, t.steps.length - 1);
  if (t.index > t.steps.length - 1) t.index = t.steps.length - 1;
  t._sync();
}

// Trace the residual stream through depth for token `t` from the (possibly
// edited) embed + deltas. Returns the [ND×D] trace + the per-depth rms[]. This
// is the single source of truth shared by draw() (render) and the drag path
// (so the edit + the picture never disagree).
function traceStream(embed, deltas, N, D, t, pre) {
  const ND = DEPTHS.length;
  const trace = { data: new Float32Array(ND * D), rows: ND, cols: D };
  const mag = new Float32Array(ND);
  let v = Float32Array.from(embed.data.subarray(t * D, t * D + D));
  const setRow = (d, vec) => { for (let j = 0; j < D; j++) trace.data[d * D + j] = vec[j]; mag[d] = rmsOf(vec); };
  setRow(0, v);
  for (let b = 0; b < 4; b++) {
    let nv = new Float32Array(D); for (let j = 0; j < D; j++) nv[j] = v[j] + deltas[b].data[t * D + j];
    if (!pre) nv = rmsNorm(nv);
    v = nv; setRow(b + 1, v);
  }
  setRow(5, pre ? rmsNorm(v) : Float32Array.from(v));
  return { trace, mag };
}

// What operand a depth-row hit edits: row 0 -> the embedding x[t,c]; rows 1..4
// -> the contribution (delta) block b-1 writes into the stream at depth b.
function operandAt(d) {
  if (d === 0) return { arr: cur.embed, kind: 'embed', label: DEPTHS[0] };
  if (d >= 1 && d <= 4) return { arr: cur.deltas[d - 1], kind: 'delta', label: DEPTHS[d] };
  return null;  // depth 5 is the final norm (derived) -- not a draggable operand
}

function flowBox(ctx, x, y, w, h, label, hi) {
  ctx.save();
  ctx.fillStyle = hi ? 'rgba(31,111,235,0.16)' : '#f3f4f7'; ctx.strokeStyle = hi ? BLUE : '#c4ccd3'; ctx.lineWidth = hi ? 2.5 : 1.5;
  ctx.fillRect(x, y, w, h); ctx.strokeRect(x, y, w, h);
  ctx.fillStyle = hi ? BLUE : '#3a4047'; ctx.font = '11px ui-monospace, monospace'; ctx.textAlign = 'center'; ctx.textBaseline = 'middle';
  ctx.fillText(label, x + w / 2, y + h / 2);
  ctx.restore();
}

mount({
  mount: 'body',
  title: 'residual-stream — the residual highway',
  blurb: 'Every block reads the stream (via norm), computes, and ADDS back — never overwrites; the stream is a running sum straight through the model. Drag a depth×D cell to edit one block\'s contribution (top row = the embedding, the rows below = each block\'s delta) and watch the running sum from that depth DOWNWARD recompute live, the ‖rms‖ column following — one block\'s write propagating down the highway. Hover any cell for its value, the rms column for how the magnitude grew, or a flow box for what that step does. Let it play (or scrub) across tokens; toggle pre/post-norm and watch the magnitude grow (pre) or stay flat (post).',
  prefer: 'canvas2d',
  aspect: '2 / 1',
  autoplay: true,
  controls: (c, page) => {
    c.stepper('N', { label: 'tokens (N)', min: 3, max: 6, value: 4 });
    c.stepper('D', { label: 'features (D)', min: 4, max: 8, value: 6 });
    c.slider('seed', { label: 'seed', min: 0, max: 99, step: 1, value: 3, rebuild: true });
    c.toggle('prenorm', { label: 'pre-norm (else post)', value: true });
    c.transport({ compute: () => buildData(page.state), speed: 1.5, loop: true });
  },
  // Direct manipulation: grab a depth×D cell and drag vertically to change that
  // block's contribution. Row 0 edits the embedding x[t,c]; rows 1..4 edit that
  // block's delta. The trace is rebuilt from cur every draw, so the running sum
  // from that depth downward + the rms column recompute under the cursor.
  onPointer: (page, ev) => {
    if (!cur) return;
    const { D } = cur, ND = DEPTHS.length;
    if (ev.type === 'down') {
      grab = null;
      const h = Trect && cellAt(Trect, ND, D, ev.x, ev.y);
      if (h && operandAt(h.r)) grab = { d: h.r, c: h.c };   // depth 5 (final norm) is not draggable
    } else if (ev.type === 'up' || ev.type === 'leave') {
      grab = null;
    } else if (ev.type === 'move' && grab && page.pointer.down) {
      const op = operandAt(grab.d);
      const t = (page.step() || { t: 0 }).t;
      const idx = t * D + grab.c;
      op.arr.data[idx] = Math.max(-3, Math.min(3, op.arr.data[idx] - ev.dy * 0.02));  // drag up = larger
      resync(page);
    }
  },
  draw: (page) => {
    const r = page.renderer, ctx = page.ctx, st = page.state;
    if (!cur) return;
    const { embed, deltas, N, D } = cur;
    r.clear('#ffffff');
    const s = page.step(), t = s ? s.t : 0, pre = st.prenorm, ND = DEPTHS.length;

    const { trace, mag } = traceStream(embed, deltas, N, D, t, pre);

    const pad = 16, topY = 64;
    const cell = Math.max(13, Math.min(30, Math.min((page.W * 0.46) / D, (page.H - topY - 150) / ND)));
    const Tx = pad + 84;
    Trect = { x: Tx, y: topY, w: D * cell, h: ND * cell };
    r.label(`token ${t} — residual stream through depth`, pad, topY - 14, { color: INK, font: '12px ui-monospace, monospace' });
    r.heatmap(trace, { rows: ND, cols: D, rect: Trect, ramp: ramps.diverging, domain: [-maxAbs(trace.data), maxAbs(trace.data)] });
    r.grid({ stroke: 'rgba(0,0,0,0.10)' });
    for (let d = 0; d < ND; d++) r.label(DEPTHS[d], Tx - 6, topY + d * cell + cell / 2 + 3, { color: d === ND - 1 ? BLUE : '#586069', font: '10px ui-monospace, monospace', align: 'right' });
    // outline the depth row being dragged
    if (grab) { ctx.save(); ctx.strokeStyle = INK; ctx.lineWidth = 2.5; ctx.strokeRect(Trect.x - 1, topY + grab.d * cell - 1, D * cell + 2, cell + 2); ctx.restore(); }

    // magnitude column (grows with depth for pre-norm; ~1 for post-norm)
    const mx = Tx + D * cell + 16, mw = 46, magMax = Math.max(...mag);
    r.label('‖rms‖', mx, topY - 6, { color: '#586069', font: '10px ui-monospace, monospace' });
    magRects = [];
    ctx.save();
    for (let d = 0; d < ND; d++) {
      const y = topY + d * cell;
      magRects.push({ x: mx, y: y + 2, w: mw, h: cell - 4 });
      ctx.strokeStyle = '#e3e6ea'; ctx.strokeRect(mx, y + 2, mw, cell - 4);
      ctx.fillStyle = 'rgba(31,111,235,0.5)'; ctx.fillRect(mx, y + 2, mw * (mag[d] / magMax), cell - 4);
      ctx.fillStyle = '#1a1d21'; ctx.font = '9px ui-monospace, monospace'; ctx.textAlign = 'left'; ctx.textBaseline = 'middle'; ctx.fillText(mag[d].toFixed(2), mx + mw + 4, y + cell / 2);
    }
    ctx.restore();

    // read-and-add unit diagram (pre vs post norm placement)
    const dy = topY + ND * cell + 44, bh = 28, bw = 76, gap = 14;
    r.label(pre ? 'each block (×n layers):  stream ← stream + sublayer(norm(stream))' : 'each block (×n layers):  stream ← norm(stream + sublayer(stream))', pad, dy - 10, { color: INK, font: '12px ui-monospace, monospace' });
    const seq = pre ? ['stream', 'norm', 'sublayer', '⊕', 'stream'] : ['stream', 'sublayer', '⊕', 'norm', 'stream'];
    let bx = pad + 6; const cx = [];
    flowRects = [];
    for (let k = 0; k < seq.length; k++) {
      flowBox(ctx, bx, dy, bw, bh, seq[k], seq[k] === 'norm'); cx.push(bx + bw / 2);
      flowRects.push({ x: bx, y: dy, w: bw, h: bh, label: seq[k] });
      if (k < seq.length - 1) { ctx.save(); ctx.strokeStyle = '#9aa4ad'; ctx.beginPath(); ctx.moveTo(bx + bw, dy + bh / 2); ctx.lineTo(bx + bw + gap, dy + bh / 2); ctx.stroke(); ctx.restore(); }
      bx += bw + gap;
    }
    const addIdx = seq.indexOf('⊕');
    ctx.save(); ctx.strokeStyle = GREEN; ctx.lineWidth = 1.5; ctx.beginPath(); ctx.moveTo(cx[0], dy); ctx.bezierCurveTo(cx[0], dy - 22, cx[addIdx], dy - 22, cx[addIdx], dy); ctx.stroke(); ctx.fillStyle = GREEN; ctx.font = '9px ui-monospace, monospace'; ctx.textAlign = 'center'; ctx.fillText('residual (identity)', (cx[0] + cx[addIdx]) / 2, dy - 25); ctx.restore();

    // Hover-to-inspect: depth×D cell -> value; rms cell -> magnitude + growth;
    // flow box -> what that step does. Suppressed while dragging.
    if (page.pointer.over && !grab) {
      const p = page.pointer;
      let tip = null;
      const th = Trect && cellAt(Trect, ND, D, p.x, p.y);
      if (th) {
        const val = trace.data[th.r * D + th.c];
        const op = operandAt(th.r);
        const verb = th.r === 0 ? 'token embedding' : th.r === ND - 1 ? 'final-norm output' : 'running stream after this block';
        tip = `depth "${DEPTHS[th.r]}"  dim ${th.c} = ${val.toFixed(3)}\n${verb}` + (op ? `\ndrag ↕ to edit this ${op.kind === 'embed' ? 'embedding' : 'block contribution'}` : '');
      } else {
        // rms column
        let mi = -1; for (let d = 0; d < ND; d++) { const R = magRects[d]; if (p.x >= R.x && p.x <= R.x + R.w && p.y >= R.y && p.y <= R.y + R.h) { mi = d; break; } }
        if (mi >= 0) {
          if (mi === 0) tip = `‖stream‖_rms at ${DEPTHS[0]} = ${mag[0].toFixed(3)}\n(the embedding's own magnitude — the highway's starting point)`;
          else {
            const dm = mag[mi] - mag[mi - 1], pct = mag[mi - 1] > 1e-6 ? (dm / mag[mi - 1] * 100) : 0;
            const how = pre
              ? `${dm >= 0 ? 'grew' : 'shrank'} ${Math.abs(dm).toFixed(3)} (${pct >= 0 ? '+' : ''}${pct.toFixed(0)}%) from ${DEPTHS[mi - 1]} (${mag[mi - 1].toFixed(3)}) — the add ${mi === ND - 1 ? 'then final-norm rescales' : 'pushed magnitude up'}`
              : `held ~1 vs ${DEPTHS[mi - 1]} (${mag[mi - 1].toFixed(3)}) — post-norm re-normalizes after every add`;
            tip = `‖stream‖_rms at ${DEPTHS[mi]} = ${mag[mi].toFixed(3)}\n${how}`;
          }
        } else if (flowRects) {
          // flow-diagram boxes
          let fb = null; for (const R of flowRects) { if (p.x >= R.x && p.x <= R.x + R.w && p.y >= R.y && p.y <= R.y + R.h) { fb = R.label; break; } }
          if (fb) {
            const desc = {
              stream: 'the residual stream itself — the running sum carried straight through; every block reads from + writes back to it',
              norm: pre ? 'pre-norm: normalize a COPY of the stream to feed the sublayer; the stream itself is left un-normalized so it keeps growing' : 'post-norm: re-normalize the stream AFTER the add, so its magnitude is reset to ~1 each block',
              sublayer: 'the block (attention or MLP) computes its contribution from the normalized read',
              '⊕': 'ADD the contribution back into the stream — the residual (identity) skip means the stream is never overwritten',
            };
            tip = `"${fb}"\n${desc[fb] || ''}`;
          }
        }
      }
      if (tip) page.setTip(tip);
    }

    let o = `Residual stream: each block reads (norm), computes, and ADDS back — the stream is a running sum, never overwritten.    tier:${r.name}\n`;
    o += pre
      ? `pre-norm: stream magnitude grows with depth (rms ${mag[0].toFixed(2)} → ${mag[4].toFixed(2)}), then the final norm rescales → ${mag[5].toFixed(2)}.    (drag a depth×D cell ↕ to edit a block's contribution and watch the highway downstream recompute)`
      : `post-norm: re-normalized after every add, so rms stays ~1 (${mag[1].toFixed(2)}, ${mag[2].toFixed(2)}, …) throughout.    (drag a depth×D cell ↕ to edit a block's contribution and watch the highway downstream recompute)`;
    page.setReadout(o);
  },
}).then((page) => {
  window.__resPage = page;
  const q = new URLSearchParams(location.search);
  const t = page.controls._transport;
  if (q.has('prenorm')) { page.controls.set('prenorm', q.get('prenorm') !== '0'); }
  // ?drag=row,col,val edits the depth×D heatmap (headless stand-in for a vertical
  // drag, since --screenshot has no pointer): row 0 sets the embedding x[t,col],
  // rows 1..4 set that block's delta[t,col]; the trace from that depth downward
  // recomputes. e.g. ?drag=1,2,2.5 sets attn-0's contribution at dim 2 to 2.5 for
  // the current token (the running sum + rms column update from depth 1 down).
  if (q.has('drag') && cur) {
    const parts = q.get('drag').split(',').map(Number);
    const [d, cc, v] = parts.length >= 3 ? parts : [parts[0], 0, parts[1]];
    const op = operandAt(d);
    const tt = (page.step() || { t: 0 }).t;
    if (op && cc >= 0 && cc < cur.D) { op.arr.data[tt * cur.D + cc] = v; resync(page); }
  }
  // ?hover=x,y fakes the cursor position (headless stand-in for a real hover,
  // since --screenshot has no pointer) so the tooltip path is verifiable.
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
