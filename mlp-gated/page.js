// mlp-gated concept page -- SwiGLU/GeGLU gated FFN. Uses the verified framework:
// layout.mount() + controls + a per-stage Transport.
//
// Interactive per the framework contract (plan/framework.md): drag any input (x)
// cell vertically and watch gate, up, gated (act(g)⊙up), and out all recompute
// live -- the element-wise gating "aha"; hover any gate / up / gated cell for its
// derivation (gate[j]=SiLU((xW_g)[j]); gated[j]=gate[j]·up[j]); the five stages
// gate → up → act → ⊙ → down auto-play + loop. Toggle SiLU→SwiGLU / GELU→GeGLU.
import { mount } from '../framework/layout.js';
import { ramps, cellAt } from '../framework/render.js';
import { seededRandn, silu, gelu } from '../framework/tensor.js';

const INK = '#111', BLUE = '#1f6feb', ORANGE = '#d2691e';
const maxAbs = (a) => { let m = 1e-9; for (let i = 0; i < a.length; i++) if (Math.abs(a[i]) > m) m = Math.abs(a[i]); return m; };
const fx = (v) => (v >= 0 ? ' ' : '') + v.toFixed(2);

const STAGES = [
  { key: 'gate', label: 'gate = x · W_gate  (project up to intermediate width I)' },
  { key: 'up', label: 'up = x · W_up  (a second, parallel projection to width I)' },
  { key: 'act', label: 'act(gate) — SiLU (→ SwiGLU) or GELU (→ GeGLU)' },
  { key: 'hidden', label: 'hidden = act(gate) ⊙ up  — the gate, element by element' },
  { key: 'down', label: 'out = hidden · W_down  (project back down to D)' },
];
const HL = { gate: [1], up: [2], act: [3], hidden: [2, 3, 4], down: [4, 5] };  // rows to outline per stage

// Shared between compute() (builds cur + steps), draw() (renders + captures
// rects), and onPointer() (hit-tests + edits the input x). The transport
// rebuild runs compute() before the matching draw, so `cur` is always fresh.
// A drag mutates cur.x in place; resync() re-projects gate/up from the edited x
// so the element-wise gate downstream recomputes live.
let cur = null;
let rowRects = [];   // [{key, rect, cols}] captured in draw for hit-testing
let grab = null;     // {i} while dragging an input (x) cell

// Re-project gate = x·W_gate and up = x·W_up from the (possibly edited) cur.x,
// without regenerating from the seed -- so a drag edit survives and the whole
// gate→up→⊙→down chain recomputes. Resync the transport's step list too.
function recompute() {
  const { x, Wg, Wu, D, I } = cur;
  const gate = new Float32Array(I), up = new Float32Array(I);
  for (let o = 0; o < I; o++) for (let k = 0; k < D; k++) { gate[o] += x[k] * Wg.data[k * I + o]; up[o] += x[k] * Wu.data[k * I + o]; }
  cur.gate = gate; cur.up = up;
}
function resync(page) {
  recompute();
  const t = page.controls._transport;
  if (!t) return;
  t.steps = STAGES.map((s) => ({ ...s }));
  t.scrub.max = Math.max(0, t.steps.length - 1);
  if (t.index > t.steps.length - 1) t.index = t.steps.length - 1;
  t._sync();
}

function buildData(st) {
  const D = st.D, I = 2 * D, seed = st.seed | 0;
  const x = seededRandn(seed, D);
  const Wg = seededRandn(seed + 1, [D, I]), Wu = seededRandn(seed + 2, [D, I]), Wd = seededRandn(seed + 3, [I, D]);
  cur = { x, Wg, Wu, Wd, gate: null, up: null, D, I };
  recompute();
  return STAGES.map((s) => ({ ...s }));
}

mount({
  mount: 'body',
  title: 'mlp-gated — SwiGLU / GeGLU feed-forward',
  blurb: 'The gated MLP in modern LLMs: two parallel projections (gate, up), an element-wise gate act(gate)⊙up, then a down projection. Drag any input x cell vertically and watch gate, up, gated, and out all recompute — the element-wise gating made tangible. Hover a gate / up / gated cell for its derivation. Scrub (or let it play) the five stages. Toggle the activation: SiLU→SwiGLU, GELU→GeGLU.',
  prefer: 'canvas2d',
  aspect: '2 / 1',
  autoplay: true,
  controls: (c, page) => {
    c.stepper('D', { label: 'dim (D)', min: 4, max: 8, value: 6 });
    c.slider('seed', { label: 'seed', min: 0, max: 99, step: 1, value: 4, rebuild: true });
    c.select('act', { label: 'activation', value: 'silu', options: [{ value: 'silu', label: 'SiLU → SwiGLU' }, { value: 'gelu', label: 'GELU → GeGLU' }] });
    c.transport({ compute: () => buildData(page.state), speed: 1.4, loop: true });
  },
  // Direct manipulation: grab an input (x) cell, drag vertically to change its
  // value; gate/up/gated/down all recompute live (the element-wise gate aha).
  onPointer: (page, ev) => {
    if (!cur) return;
    if (ev.type === 'down') {
      grab = null;
      const xr = rowRects.find((r) => r.key === 'x');
      const hit = xr && cellAt(xr.rect, 1, xr.cols, ev.x, ev.y);
      if (hit) grab = { i: hit.c };
    } else if (ev.type === 'up' || ev.type === 'leave') {
      grab = null;
    } else if (ev.type === 'move' && grab && page.pointer.down) {
      cur.x[grab.i] = Math.max(-3, Math.min(3, cur.x[grab.i] - ev.dy * 0.02));  // drag up = larger
      resync(page);
    }
  },
  draw: (page) => {
    const r = page.renderer, ctx = page.ctx, st = page.state;
    if (!cur) return;
    const { x, Wd, gate, up, D, I } = cur;
    r.clear('#ffffff');
    const actName = st.act === 'gelu' ? 'GELU' : 'SiLU', glu = st.act === 'gelu' ? 'GeGLU' : 'SwiGLU';
    const act = st.act === 'gelu' ? gelu(gate) : silu(gate);
    const hidden = new Float32Array(I); for (let i = 0; i < I; i++) hidden[i] = act[i] * up[i];
    const out = new Float32Array(D); for (let o = 0; o < D; o++) for (let i = 0; i < I; i++) out[o] += hidden[i] * Wd.data[i * D + o];

    const s = page.step(), si = s ? STAGES.findIndex((x2) => x2.key === s.key) : STAGES.length - 1;
    const hl = s ? HL[s.key] : [];
    const rows = [
      { lab: 'x', key: 'x', d: x, cols: D, stage: -1 },
      { lab: 'gate', key: 'gate', d: gate, cols: I, stage: 0 },
      { lab: 'up', key: 'up', d: up, cols: I, stage: 1 },
      { lab: `act(g) ${actName}`, key: 'act', d: act, cols: I, stage: 2 },
      { lab: 'hidden ⊙', key: 'hidden', d: hidden, cols: I, stage: 3 },
      { lab: 'out', key: 'out', d: out, cols: D, stage: 4 },
    ];

    const pad = 16, topY = 52, labelW = 96;
    const cell = Math.max(12, Math.min(26, (page.W - 2 * pad - labelW) / I));
    const x0 = pad + labelW, rowH = cell + 16;
    rowRects = [];
    for (let ri = 0; ri < rows.length; ri++) {
      const row = rows[ri], y = topY + ri * rowH, revealed = row.stage <= si, rect = { x: x0, y, w: row.cols * cell, h: cell };
      rowRects.push({ key: row.key, rect, cols: row.cols });
      r.label(row.lab, pad, y + cell / 2 + 3, { color: ri === 0 || ri === 5 ? INK : '#586069', font: '11px ui-monospace, monospace' });
      if (revealed) {
        r.heatmap(row.d, { rows: 1, cols: row.cols, rect, ramp: ramps.diverging, domain: [-maxAbs(row.d), maxAbs(row.d)] });
        r.grid({ stroke: 'rgba(0,0,0,0.10)' });
        if (cell >= 22 && row.cols <= 16) {
          for (let i = 0; i < row.cols; i++) {
            const v = row.d[i];
            r.cell(0, i, { stroke: false, label: v.toFixed(1), labelColor: Math.abs(v) > maxAbs(row.d) * 0.6 ? '#fff' : '#222', font: '9px ui-monospace, monospace' });
          }
        }
      } else { ctx.save(); ctx.strokeStyle = '#e3e6ea'; ctx.setLineDash([4, 3]); ctx.strokeRect(rect.x, rect.y, rect.w, rect.h); ctx.restore(); }
      if (hl.includes(ri)) { ctx.save(); ctx.strokeStyle = INK; ctx.lineWidth = 2.5; ctx.strokeRect(rect.x - 1, rect.y - 1, rect.w + 2, rect.h + 2); ctx.restore(); }
    }
    // mark x as the draggable operand + the gate symbol
    const xRect = rowRects[0].rect;
    r.label('↕ drag', xRect.x + xRect.w + 10, xRect.y + cell / 2 + 3, { color: BLUE, font: '10px ui-monospace, monospace' });
    r.label('↘ two parallel projections of x, then gate ⊙', x0 + I * cell + 12, topY + rowH + cell, { color: '#9aa4ad', font: '10px ui-monospace, monospace' });

    // Hover-to-inspect: gate/up/gated cell -> derivation; x/out cell -> value.
    if (page.pointer.over && !grab) {
      const p = page.pointer;
      let tip = null;
      for (const rr of rowRects) {
        const hit = cellAt(rr.rect, 1, rr.cols, p.x, p.y);
        if (!hit) continue;
        const j = hit.c;
        if (rr.key === 'x') tip = `x[${j}] = ${x[j].toFixed(3)}\ndrag ↕ to change — gate, up, ⊙, out recompute`;
        else if (rr.key === 'gate') tip = `gate[${j}] = (x·W_gate)[${j}]\n= ${gate[j].toFixed(3)}`;
        else if (rr.key === 'up') tip = `up[${j}] = (x·W_up)[${j}]\n= ${up[j].toFixed(3)}`;
        else if (rr.key === 'act') tip = `${actName}(gate[${j}]) = ${actName}(${gate[j].toFixed(2)})\n= ${act[j].toFixed(3)}`;
        else if (rr.key === 'hidden') tip = `hidden[${j}] = ${actName}(gate[${j}]) · up[${j}]\n= ${fx(act[j])} · ${fx(up[j])} = ${hidden[j].toFixed(3)}`;
        else if (rr.key === 'out') {
          const terms = [];
          for (let i = 0; i < I; i++) terms.push(`${fx(hidden[i])}·${fx(Wd.data[i * D + j])}`);
          const shown = terms.length <= 4 ? terms.join(' + ') : terms.slice(0, 3).join(' + ') + ' + … (' + I + ' terms)';
          tip = `out[${j}] = hidden · col ${j} of W_down\n= ${shown}\n= ${out[j].toFixed(3)}`;
        }
        if (tip) break;
      }
      if (tip) page.setTip(tip);
    }

    let o = `Gated MLP (${glu}):  hidden = ${actName}(x·W_gate) ⊙ (x·W_up);  out = hidden·W_down.  intermediate I = 2·D = ${I}.  tier:${r.name}\n`;
    if (!s) o += '(drag an x cell to edit · press ▶ or scrub the five stages: gate → up → act → ⊙ → down)';
    else o += `stage ${si + 1}/5 — ${s.label}`;
    page.setReadout(o);
  },
}).then((page) => {
  window.__mlpPage = page;
  const q = new URLSearchParams(location.search);
  const t = page.controls._transport;
  if (q.has('act')) { page.controls.set('act', q.get('act')); }
  // ?drag=i,val sets an input (x) cell to a value (headless stand-in for a
  // vertical drag, since --screenshot has no pointer). e.g. ?drag=2,1.5
  if (q.has('drag')) {
    const [i, v] = q.get('drag').split(',');
    if (cur && cur.x && +i >= 0 && +i < cur.x.length) { cur.x[+i] = +v; resync(page); }
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
