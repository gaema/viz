// mamba-block concept page -- the full Mamba block, the SSM analog of a
// transformer block. On a residual stream:
//   RMSNorm(x) -> in_proj (split into u, gate)
//   u -> causal Conv1d -> SiLU -> selective SSM -> y
//   gate -> SiLU -> g
//   out = x + out_proj(y ⊙ g)
// Sequence mixing is conv + a per-channel selective scan (linear time, constant
// state) instead of attention, gated by a parallel SiLU branch. The transport
// steps through the SEQUENCE: the SSM state evolves token by token and every
// heatmap fills column by column. Drag any x cell to steer the input.
import { mount } from '../framework/layout.js';
import { ramps, cellAt } from '../framework/render.js';
import { seededRandn } from '../framework/tensor.js';

const INK = '#111', BLUE = '#1f6feb', GREEN = '#2ca02c', PURPLE = '#8957e5', ORANGE = '#d2691e', TEAL = '#0a9396', GREY = '#9aa4ad';
const softplus = (x) => Math.log1p(Math.exp(-Math.abs(x))) + Math.max(x, 0);
const silu = (v) => v / (1 + Math.exp(-v));
const maxAbs = (a) => { let m = 1e-9; for (let i = 0; i < a.length; i++) if (Math.abs(a[i]) > m) m = Math.abs(a[i]); return m; };

let cur = null;
let xR = null, yR = null, gR = null, gatedR = null, outR = null, cell = 11;  // captured in draw
let grab = null;                                                            // {d,t} dragging an x cell

function buildData(st) {
  const D = st.D | 0, L = st.L | 0, E = (st.expand | 0) * D, kc = 3, seed = st.seed | 0;
  const x = seededRandn(seed, [D, L], { std: 1 });                          // [D×L]: x.data[d*L+t]
  const Wproj = seededRandn(seed + 1, [D, 2 * E], { std: 1 / Math.sqrt(D) });
  const Kc = seededRandn(seed + 2, [E, kc], { std: 0.7 });
  const A = Float32Array.from({ length: E }, (_, e) => -(0.25 + 1.6 * e / Math.max(1, E - 1)));
  const B = new Float32Array(E).fill(1), C = new Float32Array(E).fill(1);
  const Wout = seededRandn(seed + 3, [E, D], { std: 1 / Math.sqrt(E) });
  cur = { x, Wproj, Kc, A, B, C, Wout, D, L, E, kc };
  return Array.from({ length: L }, (_, t) => ({ t, label: `position ${t}: conv + scan, gate, project, add` }));
}

function block(st) {
  const { x, Wproj, Kc, A, B, C, Wout, D, L, E, kc } = cur, sel = st.sel, bias = 0.6;
  const norm = new Float32Array(D * L);
  for (let t = 0; t < L; t++) { let ms = 0; for (let d = 0; d < D; d++) { const v = x.data[d * L + t]; ms += v * v; } const inv = 1 / Math.sqrt(ms / D + 1e-6); for (let d = 0; d < D; d++) norm[d * L + t] = x.data[d * L + t] * inv; }
  const u = new Float32Array(E * L), gate = new Float32Array(E * L);
  for (let t = 0; t < L; t++) for (let j = 0; j < 2 * E; j++) { let s = 0; for (let d = 0; d < D; d++) s += norm[d * L + t] * Wproj.data[d * (2 * E) + j]; if (j < E) u[j * L + t] = s; else gate[(j - E) * L + t] = s; }
  const act = new Float32Array(E * L);                                       // causal depthwise conv -> SiLU
  for (let e = 0; e < E; e++) for (let t = 0; t < L; t++) { let s = 0; for (let k = 0; k < kc; k++) { const tt = t - k; if (tt >= 0) s += Kc.data[e * kc + k] * u[e * L + tt]; } act[e * L + t] = silu(s); }
  const y = new Float32Array(E * L);                                         // per-channel selective SSM
  for (let e = 0; e < E; e++) { let h = 0; for (let t = 0; t < L; t++) { const dt = softplus(bias + sel * act[e * L + t]); const Ab = Math.exp(dt * A[e]), Bb = (Ab - 1) / A[e] * B[e]; h = Ab * h + Bb * act[e * L + t]; y[e * L + t] = C[e] * h; } }
  const g = new Float32Array(E * L); for (let i = 0; i < E * L; i++) g[i] = silu(gate[i]);
  const gated = new Float32Array(E * L); for (let i = 0; i < E * L; i++) gated[i] = y[i] * g[i];
  const out = new Float32Array(D * L);
  for (let t = 0; t < L; t++) for (let d = 0; d < D; d++) { let s = 0; for (let e = 0; e < E; e++) s += gated[e * L + t] * Wout.data[e * D + d]; out[d * L + t] = x.data[d * L + t] + s; }
  return { y, g, gated, out, E, D, L };
}

function box(ctx, x, y, w, h, label, color) {
  ctx.save(); ctx.fillStyle = '#f6f7f9'; ctx.fillRect(x, y, w, h); ctx.strokeStyle = color; ctx.lineWidth = 1.6; ctx.strokeRect(x, y, w, h);
  ctx.fillStyle = color; ctx.font = '9px ui-monospace, monospace'; ctx.textAlign = 'center'; ctx.textBaseline = 'middle';
  label.split('\n').forEach((ln, i, a) => ctx.fillText(ln, x + w / 2, y + h / 2 + (i - (a.length - 1) / 2) * 10)); ctx.restore();
}
function arrow(ctx, x1, y1, x2, y2, col) {
  ctx.save(); ctx.strokeStyle = col; ctx.fillStyle = col; ctx.lineWidth = 1.5; ctx.beginPath(); ctx.moveTo(x1, y1); ctx.lineTo(x2, y2); ctx.stroke();
  const a = Math.atan2(y2 - y1, x2 - x1); ctx.beginPath(); ctx.moveTo(x2, y2); ctx.lineTo(x2 - 6 * Math.cos(a - 0.4), y2 - 6 * Math.sin(a - 0.4)); ctx.lineTo(x2 - 6 * Math.cos(a + 0.4), y2 - 6 * Math.sin(a + 0.4)); ctx.closePath(); ctx.fill(); ctx.restore();
}

mount({
  mount: 'body',
  title: 'mamba-block — conv + selective SSM, gated, on a residual stream',
  blurb: 'The full Mamba block (the SSM analog of a transformer block). RMSNorm(x) is projected and split into two branches: the main branch runs a causal Conv1d → SiLU → selective SSM (the per-channel scan with input-dependent Δ); the gate branch runs SiLU. The SSM output is gated y⊙g, projected back, and added to the residual: out = x + out_proj(y⊙g). So Mamba mixes the sequence with conv + a constant-size recurrent state (linear time) instead of attention. Step through the sequence to watch the state evolve token by token and the heatmaps fill column by column; drag any x cell to steer the input.',
  prefer: 'webgl2',
  aspect: '2 / 1',
  autoplay: true,
  controls: (c, page) => {
    c.stepper('L', { label: 'sequence length (L)', min: 4, max: 12, value: 8 });
    c.stepper('D', { label: 'model dim (D)', min: 2, max: 6, value: 4 });
    c.stepper('expand', { label: 'expand (E=eD)', min: 1, max: 2, value: 2 });
    c.slider('sel', { label: 'selectivity (Δ←x)', min: 0, max: 2, step: 0.1, value: 1 });
    c.slider('seed', { label: 'seed', min: 0, max: 99, step: 1, value: 3, rebuild: true });
    c.transport({ compute: () => buildData(page.state), speed: 1.3, loop: true });
  },
  onPointer: (page, ev) => {
    if (!cur || !xR) return;
    const D = cur.D, L = cur.L;
    if (ev.type === 'down') grab = cellAt(xR, D, L, ev.x, ev.y);
    else if (ev.type === 'up' || ev.type === 'leave') grab = null;
    else if (ev.type === 'move' && grab && page.pointer.down) { const i = grab.r * L + grab.c; cur.x.data[i] = Math.max(-3, Math.min(3, cur.x.data[i] - ev.dy * 0.02)); page.redraw(); }
  },
  draw: (page) => {
    const r = page.renderer, ctx = page.ctx, st = page.state;
    if (!cur) return;                       // built by the transport compute (rebuilds on L/D/expand/seed)
    const { D, L, E } = cur;
    r.clear('#ffffff');
    const Bk = block(st);
    const s = page.step(), tcur = s ? s.t : L - 1;
    const xdom = maxAbs(cur.x.data), edom = Math.max(maxAbs(Bk.y), maxAbs(Bk.g), maxAbs(Bk.gated)), odom = maxAbs(Bk.out);

    cell = Math.max(8, Math.min(15, Math.min((page.W * 0.085) / L, (page.H * 0.30) / E)));
    const hw = L * cell;
    const midY = page.H * 0.46, yTop = midY - (E * cell) / 2 - 44, yBot = midY + 30;

    const dh = (data, rows, dom, leftX, topY, dimFuture) => {
      const rect = { x: leftX, y: topY, w: hw, h: rows * cell };
      r.heatmap({ data, rows, cols: L }, { rows, cols: L, rect, ramp: ramps.diverging, domain: [-dom, dom] });
      r.grid({ stroke: 'rgba(0,0,0,0.08)' });
      if (dimFuture && tcur < L - 1) { ctx.save(); ctx.fillStyle = 'rgba(255,255,255,0.66)'; ctx.fillRect(leftX + (tcur + 1) * cell, topY, (L - 1 - tcur) * cell, rows * cell); ctx.restore(); }
      ctx.save(); ctx.strokeStyle = 'rgba(31,111,235,0.8)'; ctx.lineWidth = 1.6; ctx.strokeRect(leftX + tcur * cell, topY, cell, rows * cell); ctx.restore();   // current position column
      return rect;
    };

    // x (input, left, on the residual spine)
    const x0 = 20, xY = midY - (D * cell) / 2;
    r.label('x [D×L]', x0, xY - 8, { color: INK, font: '11px ui-monospace, monospace' });
    xR = dh(cur.x.data, D, xdom, x0, xY, false);
    r.label('drag ↕', x0, xY + D * cell + 12, { color: GREEN, font: '9px ui-monospace, monospace' });

    // in_proj (split)
    const ipx = x0 + hw + 14, ipw = 52;
    box(ctx, ipx, midY - 22, ipw, 44, 'RMSNorm\nin_proj\n(split)', BLUE);
    arrow(ctx, x0 + hw, midY, ipx, midY, GREY);

    // main lane: conv -> SiLU -> SSM -> y
    const opx = ipx + ipw + 14, opw = 96;
    box(ctx, opx, yTop + (E * cell) / 2 - 17, opw, 34, 'Conv1d→SiLU\n→ SSM (scan)', PURPLE);
    arrow(ctx, ipx + ipw, midY - 6, opx + opw / 2, yTop + (E * cell) / 2 + 17, PURPLE);
    const yX = opx + opw + 16;
    r.label('y = SSM(u) [E×L]', yX, yTop - 8, { color: PURPLE, font: '10px ui-monospace, monospace' });
    yR = dh(Bk.y, E, edom, yX, yTop, true);

    // gate lane: SiLU -> g
    box(ctx, opx, yBot + (E * cell) / 2 - 13, opw, 26, 'SiLU (gate)', ORANGE);
    arrow(ctx, ipx + ipw, midY + 6, opx + opw / 2, yBot + (E * cell) / 2 - 13, ORANGE);
    r.label('g = SiLU(gate)', yX, yBot - 8, { color: ORANGE, font: '10px ui-monospace, monospace' });
    gR = dh(Bk.g, E, edom, yX, yBot, true);

    // gate merge ⊙ -> gated
    const mx = yX + hw + 26;
    ctx.save(); ctx.strokeStyle = INK; ctx.fillStyle = '#fff'; ctx.lineWidth = 1.8; ctx.beginPath(); ctx.arc(mx, midY, 12, 0, 7); ctx.fill(); ctx.stroke(); ctx.fillStyle = INK; ctx.font = '13px ui-monospace, monospace'; ctx.textAlign = 'center'; ctx.textBaseline = 'middle'; ctx.fillText('⊙', mx, midY + 1); ctx.restore();
    arrow(ctx, yX + hw, yTop + (E * cell) / 2, mx - 12, midY - 8, PURPLE);
    arrow(ctx, yX + hw, yBot + (E * cell) / 2, mx - 12, midY + 8, ORANGE);
    const gX = mx + 18;
    r.label('y⊙g [E×L]', gX, midY - (E * cell) / 2 - 8, { color: TEAL, font: '10px ui-monospace, monospace' });
    gatedR = dh(Bk.gated, E, edom, gX, midY - (E * cell) / 2, true);

    // out_proj -> out -> residual add
    const ox = gX + hw + 14, oboxw = 46;
    box(ctx, ox, midY - 15, oboxw, 30, 'out_proj\nE→D', TEAL);
    arrow(ctx, gX + hw, midY, ox, midY, TEAL);
    const outX = ox + oboxw + 28;
    // residual ⊕
    ctx.save(); ctx.strokeStyle = GREEN; ctx.fillStyle = '#fff'; ctx.lineWidth = 1.8; ctx.beginPath(); ctx.arc(outX - 16, midY, 11, 0, 7); ctx.fill(); ctx.stroke(); ctx.beginPath(); ctx.moveTo(outX - 22, midY); ctx.lineTo(outX - 10, midY); ctx.moveTo(outX - 16, midY - 6); ctx.lineTo(outX - 16, midY + 6); ctx.stroke(); ctx.restore();
    arrow(ctx, ox + oboxw, midY, outX - 27, midY, TEAL);
    // residual arc from x down/around to ⊕
    ctx.save(); ctx.strokeStyle = GREEN; ctx.lineWidth = 1.4; ctx.setLineDash([4, 3]); const ry = page.H - 26; ctx.beginPath(); ctx.moveTo(x0 + hw / 2, xY + D * cell); ctx.lineTo(x0 + hw / 2, ry); ctx.lineTo(outX - 16, ry); ctx.lineTo(outX - 16, midY + 11); ctx.stroke(); ctx.setLineDash([]); r.label('residual', x0 + hw / 2 + 6, ry - 6, { color: GREEN, font: '9px ui-monospace, monospace' }); ctx.restore();
    const oY = midY - (D * cell) / 2;
    r.label('out [D×L]', outX, oY - 8, { color: INK, font: '11px ui-monospace, monospace' });
    outR = dh(Bk.out, D, odom, outX, oY, true);

    // hover
    if (page.pointer.over && !grab) {
      const p = page.pointer;
      const tens = [['x', cur.x.data, xR, D, 'input'], ['y', Bk.y, yR, E, 'SSM output'], ['g', Bk.g, gR, E, 'SiLU gate'], ['y⊙g', Bk.gated, gatedR, E, 'gated'], ['out', Bk.out, outR, D, 'x + out_proj(y⊙g)']];
      for (const [name, data, rect, rows, note] of tens) { const hit = rect && cellAt(rect, rows, L, p.x, p.y); if (hit) { page.setTip(`${name}[${hit.r}, t${hit.c}] = ${data[hit.r * L + hit.c].toFixed(3)}\n${note}${name === 'x' ? '  ·  drag ↕' : ''}`); break; } }
    }

    let o = `Mamba block: out = x + out_proj( SSM(SiLU(Conv1d(in_proj·x))) ⊙ SiLU(gate) ).  Sequence mixing = conv + selective scan (linear time, constant state), not attention.    tier:${r.name}\n`;
    o += s ? `position ${tcur}/${L - 1}: the SSM state has advanced to t=${tcur}; columns 0..${tcur} are computed, the rest pending.`
      : `whole sequence processed. The selective SSM (per channel) carries the state; the SiLU gate modulates it; out_proj + residual finish the block.`;
    page.setReadout(o);
  },
}).then((page) => {
  window.__mbPage = page;
  const q = new URLSearchParams(location.search);
  const t = page.controls._transport;
  if (q.has('drag')) { const [d, c, v] = q.get('drag').split(',').map(Number); if (cur && d * cur.L + c < cur.x.data.length) cur.x.data[d * cur.L + c] = v; }
  if (q.has('hover')) { const [hx, hy] = q.get('hover').split(',').map(Number); page.pointer.x = hx; page.pointer.y = hy; page.pointer.over = true; }
  if (q.has('step') || q.has('drag') || q.has('hover')) { if (t) t.pause(); }
  if (q.has('step') && t) t.seek(parseInt(q.get('step'), 10));
  if (q.get('play') === '1' && t) t.play();
  page.redraw();
});
