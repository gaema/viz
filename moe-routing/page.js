// moe-routing concept page -- how a Mixture-of-Experts layer routes tokens:
// router_logits [N×E] -> softmax gate -> top-k experts per token -> per-expert
// load / capacity / dropped tokens -> load-balance aux loss. Interactive: drag
// a router-gate cell to re-score a token and watch the load (and drops) shift,
// hover a gate cell or expert bar, step the transport to route token by token.
import { mount } from '../framework/layout.js';
import { ramps, cellAt } from '../framework/render.js';
import { softmax, seededRandn } from '../framework/tensor.js';

const INK = '#111', BLUE = '#1f6feb', GREEN = '#2ca02c', RED = '#d1242f', ORANGE = '#d2691e', GREY = '#8a939b';
const M = (r, c) => ({ data: new Float32Array(r * c), rows: r, cols: c });
const CAT = ['#1f6feb', '#2ca02c', '#d2691e', '#8957e5', '#d1242f', '#0a9396', '#bc6c25', '#5e548e'];

let cur = null;
let gateRect = null, barRects = null;   // captured in draw
let grab = null;                         // {i,e} while dragging a gate cell

function buildData(st) {
  const N = st.N | 0, E = st.E | 0;
  cur = { logits: seededRandn(st.seed | 0, [N, E], { std: 1.2 }), N, E };
  return Array.from({ length: N }, (_, t) => ({ t, label: `route token ${t} to its top-${st.k} experts` }));
}

function route(logits, N, E, k, cap, upto) {
  const g = M(N, E);
  for (let i = 0; i < N; i++) g.data.set(softmax(logits.data.subarray(i * E, (i + 1) * E)), i * E);
  const sel = [];
  for (let i = 0; i < N; i++) sel.push(Array.from({ length: E }, (_, e) => e).sort((a, b) => g.data[i * E + b] - g.data[i * E + a]).slice(0, k));
  const load = new Int32Array(E), drop = new Int32Array(E); let drops = 0, assigned = 0;
  for (let i = 0; i <= upto && i < N; i++) for (const e of sel[i]) { if (load[e] < cap) { load[e]++; assigned++; } else { drop[e]++; drops++; } }
  const P = new Float32Array(E); for (let i = 0; i < N; i++) for (let e = 0; e < E; e++) P[e] += g.data[i * E + e] / N;
  const f = new Float32Array(E); for (let e = 0; e < E; e++) f[e] = assigned > 0 ? load[e] / assigned : 0;
  let aux = 0; for (let e = 0; e < E; e++) aux += f[e] * P[e]; aux *= E;
  return { g, sel, load, drop, drops, P, aux, assigned };
}

mount({
  mount: 'body',
  title: 'moe-routing — router, top-k experts, load balance',
  blurb: 'A Mixture-of-Experts layer replaces the dense MLP with many experts and a router that sends each token to just a few. The router scores token×expert (softmax gate); each token goes to its top-k experts (k=2), renormalized. Each expert has a capacity ≈ factor·N·k/E; tokens routed to a full expert are dropped (red). A lopsided router overloads a few experts and drops tokens — the load-balance aux loss E·Σ fₑ·Pₑ penalizes that. Drag any gate cell to re-score a token and watch the load + drops shift; hover a cell or bar; step to route token by token.',
  prefer: 'webgl2',
  aspect: '2 / 1',
  autoplay: true,
  controls: (c, page) => {
    c.stepper('N', { label: 'tokens (N)', min: 4, max: 10, value: 8 });
    c.stepper('E', { label: 'experts (E)', min: 2, max: 6, value: 4 });
    c.stepper('k', { label: 'top-k', min: 1, max: 3, value: 2, rebuild: false });
    c.slider('cap', { label: 'capacity factor', min: 1, max: 2.5, step: 0.05, value: 1.25 });
    c.slider('seed', { label: 'seed', min: 0, max: 99, step: 1, value: 3, rebuild: true });
    c.transport({ compute: () => buildData(page.state), speed: 1.6, loop: true });
  },
  onPointer: (page, ev) => {
    if (!cur) return;
    const E = cur.E, N = cur.N;
    if (ev.type === 'down') grab = gateRect && cellAt(gateRect, N, E, ev.x, ev.y);
    else if (ev.type === 'up' || ev.type === 'leave') grab = null;
    else if (ev.type === 'move' && grab && page.pointer.down) {
      const idx = grab.r * E + grab.c;
      cur.logits.data[idx] = Math.max(-4, Math.min(4, cur.logits.data[idx] - ev.dy * 0.025));
      page.redraw();
    }
  },
  draw: (page) => {
    const r = page.renderer, ctx = page.ctx, st = page.state;
    if (!cur) return;
    const { N, E } = cur, k = st.k | 0;
    r.clear('#ffffff');
    const cap = Math.max(1, Math.ceil(st.cap * N * k / E));
    const s = page.step(), tok = s ? s.t : N - 1;        // route cumulatively up to this token
    const R = route(cur.logits, N, E, k, cap, tok);

    const pad = 16, topY = 64;
    const cell = Math.max(16, Math.min(34, Math.min((page.H - topY - 120) / N, (page.W * 0.30) / E)));
    gateRect = { x: pad + 28, y: topY, w: E * cell, h: N * cell };

    // ---- router gate heatmap [N×E] ----
    r.label('gate = softmax(logits) [N×E]', gateRect.x, topY - 14, { color: INK, font: '12px ui-monospace, monospace' });
    r.heatmap(R.g, { rows: N, cols: E, rect: gateRect, ramp: ramps.sequential, domain: [0, 1] });
    r.grid({ stroke: 'rgba(0,0,0,0.12)' });
    ctx.save();
    ctx.font = '10px ui-monospace, monospace'; ctx.textBaseline = 'middle';
    for (let i = 0; i < N; i++) {
      ctx.fillStyle = i === tok ? INK : '#586069'; ctx.textAlign = 'right';
      ctx.fillText(`t${i}`, gateRect.x - 5, topY + i * cell + cell / 2);
      // outline the token's top-k experts
      for (const e of R.sel[i]) { ctx.strokeStyle = i <= tok ? CAT[e % CAT.length] : '#c4ccd3'; ctx.lineWidth = i === tok ? 2.6 : 1.6; ctx.strokeRect(gateRect.x + e * cell + 1.5, topY + i * cell + 1.5, cell - 3, cell - 3); }
    }
    for (let e = 0; e < E; e++) { ctx.fillStyle = CAT[e % CAT.length]; ctx.textAlign = 'center'; ctx.fillText(`e${e}`, gateRect.x + e * cell + cell / 2, topY + N * cell + 12); }
    // highlight current token row
    ctx.strokeStyle = INK; ctx.lineWidth = 1.5; ctx.setLineDash([3, 2]); ctx.strokeRect(gateRect.x - 1, topY + tok * cell - 1, E * cell + 2, cell + 2); ctx.setLineDash([]);
    ctx.restore();

    // ---- per-expert load bars ----
    const barsX = gateRect.x + E * cell + 116, baseY = topY + N * cell, maxH = N * cell;
    const slot = Math.min(64, (page.W - barsX - pad) / E), bw = slot * 0.62;
    let axisMax = cap; for (let e = 0; e < E; e++) axisMax = Math.max(axisMax, R.load[e] + R.drop[e]);
    const sc = maxH / Math.max(1, axisMax);
    barRects = [];
    r.label('per-expert load (tokens routed)', barsX, topY - 14, { color: INK, font: '12px ui-monospace, monospace' });
    ctx.save();
    // capacity line
    const capY = baseY - cap * sc;
    ctx.strokeStyle = RED; ctx.setLineDash([5, 4]); ctx.lineWidth = 1.4; ctx.beginPath(); ctx.moveTo(barsX - 6, capY); ctx.lineTo(barsX + E * slot, capY); ctx.stroke(); ctx.setLineDash([]);
    r.label(`capacity ${cap}`, barsX + E * slot + 2, capY, { color: RED, font: '10px ui-monospace, monospace' });
    ctx.strokeStyle = '#e3e6ea'; ctx.beginPath(); ctx.moveTo(barsX - 6, baseY); ctx.lineTo(barsX + E * slot, baseY); ctx.stroke();
    ctx.font = '10px ui-monospace, monospace';
    for (let e = 0; e < E; e++) {
      const x = barsX + e * slot, lh = R.load[e] * sc, dh = R.drop[e] * sc;
      const rect = { x, y: baseY - lh - dh, w: bw, h: lh + dh, e };
      barRects.push(rect);
      ctx.fillStyle = CAT[e % CAT.length]; ctx.globalAlpha = 0.8; ctx.fillRect(x, baseY - lh, bw, lh); ctx.globalAlpha = 1;       // load
      if (dh > 0) { ctx.fillStyle = RED; ctx.fillRect(x, baseY - lh - dh, bw, dh); }                                              // dropped (over capacity)
      ctx.fillStyle = '#1a1d21'; ctx.textAlign = 'center'; ctx.textBaseline = 'bottom'; ctx.fillText(String(R.load[e]), x + bw / 2, baseY - lh - dh - 3);
      ctx.fillStyle = CAT[e % CAT.length]; ctx.textBaseline = 'top'; ctx.fillText(`e${e}`, x + bw / 2, baseY + 4);
      ctx.fillStyle = '#586069'; ctx.fillText(`P${R.P[e].toFixed(2)}`, x + bw / 2, baseY + 16);
    }
    ctx.restore();

    // ---- routing arrows: current token -> its experts' bars ----
    if (tok >= 0 && tok < N) {
      const fromY = topY + tok * cell + cell / 2, fromX = gateRect.x + E * cell + 2;
      for (const e of R.sel[tok]) { const br = barRects[e]; ctx.save(); ctx.strokeStyle = CAT[e % CAT.length]; ctx.lineWidth = 1.8; ctx.beginPath(); ctx.moveTo(fromX, fromY); ctx.lineTo(br.x + bw / 2, baseY - R.load[e] * sc + 2); ctx.stroke(); ctx.restore(); }
    }

    // ---- hover ----
    if (page.pointer.over && !grab) {
      const p = page.pointer;
      const hg = gateRect && cellAt(gateRect, N, E, p.x, p.y);
      if (hg) { const picked = R.sel[hg.r].includes(hg.c); page.setTip(`token ${hg.r} → expert ${hg.c}\ngate ${R.g.data[hg.r * E + hg.c].toFixed(3)}  ${picked ? '(top-' + k + ' ✓ routed)' : '(not selected)'}\ndrag ↕ to re-score`); }
      else { for (const br of barRects || []) if (p.x >= br.x && p.x <= br.x + bw && p.y >= br.y - 4 && p.y <= baseY) { page.setTip(`expert ${br.e}: load ${R.load[br.e]} / capacity ${cap}${R.drop[br.e] ? `\nDROPPED ${R.drop[br.e]} (over capacity)` : ''}\nmean router prob P = ${R.P[br.e].toFixed(3)}`); break; } }
    }

    const counts = Array.from(R.load).join(' / ');
    let o = `MoE: router → top-${k} of ${E} experts/token; capacity ${cap}; load-balance aux = E·Σ fₑ·Pₑ = ${R.aux.toFixed(3)} (1.0 = perfectly uniform).    tier:${r.name}\n`;
    o += s ? `routing token ${tok}/${N - 1} → experts {${R.sel[tok].map((e) => 'e' + e).join(', ')}}.   loads [${counts}]${R.drops ? `,  DROPPED ${R.drops} (router imbalanced)` : ',  no drops'}.`
      : `all ${N} tokens routed.   loads [${counts}]${R.drops ? `,  DROPPED ${R.drops} — overloaded experts` : ',  balanced (no drops)'}.`;
    page.setReadout(o);
  },
}).then((page) => {
  window.__moePage = page;
  const q = new URLSearchParams(location.search);
  const t = page.controls._transport;
  if (q.has('cap')) page.controls.set('cap', parseFloat(q.get('cap')));
  // ?drag=i,e,val sets router_logits[token i, expert e] (headless stand-in).
  if (q.has('drag')) { const [i, e, v] = q.get('drag').split(',').map(Number); if (cur && i * cur.E + e < cur.logits.data.length) cur.logits.data[i * cur.E + e] = v; }
  if (q.has('hover')) { const [hx, hy] = q.get('hover').split(',').map(Number); page.pointer.x = hx; page.pointer.y = hy; page.pointer.over = true; }
  if (q.has('step') || q.has('drag') || q.has('hover') || q.has('cap')) { if (t) t.pause(); }
  if (q.has('step') && t) t.seek(parseInt(q.get('step'), 10));
  if (q.get('play') === '1' && t) t.play();
  page.redraw();
});
