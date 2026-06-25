// moe-balance concept page -- the MoE load-balance problem over a batch.
// A balance knob λ interpolates routing from the router's skewed preference
// (λ=0: a few experts hog everything, the rest STARVE -- rich-get-richer
// collapse) to uniform (λ=1). Per-expert load bars with a capacity line +
// dropped-token overflow, the Switch load-balance aux loss, CV, and starved
// count. Direct manipulation: DRAG A BAR up/down to shift load onto/off that
// expert (the others rebalance); drag the capacity line; the bars ease to
// their targets (animate).
import { mount } from '../framework/layout.js';
import { softmax, seededRandn } from '../framework/tensor.js';

const INK = '#111', RED = '#d1242f', GREY = '#9aa4ad', TEAL = '#0a9396';
const CAT = ['#1f6feb', '#2ca02c', '#d2691e', '#8957e5', '#d1242f', '#0a9396', '#bc6c25', '#5e548e'];
const TOKENS = 120;

let cur = null, builtSig = null;       // {skew, E}
let userMult = null, displayed = null; // per-expert: drag multiplier + eased bar value
let geom = null, grab = null;          // grab: number (bar index) | 'cap'
let pendingShift = null;               // ?shift hook, applied once userMult exists

function buildSkew(seed, E) {
  const aff = seededRandn(seed | 0, E, { std: 1.5 });
  return softmax(Float32Array.from(aff, (x) => x * 1.6));   // a skewed router preference
}
const std = (a) => { const m = a.reduce((s, x) => s + x, 0) / a.length; return Math.sqrt(a.reduce((s, x) => s + (x - m) * (x - m), 0) / a.length); };

mount({
  mount: 'body',
  title: 'moe-balance — load balancing, collapse, capacity',
  blurb: 'The MoE balance problem. Without a balancing pressure the router collapses: a few experts take most tokens and the rest STARVE (no tokens → no gradients → stay bad → never picked — the rich-get-richer loop). The balance knob λ interpolates from the router\'s raw skewed preference (λ=0) to uniform (λ=1). Each expert holds ≈ factor·T/E; tokens to a full expert are dropped (red). The aux loss E·Σ fₑ·Pₑ, the load CV, and the starved count quantify the imbalance. Drag any bar up/down to shift load onto/off it (the others rebalance); drag the capacity line; hover a bar.',
  prefer: 'canvas2d',
  aspect: '2 / 1',
  compare: { key: 'lam', a: 0, b: 1, labelA: 'λ=0 — router collapse (skewed)', labelB: 'λ=1 — balanced (uniform)' },
  animate: true,
  challenges: [
    { goal: 'Balance the experts — get the aux (imbalance) loss below 1.05.', hint: 'raise the balance knob λ toward 1 (or drag the tall bars down).', check: (api) => ({ solved: (api.probe.aux ?? 9) < 1.05, detail: `aux = ${(api.probe.aux ?? 9).toFixed(3)} (1.0 = perfectly balanced)` }) },
    { goal: 'Overload an expert — cause at least one dropped token.', hint: 'lower the capacity factor, or drag one bar above the capacity line.', check: (api) => ({ solved: (api.probe.drops ?? 0) > 0, detail: `${api.probe.drops ?? 0} dropped` }) },
  ],
  controls: (c, page) => {
    c.stepper('E', { label: 'experts (E)', min: 3, max: 8, value: 6 });
    c.slider('lam', { label: 'balance loss λ', min: 0, max: 1, step: 0.05, value: 0 });
    c.toggle('shared', { label: 'shared expert', value: false });
    c.slider('cap', { label: 'capacity factor', min: 1, max: 2, step: 0.05, value: 1.3 });
    c.slider('seed', { label: 'seed', min: 0, max: 99, step: 1, value: 5, rebuild: true });
  },
  onPointer: (page, ev) => {
    if (!geom) return;
    const { barsX, barsW, slot, baseY, topBars, sc, capY, E } = geom;
    if (ev.type === 'down') {
      grab = null;
      if (Math.abs(ev.y - capY) < 9 && ev.x >= barsX - 6 && ev.x <= barsX + barsW) grab = 'cap';
      else { const e = Math.floor((ev.x - barsX) / slot); if (e >= 0 && e < E && ev.y >= topBars && ev.y <= baseY + 6) grab = e; }
    } else if (ev.type === 'up' || ev.type === 'leave') grab = null;
    else if (ev.type === 'move' && page.pointer.down) {
      if (grab === 'cap') { const capCount = Math.max(1, (baseY - ev.y) / sc); page.controls.set('cap', Math.max(1, Math.min(2, Math.round((capCount * E / TOKENS) * 20) / 20)), { silent: true }); }
      else if (typeof grab === 'number') { userMult[grab] = Math.max(0.04, Math.min(25, userMult[grab] * Math.exp(-ev.dy * 0.012))); page.redraw(); }
    }
  },
  draw: (page) => {
    const ctx = page.ctx, st = page.state, r = page.renderer;
    const E = st.E | 0, lam = st.lam, sig = `${st.seed}|${E}`;
    if (builtSig !== sig) { cur = { skew: buildSkew(st.seed, E), E }; userMult = new Float32Array(E).fill(1); displayed = new Float32Array(E); builtSig = sig; }
    if (pendingShift && pendingShift.e < E) { userMult[pendingShift.e] = pendingShift.m; pendingShift = null; }
    r.clear('#ffffff');

    // effective routing = (skew lerp uniform by λ) modulated by drag, renormalized
    const inv = 1 / E;
    const raw = Float32Array.from(cur.skew, (s, e) => ((1 - lam) * s + lam * inv) * userMult[e]);
    let rs = 0; for (const x of raw) rs += x;
    const eff = Float32Array.from(raw, (x) => x / rs);
    const cap = Math.max(1, Math.ceil(st.cap * TOKENS / E));
    const target = Float32Array.from(eff, (x) => x * TOKENS);
    for (let e = 0; e < E; e++) displayed[e] += (target[e] - displayed[e]) * 0.18;   // ease toward target

    // metrics (from the settled target, not the easing display)
    const load = Float32Array.from(target, (x) => Math.round(x));
    let disp = 0, drops = 0; const dispatched = new Float32Array(E);
    for (let e = 0; e < E; e++) { dispatched[e] = Math.min(load[e], cap); disp += dispatched[e]; drops += Math.max(0, load[e] - cap); }
    let aux = 0; for (let e = 0; e < E; e++) aux += (disp ? dispatched[e] / disp : 0) * eff[e]; aux *= E;
    page.probe = { aux, drops };
    const cv = std(load) / (load.reduce((s, x) => s + x, 0) / E || 1);
    const starved = []; for (let e = 0; e < E; e++) if (eff[e] < 0.4 * inv) starved.push(e);

    // layout
    const pad = 16, barsX = pad + 36, panelW = 210, px = page.W - panelW, barsW = px - barsX - 96;
    const baseY = page.H - pad - 46, barsH = page.H - 108, topBars = baseY - barsH;
    const nSlots = E + (st.shared ? 1.5 : 0);
    const slot = barsW / nSlots, bw = Math.min(slot * 0.66, 60);
    let axisMax = cap * 1.25; for (let e = 0; e < E; e++) axisMax = Math.max(axisMax, target[e]); if (st.shared) axisMax = Math.max(axisMax, TOKENS);
    const sc = barsH / axisMax, capY = baseY - cap * sc;
    geom = { barsX, barsW, slot, bw, baseY, topBars, sc, capY, E };

    r.label('per-expert load (tokens) — drag a bar ↕ to shift load', barsX, topBars - 12, { color: INK, font: '12px ui-monospace, monospace' });
    ctx.save();
    // baseline + capacity line
    ctx.strokeStyle = '#e3e6ea'; ctx.lineWidth = 1; ctx.beginPath(); ctx.moveTo(barsX - 6, baseY); ctx.lineTo(barsX + barsW, baseY); ctx.stroke();
    ctx.strokeStyle = RED; ctx.setLineDash([5, 4]); ctx.lineWidth = 1.5; ctx.beginPath(); ctx.moveTo(barsX - 6, capY); ctx.lineTo(barsX + E * slot, capY); ctx.stroke(); ctx.setLineDash([]);
    r.label(`cap ${cap} (drag ↕)`, barsX + E * slot + 2, capY, { color: RED, font: '10px ui-monospace, monospace' });
    // uniform reference line (T/E)
    const uniY = baseY - (TOKENS / E) * sc; ctx.strokeStyle = 'rgba(44,160,44,0.5)'; ctx.setLineDash([2, 3]); ctx.beginPath(); ctx.moveTo(barsX - 6, uniY); ctx.lineTo(barsX + E * slot, uniY); ctx.stroke(); ctx.setLineDash([]);
    r.label(`uniform ${Math.round(TOKENS / E)}`, barsX + E * slot + 2, uniY, { color: '#2ca02c', font: '10px ui-monospace, monospace' });

    ctx.font = '10px ui-monospace, monospace';
    for (let e = 0; e < E; e++) {
      const x = barsX + e * slot + (slot - bw) / 2, h = displayed[e] * sc, isStarved = starved.includes(e);
      const loadH = Math.min(displayed[e], cap) * sc, dropH = Math.max(0, displayed[e] - cap) * sc;
      ctx.fillStyle = isStarved ? '#dfe3e6' : CAT[e % CAT.length]; ctx.globalAlpha = isStarved ? 1 : 0.82; ctx.fillRect(x, baseY - loadH, bw, loadH); ctx.globalAlpha = 1;
      if (dropH > 0) { ctx.fillStyle = RED; ctx.fillRect(x, baseY - loadH - dropH, bw, dropH); }
      ctx.fillStyle = '#1a1d21'; ctx.textAlign = 'center'; ctx.textBaseline = 'bottom'; ctx.fillText(String(load[e]), x + bw / 2, baseY - h - 3);
      ctx.fillStyle = isStarved ? RED : CAT[e % CAT.length]; ctx.textBaseline = 'top'; ctx.fillText(`e${e}`, x + bw / 2, baseY + 4);
      if (isStarved) { ctx.fillStyle = RED; ctx.fillText('starving', x + bw / 2, baseY + 16); }
    }
    // shared expert (always-on)
    if (st.shared) {
      const x = barsX + E * slot + slot * 0.4 + (slot - bw) / 2, h = TOKENS * sc;
      ctx.fillStyle = TEAL; ctx.globalAlpha = 0.82; ctx.fillRect(x, baseY - h, bw, h); ctx.globalAlpha = 1;
      ctx.fillStyle = '#1a1d21'; ctx.textAlign = 'center'; ctx.textBaseline = 'bottom'; ctx.fillText(String(TOKENS), x + bw / 2, baseY - h - 3);
      ctx.fillStyle = TEAL; ctx.textBaseline = 'top'; ctx.fillText('shared', x + bw / 2, baseY + 4);
      ctx.fillText('always', x + bw / 2, baseY + 16);
    }
    ctx.restore();

    // metrics panel
    const py = topBars + 6;
    const lamState = lam < 0.2 ? 'COLLAPSED' : lam < 0.7 ? 'partial' : 'BALANCED';
    const lines = [
      ['balance λ', `${lam.toFixed(2)}  (${lamState})`, lam < 0.2 ? RED : lam > 0.7 ? '#2ca02c' : '#bc6c25'],
      ['aux loss  E·Σfₑ·Pₑ', `${aux.toFixed(3)}`, aux > 1.4 ? RED : aux > 1.12 ? '#bc6c25' : '#2ca02c'],
      ['  (1.0 = uniform)', '', GREY],
      ['load CV', `${cv.toFixed(3)}`, cv > 0.5 ? RED : '#3a4047'],
      ['starved experts', `${starved.length} / ${E}`, starved.length ? RED : '#2ca02c'],
      ['dropped tokens', `${drops}`, drops ? RED : '#2ca02c'],
    ];
    ctx.save(); ctx.textAlign = 'left';
    for (let i = 0; i < lines.length; i++) {
      const [k, v, col] = lines[i];
      r.label(k, px, py + i * 22, { color: '#586069', font: '11px ui-monospace, monospace' });
      r.label(v, px + 4, py + i * 22 + 11, { color: col, font: '12px ui-monospace, monospace' });
    }
    ctx.restore();

    // hover
    if (page.pointer.over && grab === null) {
      const e = Math.floor((page.pointer.x - barsX) / slot);
      if (e >= 0 && e < E && page.pointer.y >= topBars && page.pointer.y <= baseY) {
        page.setTip(`expert ${e}: ${load[e]} tokens (${(eff[e] * 100).toFixed(1)}%)\ncapacity ${cap}${load[e] > cap ? `, dropped ${load[e] - cap}` : ''}${starved.includes(e) ? '\nSTARVING (≈ no tokens → no gradient)' : ''}\ndrag ↕ to shift load`);
      }
    }

    let o = `MoE balance: λ=${lam.toFixed(2)} (${lamState}).  aux=${aux.toFixed(3)} (1.0=uniform), CV=${cv.toFixed(2)}, ${starved.length} starved, ${drops} dropped.    tier:${r.name}\n`;
    o += lam < 0.2
      ? `no balance loss → the router collapses onto its favorites; ${starved.length} expert${starved.length === 1 ? '' : 's'} starve (no tokens, no gradient). Raise λ or drag a starved bar up.`
      : lam > 0.7 ? `strong balance loss → load near uniform (${Math.round(TOKENS / E)}/expert); no starvation.`
        : `partial balance: load is spreading toward uniform but still skewed.`;
    if (st.shared) o += `  shared expert: every token (always active) — absorbs common patterns so routed experts specialize.`;
    page.setReadout(o);
  },
}).then((page) => {
  window.__mobPage = page;
  const q = new URLSearchParams(location.search);
  if (q.has('lam')) page.controls.set('lam', parseFloat(q.get('lam')));
  if (q.has('cap')) page.controls.set('cap', parseFloat(q.get('cap')));
  if (q.has('shared')) page.controls.set('shared', q.get('shared') !== '0');
  // ?shift=e,mult sets the drag multiplier for expert e (headless stand-in).
  if (q.has('shift')) { const [e, m] = q.get('shift').split(',').map(Number); pendingShift = { e, m }; }
  if (q.has('hover')) { const [hx, hy] = q.get('hover').split(',').map(Number); page.pointer.x = hx; page.pointer.y = hy; page.pointer.over = true; }
  page.redraw();
});
