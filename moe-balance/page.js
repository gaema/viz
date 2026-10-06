// moe-balance concept page -- the MoE load-balance problem over a batch.
// Auxiliary-loss mode: λ interpolates routing from the skewed preference
// (λ=0) toward uniform (λ=1). Selection-bias mode: a per-expert offset is
// used only for top-k, the combine weight stays the raw score, and the
// auxiliary-loss weight is 0. Drag a bar in loss mode; the bias step is the
// other control.
import { mount } from '../framework/layout.js';
import { softmax, seededRandn } from '../framework/tensor.js';
import { T, alphaOf } from '../framework/theme.js';
import { auxMix, biasRun, biasBarLoad, shownDrops, moveSentence, compareCaptions, lossReadout, loadShare, cardBlurb } from './math.js';



const CAT = () => [T.accent, T.ok, T.warn, T.violet, T.bad, T.tealDeep, T.warn, T.violetDeep];
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
  blurb: cardBlurb,
  prefer: 'canvas2d',
  aspect: '2 / 1',
  compare: { key: 'lam', a: 0, b: 1, labelA: compareCaptions('loss').a, labelB: compareCaptions('loss').b },
  animate: true,
  challenges: [
    { goal: 'In auxiliary-loss mode, get the aux loss below 1.05. Selection-bias mode keeps the auxiliary weight at 0 and does not use λ for the route.', hint: 'stay on auxiliary loss and raise λ toward 1 (or drag the tall bars down).', check: (api) => ({ solved: api.probe.balance !== 'bias' && (api.probe.aux ?? 9) < 1.05, detail: `mode ${api.probe.balance}, aux = ${(api.probe.aux ?? 9).toFixed(3)}` }) },
    { goal: 'Overload an expert — cause at least one dropped token.', hint: 'lower the capacity factor. In auxiliary-loss mode, drag one bar above the capacity line.', check: (api) => ({ solved: (api.probe.drops ?? 0) > 0, detail: `${api.probe.drops ?? 0} dropped` }) },
  ],
  controls: (c, page) => {
    c.stepper('E', { label: 'experts (E)', min: 3, max: 8, value: 6 });
    c.select('balance', { label: 'balance route', options: [{ value: 'loss', label: 'auxiliary loss λ' }, { value: 'bias', label: 'selection bias' }], value: 'loss' });
    c.slider('lam', { label: 'balance loss λ', min: 0, max: 1, step: 0.05, value: 0 });
    c.stepper('k', { label: 'experts kept (k)', min: 1, max: 4, value: 2 });
    c.slider('step', { label: 'bias step', min: 0.05, max: 1, step: 0.05, value: 0.25 });
    c.stepper('rounds', { label: 'bias steps', min: 1, max: 12, value: 1 });
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
    r.clear(T.n0);

    // Loss mode: skew lerped toward uniform by λ, modulated by drag, renormalized.
    // Bias mode: the offset chooses top-k; combine weights stay the raw scores.
    const inv = 1 / E;
    const loss = auxMix(cur.skew, lam, userMult);
    let biasRow = null;
    if (st.balance === 'bias') {
      const logits = Array.from(cur.skew, (s) => Math.log(Math.max(s, 1e-9)));
      biasRow = biasRun(logits, st.k | 0, +st.step, TOKENS, Math.max(1, st.rounds | 0));
    }
    const target = biasRow
      ? biasBarLoad(biasRow)
      : Float32Array.from(loss.eff, (x) => x * TOKENS);
    const eff = biasRow
      ? Float32Array.from(biasRow.load, (x) => x / TOKENS)
      : loss.eff;
    const cap = Math.max(1, Math.ceil(st.cap * TOKENS / E));
    for (let e = 0; e < E; e++) displayed[e] += (target[e] - displayed[e]) * 0.18;   // ease toward target

    // metrics (from the settled target, not the easing display)
    const load = Float32Array.from(target, (x) => Math.round(x));
    let disp = 0; const dispatched = new Float32Array(E);
    for (let e = 0; e < E; e++) { dispatched[e] = Math.min(load[e], cap); disp += dispatched[e]; }
    const drops = shownDrops(biasRow ? 'bias' : 'loss', load, cap);
    let aux = 0; for (let e = 0; e < E; e++) aux += (disp ? dispatched[e] / disp : 0) * eff[e]; aux *= E;
    page.probe = { aux: biasRow ? 9 : aux, drops, balance: biasRow ? 'bias' : 'loss' };
    if (typeof document !== 'undefined') {
      const caps = compareCaptions(biasRow ? 'bias' : 'loss');
      const capA = document.querySelector('.vz-cmp-cap-a');
      const capB = document.querySelector('.vz-cmp-cap-b');
      if (capA) capA.textContent = caps.a;
      if (capB) capB.textContent = caps.b;
    }
    const cv = std(load) / (load.reduce((s, x) => s + x, 0) / E || 1);
    const starved = [];
    if (!biasRow) for (let e = 0; e < E; e++) if (eff[e] < 0.4 * inv) starved.push(e);

    // layout
    const pad = 16, barsX = pad + 36, panelW = 210, px = page.W - panelW, barsW = px - barsX - 96;
    const baseY = page.H - pad - 46, barsH = page.H - 108, topBars = baseY - barsH;
    const nSlots = E + (st.shared ? 1.5 : 0);
    const slot = barsW / nSlots, bw = Math.min(slot * 0.66, 60);
    let axisMax = cap * 1.25; for (let e = 0; e < E; e++) axisMax = Math.max(axisMax, target[e]); if (st.shared) axisMax = Math.max(axisMax, TOKENS);
    const sc = barsH / axisMax, capY = baseY - cap * sc;
    geom = { barsX, barsW, slot, bw, baseY, topBars, sc, capY, E };

    r.label(biasRow
      ? 'per-expert load (tokens) — the bias step sets these bars'
      : 'per-expert load (tokens) — drag a bar ↕ to shift load', barsX, topBars - 12, { color: T.n14, font: '12px ui-monospace, monospace' });
    ctx.save();
    // baseline + capacity line
    ctx.strokeStyle = T.n4; ctx.lineWidth = 1; ctx.beginPath(); ctx.moveTo(barsX - 6, baseY); ctx.lineTo(barsX + barsW, baseY); ctx.stroke();
    if (!biasRow) {
      ctx.strokeStyle = T.bad; ctx.setLineDash([5, 4]); ctx.lineWidth = 1.5; ctx.beginPath(); ctx.moveTo(barsX - 6, capY); ctx.lineTo(barsX + E * slot, capY); ctx.stroke(); ctx.setLineDash([]);
      r.label(`cap ${cap} (drag ↕)`, barsX + E * slot + 2, capY, { color: T.bad, font: '10px ui-monospace, monospace' });
      const uniY = baseY - (TOKENS / E) * sc; ctx.strokeStyle = alphaOf(T.ok, 0.5); ctx.setLineDash([2, 3]); ctx.beginPath(); ctx.moveTo(barsX - 6, uniY); ctx.lineTo(barsX + E * slot, uniY); ctx.stroke(); ctx.setLineDash([]);
      r.label(`uniform ${Math.round(TOKENS / E)}`, barsX + E * slot + 2, uniY, { color: T.ok, font: '10px ui-monospace, monospace' });
    }

    ctx.font = '10px ui-monospace, monospace';
    for (let e = 0; e < E; e++) {
      const x = barsX + e * slot + (slot - bw) / 2, h = displayed[e] * sc, isStarved = starved.includes(e);
      const loadH = (biasRow ? displayed[e] : Math.min(displayed[e], cap)) * sc;
      const dropH = biasRow ? 0 : Math.max(0, displayed[e] - cap) * sc;
      ctx.fillStyle = isStarved ? T.n5 : CAT()[e % CAT().length]; ctx.globalAlpha = isStarved ? 1 : 0.82; ctx.fillRect(x, baseY - loadH, bw, loadH); ctx.globalAlpha = 1;
      if (!biasRow && dropH > 0) { ctx.fillStyle = T.bad; ctx.fillRect(x, baseY - loadH - dropH, bw, dropH); }
      ctx.fillStyle = T.n14; ctx.textAlign = 'center'; ctx.textBaseline = 'bottom'; ctx.fillText(String(load[e]), x + bw / 2, baseY - h - 3);
      ctx.fillStyle = isStarved ? T.bad : CAT()[e % CAT().length]; ctx.textBaseline = 'top'; ctx.fillText(`e${e}`, x + bw / 2, baseY + 4);
      if (isStarved) { ctx.fillStyle = T.bad; ctx.fillText('starving', x + bw / 2, baseY + 16); }
    }
    // shared expert (always-on)
    if (st.shared) {
      const x = barsX + E * slot + slot * 0.4 + (slot - bw) / 2, h = TOKENS * sc;
      ctx.fillStyle = T.tealDeep; ctx.globalAlpha = 0.82; ctx.fillRect(x, baseY - h, bw, h); ctx.globalAlpha = 1;
      ctx.fillStyle = T.n14; ctx.textAlign = 'center'; ctx.textBaseline = 'bottom'; ctx.fillText(String(TOKENS), x + bw / 2, baseY - h - 3);
      ctx.fillStyle = T.tealDeep; ctx.textBaseline = 'top'; ctx.fillText('shared', x + bw / 2, baseY + 4);
      ctx.fillText('always', x + bw / 2, baseY + 16);
    }
    ctx.restore();

    // metrics panel
    const py = topBars + 6;
    const report = biasRow ? null : lossReadout({ lam, load: Array.from(load), starved: starved.length, tokens: TOKENS, experts: E });
    const lines = [
      biasRow
        ? ['balance route', 'selection bias, auxiliary weight 0', T.teal]
        : ['balance λ', `${lam.toFixed(2)}  (${report.label})`, report.near ? T.ok : report.starved ? T.bad : T.warn],
      biasRow
        ? ['aux weight', '0', T.n12]
        : ['aux loss  E·Σfₑ·Pₑ (f = kept/total, drops out)', `${aux.toFixed(3)}`, aux > 1.4 ? T.bad : aux > 1.12 ? T.warn : T.ok],
      biasRow ? ['bias chooses, combine stays raw', '', T.n9] : ['  (1.0 = uniform)', '', T.n9],
      ['load CV', `${cv.toFixed(3)}`, cv > 0.5 ? T.bad : T.n12],
      biasRow
        ? ['not selected', `${E - biasRow.picked.length} / ${E}`, T.n12]
        : ['starved experts', `${starved.length} / ${E}`, starved.length ? T.bad : T.ok],
      biasRow
        ? ['capacity drops', 'auxiliary-loss mode', T.n12]
        : ['dropped tokens', `${drops}`, drops ? T.bad : T.ok],
    ];
    ctx.save(); ctx.textAlign = 'left';
    for (let i = 0; i < lines.length; i++) {
      const [k, v, col] = lines[i];
      r.label(k, px, py + i * 22, { color: T.n11, font: '11px ui-monospace, monospace' });
      r.label(v, px + 4, py + i * 22 + 11, { color: col, font: '12px ui-monospace, monospace' });
    }
    ctx.restore();

    // hover
    if (page.pointer.over && grab === null) {
      const e = Math.floor((page.pointer.x - barsX) / slot);
      if (e >= 0 && e < E && page.pointer.y >= topBars && page.pointer.y <= baseY) {
        const over = !biasRow && load[e] > cap ? `, dropped ${load[e] - cap}` : '';
        const share = loadShare(load[e], TOKENS);
        page.setTip(`expert ${e}: ${share.sentence}\n${biasRow ? 'selection count' : `capacity ${cap}`}${over}${starved.includes(e) ? '\nSTARVING (≈ no tokens → no gradient)' : ''}${biasRow ? '\nthe bias step sets this bar' : '\ndrag ↕ to shift load'}`);
      }
    }

    let o;
    if (biasRow) {
      const moves = biasRow.moves.map(moveSentence).join(' ');
      o = `selection bias, auxiliary weight 0. kept experts ${biasRow.picked.join(',')}. ${moves}    tier:${r.name}\n`;
      o += `combine weights use the raw scores of the selected experts only. λ is not the route in this mode.`;
    } else {
      o = `MoE balance: λ=${lam.toFixed(2)} (${report.label}).  aux=${aux.toFixed(3)} (1.0=uniform; f counts kept tokens only, drops are out), CV=${cv.toFixed(2)}, ${starved.length} starved, ${drops} dropped.    tier:${r.name}\n`;
      o += report.sentence;
    }
    if (st.shared) o += `  shared expert: every token (always active) — absorbs common patterns so routed experts specialize.`;
    page.setReadout(o);
  },
}).then((page) => {
  window.__mobPage = page;
  const q = new URLSearchParams(location.search);
  if (q.has('lam')) page.controls.set('lam', parseFloat(q.get('lam')));
  if (q.get('balance') === 'bias' || q.get('balance') === 'loss') page.controls.set('balance', q.get('balance'));
  if (q.has('cap')) page.controls.set('cap', parseFloat(q.get('cap')));
  if (q.has('shared')) page.controls.set('shared', q.get('shared') !== '0');
  // ?shift=e,mult sets the drag multiplier for expert e (headless stand-in).
  if (q.has('shift')) { const [e, m] = q.get('shift').split(',').map(Number); pendingShift = { e, m }; }
  if (q.has('hover')) { const [hx, hy] = q.get('hover').split(',').map(Number); page.pointer.x = hx; page.pointer.y = hy; page.pointer.over = true; }
  page.redraw();
});
