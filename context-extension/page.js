// context-extension concept page -- stretching a rotary-position model past the
// context length it was TRAINED on, and exactly where each method breaks.
//
// The `rope` page owns the rotation mechanism itself (pair i turns by
// Δ = p·θᵢ). This page starts one level up from that and asks a different
// question: each pair has a WAVELENGTH λᵢ = 2π/θᵢ, and the trained context L
// covers L/λᵢ turns of it. A pair that completes more than a full turn inside L
// has shown the model every angle it can ever produce; a pair whose wavelength
// is LONGER than L has only ever been seen over a slice of its circle, so a
// position past L drives it into angles that were never in the training data.
//
// That single fact separates the four responses, and the separation is a
// PICTURE rather than a table: naive extrapolation moves nothing and lets every
// long-wavelength pair fall off the end; position interpolation divides all
// positions by the extension factor, so nothing is out of distribution but
// EVERY pair loses fine detail; NTK/frequency scaling rescales the base, which
// spreads the change unevenly across pairs; YaRN sorts pairs by wavelength and
// only moves the ones long enough to justify it. LongRoPE drops the single rule
// entirely and searches a factor per dimension.
//
// Drag the extension factor, drag the trained-context line, step the position
// from inside the trained range to far outside, and watch which pairs recolour.
import { mount } from '../framework/layout.js';
import { T, alphaOf, rgbaToken } from '../framework/theme.js';

const TAU = Math.PI * 2;
const SMAX = 32;              // extension-factor axis of the trade chart
const YARN_ALPHA = 1;         // YaRN ramp: below this many turns inside L -> interpolate fully
const YARN_BETA = 32;         // above this many turns -> leave the pair alone
const NSTEP = 25;             // transport steps across [0, s·L]

const METHODS = [
  { id: 'none', label: 'naive extrapolation', tok: 'n9' },
  { id: 'pi', label: 'position interpolation', tok: 'teal' },
  { id: 'ntk', label: 'NTK / frequency scaling', tok: 'goldDeep' },
  { id: 'yarn', label: 'YaRN (by wavelength)', tok: 'accent' },
  { id: 'longrope', label: 'LongRoPE (searched)', tok: 'violet' },
];
const methodLabel = (id) => (METHODS.find((m) => m.id === id) || METHODS[0]).label;
// Read the colour at DRAW time -- T is mutated in place on a theme change, so a
// captured colour would freeze the theme the page loaded with.
const methodColor = (id) => T[(METHODS.find((m) => m.id === id) || METHODS[0]).tok];

const fmt = (x) => (x === 0 ? '0' : Math.abs(x) < 1e-3 || Math.abs(x) >= 1e5 ? x.toExponential(2) : String(Number(x.toPrecision(4))));
const ifmt = (x) => Math.round(x).toLocaleString('en-US');

// --- the math ---------------------------------------------------------------

/** Un-extended RoPE frequencies: θᵢ = base^(−2i/d), one per dimension PAIR. */
function baseFreqs(d, base) {
  const np = d >> 1, th = new Float64Array(np);
  for (let i = 0; i < np; i++) th[i] = Math.pow(base, -2 * i / d);
  return th;
}

/** YaRN's ramp on the number of turns r a pair completes inside the trained context. */
function yarnGamma(r) {
  if (r > YARN_BETA) return 1;              // many turns inside L -> already fully seen
  if (r < YARN_ALPHA) return 0;             // less than one turn -> never seen past its slice
  return (r - YARN_ALPHA) / (YARN_BETA - YARN_ALPHA);
}

// LongRoPE searches a rescale factor PER DIMENSION instead of deriving one from
// a rule, so its factor curve is not smooth. The real factors are per-model and
// come out of an evolutionary search; this is a deterministic ILLUSTRATIVE
// stand-in with the two properties the search result has -- non-uniform across
// dimensions, and monotone non-decreasing with wavelength -- so the reader can
// see what "searched, not derived" looks like beside the three rules.
function longropeFactors(np, s) {
  let x = 20240613 >>> 0;
  const rnd = () => { x = (Math.imul(x, 1664525) + 1013904223) >>> 0; return x / 4294967296; };
  const out = new Float64Array(np);
  let cur = 1;
  for (let i = 0; i < np; i++) {
    const t = np <= 1 ? 0 : i / (np - 1);
    const target = 1 + (s - 1) * Math.pow(t, 1.7);
    const jitter = 0.70 + 0.60 * rnd();
    cur = Math.max(cur, Math.min(s, Math.max(1, target * jitter)));
    out[i] = cur;
  }
  return out;
}

/**
 * Per-pair rescale factor fᵢ = θᵢ / θ′ᵢ for one method at extension factor s.
 * fᵢ = 1 means "this pair is untouched"; fᵢ = s means "fully interpolated".
 */
function factorsFor(d, base, L, s, method) {
  const np = d >> 1, th = baseFreqs(d, base), f = new Float64Array(np);
  if (method === 'longrope') return longropeFactors(np, s);
  for (let i = 0; i < np; i++) {
    if (method === 'pi') f[i] = s;                                    // every position divided by s
    else if (method === 'ntk') f[i] = Math.pow(s, 2 * i / (d - 2));   // base·s^(d/(d−2)) -> θ′ᵢ = θᵢ·s^(−2i/(d−2))
    else if (method === 'yarn') {
      const turns = L / (TAU / th[i]);
      const g = yarnGamma(turns);
      f[i] = 1 / ((1 - g) / s + g);                                   // θ′ = (1−γ)·θ/s + γ·θ
    } else f[i] = 1;                                                  // naive: change nothing
  }
  return f;
}

/** Everything the drawing needs for one (state, position) pair. */
function analyse(st, pos) {
  const d = st.d, base = st.base, L = st.ctx, s = st.scale, np = d >> 1;
  const th = baseFreqs(d, base);
  const fac = factorsFor(d, base, L, s, st.method);
  const rows = [];
  let ood = 0, sumRel = 0, sumSqE = 0, sumSqO = 0;
  for (let i = 0; i < np; i++) {
    const eff = th[i] / fac[i];
    const lam = TAU / th[i], lamEff = TAU / eff;
    const turns = L / lam;                                  // turns completed inside the trained context
    const cover = Math.min(TAU, (L - 1) * th[i]);            // angular slice the model actually saw
    const angle = pos * eff;                                 // angle this method asks for at `pos`
    const wrapped = angle - TAU * Math.floor(angle / TAU);
    const seen = cover >= TAU - 1e-9 || wrapped <= cover + 1e-9;
    if (!seen) ood++;
    sumRel += 1 / fac[i];
    sumSqE += eff * eff; sumSqO += th[i] * th[i];
    rows.push({ i, th: th[i], eff, lam, lamEff, turns, cover, angle, wrapped, seen, fac: fac[i] });
  }
  return {
    rows, np,
    meanSep: 100 * sumRel / np,                              // mean per-pair adjacent-token separation, % of un-extended
    l2Sep: 100 * Math.sqrt(sumSqE) / Math.sqrt(sumSqO || 1), // same thing in L2, dominated by the fastest pair
    fastSep: th[0] / fac[0],                                 // radians per token on pair 0
    ood,
  };
}

/** Mean per-pair separation as a % of the un-extended model's, for the trade chart. */
function meanSepAt(d, base, L, s, method) {
  const f = factorsFor(d, base, L, s, method);
  let acc = 0;
  for (let i = 0; i < f.length; i++) acc += 1 / f[i];
  return 100 * acc / f.length;
}

// --- live view state (shared between draw / onPointer / the URL hooks) -------
let posOverride = null;       // a dragged / deep-linked position pins the view
let grab = null;              // 'pos' | 'ctx' | 'scale' while dragging
let hit = { spec: null, trade: null, track: null, ctxY: 0 };

function buildSteps(st) {
  const span = st.scale * st.ctx;
  return Array.from({ length: NSTEP }, (_, k) => {
    const pos = Math.round((k / (NSTEP - 1)) * span);
    return { k, pos, label: `position ${ifmt(pos)}  —  ${(pos / st.ctx).toFixed(2)}× the trained context` };
  });
}

function currentPos(page) {
  if (posOverride != null) return posOverride;
  const t = page.controls._transport;
  if (t && t.steps.length) return t.steps[Math.max(0, Math.min(t.steps.length - 1, t.index))].pos;
  return 0;
}

mount({
  mount: 'body',
  title: 'context-extension — running past the trained context length',
  blurb: 'Every dimension pair rotates at its own frequency, so it has its own WAVELENGTH λᵢ = 2π/θᵢ. Inside the trained context L a fast pair turns many times (the model has seen all of its angles); a pair whose wavelength is longer than L has only ever been seen over a slice of its circle. Push the position past L and those long pairs are the ones that go out of distribution. The spectrum below plots every pair’s wavelength against L and colours it by whether its angle at the current position was ever seen. DRAG the extension factor (on the trade chart), DRAG the trained-context line, step the position from inside the trained range to far outside, and switch methods to redraw the same spectrum. The trade — long-range coverage bought with short-range resolution — is computed live at the bottom left.',
  prefer: 'canvas2d',
  aspect: '16 / 11',
  autoplay: true,
  compare: { key: 'method', a: 'pi', b: 'yarn', labelA: 'position interpolation', labelB: 'YaRN' },
  challenges: [
    { goal: 'Push the position past the trained context with naive extrapolation until at least one pair is out of distribution.',
      hint: 'method = naive extrapolation, then step the transport past 1.0× the trained context.',
      check: (api) => ({ solved: api.state.method === 'none' && (api.probe.ood || 0) > 0, detail: `method=${api.state.method}, ${api.probe.ood || 0} pair(s) out of distribution` }) },
    { goal: 'Find a method that keeps every pair in distribution AND leaves the fastest pair at full resolution.',
      hint: 'position interpolation moves every pair; try the one that sorts pairs by wavelength.',
      check: (api) => ({ solved: (api.probe.ood || 0) === 0 && (api.probe.fastFac || 99) < 1.001 && api.state.scale > 1, detail: `${api.probe.ood || 0} out of distribution, pair-0 rescale ×${(api.probe.fastFac || 1).toFixed(2)}` }) },
  ],
  controls: (c, page) => {
    c.select('method', { label: 'extension method', options: METHODS.map((m) => ({ value: m.id, label: m.label })), value: 'yarn' });
    c.slider('scale', { label: 'extension factor s', min: 1, max: SMAX, step: 0.5, value: 8, rebuild: true, format: (v) => `${(+v).toFixed(1)}×` });
    c.slider('ctx', { label: 'trained context L', min: 512, max: 32768, step: 512, value: 4096, rebuild: true, format: (v) => ifmt(v) });
    c.stepper('d', { label: 'head_dim', min: 8, max: 128, step: 8, value: 64, rebuild: false });
    c.slider('base', { label: 'rotary base θ', min: 1000, max: 50000, step: 1000, value: 10000, rebuild: false, format: (v) => ifmt(v) });
    c.transport({ compute: () => buildSteps(page.state), speed: 2.5, loop: true });
  },

  // Direct manipulation: the trade chart sets the extension factor, the dashed
  // line on the spectrum sets the trained context length, the top track sets the
  // position. Each one recomputes the whole picture under the hand.
  onPointer: (page, ev) => {
    if (ev.type === 'leave') { page.pointer.over = false; grab = null; return; }
    if (ev.type === 'up') { grab = null; return; }
    if (ev.type === 'down') {
      const tr = hit.trade, sp = hit.spec;
      if (tr && ev.x >= tr.x && ev.x <= tr.x + tr.w && ev.y >= tr.y && ev.y <= tr.y + tr.h) grab = 'scale';
      else if (sp && Math.abs(ev.y - hit.ctxY) <= 12 && ev.x >= sp.x && ev.x <= sp.x + sp.w) grab = 'ctx';
      else if (hit.track && ev.y <= hit.track.y + hit.track.h + 10) grab = 'pos';
      else grab = null;
    }
    if (!grab || (ev.type === 'move' && !page.pointer.down)) return;
    if (grab === 'scale') {
      const tr = hit.trade;
      const f = Math.max(0, Math.min(1, (ev.x - tr.x) / (tr.w || 1)));
      page.controls.set('scale', Math.round((1 + f * (SMAX - 1)) * 2) / 2, { rebuild: true });
    } else if (grab === 'ctx') {
      const lam = hit.spec.lamOf(ev.y);
      const v = Math.max(512, Math.min(32768, Math.round(lam / 512) * 512));
      page.controls.set('ctx', v, { rebuild: true });
    } else if (grab === 'pos') {
      const tk = hit.track;
      const f = Math.max(0, Math.min(1, (ev.x - tk.x) / (tk.w || 1)));
      posOverride = Math.round(f * page.state.scale * page.state.ctx);
      const t = page.controls._transport;
      if (t && t.steps.length) { t.pause(); t.index = Math.round(f * (t.steps.length - 1)); t._sync(); }
    }
  },

  draw: (page) => {
    const r = page.renderer, ctx = page.ctx, st = page.state;
    const W = page.W, H = page.H, pad = 14, padL = 58;
    r.clear(T.n0);

    const L = st.ctx, s = st.scale, target = Math.round(L * s);
    const pos = Math.max(0, Math.min(target, currentPos(page)));
    const A = analyse(st, pos);
    page.probe = { ood: A.ood, fastFac: A.rows.length ? A.rows[0].fac : 1, meanSep: A.meanSep, pos };

    // ---- position track ----------------------------------------------------
    const trkY = 32, trkX = padL, trkW = W - pad - padL;
    hit.track = { x: trkX, y: trkY - 9, w: trkW, h: 18 };
    const fracL = target > 0 ? Math.min(1, L / target) : 1;
    ctx.save();
    ctx.fillStyle = T.okBg; ctx.fillRect(trkX, trkY - 7, trkW * fracL, 14);
    ctx.fillStyle = T.warnBg; ctx.fillRect(trkX + trkW * fracL, trkY - 7, trkW * (1 - fracL), 14);
    ctx.strokeStyle = T.n6; ctx.lineWidth = 1; ctx.strokeRect(trkX + 0.5, trkY - 6.5, trkW - 1, 13);
    ctx.strokeStyle = T.accent; ctx.lineWidth = 1.5;
    ctx.beginPath(); ctx.moveTo(trkX + trkW * fracL, trkY - 9); ctx.lineTo(trkX + trkW * fracL, trkY + 9); ctx.stroke();
    const hx = trkX + (target > 0 ? (pos / target) * trkW : 0);
    ctx.fillStyle = grab === 'pos' ? T.warn : T.n14;
    ctx.beginPath(); ctx.arc(hx, trkY, 6.5, 0, TAU); ctx.fill();
    ctx.restore();
    r.label('position', pad, trkY + 4, { color: T.n11, font: '10px ui-monospace, monospace' });
    r.label(`p = ${ifmt(pos)}  (${(pos / L).toFixed(2)}× trained)`, trkX, trkY - 14, { color: pos > L ? T.warn : T.n12, font: '11px ui-monospace, monospace' });
    r.label(`target ${ifmt(target)}`, trkX + trkW, trkY - 14, { color: T.n10, font: '10px ui-monospace, monospace', align: 'right' });
    r.label(`trained L = ${ifmt(L)}`, trkX + trkW * fracL, trkY + 20, { color: T.accent, font: '10px ui-monospace, monospace', align: fracL > 0.75 ? 'right' : 'left' });

    // ---- wavelength spectrum ----------------------------------------------
    const specTop = trkY + 46, specBot = Math.round(H * 0.505);
    const specX = padL, specW = W - pad - padL, colW = specW / A.np;
    let maxLam = Math.max(L * 2.5, pos * 1.15, 100);
    for (const row of A.rows) maxLam = Math.max(maxLam, row.lamEff * 1.3);
    const lo = Math.log10(3), hi = Math.log10(maxLam);
    const yOf = (lam) => specBot - (Math.log10(Math.max(lam, 1e-3)) - lo) / (hi - lo) * (specBot - specTop);
    const lamOf = (y) => Math.pow(10, lo + (specBot - y) / ((specBot - specTop) || 1) * (hi - lo));
    hit.spec = { x: specX, y: specTop, w: specW, h: specBot - specTop, lamOf };

    ctx.save();
    ctx.fillStyle = T.n1; ctx.fillRect(specX, specTop, specW, specBot - specTop);
    ctx.strokeStyle = rgbaToken('n14', 0.08); ctx.lineWidth = 1;
    for (let e = 0; e <= 7; e++) {
      const y = yOf(Math.pow(10, e));
      if (y < specTop || y > specBot) continue;
      ctx.beginPath(); ctx.moveTo(specX, y); ctx.lineTo(specX + specW, y); ctx.stroke();
      r.label(`1e${e}`, specX - 6, y + 3, { color: T.n9, font: '10px ui-monospace, monospace', align: 'right' });
    }
    ctx.restore();
    r.label('wavelength λᵢ = 2π/θᵢ  (tokens, log)', pad, specTop - 8, { color: T.n11, font: '11px ui-monospace, monospace' });
    r.label(`${methodLabel(st.method)} · s = ${s.toFixed(1)}×`, specX + specW, specTop - 8, { color: methodColor(st.method), font: '11px ui-monospace, monospace', align: 'right' });

    // the two horizontal references: trained context (draggable) and the position
    const ctxY = yOf(L);
    hit.ctxY = ctxY;
    ctx.save();
    ctx.setLineDash([6, 4]); ctx.strokeStyle = T.accent; ctx.lineWidth = grab === 'ctx' ? 2.6 : 1.8;
    ctx.beginPath(); ctx.moveTo(specX, ctxY); ctx.lineTo(specX + specW, ctxY); ctx.stroke();
    if (pos > 0) {
      const py = yOf(pos);
      if (py >= specTop && py <= specBot) {
        ctx.setLineDash([2, 3]); ctx.strokeStyle = T.warn; ctx.lineWidth = 1.4;
        ctx.beginPath(); ctx.moveTo(specX, py); ctx.lineTo(specX + specW, py); ctx.stroke();
        r.label(`p = ${ifmt(pos)}`, specX + specW - 4, py - 4, { color: T.warn, font: '10px ui-monospace, monospace', align: 'right' });
      }
    }
    ctx.restore();
    r.label(`trained context L = ${ifmt(L)}  (drag me)`, specX + 4, ctxY - 5, { color: T.accent, font: '10px ui-monospace, monospace' });

    // one column per pair: ghost dot = un-extended λᵢ, filled dot = λ′ᵢ after the
    // method, stem between them, colour = in / out of distribution at `pos`.
    for (const row of A.rows) {
      const cx = specX + (row.i + 0.5) * colW;
      const y0 = yOf(row.lam), y1 = yOf(row.lamEff);
      ctx.save();
      if (Math.abs(y1 - y0) > 0.6) {
        ctx.strokeStyle = alphaOf(methodColor(st.method), 0.55); ctx.lineWidth = Math.max(1.4, Math.min(6, colW * 0.34));
        ctx.beginPath(); ctx.moveTo(cx, y0); ctx.lineTo(cx, y1); ctx.stroke();
      }
      ctx.fillStyle = alphaOf('n9', 0.75);
      ctx.beginPath(); ctx.arc(cx, y0, Math.max(1.6, Math.min(3.2, colW * 0.18)), 0, TAU); ctx.fill();
      ctx.fillStyle = row.seen ? T.ok : T.bad;
      ctx.beginPath(); ctx.arc(cx, y1, Math.max(2.2, Math.min(4.6, colW * 0.26)), 0, TAU); ctx.fill();
      ctx.restore();
    }
    r.label('pair 0 (fast, short λ)', specX, specBot + 13, { color: T.n10, font: '10px ui-monospace, monospace' });
    r.label(`pair ${A.np - 1} (slow, long λ) →`, specX + specW, specBot + 13, { color: T.n10, font: '10px ui-monospace, monospace', align: 'right' });
    let lx = specX + 10;
    for (const [col, txt] of [[T.ok, 'in distribution'], [T.bad, 'never seen at this position'], [alphaOf('n9', 0.75), 'un-extended λᵢ']]) {
      ctx.save(); ctx.fillStyle = col; ctx.beginPath(); ctx.arc(lx, specTop + 11, 3.2, 0, TAU); ctx.fill(); ctx.restore();
      r.label(txt, lx + 7, specTop + 15, { color: T.n10, font: '10px ui-monospace, monospace' });
      lx += 7 + txt.length * 6 + 18;
    }

    // ---- per-pair rescale-factor strip ------------------------------------
    const facTop = specBot + 38, facH = Math.max(30, Math.round(H * 0.095)), facBot = facTop + facH;
    r.label('per-pair rescale θᵢ/θ′ᵢ  —  how far this method moved each pair', specX + specW / 2, facTop - 8, { color: T.n11, font: '11px ui-monospace, monospace', align: 'center' });
    ctx.save();
    ctx.fillStyle = T.n1; ctx.fillRect(specX, facTop, specW, facH);
    ctx.restore();
    const fy = (f) => facBot - (Math.log(Math.max(1, f)) / Math.log(Math.max(1.0001, s))) * facH;
    for (const row of A.rows) {
      const cx = specX + (row.i + 0.5) * colW;
      const bw = Math.max(1.5, colW * 0.66);
      ctx.save();
      ctx.fillStyle = alphaOf(methodColor(st.method), 0.85);
      const y = fy(row.fac);
      ctx.fillRect(cx - bw / 2, y, bw, facBot - y);
      ctx.restore();
    }
    ctx.save();
    ctx.strokeStyle = rgbaToken('n14', 0.18); ctx.lineWidth = 1;
    ctx.beginPath(); ctx.moveTo(specX, facBot); ctx.lineTo(specX + specW, facBot); ctx.stroke();
    ctx.restore();
    let maxFac = 1;
    for (const row of A.rows) maxFac = Math.max(maxFac, row.fac);
    if (maxFac < 1.001) r.label('nothing moved — every pair keeps its trained frequency', specX + specW / 2, facTop + facH / 2 + 4, { color: T.n9, font: '10.5px ui-monospace, monospace', align: 'center' });
    r.label(`×${s.toFixed(1)}`, specX - 6, facTop + 9, { color: T.n9, font: '10px ui-monospace, monospace', align: 'right' });
    r.label('×1', specX - 6, facBot, { color: T.n9, font: '10px ui-monospace, monospace', align: 'right' });

    // ---- the trade, computed: short-range separation vs extension factor ----
    const botTop = facBot + 30, botBot = H - 22;   // room below for the x-axis labels
    const trX = padL, trW = Math.round(W * 0.50) - padL, trY = botTop, trH = botBot - botTop;
    hit.trade = { x: trX, y: trY, w: trW, h: trH };
    ctx.save();
    ctx.fillStyle = T.n1; ctx.fillRect(trX, trY, trW, trH);
    ctx.strokeStyle = rgbaToken('n14', 0.08); ctx.lineWidth = 1;
    for (let q = 0; q <= 4; q++) {
      const y = trY + (q / 4) * trH;
      ctx.beginPath(); ctx.moveTo(trX, y); ctx.lineTo(trX + trW, y); ctx.stroke();
      r.label(`${100 - q * 25}%`, trX - 6, y + 3, { color: T.n9, font: '10px ui-monospace, monospace', align: 'right' });
    }
    ctx.restore();
    r.label('short-range resolution kept (% of un-extended)', pad, trY - 16, { color: T.n11, font: '11px ui-monospace, monospace' });
    r.label('extension factor s →', trX + trW, botBot + 12, { color: T.n10, font: '10px ui-monospace, monospace', align: 'right' });
    r.label('1×', trX, botBot + 12, { color: T.n9, font: '10px ui-monospace, monospace' });

    const xOfS = (v) => trX + ((v - 1) / (SMAX - 1)) * trW;
    const yOfPct = (v) => trY + (1 - Math.max(0, Math.min(100, v)) / 100) * trH;
    const NS = 65;
    for (const m of METHODS) {
      const active = m.id === st.method;
      ctx.save();
      ctx.strokeStyle = active ? methodColor(m.id) : alphaOf(methodColor(m.id), 0.42);
      ctx.lineWidth = active ? 2.6 : 1.3;
      ctx.beginPath();
      for (let k = 0; k < NS; k++) {
        const sv = 1 + (k / (NS - 1)) * (SMAX - 1);
        const x = xOfS(sv), y = yOfPct(meanSepAt(st.d, st.base, L, sv, m.id));
        k === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y);
      }
      ctx.stroke();
      ctx.restore();
    }
    // current-s marker + a dot per method
    ctx.save();
    ctx.strokeStyle = T.n11; ctx.lineWidth = grab === 'scale' ? 2.2 : 1.2; ctx.setLineDash([3, 3]);
    ctx.beginPath(); ctx.moveTo(xOfS(s), trY); ctx.lineTo(xOfS(s), trY + trH); ctx.stroke();
    ctx.setLineDash([]);
    for (const m of METHODS) {
      const y = yOfPct(meanSepAt(st.d, st.base, L, s, m.id));
      ctx.fillStyle = methodColor(m.id);
      ctx.beginPath(); ctx.arc(xOfS(s), y, m.id === st.method ? 4.4 : 2.8, 0, TAU); ctx.fill();
    }
    ctx.restore();
    r.label(`s = ${s.toFixed(1)}×  (drag ↔)`, xOfS(s), trY + 12, { color: T.n11, font: '10px ui-monospace, monospace', align: s > SMAX * 0.6 ? 'right' : 'left' });

    // ---- side-by-side comparison of all five methods at the current setting --
    const tbX = trX + trW + 26, tbW = W - pad - tbX;
    let ty = botTop + 4;
    r.label('all methods at this s, L and position', tbX, trY - 16, { color: T.n11, font: '11px ui-monospace, monospace' });
    const c1 = tbX + tbW * 0.50, c2 = tbX + tbW * 0.74, c3 = tbX + tbW;
    r.label('method', tbX, ty, { color: T.n10, font: '10px ui-monospace, monospace' });
    r.label('unseen', c1, ty, { color: T.n10, font: '10px ui-monospace, monospace', align: 'right' });
    r.label('res %', c2, ty, { color: T.n10, font: '10px ui-monospace, monospace', align: 'right' });
    r.label('rad/tok', c3, ty, { color: T.n10, font: '10px ui-monospace, monospace', align: 'right' });
    ty += 6;
    ctx.save(); ctx.strokeStyle = rgbaToken('n14', 0.14); ctx.lineWidth = 1;
    ctx.beginPath(); ctx.moveTo(tbX, ty); ctx.lineTo(c3, ty); ctx.stroke(); ctx.restore();
    const rowH = Math.max(15, Math.min(22, (botBot - ty - 6) / METHODS.length));
    for (const m of METHODS) {
      ty += rowH;
      const B = analyse({ ...st, method: m.id }, pos);
      const on = m.id === st.method;
      if (on) { ctx.save(); ctx.fillStyle = rgbaToken('n14', 0.06); ctx.fillRect(tbX - 4, ty - rowH + 4, (c3 - tbX) + 8, rowH); ctx.restore(); }
      ctx.save();
      ctx.fillStyle = methodColor(m.id);
      ctx.beginPath(); ctx.arc(tbX + 3, ty - 4, 3.2, 0, TAU); ctx.fill();
      ctx.restore();
      r.label(m.label.replace(/ \(.*\)$/, ''), tbX + 11, ty, { color: on ? T.n14 : T.n12, font: `${on ? 'bold ' : ''}10.5px ui-monospace, monospace` });
      r.label(String(B.ood), c1, ty, { color: B.ood ? T.bad : T.ok, font: 'bold 10.5px ui-monospace, monospace', align: 'right' });
      r.label(B.meanSep.toFixed(1), c2, ty, { color: T.n12, font: '10.5px ui-monospace, monospace', align: 'right' });
      r.label(fmt(B.fastSep), c3, ty, { color: T.n12, font: '10.5px ui-monospace, monospace', align: 'right' });
    }

    // ---- hover-to-inspect --------------------------------------------------
    if (page.pointer.over && !grab) {
      const pt = page.pointer;
      let tip = null;
      if (pt.x >= specX && pt.x <= specX + specW && pt.y >= specTop && pt.y <= facBot) {
        const i = Math.floor((pt.x - specX) / colW);
        const row = A.rows[Math.max(0, Math.min(A.np - 1, i))];
        tip = `pair ${row.i}   θ${row.i} = ${fmt(row.th)}   →  θ′ = ${fmt(row.eff)}  (÷${row.fac.toFixed(2)})\n` +
              `wavelength ${fmt(row.lam)} tok  →  ${fmt(row.lamEff)} tok\n` +
              `turns inside trained L: ${row.turns.toFixed(2)}   angles ever seen: 0…${fmt(row.cover)} rad\n` +
              `angle at p=${ifmt(pos)}: ${fmt(row.angle)} rad  (mod 2π = ${row.wrapped.toFixed(3)})\n` +
              (row.seen ? '✓ in distribution — this angle was seen in training'
                        : '✗ out of distribution — this angle was never seen');
      } else if (pt.x >= trX && pt.x <= trX + trW && pt.y >= trY && pt.y <= trY + trH) {
        const sv = 1 + Math.max(0, Math.min(1, (pt.x - trX) / trW)) * (SMAX - 1);
        tip = `extension factor ${sv.toFixed(1)}×  (context ${ifmt(Math.round(L * sv))})\n` +
              METHODS.map((m) => `  ${m.label}: ${meanSepAt(st.d, st.base, L, sv, m.id).toFixed(1)}%`).join('\n');
      }
      if (tip) page.setTip(tip);
    }

    // ---- readout -----------------------------------------------------------
    const temp = 0.1 * Math.log(Math.max(1, s)) + 1;
    let o = `${methodLabel(st.method)} · trained L = ${ifmt(L)} · s = ${s.toFixed(1)}× → target ${ifmt(target)} · head_dim ${st.d} (${A.np} pairs) · base ${ifmt(st.base)} · tier:${r.name}\n`;
    o += `position ${ifmt(pos)} (${(pos / L).toFixed(2)}× trained): ${A.ood} of ${A.np} pairs are at an angle never seen in training`;
    o += A.ood ? `  ✗  (the long-wavelength end)\n` : `  ✓\n`;
    o += `short-range cost: mean adjacent-token separation = ${A.meanSep.toFixed(1)}% of the un-extended model’s (higher is better; 100% = untouched) · in L2 ${A.l2Sep.toFixed(1)}% · fastest pair ${fmt(A.fastSep)} rad/token\n`;
    if (st.method === 'yarn') o += `YaRN also scales attention logits by 0.1·ln(s)+1 = ${temp.toFixed(3)} to compensate the longer sequence; pairs turning >${YARN_BETA}× inside L are left alone, <${YARN_ALPHA}× are fully interpolated.`;
    else if (st.method === 'none') o += `naive extrapolation pays nothing in resolution and buys nothing: every pair whose wavelength exceeds L is asked for an angle outside the arc it was trained on.`;
    else if (st.method === 'pi') o += `position interpolation divides EVERY position by s, so no pair leaves its trained arc — and every pair, including the fastest, loses the same factor of fine positional detail.`;
    else if (st.method === 'ntk') o += `NTK/frequency scaling rescales the base (base·s^(d/(d−2))), so pair 0 is untouched and the slowest pair is divided by exactly s, with a smooth ramp in between.`;
    else o += `LongRoPE searches a rescale factor per dimension rather than deriving one from a rule — the factor strip is not smooth. (Illustrative factors: real ones are per-model search results.)`;
    page.setReadout(o);
  },
}).then((page) => {
  const q = new URLSearchParams(location.search);
  const t = page.controls._transport;

  // Restore the deep-linked control state the framework mirrors into the URL.
  const num = (k, cast) => { if (q.has(k)) { const v = cast(q.get(k)); if (!Number.isNaN(v)) page.controls.set(k, v, { rebuild: true, silent: true }); } };
  if (q.has('method') && METHODS.some((m) => m.id === q.get('method'))) page.controls.set('method', q.get('method'), { silent: true });
  num('scale', parseFloat); num('ctx', (v) => parseInt(v, 10));
  num('d', (v) => parseInt(v, 10)); num('base', (v) => parseInt(v, 10));
  if (t) t.rebuild();

  // ?pos=N — pin an exact position (deterministic frame for a capture).
  if (q.has('pos')) { const v = parseFloat(q.get('pos')); if (!Number.isNaN(v)) posOverride = v; }
  // ?step=N — seek the transport to that step.
  if (q.has('step') && t) { const si = parseInt(q.get('step'), 10); if (!Number.isNaN(si)) { t.seek(si); posOverride = null; } }
  // ?hover=x,y — stand in for a real cursor (a capture has no pointer).
  if (q.has('hover')) { const [hx, hy] = q.get('hover').split(',').map(Number); page.pointer.x = hx; page.pointer.y = hy; page.pointer.over = true; }
  if (q.has('pos') || q.has('step') || q.has('hover')) { if (t) t.pause(); }
  if (q.get('play') === '1' && t) t.play();
  page.redraw();
});
