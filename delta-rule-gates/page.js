// delta-rule-gates concept page -- ONE recurrent matrix state, THREE gate designs.
//
// Every member of the delta-rule linear-attention family carries the same object:
// a fixed [d x d] matrix state S (a key->value associative memory), updated once
// per token. What changed across three generations is the GATE -- the thing that
// decides how much of S survives the step, and how much of the new value is
// committed:
//
//   S_t = diag(b_t) . S_{t-1} . (I - k_t k_t^T)  +  diag(w_t) . v_t k_t^T
//
//   Gated DeltaNet (GDN)  : b_t = w_t = alpha_t . 1   -- ONE scalar per token.
//   Kimi Delta Attention  : b_t = w_t = gamma_t       -- one rate PER CHANNEL.
//   Gated DeltaNet-2      : b_t and w_t INDEPENDENT   -- erase and write split.
//
// The last line is the whole lesson: GDN and KDA tie erase and write to ONE
// gate; GDN-2 separates them, so the state can keep old content (b near 1)
// while still admitting new content (w free), or the reverse.
//
// THE b/w FORM ABOVE IS THIS PAGE'S TEACHING STAND-IN, NOT ANY PAPER'S NOTATION.
// Every one of the three designs carries a decay AND a delta strength, and this
// two-gate form folds them together to put the three rows on one widget. The
// actual rules, so a reader can check the page against the sources:
//
//   GDN   S_t = alpha_t . S_{t-1} . (I - beta_t k k^T) + beta_t v k^T
//         scalar decay alpha_t, scalar delta strength beta_t
//   KDA   S_t = (I - beta_t k k^T) . Diag(alpha_t) . S_{t-1} + beta_t k v^T
//         CHANNEL-WISE decay alpha_t, still a SCALAR beta_t (arXiv:2510.26692)
//   GDN-2 S_t = (I - k (b_t . k)^T) . Diag(alpha_t) . S_{t-1} + k (w_t . v)^T
//         channel-wise decay, PLUS separate erase b_t (key axis, R^d_k) and
//         write w_t (value axis, R^d_v) -- the tie broken (arXiv:2605.22791)
//
// So "scalar -> per-channel -> decoupled" is right when "per-channel" is read as
// THE DECAY. It is NOT that KDA made the erase/write gate channel-wise: KDA's
// beta_t is a scalar per head, and GDN-2's own abstract says KDA "still uses a
// single scalar gate to control two different things". The reduction runs the
// other way from what this comment used to claim: GDN-2 recovers KDA when
// b_t = w_t = beta_t . 1 -- collapsing to the same SCALAR, with the channel-wise
// decay retained -- and recovers GDN by then also setting alpha_t = alpha_t . 1.
// Note beta cannot in fact be folded into a left diag(b): diag(b).S.(I-beta kk^T)
// is not of the form diag(b').S.(I-kk^T) unless beta = 1.
//
// The page draws the gate as the thing that differs: one handle that dims the
// whole matrix, a column of per-row handles that dims it unevenly, or two
// independent columns acting on the erase term and the write term. Drag them and
// watch an early-written memory decay under your hand.
//
// NOTE: `seq` (the token stream) is built ONLY by the transport compute -- never
// rebuilt inside draw(), which would wipe the drag edits.
import { mount } from '../framework/layout.js';
import { ramps, cellAt } from '../framework/render.js';
import { seededRandn } from '../framework/tensor.js';
import { T, alphaOf } from '../framework/theme.js';

const MODES = {
  gdn:  { label: 'GDN — scalar gate',        title: 'Gated DeltaNet' },
  kda:  { label: 'KDA — channel-wise gate',  title: 'Kimi Delta Attention' },
  gdn2: { label: 'GDN-2 — erase + write',    title: 'Gated DeltaNet-2' },
};

const clamp = (v, lo, hi) => (v < lo ? lo : v > hi ? hi : v);
const maxAbs = (a) => { let m = 1e-9; for (let i = 0; i < a.length; i++) if (Math.abs(a[i]) > m) m = Math.abs(a[i]); return m; };
// Read tokens at DRAW time -- never capture them at module scope.
const rowInk = (i) => [T.accent, T.warn, T.ok, T.violet, T.teal, T.goldDeep][i % 6];

let seq = null;                 // {K, V, L, d} -- the token stream (drag-editable)
let gates = { d: 0, g: [], b: [], w: [] };
let curStep = 0;
let vRect = null, sRect = null, gateHits = [], grab = null;

// Defaults chosen so each design's character is visible the moment the page opens:
// KDA's channel rates fan out, GDN-2 keeps everything (b near 1) while writing unevenly.
const defG = (i, d) => +(0.98 - 0.55 * (d > 1 ? i / (d - 1) : 0)).toFixed(2);
const defB = (i, d) => +(0.99 - 0.10 * (d > 1 ? i / (d - 1) : 0)).toFixed(2);
const defW = (i, d) => +(0.25 + 0.72 * (d > 1 ? i / (d - 1) : 0)).toFixed(2);

function resizeGates(d) {
  if (gates.d === d) return;
  gates = { d, g: [], b: [], w: [] };
  for (let i = 0; i < d; i++) { gates.g.push(defG(i, d)); gates.b.push(defB(i, d)); gates.w.push(defW(i, d)); }
}

function buildData(st) {
  const L = st.L | 0, d = st.d | 0, seed = st.seed | 0;
  resizeGates(d);
  const K = seededRandn(seed, [L, d], { std: 1 }).data;
  for (let t = 0; t < L; t++) {                       // unit keys: the delta projector needs |k| = 1
    let n = 0; for (let i = 0; i < d; i++) n += K[t * d + i] ** 2;
    n = Math.sqrt(n) || 1;
    for (let i = 0; i < d; i++) K[t * d + i] /= n;
  }
  const V = seededRandn(seed + 1, [L, d], { std: 1 }).data;
  seq = { K, V, L, d };
  return Array.from({ length: L }, (_, t) => ({ t, label: `step ${t}` }));
}

// b / w for one mode, given the live gate state + the scalar slider.
function gatesFor(mode, st, d) {
  const b = new Float32Array(d), w = new Float32Array(d);
  for (let i = 0; i < d; i++) {
    if (mode === 'gdn') { b[i] = st.alpha; w[i] = st.alpha; }
    else if (mode === 'kda') { b[i] = gates.g[i]; w[i] = gates.g[i]; }
    else { b[i] = gates.b[i]; w[i] = gates.w[i]; }
  }
  return { b, w };
}

// Run the recurrence over the whole sequence. Token 0 is the "probe" write: its
// key k* is what we later interrogate the memory with.
//
// P is that ONE write, propagated. The recurrence is linear in the state (the
// gates and the projector do not depend on S), so the step-0 write's own
// contribution to S can be carried separately: it is decayed and erased exactly
// as it is inside S, but never has later writes added to it. P_t k* is therefore
// "how much of the memory written at step 0 is still in the state at step t",
// with the other tokens' memories -- which are different associations, not
// survival of this one -- kept out of the number.
function scan(mode, st) {
  const { K, V, L, d } = seq;
  const { b, w } = gatesFor(mode, st, d);
  const S = new Float32Array(d * d), P = new Float32Array(d * d), states = [];
  const kStar = K.subarray(0, d);
  for (let t = 0; t < L; t++) {
    const k = K.subarray(t * d, t * d + d), v = V.subarray(t * d, t * d + d);
    const Sk = new Float32Array(d), Pk = new Float32Array(d);
    for (let i = 0; i < d; i++) {
      let a = 0, c = 0;
      for (let j = 0; j < d; j++) { a += S[i * d + j] * k[j]; c += P[i * d + j] * k[j]; }
      Sk[i] = a; Pk[i] = c;
    }
    const Er = new Float32Array(d * d), Wr = new Float32Array(d * d);
    for (let i = 0; i < d; i++) for (let j = 0; j < d; j++) {
      Er[i * d + j] = b[i] * (S[i * d + j] - Sk[i] * k[j]);   // gated erase term
      Wr[i * d + j] = w[i] * v[i] * k[j];                     // gated write term
      S[i * d + j] = Er[i * d + j] + Wr[i * d + j];
      // the step-0 write, carried forward: gated + erased, never re-written
      P[i * d + j] = t === 0 ? Wr[i * d + j] : b[i] * (P[i * d + j] - Pk[i] * k[j]);
    }
    const r = new Float32Array(d), rAll = new Float32Array(d);
    for (let i = 0; i < d; i++) {
      let a = 0, c = 0;
      for (let j = 0; j < d; j++) { a += P[i * d + j] * kStar[j]; c += S[i * d + j] * kStar[j]; }
      r[i] = a; rAll[i] = c;
    }
    states.push({ S: Float32Array.from(S), Er, Wr, r, rAll, Sk });
  }
  // retention: how much of the step-0 recall is left, per row and overall.
  const r0 = states[0].r;
  let den = 1e-9; for (let i = 0; i < d; i++) den += r0[i] * r0[i];
  for (const s of states) {
    let num = 0; for (let i = 0; i < d; i++) num += s.r[i] * r0[i];
    s.ret = num / den;
    s.retRow = new Float32Array(d);
    for (let i = 0; i < d; i++) s.retRow[i] = Math.abs(r0[i]) < 1e-4 ? NaN : s.r[i] / r0[i];
  }
  return { states, b, w, r0 };
}

const pct = (x) => (Number.isFinite(x) ? `${(x * 100).toFixed(1)}%` : '--');

// One 0..1 rate as a horizontal fill bar. Returns its rect for hit-testing.
function rateBar(page, x, y, w, h, val, color, text) {
  const ctx = page.ctx;
  ctx.save();
  ctx.fillStyle = T.n3; ctx.fillRect(x, y, w, h);
  ctx.fillStyle = alphaOf(color, 0.75); ctx.fillRect(x, y, w * clamp(val, 0, 1), h);
  ctx.strokeStyle = T.n7; ctx.lineWidth = 1; ctx.strokeRect(x + 0.5, y + 0.5, w - 1, h - 1);
  ctx.fillStyle = color; ctx.fillRect(x + w * clamp(val, 0, 1) - 1.5, y, 3, h);   // the handle
  ctx.font = '9px ui-monospace, monospace'; ctx.textBaseline = 'middle';
  ctx.fillStyle = T.n12; ctx.fillText(text, x + w + 5, y + h / 2);
  ctx.restore();
  return { x, y, w, h };
}

mount({
  mount: 'body',
  title: 'delta-rule-gates — scalar, then per-channel, then decoupled',
  blurb: 'Every delta-rule linear-attention layer carries the same object: a fixed [d×d] matrix state S — a key→value memory — rewritten once per token. What changed across three generations is the GATE. Gated DeltaNet decays the whole matrix at ONE scalar rate. Kimi Delta Attention makes that DECAY a diagonal — one rate per channel — so a feature can be kept while its neighbour is dropped, though its delta strength β is still a single scalar doing erase and write together. Gated DeltaNet-2 is the one that splits that scalar, into two independent channel-wise gates: b (how much old content survives) and w (how much new content is committed). Drag the gate handles and watch a memory written at step 0 decay under your hand; the closing panel scores the same sequence under all three designs. The b/w handles are a teaching stand-in that folds each design\'s decay and delta strength together — the README gives all three real update rules.',
  prefer: 'webgl2',
  aspect: '16 / 10',
  autoplay: true,
  controls: (c, page) => {
    c.select('mode', { label: 'gate design', options: Object.keys(MODES).map((k) => ({ value: k, label: MODES[k].label })), value: 'gdn' });
    c.slider('alpha', { label: 'GDN scalar α', min: 0.3, max: 1, step: 0.01, value: 0.9 });
    c.stepper('L', { label: 'sequence length (L)', min: 4, max: 12, value: 8 });
    c.stepper('d', { label: 'state dim (d)', min: 3, max: 6, value: 4 });
    c.slider('seed', { label: 'seed', min: 0, max: 99, step: 1, value: 7, rebuild: true });
    c.transport({ compute: () => buildData(page.state), speed: 1.2, loop: true });
  },
  challenges: [
    {
      goal: 'Keep one channel of the step-0 memory while forgetting another: get the gap between the best- and worst-retained channels above 0.20.',
      hint: 'Under the scalar gate the gap is exactly 0.00 and no drag can change it — one handle moves every row together. Switch to KDA (or GDN-2) and pull the per-channel rates apart.',
      check: (api) => ({ solved: (api.probe.spread || 0) > 0.20, detail: `channel-retention gap = ${(api.probe.spread || 0).toFixed(2)} (scalar gate: always 0.00)` }),
    },
    {
      goal: 'GDN-2 only: hold on to the old memory while writing almost nothing new — every write gate under 0.25, and end retention still above 25%.',
      hint: 'Erase and write are separate gates here: push the w column left and the b column right. Under GDN or KDA they are TIED, so you cannot write little without also forgetting fast.',
      check: (api) => ({
        solved: api.state.mode === 'gdn2' && (api.probe.maxW ?? 1) < 0.25 && (api.probe.retEnd || 0) > 0.25,
        detail: `mode=${api.state.mode} max w=${(api.probe.maxW ?? 1).toFixed(2)} end retention=${pct(api.probe.retEnd || 0)}`,
      }),
    },
  ],
  onPointer: (page, ev) => {
    if (!seq) return;
    const d = seq.d;
    if (ev.type === 'down') {
      grab = null;
      for (const h of gateHits) {
        if (ev.x >= h.x - 3 && ev.x <= h.x + h.w + 3 && ev.y >= h.y - 2 && ev.y <= h.y + h.h + 2) { grab = { gate: h }; break; }
      }
      if (!grab && vRect) { const c = cellAt(vRect, 1, d, ev.x, ev.y); if (c) grab = { v: c.c }; }
      if (grab && grab.gate) { setGate(grab.gate, ev.x, page); }
    } else if (ev.type === 'up' || ev.type === 'leave') grab = null;
    else if (ev.type === 'move' && grab && page.pointer.down) {
      if (grab.gate) setGate(grab.gate, ev.x, page);
      else { const idx = curStep * d + grab.v; seq.V[idx] = clamp(seq.V[idx] - ev.dy * 0.02, -3, 3); page.redraw(); }
    }
  },
  draw: (page) => {
    const r = page.renderer, ctx = page.ctx, st = page.state;
    if (!seq) return;                            // built by the transport compute
    const { K, V, L, d } = seq;
    resizeGates(d);
    const mode = MODES[st.mode] ? st.mode : 'gdn';
    r.clear(T.n0);
    gateHits = [];

    const run = scan(mode, st);
    const s = page.step(), t = s ? s.t : L - 1;
    curStep = t;
    const cs = run.states[t], end = run.states[L - 1];

    // ---- header: the one update rule, with this mode's gates named ---------
    r.label('Sₜ = diag(bₜ) · Sₜ₋₁ · (I − kₜkₜᵀ)  +  diag(wₜ) · vₜkₜᵀ', 16, 26, { color: T.n14, font: '13px ui-monospace, monospace' });
    const sub = {
      gdn:  'GDN:    bₜ = wₜ = αₜ·1        one scalar — the whole matrix forgets at one rate',
      kda:  'KDA:    bₜ = wₜ = γₜ          per-channel DECAY — rows forget independently (β itself stays scalar)',
      gdn2: 'GDN-2:  bₜ and wₜ independent — erase and write are separate channel-wise gates',
    }[mode];
    r.label(sub, 16, 44, { color: mode === 'gdn2' ? T.violet : T.n11, font: '11px ui-monospace, monospace' });

    const pad = 14, topY = 60;
    const splitY = Math.round(page.H * 0.60);
    const cw = 18;

    // ---- column A: the current token (k, v) -------------------------------
    const tw = d * cw;
    const krow = topY + 24, vrow = krow + 40;
    r.label(`token t = ${t}`, pad, topY + 10, { color: T.n12, font: '11px ui-monospace, monospace' });
    const kvdom = Math.max(maxAbs(K), maxAbs(V), 0.5);
    const kRect = { x: pad, y: krow, w: tw, h: 20 };
    r.heatmap({ data: K.subarray(t * d, t * d + d), rows: 1, cols: d }, { rows: 1, cols: d, rect: kRect, ramp: ramps.diverging, domain: [-kvdom, kvdom] });
    r.grid({ stroke: alphaOf('n14', 0.12) });
    vRect = { x: pad, y: vrow, w: tw, h: 20 };
    r.heatmap({ data: V.subarray(t * d, t * d + d), rows: 1, cols: d }, { rows: 1, cols: d, rect: vRect, ramp: ramps.diverging, domain: [-kvdom, kvdom] });
    r.grid({ stroke: alphaOf('n14', 0.12) });
    r.label('kₜ  key (unit)', pad, krow - 5, { color: T.accent, font: '10px ui-monospace, monospace' });
    r.label('vₜ  value — drag ↕', pad, vrow - 5, { color: T.warn, font: '10px ui-monospace, monospace' });
    r.label(t === 0 ? 'this IS the probe write' : 'probe key  k★ = k₀', pad, vrow + 34, { color: t === 0 ? T.ok : T.n10, font: '10px ui-monospace, monospace' });

    // ---- column B: the gate editor, row-aligned with the state matrix ------
    const gx = pad + Math.max(tw, 132) + 16;
    const two = mode === 'gdn2';
    const barW = two ? 52 : 74;
    const gateW = two ? barW * 2 + 40 : barW + 30;
    const Scell = clamp(Math.min((splitY - topY - 46) / d, (page.W - gx - gateW - 200) / d), 15, 34);
    const Sy = topY + 26;
    const rh = Math.max(11, Scell - 5);

    r.label(two ? 'gates b · w' : mode === 'kda' ? 'gate γ' : 'gate α', gx, Sy - 12, { color: T.n12, font: '11px ui-monospace, monospace' });
    for (let i = 0; i < d; i++) {
      const y = Sy + i * Scell + (Scell - rh) / 2;
      if (two) {
        gateHits.push({ ...rateBar(page, gx, y, barW, rh, gates.b[i], T.violet, gates.b[i].toFixed(2)), kind: 'b', i });
        gateHits.push({ ...rateBar(page, gx + barW + 32, y, barW, rh, gates.w[i], T.warn, gates.w[i].toFixed(2)), kind: 'w', i });
      } else if (mode === 'kda') {
        gateHits.push({ ...rateBar(page, gx, y, barW, rh, gates.g[i], T.violet, gates.g[i].toFixed(2)), kind: 'g', i });
      } else {
        gateHits.push({ ...rateBar(page, gx, y, barW, rh, st.alpha, T.violet, st.alpha.toFixed(2)), kind: 'a', i });
      }
    }
    if (mode === 'gdn') {
      ctx.save();
      ctx.strokeStyle = alphaOf(T.violet, 0.7); ctx.lineWidth = 1.4;
      ctx.beginPath(); ctx.moveTo(gx - 6, Sy + 2); ctx.lineTo(gx - 10, Sy + 2); ctx.lineTo(gx - 10, Sy + d * Scell - 2); ctx.lineTo(gx - 6, Sy + d * Scell - 2); ctx.stroke();
      ctx.restore();
      r.label('one handle — all rows locked together', gx, Sy + d * Scell + 26, { color: T.bad, font: '10px ui-monospace, monospace' });
    } else if (mode === 'kda') {
      r.label('one handle per row — drag ↔', gx, Sy + d * Scell + 26, { color: T.ok, font: '10px ui-monospace, monospace' });
    } else {
      r.label('erase', gx, Sy - 1, { color: T.violet, font: '9px ui-monospace, monospace' });
      r.label('write', gx + barW + 32, Sy - 1, { color: T.warn, font: '9px ui-monospace, monospace' });
      r.label('two independent handles per row', gx, Sy + d * Scell + 26, { color: T.ok, font: '10px ui-monospace, monospace' });
    }

    // ---- column C: the state matrix S -------------------------------------
    const Sx = gx + gateW + 26;
    const sdom = Math.max(maxAbs(cs.S), 1e-3);
    sRect = { x: Sx, y: Sy, w: d * Scell, h: d * Scell };
    r.label('S [d×d]', Sx, Sy - 12, { color: T.n12, font: '11px ui-monospace, monospace' });
    r.heatmap({ data: cs.S, rows: d, cols: d }, { rows: d, cols: d, rect: sRect, ramp: ramps.diverging, domain: [-sdom, sdom] });
    r.grid({ stroke: alphaOf('n14', 0.12) });
    ctx.save();
    ctx.font = '9px ui-monospace, monospace'; ctx.fillStyle = T.n9; ctx.textAlign = 'center';
    ctx.fillText('key dim j →', Sx + d * Scell / 2, Sy + d * Scell + 12);
    ctx.restore();
    // per-row tint tying gate row i to state row i
    ctx.save();
    for (let i = 0; i < d; i++) { ctx.fillStyle = alphaOf(rowInk(i), 0.85); ctx.fillRect(Sx - 5, Sy + i * Scell + 2, 3, Scell - 4); }
    ctx.restore();

    // ---- column D: the two terms of the update ----------------------------
    const wcell = Math.max(8, Scell * 0.58);
    const Ex = Sx + d * Scell + 34, Ey = Sy + 6;
    const edom = Math.max(maxAbs(cs.Er), 1e-3), wdom = Math.max(maxAbs(cs.Wr), 1e-3);
    r.heatmap({ data: cs.Er, rows: d, cols: d }, { rows: d, cols: d, rect: { x: Ex, y: Ey, w: d * wcell, h: d * wcell }, ramp: ramps.diverging, domain: [-edom, edom] });
    r.grid({ stroke: alphaOf('n14', 0.10) });
    r.label('kept', Ex, Ey - 6, { color: T.violet, font: '9px ui-monospace, monospace' });
    const Wy = Ey + d * wcell + 24;
    r.heatmap({ data: cs.Wr, rows: d, cols: d }, { rows: d, cols: d, rect: { x: Ex, y: Wy, w: d * wcell, h: d * wcell }, ramp: ramps.diverging, domain: [-wdom, wdom] });
    r.grid({ stroke: alphaOf('n14', 0.10) });
    r.label('written', Ex, Wy - 6, { color: T.warn, font: '9px ui-monospace, monospace' });

    // ---- column E: recall of the step-0 memory ----------------------------
    const rx = Ex + d * wcell + 40, rw = 58;
    const rdom = Math.max(maxAbs(run.r0), maxAbs(cs.r), maxAbs(cs.rAll), 0.3);
    r.label('memory', rx, Sy - 12, { color: T.ok, font: '11px ui-monospace, monospace' });
    ctx.save();
    for (let i = 0; i < d; i++) {
      const y0 = Sy + i * Scell;
      ctx.strokeStyle = T.n6; ctx.strokeRect(rx + 0.5, y0 + 0.5, rw - 1, Scell - 1);
      const inner = rw - 6, w0 = Math.abs(run.r0[i]) / rdom * inner, w1 = Math.abs(cs.r[i]) / rdom * inner, wa = Math.abs(cs.rAll[i]) / rdom * inner;
      ctx.fillStyle = alphaOf(T.n9, 0.5); ctx.fillRect(rx + 3, y0 + Scell * 0.22, w0, 4);        // magnitude as written at step 0
      ctx.fillStyle = alphaOf(rowInk(i), 0.95); ctx.fillRect(rx + 3, y0 + Scell * 0.5, w1, 5);   // how much of it is left
      ctx.fillStyle = alphaOf('n14', 0.5); ctx.fillRect(rx + 3 + Math.min(wa, inner) - 1, y0 + Scell * 0.76, 2, 5);  // full S·k★
    }
    ctx.restore();
    r.label('grey = written', rx, Sy + d * Scell + 12, { color: T.n9, font: '9px ui-monospace, monospace' });
    r.label('bar = left', rx, Sy + d * Scell + 24, { color: T.n9, font: '9px ui-monospace, monospace' });
    r.label('tick = S·k★', rx, Sy + d * Scell + 36, { color: T.n9, font: '9px ui-monospace, monospace' });

    // ---- bottom-left: retention of the step-0 memory over the sequence ----
    const cy0 = splitY + 36, cy1 = page.H - 30, cx0 = pad + 34, cx1 = Math.round(page.W * 0.53);
    const yOf = (v) => cy1 - clamp(v, -0.15, 1.15) * (cy1 - cy0) / 1.15;
    const xOf = (i) => cx0 + (L > 1 ? (i / (L - 1)) * (cx1 - cx0) : 0);
    r.label(`step-0 memory still in the state — ${MODES[mode].title}`, pad, splitY + 12, { color: T.n12, font: '11px ui-monospace, monospace' });
    ctx.save();
    ctx.strokeStyle = alphaOf('n14', 0.18); ctx.lineWidth = 1;
    ctx.beginPath(); ctx.moveTo(cx0, yOf(1)); ctx.lineTo(cx1, yOf(1)); ctx.moveTo(cx0, yOf(0)); ctx.lineTo(cx1, yOf(0)); ctx.stroke();
    ctx.font = '9px ui-monospace, monospace'; ctx.fillStyle = T.n9; ctx.textAlign = 'right';
    ctx.fillText('100%', cx0 - 5, yOf(1) + 3); ctx.fillText('0%', cx0 - 5, yOf(0) + 3);
    ctx.textAlign = 'left';
    for (let i = 0; i < d; i++) {                       // per-channel retention
      ctx.strokeStyle = alphaOf(rowInk(i), 0.85); ctx.lineWidth = 1.6; ctx.beginPath();
      let started = false;
      for (let p = 0; p < L; p++) { const v = run.states[p].retRow[i]; if (!Number.isFinite(v)) continue; if (!started) { ctx.moveTo(xOf(p), yOf(v)); started = true; } else ctx.lineTo(xOf(p), yOf(v)); }
      ctx.stroke();
    }
    ctx.strokeStyle = T.n13; ctx.lineWidth = 2.4; ctx.setLineDash([4, 3]); ctx.beginPath();
    for (let p = 0; p < L; p++) { const y = yOf(run.states[p].ret); p === 0 ? ctx.moveTo(xOf(p), y) : ctx.lineTo(xOf(p), y); }
    ctx.stroke(); ctx.setLineDash([]);
    ctx.strokeStyle = alphaOf(T.accent, 0.8); ctx.lineWidth = 1.2;               // the step cursor
    ctx.beginPath(); ctx.moveTo(xOf(t), cy0 - 4); ctx.lineTo(xOf(t), cy1 + 4); ctx.stroke();
    ctx.fillStyle = T.n9; ctx.font = '9px ui-monospace, monospace';
    ctx.fillText('step 0', cx0, cy1 + 12); ctx.fillText(`step ${L - 1}`, cx1 - 34, cy1 + 12);
    ctx.fillStyle = T.n11; ctx.fillText('dashed = whole memory · colours = per channel', cx0, cy0 - 10);
    ctx.restore();

    // ---- bottom-right: the same sequence, scored under all three gates ----
    const bx0 = Math.round(page.W * 0.58), bw = (page.W - pad - bx0) / 3;
    r.label('three designs, same sequence', bx0, splitY + 12, { color: T.n12, font: '11px ui-monospace, monospace' });
    const finals = {};
    ctx.save();
    ctx.font = '10px ui-monospace, monospace';
    Object.keys(MODES).forEach((m, mi) => {
      const rn = m === mode ? run : scan(m, st), fin = rn.states[L - 1];
      finals[m] = fin;
      const x = bx0 + mi * bw, barTop = cy0 + 16, barH = cy1 - barTop - 14;
      ctx.fillStyle = m === mode ? T.n13 : T.n10;
      ctx.fillText(m === 'gdn2' ? 'GDN-2' : m.toUpperCase(), x, cy0 + 8);
      ctx.fillStyle = m === mode ? T.accent : T.n11;
      ctx.fillText(pct(fin.ret), x + 52, cy0 + 8);
      const cwid = Math.min(16, (bw - 20) / d);
      for (let i = 0; i < d; i++) {                     // per-channel retention at the end
        const v = Number.isFinite(fin.retRow[i]) ? clamp(fin.retRow[i], 0, 1.15) : 0;
        const h = v * barH / 1.15, bxx = x + i * cwid;
        ctx.fillStyle = T.n3; ctx.fillRect(bxx, barTop, cwid - 3, barH);
        ctx.fillStyle = alphaOf(rowInk(i), m === mode ? 0.95 : 0.4); ctx.fillRect(bxx, barTop + barH - h, cwid - 3, h);
      }
      ctx.fillStyle = T.n9; ctx.font = '9px ui-monospace, monospace';
      ctx.fillText(m === 'gdn' ? 'all rows equal' : m === 'kda' ? 'rows differ' : 'erase ≠ write', x, cy1 + 4);
      ctx.font = '10px ui-monospace, monospace';
    });
    ctx.restore();

    // ---- hover: value + the update that produced it -----------------------
    if (page.pointer.over && !grab) {
      const p = page.pointer;
      const hs = cellAt(sRect, d, d, p.x, p.y);
      if (hs) {
        const i = hs.r, j = hs.c, k = K.subarray(t * d, t * d + d), v = V.subarray(t * d, t * d + d);
        const prev = t > 0 ? run.states[t - 1].S[i * d + j] : 0;
        page.setTip(
          `S[val ${i}, key ${j}] = ${cs.S[i * d + j].toFixed(4)}\n` +
          `  = b[${i}]·(Sprev − (Sprev·k)[${i}]·k[${j}]) + w[${i}]·v[${i}]·k[${j}]\n` +
          `  = ${run.b[i].toFixed(2)}·(${prev.toFixed(3)} − ${cs.Sk[i].toFixed(3)}·${k[j].toFixed(3)}) + ${run.w[i].toFixed(2)}·${v[i].toFixed(3)}·${k[j].toFixed(3)}\n` +
          `  kept ${cs.Er[i * d + j].toFixed(4)} + written ${cs.Wr[i * d + j].toFixed(4)}`);
      } else {
        let hit = false;
        for (const h of gateHits) {
          if (p.x >= h.x && p.x <= h.x + h.w && p.y >= h.y && p.y <= h.y + h.h) {
            const nm = h.kind === 'a' ? 'α (scalar — shared by every row)' : h.kind === 'g' ? `γ[${h.i}] (channel ${h.i} — KDA's per-channel decay; its β is scalar and ties erase to write)` : h.kind === 'b' ? `b[${h.i}] (erase — how much of row ${h.i} survives)` : `w[${h.i}] (write — how much of v[${h.i}] is committed)`;
            const val = h.kind === 'a' ? st.alpha : h.kind === 'g' ? gates.g[h.i] : h.kind === 'b' ? gates.b[h.i] : gates.w[h.i];
            page.setTip(`${nm} = ${val.toFixed(2)}\ndrag ↔ to change`);
            hit = true; break;
          }
        }
        if (!hit && vRect) { const c = cellAt(vRect, 1, d, p.x, p.y); if (c) page.setTip(`v[${c.c}] = ${V[t * d + c.c].toFixed(3)}\nthe value written this step — drag ↕`); }
      }
    }

    // ---- probes for challenge mode ---------------------------------------
    let lo = Infinity, hi = -Infinity;
    for (let i = 0; i < d; i++) { const v = end.retRow[i]; if (Number.isFinite(v)) { lo = Math.min(lo, v); hi = Math.max(hi, v); } }
    page.probe = {
      spread: Number.isFinite(hi - lo) ? hi - lo : 0,
      maxW: mode === 'gdn2' ? Math.max(...gates.w) : 1,
      retEnd: end.ret,
    };

    const gateTxt = mode === 'gdn' ? `α=${st.alpha.toFixed(2)}` : mode === 'kda' ? `γ=[${gates.g.map((x) => x.toFixed(2)).join(' ')}]` : `b=[${gates.b.map((x) => x.toFixed(2)).join(' ')}] w=[${gates.w.map((x) => x.toFixed(2)).join(' ')}]`;
    page.setReadout(
      `${MODES[mode].title}: ${sub.split(':').slice(1).join(':').trim()}   ${gateTxt}   tier:${r.name}\n` +
      `step ${t}/${L - 1}: ${pct(cs.ret)} of the memory written at step 0 is still in the state.\n` +
      `survives to step ${L - 1} — GDN ${pct(finals.gdn.ret)} · KDA ${pct(finals.kda.ret)} · GDN-2 ${pct(finals.gdn2.ret)}` +
      `   (per-channel spread at the end: GDN ${spreadOf(finals.gdn, d).toFixed(2)} · KDA ${spreadOf(finals.kda, d).toFixed(2)} · GDN-2 ${spreadOf(finals.gdn2, d).toFixed(2)})`);
  },
}).then(boot);

function spreadOf(state, d) {
  let lo = Infinity, hi = -Infinity;
  for (let i = 0; i < d; i++) { const v = state.retRow[i]; if (Number.isFinite(v)) { lo = Math.min(lo, v); hi = Math.max(hi, v); } }
  return Number.isFinite(hi - lo) ? hi - lo : 0;
}

function setGate(h, x, page) {
  const v = +clamp((x - h.x) / h.w, 0.02, 1).toFixed(2);
  if (h.kind === 'a') page.controls.set('alpha', v);
  else if (h.kind === 'g') gates.g[h.i] = v;
  else if (h.kind === 'b') gates.b[h.i] = v;
  else gates.w[h.i] = v;
  page.redraw();
}

// URL hooks: every draggable handle has one, so a manipulated view is
// reproducible (and headless-screenshottable) without a pointer.
//   ?mode=gdn|kda|gdn2  ?alpha=0.9  ?g=.98,.8,.6,.4  ?b=...  ?w=...
//   ?v=i,val (set component i of the current step's value)  ?step=N  ?play=1
function boot(page) {
  window.__gatePage = page;
  const q = new URLSearchParams(location.search);
  const tp = page.controls._transport;
  const vec = (name, arr) => { if (!q.has(name)) return; q.get(name).split(',').map(Number).forEach((v, i) => { if (i < arr.length && Number.isFinite(v)) arr[i] = clamp(v, 0, 1); }); };
  if (q.has('mode') && MODES[q.get('mode')]) page.controls.set('mode', q.get('mode'));
  for (const k of ['L', 'd', 'seed']) if (q.has(k)) page.controls.set(k, parseInt(q.get(k), 10), { rebuild: true });
  if (q.has('alpha')) page.controls.set('alpha', clamp(parseFloat(q.get('alpha')), 0.3, 1));
  if (q.has('step') || q.has('v') || q.has('hover')) { if (tp) tp.pause(); }
  if (q.has('step') && tp) tp.seek(parseInt(q.get('step'), 10));
  vec('g', gates.g); vec('b', gates.b); vec('w', gates.w);
  if (q.has('v') && seq) {
    const [i, val] = q.get('v').split(',').map(Number);
    const tt = tp && tp.index >= 0 ? tp.index : seq.L - 1;
    if (i >= 0 && i < seq.d && Number.isFinite(val)) seq.V[tt * seq.d + i] = clamp(val, -3, 3);
  }
  if (q.has('hover')) { const [hx, hy] = q.get('hover').split(',').map(Number); page.pointer.x = hx; page.pointer.y = hy; page.pointer.over = true; }
  if (q.get('play') === '1' && tp) tp.play();
  page.redraw();
}
