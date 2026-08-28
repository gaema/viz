// attention-sinks concept page -- why a head parks most of its attention on the
// first token or two, why deleting them breaks the model, and the two shipped
// remedies that pull in OPPOSITE directions and both work.
//
// THE MECHANISM. Softmax must sum to 1. A head that has nothing it wants to
// read still has to put its whole probability mass somewhere, and the only
// positions every query can see are the earliest ones -- so it dumps the mass
// there. Those positions become "attention sinks": enormous attention weight,
// almost no contribution to the output vector. Evict them and the distribution
// that was leaning on them has to redistribute onto tokens that DO carry value,
// and the output moves a long way. StreamingLLM (Xiao et al.,
// https://arxiv.org/abs/2309.17453) is where this was named and measured; it
// reports that keeping a couple of early tokens resident is what stops
// perplexity from exploding under a sliding-window cache.
//
// THE TWO REMEDIES, and why they are opposites.
//   * a learned per-head SINK LOGIT appended to the softmax denominator -- an
//     explicit no-op option with no value vector behind it. Mass now has a
//     legitimate home, so it stops being parked on token 0. This is the
//     "off-by-one softmax" idea (Miller, "Attention Is Off By One"), shipped as
//     a learned per-head sink logit in current open-weight models.
//   * a per-head OUTPUT GATE -- the head scales its own contribution down
//     instead. It never needed to signal "nothing here" through the attention
//     distribution at all. Gated Attention (Qiu et al.,
//     https://arxiv.org/abs/2505.06708) reports that head-wise output gating
//     removes the sink outright.
//   One adds a place for the mass to go; the other removes the need for the
//   mass to mean anything. Both free the pinned cache slot, and both cost a
//   trained parameter the base model does not have.
//
// WHAT IS DERIVED HERE AND WHAT IS SHAPED. There is no model behind this page,
// so -- exactly as the sibling eviction page does -- the inputs are synthetic
// and deliberately shaped, and everything downstream is then computed honestly
// from them:
//   derived in-page: the softmax (including the extra denominator term), the
//     attention weights, the output vector, the contribution shares, the drift
//     when the sinks are deleted, every number in the readout and tooltips.
//   shaped inputs: a content-independent positional bias decaying from position
//     0 (the learned sink bias), per-position value vectors whose norm is tiny
//     at the sink positions (which is the reported shape: high attention, near-
//     zero value), and per-head variation in sink strength.
//   modelled, and labelled as such on screen: BOTH remedies shrink that learned
//     bias in proportion to how much no-op capacity they hand the head. That
//     relationship is the reported result -- a trained model that has somewhere
//     legitimate to put its spare mass does not learn the big bias in the first
//     place -- and it is not something this page can derive from one forward
//     pass. What IS derived, live, is everything the shrunken bias and the extra
//     denominator term then do to the distribution and the output.
//
// Interactive per the shared render framework's contract: the query position
// AUTO-PLAYS + LOOPS; DRAG the sink-logit handle (the no-op column at the right)
// or the gate handle (under the output row) and watch mass leave position 0
// live; TOGGLE "delete the sinks" and watch the distribution collapse and the
// output drift; HOVER any position for its weight, its value norm, its share of
// the output and the arithmetic behind all three.
import { mount } from '../framework/layout.js';
import { seededRandn } from '../framework/tensor.js';
import { T, alphaOf, mixColor, inkOn } from '../framework/theme.js';

const D = 6;                 // value-vector width (small enough to print)
const SINK_B = 5.4;          // learned, content-INDEPENDENT bias at position 0
const SINK_TAU = 1.5;        // how fast that bias falls off with position
const VN_TAU = 2.4;          // where the value norm recovers from its sink floor
const ELL_MIN = -2, ELL_MAX = 6;   // sink-logit range (also the drag range)

// The two shaped curves, kept side by side because the whole lesson is that
// they point in OPPOSITE directions at the same positions.
const sinkBias = (i) => Math.exp(-((i / SINK_TAU) ** 2));           // big at 0, gone by 3
const valueScale = (i) => 0.012 + 0.988 * (1 - Math.exp(-((i / VN_TAU) ** 6)));  // ~0 at 0-1, ~1 by 3

// How much legitimate no-op capacity a remedy hands the head, in [0,1]. The
// learned bias is scaled by (1 - capacity): this is the MODELLED part (see the
// header note) and it is the same relationship for both remedies, which is what
// makes them comparable even though they act at opposite ends of the head.
function noopCapacity(remedy, ell, g) {
  if (remedy === 'sinklogit') return clamp((ell - ELL_MIN) / (ELL_MAX - ELL_MIN), 0, 1);
  if (remedy === 'gate') return 1 - clamp(g, 0, 1);
  return 0;
}

const REMEDIES = [
  { key: 'none', name: 'none (base model)',
    desc: 'the softmax has no no-op option and the head has no volume control, so the mass lands on position 0' },
  { key: 'sinklogit', name: 'learned sink logit (extra denominator term)',
    desc: 'an extra logit ℓ in the denominator with no value behind it — an explicit no-op, so the mass has a legitimate home' },
  { key: 'gate', name: 'output gate (head scales itself down)',
    desc: 'the head multiplies its own output by g instead of signalling “nothing here” through the attention distribution' },
];
const remedyAt = (k) => REMEDIES.find((r) => r.key === k) || REMEDIES[0];

const clamp = (v, lo, hi) => Math.max(lo, Math.min(hi, v));
const pct = (v) => (100 * v).toFixed(1) + '%';
const norm = (v) => { let s = 0; for (let i = 0; i < v.length; i++) s += v[i] * v[i]; return Math.sqrt(s); };

// The whole subject of this page is an EXTRA TERM in the softmax denominator,
// which tensor.js's softmax() has no way to express -- so the normalization is
// written out here, one line per piece, and the denominator is a value the
// readout and the tooltips can quote.
function attend(logit, live, ell) {
  const N = logit.length;
  let mx = -Infinity;
  for (let i = 0; i < N; i++) if (live[i]) mx = Math.max(mx, logit[i]);
  if (ell != null) mx = Math.max(mx, ell);
  if (!isFinite(mx)) mx = 0;
  const e = new Float64Array(N);
  let denom = 0;
  for (let i = 0; i < N; i++) if (live[i]) { e[i] = Math.exp(logit[i] - mx); denom += e[i]; }
  const eNoop = ell == null ? 0 : Math.exp(ell - mx);
  denom += eNoop;
  const a = new Float64Array(N);
  for (let i = 0; i < N; i++) a[i] = live[i] ? e[i] / denom : 0;
  return { a, aNoop: eNoop / denom, denom, mx };
}

// One head, one query position: weights -> output vector -> per-position share
// of the output's total contributed magnitude. `g` scales the head's output
// (the gate); it cancels out of the shares, which is the point of a gate.
function headOut(model, h, q, logit, live, ell, g) {
  const { a, aNoop, denom } = attend(logit, live, ell);
  const out = new Float64Array(D);
  const contrib = new Float64Array(logit.length);
  let total = 0;
  for (let i = 0; i <= q; i++) {
    if (!live[i]) continue;
    const v = model.v[h][i];
    for (let d = 0; d < D; d++) out[d] += a[i] * v[d];
    contrib[i] = a[i] * norm(v);
    total += contrib[i];
  }
  for (let d = 0; d < D; d++) out[d] *= g;
  const share = new Float64Array(logit.length);
  for (let i = 0; i <= q; i++) share[i] = total > 1e-12 ? contrib[i] / total : 0;
  return { a, aNoop, denom, out, contrib, share, outNorm: norm(out) };
}

// Synthetic-but-shaped inputs (see the header note). Deterministic in `seed`,
// so one URL replays one run.
function buildModel(st) {
  const N = st.N | 0, H = st.heads | 0;
  const zr = seededRandn((st.seed | 0) * 7919 + 13, H * N);
  const vr = seededRandn((st.seed | 0) * 104729 + 7, H * N * D);
  const hr = seededRandn((st.seed | 0) * 31 + 5, 8);
  const z = [], v = [], hs = [];
  for (let h = 0; h < H; h++) {
    hs.push(0.68 + 0.34 * Math.abs(hr[h % 8]));          // per-head sink strength
    const zi = new Float32Array(N), vi = [];
    for (let i = 0; i < N; i++) {
      zi[i] = 1.25 * zr[h * N + i];                      // content logit
      const scale = valueScale(i);                       // sinks carry almost no value
      const vec = new Float32Array(D);
      for (let d = 0; d < D; d++) vec[d] = scale * vr[(h * N + i) * D + d];
      vi.push(vec);
    }
    z.push(zi); v.push(vi);
  }
  return { N, H, z, v, hs };
}

let cur = null;

function simulate(st) {
  const model = buildModel(st);
  const N = model.N, H = model.H;
  const S = clamp(st.sinks | 0, 0, 4);
  const Wn = clamp(st.window | 0, 2, N);
  const remedy = st.remedy || 'none';
  const g = remedy === 'gate' ? clamp(+st.gate, 0, 1) : 1;
  const ell = remedy === 'sinklogit' ? clamp(+st.sinkl, ELL_MIN, ELL_MAX) : null;
  const cap = noopCapacity(remedy, +st.sinkl, +st.gate);
  const biasScale = 1 - cap;
  const steps = [];

  for (let q = 0; q < N; q++) {
    const heads = [];
    let mMass = 0, mShare = 0, mDrift = 0;
    let slots = 0, pinned = 0;
    for (let h = 0; h < H; h++) {
      const bias = new Float32Array(N), logit = new Float32Array(N);
      const live = new Uint8Array(N), liveDel = new Uint8Array(N);
      for (let i = 0; i <= q; i++) {
        // the bias is content-independent: it does not consult z at all. Either
        // remedy shrinks it via biasScale -- see the header note on what is
        // modelled and what is derived.
        bias[i] = SINK_B * model.hs[h] * sinkBias(i) * biasScale;
        logit[i] = model.z[h][i] + bias[i];
        const inWindow = i >= q - Wn + 1;
        const isSink = i < S;
        live[i] = (inWindow || isSink) ? 1 : 0;          // sinks are pinned: never evicted
        liveDel[i] = (inWindow && !isSink) ? 1 : 0;      // ...unless the reader deletes them
      }
      if (h === 0) {
        for (let i = 0; i <= q; i++) if (live[i]) slots++;
        for (let i = 0; i < Math.min(S, q + 1); i++) if (i < q - Wn + 1) pinned++;
      }
      const keep = headOut(model, h, q, logit, live, ell, g);
      const del = headOut(model, h, q, logit, liveDel, ell, g);
      let dd = 0; for (let d = 0; d < D; d++) dd += (del.out[d] - keep.out[d]) ** 2;
      const drift = keep.outNorm > 1e-9 ? Math.sqrt(dd) / keep.outNorm : 0;

      let sinkMass = 0, sinkShare = 0;
      for (let i = 0; i < Math.min(S, q + 1); i++) { sinkMass += keep.a[i]; sinkShare += keep.share[i]; }
      heads.push({ bias, logit, live, liveDel, keep, del, drift, sinkMass, sinkShare });
      mMass += sinkMass; mShare += sinkShare; mDrift += drift;
    }
    steps.push({
      q, heads, S, Wn, g, ell, cap, biasScale,
      agg: { sinkMass: mMass / H, sinkShare: mShare / H, drift: mDrift / H, slots, pinned },
      label: `query at position ${q} — ${q + 1} token${q ? 's' : ''} of context, ${slots} cache slot${slots === 1 ? '' : 's'} readable`,
    });
  }
  cur = { N, H, S, Wn, remedy, g, ell, cap, biasScale, model, steps };
  return steps;
}

// Rects captured in draw() so the pointer layer tests exactly what was painted.
let rRows = null, rNoop = null, rGate = null, rHeads = null;
let slotW = 0, x0 = 0, gridW = 0;
let grab = null;

const ROWS = [
  { key: 'logit',  t1: 'logit',     t2: 'content + bias' },
  { key: 'attn',   t1: 'attention', t2: 'a[q][i]' },
  { key: 'vnorm',  t1: 'value norm', t2: '‖v_i‖' },
  { key: 'share',  t1: 'output share', t2: 'a_i‖v_i‖ / Σ' },
];

mount({
  mount: 'body',
  title: 'attention-sinks — the position everything attends to and nothing reads',
  blurb: 'Softmax has to sum to 1. A head with nothing it wants to read still has to put its mass somewhere, so it parks it on a position every query can see — the first token. That is an attention sink: enormous attention weight, almost no contribution to the output. The two big numbers below the rows are the whole point, and they disagree wildly. Delete the sinks and watch the distribution collapse and the output drift. Then try the two shipped remedies, which pull in opposite directions: drag the sink-logit handle (an explicit no-op in the denominator) or the gate handle (the head turning its own volume down). Both free the pinned slot; both cost a trained parameter the base model does not have.',
  prefer: 'webgl2',
  aspect: '2 / 1',
  autoplay: true,
  animate: false,
  compare: { key: 'remedy', a: 'none', b: 'sinklogit', labelA: 'base model (no remedy)', labelB: 'learned sink logit', rebuild: true },
  challenges: [
    {
      goal: 'Read the gap: find a view where the sinks hold at least 6× more attention mass than their share of the output.',
      hint: 'the base model, well into the sequence — attention piles up on position 0 while its value vector stays tiny.',
      check: (api) => ({
        solved: (api.probe.sinkMass ?? 0) >= 6 * (api.probe.sinkShare ?? 1),
        detail: `mass ${pct(api.probe.sinkMass ?? 0)} vs output share ${pct(api.probe.sinkShare ?? 0)} — ratio ${((api.probe.sinkMass ?? 0) / Math.max(1e-9, api.probe.sinkShare ?? 0)).toFixed(1)}× (need ≥ 6×)`,
      }),
    },
    {
      goal: 'Break it: with no remedy, delete the sinks and push the output drift past 40%.',
      hint: 'turn the delete toggle on with remedy = none — the mass that was parked on position 0 has to land on tokens that actually carry value.',
      check: (api) => ({
        solved: api.state.remedy === 'none' && !!api.state.del && (api.probe.drift ?? 0) >= 0.4,
        detail: `remedy=${api.state.remedy} · deleted=${api.state.del ? 'yes' : 'no'} · drift ${pct(api.probe.drift ?? 0)} (need ≥ 40.0%)`,
      }),
    },
    {
      goal: 'Free the slot: with either remedy, get sink mass under 15% AND deletion drift under 10%.',
      hint: 'drag the sink logit ℓ up, or the gate g down — either way the head stops needing position 0, so evicting it costs almost nothing.',
      check: (api) => ({
        solved: api.state.remedy !== 'none' && (api.probe.sinkMass ?? 1) < 0.15 && (api.probe.drift ?? 1) < 0.10,
        detail: `remedy=${api.state.remedy} · sink mass ${pct(api.probe.sinkMass ?? 0)} (need < 15.0%) · drift ${pct(api.probe.drift ?? 0)} (need < 10.0%)`,
      }),
    },
  ],

  controls: (c, page) => {
    c.select('remedy', {
      label: 'remedy',
      value: 'none',
      options: REMEDIES.map((r) => ({ value: r.key, label: r.name })),
      rebuild: true,
    });
    c.slider('sinkl', { label: 'sink logit ℓ (drag the no-op column too)', min: ELL_MIN, max: ELL_MAX, step: 0.25, value: 0, rebuild: true });
    c.slider('gate', { label: 'output gate g (drag the handle too)', min: 0, max: 1, step: 0.05, value: 1, rebuild: true });
    c.toggle('del', { label: 'delete the sink tokens', value: false, rebuild: true });
    c.slider('sinks', { label: 'sink tokens S (pinned, never evicted)', min: 0, max: 4, step: 1, value: 2, rebuild: true });
    c.slider('window', { label: 'window W (older slots evicted)', min: 4, max: 48, step: 1, value: 48, rebuild: true });
    c.stepper('heads', { label: 'heads H', min: 1, max: 8, value: 4, rebuild: true });
    c.slider('head', { label: 'head shown in the rows', min: 0, max: 7, step: 1, value: 0, rebuild: false });
    c.stepper('N', { label: 'context length (tokens)', min: 8, max: 48, value: 24, rebuild: true });
    c.slider('seed', { label: 'seed (token content)', min: 0, max: 99, step: 1, value: 5, rebuild: true });
    c.transport({ compute: () => simulate(page.state), speed: 4, loop: true });
  },

  // Direct manipulation: grab the no-op column to raise/lower the sink logit ℓ,
  // or the gate handle to scale the head down. Either grab also switches the
  // remedy to the one being dragged, so the picture always matches the hand.
  onPointer: (page, ev) => {
    if (!cur || !rNoop) return;
    if (ev.type === 'down') {
      grab = null;
      const near = (r, mx, my) => r && ev.x >= r.x - mx && ev.x <= r.x + r.w + mx && ev.y >= r.y - my && ev.y <= r.y + r.h + my;
      if (near(rNoop, 14, 8)) grab = 'sinkl';
      else if (near(rGate, 10, 10)) grab = 'gate';
      if (grab) {
        const t = page.controls._transport; if (t) t.pause();
        page.controls.set('remedy', grab === 'sinkl' ? 'sinklogit' : 'gate', { rebuild: true });
      }
    } else if (ev.type === 'up' || ev.type === 'leave') {
      grab = null;
    }
    if (grab && (ev.type === 'move' || ev.type === 'down') && (page.pointer.down || ev.type === 'down')) {
      if (grab === 'sinkl') {
        // the column is a vertical scale: top = ℓ max, bottom = ℓ min.
        const f = 1 - clamp((ev.y - rNoop.y) / Math.max(1, rNoop.h), 0, 1);
        const v = Math.round((ELL_MIN + (ELL_MAX - ELL_MIN) * f) * 4) / 4;
        if (+page.state.sinkl !== v) page.controls.set('sinkl', v, { rebuild: true });
      } else {
        const f = clamp((ev.x - rGate.x) / Math.max(1, rGate.w), 0, 1);
        const v = Math.round(f * 20) / 20;
        if (+page.state.gate !== v) page.controls.set('gate', v, { rebuild: true });
      }
    }
  },

  draw: (page) => {
    const r = page.renderer, ctx = page.ctx, st = page.state;
    r.clear(T.n0);
    const sim = cur; if (!sim || !sim.steps.length) return;
    const N = sim.N, H = sim.H, S = sim.S;
    const s = page.step();
    const q = s ? clamp(s.q, 0, N - 1) : N - 1;
    const rec = sim.steps[q];
    const hi = clamp(st.head | 0, 0, H - 1);
    const hd = rec.heads[hi];
    const del = !!st.del;
    const view = del ? hd.del : hd.keep;
    const live = del ? hd.liveDel : hd.live;
    const remedy = remedyAt(sim.remedy);

    // ---- geometry -------------------------------------------------------
    const pad = 14, gut = 104, noopW = 40, headsW = 96;
    const topY = 34;
    x0 = pad + gut;
    gridW = Math.max(60, page.W - x0 - pad - noopW - 16 - headsW - 14);
    slotW = gridW / N;
    const panelH = 74, gateH = 24;      // gateH keeps the gate strip off row 4
    const availH = Math.max(150, page.H - topY - panelH - gateH - 10);
    const rowGap = 13;
    const rowH = (availH - ROWS.length * rowGap) / ROWS.length;
    const yOf = (k) => topY + k * (rowH + rowGap);
    rRows = { x: x0, y: topY, w: gridW, h: ROWS.length * (rowH + rowGap) };

    const lab = (txt, x, y, col, font) => r.label(txt, x, y, { color: col || T.n11, font: font || '11px ui-monospace, monospace' });
    const isSink = (i) => i < S;

    // shared bar painter: values in [0,1] of the row height, drawn up from base
    const bars = (yTop, get, colorOf) => {
      const base = yTop + rowH;
      for (let i = 0; i < N; i++) {
        const x = x0 + i * slotW, w = Math.max(1, slotW - 2);
        if (i > q) {                                   // not in the sequence yet
          ctx.fillStyle = alphaOf('n6', 0.16);
          ctx.fillRect(x + 1, base - 2, w, 2);
          continue;
        }
        const v = clamp(get(i), 0, 1);
        const h = Math.max(1.2, v * rowH);
        ctx.fillStyle = colorOf(i);
        ctx.fillRect(x + 1, base - h, w, h);
      }
      ctx.strokeStyle = alphaOf('n14', 0.22); ctx.lineWidth = 1;
      ctx.beginPath(); ctx.moveTo(x0, base + 0.5); ctx.lineTo(x0 + gridW, base + 0.5); ctx.stroke();
    };
    const dead = () => alphaOf('n9', 0.30);

    // ---- row 1: logits (content + the content-independent sink bias) -----
    let lMin = 0, lMax = 1e-6;
    for (let i = 0; i <= q; i++) { lMin = Math.min(lMin, hd.logit[i]); lMax = Math.max(lMax, hd.logit[i]); }
    const lSpan = Math.max(1e-6, lMax - lMin);
    const y0 = yOf(0);
    ctx.save();
    bars(y0, (i) => (hd.logit[i] - lMin) / lSpan, (i) => !live[i] ? dead() : isSink(i) ? T.warn : T.accent);
    // the bias portion, drawn as a lighter cap so the content-independent part
    // of the logit is visibly separable from the content part
    for (let i = 0; i <= q; i++) {
      if (!live[i] || hd.bias[i] <= 0.001) continue;
      const base = y0 + rowH;
      const hTot = ((hd.logit[i] - lMin) / lSpan) * rowH;
      const hBias = Math.min(hTot, (hd.bias[i] / lSpan) * rowH);
      ctx.fillStyle = alphaOf('warn', 0.55);
      ctx.fillRect(x0 + i * slotW + 1, base - hTot, Math.max(1, slotW - 2), hBias);
    }
    ctx.restore();
    lab(ROWS[0].t1, pad, y0 + 12, T.n12);
    lab(ROWS[0].t2, pad, y0 + 24, T.n10, '10px ui-monospace, monospace');
    lab(sim.cap > 0.001
      ? `shaded cap = learned bias, shrunk ×${sim.biasScale.toFixed(2)} by the remedy (modelled — see below)`
      : 'shaded cap = learned bias (content-independent: it never consults the token)',
      x0 + 2, y0 - 3, T.n9, '10px ui-monospace, monospace');

    // ---- row 2: the attention weights ------------------------------------
    const y1 = yOf(1);
    // The token bars are scaled to the largest TOKEN weight, not to the no-op
    // slot: once the no-op holds most of the mass, a shared scale flattens every
    // token to a hairline and the row stops teaching anything. The no-op column
    // beside it is drawn on its own honest 0-100% scale and prints its percent,
    // so nothing is hidden by the split.
    let aMax = 1e-6; for (let i = 0; i <= q; i++) aMax = Math.max(aMax, view.a[i]);
    ctx.save();
    bars(y1, (i) => view.a[i] / aMax, (i) => !live[i] ? dead() : isSink(i) ? T.warn : T.accent);
    ctx.restore();
    lab(ROWS[1].t1, pad, y1 + 12, T.n12);
    lab(ROWS[1].t2, pad, y1 + 24, T.n10, '10px ui-monospace, monospace');
    lab(del ? 'sinks DELETED — the mass had to go somewhere'
        : sim.ell != null ? `tokens scaled to their own max (${pct(aMax)}); the no-op column is a full 0–100% scale`
        : sim.cap > 0.001 ? `the gate shrank the bias, so the mass is spread over the tokens that carry value`
        : 'the sink columns are the tall ones',
      x0 + 2, y1 - 3, del ? T.bad : T.n9, '10px ui-monospace, monospace');

    // ---- the no-op column: the extra denominator term ---------------------
    const nx = x0 + gridW + 14;
    rNoop = { x: nx, y: y1, w: noopW - 6, h: rowH };
    ctx.save();
    ctx.fillStyle = alphaOf('n9', 0.12);
    ctx.fillRect(rNoop.x, rNoop.y, rNoop.w, rNoop.h);
    if (sim.ell != null) {
      const h = Math.max(1.2, clamp(view.aNoop, 0, 1) * rowH);
      ctx.fillStyle = T.violet;
      ctx.fillRect(rNoop.x, rNoop.y + rowH - h, rNoop.w, h);
      ctx.strokeStyle = grab === 'sinkl' ? T.violet : alphaOf('violet', 0.75);
      ctx.lineWidth = grab === 'sinkl' ? 2 : 1.2;
      ctx.strokeRect(rNoop.x - 0.5, rNoop.y - 0.5, rNoop.w + 1, rNoop.h + 1);
    } else {
      ctx.strokeStyle = alphaOf('n9', 0.6); ctx.setLineDash([2, 3]); ctx.lineWidth = 1;
      ctx.strokeRect(rNoop.x + 0.5, rNoop.y + 0.5, rNoop.w - 1, rNoop.h - 1);
      ctx.setLineDash([]);
    }
    ctx.font = '9px ui-monospace, monospace'; ctx.textAlign = 'center';
    ctx.fillStyle = sim.ell != null ? T.violet : T.n9;
    ctx.fillText('no-op', rNoop.x + rNoop.w / 2, rNoop.y - 10);
    ctx.fillText(sim.ell != null ? pct(view.aNoop) : 'absent', rNoop.x + rNoop.w / 2, rNoop.y + rowH + 11);
    ctx.fillText('◂▸', rNoop.x + rNoop.w / 2, rNoop.y + rowH + 21);
    ctx.restore();

    // ---- row 3: value norms ----------------------------------------------
    const y2 = yOf(2);
    let vMax = 1e-6; const vn = [];
    for (let i = 0; i < N; i++) { vn[i] = norm(sim.model.v[hi][i]); vMax = Math.max(vMax, vn[i]); }
    ctx.save();
    bars(y2, (i) => vn[i] / vMax, (i) => !live[i] ? dead() : isSink(i) ? alphaOf('warn', 0.55) : T.teal);
    ctx.restore();
    lab(ROWS[2].t1, pad, y2 + 12, T.n12);
    lab(ROWS[2].t2, pad, y2 + 24, T.n10, '10px ui-monospace, monospace');
    lab('the sink positions barely carry a value vector at all', x0 + 2, y2 - 3, T.n9, '10px ui-monospace, monospace');

    // ---- row 4: share of the output vector -------------------------------
    const y3 = yOf(3);
    let sMax = 1e-6; for (let i = 0; i <= q; i++) sMax = Math.max(sMax, view.share[i]);
    ctx.save();
    bars(y3, (i) => view.share[i] / sMax, (i) => !live[i] ? dead() : isSink(i) ? T.warn : T.violet);
    ctx.restore();
    lab(ROWS[3].t1, pad, y3 + 12, T.n12);
    lab(ROWS[3].t2, pad, y3 + 24, T.n10, '10px ui-monospace, monospace');
    lab('same head, same step — the sink columns are now no bigger than anyone else’s', x0 + 2, y3 - 3, T.n9, '10px ui-monospace, monospace');

    // ---- the gate handle -------------------------------------------------
    const gy = y3 + rowH + 12;
    rGate = { x: x0, y: gy, w: Math.min(gridW, 210), h: 7 };
    ctx.save();
    ctx.fillStyle = alphaOf('n9', 0.18);
    ctx.fillRect(rGate.x, rGate.y, rGate.w, rGate.h);
    const gOn = sim.remedy === 'gate';
    const gw = rGate.w * clamp(sim.g, 0, 1);
    ctx.fillStyle = alphaOf('ok', gOn ? (grab === 'gate' ? 0.75 : 0.5) : 0.18);
    ctx.fillRect(rGate.x, rGate.y, gw, rGate.h);
    ctx.strokeStyle = gOn ? T.okDeep : alphaOf('n9', 0.7); ctx.lineWidth = grab === 'gate' ? 2 : 1.3;
    ctx.beginPath(); ctx.moveTo(rGate.x + gw, rGate.y - 4); ctx.lineTo(rGate.x + gw, rGate.y + rGate.h + 4); ctx.stroke();
    ctx.fillStyle = ctx.strokeStyle; ctx.font = '10px ui-monospace, monospace'; ctx.textAlign = 'center';
    ctx.fillText('◂▸', rGate.x + gw, rGate.y + rGate.h + 13);
    ctx.restore();
    lab(`output gate g=${sim.g.toFixed(2)}${gOn ? '' : ' (off)'}`, pad, gy + 8, gOn ? T.okDeep : T.n9, '10px ui-monospace, monospace');

    // ---- per-head strip ---------------------------------------------------
    const hx = x0 + gridW + 14 + noopW + 8;
    rHeads = { x: hx, y: topY, w: headsW, h: Math.min(availH, H * 22 + 18) };
    lab('per head', hx, topY - 3, T.n11, '10px ui-monospace, monospace');
    lab('mass / share', hx, topY + 9, T.n9, '9px ui-monospace, monospace');
    ctx.save();
    for (let h = 0; h < H; h++) {
      const yy = topY + 16 + h * 22, bw = headsW - 26;
      const hh = rec.heads[h];
      ctx.fillStyle = h === hi ? T.n12 : T.n10;
      ctx.font = '9px ui-monospace, monospace'; ctx.textAlign = 'left';
      ctx.fillText('h' + h, hx, yy + 8);
      ctx.fillStyle = alphaOf('n9', 0.16); ctx.fillRect(hx + 18, yy, bw, 7);
      ctx.fillStyle = T.warn; ctx.fillRect(hx + 18, yy, bw * clamp(hh.sinkMass, 0, 1), 7);
      ctx.fillStyle = alphaOf('n9', 0.16); ctx.fillRect(hx + 18, yy + 8, bw, 7);
      ctx.fillStyle = T.violet; ctx.fillRect(hx + 18, yy + 8, bw * clamp(hh.sinkShare, 0, 1), 7);
      if (h === hi) {
        ctx.strokeStyle = T.n14; ctx.lineWidth = 1;
        ctx.strokeRect(hx - 2.5, yy - 2.5, headsW - 4, 20);
      }
    }
    ctx.restore();

    // ---- the panel: the two numbers that disagree ------------------------
    const py = topY + availH + gateH + 6;
    const mass = del ? 0 : rec.agg.sinkMass, share = del ? 0 : rec.agg.sinkShare;
    const ratio = share > 1e-9 ? mass / share : 0;
    ctx.save();
    ctx.font = '11px ui-monospace, monospace'; ctx.textAlign = 'left';
    const cardW = Math.min(230, (page.W - 2 * pad - 24) / 3);
    const card = (k, title, value, sub, col) => {
      const cx = pad + k * (cardW + 12);
      ctx.fillStyle = alphaOf(col, 0.10);
      ctx.fillRect(cx, py, cardW, panelH - 14);
      ctx.strokeStyle = alphaOf(col, 0.55); ctx.lineWidth = 1;
      ctx.strokeRect(cx + 0.5, py + 0.5, cardW - 1, panelH - 15);
      ctx.fillStyle = T.n11; ctx.font = '10px ui-monospace, monospace';
      ctx.fillText(title, cx + 8, py + 14);
      ctx.fillStyle = T[col] || col; ctx.font = '20px ui-monospace, monospace';
      ctx.fillText(value, cx + 8, py + 37);
      ctx.fillStyle = T.n10; ctx.font = '9px ui-monospace, monospace';
      ctx.fillText(sub, cx + 8, py + 51);
    };
    card(0, `attention mass on the ${S} sink${S === 1 ? '' : 's'}`, pct(mass),
      del ? 'deleted — they are not there to hold any' : 'what the softmax hands them', 'warn');
    card(1, 'their share of the output', pct(share),
      del ? 'the mass moved onto the tokens above' : ratio >= 1.05 ? `${ratio.toFixed(1)}× less than the mass` : 'the two numbers agree', 'violet');
    card(2, 'cost of deleting the sinks', pct(rec.agg.drift), '‖o′ − o‖ / ‖o‖, averaged over heads', rec.agg.drift >= 0.25 ? 'bad' : 'ok');
    ctx.restore();

    page.probe = {
      q, N, H, S, sinkMass: rec.agg.sinkMass, sinkShare: rec.agg.sinkShare,
      drift: rec.agg.drift, aNoop: view.aNoop, slots: rec.agg.slots, pinned: rec.agg.pinned,
    };

    // ---- hover-to-inspect -------------------------------------------------
    if (page.pointer.over && !grab) {
      const p = page.pointer;
      const inNoop = rNoop && p.x >= rNoop.x - 4 && p.x <= rNoop.x + rNoop.w + 4 && p.y >= rNoop.y - 12 && p.y <= rNoop.y + rNoop.h + 12;
      const inGrid = p.x >= x0 && p.x <= x0 + gridW && p.y >= rRows.y - 6 && p.y <= rRows.y + rRows.h + 6;
      if (inNoop) {
        page.setTip(sim.ell == null
          ? ['the no-op slot — absent in the base model',
             'there is no term here, so the denominator is the tokens alone and',
             'every unit of probability must land on one of them.',
             'switch the remedy to “learned sink logit” (or drag this column) to add it'].join('\n')
          : [`no-op slot — an extra term in the softmax denominator, no value behind it`,
             `ℓ = ${sim.ell.toFixed(2)}   exp(ℓ − max) / denom = ${view.aNoop.toFixed(4)}  (${pct(view.aNoop)} of the mass)`,
             `it contributes 0 to the output vector by construction, so the head can`,
             `say “nothing here” without parking mass on a real token`].join('\n'));
      } else if (inGrid) {
        const i = clamp(Math.floor((p.x - x0) / slotW), 0, N - 1);
        const lines = [`position ${i}${i === q ? '  (the query itself)' : ''}${isSink(i) ? '  — SINK' : ''}`];
        if (i > q) {
          lines.push('not in the sequence yet');
        } else if (!live[i]) {
          lines.push(del && isSink(i)
            ? 'DELETED by the toggle — its mass was redistributed over the rest'
            : `EVICTED — outside the window W=${sim.Wn} (positions ${Math.max(0, q - sim.Wn + 1)}…${q}) and not pinned as a sink`);
        } else {
          const b = hd.bias[i], z = hd.logit[i] - b;
          lines.push(`logit = content ${z.toFixed(3)} + learned bias ${b.toFixed(3)} = ${hd.logit[i].toFixed(3)}`);
          lines.push(`weight a[${q}][${i}] = exp(${hd.logit[i].toFixed(2)} − max) / denom = ${view.a[i].toFixed(4)}   (${pct(view.a[i])})`);
          lines.push(`value norm ‖v_${i}‖ = ${vn[i].toFixed(3)}`);
          lines.push(`contribution a·‖v‖ = ${view.contrib[i].toFixed(4)}  →  ${pct(view.share[i])} of the output`);
          if (isSink(i)) lines.push(`pinned: this slot is never evicted, and it is ${(view.a[i] / Math.max(1e-9, view.share[i])).toFixed(1)}× bigger in attention than in output`);
        }
        page.setTip(lines.join('\n'));
      }
    }

    // ---- readout ----------------------------------------------------------
    let o = `remedy: ${remedy.name} — ${remedy.desc}\n`;
    o += `step ${q + 1}/${N}: head ${hi} of ${H}, ${rec.agg.slots} readable slot${rec.agg.slots === 1 ? '' : 's'} (window W=${sim.Wn}, ${S} pinned sink${S === 1 ? '' : 's'}`;
    o += rec.agg.pinned ? `, ${rec.agg.pinned} of them held only by the pin)` : ')';
    o += `    tier:${r.name}\n`;
    o += `sinks hold ${pct(rec.agg.sinkMass)} of the attention and contribute ${pct(rec.agg.sinkShare)} of the output`;
    o += sim.ell != null ? `; the no-op term holds ${pct(view.aNoop)}\n` : `\n`;
    o += del
      ? `DELETED: with the sinks gone the mass landed on tokens that do carry value, and the output moved ${pct(rec.agg.drift)}. That is the collapse — the head never wanted those tokens' content, it wanted somewhere to put the mass.`
      : sim.remedy === 'none'
      ? `The base model has no no-op option and no volume control, so the mass is stuck on position 0 — and that slot can never be evicted. Toggle "delete the sink tokens" to see what it is worth, then try a remedy.`
      : sim.remedy === 'sinklogit'
      ? `The extra denominator term is doing the job the sink token used to do: ℓ=${sim.ell.toFixed(2)} gives it ${pct(view.aNoop)} of the mass, and a head with somewhere legitimate to put spare mass does not learn the big positional bias — so it is shown shrunk ×${sim.biasScale.toFixed(2)} (that shrink is modelled after the reported result; everything downstream of it is computed here). Deleting the sinks now costs ${pct(rec.agg.drift)}. The price is one trained scalar per head the base model does not have.`
      : `The gate lets this head scale its own output to ${sim.g.toFixed(2)}, so it never needs the attention distribution to say "nothing here" — the positional bias is shown shrunk ×${sim.biasScale.toFixed(2)} (modelled after the reported result; everything downstream of it is computed here). Deleting the sinks now costs ${pct(rec.agg.drift)}. The price is one trained gate per head the base model does not have.`;
    page.setReadout(o);
  },
}).then((page) => {
  window.__attnSinksPage = page;
  const q = new URLSearchParams(location.search);
  const t = page.controls._transport;
  const num = (k, lo, ho, key, rebuild) => {
    if (!q.has(k)) return false;
    page.controls.set(key, clamp(+q.get(k), lo, ho), { rebuild });
    return true;
  };
  let dirty = false;
  if (q.has('remedy') && REMEDIES.some((x) => x.key === q.get('remedy'))) { page.controls.set('remedy', q.get('remedy'), { rebuild: true }); dirty = true; }
  dirty = num('n', 8, 48, 'N', true) || dirty;
  dirty = num('seed', 0, 99, 'seed', true) || dirty;
  dirty = num('heads', 1, 8, 'heads', true) || dirty;
  dirty = num('s', 0, 4, 'sinks', true) || dirty;
  dirty = num('w', 4, 48, 'window', true) || dirty;
  // ?sinkl= and ?gate= are the headless stand-ins for dragging the two handles
  // -- --screenshot has no pointer, so every handle needs a URL twin.
  dirty = num('sinkl', ELL_MIN, ELL_MAX, 'sinkl', true) || dirty;
  dirty = num('gate', 0, 1, 'gate', true) || dirty;
  if (q.has('del')) { page.controls.set('del', q.get('del') === '1', { rebuild: true }); dirty = true; }
  if (q.has('head')) page.controls.set('head', clamp(+q.get('head') | 0, 0, 7));
  if (q.has('hover')) {
    const [hx, hy] = q.get('hover').split(',').map(Number);
    page.pointer.x = hx; page.pointer.y = hy; page.pointer.over = true;
  }
  if (q.has('step') || q.has('hover') || dirty) { if (t) t.pause(); }
  // a hook that changed the sequence length must rebuild BEFORE the seek, or
  // ?step= is clamped against the previous (shorter) step list.
  if (t && dirty) t.rebuild();
  if (q.has('step') && t) t.seek(parseInt(q.get('step'), 10));
  if (q.get('play') === '1' && t) t.play();
  page.redraw();
});
