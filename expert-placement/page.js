// expert-placement -- WHERE an expert lives, which is a different question from
// which expert a token picks.
//
// The routing page answers "which experts does this token want". This page
// answers "and where do those experts physically live", because that is what
// decides how much of the batch crosses the interconnect. Same routing
// decisions, different placement, wildly different cost.
//
// THE MECHANISM. Under expert parallelism each device holds a SUBSET of the
// experts. A token whose chosen expert lives elsewhere has to be shipped to
// that device and its result shipped back -- the all-to-all. The dispatch sends
// a token once per DISTINCT destination device, not once per expert, so:
//
//   * co-locating experts that are frequently chosen TOGETHER collapses a
//     token's destination set and keeps its work local;
//   * scattering them maximises the number of devices each token must visit;
//   * expert popularity is heavily skewed in practice, so a hot expert sitting
//     alone on one device makes that device the bottleneck while the others
//     idle -- the step waits for the slowest device either way.
//
// Two shipped responses are drawn here: REPLICATING the hottest experts onto
// every device (they then always execute at home -- paid for in memory), and
// REBALANCING the placement periodically from observed load and co-occurrence.
//
// Public sources for the mechanism: the DeepSeek-V3 technical report
// (https://arxiv.org/abs/2412.19437), whose deployment section describes
// redundant copies of high-load experts plus periodic rebalancing of the expert
// placement from observed load; and the published expert-parallel
// load-balancing literature, e.g. FasterMoE (https://arxiv.org/abs/2203.10924),
// SmartMoE (https://arxiv.org/abs/2304.11414) and Tutel
// (https://arxiv.org/abs/2206.03382), which all treat placement -- not routing
// -- as the tunable. Interconnect bandwidths here are reader-set sliders whose
// defaults are public nominal figures, not measurements.
//
// Interactive per the shared framework's contract:
//  - TRANSPORT walks one MoE layer: routing -> dispatch -> compute -> combine,
//    autoplaying and looping.
//  - DIRECT MANIPULATION: pick an expert chip up and DROP IT on another device
//    -- traffic, per-device load and step time recompute immediately. Drag the
//    popularity ribbon vertically to change the SKEW and horizontally to change
//    how many hot experts are REPLICATED.
//  - HOVER a device for what it holds, its load and its idle time; hover a wire
//    for the tokens crossing it and which experts pulled them there.
//  - Buttons place by round-robin (the named baseline), by observed
//    co-occurrence + load (rebalance), and worst-case.
//  - URL hooks reproduce every view headlessly, ?step=N included; the batch is
//    seeded, so a reload shows the same picture.
import { mount } from '../framework/layout.js';
import { T, alphaOf, inkOn, rgbaToken } from '../framework/theme.js';

const BPE = 2;                       // bytes per bf16 activation element
const clamp = (v, lo, hi) => (v < lo ? lo : v > hi ? hi : v);
const CATS = () => [T.accent, T.ok, T.violet, T.warn, T.tealDeep, T.goldDeep];

function fmtB(b) {
  if (!isFinite(b)) return '–';
  if (b >= 1e9) return (b / 1e9).toFixed(2) + ' GB';
  if (b >= 1e6) return (b / 1e6).toFixed(1) + ' MB';
  if (b >= 1e3) return (b / 1e3).toFixed(1) + ' kB';
  return b.toFixed(0) + ' B';
}

// Deterministic PRNG so the batch, and therefore every number on the page, is
// reproducible from ?seed alone.
function rng(seed) {
  let s = (seed >>> 0) || 1;
  return () => {
    s ^= s << 13; s >>>= 0; s ^= s >>> 17; s ^= s << 5; s >>>= 0;
    return s / 4294967296;
  };
}

// ---- the simulated batch --------------------------------------------------
// Experts sit in latent GROUPS (a token that wants one member usually wants
// another), and popularity is Zipf-skewed over a seeded permutation so the hot
// experts are not simply the low-numbered ones. Everything downstream -- the
// crossings, the per-device load, the step time -- is counted off this batch.
let batch = null, batchKey = '';
function buildBatch(st) {
  const E = st.experts | 0, N = st.tokens | 0, k = Math.min(st.topk | 0, E);
  const key = [E, N, k, st.skew, st.seed].join('|');
  if (batch && batchKey === key) return batch;
  const rnd = rng((st.seed | 0) * 2654435761 + 12345);

  const order = Array.from({ length: E }, (_, i) => i);
  for (let i = E - 1; i > 0; i--) { const j = (rnd() * (i + 1)) | 0; const t = order[i]; order[i] = order[j]; order[j] = t; }
  const w = new Float64Array(E); let sw = 0;
  for (let r = 0; r < E; r++) { const v = Math.pow(1 / (1 + r), st.skew); w[order[r]] = v; sw += v; }
  for (let e = 0; e < E; e++) w[e] /= sw;

  const G = clamp(Math.round(E / 4), 2, 6);
  const grp = new Int32Array(E);
  for (let e = 0; e < E; e++) grp[e] = Math.min(G - 1, Math.floor(e * G / E));
  const gw = new Float64Array(G);
  for (let e = 0; e < E; e++) gw[grp[e]] += w[e];

  const COH = 0.78;                  // how often a token stays inside one group
  const pickFrom = (pool) => {
    let tot = 0; for (const e of pool) tot += w[e];
    let x = rnd() * tot;
    for (const e of pool) { x -= w[e]; if (x <= 0) return e; }
    return pool[pool.length - 1];
  };
  const all = order.slice();
  const byGroup = Array.from({ length: G }, () => []);
  for (let e = 0; e < E; e++) byGroup[grp[e]].push(e);

  const sel = new Int32Array(N * k);
  for (let t = 0; t < N; t++) {
    let x = rnd(); let g = 0;
    for (; g < G - 1; g++) { x -= gw[g]; if (x <= 0) break; }
    const chosen = [];
    let guard = 0;
    while (chosen.length < k && guard++ < 200) {
      const pool = (rnd() < COH && byGroup[g].length) ? byGroup[g] : all;
      const e = pickFrom(pool);
      if (!chosen.includes(e)) chosen.push(e);
    }
    for (let i = 0; i < k; i++) sel[t * k + i] = chosen[i % chosen.length];
  }

  // per-expert counts + the co-occurrence matrix the rebalancer reads
  const cnt = new Int32Array(E);
  const co = Array.from({ length: E }, () => new Int32Array(E));
  for (let t = 0; t < N; t++) {
    for (let i = 0; i < k; i++) {
      const a = sel[t * k + i]; cnt[a]++;
      for (let j = i + 1; j < k; j++) { const b = sel[t * k + j]; co[a][b]++; co[b][a]++; }
    }
  }
  const hot = Array.from({ length: E }, (_, e) => e).sort((a, b) => cnt[b] - cnt[a]);
  batchKey = key;
  batch = { E, N, k, sel, cnt, co, w, grp, G, hot };
  return batch;
}

// ---- placement ------------------------------------------------------------
// A placement is one device digit per expert, so it round-trips through the URL
// as ?place=0123... and a hand-made arrangement is shareable.
const roundRobin = (E, D) => Array.from({ length: E }, (_, e) => e % D);
const encode = (pl) => pl.join('');
function decode(s, E, D) {
  const out = [];
  for (let i = 0; i < E; i++) {
    const c = s && s.charCodeAt(i) - 48;
    out.push(Number.isFinite(c) && c >= 0 && c < D ? c : i % D);
  }
  return out;
}

// Worst case: pile the hottest experts onto one device, and scatter whatever is
// left so that co-chosen experts land apart. Both failure modes at once.
function worstPlacement(b, D) {
  const E = b.E, cap = Math.ceil(E / D);
  const pl = new Array(E).fill(-1);
  const on = Array.from({ length: D }, () => []);
  for (let i = 0; i < Math.min(cap, E); i++) { const e = b.hot[i]; pl[e] = 0; on[0].push(e); }
  for (const e of b.hot) {
    if (pl[e] >= 0) continue;
    let best = -1, bestScore = Infinity;
    for (let d = 0; d < D; d++) {
      if (on[d].length >= cap) continue;
      let aff = 0; for (const f of on[d]) aff += b.co[e][f];
      const score = aff;                       // MINIMISE affinity: scatter
      if (score < bestScore) { bestScore = score; best = d; }
    }
    if (best < 0) best = on.findIndex((x) => x.length < cap);
    if (best < 0) best = 0;
    pl[e] = best; on[best].push(e);
  }
  return pl;
}

// Rebalance from observed load. Two ideas, in order: SEED each device with one
// of the hottest experts so the demand peaks start apart, then GROW each device
// by co-occurrence -- always letting the currently-lightest device pick next,
// and refusing a pick that would blow it past its fair share of the observed
// assignments. Replicated experts are excluded: they are everywhere already.
// This is the shipped shape -- placement recomputed from measured traffic, not
// from the model definition.
function rebalancePlacement(b, D, rep) {
  const E = b.E, cap = Math.ceil(E / D);
  const replicated = new Uint8Array(E);
  for (let i = 0; i < Math.min(rep || 0, E); i++) replicated[b.hot[i]] = 1;
  const pool = b.hot.filter((e) => !replicated[e]);
  const pl = Array.from({ length: E }, (_, e) => e % D);
  const unplaced = new Set(pool);
  const on = Array.from({ length: D }, () => []);
  const load = new Float64Array(D);
  const fair = Math.max(1, pool.reduce((a, e) => a + b.cnt[e], 0) / D);

  for (let d = 0; d < D && unplaced.size; d++) {
    const e = pool.find((x) => unplaced.has(x));
    unplaced.delete(e); pl[e] = d; on[d].push(e); load[d] += b.cnt[e];
  }
  while (unplaced.size) {
    let d = -1;
    for (let i = 0; i < D; i++) if (on[i].length < cap && (d < 0 || load[i] < load[d])) d = i;
    if (d < 0) d = 0;
    let best = -1, bestScore = -Infinity;
    for (const e of unplaced) {
      let aff = 0; for (const f of on[d]) aff += b.co[e][f];
      const score = aff / Math.max(1, b.N) - 0.35 * Math.max(0, (load[d] + b.cnt[e]) - fair) / fair;
      if (score > bestScore) { bestScore = score; best = e; }
    }
    if (best < 0) break;
    unplaced.delete(best); pl[best] = d; on[d].push(best); load[d] += b.cnt[best];
  }
  return pl;
}

// ---- the measurement ------------------------------------------------------
// Everything below is counted off the simulated batch: token copies crossing,
// bytes on each directed link, per-device assignments, and the step time set by
// whichever of (slowest device, busiest link) is larger.
function measure(b, pl, rep, st) {
  const D = st.devices | 0, E = b.E, k = b.k, N = b.N, dm = st.dmodel | 0;
  const replicated = new Uint8Array(E);
  for (let i = 0; i < Math.min(rep, E); i++) replicated[b.hot[i]] = 1;

  const assign = new Float64Array(D);           // (token, expert) pairs run here
  const link = Array.from({ length: D }, () => new Float64Array(D));   // token copies src->dst
  const linkBy = Array.from({ length: D }, () => Array.from({ length: D }, () => new Int32Array(E)));
  const localPairs = new Float64Array(D);
  let crossCopies = 0, localTokens = 0;

  const dests = new Set();
  for (let t = 0; t < N; t++) {
    const home = t % D;
    dests.clear();
    for (let i = 0; i < k; i++) {
      const e = b.sel[t * k + i];
      const d = replicated[e] ? home : pl[e];
      assign[d]++;
      if (d === home) localPairs[home]++;
      else { dests.add(d); linkBy[home][d][e]++; }
    }
    if (dests.size === 0) localTokens++;
    for (const d of dests) { link[home][d]++; crossCopies++; }
  }

  const copyBytes = dm * BPE;
  // A directed link carries dispatch out and the matching combine coming back.
  let busyBytes = 0, busySrc = 0, busyDst = 0, totalBytes = 0;
  for (let a = 0; a < D; a++) {
    for (let c = 0; c < D; c++) {
      if (a === c) continue;
      const bytes = (link[a][c] + link[c][a]) * copyBytes;
      totalBytes += link[a][c] * copyBytes * 2;
      if (bytes > busyBytes) { busyBytes = bytes; busySrc = a; busyDst = c; }
    }
  }

  // One token step passes through L MoE layers, each with the same placement,
  // so the per-layer counts above are paid L times over.
  const L = Math.max(1, st.layers | 0);
  const dff = Math.round(dm / 2);
  const flopsPerPair = 6 * dm * dff;            // gate + up + down, 2 FLOPs per MAC
  const compMs = Array.from(assign, (a) => (a * L * flopsPerPair) / (st.tflops * 1e12) * 1000);
  const commMs = st.bw > 0 ? (busyBytes * L / (st.bw * 1e9)) * 1000 : 0;
  const slowest = compMs.length ? Math.max(...compMs) : 0;
  const slowDev = compMs.indexOf(slowest);
  const stepMs = Math.max(slowest, commMs);
  const idleMs = compMs.map((c) => stepMs - c);
  const meanAssign = (N * k) / D;
  const imbalance = meanAssign > 0 ? Math.max(...assign) / meanAssign : 1;
  const util = compMs.map((c) => (stepMs > 0 ? c / stepMs : 0));

  const expertBytes = 3 * dm * dff * BPE;
  const resident = Array.from({ length: D }, (_, d) => {
    let n = 0; for (let e = 0; e < E; e++) if (replicated[e] || pl[e] === d) n++;
    return n;
  });
  const replicas = Array.from(replicated).reduce((a, v) => a + v, 0);

  return {
    D, E, k, N, pl, replicated, replicas, assign, link, linkBy, localPairs, localTokens,
    crossCopies, copyBytes, busyBytes, busySrc, busyDst, totalBytes,
    compMs, commMs, stepMs, slowest, slowDev, idleMs, util, imbalance, resident, L,
    expertBytes, memPerDev: resident.map((n) => n * expertBytes * L),
    stepBytes: totalBytes * L, busyStepBytes: busyBytes * L,
    crossFrac: N > 0 ? crossCopies / N : 0,
  };
}

const STAGES = [
  { key: 'route', label: 'routing — each token picks its top-k experts (this page does NOT change that)' },
  { key: 'dispatch', label: 'ALL-TO-ALL dispatch — each token is copied once to every DISTINCT device holding an expert it chose' },
  { key: 'compute', label: 'expert compute — each device runs the experts resident on it, on whatever arrived' },
  { key: 'combine', label: 'ALL-TO-ALL combine — every expert output travels back to the token’s home device' },
];

// ---- drawing helpers ------------------------------------------------------
function roundRect(ctx, x, y, w, h, r) {
  const rr = Math.max(0, Math.min(r, w / 2, h / 2));
  ctx.beginPath(); ctx.moveTo(x + rr, y);
  ctx.arcTo(x + w, y, x + w, y + h, rr); ctx.arcTo(x + w, y + h, x, y + h, rr);
  ctx.arcTo(x, y + h, x, y, rr); ctx.arcTo(x, y, x + w, y, rr); ctx.closePath();
}
function hatch(ctx, x, y, w, h, color) {
  if (w <= 0 || h <= 0) return;
  ctx.save();
  ctx.beginPath(); ctx.rect(x, y, w, h); ctx.clip();
  ctx.strokeStyle = color; ctx.lineWidth = 1;
  for (let i = -h; i < w; i += 5) { ctx.beginPath(); ctx.moveTo(x + i, y + h); ctx.lineTo(x + i + h, y); ctx.stroke(); }
  ctx.restore();
}
function curve(ctx, x1, y1, x2, y2, lift) {
  ctx.beginPath(); ctx.moveTo(x1, y1);
  ctx.quadraticCurveTo((x1 + x2) / 2, Math.min(y1, y2) - lift, x2, y2);
  ctx.stroke();
}
function arrowHead(ctx, x, y, ang, size) {
  ctx.beginPath(); ctx.moveTo(x, y);
  ctx.lineTo(x - size * Math.cos(ang - 0.42), y - size * Math.sin(ang - 0.42));
  ctx.lineTo(x - size * Math.cos(ang + 0.42), y - size * Math.sin(ang + 0.42));
  ctx.closePath(); ctx.fill();
}

// ---- live geometry, captured each draw for hit-testing --------------------
let panels = null;      // [{d, x, y, w, h, cx}]
let chips = null;       // [{e, d, x, y, w, h, replica}]
let wires = null;       // [{a, b, mid, why}]
let ribbon = null;      // {x, y, w, h, bars:[{e,x,w}]}
let grab = null;        // {e, from} while an expert chip is being dragged
let ribDrag = null;     // {y0, s0, x0, r0} while the ribbon is being dragged

mount({
  mount: 'body',
  title: 'expert-placement — which GPU an expert lives on, and what that costs',
  blurb: 'Routing decides which experts a token wants. PLACEMENT decides where those experts physically live — and that is what sets how much of the batch crosses the interconnect. Under expert parallelism each device holds a subset of the experts, so a token whose expert lives elsewhere has to be shipped there and its result shipped back. The dispatch copies a token once per DISTINCT destination device, so experts that are frequently chosen TOGETHER are cheap to co-locate and expensive to scatter; and because expert popularity is heavily skewed, a hot expert alone on one device makes that device the bottleneck while the others idle. Pick an expert chip up and drop it on another device — traffic, per-device load and step time recompute under your hand. Drag the popularity ribbon ↕ to skew demand and ↔ to replicate the hottest experts onto every device. Then try the buttons: round-robin is the named baseline, rebalance places by observed co-occurrence and load, and worst case does both things wrong at once. Every number is counted off the seeded batch; the interconnect and compute rates are yours to set.',
  prefer: 'canvas2d',
  aspect: '16 / 10',
  autoplay: true,
  animate: true,
  challenges: [
    {
      goal: 'Beat round-robin placement: get the step time under 80% of the baseline.',
      hint: 'co-locate experts that are chosen together, and replicate the hottest ones so they never travel — press “rebalance from observed load”, then push the replication slider up.',
      check: (api) => ({ solved: (api.probe.ratio || 9) < 0.80, detail: `step time is ${(100 * (api.probe.ratio || 1)).toFixed(1)}% of round-robin (lower is better) — needs < 80.0%` }),
    },
    {
      goal: 'Make one device the bottleneck: push its share of the work past 1.8× the fair share.',
      hint: 'drag the hottest experts onto the same device, and turn the popularity skew up.',
      check: (api) => ({ solved: (api.probe.imbalance || 1) > 1.8, detail: `busiest device holds ${(api.probe.imbalance || 1).toFixed(2)}× the fair share — needs > 1.80×` }),
    },
    {
      goal: 'Get more than half the batch to stay home: fewer than 0.5 device-crossings per token.',
      hint: 'replication makes a hot expert local everywhere — at the price of a copy of its weights on every device.',
      check: (api) => ({ solved: (api.probe.crossFrac || 9) < 0.5, detail: `${(api.probe.crossFrac || 0).toFixed(2)} crossings per token — needs < 0.50` }),
    },
  ],
  controls: (c, page) => {
    c.stepper('devices', { label: 'devices (expert-parallel group)', min: 2, max: 6, value: 4 });
    c.stepper('experts', { label: 'experts (E)', min: 4, max: 24, value: 16 });
    c.stepper('topk', { label: 'experts per token (k)', min: 1, max: 4, value: 2 });
    c.slider('tokens', { label: 'tokens in the batch', min: 32, max: 512, step: 32, value: 256, rebuild: true });
    c.slider('skew', { label: 'popularity skew', min: 0, max: 3, step: 0.05, value: 1.2, rebuild: true });
    c.slider('rep', { label: 'replicate hottest N experts', min: 0, max: 4, step: 1, value: 0, rebuild: true });
    c.slider('layers', { label: 'MoE layers per token step', min: 1, max: 64, step: 1, value: 32, rebuild: true });
    c.slider('dmodel', { label: 'hidden size d', min: 1024, max: 8192, step: 512, value: 4096, rebuild: true });
    c.slider('bw', { label: 'per-device link (GB/s)', min: 10, max: 600, step: 10, value: 100, rebuild: true });
    c.slider('tflops', { label: 'per-device compute (TFLOP/s)', min: 20, max: 1000, step: 20, value: 200, rebuild: true });
    c.slider('seed', { label: 'batch seed', min: 0, max: 99, step: 1, value: 7, rebuild: true });
    c.button('round-robin placement (baseline)', () => {
      const b = buildBatch(page.state);
      page.controls.set('place', encode(roundRobin(b.E, page.state.devices | 0)));
    });
    c.button('rebalance from observed load', () => {
      const b = buildBatch(page.state);
      page.controls.set('place', encode(rebalancePlacement(b, page.state.devices | 0, page.state.rep | 0)));
    });
    c.button('worst case', () => {
      const b = buildBatch(page.state);
      page.controls.set('place', encode(worstPlacement(b, page.state.devices | 0)));
    });
    c.transport({ compute: () => STAGES.slice(), speed: 1.1, loop: true });
  },

  // Direct manipulation. Chips are picked up and dropped on a device; the
  // popularity ribbon is a two-axis drag (skew vertically, replication count
  // horizontally).
  onPointer: (page, ev) => {
    const D = page.state.devices | 0;
    if (ev.type === 'down') {
      if (chips) {
        for (const ch of chips) {
          if (ch.replica) continue;
          if (ev.x >= ch.x && ev.x <= ch.x + ch.w && ev.y >= ch.y && ev.y <= ch.y + ch.h) { grab = { e: ch.e, from: ch.d }; return; }
        }
      }
      if (ribbon && ev.x >= ribbon.x && ev.x <= ribbon.x + ribbon.w && ev.y >= ribbon.y - 6 && ev.y <= ribbon.y + ribbon.h + 6) {
        ribDrag = { y0: ev.y, s0: +page.state.skew, x0: ev.x, r0: +page.state.rep };
      }
    } else if (ev.type === 'up' || ev.type === 'leave') {
      if (grab && panels) {
        for (const p of panels) {
          if (ev.x >= p.x && ev.x <= p.x + p.w && ev.y >= p.y && ev.y <= p.y + p.h && p.d !== grab.from) {
            const b = buildBatch(page.state);
            const pl = decode(page.state.place, b.E, D);
            pl[grab.e] = p.d;
            page.controls.set('place', encode(pl));
            break;
          }
        }
      }
      grab = null; ribDrag = null;
    } else if (ev.type === 'move') {
      if (ribDrag && page.pointer.down) {
        const s = clamp(+(ribDrag.s0 - (ev.y - ribDrag.y0) * 0.012).toFixed(2), 0, 3);
        if (s !== +page.state.skew) page.controls.set('skew', s, { rebuild: true });
        const r = clamp(ribDrag.r0 + Math.round((ev.x - ribDrag.x0) / 46), 0, 4);
        if (r !== +page.state.rep) page.controls.set('rep', r);
      }
      page.redraw();
    }
  },

  draw: (page) => {
    const r = page.renderer, ctx = page.ctx, st = page.state;
    const D = clamp(st.devices | 0, 1, 6);
    const b = buildBatch(st);
    const E = b.E;

    // Normalise the placement against the live E / D. Silent, so a resize of
    // the problem does not fight the redraw loop.
    let pl = decode(st.place, E, D);
    const enc = encode(pl);
    if (enc !== st.place) page.controls.set('place', enc, { silent: true });

    const rep = clamp(st.rep | 0, 0, 4);
    const m = measure(b, pl, rep, st);
    const base = measure(b, roundRobin(E, D), 0, st);
    const ratio = base.stepMs > 0 ? m.stepMs / base.stepMs : 1;

    page.probe = { ratio, imbalance: m.imbalance, crossFrac: m.crossFrac, stepMs: m.stepMs, baseMs: base.stepMs };

    r.clear(T.n0);
    const W = page.W, H = page.H, pad = 16;
    const tr = page.controls._transport;
    const sIdx = tr && tr.index >= 0 ? Math.min(tr.index, STAGES.length - 1) : STAGES.length - 1;
    const stage = STAGES[sIdx];
    const stageCol = stage.key === 'compute' ? T.ok : stage.key === 'route' ? T.n10 : T.violet;

    // ---- header ----------------------------------------------------------
    r.label(`${E} experts over ${D} devices · top-${b.k} of ${E} per token · ${b.N} tokens in the batch · ${m.L} MoE layers · d=${st.dmodel}`,
      pad, 18, { color: T.n14, font: '13px ui-monospace, monospace' });
    r.label(`step ${sIdx + 1}/${STAGES.length} — ${stage.label}`, pad, 36, { color: stageCol, font: '11.5px ui-monospace, monospace' });

    // ---- geometry --------------------------------------------------------
    const ribY = 50, ribH = 34;
    const panelTop = ribY + ribH + 26;
    const barBot = H - 52;
    const barH = Math.max(44, H * 0.15);
    const barTop = barBot - barH;
    const wireBot = barTop - 22;
    const slot = (W - 2 * pad) / D;
    const panelW = Math.max(60, slot - 12);
    // Panels are only as tall as the chips they hold; whatever is left goes to
    // the wire band, which is the part that actually has to be legible.
    const ncol = Math.max(2, Math.floor((panelW - 8) / 46));
    let maxList = 0;
    for (let d = 0; d < D; d++) {
      let n = m.replicas;
      for (let e = 0; e < E; e++) if (!m.replicated[e] && pl[e] === d) n++;
      maxList = Math.max(maxList, n);
    }
    const rows = Math.ceil(maxList / ncol);
    const panelH = clamp(44 + rows * 19 + 8, 84, Math.max(84, (wireBot - panelTop) * 0.58));
    const panelBot = panelTop + panelH;
    const wireTop = panelBot + 22;

    // ---- popularity ribbon (drag: ↕ skew, ↔ replication) -----------------
    {
      const x0 = pad + 138, w = W - pad - x0;
      ribbon = { x: x0, y: ribY, w, h: ribH, bars: [] };
      r.label('expert popularity', pad, ribY + 11, { color: T.n12, font: '11px ui-monospace, monospace' });
      r.label('↕ skew  ↔ replicate', pad, ribY + 25, { color: T.accent, font: '9.5px ui-monospace, monospace' });
      let maxw = 0; for (let e = 0; e < E; e++) maxw = Math.max(maxw, b.w[e]);
      const bw = w / E;
      for (let i = 0; i < E; i++) {
        const e = b.hot[i];                       // hottest first, left to right
        const x = x0 + i * bw;
        const h = Math.max(1.5, (b.w[e] / (maxw || 1)) * ribH);
        const isRep = m.replicated[e] === 1;
        ctx.save();
        ctx.fillStyle = alphaOf(isRep ? T.ok : CATS()[pl[e] % CATS().length], isRep ? 0.95 : 0.7);
        ctx.fillRect(x + 1, ribY + ribH - h, Math.max(1, bw - 2), h);
        ctx.restore();
        ribbon.bars.push({ e, x, w: bw });
      }
      if (rep > 0) {
        ctx.save();
        ctx.strokeStyle = T.ok; ctx.lineWidth = 1.4; ctx.setLineDash([4, 3]);
        ctx.strokeRect(x0, ribY - 3, bw * rep, ribH + 6); ctx.setLineDash([]);
        ctx.restore();
        r.label(`${rep} replicated on every device`, x0 + bw * rep + 6, ribY + 11, { color: T.ok, font: '9.5px ui-monospace, monospace' });
      }
    }

    // ---- device panels ---------------------------------------------------
    r.label('who holds what', pad, panelTop - 7, { color: T.n12, font: '11px ui-monospace, monospace' });
    r.label('drag an expert chip onto another device', W - pad, panelTop - 7,
      { color: T.accent, font: '10px ui-monospace, monospace', align: 'right' });
    panels = []; chips = [];
    for (let d = 0; d < D; d++) {
      const x = pad + d * slot + (slot - panelW) / 2;
      const cx = x + panelW / 2;
      const hot = m.slowDev === d && m.imbalance > 1.05;
      ctx.save();
      roundRect(ctx, x, panelTop, panelW, panelH, 8);
      ctx.fillStyle = rgbaToken('n14', 0.035); ctx.fill();
      ctx.strokeStyle = hot ? T.bad : rgbaToken('n14', 0.16);
      ctx.lineWidth = hot ? 1.8 : 1; ctx.stroke();
      ctx.restore();
      r.label(`device ${d}`, cx, panelTop + 14, { color: T.n13, font: '11px ui-monospace, monospace', align: 'center' });
      r.label(`${m.assign[d].toFixed(0)} pairs · ${m.compMs[d].toFixed(2)} ms`, cx, panelTop + 27,
        { color: hot ? T.bad : T.n11, font: '9.5px ui-monospace, monospace', align: 'center' });

      // resident experts, as chips. A replica is dashed: it is a COPY, and it
      // is why replication costs memory rather than being free.
      const list = [];
      for (let e = 0; e < E; e++) if (m.replicated[e]) list.push({ e, replica: true });
      for (let e = 0; e < E; e++) if (!m.replicated[e] && pl[e] === d) list.push({ e, replica: false });
      const cw = (panelW - 10 - (ncol - 1) * 4) / ncol, chH = 15;
      for (let i = 0; i < list.length; i++) {
        const cxx = x + 5 + (i % ncol) * (cw + 4);
        const cyy = panelTop + 36 + Math.floor(i / ncol) * (chH + 4);
        if (cyy + chH > panelTop + panelH - 3) break;
        const { e, replica } = list[i];
        const col = replica ? T.ok : CATS()[d % CATS().length];
        const dragging = grab && grab.e === e && !replica;
        ctx.save();
        roundRect(ctx, cxx, cyy, cw, chH, 4);
        ctx.fillStyle = alphaOf(col, dragging ? 0.10 : 0.26); ctx.fill();
        ctx.strokeStyle = alphaOf(col, dragging ? 0.35 : 0.85); ctx.lineWidth = 1;
        if (replica) ctx.setLineDash([3, 2]);
        ctx.stroke(); ctx.setLineDash([]);
        ctx.fillStyle = col; ctx.font = '9.5px ui-monospace, monospace';
        ctx.textAlign = 'center'; ctx.textBaseline = 'middle';
        ctx.fillText(`e${e}${replica ? '*' : ''}`, cxx + cw / 2, cyy + chH / 2 + 0.5);
        ctx.restore();
        chips.push({ e, d, x: cxx, y: cyy, w: cw, h: chH, replica });
      }
      panels.push({ d, x, y: panelTop, w: panelW, h: panelH, cx });
    }

    // the chip being dragged rides the cursor, so the drop target is obvious
    if (grab) {
      const p = page.pointer;
      ctx.save();
      roundRect(ctx, p.x - 20, p.y - 8, 40, 16, 4);
      ctx.fillStyle = alphaOf(T.accent, 0.85); ctx.fill();
      ctx.fillStyle = inkOn(T.accent); ctx.font = '9.5px ui-monospace, monospace';
      ctx.textAlign = 'center'; ctx.textBaseline = 'middle';
      ctx.fillText(`e${grab.e}`, p.x, p.y + 0.5);
      ctx.restore();
    }

    // ---- the all-to-all wires --------------------------------------------
    wires = [];
    const yMid = (wireTop + wireBot) / 2;
    const showTraffic = stage.key === 'dispatch' || stage.key === 'combine';
    let maxLink = 0;
    for (let a = 0; a < D; a++) for (let c = 0; c < D; c++) if (a !== c) maxLink = Math.max(maxLink, m.link[a][c]);
    if (D > 1 && maxLink > 0) {
      const combine = stage.key === 'combine';
      for (let a = 0; a < D; a++) {
        for (let c = 0; c < D; c++) {
          if (a === c || m.link[a][c] <= 0) continue;
          const v = m.link[a][c], tt = v / maxLink;
          const src = panels[combine ? c : a], dst = panels[combine ? a : c];
          const lift = 12 + Math.abs(c - a) * Math.max(10, (wireBot - wireTop) * 0.30);
          ctx.save();
          const alpha = (showTraffic ? 0.25 : 0.10) + (showTraffic ? 0.7 : 0.22) * tt;
          ctx.strokeStyle = alphaOf(T.violet, alpha);
          ctx.fillStyle = alphaOf(T.violet, alpha);
          ctx.lineWidth = 0.8 + 3.6 * tt;
          curve(ctx, src.cx, yMid, dst.cx, yMid, lift);
          arrowHead(ctx, dst.cx, yMid, dst.cx > src.cx ? 0.9 : Math.PI - 0.9, 7);
          ctx.restore();
          if (showTraffic) {
            const off = (a * D + c) / (D * D);
            const ph = ((page.t || 0) * 0.5 + off) % 1;
            const mx = (src.cx + dst.cx) / 2, my = yMid - lift, u = 1 - ph;
            const bx = u * u * src.cx + 2 * u * ph * mx + ph * ph * dst.cx;
            const by = u * u * yMid + 2 * u * ph * my + ph * ph * yMid;
            ctx.save(); ctx.fillStyle = T.violet; ctx.beginPath(); ctx.arc(bx, by, 2.6, 0, Math.PI * 2); ctx.fill(); ctx.restore();
          }
          const pulls = Array.from({ length: E }, (_, e) => e)
            .filter((e) => m.linkBy[a][c][e] > 0)
            .sort((x, y) => m.linkBy[a][c][y] - m.linkBy[a][c][x]).slice(0, 3)
            .map((e) => `e${e} (${m.linkBy[a][c][e]})`).join(', ');
          wires.push({
            a, c, mid: { x: (src.cx + dst.cx) / 2, y: yMid - lift * 0.55 },
            why: `device ${a} → device ${c}\n` +
              `${v} of the ${m.N} tokens are copied across this link: they live on device ${a}\n` +
              `and chose at least one expert that is placed on device ${c}.\n` +
              `dispatch ${v} × ${st.dmodel} × ${BPE} B = ${fmtB(v * m.copyBytes)} per layer, and the combine\n` +
              `brings the same volume back, so this link carries ${fmtB((m.link[a][c] + m.link[c][a]) * m.copyBytes * m.L)} per step across ${m.L} layers.\n` +
              `the experts pulling them there: ${pulls || '—'}\n` +
              `move one of those onto device ${a} and this link's volume drops.`,
          });
        }
      }
    }
    r.label(D > 1
      ? `${m.crossCopies} token copies cross per layer (${m.crossFrac.toFixed(2)} per token) · busiest link device ${m.busySrc}↔${m.busyDst} carries ${fmtB(m.busyStepBytes)} per step = ${m.commMs.toFixed(2)} ms at ${st.bw} GB/s`
      : 'one device — every expert is local and nothing crosses',
      pad, wireTop - 4, { color: T.violet, font: '10.5px ui-monospace, monospace' });

    // ---- per-device utilisation bars -------------------------------------
    {
      const scale = m.stepMs > 0 ? barH / m.stepMs : 0;
      ctx.save();
      ctx.strokeStyle = T.n5; ctx.lineWidth = 1;
      ctx.beginPath(); ctx.moveTo(pad, barBot); ctx.lineTo(W - pad, barBot); ctx.stroke();
      ctx.restore();
      const bwid = Math.min(72, slot * 0.5);
      for (let d = 0; d < D; d++) {
        const cxx = pad + d * slot + slot / 2 - bwid / 2;
        const ch = m.compMs[d] * scale;
        ctx.save();
        ctx.fillStyle = alphaOf(CATS()[d % CATS().length], 0.75);
        ctx.fillRect(cxx, barBot - ch, bwid, ch);
        ctx.restore();
        hatch(ctx, cxx, barTop, bwid, Math.max(0, barBot - ch - barTop), alphaOf(T.n9, 0.55));
        // A full-height bar has no room above it, so the percentage moves
        // inside rather than colliding with the axis caption.
        const inside = barBot - ch - barTop < 14;
        r.label(`${(100 * m.util[d]).toFixed(0)}%`, cxx + bwid / 2, inside ? barBot - ch + 12 : barBot - ch - 4,
          { color: inside ? inkOn(CATS()[d % CATS().length]) : T.n12, font: '9.5px ui-monospace, monospace', align: 'center' });
        r.label(`device ${d} · idle ${m.idleMs[d].toFixed(2)} ms`, cxx + bwid / 2, barBot + 12,
          { color: T.n10, font: '9px ui-monospace, monospace', align: 'center' });
      }
      ctx.save();
      ctx.strokeStyle = T.bad; ctx.setLineDash([5, 4]); ctx.lineWidth = 1.3;
      ctx.beginPath(); ctx.moveTo(pad, barTop); ctx.lineTo(W - pad, barTop); ctx.stroke();
      ctx.setLineDash([]); ctx.restore();
      r.label(`step ${m.stepMs.toFixed(2)} ms — set by ${m.commMs >= m.slowest ? `the busiest link` : `device ${m.slowDev}`}`,
        W - pad, barTop - 4, { color: T.bad, font: '10px ui-monospace, monospace', align: 'right' });
      r.label('per-device compute · hatched = idle, waiting for the step', pad, barTop - 4,
        { color: T.n11, font: '10px ui-monospace, monospace' });
    }

    // ---- hover ------------------------------------------------------------
    if (page.pointer.over && !grab && !ribDrag) {
      const p = page.pointer;
      let tip = null;
      let best = 1e9;
      for (const wr of wires) {
        const dx = p.x - wr.mid.x, dy = p.y - wr.mid.y, dd = dx * dx + dy * dy;
        if (dd < best && dd < 26 * 26) { best = dd; tip = wr.why; }
      }
      if (!tip && ribbon && p.x >= ribbon.x && p.x <= ribbon.x + ribbon.w && p.y >= ribbon.y - 6 && p.y <= ribbon.y + ribbon.h + 6) {
        const bar = ribbon.bars.find((bb) => p.x >= bb.x && p.x < bb.x + bb.w);
        if (bar) {
          tip = `expert ${bar.e} — ${(100 * b.w[bar.e]).toFixed(1)}% of routed demand\n` +
            `${b.cnt[bar.e]} of the ${b.N * b.k} assignments in this batch\n` +
            `${m.replicated[bar.e] ? 'REPLICATED on every device — always local, and its weights are resident ' + D + ' times' : `placed on device ${pl[bar.e]}`}\n` +
            `drag this ribbon ↕ to change the skew, ↔ to change how many hot experts are replicated`;
        }
      }
      if (!tip) {
        for (const ch of chips) {
          if (p.x >= ch.x && p.x <= ch.x + ch.w && p.y >= ch.y && p.y <= ch.y + ch.h) {
            tip = ch.replica
              ? `expert ${ch.e} — a REPLICA.\nIt is resident on all ${D} devices, so every token that wants it runs at home\nand nothing crosses for it. The bill is ${fmtB(m.expertBytes * m.L)} of weights per extra copy\n(${fmtB(m.expertBytes)} per layer × ${m.L} layers).`
              : `expert ${ch.e} on device ${ch.d} — ${b.cnt[ch.e]} assignments this batch\n${(100 * b.w[ch.e]).toFixed(1)}% of routed demand · weights ${fmtB(m.expertBytes * m.L)} across ${m.L} layers\ndrag it onto another device and watch the wires and the step time move`;
            break;
          }
        }
      }
      if (!tip) {
        for (const pn of panels) {
          if (p.x >= pn.x && p.x <= pn.x + pn.w && p.y >= pn.y && p.y <= pn.y + pn.h) {
            const res = [];
            for (let e = 0; e < E; e++) if (m.replicated[e] || pl[e] === pn.d) res.push('e' + e + (m.replicated[e] ? '*' : ''));
            tip = `device ${pn.d} holds ${res.length} experts: ${res.join(' ')}\n` +
              `${m.assign[pn.d].toFixed(0)} of the ${m.N * m.k} (token, expert) pairs run here — ` +
              `${(m.assign[pn.d] / Math.max(1, (m.N * m.k) / D)).toFixed(2)}× the fair share\n` +
              `compute ${m.compMs[pn.d].toFixed(2)} ms of a ${m.stepMs.toFixed(2)} ms step → ${(100 * m.util[pn.d]).toFixed(0)}% busy, ` +
              `idle ${m.idleMs[pn.d].toFixed(2)} ms\n` +
              `${m.localPairs[pn.d].toFixed(0)} of its pairs came from tokens already living here\n` +
              `expert weights resident ${fmtB(m.memPerDev[pn.d])}${m.replicas ? ` (includes ${m.replicas} replica${m.replicas > 1 ? 's' : ''})` : ''}` +
              (pn.d === m.slowDev && m.slowest >= m.commMs ? '\nthis device sets the step time — every other device waits for it' : '');
            break;
          }
        }
      }
      if (tip) page.setTip(tip);
    }

    // ---- readout ----------------------------------------------------------
    const dir = ratio < 1 ? 'FASTER than' : ratio > 1 ? 'SLOWER than' : 'level with';
    let o = `PLACEMENT ${enc}  ·  ${E} experts / ${D} devices / top-${b.k} / ${m.N} tokens / ${m.L} MoE layers  ·  ${m.replicas} replicated     tier:${r.name}\n`;
    o += `STEP  ${m.stepMs.toFixed(2)} ms = ${(100 * ratio).toFixed(1)}% of round-robin placement's ${base.stepMs.toFixed(2)} ms `;
    o += `(lower is better; 100% = parity — this placement is ${dir} the baseline). Set by ${m.commMs >= m.slowest ? `the busiest link (${m.commMs.toFixed(2)} ms vs ${m.slowest.toFixed(2)} ms of compute)` : `device ${m.slowDev} (${m.slowest.toFixed(2)} ms of compute vs ${m.commMs.toFixed(2)} ms on the wire)`}.\n`;
    o += `WIRE  ${m.crossCopies} token copies cross per layer (${m.crossFrac.toFixed(2)} per token; round-robin: ${base.crossFrac.toFixed(2)}) · ${fmtB(m.stepBytes)} moved per step across all ${m.L} layers · busiest link ${fmtB(m.busyStepBytes)} = ${m.commMs.toFixed(2)} ms at ${st.bw} GB/s.\n`;
    o += `LOAD  busiest device ${m.imbalance.toFixed(2)}× the fair share (round-robin: ${base.imbalance.toFixed(2)}×) · utilisation ${m.util.map((u) => (100 * u).toFixed(0) + '%').join(' / ')} · idle ${m.idleMs.map((i) => i.toFixed(2)).join(' / ')} ms.\n`;
    o += `MEM   ${fmtB(m.expertBytes)} of weights per expert per layer; resident per device across ${m.L} layers ${m.memPerDev.map((x) => fmtB(x)).join(' / ')}${m.replicas ? ` — replication is bought with memory, not for free` : ''}.\n`;
    o += `Routing is unchanged by anything on this page: the same tokens want the same experts, and only WHERE those experts live is moving.`;
    page.setReadout(o);
  },
}).then((page) => {
  window.__epPage = page;
  const q = new URLSearchParams(location.search);
  const t = page.controls._transport;
  const num = (k, key, lo, hi) => { if (q.has(k)) page.controls.set(key, clamp(parseFloat(q.get(k)) || lo, lo, hi), { rebuild: true, silent: true }); };
  num('devices', 'devices', 2, 6);
  num('experts', 'experts', 4, 24);
  num('topk', 'topk', 1, 4);
  num('tokens', 'tokens', 32, 512);
  num('skew', 'skew', 0, 3);
  num('rep', 'rep', 0, 4);
  num('layers', 'layers', 1, 64);
  num('dmodel', 'dmodel', 1024, 8192);
  num('bw', 'bw', 10, 600);
  num('tflops', 'tflops', 20, 1000);
  num('seed', 'seed', 0, 99);
  // ?place=<one device digit per expert> reproduces a hand-made arrangement;
  // ?preset=rebalance|worst|rr computes one. Placement first, preset wins.
  const bt = buildBatch(page.state);
  const D0 = page.state.devices | 0;
  let place = q.has('place') ? decode(q.get('place'), bt.E, D0) : roundRobin(bt.E, D0);
  const preset = q.get('preset');
  if (preset === 'rebalance') place = rebalancePlacement(bt, D0, page.state.rep | 0);
  else if (preset === 'worst') place = worstPlacement(bt, D0);
  else if (preset === 'rr') place = roundRobin(bt.E, D0);
  page.controls.set('place', encode(place), { silent: true });
  if (t) t.rebuild();
  // ?hover=x,y fakes the cursor (a screenshot run has no pointer). Canvas px.
  if (q.has('hover')) {
    const [hx, hy] = q.get('hover').split(',').map(Number);
    page.pointer.x = hx; page.pointer.y = hy; page.pointer.over = true;
  }
  if (q.has('step') || q.has('hover')) { if (t) t.pause(); }
  if (q.has('step') && t) t.seek(parseInt(q.get('step'), 10));
  if (q.get('play') === '1' && t) t.play();
  page.redraw();
});
