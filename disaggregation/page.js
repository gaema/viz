// disaggregation concept page -- prefill and decode on SEPARATE machine pools,
// with the KV cache shipped between them.
//
// The two phases of autoregressive inference want opposite machines. PREFILL is
// one big parallel GEMM over the whole prompt: it saturates the math units and
// scales with FLOPs (compute-bound). DECODE is one skinny step per token that
// re-reads the whole KV cache and the weights: the math units idle and the run
// is set by memory bandwidth (memory-bound). That is the lesson of the
// prefill-vs-decode page; this page is its DEPLOYMENT consequence.
//
// Share one pool and the two phases fight: a prefill occupies the machine for
// its whole duration, so every sequence already decoding on that machine emits
// nothing until it ends (head-of-line blocking -- the hatched gaps on the lower
// timeline). Split the fleet instead: a prefill pool sized and tuned for
// compute, a decode pool sized and tuned for bandwidth, and a request that has
// been prefilled has its KV cache transferred to a decode machine to finish.
//
// THE NEW COST IS REAL AND THIS PAGE CHARGES IT. The KV cache is bytes on a
// wire: K and V for every layer for every prompt token, so the transfer is
// proportional to prompt length, and it sits between the first token (emitted
// by the prefill machine) and all the rest. Drag the interconnect thin enough
// and disaggregation LOSES -- the page says so in the readout when it does.
//
// Everything on screen is simulated in-page, tick by tick, from the request
// list; every number in a tooltip or the readout is derived from that run.
//
// Interactive per the shared render framework's contract:
//  - TRANSPORT scrubs time (one tick = one decode step); BOTH deployments
//    advance together against one shared x axis. Auto-plays and loops.
//  - DIRECT MANIPULATION: drag the allocation bar to move machines between the
//    prefill pool and the decode pool, and drag the interconnect pipe up/down
//    to widen or throttle the link. Throughput and latency re-balance under
//    your hand; there is a sweet spot, a too-few-prefill failure and a
//    too-slow-link failure.
//  - HOVER a request (on either timeline or in the breakdown strip) for its
//    phase breakdown including the KV bytes and the transfer time; hover a
//    worker for what it is doing, and why it is idle when it is.
//  - STEPPERS resize the problem: machines, requests, arrival gap, prompt
//    length.
//  - URL hooks reproduce every view headlessly: ?step, ?M, ?split, ?bw, ?nreq,
//    ?gap, ?plen, ?seed, ?hover=x,y.
import { mount } from '../framework/layout.js';
import { categorical } from '../framework/render.js';
import { T, alphaOf, inkOn, rgbaToken } from '../framework/theme.js';

// ---- the cost model (one place, so every number on screen is traceable) ----
// A tick is one decode step of the model being served. The constants are
// round, order-of-magnitude figures for a large dense model on a modern
// accelerator -- they set the SHAPE of the tradeoff, which is the lesson; the
// page is not a calculator for any particular machine.
const TICK_MS = 10;            // one decode step = one tick on the timelines
const PRE_MS_PER_TOK = 0.25;   // prefill compute, per prompt token, per machine
const KV_MB_PER_TOK = 0.5;     // K+V for EVERY layer, per token -- what must move
const BATCH = 4;               // sequences a decode machine batches per step

const clamp = (v, lo, hi) => (v < lo ? lo : v > hi ? hi : v);
const preTicks = (P) => Math.max(1, Math.ceil((P * PRE_MS_PER_TOK) / TICK_MS));
const kvMB = (P) => P * KV_MB_PER_TOK;
const xferMs = (P, bw) => (kvMB(P) / 1024 / bw) * 1000;      // MB -> GiB -> s -> ms
const xferTicks = (P, bw) => Math.max(1, Math.ceil(xferMs(P, bw) / TICK_MS));

function hash32(a, b) {
  let x = (Math.imul(a + 1, 374761393) + Math.imul(b + 1, 668265263)) >>> 0;
  x = (x ^ (x >>> 13)) >>> 0;
  x = Math.imul(x, 1274126177) >>> 0;
  return (x ^ (x >>> 16)) >>> 0;
}

// The workload: one arrival every `gap` ticks, prompt lengths spread around the
// requested mean (a real serving mix is never uniform), output lengths spread
// too. Seeded, so every reload shows the same picture.
function buildRequests(st) {
  const n = st.nreq | 0, gap = st.gap | 0, base = st.plen | 0, seed = st.seed | 0;
  const out = [];
  for (let i = 0; i < n; i++) {
    const h = hash32(seed, i);
    const P = Math.max(64, Math.round((base * (0.55 + ((h >>> 3) % 90) / 100)) / 64) * 64);
    const G = 24 + (h % 28);        // outputs are many decode steps -- decode is where the time goes
    out.push({ id: i, P, G, arrival: i * gap });
  }
  return out;
}

const mkRec = (r, bw) => ({
  ...r, pT: preTicks(r.P), xT: xferTicks(r.P, bw), xMs: xferMs(r.P, bw), mb: kvMB(r.P),
  preStart: null, preEnd: null, xStart: null, xEnd: null, decStart: null, decEnd: null,
  preM: null, decM: null, stall: 0, left: 0, fin: false,
});

// ---- deployment A: two pools + one interconnect ---------------------------
// A prefill machine runs one prompt at a time (the GEMM already fills it). The
// link is modelled as ONE shared point-to-point path: transfers queue on it,
// which is exactly how a thin link turns into the system bottleneck. A decode
// machine batches up to BATCH sequences and advances all of them per tick --
// batching is nearly free when you are bandwidth-bound, because the weight read
// is paid once for the whole batch.
function simDisagg(reqs, nP, nD, bw) {
  const recs = reqs.map((r) => mkRec(r, bw));
  const pre = new Array(nP).fill(null);          // {rec, endT}
  const dec = Array.from({ length: nD }, () => []);
  const link = { rec: null, endT: 0 };
  const qPre = [], qX = [], qDec = [];
  const lanePre = Array.from({ length: nP }, () => []);
  const laneDec = Array.from({ length: nD }, () => []);
  const laneLink = [], qLen = [];
  let t = 0, done = 0, guard = 0;

  while (done < recs.length && guard++ < 4000) {
    for (const r of recs) if (r.preStart == null && !r.queued && r.arrival <= t) { r.queued = true; qPre.push(r); }
    for (let m = 0; m < nP; m++) {
      if (!pre[m] && qPre.length) {
        const r = qPre.shift();
        r.preM = m; r.preStart = t; r.preEnd = t + r.pT;
        pre[m] = { rec: r, endT: r.preEnd };
      }
    }
    if (!link.rec && qX.length) {
      const r = qX.shift();
      r.xStart = t; r.xEnd = t + r.xT; link.rec = r; link.endT = r.xEnd;
    }
    for (;;) {
      if (!qDec.length) break;
      let best = -1, bl = BATCH;
      for (let m = 0; m < nD; m++) if (dec[m].length < bl) { bl = dec[m].length; best = m; }
      if (best < 0) break;
      const r = qDec.shift();
      r.decM = best; r.decStart = t; r.left = Math.max(1, r.G - 1); dec[best].push(r);
    }

    for (let m = 0; m < nP; m++) lanePre[m][t] = pre[m] ? pre[m].rec.id : null;
    laneLink[t] = link.rec ? link.rec.id : null;
    for (let m = 0; m < nD; m++) laneDec[m][t] = dec[m].map((r) => r.id);
    qLen[t] = { pre: qPre.length, x: qX.length, dec: qDec.length };

    t++;
    for (let m = 0; m < nP; m++) if (pre[m] && pre[m].endT <= t) { qX.push(pre[m].rec); pre[m] = null; }
    if (link.rec && link.endT <= t) { qDec.push(link.rec); link.rec = null; }
    for (let m = 0; m < nD; m++) {
      for (const r of dec[m]) { r.left--; if (r.left <= 0 && !r.fin) { r.decEnd = t; r.fin = true; done++; } }
      dec[m] = dec[m].filter((r) => !r.fin);
    }
  }
  for (const r of recs) { if (r.preEnd == null) r.preEnd = t; if (r.decStart == null) r.decStart = t; if (r.decEnd == null) r.decEnd = t; }
  return { kind: 'disagg', recs, lanePre, laneLink, laneDec, qLen, span: Math.max(1, t), nP, nD };
}

// ---- deployment B: one shared pool (the baseline) -------------------------
// Every machine does both phases. It runs its decode batch a tick at a time --
// until a prompt lands on it, and then the whole machine goes to that prefill
// and its batch emits NOTHING until the prefill ends. That is head-of-line
// blocking, and it is what couples time-to-first-token to inter-token latency.
function simSingle(reqs, M, bw) {
  const recs = reqs.map((r) => mkRec(r, bw));
  for (const r of recs) { r.xT = 0; r.xMs = 0; }   // nothing moves: same machine
  const mach = Array.from({ length: M }, () => ({ pre: null, act: [], q: [] }));
  const lane = Array.from({ length: M }, () => []);
  let t = 0, done = 0, guard = 0;

  while (done < recs.length && guard++ < 4000) {
    for (const r of recs) {
      if (r.queued || r.arrival > t) continue;
      let best = 0, bl = Infinity;
      for (let m = 0; m < M; m++) {
        const q = mach[m], load = q.q.reduce((s, x) => s + x.pT, 0)
          + (q.pre ? q.pre.endT - t : 0) + q.act.length;
        if (load < bl) { bl = load; best = m; }
      }
      r.queued = true; mach[best].q.push(r);
    }
    for (let m = 0; m < M; m++) {
      const q = mach[m];
      if (!q.pre && q.q.length && q.act.length < BATCH) {
        const r = q.q.shift();
        r.preM = m; r.decM = m; r.preStart = t; r.preEnd = t + r.pT;
        q.pre = { rec: r, endT: r.preEnd };
      }
    }
    for (let m = 0; m < M; m++) lane[m][t] = { pre: mach[m].pre ? mach[m].pre.rec.id : null, act: mach[m].act.map((r) => r.id) };

    t++;
    for (let m = 0; m < M; m++) {
      const q = mach[m];
      if (q.pre) {
        for (const r of q.act) r.stall++;                 // the cost of sharing
        if (q.pre.endT <= t) {
          const r = q.pre.rec; q.pre = null;
          r.decStart = t; r.left = Math.max(1, r.G - 1); q.act.push(r);
        }
      } else {
        for (const r of q.act) { r.left--; if (r.left <= 0 && !r.fin) { r.decEnd = t; r.fin = true; done++; } }
        q.act = q.act.filter((r) => !r.fin);
      }
    }
  }
  for (const r of recs) { if (r.preEnd == null) r.preEnd = t; if (r.decStart == null) r.decStart = t; if (r.decEnd == null) r.decEnd = t; }
  return { kind: 'single', recs, lane, span: Math.max(1, t), M };
}

// ---- metrics -------------------------------------------------------------
// TTFT is arrival -> end of prefill: the first token comes out of the machine
// that ran the prompt, in BOTH deployments. Everything after it -- the KV
// transfer, the wait for a decode slot, the stalls -- lands on TPOT, the
// average gap between the tokens that follow.
function metrics(sim) {
  const recs = sim.recs, n = Math.max(1, recs.length);
  let ttft = 0, e2e = 0, tpot = 0, xf = 0, lq = 0, st = 0, worst = 0, mb = 0;
  for (const r of recs) {
    ttft += r.preEnd - r.arrival;
    e2e += r.decEnd - r.arrival;
    worst = Math.max(worst, r.decEnd - r.arrival);
    tpot += (r.decEnd - r.preEnd) / Math.max(1, r.G - 1);
    xf += r.xEnd != null ? r.xEnd - r.xStart : 0;
    lq += r.xStart != null ? r.xStart - r.preEnd : 0;   // queued for a busy link
    st += r.stall; mb += r.xT ? r.mb : 0;
  }
  return {
    n, span: sim.span,
    ttft: (ttft / n) * TICK_MS, e2e: (e2e / n) * TICK_MS, tpot: (tpot / n) * TICK_MS,
    xfer: (xf / n) * TICK_MS, linkq: (lq / n) * TICK_MS, stall: (st / n) * TICK_MS, worst: worst * TICK_MS, mb: mb / n,
    thr: n / ((sim.span * TICK_MS) / 1000),
  };
}

// ---- drawing helpers -----------------------------------------------------
function roundRect(ctx, x, y, w, h, r) {
  const rr = Math.max(0, Math.min(r, w / 2, h / 2));
  ctx.beginPath(); ctx.moveTo(x + rr, y);
  ctx.arcTo(x + w, y, x + w, y + h, rr); ctx.arcTo(x + w, y + h, x, y + h, rr);
  ctx.arcTo(x, y + h, x, y, rr); ctx.arcTo(x, y, x + w, y, rr); ctx.closePath();
}
// Diagonal hatch = time paid for and not spent generating.
function hatch(ctx, x, y, w, h, color, bg) {
  if (w <= 0 || h <= 0) return;
  ctx.save();
  ctx.beginPath(); ctx.rect(x, y, w, h); ctx.clip();
  if (bg) { ctx.fillStyle = bg; ctx.fillRect(x, y, w, h); }
  ctx.strokeStyle = color; ctx.lineWidth = 1;
  for (let k = -h; k < w; k += 5) { ctx.beginPath(); ctx.moveTo(x + k, y + h); ctx.lineTo(x + k + h, y); ctx.stroke(); }
  ctx.restore();
}

let cur = null;     // the two simulated deployments + their metrics
let geom = null;    // hit-test geometry captured each draw
let drag = null;    // {mode:'split'|'bw', ...} while a handle is held

// The sweet spot is a real curve, so draw it rather than assert it: re-run the
// disaggregated deployment at EVERY possible split and keep the end-to-end
// latency of each. Too few prefill machines and prompts queue; too few decode
// machines and prefilled requests queue for a slot.
function sweep(reqs, M, bw) {
  const out = [];
  for (let k = 1; k <= M - 1; k++) {
    const m = metrics(simDisagg(reqs, k, M - k, bw));
    out.push({ k, e2e: m.e2e, thr: m.thr });
  }
  return out;
}

// Memoized on the full control signature. The transport rebuilds on every
// state change -- and in A/B compare it rebuilds TWICE per animation frame,
// once per pane -- so recomputing both deployments plus the split sweep each
// time would pin a core for no new information.
const simCache = new Map();

function rebuild(st) {
  const key = [st.M, st.split, st.bw, st.nreq, st.gap, st.plen, st.seed].join('|');
  const hit = simCache.get(key);
  if (hit) { cur = hit; return hit.steps; }
  const M = st.M | 0, bw = +st.bw;
  const nP = clamp(st.split | 0, 1, M - 1), nD = M - nP;
  const reqs = buildRequests(st);
  const D = simDisagg(reqs, nP, nD, bw), S = simSingle(reqs, M, bw);
  const span = Math.max(D.span, S.span);
  const sw = sweep(reqs, M, bw);
  let best = sw[0];
  for (const p of sw) if (p.e2e < best.e2e) best = p;
  const steps = Array.from({ length: span }, (_, i) => ({ t: i, label: `t = ${i * TICK_MS} ms  (decode step ${i + 1} / ${span})` }));
  cur = { reqs, M, nP, nD, bw, D, S, span, sw, best, mD: metrics(D), mS: metrics(S), steps };
  if (simCache.size > 24) simCache.clear();
  simCache.set(key, cur);
  return steps;
}

// One request's phase breakdown, in ms, from whichever deployment's record.
function phases(r) {
  const wait = (r.preStart - r.arrival) * TICK_MS;
  const pre = (r.preEnd - r.preStart) * TICK_MS;
  const linkQ = r.xStart != null ? (r.xStart - r.preEnd) * TICK_MS : 0;
  const move = r.xEnd != null ? (r.xEnd - r.xStart) * TICK_MS : 0;
  const slotQ = (r.decStart - (r.xEnd != null ? r.xEnd : r.preEnd)) * TICK_MS;
  const dec = (r.decEnd - r.decStart) * TICK_MS;
  return { wait, pre, linkQ, move, slotQ, dec, stall: r.stall * TICK_MS };
}

function tipFor(r, kind, bw) {
  const p = phases(r);
  const head = kind === 'disagg'
    ? `request ${r.id} · disaggregated (prefill pool → link → decode pool)`
    : `request ${r.id} · one shared pool`;
  const lines = [
    head,
    `prompt ${r.P} tokens · output ${r.G} tokens · arrived at ${r.arrival * TICK_MS} ms`,
    `queued ${p.wait.toFixed(0)} ms → prefill ${p.pre.toFixed(0)} ms on machine ${r.preM}`,
  ];
  if (kind === 'disagg') {
    lines.push(`KV cache = ${r.P} tok × ${KV_MB_PER_TOK} MB/tok = ${r.mb.toFixed(0)} MB (K+V, every layer)`);
    lines.push(`transfer = ${r.mb.toFixed(0)} MB ÷ ${bw} GB/s = ${r.xMs.toFixed(1)} ms` + (p.linkQ > 0 ? ` (+ ${p.linkQ.toFixed(0)} ms queued for the link)` : ''));
    if (p.slotQ > 0) lines.push(`then waited ${p.slotQ.toFixed(0)} ms for a free decode slot`);
    lines.push(`decode ${p.dec.toFixed(0)} ms on decode machine ${r.decM}`);
  } else {
    lines.push(`decode ${p.dec.toFixed(0)} ms on the same machine, of which ${p.stall.toFixed(0)} ms blocked by a co-located prefill`);
  }
  const ttft = (r.preEnd - r.arrival) * TICK_MS, e2e = (r.decEnd - r.arrival) * TICK_MS;
  lines.push(`TTFT ${ttft.toFixed(0)} ms · end-to-end ${e2e.toFixed(0)} ms · TPOT ${((r.decEnd - r.preEnd) * TICK_MS / Math.max(1, r.G - 1)).toFixed(1)} ms/token`);
  return lines.join('\n');
}

mount({
  mount: 'body',
  title: 'disaggregation — prefill pool, decode pool, KV cache on the wire',
  blurb: 'The two phases of inference want opposite machines: prefill is one big parallel GEMM (compute-bound), decode is one skinny step per token that re-reads the whole cache (memory-bandwidth-bound). Share one pool and they fight — a prefill takes the machine, and every sequence decoding on it emits nothing until the prefill ends (the hatched gaps below). Split the fleet instead, and ship each request\'s KV cache from the prefill machine to a decode machine. That transfer is not free: it is K and V for every layer of every prompt token, on a real link, sitting between the first token and all the rest. DRAG the allocation bar to move machines between the pools, and drag the interconnect pipe up or down to change its bandwidth — there is a sweet spot, and there is a link thin enough to make the whole idea a loss. Hover any request for its phase breakdown, or any worker for what it is doing.',
  prefer: 'canvas2d',
  aspect: '16 / 13',
  autoplay: true,
  animate: true,
  compare: { key: 'split', a: 1, b: 5, labelA: '1 machine prefilling', labelB: '5 machines prefilling', rebuild: true },
  challenges: [
    {
      goal: 'Land on the sweet spot: a split within 5% of the best end-to-end latency this fleet can reach.',
      hint: 'too few prefill machines and prompts queue for compute; too few decode machines and prefilled requests queue for a slot. The sweep strip above the allocation bar plots every split — walk the divider to the short bar.',
      check: (api) => ({
        solved: (api.probe.e2eD ?? 1e9) <= 1.05 * (api.probe.bestE2e ?? 0),
        detail: `end-to-end ${(api.probe.e2eD ?? 0).toFixed(0)} ms at ${api.probe.split} prefill · best is ${(api.probe.bestE2e ?? 0).toFixed(0)} ms at ${api.probe.bestSplit}`,
      }),
    },
    {
      goal: 'Throttle the link until disaggregation LOSES on end-to-end latency.',
      hint: 'drag the interconnect pipe down (or lower the bandwidth slider) — and remember the transfer grows with prompt length.',
      check: (api) => ({
        solved: (api.probe.e2eD ?? 0) > (api.probe.e2eS ?? 1e9),
        detail: `end-to-end ${(api.probe.e2eD ?? 0).toFixed(0)} ms vs ${(api.probe.e2eS ?? 0).toFixed(0)} ms shared-pool`,
      }),
    },
  ],
  controls: (c, page) => {
    c.stepper('M', { label: 'machines in the fleet', min: 3, max: 8, value: 6 });
    c.slider('split', { label: 'machines in the PREFILL pool', min: 1, max: 7, step: 1, value: 4, rebuild: true });
    c.slider('bw', { label: 'interconnect (GB/s)', min: 2, max: 100, step: 1, value: 25, rebuild: true });
    c.stepper('nreq', { label: 'requests', min: 4, max: 20, value: 14 });
    c.stepper('gap', { label: 'arrival gap (steps)', min: 1, max: 8, value: 2 });
    c.stepper('plen', { label: 'prompt tokens (mean)', min: 128, max: 1024, step: 128, value: 512 });
    c.slider('seed', { label: 'workload seed', min: 0, max: 99, step: 1, value: 7, rebuild: true });
    c.transport({ compute: () => rebuild(page.state), speed: 24, loop: true });
  },

  // Direct manipulation: the allocation bar re-partitions the fleet, the pipe
  // sets the link bandwidth. Both write through controls.set() so the widget,
  // the URL and the simulation all move together.
  onPointer: (page, ev) => {
    if (!geom || !cur) return;
    const st = page.state;
    const inRect = (R, pad = 0) => R && ev.x >= R.x - pad && ev.x <= R.x + R.w + pad && ev.y >= R.y - pad && ev.y <= R.y + R.h + pad;
    if (ev.type === 'down') {
      if (inRect(geom.alloc, 8)) drag = { mode: 'split' };
      else if (inRect(geom.pipe, 16)) drag = { mode: 'bw', y0: ev.y, bw0: +st.bw };
      else drag = null;
    } else if (ev.type === 'up' || ev.type === 'leave') {
      drag = null;
    }
    if (!drag || !page.pointer.down) return;
    if (drag.mode === 'split') {
      const M = cur.M;
      const want = clamp(Math.round(((ev.x - geom.alloc.x) / Math.max(1, geom.alloc.w)) * M), 1, M - 1);
      if (want !== (st.split | 0)) page.controls.set('split', want, { rebuild: true });
    } else if (drag.mode === 'bw') {
      // vertical drag on a log axis: up = wider pipe.
      const want = clamp(Math.round(drag.bw0 * Math.pow(10, (drag.y0 - ev.y) / 110)), 2, 100);
      if (want !== (st.bw | 0)) page.controls.set('bw', want, { rebuild: true });
    }
  },

  draw: (page) => {
    const r = page.renderer, ctx = page.ctx, st = page.state;
    if (!cur) rebuild(st);
    const { reqs, M, nP, nD, bw, D, S, span, sw, best, mD, mS } = cur;
    r.clear(T.n0);
    const s = page.step();
    const now = clamp(s ? s.t : span - 1, 0, span - 1);
    page.probe = { tpotD: mD.tpot, tpotS: mS.tpot, thrD: mD.thr, thrS: mS.thr, e2eD: mD.e2e, e2eS: mS.e2e, bestE2e: best.e2e, bestSplit: best.k, split: nP, bw };

    const W = page.W, H = page.H, padL = 78, padR = 12;
    const gx = padL, gw = Math.max(60, W - padL - padR);
    const cellW = gw / span;
    const xOf = (t) => gx + t * cellW;
    const mono = (px) => `${px}px ui-monospace, monospace`;

    // ---- header ---------------------------------------------------------
    r.label(`${reqs.length} requests · ${M} machines · ${nP} prefill + ${nD} decode · link ${bw} GB/s · t = ${now * TICK_MS} ms`,
      gx, 15, { color: T.n14, font: mono(13) });
    // colour key, painted with the same tokens the timelines use
    let lx = gx;
    const key = (col, text, hatched) => {
      if (hatched) hatch(ctx, lx, 20, 9, 8, alphaOf(col, 0.9), alphaOf(col, 0.15));
      else { ctx.fillStyle = alphaOf(col, 0.85); ctx.fillRect(lx, 20, 9, 8); }
      r.label(text, lx + 12, 27, { color: T.n11, font: mono(10) });
      ctx.save(); ctx.font = mono(10); lx += 12 + ctx.measureText(text).width + 14; ctx.restore();
    };
    key(T.accent, 'prefill (compute-bound)');
    key(T.violet, 'KV transfer');
    key(T.ok, 'decode (bandwidth-bound)');
    key(T.warn, 'blocked by a co-located prefill', true);
    key(T.n9, 'queued', true);

    // ---- the deployment picture: two pools and one interconnect ----------
    const poolTop = 34, poolH = clamp(H * 0.15, 84, 104);   // two chip rows at most
    const xa = 8, xb = W - 8, pipeW = Math.min(132, (xb - xa) * 0.17);
    const avail = (xb - xa) - pipeW - 16;
    const fr = nP / M;
    const boxA = { x: xa, y: poolTop, w: Math.max(70, avail * fr), h: poolH };
    const pipe = { x: boxA.x + boxA.w + 8, y: poolTop + poolH / 2 - 6, w: pipeW, h: 12 };
    const boxB = { x: pipe.x + pipeW + 8, y: poolTop, w: Math.max(70, xb - (pipe.x + pipeW + 8)), h: poolH };

    const drawPool = (box, title, sub, tone, count, chipFn) => {
      ctx.save();
      roundRect(ctx, box.x, box.y, box.w, box.h, 8);
      ctx.fillStyle = alphaOf(tone, 0.06); ctx.fill();
      ctx.strokeStyle = alphaOf(tone, 0.55); ctx.lineWidth = 1.2; ctx.stroke();
      ctx.restore();
      r.label(title, box.x + 9, box.y + 15, { color: tone, font: mono(11) });
      r.label(sub, box.x + 9, box.y + 27, { color: T.n10, font: mono(9.5) });
      const cols = Math.max(1, Math.min(count, Math.floor((box.w - 14) / 62)));
      const rows = Math.ceil(count / cols);
      const cwd = (box.w - 14) / cols, chh = Math.min(30, (box.h - 40) / rows);
      const out = [];
      for (let i = 0; i < count; i++) {
        const cx = box.x + 7 + (i % cols) * cwd, cy = box.y + 34 + Math.floor(i / cols) * chh;
        const R = { x: cx + 2, y: cy + 2, w: cwd - 5, h: chh - 5 };
        chipFn(i, R);
        out.push(R);
      }
      return out;
    };

    // prefill workers: one prompt each, with a progress bar of its own prefill.
    const chipsA = drawPool(boxA, `PREFILL POOL · ${nP}`, 'compute-bound: one big GEMM', T.accent, nP, (m, R) => {
      const id = D.lanePre[m] ? D.lanePre[m][now] : null;
      ctx.save();
      roundRect(ctx, R.x, R.y, R.w, R.h, 5);
      if (id != null) { ctx.fillStyle = alphaOf(categorical(id), 0.9); ctx.fill(); }
      else { ctx.fillStyle = rgbaToken('n14', 0.05); ctx.fill(); ctx.strokeStyle = rgbaToken('n14', 0.16); ctx.lineWidth = 1; ctx.stroke(); }
      ctx.font = mono(10); ctx.textBaseline = 'middle'; ctx.textAlign = 'left';
      ctx.fillStyle = id != null ? inkOn(categorical(id)) : T.n9;
      ctx.fillText(id != null ? `r${id}` : 'idle', R.x + 6, R.y + R.h / 2);
      if (id != null) {
        const rec = D.recs[id], f = clamp((now - rec.preStart + 1) / Math.max(1, rec.pT), 0, 1);
        ctx.fillStyle = alphaOf(inkOn(categorical(id)), 0.45);
        ctx.fillRect(R.x + 4, R.y + R.h - 5, (R.w - 8) * f, 2.5);
      }
      ctx.restore();
    });

    // decode workers: BATCH slots each, filled by whoever is resident.
    const chipsB = drawPool(boxB, `DECODE POOL · ${nD}`, `bandwidth-bound: batches ${BATCH} seqs`, T.ok, nD, (m, R) => {
      const ids = (D.laneDec[m] && D.laneDec[m][now]) || [];
      ctx.save();
      roundRect(ctx, R.x, R.y, R.w, R.h, 5);
      ctx.fillStyle = rgbaToken('n14', 0.05); ctx.fill();
      ctx.strokeStyle = rgbaToken('n14', 0.16); ctx.lineWidth = 1; ctx.stroke();
      const sw = (R.w - 10) / BATCH;
      for (let k = 0; k < BATCH; k++) {
        const sx = R.x + 5 + k * sw;
        ctx.fillStyle = k < ids.length ? alphaOf(categorical(ids[k]), 0.9) : rgbaToken('n14', 0.09);
        ctx.fillRect(sx, R.y + 5, sw - 2, R.h - 10);
      }
      ctx.restore();
    });

    // the interconnect: thickness is the bandwidth (log), and the KV of the
    // request currently in flight rides along it.
    const pipeH = 5 + 30 * (Math.log10(bw) / 2);
    pipe.y = poolTop + poolH / 2 - pipeH / 2; pipe.h = pipeH;
    ctx.save();
    roundRect(ctx, pipe.x, pipe.y, pipe.w, pipe.h, Math.min(6, pipeH / 2));
    ctx.fillStyle = alphaOf(T.violet, 0.16); ctx.fill();
    ctx.strokeStyle = alphaOf(T.violet, 0.7); ctx.lineWidth = 1.2; ctx.stroke();
    const flying = D.laneLink[now];
    if (flying != null) {
      ctx.beginPath(); roundRect(ctx, pipe.x, pipe.y, pipe.w, pipe.h, Math.min(6, pipeH / 2)); ctx.clip();
      for (let k = 0; k < 5; k++) {
        const f = ((page.t * 0.9 + k / 5) % 1);
        ctx.fillStyle = alphaOf(categorical(flying), 0.95);
        ctx.beginPath(); ctx.arc(pipe.x + f * pipe.w, pipe.y + pipe.h / 2, Math.min(3.5, pipeH / 3), 0, 6.2832); ctx.fill();
      }
    }
    ctx.restore();
    r.label(`${bw} GB/s  ↕drag`, pipe.x + pipe.w / 2, pipe.y - 6, { color: T.violet, font: mono(10), align: 'center' });
    r.label(flying != null ? `KV of r${flying}: ${D.recs[flying].mb.toFixed(0)} MB` : 'link idle',
      pipe.x + pipe.w / 2, pipe.y + pipe.h + 12, { color: flying != null ? T.violetDeep : T.n9, font: mono(9.5), align: 'center' });

    // ---- the split sweep: every possible allocation, measured -----------
    // The sweet spot is not asserted, it is re-simulated: one bar per split,
    // height = that split's end-to-end latency. The two failure modes are the
    // two tall ends -- prefill-starved on the left, decode-starved on the right.
    const swY = poolTop + poolH + 6, swH = 24;
    const swMax = sw.reduce((a, p) => Math.max(a, p.e2e), 0);
    const swMin = sw.reduce((a, p) => Math.min(a, p.e2e), Infinity);
    const swBarW = Math.min(26, gw / (M * 1.4));
    const swX = (k) => gx + (k / M) * gw;
    ctx.save();
    for (const p of sw) {
      const hh = swH * (0.16 + 0.84 * ((p.e2e - swMin) / Math.max(1e-6, swMax - swMin)));
      ctx.fillStyle = p.k === nP ? alphaOf(T.accent, 0.95) : alphaOf(p.k === best.k ? T.ok : T.n9, 0.5);
      ctx.fillRect(swX(p.k) - swBarW / 2, swY + swH - hh, swBarW, hh);
    }
    ctx.restore();
    ctx.save();     // ▼ marks the best split the sweep found
    ctx.fillStyle = T.ok;
    ctx.beginPath(); ctx.moveTo(swX(best.k) - 4, swY + 1); ctx.lineTo(swX(best.k) + 4, swY + 1); ctx.lineTo(swX(best.k), swY + 7); ctx.closePath(); ctx.fill();
    ctx.restore();
    r.label('sweep', gx - 6, swY + swH * 0.8, { color: T.n10, font: mono(9), align: 'right' });
    r.label('end-to-end at every split (shorter = better) · ▼ best', gx + gw, swY - 2, { color: T.n10, font: mono(9), align: 'right' });

    // ---- the allocation bar: drag it to re-partition the fleet -----------
    const alloc = { x: gx, y: swY + swH + 5, w: gw, h: 15 };
    ctx.save();
    const cwAl = alloc.w / M;
    for (let m = 0; m < M; m++) {
      ctx.fillStyle = alphaOf(m < nP ? T.accent : T.ok, drag && drag.mode === 'split' ? 0.85 : 0.6);
      ctx.fillRect(alloc.x + m * cwAl + 1, alloc.y, cwAl - 2, alloc.h);
    }
    ctx.strokeStyle = T.n14; ctx.lineWidth = 2.5;
    ctx.beginPath(); ctx.moveTo(alloc.x + nP * cwAl, alloc.y - 4); ctx.lineTo(alloc.x + nP * cwAl, alloc.y + alloc.h + 4); ctx.stroke();
    ctx.restore();
    r.label('fleet', alloc.x - 8, alloc.y + 11, { color: T.n10, font: mono(10), align: 'right' });
    r.label('◀ drag the divider: machines move between the pools ▶', alloc.x + alloc.w / 2, alloc.y + 11.5,
      { color: T.n13, font: mono(9.5), align: 'center' });

    // ---- the two timelines, one shared x axis ---------------------------
    const bdH = Math.max(64, H * 0.24);
    const tlTop = alloc.y + alloc.h + 20;
    const tlAll = Math.max(70, H - tlTop - bdH - 16);
    // A decode lane carries BATCH stacked sequences, so it needs more height
    // than a prefill lane (which is one request wide) -- otherwise the batch
    // slots collapse into invisible hairlines.
    const units = nP + 1 + (nD + M) * 1.8;
    const laneH = clamp((tlAll - 66) / units, 3.5, 16);
    const decH = laneH * 1.8;

    const lanes = [];         // hit-test: {y,h,kind,m,dep}
    const drawGridRules = (y0, h) => {
      ctx.save(); ctx.strokeStyle = rgbaToken('n14', 0.1); ctx.lineWidth = 1;
      const tick = Math.max(5, Math.ceil(span / 24 / 5) * 5);
      for (let t = 0; t <= span; t += tick) { ctx.beginPath(); ctx.moveTo(xOf(t), y0); ctx.lineTo(xOf(t), y0 + h); ctx.stroke(); }
      ctx.restore();
    };
    const cellFill = (t, x, y, w, h, fill) => {
      ctx.save(); if (t > now) ctx.globalAlpha = 0.18;
      ctx.fillStyle = fill; ctx.fillRect(x, y, w, h); ctx.restore();
    };

    // -- panel A: disaggregated
    let y = tlTop;
    r.label(`DISAGGREGATED — ${nP} prefill + ${nD} decode machines, KV over a ${bw} GB/s link`, gx, y + 9, { color: T.accent, font: mono(11.5) });
    r.label(`${(mD.span * TICK_MS)} ms to drain · ${mD.thr.toFixed(2)} req/s · TTFT ${mD.ttft.toFixed(0)} ms · TPOT ${mD.tpot.toFixed(1)} ms/tok · transfer ${mD.xfer.toFixed(0)} ms/req`,
      gx, y + 21, { color: T.n11, font: mono(10) });
    y += 27;
    const panelAy = y, panelAh = (nP + 1) * laneH + nD * decH;
    drawGridRules(panelAy, panelAh);
    for (let m = 0; m < nP; m++) {
      lanes.push({ y, h: laneH, kind: 'pre', m, dep: 'disagg' });
      r.label(`prefill ${m}`, gx - 6, y + laneH * 0.78, { color: T.accent, font: mono(9), align: 'right' });
      for (let t = 0; t < span; t++) {
        const id = D.lanePre[m][t];
        cellFill(t, xOf(t), y + 0.8, cellW + 0.5, laneH - 1.6,
          id != null ? alphaOf(categorical(id), 0.9) : rgbaToken('n14', 0.05));
      }
      y += laneH;
    }
    lanes.push({ y, h: laneH, kind: 'link', m: 0, dep: 'disagg' });
    r.label('link', gx - 6, y + laneH * 0.78, { color: T.violet, font: mono(9), align: 'right' });
    for (let t = 0; t < span; t++) {
      const id = D.laneLink[t];
      cellFill(t, xOf(t), y + 0.8, cellW + 0.5, laneH - 1.6, id != null ? alphaOf(categorical(id), 0.75) : rgbaToken('n14', 0.05));
      if (id != null && t <= now) hatch(ctx, xOf(t), y + 0.8, cellW + 0.5, laneH - 1.6, alphaOf(T.violet, 0.85), null);
    }
    y += laneH;
    for (let m = 0; m < nD; m++) {
      lanes.push({ y, h: decH, kind: 'dec', m, dep: 'disagg' });
      r.label(`decode ${m}`, gx - 6, y + decH * 0.7, { color: T.ok, font: mono(9), align: 'right' });
      const sh = (decH - 1.6) / BATCH;
      for (let t = 0; t < span; t++) {
        const ids = D.laneDec[m][t] || [];
        cellFill(t, xOf(t), y + 0.8, cellW + 0.5, decH - 1.6, rgbaToken('n14', 0.05));
        for (let k = 0; k < ids.length; k++) cellFill(t, xOf(t), y + 0.8 + k * sh, cellW + 0.5, sh - 0.4, alphaOf(categorical(ids[k]), 0.9));
      }
      y += decH;
    }

    // -- panel B: one shared pool
    y += 12;
    r.label(`ONE SHARED POOL — ${M} machines, each doing both phases (no transfer, but they interfere)`, gx, y + 9, { color: T.warnDeep, font: mono(11.5) });
    r.label(`${(mS.span * TICK_MS)} ms to drain · ${mS.thr.toFixed(2)} req/s · TTFT ${mS.ttft.toFixed(0)} ms · TPOT ${mS.tpot.toFixed(1)} ms/tok · blocked ${mS.stall.toFixed(0)} ms/req`,
      gx, y + 21, { color: T.n11, font: mono(10) });
    y += 27;
    drawGridRules(y, M * decH);
    for (let m = 0; m < M; m++) {
      lanes.push({ y, h: decH, kind: 'both', m, dep: 'single' });
      r.label(`machine ${m}`, gx - 6, y + decH * 0.7, { color: T.n11, font: mono(9), align: 'right' });
      const preH = Math.max(2, (decH - 1.6) * 0.3), dh = (decH - 1.6) - preH;
      const sh = dh / BATCH;
      for (let t = 0; t < span; t++) {
        const c = S.lane[m][t] || { pre: null, act: [] };
        cellFill(t, xOf(t), y + 0.8, cellW + 0.5, preH, c.pre != null ? alphaOf(categorical(c.pre), 0.9) : rgbaToken('n14', 0.05));
        cellFill(t, xOf(t), y + 0.8 + preH, cellW + 0.5, dh, rgbaToken('n14', 0.05));
        for (let k = 0; k < c.act.length; k++) {
          cellFill(t, xOf(t), y + 0.8 + preH + k * sh, cellW + 0.5, sh - 0.3, alphaOf(categorical(c.act[k]), 0.9));
        }
        // the whole point: while this machine prefills, its batch generates nothing.
        if (c.pre != null && c.act.length && t <= now) hatch(ctx, xOf(t), y + 0.8 + preH, cellW + 0.5, dh, alphaOf(T.warn, 0.9), null);
      }
      y += decH;
    }

    // shared x axis + playhead across both panels
    const axY = y + 12;
    ctx.save(); ctx.strokeStyle = rgbaToken('n14', 0.25); ctx.lineWidth = 1;
    ctx.beginPath(); ctx.moveTo(gx, axY - 8); ctx.lineTo(gx + gw, axY - 8); ctx.stroke();
    ctx.strokeStyle = T.n13; ctx.lineWidth = 1.5;
    ctx.beginPath(); ctx.moveTo(xOf(now + 1), panelAy - 4); ctx.lineTo(xOf(now + 1), axY - 8); ctx.stroke();
    ctx.restore();
    r.label('time (ms) →', gx + gw, axY - 12, { color: T.n10, font: mono(10), align: 'right' });
    const atick = Math.max(5, Math.ceil(span / 12 / 5) * 5);
    for (let t = 0; t < span; t += atick) r.label(String(t * TICK_MS), xOf(t) + cellW / 2, axY, { color: T.n10, font: mono(9), align: 'center' });

    // ---- per-request latency breakdown ----------------------------------
    const by = Math.max(axY + 4, H - bdH);
    r.label('per-request latency — upper bar: disaggregated (wait · prefill · link queue · KV transfer · slot queue · decode) · lower bar: one shared pool',
      gx, by + 8, { color: T.n11, font: mono(9.5) });
    const rowH = clamp((bdH - 26) / Math.max(1, reqs.length), 4, 15);
    const bars = [];
    reqs.forEach((q, i) => {
      const ry = by + 14 + i * rowH;
      r.label(`r${q.id}`, gx - 6, ry + rowH * 0.7, { color: T.n11, font: mono(9), align: 'right' });
      [{ sim: D, kind: 'disagg' }, { sim: S, kind: 'single' }].forEach((side, k) => {
        const rec = side.sim.recs[q.id];
        const yy = ry + k * (rowH / 2), hh = Math.max(2, rowH / 2 - 1);
        let x = xOf(q.arrival);
        const seg = (ticks, fill, hatched) => {
          const w = ticks * cellW;
          if (w <= 0) return;
          if (hatched) hatch(ctx, x, yy, w, hh, alphaOf(fill, 0.8), alphaOf(fill, 0.14));
          else { ctx.fillStyle = alphaOf(fill, 0.85); ctx.fillRect(x, yy, w, hh); }
          x += w;
        };
        seg(rec.preStart - q.arrival, T.n9, true);
        seg(rec.preEnd - rec.preStart, T.accent, false);
        if (side.kind === 'disagg') {
          seg(rec.xStart - rec.preEnd, T.violet, true);
          seg(rec.xEnd - rec.xStart, T.violet, false);
          seg(rec.decStart - rec.xEnd, T.teal, true);
          seg(rec.decEnd - rec.decStart, T.ok, false);
        } else {
          const dec = rec.decEnd - rec.decStart, stall = Math.min(dec, rec.stall);
          seg(rec.decStart - rec.preEnd, T.teal, true);
          seg(dec - stall, T.ok, false);
          seg(stall, T.warn, true);
        }
        bars.push({ x0: xOf(q.arrival), x1: x, y: yy, h: hh, rec, kind: side.kind });
      });
    });

    // ---- hit-test geometry ----------------------------------------------
    geom = {
      alloc, pipe, cellW, gx, gw, span,
      chips: [
        ...chipsA.map((R, m) => ({ R, kind: 'pre', m })),
        ...chipsB.map((R, m) => ({ R, kind: 'dec', m })),
      ],
      lanes, bars,
      tickAt: (x) => (x < gx || x > gx + gw ? null : clamp(Math.floor((x - gx) / cellW), 0, span - 1)),
    };

    // ---- hover-to-inspect -------------------------------------------------
    if (page.pointer.over && !drag) {
      const px = page.pointer.x, py = page.pointer.y;
      const inR = (R, pad = 0) => px >= R.x - pad && px <= R.x + R.w + pad && py >= R.y - pad && py <= R.y + R.h + pad;
      let tip = null;

      for (const c of geom.chips) {
        if (!inR(c.R)) continue;
        if (c.kind === 'pre') {
          const id = D.lanePre[c.m][now];
          if (id != null) {
            const rec = D.recs[id];
            tip = `prefill machine ${c.m}\nrunning request ${id}: ${rec.P} prompt tokens in one parallel pass\n${rec.pT * TICK_MS} ms of compute (${PRE_MS_PER_TOK} ms/token) — the math units are saturated\nwhen it ends, ${rec.mb.toFixed(0)} MB of KV goes on the wire`;
          } else {
            const q = D.qLen[now] || { pre: 0 };
            tip = `prefill machine ${c.m}: IDLE\n${q.pre ? `${q.pre} prompt(s) queued but every other prefill machine is busy` : 'no prompt is waiting — this pool has spare compute for this arrival rate'}\nmoving a machine to the decode pool would raise decode capacity`;
          }
        } else {
          const ids = (D.laneDec[c.m] && D.laneDec[c.m][now]) || [];
          if (ids.length) {
            tip = `decode machine ${c.m}\n${ids.length} of ${BATCH} slots busy: ${ids.map((i) => 'r' + i).join(', ')}\none step advances ALL of them — batching is nearly free when you are bandwidth-bound\nthe weights and each cache are re-read every ${TICK_MS} ms step`;
          } else {
            const q = D.qLen[now] || { pre: 0, x: 0, dec: 0 };
            const why = q.dec ? `${q.dec} request(s) waiting for a slot elsewhere — this machine will pick one up`
              : q.x || D.laneLink[now] != null ? 'starved: KV is still crossing the link — a wider link would feed it sooner'
                : q.pre ? `starved: ${q.pre} prompt(s) are queued in the prefill pool — that pool is the bottleneck`
                  : 'idle: nothing has been prefilled yet';
            tip = `decode machine ${c.m}: IDLE\n${why}`;
          }
        }
        break;
      }

      if (!tip) {
        for (const b of geom.bars) {
          if (px >= b.x0 && px <= b.x1 && py >= b.y && py <= b.y + b.h) { tip = tipFor(b.rec, b.kind, bw); break; }
        }
      }
      if (!tip) {
        const t = geom.tickAt(px);
        if (t != null) {
          for (const L of lanes) {
            if (py < L.y || py >= L.y + L.h) continue;
            let id = null;
            if (L.dep === 'disagg') {
              if (L.kind === 'pre') id = D.lanePre[L.m][t];
              else if (L.kind === 'link') id = D.laneLink[t];
              else { const ids = D.laneDec[L.m][t] || []; id = ids.length ? ids[clamp(Math.floor(((py - L.y) / L.h) * BATCH), 0, ids.length - 1)] : null; }
              if (id != null) tip = tipFor(D.recs[id], 'disagg', bw);
            } else {
              const c = S.lane[L.m][t] || { pre: null, act: [] };
              const preH = (L.h - 1.6) * 0.3;
              if (py - L.y < preH + 0.8 && c.pre != null) id = c.pre;
              else if (c.act.length) id = c.act[clamp(Math.floor(((py - L.y - preH) / Math.max(1, L.h - preH)) * BATCH), 0, c.act.length - 1)];
              if (id != null) {
                tip = tipFor(S.recs[id], 'single', bw);
                if (c.pre != null && c.act.length) tip += `\nmachine ${L.m} is prefilling r${c.pre} this step, so its ${c.act.length} decoding sequence(s) emit nothing`;
              }
            }
            if (!tip) tip = `${L.dep === 'disagg' ? (L.kind === 'link' ? 'the interconnect' : L.kind === 'pre' ? `prefill machine ${L.m}` : `decode machine ${L.m}`) : `machine ${L.m}`} · t = ${t * TICK_MS} ms\nidle this step`;
            break;
          }
        }
      }
      if (!tip && px >= geom.pipe.x - 16 && px <= geom.pipe.x + geom.pipe.w + 16 && py >= geom.pipe.y - 16 && py <= geom.pipe.y + geom.pipe.h + 16) {
        tip = `interconnect: ${bw} GB/s\nevery disaggregated request must move ${KV_MB_PER_TOK} MB per prompt token\nmean prompt here = ${(reqs.reduce((a, q) => a + q.P, 0) / reqs.length).toFixed(0)} tokens → ${(mD.mb).toFixed(0)} MB → ${(mD.mb / 1024 / bw * 1000).toFixed(1)} ms on the wire\n↕ drag to widen or throttle the link`;
      }
      if (!tip && px >= gx && px <= gx + gw && py >= swY - 4 && py <= swY + swH + 2) {
        tip = 'end-to-end latency at every possible split (re-simulated, not interpolated):\n'
          + sw.map((p) => `  ${p.k} prefill + ${M - p.k} decode → ${p.e2e.toFixed(0)} ms · ${p.thr.toFixed(2)} req/s${p.k === best.k ? '   ← best' : ''}${p.k === nP ? '   ← you are here' : ''}`).join('\n');
      }
      if (!tip && px >= alloc.x && px <= alloc.x + alloc.w && py >= alloc.y - 8 && py <= alloc.y + alloc.h + 8) {
        tip = `fleet allocation: ${nP} prefill + ${nD} decode\nprefill capacity ≈ ${(nP * TICK_MS / PRE_MS_PER_TOK).toFixed(0)} prompt tokens per ${TICK_MS} ms step\ndecode capacity ≈ ${nD * BATCH} concurrent sequences\n◀ drag to move a machine across ▶`;
      }
      if (tip) page.setTip(tip);
    }

    // ---- readout ----------------------------------------------------------
    const pc = (a, b) => (b > 0 ? (100 * a / b).toFixed(0) : '—');
    let o = `same ${reqs.length} requests, same prompts, same ${M} machines — only the DEPLOYMENT differs.    tier:${r.name}\n`;
    o += `DISAGGREGATED  ${nP} prefill + ${nD} decode · ${mD.thr.toFixed(2)} req/s · TTFT ${mD.ttft.toFixed(0)} ms · TPOT ${mD.tpot.toFixed(1)} ms/tok · e2e ${mD.e2e.toFixed(0)} ms · KV transfer ${mD.mb.toFixed(0)} MB = ${mD.xfer.toFixed(0)} ms on the wire + ${mD.linkq.toFixed(0)} ms queued for it\n`;
    o += `SHARED POOL    ${M} mixed         · ${mS.thr.toFixed(2)} req/s · TTFT ${mS.ttft.toFixed(0)} ms · TPOT ${mS.tpot.toFixed(1)} ms/tok · e2e ${mS.e2e.toFixed(0)} ms · blocked by a co-located prefill ${mS.stall.toFixed(0)} ms/req\n`;
    o += `vs the shared pool: throughput ${pc(mD.thr, mS.thr)}% (higher is better) · TPOT ${pc(mD.tpot, mS.tpot)}% (lower is better; 100% = parity) · end-to-end ${pc(mD.e2e, mS.e2e)}% (lower is better)\n`;
    const share = mD.e2e > 0 ? (100 * (mD.xfer + mD.linkq) / mD.e2e) : 0;
    o += `The KV transfer costs ${(mD.xfer + mD.linkq).toFixed(0)} ms of the ${mD.e2e.toFixed(0)} ms end-to-end (${share.toFixed(0)}%) — ${mD.xfer.toFixed(0)} ms of wire time plus ${mD.linkq.toFixed(0)} ms waiting for the link to be free — and it grows with prompt length: ${KV_MB_PER_TOK} MB per token, K and V for every layer. `;
    if (mD.e2e > mS.e2e) {
      o += share > 20
        ? `RIGHT NOW DISAGGREGATION IS A LOSS, and the link is why: ${share.toFixed(0)}% of every request's life goes to the KV transfer at ${bw} GB/s. Widen the link, shorten the prompts, or keep one pool.`
        : `RIGHT NOW DISAGGREGATION IS A LOSS at this split (${nP} prefill + ${nD} decode); the sweep says ${best.k} + ${M - best.k} reaches ${best.e2e.toFixed(0)} ms. Drag the divider there before blaming the idea.`;
    } else {
      o += `The win is inter-token latency: no prefill can ever land on a decode machine, so nothing stalls mid-generation. That is what disaggregation buys — the first-token target and the per-token target stop being one number — and it is not automatically raw throughput, since the shared pool can put every machine on whichever phase is short of capacity.`;
    }
    if (nP !== best.k) o += `  (Best split for this workload: ${best.k} prefill + ${M - best.k} decode → ${best.e2e.toFixed(0)} ms end-to-end.)`;
    page.setReadout(o);
  },
}).then((page) => {
  window.__disaggPage = page;
  const q = new URLSearchParams(location.search);
  const t = page.controls._transport;
  const restore = (k, lo, hi) => { if (q.has(k)) page.controls.set(k, clamp(parseInt(q.get(k), 10) || lo, lo, hi), { rebuild: true, silent: true }); };
  restore('M', 3, 8);
  restore('split', 1, 7);
  restore('bw', 2, 100);
  restore('nreq', 4, 20);
  restore('gap', 1, 8);
  restore('plen', 128, 1024);
  restore('seed', 0, 99);
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
