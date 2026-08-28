// prefix-routing concept page -- choosing WHICH replica serves a request when
// several replicas of the same model each hold their own prefix cache.
//
// One replica reusing a prefix it already computed is the sibling page
// radix-attention. This page is the layer ABOVE that: with N replicas behind a
// load balancer, the cache a request wants may live on a replica the balancer
// was not going to pick. Round-robin throws the reuse away -- a request whose
// long system prompt is already resident on replica 2 lands on replica 3 and
// recomputes every block of it. A prefix-aware router instead hashes the
// prompt's leading blocks and sends the request where the matched prefix is
// longest.
//
// THE FAILURE IS THE POINT. Pure prefix affinity is a pinning function: every
// request that shares one popular system prompt scores highest on the single
// replica that cached it, so they all queue there while the siblings sit idle.
// Cache hit rate goes UP and throughput goes DOWN. So a real router mixes the
// two signals, and the mixing weight is the knob this page hands the reader:
// both extremes are reachable and both are bad in different ways, and the
// middle beats both.
//
// Interactive per the shared render framework's contract:
//  - TRANSPORT: one step per arriving request. Autoplays and loops; the current
//    arrival lights up in its replica's lane with the score that sent it there.
//  - DIRECT MANIPULATION: drag the ◆ marker across the trade-off chart to move
//    the affinity-vs-balance weight, and drag the ▲ handle under the popularity
//    histogram to move the prompt skew. Queue depths, cached prefixes, hit rate
//    and latency all recompute under your hand.
//  - HOVER: a replica card reports its queue depth, which prompt prefixes it
//    holds and how full its cache is, plus its utilisation; an arrival dot
//    reports where it was sent and WHY (per-replica affinity, load and score);
//    the chart reports every metric at the weight under the cursor.
//  - REPLICA COUNT: a stepper from 2 to 8 replicas, with the arrival rate held
//    at a constant per-replica load so the policies stay comparable.
//
// Every number on screen is produced by the simulated run below -- a seeded
// arrival stream, a per-replica FIFO queue, an LRU block cache per replica --
// and is recomputed from scratch whenever a control moves. Nothing is asserted.
//
// Mechanism sources (both public):
//   SGLang -- "SGLang: Efficient Execution of Structured Language Model
//     Programs", https://arxiv.org/abs/2312.07104, for RadixAttention (KV reuse
//     INSIDE one worker). The cross-worker cache-aware balancer this page draws
//     is the separate SGLang Router (sgl-router), shipped in v0.4:
//     https://lmsys.org/blog/2024-12-04-sglang-v0-4/
//   vLLM production-stack -- the KV-cache-aware routing logic in its router:
//     https://docs.vllm.ai/projects/production-stack/en/latest/
import { mount } from '../framework/layout.js';
import { T, alphaOf, rgbaToken, inkOn } from '../framework/theme.js';
import { categorical } from '../framework/render.js';

// ---- the cost model -------------------------------------------------------
// A prompt is a run of fixed-size blocks. The leading PREFIX_BLOCKS are the
// shared system prompt (this is what a router can match on); the trailing
// TAIL_BLOCKS are that request's own text and are never shared with anybody.
const PREFIX_BLOCKS = 8;
const TAIL_BLOCKS = 2;
const TOTAL_BLOCKS = PREFIX_BLOCKS + TAIL_BLOCKS;
const BLOCK_TOKENS = 16;

// Prefill a block that is NOT cached vs. reading one that is. The gap between
// these two numbers is the entire reason a prefix router exists.
const PREFILL_MS_PER_BLOCK = 6.0;
const CACHED_MS_PER_BLOCK = 0.15;
// Generating the answer occupies the replica for a fixed span, so a replica is
// never free just because prefill was cheap.
const DECODE_MS = 40;

// Offered load per replica. Arrivals are scaled to this so that changing the
// replica count compares routing policies instead of comparing an overloaded
// system with an idle one.
const RHO = 0.85;
const SERVICE_NOMINAL = DECODE_MS + (PREFIX_BLOCKS * 0.45 + TAIL_BLOCKS) * PREFILL_MS_PER_BLOCK;

const SWEEP_N = 20;   // trade-off chart resolution: SWEEP_N + 1 weights in [0,1]

// Deterministic PRNG so one URL replays exactly one arrival stream.
function rng(seed) {
  let a = (seed >>> 0) || 1;
  return () => {
    a |= 0; a = (a + 0x6D2B79F5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

const f1 = (v) => (Number.isFinite(v) ? v.toFixed(1) : '–');
const pct = (v) => (Number.isFinite(v) ? (100 * v).toFixed(1) + '%' : '–');
const promptName = (p) => 'prompt ' + String.fromCharCode(65 + p);

// Cache key for "the first k+1 blocks of system prompt p". Two requests share a
// key exactly when they share that prefix, which is what block hashing buys a
// real router: identity by content, not by request id.
const pkey = (p, k) => 'p' + p + ':' + k;

// ---- the arrival stream ---------------------------------------------------
// Prompt popularity follows 1/(rank+1)^skew. skew = 0 is a uniform mix of
// distinct system prompts; a large skew is one dominant prompt with a thin
// tail, which is the regime where pure affinity pins.
function buildArrivals(st) {
  const P = st.prompts, A = st.arrivals, N = st.replicas;
  const w = [];
  let sum = 0;
  for (let i = 0; i < P; i++) { const v = 1 / Math.pow(i + 1, st.skew); w.push(v); sum += v; }
  const probs = w.map((v) => v / sum);
  const cum = []; let c = 0;
  for (let i = 0; i < P; i++) { c += probs[i]; cum.push(c); }

  const iat = SERVICE_NOMINAL / (N * RHO);
  const r = rng(st.seed * 7919 + 13);
  const list = [];
  let t = 0;
  for (let k = 0; k < A; k++) {
    const u = r();
    let p = 0;
    while (p < P - 1 && u > cum[p]) p++;
    list.push({ id: k, t, prompt: p });
    t += iat * (0.4 + 1.2 * r());   // mean 1.0 x iat, so the offered load is RHO
  }
  return { list, probs, iat };
}

// ---- the simulated run ----------------------------------------------------
// Per replica: one FIFO server and one LRU block cache. mode 'rr' is the
// round-robin baseline (the named comparison for every ratio this page
// reports); mode 'aff' is the prefix-aware router at mixing weight `w`.
function simulate(arr, st, mode, w) {
  const N = st.replicas, P = st.prompts, CAP = st.capacity;
  const free = new Array(N).fill(0);
  const busy = new Array(N).fill(0);
  const served = new Array(N).fill(0);
  const cache = [];                      // per replica: Map(key -> recency), insertion-ordered = LRU
  const finished = [];                   // per replica: finish times, for queue depth
  const spans = [];                      // per replica: [start, finish] busy intervals
  for (let i = 0; i < N; i++) { cache.push(new Map()); finished.push([]); spans.push([]); }
  // Work ALREADY DONE by time `now`. Charging the whole service the moment a
  // request is assigned would count future work as done and read 100% at the
  // second arrival, which is not a utilisation.
  const busyBy = (i, now) => (now <= 0 ? 0 : spans[i].reduce((s, v) => s + Math.max(0, Math.min(v[1], now) - v[0]), 0));

  const recs = [];
  let hitBlocks = 0, allBlocks = 0;

  for (let k = 0; k < arr.length; k++) {
    const a = arr[k];

    // What the router can see at this instant.
    const backlog = [], qd = [], match = [], util = [];
    for (let i = 0; i < N; i++) {
      backlog.push(Math.max(0, free[i] - a.t));                 // unfinished work, ms
      qd.push(finished[i].reduce((n, f) => n + (f > a.t ? 1 : 0), 0));
      let m = 0;
      while (m < PREFIX_BLOCKS && cache[i].has(pkey(a.prompt, m))) m++;
      match.push(m);
      util.push(a.t > 0 ? Math.min(1, busyBy(i, a.t) / a.t) : 0);
    }
    const maxB = Math.max(...backlog);

    // The routing score. Affinity is the matched fraction of the shared prefix;
    // load is this replica's backlog relative to the busiest one, so the two
    // terms are on one scale and `w` really does sweep between the extremes.
    const scores = [];
    for (let i = 0; i < N; i++) {
      const aff = match[i] / PREFIX_BLOCKS;
      const load = maxB > 0 ? backlog[i] / maxB : 0;
      scores.push(w * aff - (1 - w) * load);
    }

    let j = 0;
    if (mode === 'rr') {
      j = k % N;
    } else {
      for (let i = 1; i < N; i++) {
        const d = scores[i] - scores[j];
        if (d > 1e-9 || (Math.abs(d) <= 1e-9 && backlog[i] < backlog[j])) j = i;
      }
    }

    const hit = match[j], comp = TOTAL_BLOCKS - hit;
    const prefill = hit * CACHED_MS_PER_BLOCK + comp * PREFILL_MS_PER_BLOCK;
    const start = Math.max(a.t, free[j]);
    const wait = start - a.t;
    const ttft = wait + prefill;                 // queueing delay + prefill = time to first token
    const finish = start + prefill + DECODE_MS;
    free[j] = finish; busy[j] += prefill + DECODE_MS; served[j]++;
    finished[j].push(finish); spans[j].push([start, finish]);
    hitBlocks += hit; allBlocks += TOTAL_BLOCKS;

    // Write this request's blocks into the chosen replica's cache, newest last,
    // then evict from the front until it fits. The tail blocks are unique to
    // this request, so they occupy capacity once and are never matched again --
    // which is exactly how a real cache is pressured.
    const c = cache[j];
    const touch = (key) => { c.delete(key); c.set(key, k); };
    for (let b = 0; b < PREFIX_BLOCKS; b++) touch(pkey(a.prompt, b));
    for (let b = 0; b < TAIL_BLOCKS; b++) touch('t' + k + ':' + b);
    while (c.size > CAP) c.delete(c.keys().next().value);

    // Per-prompt cache occupancy of every replica, for the cards and tooltips.
    const holds = [];
    for (let i = 0; i < N; i++) {
      const row = [];
      for (let p = 0; p < P; p++) { let n = 0; for (let b = 0; b < PREFIX_BLOCKS; b++) if (cache[i].has(pkey(p, b))) n++; row.push(n); }
      holds.push(row);
    }

    recs.push({
      k, t: a.t, prompt: a.prompt, replica: j,
      match, backlog, qd, scores, util, holds,
      size: cache.map((m) => m.size),
      hit, comp, prefill, wait, ttft, start, finish,
      label: `request ${k} (${promptName(a.prompt)}) → replica ${j} — ${hit}/${PREFIX_BLOCKS} prefix blocks reused, ${comp} computed, TTFT ${f1(ttft)} ms`,
    });
  }

  const makespan = Math.max(1e-6, ...free);
  const utilF = busy.map((b) => Math.min(1, b / makespan));
  const maxU = Math.max(...utilF), minU = Math.min(...utilF);
  const tt = recs.map((r) => r.ttft).sort((x, y) => x - y);
  const mean = tt.reduce((s, v) => s + v, 0) / Math.max(1, tt.length);
  const p95 = tt.length ? tt[Math.min(tt.length - 1, Math.ceil(0.95 * tt.length) - 1)] : 0;

  return {
    recs, util: utilF, served, makespan,
    hitRate: allBlocks ? hitBlocks / allBlocks : 0,
    balance: maxU > 0 ? minU / maxU : 1,
    meanTtft: mean, p95Ttft: p95,
    throughput: (1000 * recs.length) / makespan,
  };
}

// The sweep does not depend on `w`, so it is memoized on everything else: the
// reader drags the weight continuously and only the current run is re-simulated.
let memoKey = null, memoVal = null;
function baseRuns(st) {
  const key = [st.replicas, st.prompts, st.skew, st.capacity, st.arrivals, st.seed].join('|');
  if (key === memoKey) return memoVal;
  const A = buildArrivals(st);
  const rr = simulate(A.list, st, 'rr', 0);
  const sweep = [];
  for (let i = 0; i <= SWEEP_N; i++) {
    const w = i / SWEEP_N;
    const s = simulate(A.list, st, 'aff', w);
    sweep.push({ w, hit: s.hitRate, bal: s.balance, ttft: s.meanTtft, p95: s.p95Ttft, thr: s.throughput });
  }
  memoKey = key; memoVal = { A, rr, sweep };
  return memoVal;
}

function fmtKB(blocks) {
  const kb = blocks * BLOCK_TOKENS * 0.5;   // display-only: KV bytes per token are a model constant
  return kb >= 1024 ? (kb / 1024).toFixed(1) + ' MB' : kb.toFixed(0) + ' KB';
}

function roundRect(ctx, x, y, w, h, rr) {
  ctx.beginPath();
  ctx.moveTo(x + rr, y);
  ctx.arcTo(x + w, y, x + w, y + h, rr); ctx.arcTo(x + w, y + h, x, y + h, rr);
  ctx.arcTo(x, y + h, x, y, rr); ctx.arcTo(x, y, x + w, y, rr);
  ctx.closePath();
}

// Hit-test geometry captured each draw(), reused by onPointer + the hover tips.
let geom = null;
let drag = null;   // 'w' | 'skew' while a handle is grabbed

mount({
  mount: 'body',
  title: 'prefix-routing — sending a request to the replica that already holds its prefix',
  blurb: 'With several replicas of one model behind a load balancer, round-robin throws prefix reuse away: a request whose system prompt is already computed on replica 2 lands on replica 3 and recomputes every block of it. A prefix-aware router hashes the prompt\'s leading blocks and sends the request where the matched prefix is longest. But pure affinity is a pinning function — every request sharing one popular system prompt scores highest on the single replica that cached it, so they all queue there while the siblings idle: hit rate up, throughput down. Drag the ◆ across the chart to mix affinity against load balance, and drag the ▲ to change how skewed prompt popularity is. Both extremes are bad in different ways; the middle beats both.',
  prefer: 'canvas2d',
  aspect: '3 / 2',
  autoplay: true,
  compare: { key: 'w', a: 0, b: 1, rebuild: true, labelA: 'w = 0 — pure load balance: even queues, thrashed caches', labelB: 'w = 1 — pure cache affinity: perfect reuse, one replica carries it all' },
  onPointer: (page, ev) => {
    if (!geom) return;
    const g = geom;
    const setW = (x) => {
      const v = Math.max(0, Math.min(1, (x - g.cx) / g.cw));
      page.controls.set('w', +(Math.round(v * 50) / 50).toFixed(2), { rebuild: true });
    };
    const setSkew = (x) => {
      const v = Math.max(0, Math.min(1, (x - g.popX) / g.popW)) * 2.5;
      page.controls.set('skew', +(Math.round(v * 20) / 20).toFixed(2), { rebuild: true });
    };
    if (ev.type === 'down') {
      drag = null;
      if (ev.x >= g.cx - 12 && ev.x <= g.cx + g.cw + 12 && ev.y >= g.cy - 8 && ev.y <= g.cy + g.ch + 16) { drag = 'w'; setW(ev.x); }
      else if (ev.x >= g.popX - 10 && ev.x <= g.popX + g.popW + 10 && ev.y >= g.popY - 6 && ev.y <= g.popY + g.popH + 14) { drag = 'skew'; setSkew(ev.x); }
    } else if (ev.type === 'up' || ev.type === 'leave') {
      drag = null;
    } else if (ev.type === 'move' && drag && page.pointer.down) {
      if (drag === 'w') setW(ev.x); else setSkew(ev.x);
    }
  },
  challenges: [
    {
      goal: 'Pin the whole workload onto one replica: leave a replica that never serves a single request.',
      hint: 'push the ◆ to w = 1 (pure affinity) and the ▲ right (one dominant prompt), then run to the end.',
      check: (api) => ({ solved: (api.probe.idleReplicas ?? 0) > 0, detail: `${api.probe.idleReplicas ?? 0} replica(s) served nothing — needs at least 1` }),
    },
    {
      goal: 'Beat round-robin on mean time-to-first-token by 25% or more.',
      hint: 'neither extreme does it. Find the weight where the two curves cross — the ★ marks the best one found in the sweep.',
      check: (api) => ({ solved: (api.probe.ttftRel ?? 2) <= 0.75, detail: `mean TTFT is ${pct(api.probe.ttftRel ?? 1)} of round-robin — needs ≤ 75.0% (lower is better)` }),
    },
    {
      goal: 'Make cache affinity worthless: get pure affinity (w = 1) to hit within 5 points of pure balance (w = 0) on hit rate.',
      hint: 'drag the ▲ hard left. With many equally-popular distinct prompts and enough cache, there is no popular prefix to follow.',
      check: (api) => ({ solved: Math.abs(api.probe.hitAt1 - api.probe.hitAt0) <= 0.05, detail: `hit rate w=1 ${pct(api.probe.hitAt1)} vs w=0 ${pct(api.probe.hitAt0)} — gap ${pct(Math.abs(api.probe.hitAt1 - api.probe.hitAt0))}, needs ≤ 5.0%` }),
    },
  ],
  controls: (c, page) => {
    c.stepper('replicas', { label: 'replicas', min: 2, max: 8, value: 4 });
    c.slider('w', { label: 'router weight  (0 = balance … 1 = affinity)', min: 0, max: 1, step: 0.02, value: 0.4, rebuild: true });
    c.slider('skew', { label: 'prompt popularity skew', min: 0, max: 2.5, step: 0.05, value: 1.2, rebuild: true });
    c.stepper('prompts', { label: 'distinct system prompts', min: 2, max: 10, value: 6 });
    c.slider('capacity', { label: 'cache per replica (blocks)', min: 8, max: 64, step: 2, value: 24, rebuild: true });
    c.stepper('arrivals', { label: 'requests in the stream', min: 8, max: 48, value: 28 });
    c.stepper('seed', { label: 'arrival seed', min: 1, max: 99, value: 7 });
    c.transport({ compute: () => simulate(baseRuns(page.state).A.list, page.state, 'aff', page.state.w).recs, speed: 3, loop: true });
  },
  draw: (page) => {
    const r = page.renderer, ctx = page.ctx, st = page.state, W = page.W, H = page.H;
    r.clear(T.n0);
    const N = st.replicas, P = st.prompts, A = st.arrivals;
    const base = baseRuns(st);
    const cur = simulate(base.A.list, st, 'aff', st.w);
    const rr = base.rr;
    const s = page.step();
    const kNow = s ? s.k : -1;

    const pad = 14;
    const cardTop = 56, cardH = 78;
    const popY = cardTop + cardH + 42, popH = 26;
    const laneTop = popY + popH + 36;
    const laneH = Math.max(11, Math.min(22, 120 / N));
    const laneBot = laneTop + N * laneH;
    const cy = laneBot + 62, chBot = H - 26, ch = Math.max(60, chBot - cy);
    const cx = pad + 42, cw = W - cx - pad - 8;

    // ---- header -----------------------------------------------------------
    r.label(s ? `step ${s.k + 1} / ${A}  —  ${s.label}` : `${A} requests queued — press ▶ (or step) to admit the first one`,
      pad, 26, { color: T.n14, font: '13px ui-monospace, monospace' });
    r.label(`router: score(replica) = ${st.w.toFixed(2)} × matched-prefix-fraction − ${(1 - st.w).toFixed(2)} × relative-backlog`,
      pad, 43, { color: T.n11, font: '11px ui-monospace, monospace' });

    // ---- replica cards ----------------------------------------------------
    const gap = 6, cardW = (W - 2 * pad - (N - 1) * gap) / N;
    const cards = [];
    // Queue bars are scaled to the deepest queue anywhere in the RUN, not to the
    // deepest queue at this instant -- otherwise every replica reads "full" the
    // moment they all hold one request, and the bar carries no information.
    let maxQ = 1;
    for (const rec of cur.recs) for (const q of rec.qd) if (q > maxQ) maxQ = q;
    for (let i = 0; i < N; i++) {
      const x = pad + i * (cardW + gap), y = cardTop;
      cards.push({ i, x, y, w: cardW, h: cardH });
      const isTarget = s && s.replica === i;
      ctx.save();
      roundRect(ctx, x, y, cardW, cardH, 6);
      ctx.fillStyle = isTarget ? alphaOf(T.accent, 0.13) : rgbaToken('n14', 0.04);
      ctx.fill();
      ctx.strokeStyle = isTarget ? T.accent : T.n5; ctx.lineWidth = isTarget ? 2 : 1;
      ctx.stroke();
      ctx.restore();

      const u = s ? s.util[i] : 0;
      ctx.save();
      ctx.font = 'bold 10.5px ui-monospace, monospace'; ctx.textBaseline = 'top'; ctx.textAlign = 'left';
      ctx.fillStyle = isTarget ? T.accent : T.n13;
      // At eight replicas the card is too narrow for "replica 0" beside the
      // utilisation percent, and the two collide.
      ctx.fillText((cardW < 118 ? 'R' : 'replica ') + i, x + 6, y + 5);
      ctx.textAlign = 'right'; ctx.font = '9.5px ui-monospace, monospace';
      ctx.fillStyle = u > 0.92 ? T.bad : u < 0.25 ? T.n9 : T.n11;
      ctx.fillText(pct(u), x + cardW - 6, y + 6);
      ctx.restore();

      // utilisation bar
      const ubY = y + 20, ubW = cardW - 12;
      ctx.save();
      ctx.fillStyle = rgbaToken('n14', 0.08); ctx.fillRect(x + 6, ubY, ubW, 5);
      ctx.fillStyle = u > 0.92 ? T.bad : alphaOf(T.teal, 0.85);
      ctx.fillRect(x + 6, ubY, ubW * u, 5);
      ctx.restore();

      // queue depth
      const q = s ? s.qd[i] : 0;
      ctx.save();
      ctx.font = '9px ui-monospace, monospace'; ctx.textBaseline = 'middle'; ctx.textAlign = 'left';
      ctx.fillStyle = T.n10; ctx.fillText('queue', x + 6, y + 35);
      const qX = x + 40, qW = cardW - 46;
      ctx.fillStyle = rgbaToken('n14', 0.08); ctx.fillRect(qX, y + 31, qW, 8);
      ctx.fillStyle = q >= maxQ && q > 0 ? T.bad : alphaOf(T.warn, 0.85);
      ctx.fillRect(qX, y + 31, qW * (q / maxQ), 8);
      ctx.fillStyle = q ? inkOn(T.warn) : T.n9;
      ctx.textAlign = 'center';
      if (q) ctx.fillText(String(q), qX + Math.max(9, qW * (q / maxQ)) - 6, y + 35);
      ctx.restore();

      // cached prefixes: one bar per system prompt, height = blocks of that
      // prompt's prefix this replica currently holds
      const chY = y + 46, chH = 26, bw = Math.min(11, (cardW - 12) / P);
      for (let p = 0; p < P; p++) {
        const bx = x + 6 + p * bw;
        const held = s ? s.holds[i][p] : 0;
        ctx.save();
        ctx.fillStyle = rgbaToken('n14', 0.06);
        ctx.fillRect(bx, chY, bw - 1.5, chH);
        if (held) {
          const hh = chH * (held / PREFIX_BLOCKS);
          ctx.fillStyle = alphaOf(categorical(p), held === PREFIX_BLOCKS ? 0.95 : 0.5);
          ctx.fillRect(bx, chY + chH - hh, bw - 1.5, hh);
        }
        ctx.restore();
      }
    }
    r.label('per card: utilisation so far · queue depth (scaled to the run\'s deepest queue) · prefix blocks cached, one bar per system prompt',
      pad, cardTop + cardH + 14, { color: T.n9, font: '9.5px ui-monospace, monospace' });

    // ---- prompt popularity, draggable ------------------------------------
    r.label('prompt popularity — how the arrival stream picks a system prompt', pad, popY - 8, { color: T.n11, font: '11px ui-monospace, monospace' });
    const popX = pad, popW = Math.min(320, W * 0.42);
    const pbw = popW / P, maxProb = Math.max(...base.A.probs);
    for (let p = 0; p < P; p++) {
      const bx = popX + p * pbw, hgt = popH * (base.A.probs[p] / maxProb);
      ctx.save();
      ctx.fillStyle = alphaOf(categorical(p), 0.8);
      ctx.fillRect(bx, popY + popH - hgt, pbw - 2, hgt);
      ctx.restore();
      if (pbw > 22) r.label(String.fromCharCode(65 + p), bx + pbw / 2 - 3, popY + popH + 9, { color: T.n10, font: '8px ui-monospace, monospace' });
    }
    // the skew handle
    const skX = popX + popW * (st.skew / 2.5);
    ctx.save();
    ctx.strokeStyle = alphaOf(T.violet, 0.5); ctx.lineWidth = 1;
    ctx.beginPath(); ctx.moveTo(popX, popY + popH + 3); ctx.lineTo(popX + popW, popY + popH + 3); ctx.stroke();
    ctx.fillStyle = drag === 'skew' ? T.violet : alphaOf(T.violet, 0.9);
    ctx.beginPath(); ctx.moveTo(skX, popY + popH + 1); ctx.lineTo(skX - 5, popY + popH + 9); ctx.lineTo(skX + 5, popY + popH + 9); ctx.closePath(); ctx.fill();
    ctx.restore();
    r.label(`↔ drag ▲ — skew ${st.skew.toFixed(2)}`, popX + popW + 14, popY + 10, { color: T.violet, font: '10.5px ui-monospace, monospace' });
    r.label('left: many equally popular prompts · right: one dominant prompt', popX + popW + 14, popY + 24, { color: T.n10, font: '9.5px ui-monospace, monospace' });
    r.label(`top prompt takes ${pct(base.A.probs[0])} of arrivals`, popX + popW + 14, popY + popH + 12, { color: T.n10, font: '9.5px ui-monospace, monospace' });

    // ---- arrival lanes ----------------------------------------------------
    r.label('where each request was routed — one lane per replica, left to right in arrival order', pad, laneTop - 12, { color: T.n11, font: '11px ui-monospace, monospace' });
    const laneX = pad + 30, laneW = W - laneX - pad - 6;
    const dotStep = laneW / A, dotR = Math.max(2.2, Math.min(6, dotStep / 2 - 0.8));
    const dots = [], lanes = [];
    for (let i = 0; i < N; i++) {
      const ly = laneTop + i * laneH;
      lanes.push({ i, x: pad, y: ly, w: laneW + laneX - pad, h: laneH });
      ctx.save();
      ctx.fillStyle = i % 2 ? rgbaToken('n14', 0.035) : rgbaToken('n14', 0.015);
      ctx.fillRect(laneX, ly, laneW, laneH - 1);
      ctx.restore();
      r.label('R' + i, pad, ly + laneH / 2 + 3, { color: T.n10, font: '9px ui-monospace, monospace' });
    }
    if (s) {
      const guideX = laneX + (kNow + 0.5) * dotStep;
      ctx.save();
      ctx.strokeStyle = alphaOf(T.accent, 0.35); ctx.lineWidth = 1;
      ctx.beginPath(); ctx.moveTo(guideX, laneTop - 4); ctx.lineTo(guideX, laneBot + 4); ctx.stroke();
      ctx.restore();
    }
    for (let k = 0; k <= kNow; k++) {
      const rec = cur.recs[k];
      if (!rec) continue;
      const dx = laneX + (k + 0.5) * dotStep, dy = laneTop + rec.replica * laneH + (laneH - 1) / 2;
      dots.push({ k, x: dx, y: dy, rec });
      const frac = rec.hit / PREFIX_BLOCKS;
      ctx.save();
      ctx.beginPath(); ctx.arc(dx, dy, dotR, 0, Math.PI * 2);
      ctx.fillStyle = alphaOf(categorical(rec.prompt), 0.25 + 0.7 * frac);
      ctx.fill();
      ctx.strokeStyle = k === kNow ? T.n14 : rec.hit === 0 ? alphaOf(T.warn, 0.9) : alphaOf(categorical(rec.prompt), 0.95);
      ctx.lineWidth = k === kNow ? 2 : 1;
      ctx.stroke();
      ctx.restore();
    }
    r.label('fill = fraction of the shared prefix served from that replica\'s cache · orange ring = nothing reused, full recompute',
      laneX, laneBot + 13, { color: T.n9, font: '9.5px ui-monospace, monospace' });

    // ---- the trade-off chart ---------------------------------------------
    // Everything plotted here comes out of SWEEP_N + 1 complete re-simulations
    // of the SAME arrival stream, one per weight.
    const ttftRelOf = (p) => (rr.meanTtft > 0 ? 100 * p.ttft / rr.meanTtft : 0);
    let maxRel = 100;
    for (const p of base.sweep) maxRel = Math.max(maxRel, ttftRelOf(p));
    // The percent axis stretches to hold the TTFT curve, because the whole point
    // of the page is how far above round-robin the affinity end goes -- a fixed
    // axis flat-tops exactly the finding. It stops at 400%: past that the shape
    // below is what matters and the peak is stated in words instead.
    const yMax = Math.min(400, Math.max(150, Math.ceil(maxRel * 1.08 / 50) * 50));
    const clipped = maxRel > yMax + 0.5;
    const gstep = yMax <= 200 ? 50 : 100;
    const yOf = (v) => cy + ch * (1 - Math.min(yMax, Math.max(0, v)) / yMax);
    const xOf = (w) => cx + cw * w;
    ctx.save();
    ctx.strokeStyle = T.n5; ctx.lineWidth = 1;
    ctx.beginPath(); ctx.moveTo(cx, cy); ctx.lineTo(cx, cy + ch); ctx.lineTo(cx + cw, cy + ch); ctx.stroke();
    ctx.restore();
    for (let gv = 0; gv <= yMax + 0.5; gv += gstep) {
      const gy = yOf(gv);
      ctx.save();
      ctx.strokeStyle = gv === 100 ? alphaOf(T.n9, 0.55) : rgbaToken('n14', 0.07);
      ctx.setLineDash(gv === 100 ? [4, 3] : []);
      ctx.lineWidth = 1;
      ctx.beginPath(); ctx.moveTo(cx, gy); ctx.lineTo(cx + cw, gy); ctx.stroke();
      ctx.restore();
      r.label(gv + '%', cx - 5, gy + 3, { color: T.n9, font: '9px ui-monospace, monospace', align: 'right' });
    }
    const line = (fn, color, dash) => {
      ctx.save();
      ctx.strokeStyle = color; ctx.lineWidth = 2; ctx.setLineDash(dash || []);
      ctx.beginPath();
      base.sweep.forEach((pt, i) => { const X = xOf(pt.w), Y = yOf(fn(pt)); if (i) ctx.lineTo(X, Y); else ctx.moveTo(X, Y); });
      ctx.stroke();
      ctx.restore();
    };
    line((p) => 100 * p.hit, T.ok);
    line((p) => 100 * p.bal, T.violet);
    line(ttftRelOf, T.accent, [5, 3]);

    // best weight found in the sweep (lowest mean TTFT)
    let bestI = 0;
    for (let i = 1; i < base.sweep.length; i++) if (base.sweep[i].ttft < base.sweep[bestI].ttft) bestI = i;
    const bestPt = base.sweep[bestI];
    ctx.save();
    ctx.fillStyle = T.accent;
    ctx.font = '13px ui-monospace, monospace'; ctx.textAlign = 'center'; ctx.textBaseline = 'middle';
    ctx.fillText('★', xOf(bestPt.w), yOf(ttftRelOf(bestPt)));
    ctx.restore();

    // the draggable weight marker
    const mx = xOf(st.w);
    ctx.save();
    ctx.strokeStyle = drag === 'w' ? T.n14 : alphaOf(T.n11, 0.8);
    ctx.lineWidth = drag === 'w' ? 2 : 1.4;
    ctx.beginPath(); ctx.moveTo(mx, cy - 4); ctx.lineTo(mx, cy + ch + 4); ctx.stroke();
    ctx.fillStyle = drag === 'w' ? T.n14 : T.n12;
    ctx.beginPath();
    ctx.moveTo(mx, cy - 12); ctx.lineTo(mx + 6, cy - 6); ctx.lineTo(mx, cy); ctx.lineTo(mx - 6, cy - 6);
    ctx.closePath(); ctx.fill();
    ctx.restore();

    r.label('0.0  pure load balance', cx, cy + ch + 14, { color: T.n10, font: '9.5px ui-monospace, monospace' });
    r.label('pure cache affinity  1.0', cx + cw, cy + ch + 14, { color: T.n10, font: '9.5px ui-monospace, monospace', align: 'right' });
    r.label('↔ drag ◆', mx, cy - 15, { color: drag === 'w' ? T.n14 : T.n11, font: '9.5px ui-monospace, monospace', align: 'center' });
    if (clipped) r.label(`(TTFT peaks at ${f1(maxRel)}% — above the top of this axis)`, cx + cw, cy + 11, { color: T.bad, font: '9px ui-monospace, monospace', align: 'right' });

    // legend: one horizontal strip above the plot, so it never competes with the
    // plot for width (stacked at the right it was clipped at every replica count)
    const leg = [
      ['cache hit rate', T.ok, false],
      ['load balance', T.violet, false],
      ['mean TTFT vs round-robin (lower is better, 100% = parity)', T.accent, true],
    ];
    ctx.save();
    ctx.font = '9px ui-monospace, monospace';
    let lx = cx;
    for (const e of leg) {
      ctx.save();
      ctx.strokeStyle = e[1]; ctx.lineWidth = 2; ctx.setLineDash(e[2] ? [5, 3] : []);
      ctx.beginPath(); ctx.moveTo(lx, cy - 32); ctx.lineTo(lx + 15, cy - 32); ctx.stroke();
      ctx.restore();
      ctx.fillStyle = T.n11; ctx.textAlign = 'left'; ctx.textBaseline = 'middle';
      ctx.fillText(e[0], lx + 19, cy - 31);
      lx += 19 + ctx.measureText(e[0]).width + 16;
    }
    ctx.fillStyle = T.accent;
    ctx.fillText('★ best weight in the sweep', lx, cy - 31);
    ctx.restore();
    r.label(`router weight — the same ${A}-request arrival stream, re-simulated at ${SWEEP_N + 1} weights`,
      cx + cw / 2, cy + ch + 14, { color: T.n10, font: '9.5px ui-monospace, monospace', align: 'center' });

    geom = { cards, lanes, dots, cx, cy, cw, ch, popX, popY, popW, popH, laneX, laneW, laneTop, laneH, dotR };

    // ---- hover-to-inspect -------------------------------------------------
    if (page.pointer.over && !drag) {
      const p = page.pointer;
      let tip = null;
      for (const d of dots) {
        if (Math.abs(p.x - d.x) <= Math.max(4, dotR + 2) && Math.abs(p.y - d.y) <= Math.max(5, laneH / 2)) {
          const rec = d.rec;
          const lines = [
            `request ${rec.k} · ${promptName(rec.prompt)} · arrived at ${f1(rec.t)} ms`,
            `→ replica ${rec.replica}, because it scored highest:`,
          ];
          for (let i = 0; i < N; i++) {
            const aff = rec.match[i] / PREFIX_BLOCKS;
            const mb = Math.max(...rec.backlog);
            const load = mb > 0 ? rec.backlog[i] / mb : 0;
            lines.push(`  R${i}: ${st.w.toFixed(2)}×${aff.toFixed(2)} (${rec.match[i]}/${PREFIX_BLOCKS} blocks) − ${(1 - st.w).toFixed(2)}×${load.toFixed(2)} (${f1(rec.backlog[i])} ms backlog) = ${rec.scores[i].toFixed(3)}${i === rec.replica ? '   ← chosen' : ''}`);
          }
          lines.push(`reused ${rec.hit} block(s), computed ${rec.comp}: prefill ${f1(rec.prefill)} ms`);
          lines.push(`waited ${f1(rec.wait)} ms in queue → TTFT ${f1(rec.ttft)} ms`);
          tip = lines.join('\n');
          break;
        }
      }
      if (!tip) for (const c of cards) {
        if (p.x >= c.x && p.x <= c.x + c.w && p.y >= c.y && p.y <= c.y + c.h) {
          const i = c.i;
          const held = s ? s.holds[i] : new Array(P).fill(0);
          const names = held.map((n, q) => (n ? `${promptName(q)} ${n}/${PREFIX_BLOCKS}` : null)).filter(Boolean);
          tip = [
            `replica ${i}`,
            `queue depth now: ${s ? s.qd[i] : 0} request(s) · backlog ${s ? f1(s.backlog[i]) : '0.0'} ms`,
            `utilisation so far: ${s ? pct(s.util[i]) : '–'} · served ${cur.recs.slice(0, kNow + 1).filter((x) => x.replica === i).length} of ${kNow + 1}`,
            `cache: ${s ? s.size[i] : 0}/${st.capacity} blocks (${fmtKB(s ? s.size[i] : 0)} of KV)`,
            `holds prefixes: ${names.length ? names.join(', ') : '(none yet)'}`,
            `over the whole run this replica ends at ${pct(cur.util[i])} utilisation, ${cur.served[i]} request(s)`,
          ].join('\n');
          break;
        }
      }
      if (!tip && p.x >= cx - 10 && p.x <= cx + cw + 10 && p.y >= cy - 14 && p.y <= cy + ch + 10) {
        const wv = Math.max(0, Math.min(1, (p.x - cx) / cw));
        const pt = base.sweep[Math.round(wv * SWEEP_N)];
        tip = [
          `router weight ${pt.w.toFixed(2)}  (${pt.w === 0 ? 'pure load balance' : pt.w === 1 ? 'pure cache affinity' : `${(100 * pt.w).toFixed(0)}% affinity / ${(100 * (1 - pt.w)).toFixed(0)}% balance`})`,
          `cache hit rate      ${pct(pt.hit)}   of ${TOTAL_BLOCKS * A} prompt blocks`,
          `load balance        ${pct(pt.bal)}   (least-busy replica ÷ busiest)`,
          `mean TTFT           ${f1(pt.ttft)} ms = ${f1(ttftRelOf(pt))}% of round-robin (lower is better)`,
          `tail (p95) TTFT     ${f1(pt.p95)} ms = ${f1(rr.p95Ttft > 0 ? 100 * pt.p95 / rr.p95Ttft : 0)}% of round-robin`,
          `throughput          ${f1(pt.thr)} req/s = ${f1(rr.throughput > 0 ? 100 * pt.thr / rr.throughput : 0)}% of round-robin (higher is better)`,
          '↔ click and drag to move the router to this weight',
        ].join('\n');
      }
      if (!tip && p.x >= popX - 6 && p.x <= popX + popW + 6 && p.y >= popY - 6 && p.y <= popY + popH + 12) {
        tip = [
          `prompt popularity skew ${st.skew.toFixed(2)}  —  P(rank) ∝ 1 / (rank+1)^skew`,
          ...base.A.probs.map((v, q) => `  ${promptName(q)}: ${pct(v)} of arrivals`),
          '↔ drag to reshape. Affinity only pays when some prefix is popular enough',
          '   to be worth following — and pins hardest when ONE prompt dominates.',
        ].join('\n');
      }
      if (tip) page.setTip(tip);
    }

    // ---- readout ----------------------------------------------------------
    const ttftRel = rr.meanTtft > 0 ? cur.meanTtft / rr.meanTtft : 1;
    const p95Rel = rr.p95Ttft > 0 ? cur.p95Ttft / rr.p95Ttft : 1;
    const thrRel = rr.throughput > 0 ? cur.throughput / rr.throughput : 1;
    const idle = cur.served.filter((v) => v === 0).length;
    page.probe = {
      idleReplicas: idle, ttftRel, hitRate: cur.hitRate,
      hitAt0: base.sweep[0].hit, hitAt1: base.sweep[SWEEP_N].hit, bestW: bestPt.w,
    };

    const utilTxt = cur.util.map((u, i) => `R${i} ${(100 * u).toFixed(0)}%`).join(' · ');
    let o = s
      ? `request ${s.k} (${promptName(s.prompt)}) → replica ${s.replica}: longest matched prefix ${s.hit}/${PREFIX_BLOCKS} blocks, so ${s.comp} of ${TOTAL_BLOCKS} blocks are prefilled (${f1(s.prefill)} ms) after ${f1(s.wait)} ms of queueing → TTFT ${f1(s.ttft)} ms.\n`
      : 'nothing admitted yet — every replica cache is empty, so the first request to each replica matches nothing and prefills its whole prompt.\n';
    o += `whole run at weight ${st.w.toFixed(2)}, ${N} replicas, ${A} requests, ${P} distinct system prompts, skew ${st.skew.toFixed(2)}, ${st.capacity}-block cache per replica:\n`;
    o += `  cache hit rate ${pct(cur.hitRate)} of ${TOTAL_BLOCKS * A} prompt blocks (round-robin: ${pct(rr.hitRate)})\n`;
    o += `  mean TTFT ${f1(cur.meanTtft)} ms = ${f1(100 * ttftRel)}% of round-robin's ${f1(rr.meanTtft)} ms (lower is better; 100% = parity)\n`;
    o += `  tail p95 TTFT ${f1(cur.p95Ttft)} ms = ${f1(100 * p95Rel)}% of round-robin's ${f1(rr.p95Ttft)} ms (lower is better)\n`;
    o += `  throughput ${f1(cur.throughput)} req/s = ${f1(100 * thrRel)}% of round-robin's ${f1(rr.throughput)} req/s (higher is better)\n`;
    o += `  per-replica utilisation over the run: ${utilTxt} — balance ${pct(cur.balance)}${idle ? `, and ${idle} replica(s) served nothing at all` : ''}\n`;
    o += `The sweep re-runs this same arrival stream at ${SWEEP_N + 1} weights. Best mean TTFT is at weight ${bestPt.w.toFixed(2)} (${f1(ttftRelOf(bestPt))}% of round-robin); pure balance gives ${f1(ttftRelOf(base.sweep[0]))}% and pure affinity ${f1(ttftRelOf(base.sweep[SWEEP_N]))}%. `;
    o += 'Reuse and balance pull against each other — the green curve broadly climbs as the weight does, the violet one gives way, and the dashed latency curve bottoms out near where they cross. Drag the ▲ to the far left (many equally popular prompts) and the green curve flattens: there is no popular prefix to follow, so affinity buys nothing and only costs balance.';
    page.setReadout(o);
  },
}).then((page) => {
  const q = new URLSearchParams(location.search);
  const t = page.controls._transport;
  // Restore every control from the query string, so one URL replays exactly one
  // arrival stream at exactly one router setting -- ?w= and ?skew= are the
  // headless stand-ins for dragging the ◆ and ▲ handles, since a screenshot has
  // no pointer. The framework mirrors state OUT to the URL on every change; this
  // is the matching read-back on load.
  for (const k of ['replicas', 'prompts', 'arrivals', 'seed', 'w', 'skew', 'capacity']) {
    if (!q.has(k)) continue;
    const v = parseFloat(q.get(k));
    if (Number.isFinite(v)) page.controls.set(k, v, { rebuild: true, silent: true });
  }
  // ?hover=x,y fakes the cursor (headless stand-in for a real hover).
  if (q.has('hover')) {
    const [hx, hy] = q.get('hover').split(',').map(Number);
    page.pointer.x = hx; page.pointer.y = hy; page.pointer.over = true;
  }
  if (t) t.rebuild();   // the restored controls change the step list; seek must clamp against the new one
  // Deterministic frame for capture: any explicit hook pauses autoplay first.
  if (q.has('step') || q.has('hover')) { if (t) t.pause(); }
  if (q.has('step') && t) t.seek(parseInt(q.get('step'), 10));
  if (q.get('play') === '1' && t) t.play();
  page.redraw();
});
