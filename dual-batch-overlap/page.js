// dual-batch-overlap concept page -- hiding an expert all-to-all behind another
// microbatch's arithmetic.
//
// In a mixture-of-experts model served with EXPERT PARALLELISM, each layer has
// to route: every token is sent to whichever device holds the expert it picked,
// and the results come back. That is two all-to-all collectives per layer, and
// while they are in flight the arithmetic units have nothing to do. The
// companion page `parallelism` owns WHAT those collectives put on the wire and
// why the volume is set by the router; this page is about HIDING that cost in
// TIME, and never re-derives the byte counts.
//
// The trick: split the batch in two and interleave the halves. While microbatch
// A is in its all-to-all, microbatch B is computing; when B reaches its
// all-to-all, A is computing again. With balanced halves the communication
// disappears into the arithmetic entirely. With unbalanced halves a visible
// bubble remains -- and that bubble is what this page draws.
//
// PUBLIC SOURCES for the mechanism:
//   - DeepSeek-V3 Technical Report, https://arxiv.org/abs/2412.19437 -- the
//     DualPipe schedule and its "computation-communication overlap" section,
//     which overlaps the dispatch/combine all-to-all of one chunk with the
//     compute of another so the routing traffic is "fully hidden".
//   - MegaScale, https://arxiv.org/abs/2402.15627 -- overlapping collectives
//     with computation at scale, and what stops it from being free.
//   - Lina, https://arxiv.org/abs/2210.17223 -- MoE serving work that
//     decomposes and pipelines the all-to-all against expert compute.
//   - Tutel, https://arxiv.org/abs/2206.03382 -- adaptive MoE dispatch,
//     including splitting the all-to-all so it can be pipelined.
//
// THE PAGE IS BUILT, NOT DRAWN. `schedule()` below is a list scheduler over two
// serial resources (one compute lane, one communication lane) with real
// dependency edges between the phases of each microbatch. The step time, the
// overlap fraction and the bubble are all READ OFF the resulting schedule -- no
// closed-form formula produces them, so a configuration where splitting HURTS
// falls out on its own rather than being asserted.
//
// Interactive per the shared render framework's contract:
//  - TRANSPORT: play / step / scrub the schedule; phases fill in on both lanes.
//    Autoplays and loops.
//  - DIRECT MANIPULATION: drag any COMPUTE block to stretch the compute cost,
//    any COMM block to stretch the communication cost, and drag across the
//    microbatch-count chart to re-split the batch. The timeline re-packs under
//    your hand and the bubble grows or vanishes.
//  - HOVER any phase block -> its duration, what it waits on, and what it
//    overlaps on the other lane.
//  - RESIZE the problem: number of layers, number of microbatches.
//
// The millisecond figures come from a deliberately simple illustrative cost
// model -- a per-microbatch share of the work plus a fixed per-invocation
// overhead on each lane -- chosen so the SHAPE of the trade is readable. They
// illustrate the mechanism; they are not a measurement of any particular model,
// interconnect or accelerator.
import { mount } from '../framework/layout.js';
import { categorical } from '../framework/render.js';
import { T, alphaOf, rgbaToken, inkOn } from '../framework/theme.js';

// --- illustrative cost model -------------------------------------------------
// Splitting the batch does NOT split the cost cleanly, and that is the whole
// second half of the lesson:
//   * a compute phase pays a fixed launch + weight-read cost every time it is
//     invoked, so half the tokens is MORE than half the time;
//   * an all-to-all pays a fixed latency every time it is issued, so half the
//     payload is MORE than half the transfer.
// Split into M microbatches and you pay both fixed costs M times.
const LAUNCH_MS = 0.26;   // fixed cost of one compute phase invocation
const LAT_MS = 0.34;      // fixed cost of issuing one all-to-all
const ATTN_SHARE = 0.4;   // share of a layer's compute before the dispatch

const MB = ['A', 'B', 'C', 'D'];
const M_CHOICES = [1, 2, 4];
const mbColor = (i) => (i === 0 ? T.accent : alphaOf(categorical(i), 1));
const mbFill = (i, a) => (i === 0 ? alphaOf(T.accent, a) : alphaOf(categorical(i), a));

const clamp = (v, lo, hi) => (v < lo ? lo : v > hi ? hi : v);
const ms = (v) => `${v.toFixed(2)} ms`;
// Step time is a LOWER-is-better quantity: >100% of the baseline is WORSE.
const pctOf = (v, b) => (b > 0 ? Math.round((v / b) * 1000) / 10 : 100);
const dirWord = (p) => (p > 100 ? '▲ worse' : p < 100 ? '▼ better' : '= parity');
const dirColor = (p) => (p > 100.05 ? T.bad : p < 99.95 ? T.ok : T.n12);

const PHASES = [
  { key: 'attn', lane: 'compute', name: 'attention + router' },
  { key: 'dispatch', lane: 'comm', name: 'all-to-all dispatch' },
  { key: 'expert', lane: 'compute', name: 'expert FFN' },
  { key: 'combine', lane: 'comm', name: 'all-to-all combine' },
];

// --- the simulator -----------------------------------------------------------
// Two serial resources. Every task needs its predecessor (the previous phase of
// the same microbatch, chaining across layers) AND its lane. A task starts at
// max(predecessor end, lane free). Ready tasks are taken earliest-start-first,
// ties broken by program order, which is exactly what an interleaved
// dual-batch schedule does.
function schedule(computeMs, commMs, M, L) {
  const dur = {
    attn: (computeMs * ATTN_SHARE) / M + LAUNCH_MS,
    expert: (computeMs * (1 - ATTN_SHARE)) / M + LAUNCH_MS,
    dispatch: (commMs * 0.5) / M + LAT_MS,
    combine: (commMs * 0.5) / M + LAT_MS,
  };

  const tasks = [];
  for (let l = 0; l < L; l++) {
    for (let m = 0; m < M; m++) {
      for (let p = 0; p < PHASES.length; p++) {
        const ph = PHASES[p];
        tasks.push({
          id: tasks.length, l, m, p, key: ph.key, lane: ph.lane, name: ph.name,
          dur: dur[ph.key], prev: null, t0: null, t1: null, scheduled: false,
        });
      }
    }
  }
  // dependency chain: phase k of (l,m) follows phase k-1, and layer l follows
  // layer l-1 for the same microbatch.
  const at = (l, m, p) => tasks[(l * M + m) * PHASES.length + p];
  for (let l = 0; l < L; l++) {
    for (let m = 0; m < M; m++) {
      for (let p = 0; p < PHASES.length; p++) {
        const t = at(l, m, p);
        t.prev = p > 0 ? at(l, m, p - 1) : (l > 0 ? at(l - 1, m, PHASES.length - 1) : null);
      }
    }
  }

  const laneFree = { compute: 0, comm: 0 };
  const order = [];
  let left = tasks.length;
  while (left > 0) {
    let best = null, bestEst = Infinity;
    for (const t of tasks) {
      if (t.scheduled) continue;
      if (t.prev && !t.prev.scheduled) continue;
      const est = Math.max(t.prev ? t.prev.t1 : 0, laneFree[t.lane]);
      if (est < bestEst - 1e-9) { best = t; bestEst = est; }
    }
    best.t0 = bestEst;
    best.t1 = bestEst + best.dur;
    best.waitOn = (best.prev && best.prev.t1 >= bestEst - 1e-9)
      ? { kind: 'dep', task: best.prev }
      : (laneFree[best.lane] >= bestEst - 1e-9 && bestEst > 1e-9
        ? { kind: 'lane', task: order.filter((q) => q.lane === best.lane).pop() || null }
        : { kind: 'start', task: null });
    laneFree[best.lane] = best.t1;
    best.scheduled = true;
    best.order = order.length;
    order.push(best);
    left--;
  }

  const total = Math.max(laneFree.compute, laneFree.comm);
  const computeSum = tasks.filter((t) => t.lane === 'compute').reduce((a, t) => a + t.dur, 0);
  const commSum = tasks.filter((t) => t.lane === 'comm').reduce((a, t) => a + t.dur, 0);

  // What each comm task actually got hidden behind: the part of its interval
  // covered by a compute task. Read off the schedule, never assumed.
  let hidden = 0;
  for (const c of tasks) {
    if (c.lane !== 'comm') continue;
    let cov = 0;
    for (const k of tasks) {
      if (k.lane !== 'compute') continue;
      cov += Math.max(0, Math.min(c.t1, k.t1) - Math.max(c.t0, k.t0));
    }
    c.hidden = Math.min(cov, c.dur);
    hidden += c.hidden;
  }
  const overlapFrac = commSum > 0 ? hidden / commSum : 1;
  // The bubble is compute-lane idle time inside the step: the arithmetic units
  // sitting there with nothing to do.
  const bubble = Math.max(0, total - computeSum);
  const floor = Math.max(computeSum, commSum);   // perfect-overlap floor at this M

  return {
    tasks, order, total, computeSum, commSum, hidden, overlapFrac, bubble, floor,
    M, L, computeMs, commMs, dur,
  };
}

let cur = null;      // schedule on screen
let base = null;     // named baseline: no overlap, ONE batch (M = 1)
let geom = null;     // hit-test rects captured in draw()
let drag = null;     // 'compute' | 'comm' | 'M' while a handle is grabbed

function buildData(st) {
  const C = +st.C, V = +st.V, M = st.M | 0, L = st.L | 0;
  const s = schedule(C, V, M, L);
  base = schedule(C, V, 1, L);
  // Sweep the split so "past a point, splitting makes it WORSE" is visible
  // rather than asserted -- every point is its own full simulation.
  s.sweep = M_CHOICES.map((m) => ({ m, sched: schedule(C, V, m, L) }));
  cur = s;
  return s.order.map((t) => ({
    ...t,
    label: `step ${t.order + 1}/${s.order.length}: microbatch ${MB[t.m]} · layer ${t.l} · ${t.name} `
      + `(${t.lane === 'comm' ? 'communication' : 'compute'} lane, ${ms(t.dur)})`,
  }));
}

function setNum(page, key, v, lo, hi, step) {
  const q = clamp(Math.round(v / step) * step, lo, hi);
  if (Math.abs(q - (+page.state[key])) < step / 2) return;
  page.controls.set(key, q, { rebuild: true });
}

mount({
  mount: 'body',
  title: 'dual-batch-overlap — hiding an expert all-to-all behind another microbatch',
  blurb: 'A mixture-of-experts layer served with expert parallelism has to ROUTE: tokens out to the devices holding their experts, results back. Two all-to-alls per layer, and while they are in flight the arithmetic units idle. Split the batch in two and interleave the halves: while microbatch A is in its all-to-all, microbatch B computes, and vice versa. The two lanes below are packed by a real list scheduler — one serial compute resource, one serial communication resource, with dependency edges between each microbatch’s phases — so the step time and the bubble are CONSEQUENCES of the schedule, not a formula. DRAG a compute block to stretch the compute cost, a comm block to stretch the communication cost, or drag across the split chart to change the number of microbatches. Overlap only pays when a microbatch’s compute is at least as long as the communication it has to hide — and each split also halves that compute while paying the fixed per-invocation costs again, so past a point splitting makes it WORSE. Find that point.',
  prefer: 'canvas2d',
  aspect: '2 / 1',
  autoplay: true,
  compare: { key: 'M', a: 1, b: 2, labelA: 'one batch — communication is dead time', labelB: 'two microbatches — the all-to-all hides behind arithmetic', rebuild: true },
  challenges: [
    {
      goal: 'Hide the whole routing all-to-all: get the overlap fraction to 100%.',
      hint: 'two microbatches, and enough compute per microbatch to cover the transfer — raise the compute cost or lower the communication cost.',
      check: (api) => ({ solved: (api.probe.overlap ?? 0) > 0.999, detail: `overlap ${((api.probe.overlap ?? 0) * 100).toFixed(1)}% of communication hidden (need 100%)` }),
    },
    {
      goal: 'Now break it: find a setting where FOUR microbatches are slower than not splitting at all.',
      hint: 'shrink the compute cost. Each split halves a microbatch’s arithmetic but pays the fixed launch and link latency again, so the small kernels stop covering anything and the overhead is all that is left.',
      check: (api) => ({ solved: (api.probe.m4 ?? 0) > (api.probe.baseTotal ?? 1e9) + 1e-6, detail: `4 microbatches = ${pctOf(api.probe.m4 ?? 0, api.probe.baseTotal ?? 1)}% of the no-overlap one-batch baseline (lower is better; need > 100%)` }),
    },
  ],
  controls: (c, page) => {
    c.slider('C', { label: 'compute per layer, whole batch (ms)', min: 0.5, max: 16, step: 0.5, value: 6, rebuild: true });
    c.slider('V', { label: 'all-to-all per layer, whole batch (ms)', min: 0, max: 12, step: 0.5, value: 4, rebuild: true });
    c.select('M', { label: 'microbatches (split the batch)', options: M_CHOICES.map((m) => ({ value: String(m), label: m === 1 ? '1 — no overlap possible' : `${m} microbatches` })), value: '2', rebuild: true });
    c.stepper('L', { label: 'layers', min: 1, max: 3, value: 2 });
    c.transport({ compute: () => buildData(page.state), speed: 3.2, loop: true });
  },

  // DIRECT MANIPULATION: grab a block on either lane and stretch it, or drag
  // across the split chart. Everything re-packs under the hand.
  onPointer: (page, ev) => {
    if (!geom) return;
    const g = geom;
    const pickM = (x) => {
      let best = M_CHOICES[0], bd = Infinity;
      for (const b of g.mbars) { const d = Math.abs(x - (b.x + b.w / 2)); if (d < bd) { bd = d; best = b.m; } }
      return best;
    };
    if (ev.type === 'down') {
      if (ev.y >= g.mchart.y && ev.y <= g.mchart.y + g.mchart.h && ev.x >= g.mchart.x && ev.x <= g.mchart.x + g.mchart.w) {
        drag = 'M';
        page.controls.set('M', String(pickM(ev.x)), { rebuild: true });
        return;
      }
      const hit = g.rects.find((r) => ev.x >= r.x && ev.x <= r.x + r.w && ev.y >= r.y && ev.y <= r.y + r.h);
      drag = hit ? hit.t.lane : null;
    } else if (ev.type === 'up' || ev.type === 'leave') {
      drag = null;
    } else if (ev.type === 'move' && drag && page.pointer.down) {
      if (drag === 'M') { page.controls.set('M', String(pickM(ev.x)), { rebuild: true }); return; }
      // dx in canvas px -> ms on the shared time axis, scaled by how many
      // invocations of this lane are on screen (stretching ONE block stretches
      // the whole lane's cost).
      const perMs = g.perMs || 1;
      const n = drag === 'compute' ? 2 * cur.M * cur.L : 2 * cur.M * cur.L;
      const d = (ev.dx / perMs) * n;
      if (drag === 'compute') setNum(page, 'C', +page.state.C + d, 0.5, 16, 0.5);
      else setNum(page, 'V', +page.state.V + d, 0, 12, 0.5);
    }
  },

  draw: (page) => {
    const r = page.renderer, ctx = page.ctx;
    if (!cur) return;
    r.clear(T.n0);
    const W = page.W, H = page.H, pad = 16;
    const { tasks, order, M, L } = cur;
    const idx = page.controls._transport ? page.controls._transport.index : order.length - 1;
    const m2 = cur.sweep.find((s) => s.m === 2), m4 = cur.sweep.find((s) => s.m === 4);
    page.probe = {
      overlap: cur.overlapFrac, bubble: cur.bubble, total: cur.total,
      m2: m2 ? m2.sched.total : 0, m4: m4 ? m4.sched.total : 0, M, baseTotal: base.total,
    };

    // ---- geometry ----------------------------------------------------------
    const x0 = pad + 62, xR = Math.max(x0 + 140, W - 214);
    const tMax = Math.max(cur.total, base.total) * 1.02;
    const perMs = (xR - x0) / tMax;
    const xOf = (t) => x0 + t * perMs;
    const tOf = (x) => (x - x0) / perMs;

    const ghostY = 52, ghostH = 13;
    const laneH = clamp(H * 0.09, 22, 36);
    const compY = ghostY + ghostH + 30;
    const commY = compY + laneH + 12;
    const laneBot = commY + laneH;
    const laneOf = (lane) => (lane === 'compute' ? compY : commY);

    // ---- the named baseline: no overlap, ONE batch --------------------------
    ctx.save(); ctx.beginPath(); ctx.rect(x0, ghostY - 20, xR - x0, 16); ctx.clip();
    r.label('BASELINE — no overlap, one batch (M = 1): compute idles through both all-to-alls',
      x0, ghostY - 8, { color: T.n11, font: '10px ui-monospace, monospace' });
    ctx.restore();
    for (const t of base.tasks) {
      const bx = xOf(t.t0), bw = Math.max(1, xOf(t.t1) - xOf(t.t0) - 0.6);
      ctx.fillStyle = t.lane === 'comm' ? alphaOf(T.bad, 0.34) : rgbaToken('n14', 0.16);
      ctx.fillRect(bx, ghostY, bw, ghostH);
    }
    ctx.strokeStyle = rgbaToken('n14', 0.25); ctx.lineWidth = 1;
    ctx.strokeRect(x0 + 0.5, ghostY + 0.5, xOf(base.total) - x0 - 1, ghostH - 1);
    r.label(ms(base.total), xOf(base.total) + 6, ghostY + ghostH - 2, { color: T.n11, font: '10px ui-monospace, monospace' });

    // ---- the two lanes -----------------------------------------------------
    const rects = [];
    for (const lane of ['compute', 'comm']) {
      const ly = laneOf(lane);
      ctx.fillStyle = rgbaToken('n14', 0.05);
      ctx.fillRect(x0, ly, xR - x0, laneH);
      r.label(lane === 'compute' ? 'COMPUTE' : 'COMM', x0 - 10, ly + laneH / 2 - 3, { color: T.n12, font: '10px ui-monospace, monospace', align: 'right' });
      r.label(lane === 'compute' ? 'arithmetic' : 'all-to-all', x0 - 10, ly + laneH / 2 + 9, { color: T.n10, font: '9px ui-monospace, monospace', align: 'right' });
    }

    // compute-lane idle (the bubble) drawn as hatching on the compute lane
    const cTasks = tasks.filter((t) => t.lane === 'compute').sort((a, b) => a.t0 - b.t0);
    let prevEnd = 0;
    const gaps = [];
    for (const t of cTasks) { if (t.t0 > prevEnd + 1e-9) gaps.push([prevEnd, t.t0]); prevEnd = Math.max(prevEnd, t.t1); }
    if (cur.total > prevEnd + 1e-9) gaps.push([prevEnd, cur.total]);
    for (const [a, b] of gaps) {
      const gx = xOf(a), gw = xOf(b) - xOf(a);
      ctx.fillStyle = alphaOf(T.bad, 0.18);
      ctx.fillRect(gx, compY, gw, laneH);
      // clip FIRST, then build the hatch path -- a beginPath() after the path
      // is built silently discards it (and strokes the clip rect instead).
      ctx.save();
      ctx.beginPath(); ctx.rect(gx, compY, gw, laneH); ctx.clip();
      ctx.strokeStyle = alphaOf(T.bad, 0.5); ctx.lineWidth = 1;
      ctx.beginPath();
      for (let x = gx - laneH; x < gx + gw; x += 6) { ctx.moveTo(x, compY + laneH); ctx.lineTo(x + laneH, compY); }
      ctx.stroke();
      ctx.restore();
      if (gw > 42) r.label('bubble', gx + gw / 2, compY + laneH / 2 + 3, { color: T.bad, font: '10px ui-monospace, monospace', align: 'center' });
    }

    for (const t of order) {
      const ly = laneOf(t.lane);
      const bx = xOf(t.t0), bw = Math.max(1.5, xOf(t.t1) - xOf(t.t0) - 1);
      const past = t.order <= idx || idx < 0;
      const a = past ? 1 : 0.16;
      rects.push({ t, x: bx, y: ly, w: bw, h: laneH });
      ctx.fillStyle = mbFill(t.m, (t.lane === 'comm' ? 0.42 : 0.82) * a);
      ctx.fillRect(bx, ly, bw, laneH);
      if (t.lane === 'comm') {
        // stripes: the comm lane is a transfer, not arithmetic
        ctx.save(); ctx.beginPath(); ctx.rect(bx, ly, bw, laneH); ctx.clip();
        ctx.strokeStyle = mbFill(t.m, 0.85 * a); ctx.lineWidth = 1.2;
        ctx.beginPath();
        for (let x = bx - laneH; x < bx + bw; x += 5) { ctx.moveTo(x, ly + laneH); ctx.lineTo(x + laneH, ly); }
        ctx.stroke(); ctx.restore();
      }
      ctx.strokeStyle = mbFill(t.m, a); ctx.lineWidth = 1;
      ctx.strokeRect(bx + 0.5, ly + 0.5, bw - 1, laneH - 1);
      if (bw > 30) {
        ctx.save(); ctx.globalAlpha = a;
        r.label(`${MB[t.m]}${t.l}`, bx + bw / 2, ly + laneH / 2 - 2, { color: t.lane === 'comm' ? T.n13 : inkOn(mbColor(t.m)), font: '10px ui-monospace, monospace', align: 'center' });
        if (bw > 62) r.label(t.key, bx + bw / 2, ly + laneH / 2 + 10, { color: t.lane === 'comm' ? T.n11 : inkOn(mbColor(t.m)), font: '8.5px ui-monospace, monospace', align: 'center' });
        ctx.restore();
      }
      if (t.order === idx) {
        ctx.strokeStyle = T.n14; ctx.lineWidth = 1.6;
        ctx.strokeRect(bx - 1.5, ly - 1.5, bw + 3, laneH + 3);
      }
    }

    // step-time marker
    ctx.save();
    ctx.strokeStyle = T.n12; ctx.lineWidth = 1.2; ctx.setLineDash([4, 3]);
    ctx.beginPath(); ctx.moveTo(xOf(cur.total), ghostY - 4); ctx.lineTo(xOf(cur.total), laneBot + 8); ctx.stroke();
    ctx.setLineDash([]); ctx.restore();
    r.label(`step ${ms(cur.total)}`, xOf(cur.total) + 5, laneBot + 10, { color: T.n12, font: '10px ui-monospace, monospace' });
    r.label('time →', xR - 34, laneBot + 24, { color: T.n10, font: '10px ui-monospace, monospace' });

    // one-line explanation of the current regime, straight off the schedule
    const perMbCompute = cur.dur.attn + cur.dur.expert;
    const perMbComm = cur.dur.dispatch + cur.dur.combine;
    const line = M === 1
      ? `one batch: nothing can run during the all-to-all — ${ms(cur.commSum)} of routing is dead time`
      : perMbCompute >= perMbComm
        ? `per microbatch/layer: compute ${ms(perMbCompute)} ≥ comm ${ms(perMbComm)} — hides behind a sibling`
        : `per microbatch/layer: compute ${ms(perMbCompute)} < comm ${ms(perMbComm)} — too short to cover it`;
    // clipped to the lane column so a long sentence can never run under the
    // readout cards on the right
    ctx.save(); ctx.beginPath(); ctx.rect(x0, laneBot + 30, xR - x0, 16); ctx.clip();
    r.label(line, x0, laneBot + 42, { color: M === 1 ? T.bad : (perMbCompute >= perMbComm ? T.ok : T.warn), font: '10px ui-monospace, monospace' });
    ctx.restore();

    // ---- split chart: every M is its own full simulation --------------------
    const mcY = laneBot + 74;
    const mcH = clamp(H - mcY - 30, 46, 152);
    const mcW = xR - x0;
    r.label('split the batch — each bar is a complete re-simulation at that microbatch count (drag across)', x0, mcY - 8, { color: T.n12, font: '10px ui-monospace, monospace' });
    ctx.fillStyle = rgbaToken('n14', 0.04); ctx.fillRect(x0, mcY, mcW, mcH);
    const mMax = Math.max(base.total, ...cur.sweep.map((s) => s.sched.total)) * 1.12;
    const bw2 = Math.min(96, (mcW - 24) / cur.sweep.length - 12);
    const mbars = [];
    cur.sweep.forEach((s, i) => {
      const bx = x0 + 16 + i * (bw2 + 14);
      const h = (s.sched.total / mMax) * (mcH - 22);
      const y = mcY + mcH - 6 - h;
      const on = s.m === M;
      mbars.push({ m: s.m, x: bx, w: bw2 });
      ctx.fillStyle = on ? alphaOf(T.accent, 0.75) : rgbaToken('n14', 0.18);
      ctx.fillRect(bx, y, bw2, h);
      // the part of the bar that is bubble
      const bh = (s.sched.bubble / mMax) * (mcH - 22);
      ctx.fillStyle = alphaOf(T.bad, on ? 0.55 : 0.3);
      ctx.fillRect(bx, y, bw2, Math.min(bh, h));
      if (on) { ctx.strokeStyle = T.n14; ctx.lineWidth = 1.4; ctx.strokeRect(bx - 1.5, y - 1.5, bw2 + 3, h + 3); }
      const p = pctOf(s.sched.total, base.total);
      r.label(`${s.m}`, bx + bw2 / 2, mcY + mcH + 8, { color: on ? T.n14 : T.n10, font: '11px ui-monospace, monospace', align: 'center' });
      r.label(ms(s.sched.total), bx + bw2 / 2, y - 14, { color: on ? T.n13 : T.n10, font: '9.5px ui-monospace, monospace', align: 'center' });
      r.label(`${p}%`, bx + bw2 / 2, y - 4, { color: dirColor(p), font: '9.5px ui-monospace, monospace', align: 'center' });
    });
    // baseline rule across the chart, so "worse than one batch" is legible
    const byy = mcY + mcH - 6 - (base.total / mMax) * (mcH - 22);
    ctx.save(); ctx.strokeStyle = alphaOf(T.n12, 0.8); ctx.lineWidth = 1; ctx.setLineDash([5, 4]);
    ctx.beginPath(); ctx.moveTo(x0 + 4, byy + 0.5); ctx.lineTo(x0 + mcW - 4, byy + 0.5); ctx.stroke();
    ctx.setLineDash([]); ctx.restore();
    r.label('baseline (M = 1)', x0 + mcW - 6, byy - 4, { color: T.n11, font: '9px ui-monospace, monospace', align: 'right' });
    r.label('microbatches →   red = compute-lane bubble', x0 + 16, mcY + mcH + 20, { color: T.n10, font: '9px ui-monospace, monospace' });

    // ---- right column: the readout cards -----------------------------------
    const cx = xR + 30, cw = W - cx - pad;
    const card = (y, h) => { ctx.fillStyle = rgbaToken('n14', 0.04); ctx.fillRect(cx, y, cw, h); };
    let cy = ghostY - 12;
    r.label('vs baseline — lower is better', cx, cy - 6, { color: T.n12, font: '9.5px ui-monospace, monospace' });
    card(cy, 60);
    r.label('step time', cx + 8, cy + 14, { color: T.n11, font: '10px ui-monospace, monospace' });
    const pT = pctOf(cur.total, base.total);
    r.label(ms(cur.total), cx + 8, cy + 34, { color: dirColor(pT), font: '17px ui-monospace, monospace' });
    r.label(`${dirWord(pT)} — ${pT}% of baseline`, cx + 8, cy + 47, { color: T.n11, font: '9px ui-monospace, monospace' });
    r.label(`no overlap, one batch: ${ms(base.total)}`, cx + 8, cy + 57, { color: T.n10, font: '9px ui-monospace, monospace' });

    cy += 70; card(cy, 52);
    r.label('overlap achieved', cx + 8, cy + 14, { color: T.n11, font: '10px ui-monospace, monospace' });
    const ov = cur.overlapFrac * 100;
    r.label(`${ov.toFixed(1)}%`, cx + 8, cy + 34, { color: ov > 99.9 ? T.ok : ov > 50 ? T.warn : T.bad, font: '17px ui-monospace, monospace' });
    r.label(`${ms(cur.hidden)} of ${ms(cur.commSum)} hidden`, cx + 8, cy + 46, { color: T.n11, font: '9px ui-monospace, monospace' });
    // overlap bar
    ctx.fillStyle = rgbaToken('n14', 0.12); ctx.fillRect(cx + 8, cy + 49, cw - 16, 3);
    ctx.fillStyle = ov > 99.9 ? T.ok : T.warn; ctx.fillRect(cx + 8, cy + 49, (cw - 16) * cur.overlapFrac, 3);

    cy += 62; card(cy, 52);
    r.label('residual bubble', cx + 8, cy + 14, { color: T.n11, font: '10px ui-monospace, monospace' });
    r.label(ms(cur.bubble), cx + 8, cy + 34, { color: cur.bubble > 0.01 ? T.bad : T.ok, font: '17px ui-monospace, monospace' });
    r.label(`${((cur.bubble / cur.total) * 100).toFixed(1)}% of the step idle`, cx + 8, cy + 46, { color: T.n11, font: '9px ui-monospace, monospace' });

    cy += 62; card(cy, 60);
    r.label('per microbatch, per layer', cx + 8, cy + 14, { color: T.n11, font: '10px ui-monospace, monospace' });
    r.label(`compute ${ms(perMbCompute)}`, cx + 8, cy + 28, { color: T.n12, font: '9.5px ui-monospace, monospace' });
    r.label(`comm    ${ms(perMbComm)}`, cx + 8, cy + 40, { color: T.n12, font: '9.5px ui-monospace, monospace' });
    r.label(perMbCompute >= perMbComm ? 'compute ≥ comm → coverable' : 'compute < comm → bubble', cx + 8, cy + 53, { color: perMbCompute >= perMbComm ? T.ok : T.warn, font: '9px ui-monospace, monospace' });

    cy += 70;
    r.label(`fixed costs paid ${2 * M * L}× each:`, cx, cy, { color: T.n11, font: '9px ui-monospace, monospace' });
    r.label(`${LAUNCH_MS} ms launch · ${LAT_MS} ms link`, cx, cy + 11, { color: T.n10, font: '9px ui-monospace, monospace' });
    r.label('this is why splitting stops paying', cx, cy + 24, { color: T.n10, font: '9px ui-monospace, monospace' });

    // ---- hover-to-inspect --------------------------------------------------
    geom = { x0, xR, perMs, rects, mchart: { x: x0, y: mcY, w: mcW, h: mcH }, mbars, compY, commY, laneH };
    if (page.pointer.over && !drag) {
      const p = page.pointer;
      const hit = rects.find((q) => p.x >= q.x && p.x <= q.x + q.w && p.y >= q.y && p.y <= q.y + q.h);
      let tip = null;
      if (hit) {
        const t = hit.t;
        const w = t.waitOn || { kind: 'start' };
        const waits = w.kind === 'dep' && w.task
          ? `waits on: ${MB[w.task.m]}·layer ${w.task.l}·${w.task.name} (its own previous phase)`
          : w.kind === 'lane' && w.task
            ? `waits on: the ${t.lane} lane, busy with ${MB[w.task.m]}·layer ${w.task.l}·${w.task.name}`
            : 'waits on: nothing — it starts the step';
        const others = tasks.filter((q) => q.lane !== t.lane && Math.min(q.t1, t.t1) - Math.max(q.t0, t.t0) > 1e-9);
        const ovl = others.length
          ? `overlaps: ${others.map((q) => `${MB[q.m]}·${q.name} (${ms(Math.min(q.t1, t.t1) - Math.max(q.t0, t.t0))})`).join(', ')}`
          : `overlaps: NOTHING — the other lane is idle for all ${ms(t.dur)}`;
        tip = `microbatch ${MB[t.m]} · layer ${t.l} · ${t.name}\n`
          + `${t.lane === 'comm' ? 'communication' : 'compute'} lane   ${ms(t.t0)} → ${ms(t.t1)}   (${ms(t.dur)})\n`
          + `${waits}\n${ovl}\n`
          + (t.lane === 'comm' ? `hidden behind arithmetic: ${ms(t.hidden)} of ${ms(t.dur)}\ndrag ↔ to stretch the communication cost` : 'drag ↔ to stretch the compute cost');
      } else {
        for (const [a, b] of gaps) {
          if (p.y >= compY && p.y <= compY + laneH && p.x >= xOf(a) && p.x <= xOf(b)) {
            const blockers = tasks.filter((q) => q.lane === 'comm' && q.t1 > a + 1e-9 && q.t0 < b - 1e-9);
            tip = `BUBBLE — the arithmetic units idle for ${ms(b - a)}\n`
              + (blockers.length
                ? `nothing left to compute: ${blockers.map((q) => `${MB[q.m]}·${q.name}`).join(', ')} still in flight`
                : 'no microbatch has work ready here');
            break;
          }
        }
        if (!tip && p.y >= mcY && p.y <= mcY + mcH) {
          const b = mbars.find((q) => p.x >= q.x && p.x <= q.x + q.w);
          if (b) {
            const s = cur.sweep.find((q) => q.m === b.m).sched;
            tip = `${b.m} microbatch${b.m === 1 ? '' : 'es'}: step ${ms(s.total)} = ${pctOf(s.total, base.total)}% of baseline\n`
              + `overlap ${(s.overlapFrac * 100).toFixed(1)}% · bubble ${ms(s.bubble)}\n`
              + `per microbatch/layer: compute ${ms(s.dur.attn + s.dur.expert)} vs comm ${ms(s.dur.dispatch + s.dur.combine)}\n`
              + `perfect-overlap floor at this split: ${ms(s.floor)}`;
          }
        }
      }
      if (tip) page.setTip(tip);
    }

    // ---- readout -----------------------------------------------------------
    const s = page.step();
    let o = M === 1
      ? `M = 1: one batch, no sibling to interleave with — all ${ms(cur.commSum)} of routing all-to-all is dead time. `
      : `M = ${M}: microbatches ${MB.slice(0, M).join('/')} interleaved over ${L} layer${L === 1 ? '' : 's'} on two lanes. `;
    o += `tier:${r.name}\n`;
    o += `step time ${ms(cur.total)} = ${pctOf(cur.total, base.total)}% of the no-overlap one-batch baseline (${ms(base.total)}) `
      + `(lower is better; 100% = parity). Overlap achieved ${(cur.overlapFrac * 100).toFixed(1)}% of communication hidden `
      + `(${ms(cur.hidden)} of ${ms(cur.commSum)}); residual bubble ${ms(cur.bubble)} = ${((cur.bubble / cur.total) * 100).toFixed(1)}% of the step with the arithmetic units idle.`;
    const b2 = m2 ? m2.sched.total : 0, b4 = m4 ? m4.sched.total : 0;
    if (b2 && b4) o += `\nsplit sweep: 2 microbatches ${ms(b2)} · 4 microbatches ${ms(b4)} = ${pctOf(b4, b2)}% of the 2-microbatch step `
      + `(lower is better) — ${b4 > b2 ? 'splitting further is WORSE here' : 'splitting further still pays here'}.`;
    o += s ? `\n${s.label}` : '';
    page.setReadout(o);
  },
}).then((page) => {
  window.__dualBatchOverlapPage = page;
  const q = new URLSearchParams(location.search);
  const t = page.controls._transport;
  // Every control is a URL hook (the framework mirrors state into the query
  // string); these restore it on load. ?C= and ?V= are the headless stand-ins
  // for dragging a compute / comm block, since --screenshot has no pointer.
  const num = (k, lo, hi) => (q.has(k) ? clamp(parseFloat(q.get(k)) || lo, lo, hi) : null);
  const C = num('C', 0.5, 16); if (C != null) page.controls.set('C', C, { rebuild: true, silent: true });
  const V = num('V', 0, 12); if (V != null) page.controls.set('V', V, { rebuild: true, silent: true });
  const L = num('L', 1, 3); if (L != null) page.controls.set('L', Math.round(L), { rebuild: true, silent: true });
  if (q.has('M')) {
    const m = parseInt(q.get('M'), 10);
    if (M_CHOICES.includes(m)) page.controls.set('M', String(m), { rebuild: true, silent: true });
  }
  if (t) t.rebuild();
  // ?hover=x,y fakes the cursor (headless stand-in for a real hover) so the
  // phase / bubble tooltips are screenshot-verifiable. Canvas-space px.
  if (q.has('hover')) {
    const [hx, hy] = q.get('hover').split(',').map(Number);
    page.pointer.x = hx; page.pointer.y = hy; page.pointer.over = true;
  }
  // Deterministic frame for capture: pause before seeking, so autoplay does not
  // advance off the requested step.
  if (q.has('step') || q.has('hover') || q.has('C') || q.has('V') || q.has('M')) { if (t) t.pause(); }
  if (q.has('step') && t) t.seek(parseInt(q.get('step'), 10));
  if (q.get('play') === '1' && t) t.play();
  page.redraw();
});
