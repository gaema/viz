// kv-tiering concept page -- the KV cache spilled down a memory hierarchy, and
// the race that decides what happens on a later hit.
//
// Accelerator memory is small and fast. Host DRAM is larger and reached across
// a host link. A local SSD is larger again and slower still. A long-lived
// conversation's KV blocks do not have to be DESTROYED when they leave the top
// tier -- they can be pushed down. The interesting question is what happens
// when one of them is wanted again, because there are two ways to get it back
// and they are priced by completely different physics:
//
//   fetch_ms     = block_bytes / tier_bandwidth      (bytes over a link)
//   recompute_ms = the model's arithmetic, re-run over the block's tokens
//
// Fetch cost scales with the block's SIZE and the tier's BANDWIDTH. Recompute
// cost scales with the model's arithmetic and -- because the block's keys must
// be attended against everything before them -- with the block's POSITION in
// the sequence. So the winner FLIPS: with distance down the hierarchy (a
// deeper tier is slower, which favours recomputing), with spare arithmetic (a
// fast, idle machine recomputes almost for free), and with where the block
// sits (a block far into a conversation is expensive to rebuild).
//
// THIS PAGE BUILDS THE RACE RATHER THAN ASSERTING IT. Every tier has a
// capacity and a bandwidth you set; the model has an arithmetic rate you set;
// and BOTH candidate costs are computed for every hit, every time. The verdict
// strip shows the two bars per block and marks the winner; the race chart plots
// both curves against block size and marks where they cross. Move a slider far
// enough and the answer changes on screen.
//
// AND A DEEPER TIER IS NOT FREE EVEN ON A HIT. Each link is a real, serial
// resource here: fetches queue on it, and so do the SPILLS -- when a block is
// pushed out of the top tier, those bytes cross the link too. The occupancy
// timeline is what that costs; a link at 100% is the system's bottleneck no
// matter how good the hit rate looks.
//
// BASELINE: everything is scored against GPU-ONLY CACHING -- the same top-tier
// capacity, no tiers below it, so an evicted block is simply gone and any later
// hit is a full recompute. That is the thing tiering has to beat, and the
// readout states by how much, as a percent, with the direction named.
//
// The numbers are computed in-page from the sliders. The DEFAULTS are public
// nominal figures (a PCIe-class host link, a consumer NVMe sequential read
// rate, a small dense model at a plausible effective arithmetic rate) -- they
// set the SHAPE of the tradeoff, which is the lesson. This is not a calculator
// for any particular machine, and no measured number from any machine is in it.
//
// Interactive per the shared render framework's contract:
//  - TRANSPORT steps the arrival stream: one step = one request, blocks
//    migrating between tiers as it is served. Auto-plays and loops.
//  - DIRECT MANIPULATION: drag each tier's capacity strip to resize that tier,
//    drag the pipe between two tiers to change its bandwidth, and drag the
//    arithmetic-rate handle on the race chart. The verdicts re-decide under
//    your hand.
//  - HOVER any block for its tier, size, age and BOTH candidate costs; hover a
//    timeline segment, a race-chart curve, or a policy bar.
//  - POLICY switch: always fetch / always recompute / whichever is cheaper,
//    with each one's resulting mean latency drawn side by side.
//  - URL hooks reproduce every view headlessly: ?step, ?policy, ?cap0, ?cap1,
//    ?cap2, ?bw1, ?bw2, ?rate, ?params, ?kvkb, ?blk, ?nreq, ?gap, ?seed,
//    ?hover=x,y.
//
// Sources for the mechanism (public): the KV-offloading / cross-tier
// prefix-cache designs described in the vLLM project documentation and in the
// LMCache project's public papers and docs, which is where "store the KV below
// the accelerator and decide fetch-vs-recompute on a hit" is stated as the
// operating question. See README.md for the citations.
import { mount } from '../framework/layout.js';
import { categorical } from '../framework/render.js';
import { T, alphaOf, inkOn, rgbaToken } from '../framework/theme.js';

// ---- fixed structure of the workload ---------------------------------------
// Kept as constants rather than sliders so the control panel stays about the
// COST MODEL, which is the subject. A conversation grows one block per turn.
const NCONV = 6;          // distinct conversations in the mix
const DEPTH = 12;         // most blocks a conversation ever reaches
const DTYPE_BYTES = 2;    // KV stored at 2 bytes/element (bf16-class)
const GQA_GROUP = 4;      // query heads per KV head -- sets how much attention
                          // arithmetic each cached KV element implies

const TIER_NAME = ['accelerator memory', 'host DRAM', 'local SSD'];
const TIER_SHORT = ['GPU', 'DRAM', 'SSD'];
const clamp = (v, lo, hi) => (v < lo ? lo : v > hi ? hi : v);

// ---- the cost model (one place, so every number on screen is traceable) ----
// Bytes: K and V for every layer, for every token in the block.
const bytesPerTok = (st) => st.kvkb * 1024;
const blockBytes = (st) => st.blk * bytesPerTok(st);

// A transfer is bytes over a link. GB/s is decimal (1e9 B/s), the unit link
// vendors quote in.
const fetchMs = (bytes, bwGBs) => (bytes / (bwGBs * 1e9)) * 1000;

// Recompute is the model's own prefill arithmetic over the block's L tokens,
// sitting at offset `pos` in the sequence:
//   dense/projection work  = 2 * params * L                       (linear in L)
//   attention work         = 2 * kv_elems_per_tok * GQA * L * (pos + L/2)
// The second term is why POSITION matters: rebuilding a block near the end of a
// long conversation means attending it against everything before it.
function recomputeMs(st, L, pos) {
  const kvElems = bytesPerTok(st) / DTYPE_BYTES;
  const lin = 2 * st.params * 1e9 * L;
  const attn = 2 * kvElems * GQA_GROUP * L * (pos + L / 2);
  return ((lin + attn) / (st.rate * 1e12)) * 1000;
}
// Per-token forms, which is what the break-even algebra needs.
const linPerTokMs = (st) => ((2 * st.params * 1e9) / (st.rate * 1e12)) * 1000;
const attnPerPairMs = (st) => ((2 * (bytesPerTok(st) / DTYPE_BYTES) * GQA_GROUP) / (st.rate * 1e12)) * 1000;
const fetchPerTokMs = (st, bw) => (bytesPerTok(st) / (bw * 1e9)) * 1000;

// BREAK-EVEN BLOCK SIZE, solved rather than searched.
//   fetch:     L * fetchPerTok
//   recompute: L * (linPerTok + attnPerPair * (pos + L/2))
// Equal when  fetchPerTok = linPerTok + attnPerPair*pos + attnPerPair*L/2, so
//   L* = 2 * (fetchPerTok - linPerTok - attnPerPair*pos) / attnPerPair
// Above L* the block is cheaper to FETCH (recompute grows faster than linearly
// in L); below it, cheaper to REBUILD. A NON-POSITIVE L* is the other honest
// answer, and it points the OTHER way: it means fetching already wins at every
// block size, because this tier delivers a token's KV faster than the model can
// regenerate one even before attention is counted.
function breakEvenL(st, bw, pos) {
  const a = attnPerPairMs(st);
  if (a <= 0) return null;
  return (2 * (fetchPerTokMs(st, bw) - linPerTokMs(st) - a * pos)) / a;
}

function hash32(a, b) {
  let x = (Math.imul(a + 1, 374761393) + Math.imul(b + 1, 668265263)) >>> 0;
  x = (x ^ (x >>> 13)) >>> 0;
  x = Math.imul(x, 1274126177) >>> 0;
  return (x ^ (x >>> 16)) >>> 0;
}

// The arrival stream: one request every `gap` ms, each a turn in one of NCONV
// conversations (skewed, because a real mix always has hot and cold ones).
// Turn n of a conversation needs its first n+2 blocks -- a growing prefix, which
// is exactly the shape that makes a KV cache worth keeping across turns.
// Seeded, so one URL replays one arrival stream.
function buildRequests(st) {
  const out = [], seen = new Array(NCONV).fill(0);
  for (let i = 0; i < (st.nreq | 0); i++) {
    const h = hash32(st.seed | 0, i);
    const u = (h % 1000) / 1000;
    const c = clamp(Math.floor(Math.pow(u, 1.7) * NCONV), 0, NCONV - 1);
    const k = Math.min(DEPTH, 2 + seen[c]);
    seen[c]++;
    out.push({ id: i, conv: c, need: k, arrival: i * st.gap });
  }
  return out;
}

// ---- the simulation --------------------------------------------------------
// `nBelow` = how many tiers exist BELOW the top one. 0 gives the named
// baseline: GPU-only caching, where an evicted block is gone and every later
// hit is a full recompute.
//
// Three serial engines: one compute unit (recomputes queue on it) and one link
// per tier boundary (fetches AND spills queue on it). They run concurrently
// with each other, which is why a policy can be beaten by its own contention.
function simulate(st, policy, nBelow) {
  const caps = [st.cap0 | 0, st.cap1 | 0, st.cap2 | 0].slice(0, 1 + nBelow);
  const bw = [Infinity, +st.bw1, +st.bw2];
  const bytes = blockBytes(st);
  const tiers = caps.map(() => []);        // key lists, LRU at [0], MRU at end
  const where = new Map();                 // key -> tier index (absent = gone)
  const lastUse = new Map();               // key -> request index
  const eng = { compute: 0, link1: 0, link2: 0 };   // freeAt, ms
  const lanes = { compute: [], link1: [], link2: [] };
  const reqs = buildRequests(st);
  const frames = [], gone = new Set();
  // A lookup that finds nothing is TWO different events -- a block the
  // conversation has never reached before (unavoidable work), and one the
  // hierarchy destroyed (work tiering was supposed to save). Counted apart.
  const hits = [0, 0, 0], cold = { fresh: 0, dropped: 0 }, acted = { fetch: 0, recompute: 0, free: 0 };
  let movedBytes = 0, lat = 0, worst = 0;

  const engOf = (t) => (t === 1 ? 'link1' : 'link2');
  const occupy = (name, at, cost, entry) => {
    const t0 = Math.max(at, eng[name]), t1 = t0 + cost;
    eng[name] = t1;
    lanes[name].push({ ...entry, t0, t1 });
    return t1;
  };

  // Promote to the top tier, then let capacity push the overflow downward. A
  // spill is a real transfer: it occupies the link into the tier below.
  const touch = (key, at, reqId) => {
    const t = where.get(key);
    if (t != null) tiers[t] = tiers[t].filter((k) => k !== key);
    tiers[0].push(key); where.set(key, 0); gone.delete(key);
    for (let i = 0; i < tiers.length; i++) {
      while (tiers[i].length > caps[i]) {
        const k = tiers[i].shift();
        if (i + 1 < tiers.length) {
          occupy(engOf(i + 1), at, fetchMs(bytes, bw[i + 1]), { key: k, kind: 'spill', req: reqId });
          movedBytes += bytes;
          tiers[i + 1].push(k); where.set(k, i + 1);
        } else { where.delete(k); gone.add(k); }
      }
    }
  };

  for (const r of reqs) {
    const blocks = [];
    let ready = r.arrival;
    for (let j = 0; j < r.need; j++) {
      const key = r.conv + ':' + j;
      const t = where.has(key) ? where.get(key) : -1;
      const pos = j * st.blk;
      const rc = recomputeMs(st, st.blk, pos);
      const fc = t > 0 ? fetchMs(bytes, bw[t]) : (t === 0 ? 0 : Infinity);
      let action, cost, on = null;
      if (t === 0) { action = 'resident'; cost = 0; hits[0]++; acted.free++; }
      else if (t < 0) {
        action = 'recompute'; cost = rc; on = 'compute'; acted.recompute++;
        if (lastUse.has(key)) cold.dropped++; else cold.fresh++;
      }
      else {
        hits[t]++;
        // The whole page in one line: both candidates priced, one chosen.
        const wantFetch = policy === 'fetch' ? true : policy === 'recompute' ? false : fc <= rc;
        action = wantFetch ? 'fetch' : 'recompute';
        cost = wantFetch ? fc : rc;
        on = wantFetch ? engOf(t) : 'compute';
        acted[action]++;
        if (wantFetch) movedBytes += bytes;
      }
      let t0 = r.arrival, t1 = r.arrival;
      if (on) { t1 = occupy(on, r.arrival, cost, { key, kind: action, req: r.id }); t0 = t1 - cost; }
      ready = Math.max(ready, t1);
      blocks.push({
        key, j, pos, tier: t, rc, fc, action, cost, t0, t1,
        age: lastUse.has(key) ? r.id - lastUse.get(key) : null,
      });
      lastUse.set(key, r.id);
    }
    // Promotion happens after every block of this request has been priced, so
    // they all see the same pre-request residency.
    for (const b of blocks) touch(b.key, ready, r.id);
    const L = ready - r.arrival;
    lat += L; worst = Math.max(worst, L);
    frames.push({
      req: r, blocks, ready, lat: L,
      tiers: tiers.map((x) => x.slice()), gone: [...gone],
    });
  }

  const n = Math.max(1, reqs.length);
  // The run is not over when the last request answers: a spill queued behind it
  // is still on the link. Taking the last READY time as the span let occupancy
  // read over 100%, which is not a thing a serial resource can do.
  let end = frames.length ? frames[frames.length - 1].ready : 1;
  for (const k of ['compute', 'link1', 'link2']) for (const e of lanes[k]) end = Math.max(end, e.t1);
  const span = Math.max(1, end);
  const busy = {
    compute: lanes.compute.reduce((s, e) => s + (e.t1 - e.t0), 0),
    link1: lanes.link1.reduce((s, e) => s + (e.t1 - e.t0), 0),
    link2: lanes.link2.reduce((s, e) => s + (e.t1 - e.t0), 0),
  };
  return {
    policy, nBelow, caps, reqs, frames, lanes, span,
    mean: lat / n, worst, hits, cold, acted, movedBytes,
    busy: { compute: busy.compute / span, link1: busy.link1 / span, link2: busy.link2 / span },
  };
}

// ---- drawing helpers -------------------------------------------------------
function roundRect(ctx, x, y, w, h, r) {
  const rr = Math.max(0, Math.min(r, w / 2, h / 2));
  ctx.beginPath(); ctx.moveTo(x + rr, y);
  ctx.arcTo(x + w, y, x + w, y + h, rr); ctx.arcTo(x + w, y + h, x, y + h, rr);
  ctx.arcTo(x, y + h, x, y, rr); ctx.arcTo(x, y, x + w, y, rr); ctx.closePath();
}
function hatch(ctx, x, y, w, h, color, bg) {
  if (w <= 0 || h <= 0) return;
  ctx.save();
  ctx.beginPath(); ctx.rect(x, y, w, h); ctx.clip();
  if (bg) { ctx.fillStyle = bg; ctx.fillRect(x, y, w, h); }
  ctx.strokeStyle = color; ctx.lineWidth = 1;
  for (let k = -h; k < w; k += 5) { ctx.beginPath(); ctx.moveTo(x + k, y + h); ctx.lineTo(x + k + h, y); ctx.stroke(); }
  ctx.restore();
}
const fmtMs = (v) => (!isFinite(v) ? '∞' : v >= 100 ? v.toFixed(0) : v >= 10 ? v.toFixed(1) : v.toFixed(2));
const fmtMB = (b) => (b / 1e6 >= 1000 ? (b / 1e9).toFixed(2) + ' GB' : (b / 1e6).toFixed(1) + ' MB');

let cur = null;    // the four simulations + their metrics
let geom = null;   // hit-test geometry captured each draw
let drag = null;   // {mode, ...} while a handle is held

const simCache = new Map();
function rebuild(st) {
  const key = [st.policy, st.cap0, st.cap1, st.cap2, st.bw1, st.bw2, st.rate,
    st.params, st.kvkb, st.blk, st.nreq, st.gap, st.seed].join('|');
  const hit = simCache.get(key);
  if (hit) { cur = hit; return hit.steps; }
  const active = simulate(st, st.policy, 2);
  const alt = {
    cheaper: st.policy === 'cheaper' ? active : simulate(st, 'cheaper', 2),
    fetch: st.policy === 'fetch' ? active : simulate(st, 'fetch', 2),
    recompute: st.policy === 'recompute' ? active : simulate(st, 'recompute', 2),
  };
  const base = simulate(st, 'recompute', 0);   // GPU-only caching: the baseline
  const steps = active.frames.map((f, i) => ({
    i, label: `request ${i + 1} / ${active.frames.length} — conversation ${f.req.conv}, ${f.req.need} blocks, ${f.lat.toFixed(1)} ms`,
  }));
  cur = { active, alt, base, steps, bytes: blockBytes(st) };
  if (simCache.size > 24) simCache.clear();
  simCache.set(key, cur);
  return steps;
}

function blockTip(st, b, bytes) {
  // A block that is in no tier is one of TWO different things, and conflating
  // them misreads the picture: it may never have existed yet (the first time
  // this turn extends the conversation), or it may have been pushed off the
  // bottom of the hierarchy and destroyed.
  const where = b.tier >= 0 ? TIER_NAME[b.tier]
    : b.age == null ? 'nowhere yet — this is the first time it has ever been needed'
      : 'nowhere — it was pushed off the bottom tier and destroyed';
  const lines = [
    `block ${b.key.replace(':', ' · block ')} — ${st.blk} tokens at sequence offset ${b.pos}`,
    `lived in: ${where}`,
    `size: ${st.blk} tok × ${st.kvkb} KB/tok = ${fmtMB(bytes)} of K and V`,
    b.age == null ? 'first time this block has ever been asked for' : `age: last used ${b.age} request(s) ago`,
    '',
    `FETCH     ${b.tier > 0 ? `${fmtMB(bytes)} ÷ ${b.tier === 1 ? st.bw1 : st.bw2} GB/s = ${fmtMs(b.fc)} ms` : b.tier === 0 ? 'already resident — 0 ms' : 'impossible — no copy survives anywhere'}`,
    `RECOMPUTE ${fmtMs(b.rc)} ms  (${st.blk} tok × ${st.params}B params, attended against the ${b.pos} tokens before it, at ${st.rate} TFLOP/s)`,
    '',
    b.action === 'resident' ? 'verdict: a top-tier hit — neither path was needed'
      : b.action === 'fetch' ? `verdict: FETCHED (${fmtMs(b.fc)} ms beats ${fmtMs(b.rc)} ms), and it occupied the ${TIER_SHORT[b.tier]} link while it moved`
        : b.tier < 0 ? 'verdict: RECOMPUTED — there was nothing anywhere to fetch, so no race was run'
          : `verdict: RECOMPUTED (${fmtMs(b.rc)} ms beats ${fmtMs(b.fc)} ms), and it occupied the compute unit`,
  ];
  return lines.join('\n');
}

mount({
  mount: 'body',
  title: 'kv-tiering — spill the cache downhill, then race fetch against recompute',
  blurb: 'Accelerator memory is small and fast; host DRAM is larger and across a link; an SSD is larger again and slower still. A conversation\'s KV blocks do not have to be destroyed when they leave the top tier — they can be pushed down it. The question is what happens when one is wanted again, because there are two ways to get it back: FETCH the bytes up the hierarchy, or RECOMPUTE the block from its tokens. Fetch cost is size ÷ bandwidth. Recompute cost is the model\'s arithmetic over those tokens — and it grows with how deep into the conversation the block sits, because its keys must be attended against everything before them. Both are computed for every hit here, and the winner is drawn, never assumed. DRAG a tier\'s capacity strip to resize it, drag a pipe to change that link\'s bandwidth, drag the arithmetic-rate handle on the race chart — and watch the verdicts flip. A deeper tier is not free even on a hit: every fetch AND every spill occupies its link, which the occupancy bars charge.',
  prefer: 'canvas2d',
  aspect: '16 / 14',
  autoplay: true,
  animate: true,
  compare: {
    key: 'policy', a: 'recompute', b: 'cheaper',
    labelA: 'always recompute', labelB: 'whichever is cheaper', rebuild: true,
  },
  challenges: [
    {
      goal: 'Make RECOMPUTE the cheaper answer for a host-DRAM hit.',
      hint: 'recompute is priced by arithmetic and fetch by bandwidth — so raise the arithmetic rate, shrink the model, thin the DRAM link, or fatten each token\'s KV. The race chart\'s crossing point moves as you do.',
      check: (api) => ({
        solved: (api.probe.rc0 ?? 1e9) < (api.probe.fc1 ?? 0),
        detail: `DRAM block at offset 0: recompute ${fmtMs(api.probe.rc0 ?? 0)} ms vs fetch ${fmtMs(api.probe.fc1 ?? 0)} ms`,
      }),
    },
    {
      goal: 'Get the tiered cache under 60% of the GPU-only baseline\'s mean latency (lower is better).',
      hint: 'the top tier decides how often you pay anything at all; the tiers below decide what a miss costs. Both matter — and so does the policy, because the wrong one queues everything onto one engine.',
      check: (api) => ({
        solved: (api.probe.pct ?? 999) < 60,
        detail: `mean latency ${fmtMs(api.probe.mean ?? 0)} ms = ${(api.probe.pct ?? 0).toFixed(0)}% of the ${fmtMs(api.probe.baseMean ?? 0)} ms GPU-only baseline`,
      }),
    },
  ],

  controls: (c, page) => {
    c.select('policy', {
      label: 'on a hit below the top tier',
      options: [
        { value: 'cheaper', label: 'whichever is cheaper' },
        { value: 'fetch', label: 'always fetch' },
        { value: 'recompute', label: 'always recompute' },
      ],
      value: 'cheaper', rebuild: true,
    });
    c.slider('cap0', { label: 'accelerator memory (blocks)', min: 2, max: 48, step: 1, value: 10, rebuild: true });
    // Deliberately small by default: with DRAM able to hold the whole working
    // set, nothing ever reaches the SSD and the page opens on a picture where
    // the deepest tier is decoration. A shallow middle tier makes the third
    // tier -- and the policy disagreement that lives there -- visible on load.
    c.slider('cap1', { label: 'host DRAM (blocks)', min: 2, max: 160, step: 1, value: 12, rebuild: true });
    c.slider('cap2', { label: 'local SSD (blocks)', min: 4, max: 512, step: 4, value: 128, rebuild: true });
    c.slider('bw1', { label: 'host link (GB/s)', min: 1, max: 128, step: 1, value: 55, rebuild: true });
    c.slider('bw2', { label: 'SSD read (GB/s)', min: 0.2, max: 16, step: 0.1, value: 7, rebuild: true });
    c.slider('rate', { label: 'arithmetic rate (TFLOP/s)', min: 5, max: 1000, step: 5, value: 400, rebuild: true });
    c.slider('params', { label: 'model size (B params)', min: 1, max: 70, step: 1, value: 3, rebuild: true });
    c.slider('kvkb', { label: 'KV per token (KB)', min: 8, max: 512, step: 8, value: 128, rebuild: true });
    c.stepper('blk', { label: 'block size (tokens)', min: 32, max: 1024, step: 32, value: 256 });
    c.stepper('nreq', { label: 'requests', min: 6, max: 60, value: 30 });
    c.stepper('gap', { label: 'arrival gap (ms)', min: 1, max: 60, value: 12 });
    c.slider('seed', { label: 'arrival seed', min: 0, max: 99, step: 1, value: 5, rebuild: true });
    c.transport({ compute: () => rebuild(page.state), speed: 4, loop: true });
  },

  // Direct manipulation. Every handle writes through controls.set() so the
  // widget, the URL and the simulation move together.
  onPointer: (page, ev) => {
    if (!geom || !cur) return;
    const st = page.state;
    const inR = (R, pad = 0) => R && ev.x >= R.x - pad && ev.x <= R.x + R.w + pad && ev.y >= R.y - pad && ev.y <= R.y + R.h + pad;
    if (ev.type === 'down') {
      drag = null;
      for (const s of geom.capStrips) if (inR(s.R, 6)) { drag = { mode: 'cap', tier: s.tier, R: s.R, per: s.per, max: s.max }; break; }
      if (!drag) for (const p of geom.pipes) if (inR(p.R, 14)) { drag = { mode: 'bw', tier: p.tier, y0: ev.y, v0: +st[p.key], key: p.key }; break; }
      if (!drag && geom.rateHandle && inR(geom.rateHandle, 14)) drag = { mode: 'rate', y0: ev.y, v0: +st.rate };
    } else if (ev.type === 'up' || ev.type === 'leave') { drag = null; }
    if (!drag || !page.pointer.down) return;
    if (drag.mode === 'cap') {
      const key = 'cap' + drag.tier;
      const step = drag.tier === 2 ? 4 : 1;
      let want = Math.round(((ev.x - drag.R.x) / Math.max(1, drag.per)) / step) * step;
      want = clamp(want, drag.tier === 2 ? 4 : 2, drag.max);
      if (want !== (st[key] | 0)) page.controls.set(key, want, { rebuild: true });
    } else if (drag.mode === 'bw') {
      // vertical drag on a log axis: up = wider pipe.
      const lo = drag.key === 'bw1' ? 1 : 0.2, hi = drag.key === 'bw1' ? 128 : 16;
      let want = clamp(drag.v0 * Math.pow(10, (drag.y0 - ev.y) / 130), lo, hi);
      want = drag.key === 'bw1' ? Math.round(want) : Math.round(want * 10) / 10;
      if (want !== +st[drag.key]) page.controls.set(drag.key, want, { rebuild: true });
    } else if (drag.mode === 'rate') {
      const want = clamp(Math.round((drag.v0 * Math.pow(10, (drag.y0 - ev.y) / 150)) / 5) * 5, 5, 1000);
      if (want !== +st.rate) page.controls.set('rate', want, { rebuild: true });
    }
  },

  draw: (page) => {
    const r = page.renderer, ctx = page.ctx, st = page.state;
    if (!cur) rebuild(st);
    const { active, alt, base, bytes } = cur;
    r.clear(T.n0);
    const mono = (px) => `${px}px ui-monospace, monospace`;
    const W = page.W, H = page.H;

    const s = page.step();
    const now = clamp(s ? s.i : active.frames.length - 1, 0, Math.max(0, active.frames.length - 1));
    const F = active.frames[now];

    // Costs at the two reference points the readout and the challenges quote.
    const deepPos = st.blk * (DEPTH - 1);
    const rc0 = recomputeMs(st, st.blk, 0), rcD = recomputeMs(st, st.blk, deepPos);
    const fc1 = fetchMs(bytes, st.bw1), fc2 = fetchMs(bytes, st.bw2);
    const pct = base.mean > 0 ? (100 * active.mean / base.mean) : 100;
    page.probe = {
      rc0, rcD, fc1, fc2, mean: active.mean, baseMean: base.mean, pct,
      policy: st.policy, be1: breakEvenL(st, st.bw1, 0), be2: breakEvenL(st, st.bw2, 0),
    };

    // ---- header ------------------------------------------------------------
    r.label(`${st.blk}-token blocks of ${fmtMB(bytes)} · ${st.params}B model at ${st.rate} TFLOP/s · request ${now + 1} of ${active.frames.length} (conversation ${F.req.conv}, ${F.req.need} blocks)`,
      8, 14, { color: T.n14, font: mono(12.5) });
    let lx = 8;
    const key = (col, text, hatched) => {
      if (hatched) hatch(ctx, lx, 20, 9, 8, alphaOf(col, 0.9), alphaOf(col, 0.15));
      else { ctx.fillStyle = alphaOf(col, 0.85); ctx.fillRect(lx, 20, 9, 8); }
      r.label(text, lx + 12, 27, { color: T.n11, font: mono(10) });
      ctx.save(); ctx.font = mono(10); lx += 12 + ctx.measureText(text).width + 12; ctx.restore();
    };
    key(T.violet, 'fetched up a link');
    key(T.warn, 'recomputed from tokens');
    key(T.ok, 'already in the top tier');
    key(T.n9, 'spilled downhill (a transfer too)', true);

    // ================= the hierarchy (left) =================================
    const midX = Math.round(W * 0.5);
    const hierX = 8, hierW = midX - 20;
    const hierTop = 36, hierH = Math.max(150, H * 0.30);
    const rowH = (hierH - 16) / 3;
    const capStrips = [], pipes = [], blockRects = [];
    const inThisReq = new Map();
    for (const b of F.blocks) inThisReq.set(b.key, b);

    for (let t = 0; t < 3; t++) {
      const y = hierTop + t * rowH;
      const box = { x: hierX, y, w: hierW, h: rowH - 10 };
      const tone = t === 0 ? T.ok : t === 1 ? T.accent : T.teal;
      ctx.save();
      roundRect(ctx, box.x, box.y, box.w, box.h, 7);
      ctx.fillStyle = alphaOf(tone, 0.06); ctx.fill();
      ctx.strokeStyle = alphaOf(tone, 0.5); ctx.lineWidth = 1.1; ctx.stroke();
      ctx.restore();
      const capN = [st.cap0 | 0, st.cap1 | 0, st.cap2 | 0][t];
      const held = F.tiers[t] ? F.tiers[t].length : 0;
      r.label(`${TIER_NAME[t].toUpperCase()} · ${held} / ${capN} blocks · ${fmtMB(held * bytes)}`,
        box.x + 8, box.y + 13, { color: tone, font: mono(10.5) });
      r.label(t === 0 ? 'no transfer needed — a hit here is free'
        : `reached over ${t === 1 ? `the host link at ${st.bw1} GB/s` : `SSD read at ${st.bw2} GB/s`} · fetch a block = ${fmtMs(t === 1 ? fc1 : fc2)} ms`,
        box.x + 8, box.y + 24, { color: T.n10, font: mono(9) });

      // the capacity strip: one cell per SLOT, so an empty slot is visible and
      // the right edge is a real handle you can pull.
      const sx = box.x + 8, sy = box.y + 30, sw = box.w - 16, sh = Math.max(9, box.h - 38);
      const maxSlots = [48, 160, 512][t];
      const per = sw / maxSlots;
      const R = { x: sx, y: sy, w: sw, h: sh };
      capStrips.push({ tier: t, R, per, max: maxSlots });
      ctx.save();
      ctx.strokeStyle = rgbaToken('n14', 0.12); ctx.lineWidth = 1;
      ctx.strokeRect(sx, sy, sw, sh);
      // the allocated part of the strip
      ctx.fillStyle = rgbaToken('n14', 0.05);
      ctx.fillRect(sx, sy, per * capN, sh);
      const list = F.tiers[t] || [];
      for (let i = 0; i < list.length; i++) {
        const k = list[i], conv = +k.split(':')[0];
        const bx = sx + i * per, bwid = Math.max(1.5, per - 0.8);
        const mine = inThisReq.get(k);
        ctx.fillStyle = alphaOf(categorical(conv), mine ? 0.95 : 0.6);
        ctx.fillRect(bx, sy + 1, bwid, sh - 2);
        if (mine) { ctx.strokeStyle = T.n14; ctx.lineWidth = 1.2; ctx.strokeRect(bx - 0.5, sy + 0.5, bwid + 1, sh - 1); }
        blockRects.push({ R: { x: bx, y: sy + 1, w: Math.max(3, bwid), h: sh - 2 }, key: k, tier: t });
      }
      ctx.strokeStyle = tone; ctx.lineWidth = 2;
      ctx.beginPath(); ctx.moveTo(sx + per * capN, sy - 3); ctx.lineTo(sx + per * capN, sy + sh + 3); ctx.stroke();
      ctx.restore();
      r.label('◀ drag the edge: resize this tier ▶', sx + per * capN + 6, sy + sh * 0.7,
        { color: T.n10, font: mono(8.5) });

      // the pipe INTO this tier from the one above: thickness = bandwidth, and
      // its fill is how much of the run that link was busy.
      if (t > 0) {
        const bwv = t === 1 ? +st.bw1 : +st.bw2;
        const lo = t === 1 ? 1 : 0.2, hi = t === 1 ? 128 : 16;
        const frac = (Math.log10(bwv) - Math.log10(lo)) / (Math.log10(hi) - Math.log10(lo));
        const ph = 4 + 12 * clamp(frac, 0, 1);
        const pR = { x: box.x + box.w * 0.30, y: y - 11 - ph / 2, w: box.w * 0.24, h: ph };
        const busy = t === 1 ? active.busy.link1 : active.busy.link2;
        ctx.save();
        roundRect(ctx, pR.x, pR.y, pR.w, pR.h, Math.min(5, ph / 2));
        ctx.fillStyle = alphaOf(T.violet, 0.14); ctx.fill();
        ctx.strokeStyle = alphaOf(T.violet, 0.7); ctx.lineWidth = 1.1; ctx.stroke();
        ctx.save();
        roundRect(ctx, pR.x, pR.y, pR.w, pR.h, Math.min(5, ph / 2)); ctx.clip();
        ctx.fillStyle = alphaOf(T.violet, 0.55);
        ctx.fillRect(pR.x, pR.y, pR.w * clamp(busy, 0, 1), pR.h);
        // packets in flight, so the link reads as a moving thing, not a shape
        for (let q = 0; q < 4; q++) {
          const f = (page.t * 0.7 + q / 4) % 1;
          ctx.fillStyle = alphaOf(T.violetDeep, 0.9);
          ctx.beginPath(); ctx.arc(pR.x + f * pR.w, pR.y + pR.h / 2, Math.min(2.5, ph / 3), 0, 6.2832); ctx.fill();
        }
        ctx.restore(); ctx.restore();
        r.label(`${t === 1 ? st.bw1 : st.bw2} GB/s ↕drag · link busy ${(busy * 100).toFixed(0)}%`,
          pR.x + pR.w + 6, pR.y + pR.h / 2 + 3, { color: T.violet, font: mono(9) });
        pipes.push({ tier: t, R: pR, key: t === 1 ? 'bw1' : 'bw2' });
      }
    }
    r.label(`dropped entirely: ${F.gone.length} block(s) — pushed off the bottom of the hierarchy, so a later hit MUST be recomputed`,
      hierX, hierTop + hierH + 2, { color: T.n10, font: mono(9) });

    // ================= the race chart (right) ===============================
    // fetch_ms and recompute_ms as functions of BLOCK SIZE, both on log axes.
    // Fetch is a straight line through the origin (bytes ∝ tokens). Recompute
    // bends upward, because attention makes it grow faster than linearly. Where
    // they cross is the break-even block size, and it is drawn, not asserted.
    const chX = midX + 4, chW = W - chX - 10;
    const chY = hierTop + 6, chH = hierH - 24;
    const Lmin = 32, Lmax = 4096;
    const yMin = 0.02, yMax = Math.max(200, rcD * 3, fc2 * 3);
    const px = (L) => chX + 34 + (chW - 44) * (Math.log10(L) - Math.log10(Lmin)) / (Math.log10(Lmax) - Math.log10(Lmin));
    const py = (v) => chY + 16 + (chH - 34) * (1 - (Math.log10(Math.max(yMin, v)) - Math.log10(yMin)) / (Math.log10(yMax) - Math.log10(yMin)));
    ctx.save();
    ctx.strokeStyle = rgbaToken('n14', 0.14); ctx.lineWidth = 1;
    ctx.strokeRect(chX + 34, chY + 16, chW - 44, chH - 34);
    for (const g of [0.1, 1, 10, 100, 1000]) {
      if (g > yMax) continue;
      const gy = py(g);
      ctx.beginPath(); ctx.moveTo(chX + 34, gy); ctx.lineTo(chX + chW - 10, gy); ctx.stroke();
      r.label(g >= 1 ? `${g} ms` : `${g}`, chX + 30, gy + 3, { color: T.n10, font: mono(8.5), align: 'right' });
    }
    ctx.restore();
    r.label('the race: cost of ONE block, against block size', chX + 34, chY + 10, { color: T.n13, font: mono(10.5) });

    const curveOf = (fn) => {
      const pts = [];
      for (let i = 0; i <= 60; i++) {
        const L = Lmin * Math.pow(Lmax / Lmin, i / 60);
        pts.push([px(L), py(fn(L)), L, fn(L)]);
      }
      return pts;
    };
    const stroke = (pts, color, dash, wid) => {
      ctx.save(); ctx.strokeStyle = color; ctx.lineWidth = wid || 1.8;
      if (dash) ctx.setLineDash(dash);
      ctx.beginPath();
      pts.forEach((p, i) => (i ? ctx.lineTo(p[0], p[1]) : ctx.moveTo(p[0], p[1])));
      ctx.stroke(); ctx.restore();
    };
    const cFetch1 = curveOf((L) => fetchMs(L * bytesPerTok(st), st.bw1));
    const cFetch2 = curveOf((L) => fetchMs(L * bytesPerTok(st), st.bw2));
    const cRec0 = curveOf((L) => recomputeMs(st, L, 0));
    const cRecD = curveOf((L) => recomputeMs(st, L, deepPos));
    stroke(cFetch1, alphaOf(T.accent, 0.95));
    stroke(cFetch2, alphaOf(T.teal, 0.95));
    stroke(cRec0, alphaOf(T.warn, 0.95));
    stroke(cRecD, alphaOf(T.warnDeep, 0.9), [4, 3]);

    // the crossings, solved from the algebra above
    const marks = [
      { L: breakEvenL(st, st.bw1, 0), color: T.accent, lab: 'host link × recompute' },
      { L: breakEvenL(st, st.bw2, 0), color: T.teal, lab: 'SSD × recompute' },
    ];
    for (const m of marks) {
      if (!(m.L > Lmin && m.L < Lmax)) continue;
      const x = px(m.L), y = py(recomputeMs(st, m.L, 0));
      ctx.save();
      ctx.strokeStyle = alphaOf(m.color, 0.7); ctx.setLineDash([2, 3]); ctx.lineWidth = 1;
      ctx.beginPath(); ctx.moveTo(x, chY + 16); ctx.lineTo(x, chY + chH - 18); ctx.stroke();
      ctx.setLineDash([]);
      ctx.fillStyle = m.color; ctx.beginPath(); ctx.arc(x, y, 3.2, 0, 6.2832); ctx.fill();
      ctx.restore();
      r.label(`${Math.round(m.L)} tok`, x, chY + chH - 22, { color: m.color, font: mono(8.5), align: 'center' });
    }
    // where the reader currently is
    const nowX = px(clamp(st.blk, Lmin, Lmax));
    ctx.save(); ctx.strokeStyle = T.n13; ctx.lineWidth = 1.4;
    ctx.beginPath(); ctx.moveTo(nowX, chY + 16); ctx.lineTo(nowX, chY + chH - 18); ctx.stroke(); ctx.restore();
    r.label(`your ${st.blk}-token block`, nowX, chY + chH - 8, { color: T.n13, font: mono(9), align: 'center' });
    for (const [x, lab] of [[Lmin, '32'], [256, '256'], [1024, '1k'], [Lmax, '4k tok']]) {
      r.label(lab, px(x), chY + chH - 2, { color: T.n10, font: mono(8.5), align: 'center' });
    }

    // the arithmetic-rate handle rides the recompute curve's right end
    const rh = cRec0[cRec0.length - 1];
    const rateHandle = { x: rh[0] - 24, y: rh[1] - 7, w: 24, h: 14 };
    ctx.save();
    roundRect(ctx, rateHandle.x, rateHandle.y, rateHandle.w, rateHandle.h, 4);
    ctx.fillStyle = alphaOf(T.warn, drag && drag.mode === 'rate' ? 0.95 : 0.6); ctx.fill();
    ctx.restore();
    r.label('↕', rateHandle.x + rateHandle.w / 2, rateHandle.y + 11, { color: inkOn(T.warn), font: mono(10), align: 'center' });
    // A stacked legend rather than labels sitting on the curves: at low block
    // sizes the four lines are close enough together that on-curve text
    // overlaps itself, which is worse than no label at all.
    const legend = [
      [T.accent, `fetch · host link ${st.bw1} GB/s`],
      [T.teal, `fetch · SSD ${st.bw2} GB/s`],
      [T.warn, `recompute @ offset 0`],
      [T.warnDeep, `recompute @ offset ${deepPos}`],
    ];
    legend.forEach(([col, text], i) => {
      const ly = chY + 24 + i * 11;
      ctx.save(); ctx.strokeStyle = col; ctx.lineWidth = 2;
      if (i === 3) ctx.setLineDash([4, 3]);
      ctx.beginPath(); ctx.moveTo(chX + 40, ly - 3); ctx.lineTo(chX + 52, ly - 3); ctx.stroke();
      ctx.restore();
      r.label(text, chX + 56, ly, { color: col, font: mono(8.5) });
    });

    // ================= this request's verdicts ==============================
    const vY = hierTop + hierH + 16;
    const vH = Math.max(84, H * 0.21);
    r.label(`REQUEST ${now + 1}: every block it needs, both candidate costs, and the winner — total ${F.lat.toFixed(1)} ms`,
      8, vY + 9, { color: T.n13, font: mono(11) });
    r.label('left bar = fetch · right bar = recompute · solid = the one actually paid · hover a column for both costs',
      8, vY + 20, { color: T.n10, font: mono(9) });
    const nb = Math.max(1, F.blocks.length);
    // Cap the column width: a two-block request would otherwise draw two bars
    // half the page wide, which reads as a bar CHART rather than a per-block
    // comparison.
    const colW = Math.min(96, (W - 16) / nb);
    const barTop = vY + 40, barH = vH - 60;
    const scale = Math.max(rcD, fc2, rc0, fc1, 0.001);
    const verdictRects = [];
    F.blocks.forEach((b, i) => {
      const x = 8 + i * colW;
      const cw2 = Math.max(6, colW * 0.34);
      const hF = b.tier > 0 ? barH * clamp(b.fc / scale, 0.02, 1) : 0;
      const hR = barH * clamp(b.rc / scale, 0.02, 1);
      // fetch bar
      if (b.tier > 0) {
        ctx.fillStyle = alphaOf(T.violet, b.action === 'fetch' ? 0.9 : 0.28);
        ctx.fillRect(x + 2, barTop + barH - hF, cw2, hF);
      } else if (b.tier < 0) {
        hatch(ctx, x + 2, barTop + barH - 6, cw2, 6, alphaOf(T.n9, 0.8), null);
      } else {
        ctx.fillStyle = alphaOf(T.ok, 0.85);
        ctx.fillRect(x + 2, barTop + barH - 5, cw2, 5);
      }
      // recompute bar
      ctx.fillStyle = alphaOf(T.warn, b.action === 'recompute' ? 0.9 : 0.28);
      ctx.fillRect(x + 6 + cw2, barTop + barH - hR, cw2, hR);
      r.label(`b${b.j}`, x + 2, barTop + barH + 10, { color: T.n11, font: mono(9) });
      r.label(b.tier >= 0 ? TIER_SHORT[b.tier] : b.age == null ? 'new' : 'gone', x + 2, barTop + barH + 19, {
        color: b.tier < 0 ? T.bad : b.tier === 0 ? T.ok : b.tier === 1 ? T.accent : T.teal, font: mono(9),
      });
      const win = b.action === 'resident' ? 'free' : b.action === 'fetch' ? `fetch ${fmtMs(b.fc)}` : `recomp ${fmtMs(b.rc)}`;
      r.label(win, x + 2, barTop - 5, {
        color: b.action === 'fetch' ? T.violetDeep : b.action === 'recompute' ? T.warnDeep : T.ok, font: mono(8.5),
      });
      verdictRects.push({ R: { x, y: barTop - 14, w: colW, h: barH + 36 }, b });
    });

    // ================= engine occupancy timeline ============================
    // The honest cost of a deeper tier: fetches and spills SHARE one link, and
    // recomputes SHARE one compute unit. A policy can be beaten by its own
    // queue even when each individual decision was right.
    const tlY = vY + vH + 8;
    const tlH = Math.max(52, H * 0.16);
    const span = Math.max(1, active.span);
    const laneX = 80;
    const t2x = (t) => laneX + (W - laneX - 10) * (t / span);
    const laneNames = [['compute', 'compute unit', T.warn], ['link1', 'host link', T.accent], ['link2', 'SSD link', T.teal]];
    const laneH = (tlH - 14) / 3;
    const segRects = [];
    laneNames.forEach(([k, lab, tone], li) => {
      const y = tlY + 12 + li * laneH;
      r.label(lab, laneX - 5, y + laneH * 0.62, { color: tone, font: mono(9), align: 'right' });
      ctx.fillStyle = rgbaToken('n14', 0.05);
      ctx.fillRect(laneX, y, W - laneX - 10, laneH - 3);
      for (const e of active.lanes[k]) {
        const x0 = t2x(e.t0), x1 = Math.max(x0 + 1, t2x(e.t1));
        const fill = e.kind === 'spill' ? T.n9 : e.kind === 'fetch' ? T.violet : T.warn;
        ctx.save();
        if (e.req > now) ctx.globalAlpha = 0.16;
        if (e.kind === 'spill') hatch(ctx, x0, y, x1 - x0, laneH - 3, alphaOf(fill, 0.85), alphaOf(fill, 0.12));
        else { ctx.fillStyle = alphaOf(fill, e.req === now ? 0.95 : 0.55); ctx.fillRect(x0, y, x1 - x0, laneH - 3); }
        ctx.restore();
        segRects.push({ R: { x: x0, y, w: Math.max(2, x1 - x0), h: laneH - 3 }, e, lane: lab });
      }
      r.label(`${(active.busy[k] * 100).toFixed(0)}% busy`, W - 12, y + laneH * 0.62, { color: T.n11, font: mono(8.5), align: 'right' });
    });
    // the playhead: where this request sits on the shared time axis
    ctx.save(); ctx.strokeStyle = T.n13; ctx.lineWidth = 1.2;
    ctx.beginPath(); ctx.moveTo(t2x(F.ready), tlY + 10); ctx.lineTo(t2x(F.ready), tlY + tlH - 2); ctx.stroke(); ctx.restore();
    r.label(`who is busy, over the whole ${span.toFixed(0)} ms run — a fetch and a spill both occupy the SAME link`,
      laneX, tlY + 8, { color: T.n11, font: mono(9.5) });

    // ================= policy comparison ====================================
    const pY = tlY + tlH + 8;
    const pH = Math.max(40, H - pY - 8);
    const rows = [
      ['GPU-only caching (baseline)', base.mean, T.n9, 'base'],
      ['always fetch', alt.fetch.mean, T.violet, 'fetch'],
      ['always recompute', alt.recompute.mean, T.warn, 'recompute'],
      ['whichever is cheaper', alt.cheaper.mean, T.ok, 'cheaper'],
    ];
    const pMax = rows.reduce((a, x) => Math.max(a, x[1]), 0.001);
    r.label('mean request latency, as a percent of the GPU-only baseline (lower is better; 100% = parity)',
      180, pY + 8, { color: T.n11, font: mono(9.5) });
    const prH = (pH - 18) / rows.length;
    const policyRects = [];
    rows.forEach(([lab, v, tone, id], i) => {
      const y = pY + 14 + i * prH;
      const on = id === st.policy;
      // Leave room for the number AFTER the bar: a bar allowed to run to the
      // right edge pushes its own label off the canvas.
      const barMax = Math.max(40, W - 180 - 190);
      const bwid = barMax * clamp(v / pMax, 0, 1);
      ctx.fillStyle = alphaOf(tone, on ? 0.9 : 0.4);
      ctx.fillRect(180, y, Math.max(2, bwid), prH - 4);
      r.label((on ? '▶ ' : '  ') + lab, 174, y + prH * 0.6, { color: on ? T.n14 : T.n11, font: mono(9.5), align: 'right' });
      const rel = base.mean > 0 ? (100 * v / base.mean) : 100;
      r.label(`${fmtMs(v)} ms · ${rel.toFixed(0)}% of baseline`,
        184 + Math.max(2, bwid), y + prH * 0.6, { color: T.n11, font: mono(9) });
      policyRects.push({ R: { x: 60, y, w: W - 70, h: prH - 4 }, id, lab, v, rel });
    });

    // ---- hit-test geometry --------------------------------------------------
    geom = { capStrips, pipes, blockRects, verdictRects, segRects, policyRects, rateHandle, chart: { x: chX, y: chY, w: chW, h: chH } };

    // ---- hover-to-inspect ---------------------------------------------------
    if (page.pointer.over && !drag) {
      const hx = page.pointer.x, hy = page.pointer.y;
      const inR = (R, pad = 0) => R && hx >= R.x - pad && hx <= R.x + R.w + pad && hy >= R.y - pad && hy <= R.y + R.h + pad;
      let tip = null;
      for (const v of verdictRects) if (inR(v.R)) { tip = blockTip(st, v.b, bytes); break; }
      if (!tip) {
        for (const b of blockRects) {
          if (!inR(b.R, 1)) continue;
          const [conv, j] = b.key.split(':').map(Number);
          const pos = j * st.blk;
          const rcx = recomputeMs(st, st.blk, pos);
          const fcx = b.tier > 0 ? fetchMs(bytes, b.tier === 1 ? st.bw1 : st.bw2) : 0;
          const here = inThisReq.get(b.key);
          tip = [
            `conversation ${conv} · block ${j} — ${st.blk} tokens at sequence offset ${pos}`,
            `tier: ${TIER_NAME[b.tier]} · size ${fmtMB(bytes)}`,
            here && here.age != null ? `age: last used ${here.age} request(s) ago` : 'age: in use this request',
            `if it were wanted from here: fetch ${b.tier > 0 ? fmtMs(fcx) + ' ms' : 'not needed, it is resident'} vs recompute ${fmtMs(rcx)} ms`,
            b.tier > 0 ? (fcx <= rcx ? '→ fetching is cheaper at these settings' : '→ rebuilding it is cheaper at these settings') : '→ a top-tier hit costs nothing at all',
          ].join('\n');
          break;
        }
      }
      if (!tip) for (const p of pipes) if (inR(p.R, 12)) {
        const bwv = p.tier === 1 ? +st.bw1 : +st.bw2;
        const busy = p.tier === 1 ? active.busy.link1 : active.busy.link2;
        tip = [
          `${p.tier === 1 ? 'host link' : 'SSD read path'}: ${bwv} GB/s`,
          `one ${st.blk}-token block = ${fmtMB(bytes)} → ${fmtMs(fetchMs(bytes, bwv))} ms on this link`,
          `busy ${(busy * 100).toFixed(0)}% of the run — fetches AND spills both queue here`,
          'a deeper tier is not free even on a hit: this is the occupancy it costs',
          '↕ drag to widen or throttle it',
        ].join('\n');
        break;
      }
      if (!tip) for (const c of capStrips) if (inR(c.R, 5)) {
        const capN = [st.cap0 | 0, st.cap1 | 0, st.cap2 | 0][c.tier];
        tip = [
          `${TIER_NAME[c.tier]}: ${capN} blocks = ${fmtMB(capN * bytes)}`,
          `holding ${(F.tiers[c.tier] || []).length} block(s) right now`,
          c.tier === 0 ? 'a hit here is free; everything pushed out of it lands in the tier below'
            : c.tier === 2 ? 'the bottom: what gets pushed out of here is destroyed, and can only be recomputed'
              : 'overflow from here spills further down, and that spill is a transfer too',
          '◀ drag the edge to resize ▶',
        ].join('\n');
        break;
      }
      if (!tip && geom.rateHandle && inR(geom.rateHandle, 12)) {
        tip = [
          `arithmetic rate: ${st.rate} TFLOP/s`,
          `one ${st.blk}-token block of a ${st.params}B model costs ${fmtMs(rc0)} ms to rebuild at offset 0,`,
          `and ${fmtMs(rcD)} ms at offset ${deepPos} — attention against everything before it is the difference`,
          '↕ drag: a faster machine makes recompute cheap, which moves every crossing',
        ].join('\n');
      }
      if (!tip) for (const sg of segRects) if (inR(sg.R, 1)) {
        tip = [
          `${sg.lane}: ${sg.e.kind === 'spill' ? 'a SPILL — a block pushed down a tier' : sg.e.kind === 'fetch' ? 'a fetch up the hierarchy' : 'a recompute'}`,
          `block ${sg.e.key.replace(':', ' · block ')} · request ${sg.e.req + 1}`,
          `occupied it for ${fmtMs(sg.e.t1 - sg.e.t0)} ms (${sg.e.t0.toFixed(1)} → ${sg.e.t1.toFixed(1)} ms)`,
          sg.e.kind === 'spill' ? 'nobody asked for this — it is the cost of making room' : 'this is on some request\'s critical path',
        ].join('\n');
        break;
      }
      if (!tip) for (const p of policyRects) if (inR(p.R)) {
        tip = [
          `${p.lab}: mean request latency ${fmtMs(p.v)} ms`,
          `= ${p.rel.toFixed(1)}% of GPU-only caching (lower is better; 100% = parity)`,
          p.id === 'base' ? 'the baseline: same top tier, nothing below it, so every evicted block is recomputed'
            : p.id === 'fetch' ? 'never rebuilds — pays bandwidth even when the block is cheap to recompute'
              : p.id === 'recompute' ? 'never reads a lower tier — the links stay idle and the compute unit queues'
                : 'prices both per block and takes the smaller — which is not automatically the fastest, because it can pile every decision onto one engine',
        ].join('\n');
        break;
      }
      if (!tip && hx > geom.chart.x && hy > geom.chart.y && hy < geom.chart.y + geom.chart.h) {
        const be1 = breakEvenL(st, st.bw1, 0), be2 = breakEvenL(st, st.bw2, 0);
        tip = [
          'both costs, as a function of block size, at these settings:',
          `  fetch  = L × ${st.kvkb} KB ÷ bandwidth        (a straight line: bytes scale with tokens)`,
          `  recomp = L × (2 × ${st.params}B params + attention over everything before it) ÷ ${st.rate} TFLOP/s`,
          'recompute bends upward because a block of L tokens must attend L×(pos + L/2) pairs.',
          `break-even block size vs the host link: ${be1 > 0 ? Math.round(be1) + ' tokens' : 'no crossing — fetching wins at every size'}`,
          `break-even block size vs the SSD:       ${be2 > 0 ? Math.round(be2) + ' tokens' : 'no crossing — fetching wins at every size'}`,
        ].join('\n');
      }
      if (tip) page.setTip(tip);
    }

    // ---- readout ------------------------------------------------------------
    const be1 = breakEvenL(st, st.bw1, 0), be2 = breakEvenL(st, st.bw2, 0);
    const lookups = active.hits[0] + active.hits[1] + active.hits[2] + active.cold.fresh + active.cold.dropped;
    const hitPct = (i) => (lookups ? (100 * active.hits[i] / lookups).toFixed(0) : '0');
    const pctOf = (v) => (lookups ? (100 * v / lookups).toFixed(0) : '0');
    let o = `ONE ${st.blk}-token block = ${st.blk} × ${st.kvkb} KB = ${fmtMB(bytes)}.   fetch = bytes ÷ bandwidth · recompute = ${st.params}B params + attention, at ${st.rate} TFLOP/s.    tier:${r.name}\n`;
    o += `  host DRAM  fetch ${fmtMs(fc1)} ms  vs  recompute ${fmtMs(rc0)} ms at offset 0 / ${fmtMs(rcD)} ms at offset ${deepPos}  →  ${fc1 <= rc0 ? 'FETCH wins' : 'RECOMPUTE wins'} at the front of the sequence, ${fc1 <= rcD ? 'FETCH' : 'RECOMPUTE'} deep into it\n`;
    o += `  local SSD  fetch ${fmtMs(fc2)} ms  vs  the same recompute            →  ${fc2 <= rc0 ? 'FETCH wins' : 'RECOMPUTE wins'} at the front of the sequence, ${fc2 <= rcD ? 'FETCH' : 'RECOMPUTE'} deep into it\n`;
    const beTxt = (be) => (be > 0 ? Math.round(be) + ' tokens' : 'no crossing — fetching is cheaper at EVERY block size');
    o += `BREAK-EVEN BLOCK SIZE (above it, cheaper to fetch; below it, cheaper to rebuild): `
      + `host link ${beTxt(be1)} · SSD ${beTxt(be2)} (both at sequence offset 0; the crossing moves LEFT as the block sits deeper, because attention makes a late block dearer to rebuild)\n`;
    o += `Lookups: ${hitPct(0)}% already in the top tier (free), ${hitPct(1)}% in DRAM, ${hitPct(2)}% on the SSD, ${pctOf(active.cold.fresh)}% brand new (never cached yet — unavoidable), ${pctOf(active.cold.dropped)}% destroyed off the bottom and rebuilt from scratch. `
      + `${fmtMB(active.movedBytes)} crossed the links, which were busy ${(active.busy.link1 * 100).toFixed(0)}% (host) and ${(active.busy.link2 * 100).toFixed(0)}% (SSD) of the run; the compute unit ${(active.busy.compute * 100).toFixed(0)}%.\n`;
    o += `POLICY "${st.policy === 'cheaper' ? 'whichever is cheaper' : st.policy === 'fetch' ? 'always fetch' : 'always recompute'}": mean request latency ${fmtMs(active.mean)} ms, worst ${fmtMs(active.worst)} ms — `
      + `${pct.toFixed(0)}% of GPU-only caching's ${fmtMs(base.mean)} ms (lower is better; 100% = parity). `;
    o += pct < 100
      ? `Tiering is ahead here by ${(100 - pct).toFixed(0)} percentage points, and it bought that with ${fmtMB(active.movedBytes)} of link traffic — not for free.`
      : `Tiering is BEHIND the baseline here: keeping the blocks alive downhill costs more link time than simply rebuilding them. Widen a link, shrink the blocks, or hand the arithmetic more headroom.`;
    o += `  (always fetch ${fmtMs(alt.fetch.mean)} ms · always recompute ${fmtMs(alt.recompute.mean)} ms · cheaper-of-the-two ${fmtMs(alt.cheaper.mean)} ms.)`;
    page.setReadout(o);
  },
}).then((page) => {
  window.__kvTieringPage = page;
  const q = new URLSearchParams(location.search);
  const t = page.controls._transport;
  const num = (k, lo, hi, int) => {
    if (!q.has(k)) return;
    const v = int ? parseInt(q.get(k), 10) : parseFloat(q.get(k));
    if (Number.isNaN(v)) return;
    page.controls.set(k, clamp(v, lo, hi), { rebuild: true, silent: true });
  };
  if (q.has('policy') && ['cheaper', 'fetch', 'recompute'].includes(q.get('policy'))) {
    page.controls.set('policy', q.get('policy'), { rebuild: true, silent: true });
  }
  num('cap0', 2, 48, true);
  num('cap1', 2, 160, true);
  num('cap2', 4, 512, true);
  num('bw1', 1, 128, false);
  num('bw2', 0.2, 16, false);
  num('rate', 5, 1000, false);
  num('params', 1, 70, false);
  num('kvkb', 8, 512, true);
  num('blk', 32, 1024, true);
  num('nreq', 6, 60, true);
  num('gap', 1, 60, true);
  num('seed', 0, 99, true);
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
