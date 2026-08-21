// continuous-batching concept page -- how a serving runtime schedules a batch
// of generation requests, and why it refills a slot the instant a sequence
// stops instead of waiting for the whole batch.
//
// Two schedulers run the SAME workload (same arrivals, same output lengths) and
// are drawn as two Gantt charts sharing one x axis (x = decode step, y = batch
// slot, one coloured bar per request):
//
//   STATIC batching     -- a batch is formed once, and every slot in it is held
//                          until the LONGEST sequence in that batch finishes.
//                          A sequence that stopped early keeps its slot,
//                          emitting padding: the hatched dead area.
//   CONTINUOUS batching -- the scheduler re-admits every step. The moment a
//                          sequence emits its stop token, a waiting request
//                          takes that slot mid-flight, so the queue drains
//                          sooner and the dead area mostly disappears.
//
// Interactive per the shared render framework's contract:
//  - TRANSPORT scrubs the decode step; both schedulers advance together, so the
//    step at which they diverge is visible. Auto-plays and loops.
//  - DIRECT MANIPULATION: drag any request bar horizontally to change that
//    request's OUTPUT LENGTH. Both schedules re-flow and every counter
//    recomputes under your hand. The lesson lands here: stretching one sequence
//    is catastrophic for static batching (every other slot in its batch pads
//    along with it) and nearly free for continuous batching.
//  - HOVER a bar for its arrival / wait / output length / the steps it occupied;
//    hover a hatched cell for WHY that slot is idle (which sequence it is
//    waiting on).
//  - STEPPERS resize the problem: batch slots, number of requests, arrival gap.
//  - URL hooks reproduce every view headlessly: ?step, ?slots, ?nreq, ?gap,
//    ?seed, ?lens (per-request output lengths, "id:len,id:len"), ?hover=x,y.
import { mount } from '../framework/layout.js';
import { categorical } from '../framework/render.js';
import { T, alphaOf, inkOn, rgbaToken } from '../framework/theme.js';

const MIN_LEN = 1, MAX_LEN = 20;
const clamp = (v, lo, hi) => (v < lo ? lo : v > hi ? hi : v);
const rgb = (c, a = 1) => alphaOf(c, a);

// Deterministic per-request output length: a served workload has a wide,
// unpredictable spread of output lengths, which is exactly what makes a
// fixed-batch schedule waste slots. Seeded so every reload shows one picture.
function hash32(a, b) {
  let x = (Math.imul(a + 1, 374761393) + Math.imul(b + 1, 668265263)) >>> 0;
  x = (x ^ (x >>> 13)) >>> 0;
  x = Math.imul(x, 1274126177) >>> 0;
  return (x ^ (x >>> 16)) >>> 0;
}

function parseLens(s) {
  const out = {};
  String(s || '').split(',').forEach((part) => {
    const m = /^\s*(\d+)\s*:\s*(\d+)\s*$/.exec(part);
    if (m) out[+m[1]] = clamp(+m[2], MIN_LEN, MAX_LEN);
  });
  return out;
}
const formatLens = (o) => Object.keys(o).map(Number).sort((a, b) => a - b).map((k) => `${k}:${o[k]}`).join(',');

// The workload: R requests, each with an arrival step and an output length.
// The first `slots` requests are already queued at step 0 (they fill the first
// batch); the rest trickle in every `gap` steps.
function buildRequests(st) {
  const R = st.nreq | 0, S = st.slots | 0, gap = st.gap | 0, seed = st.seed | 0;
  const over = parseLens(st.lens);
  const reqs = [];
  for (let i = 0; i < R; i++) {
    const len = over[i] != null ? over[i] : 3 + (hash32(seed, i) % 10);
    const arrival = i < S ? 0 : (i - S + 1) * gap;
    reqs.push({ id: i, len, arrival });
  }
  return reqs;
}

// ---- schedulers ---------------------------------------------------------
// STATIC: fill a batch from whoever is queued, run it to the longest member,
// only then form the next batch. Nobody joins a batch in flight.
function simStatic(reqs, S) {
  const q = reqs.slice().sort((a, b) => a.arrival - b.arrival || a.id - b.id);
  let t = 0, qi = 0;
  const rows = [], batches = [];
  while (qi < q.length) {
    if (q[qi].arrival > t) t = q[qi].arrival;
    const members = [];
    while (members.length < S && qi < q.length && q[qi].arrival <= t) members.push(q[qi++]);
    let dur = 0, strag = members[0];
    for (const m of members) if (m.len > dur) { dur = m.len; strag = m; }
    members.forEach((m, j) => rows.push({ ...m, slot: j, start: t, end: t + m.len - 1 }));
    batches.push({ t0: t, dur, n: members.length, straggler: strag.id, straggLen: dur });
    t += dur;
  }
  return { rows, batches, scheme: 'static' };
}

// CONTINUOUS: every step, any free slot takes the head of the queue. A finished
// sequence releases its slot at the end of the step it stops on.
function simCont(reqs, S) {
  const q = reqs.slice().sort((a, b) => a.arrival - b.arrival || a.id - b.id);
  let t = 0, qi = 0, guard = 0;
  const busy = new Array(S).fill(null), rows = [];
  const anyBusy = () => busy.some(Boolean);
  while ((qi < q.length || anyBusy()) && guard++ < 20000) {
    for (let j = 0; j < S; j++) {
      if (!busy[j] && qi < q.length && q[qi].arrival <= t) {
        const rec = { ...q[qi++], slot: j, start: t, end: t + 0 };
        rec.end = rec.start + rec.len - 1;
        busy[j] = rec; rows.push(rec);
      }
    }
    if (!anyBusy()) {                       // engine drained; jump to next arrival
      if (qi < q.length) { t = q[qi].arrival; continue; }
      break;
    }
    t++;
    for (let j = 0; j < S; j++) if (busy[j] && busy[j].end < t) busy[j] = null;
  }
  return { rows, batches: null, scheme: 'continuous' };
}

// Occupancy grid + the reason each idle cell is idle (the hover derivation).
function gridFor(sim, reqs, S, span) {
  const g = Array.from({ length: S }, () => new Array(span).fill(null));
  for (const r of sim.rows) for (let t = r.start; t <= r.end && t < span; t++) g[r.slot][t] = { r };
  const nextArrival = (t) => { let a = null; for (const q of reqs) if (q.arrival > t && (a == null || q.arrival < a)) a = q.arrival; return a; };
  for (let j = 0; j < S; j++) {
    for (let t = 0; t < span; t++) {
      if (g[j][t]) continue;
      if (sim.scheme === 'static') {
        const b = sim.batches.find((x) => t >= x.t0 && t < x.t0 + x.dur);
        if (b && j < b.n) g[j][t] = { idle: true, waste: true, why: `slot idle: waiting for sequence ${b.straggler} to finish\nits ${b.straggLen} output tokens set this batch's length, so every\nshorter sequence holds its slot and emits padding` };
        else if (b) g[j][t] = { idle: true, waste: true, why: `slot idle: the batch was formed with only ${b.n} of ${S} slots filled\nnothing may join a running batch, so this slot stays empty\nuntil sequence ${b.straggler} finishes the batch` };
        else { const a = nextArrival(t); g[j][t] = { idle: true, waste: false, why: a == null ? 'slot idle: the queue is empty — all requests are done' : `slot idle: nothing has arrived yet (next arrival: step ${a})` }; }
      } else {
        const a = nextArrival(t);
        g[j][t] = { idle: true, waste: false, why: a == null ? 'slot idle: the queue is empty — all requests are done' : `slot idle: no request is waiting (next arrival: step ${a})` };
      }
    }
  }
  return g;
}

function metrics(sim, S, reqs) {
  let span = 0, busy = 0, wait = 0, lat = 0, worst = 0;
  for (const r of sim.rows) {
    span = Math.max(span, r.end + 1);
    busy += r.len;
    wait += r.start - r.arrival;
    const l = r.end + 1 - r.arrival;
    lat += l; worst = Math.max(worst, l);
  }
  const n = Math.max(1, sim.rows.length);
  const cap = Math.max(1, S * span);
  return { span, busy, idle: cap - busy, util: busy / cap, avgWait: wait / n, avgLat: lat / n, worst, n: sim.rows.length };
}

// Waiting-in-queue depth at each step (arrived but not yet admitted).
function queueDepth(sim, reqs, span) {
  const startOf = {}; for (const r of sim.rows) startOf[r.id] = r.start;
  const d = new Array(span).fill(0);
  for (let t = 0; t < span; t++) for (const q of reqs) if (q.arrival <= t && startOf[q.id] > t) d[t]++;
  return d;
}

// ---- drawing helpers ----------------------------------------------------
function roundRect(ctx, x, y, w, h, r) {
  const rr = Math.max(0, Math.min(r, w / 2, h / 2));
  ctx.beginPath(); ctx.moveTo(x + rr, y);
  ctx.arcTo(x + w, y, x + w, y + h, rr); ctx.arcTo(x + w, y + h, x, y + h, rr);
  ctx.arcTo(x, y + h, x, y, rr); ctx.arcTo(x, y, x + w, y, rr); ctx.closePath();
}
// Diagonal hatch = "dead slot-step": the slot is held but produces nothing.
function hatch(ctx, x, y, w, h, color, bg) {
  ctx.save();
  ctx.beginPath(); ctx.rect(x, y, w, h); ctx.clip();
  ctx.fillStyle = bg; ctx.fillRect(x, y, w, h);
  ctx.strokeStyle = color; ctx.lineWidth = 1;
  for (let k = -h; k < w; k += 5) { ctx.beginPath(); ctx.moveTo(x + k, y + h); ctx.lineTo(x + k + h, y); ctx.stroke(); }
  ctx.restore();
}

let cur = null;      // {reqs, S, span, A:{sim,grid,m,q}, B:{...}}
let geom = null;     // hit-test geometry captured each draw
let drag = null;     // {id, x0, len0} while a bar is being stretched

function rebuild(st) {
  const reqs = buildRequests(st), S = st.slots | 0;
  const A = simStatic(reqs, S), B = simCont(reqs, S);
  const mA = metrics(A, S, reqs), mB = metrics(B, S, reqs);
  const span = Math.max(mA.span, mB.span, 1);
  cur = {
    reqs, S, span,
    A: { sim: A, m: mA, grid: gridFor(A, reqs, S, span), q: queueDepth(A, reqs, span), title: 'STATIC batching — the batch runs until its LONGEST sequence finishes' },
    B: { sim: B, m: mB, grid: gridFor(B, reqs, S, span), q: queueDepth(B, reqs, span), title: 'CONTINUOUS batching — a freed slot takes the next queued request immediately' },
  };
  return Array.from({ length: span }, (_, i) => ({ t: i, label: `decode step ${i + 1} / ${span}` }));
}

function setLen(page, id, len) {
  const over = parseLens(page.state.lens);
  over[id] = clamp(len | 0, MIN_LEN, MAX_LEN);
  page.controls.set('lens', formatLens(over), { rebuild: true });
}

mount({
  mount: 'body',
  title: 'continuous-batching — refill the slot the moment a sequence stops',
  blurb: 'One workload, two schedulers, one shared x axis (x = decode step, y = batch slot). STATIC batching forms a batch once and holds every slot until the LONGEST sequence in it finishes — the hatched cells are slot-steps the runtime paid for and got nothing back. CONTINUOUS batching re-admits every step, so a request waiting in the queue takes a slot the instant its predecessor emits a stop token, and the queue drains sooner. DRAG any bar sideways to change that request\'s output length and watch both schedules and every counter re-flow: one long sequence is ruinous on the left and nearly free on the right. Hover a bar for its wait and latency, or a hatched cell for what it is waiting on.',
  prefer: 'canvas2d',
  aspect: '16 / 10',
  autoplay: true,
  compare: { key: 'slots', a: 2, b: 6, labelA: '2 batch slots', labelB: '6 batch slots', rebuild: true },
  challenges: [
    {
      goal: 'Starve static batching: leave it wasting more than a third of its slot-steps.',
      hint: 'drag one request bar far to the right — a single long sequence pins every other slot in its batch.',
      check: (api) => ({ solved: (api.probe.utilA ?? 1) < 0.67, detail: `static utilisation ${(100 * (api.probe.utilA ?? 1)).toFixed(0)}% — needs < 67%` }),
    },
    {
      goal: 'Find a workload where continuous batching drains the queue in at least 20% fewer steps.',
      hint: 'a wide spread of output lengths (drag a couple of bars apart) is what static batching cannot absorb.',
      check: (api) => ({ solved: (api.probe.spanB ?? 1) <= 0.8 * (api.probe.spanA ?? 1), detail: `${api.probe.spanB ?? '–'} steps vs ${api.probe.spanA ?? '–'} — needs ≤ ${(0.8 * (api.probe.spanA ?? 0)).toFixed(1)}` }),
    },
  ],
  controls: (c, page) => {
    c.stepper('slots', { label: 'batch slots', min: 2, max: 6, value: 3 });
    c.stepper('nreq', { label: 'requests', min: 3, max: 10, value: 7 });
    c.stepper('gap', { label: 'arrival gap (steps)', min: 0, max: 4, value: 1 });
    c.slider('seed', { label: 'workload seed', min: 0, max: 99, step: 1, value: 3, rebuild: true });
    c.text('lens', { label: 'output lengths', value: '', placeholder: 'id:len,id:len', rebuild: true });
    c.transport({ compute: () => rebuild(page.state), speed: 6, loop: true });
  },
  // Direct manipulation: grab a request's bar in either timeline and drag
  // horizontally; the x displacement is read in decode steps and becomes that
  // request's new output length. Both schedules rebuild on every pixel.
  onPointer: (page, ev) => {
    if (!geom || !cur) return;
    if (ev.type === 'down') {
      const hit = geom.hit(ev.x, ev.y);
      drag = hit && hit.cell && hit.cell.r ? { id: hit.cell.r.id, x0: ev.x, len0: hit.cell.r.len } : null;
    } else if (ev.type === 'up' || ev.type === 'leave') {
      drag = null;
    } else if (ev.type === 'move' && drag && page.pointer.down) {
      const want = Math.round(drag.len0 + (ev.x - drag.x0) / geom.cellW);
      const now = parseLens(page.state.lens)[drag.id];
      if (want !== now) setLen(page, drag.id, want);
    }
  },
  draw: (page) => {
    const r = page.renderer, ctx = page.ctx, st = page.state;
    if (!cur) rebuild(st);
    const { reqs, S, span, A, B } = cur;
    r.clear(T.n0);
    const s = page.step();
    const now = s ? s.t : span - 1;          // current decode step (0-based)
    page.probe = { utilA: A.m.util, utilB: B.m.util, spanA: A.m.span, spanB: B.m.span };

    const W = page.W, H = page.H, padL = 74, padR = 14, top = 40;
    const gridW = Math.max(40, W - padL - padR);
    const cellW = gridW / span;
    const stripH = Math.min(reqs.length * 13 + 30, H * 0.3);
    const panelH = (H - top - stripH - 10) / 2;
    const cellH = Math.max(7, Math.min(26, (panelH - 40) / S));

    // ---- legend / shared axis -------------------------------------------
    r.label(`workload: ${reqs.length} requests · ${S} batch slots · decode step ${now + 1} / ${span}`, padL, 18, { color: T.n14, font: '13px ui-monospace, monospace' });
    r.label('▨ = dead slot-step (paid for, produced nothing)', padL, 32, { color: T.warnDeep, font: '10px ui-monospace, monospace' });
    r.label('↔ drag a bar to restretch its output length', W - padR, 32, { color: T.accent, font: '10px ui-monospace, monospace', align: 'right' });

    const panels = [];
    const drawPanel = (P, y0, accent) => {
      const gy = y0 + 30, gh = S * cellH;
      panels.push({ P, gy, gh });
      r.label(P.title, padL, y0 + 12, { color: accent, font: '12px ui-monospace, monospace' });
      r.label(`${P.m.span} steps to drain · ${(100 * P.m.util).toFixed(0)}% slot utilisation · ${P.m.idle} dead slot-steps · avg wait ${P.m.avgWait.toFixed(1)} · avg latency ${P.m.avgLat.toFixed(1)}`,
        padL, y0 + 26, { color: T.n11, font: '11px ui-monospace, monospace' });

      // faint every-5-steps rules, so the shared x axis is readable as steps
      ctx.save();
      ctx.strokeStyle = rgbaToken('n14', 0.1); ctx.lineWidth = 1;
      for (let t = 0; t <= span; t += 5) { ctx.beginPath(); ctx.moveTo(padL + t * cellW, gy); ctx.lineTo(padL + t * cellW, gy + gh); ctx.stroke(); }
      ctx.restore();

      for (let j = 0; j < S; j++) {
        r.label(`slot ${j}`, padL - 8, gy + j * cellH + cellH / 2 + 3, { color: T.n10, font: '10px ui-monospace, monospace', align: 'right' });
        for (let t = 0; t < span; t++) {
          const c = P.grid[j][t], x = padL + t * cellW, y = gy + j * cellH;
          const future = t > now;
          ctx.save();
          if (future) ctx.globalAlpha = 0.16;
          if (c && c.r) {
            const col = categorical(c.r.id);
            const first = t === c.r.start, last = t === c.r.end;
            ctx.fillStyle = rgb(col, drag && drag.id === c.r.id ? 1 : 0.85);
            ctx.fillRect(x, y + 1.5, cellW + 0.5, cellH - 3);
            if (first) { ctx.fillStyle = rgb(col, 1); ctx.fillRect(x, y + 1.5, Math.min(2.5, cellW), cellH - 3); }
            if (last) { ctx.fillStyle = T.n0; ctx.fillRect(x + cellW - 2, y + 1.5, 2, cellH - 3); }   // stop token
            if (first && cellW * c.r.len > 26 && cellH > 11) {
              ctx.fillStyle = inkOn(col); ctx.font = '10px ui-monospace, monospace'; ctx.textAlign = 'left'; ctx.textBaseline = 'middle';
              ctx.fillText(`r${c.r.id}`, x + 3, y + cellH / 2);
            }
          } else if (c && c.waste) {
            hatch(ctx, x, y + 1.5, cellW + 0.5, cellH - 3, alphaOf(T.warn, 0.55), alphaOf(T.warn, 0.1));
          } else {
            ctx.fillStyle = rgbaToken('n14', 0.045); ctx.fillRect(x, y + 1.5, cellW + 0.5, cellH - 3);
          }
          ctx.restore();
        }
      }
      // queue depth under the grid: how many arrived requests are still waiting.
      const qy = gy + gh + 3, qh = 9;
      r.label('queue', padL - 8, qy + qh, { color: T.n10, font: '10px ui-monospace, monospace', align: 'right' });
      const qmax = Math.max(1, ...P.q);
      for (let t = 0; t < span; t++) {
        const h = (P.q[t] / qmax) * qh;
        ctx.save(); if (t > now) ctx.globalAlpha = 0.16;
        ctx.fillStyle = P.q[t] ? alphaOf(T.violet, 0.75) : rgbaToken('n14', 0.06);
        ctx.fillRect(padL + t * cellW, qy + qh - Math.max(P.q[t] ? 1.5 : 1, h), cellW + 0.5, Math.max(P.q[t] ? 1.5 : 1, h));
        ctx.restore();
      }
      // playhead
      ctx.save();
      ctx.strokeStyle = T.n13; ctx.lineWidth = 1.5;
      ctx.beginPath(); ctx.moveTo(padL + (now + 1) * cellW, y0 + 30 - 4); ctx.lineTo(padL + (now + 1) * cellW, qy + qh); ctx.stroke();
      ctx.restore();
    };

    drawPanel(A, top, T.bad);
    drawPanel(B, top + panelH, T.ok);

    // shared x axis: both schedulers are drawn against the same decode steps,
    // which is what makes the point at which they diverge readable.
    const axY = top + 2 * panelH - 4;
    ctx.save();
    ctx.strokeStyle = rgbaToken('n14', 0.25); ctx.lineWidth = 1;
    ctx.beginPath(); ctx.moveTo(padL, axY - 8); ctx.lineTo(padL + gridW, axY - 8); ctx.stroke();
    ctx.restore();
    r.label('decode step →', padL + gridW, axY - 12, { color: T.n10, font: '10px ui-monospace, monospace', align: 'right' });
    const tick = Math.max(1, Math.ceil(span / 14));
    for (let t = 0; t < span; t += tick) r.label(String(t), padL + t * cellW + cellW / 2, axY, { color: T.n10, font: '9px ui-monospace, monospace', align: 'center' });

    // ---- per-request latency: wait (hatched) + generate (solid) ----------
    // The honest tradeoff lives here: the SOLID part is identical in both
    // schedulers -- continuous batching removes queueing delay, it does not
    // make any single sequence generate one token faster.
    const sy = top + 2 * panelH + 6, rowH = Math.min(13, (stripH - 24) / Math.max(1, reqs.length));
    r.label('per-request latency  ▨ wait in queue  ▮ generate (identical under both schedulers)', padL, sy + 8, { color: T.n11, font: '10px ui-monospace, monospace' });
    const startOf = (P, id) => P.sim.rows.find((x) => x.id === id);
    reqs.forEach((q, i) => {
      const y = sy + 16 + i * rowH;
      r.label(`r${q.id}`, padL - 8, y + rowH * 0.75, { color: T.n11, font: '10px ui-monospace, monospace', align: 'right' });
      [A, B].forEach((P, k) => {
        const rec = startOf(P, q.id); if (!rec) return;
        const yy = y + k * (rowH / 2), hh = Math.max(2, rowH / 2 - 1.5);
        const x0 = padL + q.arrival * cellW, xw = (rec.start - q.arrival) * cellW;
        if (xw > 0) hatch(ctx, x0, yy, xw, hh, alphaOf(k ? T.ok : T.bad, 0.6), alphaOf(k ? T.ok : T.bad, 0.12));
        ctx.fillStyle = rgb(categorical(q.id), 0.85);
        ctx.fillRect(padL + rec.start * cellW, yy, rec.len * cellW, hh);
      });
    });

    // ---- hit-test geometry (hover + the length drag) ---------------------
    geom = {
      cellW, cellH, padL, span, S,
      hit: (x, y) => {
        if (x < padL || x > padL + gridW) return null;
        const t = Math.floor((x - padL) / cellW);
        if (t < 0 || t >= span) return null;
        for (const pn of panels) {
          if (y >= pn.gy && y < pn.gy + pn.gh) {
            const j = Math.floor((y - pn.gy) / cellH);
            if (j >= 0 && j < S) return { P: pn.P, j, t, cell: pn.P.grid[j][t] };
          }
        }
        return null;
      },
    };

    // ---- hover-to-inspect -------------------------------------------------
    if (page.pointer.over && !drag) {
      const h = geom.hit(page.pointer.x, page.pointer.y);
      if (h && h.cell && h.cell.r) {
        const rec = h.P.sim.rows.find((x) => x.id === h.cell.r.id);
        const wait = rec.start - rec.arrival, lat = rec.end + 1 - rec.arrival;
        page.setTip(
          `request ${rec.id}  (${h.P.sim.scheme} scheduler)\n` +
          `arrived at step ${rec.arrival} · admitted at step ${rec.start} · stopped at step ${rec.end}\n` +
          `output length ${rec.len} tokens → occupies slot ${rec.slot}, steps ${rec.start}..${rec.end}\n` +
          `wait ${wait} + generate ${rec.len} = latency ${lat} steps\n` +
          `↔ drag to restretch this request's output length`);
      } else if (h && h.cell) {
        page.setTip(`${h.P.sim.scheme} scheduler · slot ${h.j}, step ${h.t}\n${h.cell.why}`);
      }
    }

    const dSpan = A.m.span - B.m.span, dWait = A.m.avgWait - B.m.avgWait;
    let o = `same ${reqs.length} requests, same output lengths, same ${S} slots — only the admission policy differs.    tier:${r.name}\n`;
    o += `STATIC     ${String(A.m.span).padStart(3)} steps to drain · ${(100 * A.m.util).toFixed(0)}% utilisation · ${A.m.idle} dead slot-steps · avg wait ${A.m.avgWait.toFixed(1)} · avg latency ${A.m.avgLat.toFixed(1)} · worst ${A.m.worst}\n`;
    o += `CONTINUOUS ${String(B.m.span).padStart(3)} steps to drain · ${(100 * B.m.util).toFixed(0)}% utilisation · ${B.m.idle} dead slot-steps · avg wait ${B.m.avgWait.toFixed(1)} · avg latency ${B.m.avgLat.toFixed(1)} · worst ${B.m.worst}\n`;
    o += dSpan > 0
      ? `Continuous drains the queue ${dSpan} step(s) sooner (${(100 * B.m.span / A.m.span).toFixed(0)}% of the static makespan — lower is better) and cuts average wait by ${dWait.toFixed(1)} steps. `
      : 'With these lengths the two schedules coincide — spread the output lengths apart (drag a bar) and the static schedule starts padding. ';
    o += 'The win is throughput and queueing delay only: each sequence still needs exactly its own output length of decode steps, so a single request in an empty engine is not one token faster.';
    page.setReadout(o);
  },
}).then((page) => {
  window.__cbPage = page;
  const q = new URLSearchParams(location.search);
  const t = page.controls._transport;
  // Restore the whole control state from the query string, so a link (or the
  // copy-link button) reproduces the exact workload: ?slots ?nreq ?gap ?seed
  // and ?lens=id:len,id:len (the headless stand-in for a bar drag).
  const restoreInt = (k, lo, hi) => { if (q.has(k)) page.controls.set(k, clamp(parseInt(q.get(k), 10) || lo, lo, hi), { rebuild: true, silent: true }); };
  restoreInt('slots', 2, 6);
  restoreInt('nreq', 3, 10);
  restoreInt('gap', 0, 4);
  restoreInt('seed', 0, 99);
  if (q.has('lens')) page.controls.set('lens', formatLens(parseLens(q.get('lens'))), { rebuild: true, silent: true });
  if (t) t.rebuild();
  // ?hover=x,y fakes the cursor (headless stand-in for a real hover, since a
  // screenshot run has no pointer). Canvas-space px.
  if (q.has('hover')) {
    const [hx, hy] = q.get('hover').split(',').map(Number);
    page.pointer.x = hx; page.pointer.y = hy; page.pointer.over = true;
  }
  // Deterministic frame for capture: pause before seeking so autoplay does not
  // advance off the requested step.
  if (q.has('step') || q.has('hover') || q.has('lens')) { if (t) t.pause(); }
  if (q.has('step') && t) t.seek(parseInt(q.get('step'), 10));
  if (q.get('play') === '1' && t) t.play();
  page.redraw();
});
