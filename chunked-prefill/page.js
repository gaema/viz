// chunked-prefill concept page -- how a scheduler stops one long prompt from
// stalling everybody else.
//
// The companion page `prefill-vs-decode` shows WHY the two regimes differ in
// shape: prefill is one big parallel pass over the whole prompt (compute-bound),
// decode is one skinny step per token that re-reads the KV cache
// (memory-bound). This page is the SCHEDULING consequence of exactly that.
//
// A serving engine runs one batched step at a time. If a newly-arrived long
// prompt's prefill is a single step, every other sequence's next token waits for
// the whole thing -- a visible hole in their token stream. CHUNKED PREFILL
// splits that prefill into pieces of at most a few tokens and packs each piece
// into a step alongside the waiting sequences' decode tokens, up to a per-step
// token budget.
//
// The honest trade, both halves shown live:
//   - the long prompt's own time-to-first-token gets WORSE (its prefill is now
//     spread over many steps, each paying the fixed per-step overhead again);
//   - everyone else's inter-token latency gets much BETTER (the longest hole in
//     their stream shrinks from "the whole prefill" to "one step").
//
// Interactive per the shared render framework's contract:
//  - TRANSPORT: play / step / scrub the scheduler steps; autoplays and loops.
//  - DIRECT MANIPULATION: drag the dashed token-budget line up and down. The
//    timeline re-packs under your hand and the two latency numbers move in
//    OPPOSITE directions -- that opposition is the whole lesson.
//  - HOVER a step -> exactly what it carried (whose prefill chunk, which
//    decodes, tokens used vs budget). Hover a gap in a token strip -> what
//    occupied the scheduler while that sequence waited.
//  - RESIZE the problem: prompt length and how many sequences are waiting.
//
// The millisecond figures come from a deliberately simple illustrative cost
// model -- a fixed per-step overhead plus a per-token cost -- so the SHAPE of
// the trade is readable. They are an illustration of the mechanism, not a
// measurement of any particular engine or accelerator.
import { mount } from '../framework/layout.js';
import { categorical } from '../framework/render.js';
import { T, alphaOf, rgbaToken, inkOn } from '../framework/theme.js';

// --- illustrative cost model -------------------------------------------------
// Every scheduler step pays a fixed cost (launch + scheduling + reading the
// weights) plus a marginal cost per token it carries. The fixed part is what
// makes a very small chunk expensive: halve the chunk size and you pay that
// overhead twice as often.
const OVERHEAD_MS = 1.8;   // fixed cost of running one scheduler step
const TOK_MS = 0.30;       // marginal cost of one more token in that step

const SEQ = ['A', 'B', 'C', 'D', 'E'];
// Sequence 0 is the long prompt being prefilled; 1..S are the sequences already
// decoding. Colour 0 is the framework's blue, close to the accent used for the
// prefill blocks, so the waiting sequences start at 1.
const seqColor = (i) => (i === 0 ? T.accent : alphaOf(categorical(i), 1));
const seqFill = (i, a) => (i === 0 ? alphaOf(T.accent, a) : alphaOf(categorical(i), a));

const clamp = (v, lo, hi) => (v < lo ? lo : v > hi ? hi : v);
const ms = (v) => `${v.toFixed(1)} ms`;
// A LOWER-is-better ratio: >100% of the baseline is WORSE, <100% is BETTER.
const pctOf = (v, b) => (b > 0 ? Math.round((v / b) * 100) : 100);
// Direction word + colour for a lower-is-better ratio (100% = parity, and
// parity is neither good nor bad -- it must not read green).
const dirWord = (p) => (p > 100 ? '▲ worse' : p < 100 ? '▼ better' : '= parity');
const dirColor = (p) => (p > 100 ? T.bad : p < 100 ? T.ok : T.n12);

// Build one schedule. `chunked=false` is the stall: the whole prefill is a
// single step and nothing else runs during it.
function makeSchedule(P, S, budget, chunked) {
  const steps = [];
  const add = (chunk, decIds, kind) => steps.push({ chunk, decIds, kind, idx: steps.length });
  // Two steady steps first, so the waiting sequences have a normal token
  // rhythm to be interrupted -- without them there is no "before" to compare.
  const waiting = Array.from({ length: S }, (_, i) => i + 1);
  const all = [0, ...waiting];
  for (let i = 0; i < 2; i++) add(0, waiting, 'steady');

  // Chunk cap: decodes are scheduled first (one token each), the rest of the
  // budget goes to the prefill chunk. If the budget cannot even cover the
  // decodes, the scheduler still forces one prefill token through so the prompt
  // can finish at all.
  const cap = chunked ? Math.max(1, budget - S) : P;
  const nChunks = Math.ceil(P / cap);
  let rem = P, k = 0;
  while (rem > 0) {
    const c = Math.min(cap, rem); rem -= c; k++;
    const s = { chunk: c, decIds: chunked ? waiting : [], kind: 'prefill', idx: steps.length, chunkNo: k, chunksTotal: nChunks };
    steps.push(s);
  }
  // Once the prompt is prefilled, its first token is out and it decodes too.
  for (let i = 0; i < 2; i++) add(0, all, 'tail');

  let t = 0;
  for (const s of steps) {
    s.dec = s.decIds.length;
    s.tokens = s.chunk + s.dec;
    s.msDur = OVERHEAD_MS + s.tokens * TOK_MS;
    s.t0 = t; t += s.msDur; s.t1 = t;
  }

  // Token arrivals per sequence (a step emits one token for each sequence it
  // carried a decode slot for; the last prefill chunk emits the prompt's first
  // token).
  const arrivals = Array.from({ length: S + 1 }, () => []);
  for (const s of steps) {
    if (s.kind === 'prefill' && s.chunkNo === nChunks) arrivals[0].push(s.t1);
    for (const id of s.decIds) arrivals[id].push(s.t1);
  }

  const pf = steps.filter((s) => s.kind === 'prefill');
  const ttft = pf.length ? pf[pf.length - 1].t1 - pf[0].t0 : 0;
  let worst = 0, sum = 0, n = 0;
  for (let i = 1; i <= S; i++) {
    const a = arrivals[i];
    for (let j = 1; j < a.length; j++) { const g = a[j] - a[j - 1]; if (g > worst) worst = g; sum += g; n++; }
  }
  const yMax = Math.max(4, budget, ...steps.map((s) => s.tokens));
  return { steps, arrivals, ttft, worstITL: worst, meanITL: n ? sum / n : 0, total: t, P, S, budget, chunked, cap, nChunks, yMax };
}

let cur = null;            // the schedule on screen
let geom = null;           // hit-test rects captured in draw()
let dragBudget = false;    // true while the budget line is grabbed

function buildData(st) {
  const P = st.P | 0, S = st.S | 0, budget = st.budget | 0, chunked = st.mode !== 'stall';
  const sched = makeSchedule(P, S, budget, chunked);
  const base = makeSchedule(P, S, budget, false);      // the one-big-prefill stall
  // Trade curve: sweep the budget and record both halves of the trade, so the
  // "there is a middle" claim is visible rather than asserted.
  const sweep = [];
  for (let b = 2; b <= 48; b++) { const m = makeSchedule(P, S, b, true); sweep.push({ b, ttft: m.ttft, worst: m.worstITL }); }
  cur = { ...sched, base, sweep };
  return sched.steps.map((s) => ({
    ...s,
    label: s.kind === 'prefill'
      ? `step ${s.idx + 1}: prefill chunk ${s.chunkNo}/${s.chunksTotal} of ${SEQ[0]} (${s.chunk} tok) + ${s.dec} decode${s.dec === 1 ? '' : 's'} = ${s.tokens}/${budget} tokens`
      : `step ${s.idx + 1}: ${s.dec} decode token${s.dec === 1 ? '' : 's'} (${s.decIds.map((i) => SEQ[i]).join(', ')}) = ${s.tokens}/${budget} tokens`,
  }));
}

// Set the token budget (from a drag or a URL hook) and rebuild the schedule.
function setBudget(page, v) {
  const b = clamp(Math.round(v), 2, 48);
  if (b === (page.state.budget | 0)) return;
  page.controls.set('budget', b, { rebuild: true });
}

mount({
  mount: 'body',
  title: 'chunked-prefill — stop one long prompt from stalling everyone else',
  blurb: 'A serving engine runs one batched step at a time. Run a long prompt’s prefill as ONE step and every other sequence’s next token waits for all of it — a hole in their token stream. Chunked prefill splits that prefill into pieces of at most a few tokens and packs each piece into a step next to the waiting sequences’ decode tokens, up to a per-step token budget. DRAG the dashed budget line up and down: the timeline re-packs under your hand and the two latency numbers move in OPPOSITE directions — the long prompt’s time-to-first-token gets worse, everyone else’s worst gap gets better. Hover a step for exactly what it carried, or a gap in a token strip for what the scheduler was doing instead.',
  prefer: 'canvas2d',
  aspect: '2 / 1',
  autoplay: true,
  compare: { key: 'mode', a: 'stall', b: 'chunked', labelA: 'one big prefill — everyone waits', labelB: 'chunked prefill — interleaved with decodes', rebuild: true },
  challenges: [
    {
      goal: 'Cut the waiting sequences’ worst token gap to under half of the one-big-prefill stall.',
      hint: 'switch the scheduler to chunked and drag the budget line down.',
      check: (api) => ({ solved: (api.probe.worst ?? 1e9) < (api.probe.baseWorst ?? 0) / 2, detail: `worst gap ${pctOf(api.probe.worst ?? 0, api.probe.baseWorst ?? 1)}% of the stall (lower is better; need < 50%)` }),
    },
    {
      goal: 'Now pay for it honestly: find a budget where the long prompt’s time-to-first-token is more than 1.5× the stall’s.',
      hint: 'keep dragging the budget down — tiny chunks pay the fixed per-step overhead over and over.',
      check: (api) => ({ solved: (api.probe.ttft ?? 0) > (api.probe.baseTtft ?? 1e9) * 1.5, detail: `TTFT ${pctOf(api.probe.ttft ?? 0, api.probe.baseTtft ?? 1)}% of the stall (lower is better; need > 150%)` }),
    },
  ],
  controls: (c, page) => {
    c.stepper('P', { label: 'long prompt tokens', min: 8, max: 64, step: 4, value: 32 });
    c.stepper('S', { label: 'sequences already decoding', min: 1, max: 4, value: 3 });
    c.slider('budget', { label: 'token budget per step', min: 2, max: 48, step: 1, value: 8, rebuild: true });
    c.select('mode', { label: 'scheduler', options: [{ value: 'chunked', label: 'chunked prefill' }, { value: 'stall', label: 'one big prefill' }], value: 'chunked', rebuild: true });
    c.transport({ compute: () => buildData(page.state), speed: 2.2, loop: true });
  },
  // DIRECT MANIPULATION: grab the dashed budget line and drag it vertically.
  onPointer: (page, ev) => {
    if (!geom) return;
    const g = geom;
    const near = (y) => Math.abs(y - g.yOf(page.state.budget | 0)) <= 14;
    if (ev.type === 'down') {
      dragBudget = ev.x >= g.x0 - 10 && ev.x <= g.xR + 26 && near(ev.y);
      if (dragBudget) setBudget(page, (g.plotBot - ev.y) / g.unit);
    } else if (ev.type === 'up' || ev.type === 'leave') {
      dragBudget = false;
    } else if (ev.type === 'move' && dragBudget && page.pointer.down) {
      setBudget(page, (g.plotBot - ev.y) / g.unit);
    }
  },
  draw: (page) => {
    const r = page.renderer, ctx = page.ctx, st = page.state;
    if (!cur) return;
    r.clear(T.n0);
    const W = page.W, H = page.H, pad = 16;
    const { steps, arrivals, base, sweep, S, P, budget, chunked, cap, nChunks } = cur;
    const idx = page.controls._transport ? page.controls._transport.index : steps.length - 1;
    page.probe = { ttft: cur.ttft, worst: cur.worstITL, baseTtft: base.ttft, baseWorst: base.worstITL, budget, chunked };

    // ---- geometry ----------------------------------------------------------
    const x0 = pad + 44, xR = Math.max(x0 + 120, W - 208);
    const plotTop = 66, plotBot = plotTop + Math.max(88, H * 0.36);
    const yMax = cur.yMax * 1.08;
    const unit = (plotBot - plotTop) / yMax;
    const yOf = (tok) => plotBot - tok * unit;
    const xOf = (t) => x0 + (t / cur.total) * (xR - x0);
    const tOf = (x) => ((x - x0) / (xR - x0)) * cur.total;
    const stripTop = plotBot + 44;
    const rowH = clamp((H - stripTop - 24) / (S + 1) - 4, 8, 20);

    // ---- scheduler timeline ------------------------------------------------
    r.label('scheduler steps — bar width = duration, height = tokens carried', x0, plotTop - 30, { color: T.n12, font: '11px ui-monospace, monospace' });
    r.label('tokens', pad, plotTop - 10, { color: T.n10, font: '10px ui-monospace, monospace' });
    // token axis
    ctx.save();
    ctx.strokeStyle = T.n5; ctx.lineWidth = 1;
    ctx.beginPath(); ctx.moveTo(x0 - 6, plotBot + 0.5); ctx.lineTo(xR + 4, plotBot + 0.5); ctx.stroke();
    const tick = Math.max(1, Math.round(cur.yMax / 4));
    ctx.font = '10px ui-monospace, monospace'; ctx.textAlign = 'right'; ctx.textBaseline = 'middle';
    for (let v = 0; v <= cur.yMax; v += tick) {
      ctx.fillStyle = T.n10; ctx.fillText(String(v), x0 - 9, yOf(v));
      ctx.strokeStyle = rgbaToken('n14', 0.06);
      ctx.beginPath(); ctx.moveTo(x0 - 4, yOf(v) + 0.5); ctx.lineTo(xR + 4, yOf(v) + 0.5); ctx.stroke();
    }
    ctx.restore();

    // step bars, stacked: decode tokens at the bottom, the prefill chunk above
    for (const s of steps) {
      const bx = xOf(s.t0), bw = Math.max(1.5, xOf(s.t1) - xOf(s.t0) - 1.2);
      const past = s.idx <= idx || idx < 0;
      const a = past ? 1 : 0.22;
      let y = plotBot;
      for (const id of s.decIds) {
        const h = Math.max(1.5, unit);
        y -= h;
        ctx.fillStyle = seqFill(id, 0.85 * a);
        ctx.fillRect(bx, y, bw, h - (unit > 4 ? 0.8 : 0));
      }
      if (s.chunk > 0) {
        const h = s.chunk * unit;
        y -= h;
        ctx.fillStyle = alphaOf(T.accent, 0.8 * a);
        ctx.fillRect(bx, y, bw, h);
        ctx.strokeStyle = alphaOf(T.accent, a); ctx.lineWidth = 1; ctx.strokeRect(bx + 0.5, y + 0.5, bw - 1, h - 1);
        if (bw > 22 && h > 12) {
          ctx.save(); ctx.fillStyle = inkOn(T.accent); ctx.font = '10px ui-monospace, monospace'; ctx.textAlign = 'center'; ctx.textBaseline = 'middle';
          ctx.globalAlpha = a; ctx.fillText(String(s.chunk), bx + bw / 2, y + h / 2); ctx.restore();
        }
      }
      if (s.idx === idx) {
        ctx.strokeStyle = T.n14; ctx.lineWidth = 1.5;
        ctx.strokeRect(bx - 1.5, yOf(s.tokens) - 2.5, bw + 3, plotBot - yOf(s.tokens) + 4);
      }
    }

    // the draggable token-budget line
    const by = yOf(budget);
    ctx.save();
    ctx.strokeStyle = chunked ? T.warn : T.n8; ctx.lineWidth = dragBudget ? 2.5 : 1.5;
    ctx.setLineDash([6, 4]);
    ctx.beginPath(); ctx.moveTo(x0 - 6, by + 0.5); ctx.lineTo(xR + 6, by + 0.5); ctx.stroke();
    ctx.setLineDash([]);
    ctx.fillStyle = chunked ? T.warn : T.n8;
    ctx.beginPath(); ctx.moveTo(xR + 8, by); ctx.lineTo(xR + 22, by - 6); ctx.lineTo(xR + 22, by + 6); ctx.closePath(); ctx.fill();
    ctx.restore();
    r.label(`budget ${budget} tok/step  ${chunked ? '↕ drag' : '(ignored)'}`, x0, by - 6, { color: chunked ? T.warn : T.n9, font: '10px ui-monospace, monospace' });

    const overrun = chunked && budget < S + 1;
    if (overrun) r.label(`budget ${budget} cannot even cover ${S} decode${S === 1 ? '' : 's'} — the scheduler forces 1 prefill token through, so each step overruns to ${S + 1} tokens and the prompt crawls (${nChunks} chunks)`, x0, plotBot + 16, { color: T.warn, font: '10px ui-monospace, monospace' });
    else if (chunked) r.label(`chunk cap = budget − ${S} decode${S === 1 ? '' : 's'} = ${cap} tok → ${nChunks} chunk${nChunks === 1 ? '' : 's'} for a ${P}-token prompt`, x0, plotBot + 16, { color: T.n11, font: '10px ui-monospace, monospace' });
    else r.label(`the whole ${P}-token prefill is ONE step — no decode slot in it, so every other sequence waits`, x0, plotBot + 16, { color: T.bad, font: '10px ui-monospace, monospace' });
    r.label('time →', xR - 34, plotBot + 29, { color: T.n10, font: '10px ui-monospace, monospace' });

    // ---- per-sequence token-arrival strips ---------------------------------
    r.label('token arrivals per sequence (same time axis) — a gap is a sequence waiting', x0, stripTop - 12, { color: T.n12, font: '11px ui-monospace, monospace' });
    const rows = [];
    for (let i = 0; i <= S; i++) {
      const ry = stripTop + i * (rowH + 4);
      rows.push({ i, y: ry, h: rowH });
      ctx.fillStyle = rgbaToken('n14', 0.05);
      ctx.fillRect(x0, ry, xR - x0, rowH);
      r.label(i === 0 ? `${SEQ[0]}◆` : SEQ[i], pad + 20, ry + rowH / 2 + 3, { color: seqColor(i), font: '11px ui-monospace, monospace', align: 'right' });
      const a = arrivals[i];
      // shade the long waits
      const gapsAll = [];
      for (let j = 1; j < a.length; j++) gapsAll.push(a[j] - a[j - 1]);
      const sorted = gapsAll.slice().sort((u, v) => u - v);
      const median = sorted.length ? sorted[sorted.length >> 1] : 0;
      const rgaps = [];
      for (let j = 1; j < a.length; j++) {
        const g = a[j] - a[j - 1];
        if (g > median * 1.5) {
          rgaps.push({ a: a[j - 1], b: a[j], g });
          ctx.fillStyle = alphaOf(T.bad, 0.16);
          ctx.fillRect(xOf(a[j - 1]), ry, xOf(a[j]) - xOf(a[j - 1]), rowH);
        }
      }
      rows[rows.length - 1].gaps = rgaps;
      const revealT = idx < 0 ? -1 : steps[Math.min(idx, steps.length - 1)].t1;
      for (const t of a) {
        ctx.fillStyle = t <= revealT ? seqFill(i, 0.95) : seqFill(i, 0.22);
        ctx.fillRect(xOf(t) - 1.5, ry + 1, 3, rowH - 2);
      }
      if (i === 0) {
        // mark where the long prompt's first token finally lands
        const ft = a[0];
        if (ft != null) {
          ctx.strokeStyle = alphaOf(T.accent, 0.8); ctx.lineWidth = 1; ctx.setLineDash([2, 3]);
          ctx.beginPath(); ctx.moveTo(xOf(ft), plotBot + 22); ctx.lineTo(xOf(ft), ry); ctx.stroke(); ctx.setLineDash([]);
          r.label('first token', xOf(ft) + 4, ry - 1, { color: T.accent, font: '9px ui-monospace, monospace' });
        }
      }
    }

    // ---- right column: the two numbers, moving in opposite directions ------
    const cx = xR + 34, cw = W - cx - pad;
    r.label('the trade — lower is better', cx, plotTop - 30, { color: T.n12, font: '9.5px ui-monospace, monospace' });
    // card 1: the long prompt pays
    ctx.fillStyle = rgbaToken('n14', 0.04); ctx.fillRect(cx, plotTop - 18, cw, 62);
    r.label(`${SEQ[0]}: time to first token`, cx + 8, plotTop - 4, { color: T.n11, font: '10px ui-monospace, monospace' });
    const pT = pctOf(cur.ttft, base.ttft);
    r.label(ms(cur.ttft), cx + 8, plotTop + 17, { color: dirColor(pT), font: '18px ui-monospace, monospace' });
    r.label(`${dirWord(pT)} — ${pT}% of the stall`, cx + 8, plotTop + 31, { color: T.n11, font: '9px ui-monospace, monospace' });
    r.label(`one big prefill: ${ms(base.ttft)}`, cx + 8, plotTop + 41, { color: T.n10, font: '9px ui-monospace, monospace' });
    // card 2: everybody else gains
    const c2y = plotTop + 54;
    ctx.fillStyle = rgbaToken('n14', 0.04); ctx.fillRect(cx, c2y, cw, 62);
    r.label(`${SEQ.slice(1, S + 1).join('/')}: worst token gap`, cx + 8, c2y + 14, { color: T.n11, font: '10px ui-monospace, monospace' });
    const pW = pctOf(cur.worstITL, base.worstITL);
    r.label(ms(cur.worstITL), cx + 8, c2y + 35, { color: dirColor(pW), font: '18px ui-monospace, monospace' });
    r.label(`${dirWord(pW)} — ${pW}% of the stall`, cx + 8, c2y + 49, { color: T.n11, font: '9px ui-monospace, monospace' });
    r.label(`one big prefill: ${ms(base.worstITL)}`, cx + 8, c2y + 59, { color: T.n10, font: '9px ui-monospace, monospace' });

    // ---- trade curve: both halves vs the budget ----------------------------
    const gy = c2y + 78, gh = Math.max(52, H - gy - 30), gw = cw;
    r.label('budget sweep', cx, gy - 6, { color: T.n11, font: '10px ui-monospace, monospace' });
    ctx.fillStyle = rgbaToken('n14', 0.04); ctx.fillRect(cx, gy, gw, gh);
    const maxT = Math.max(...sweep.map((p) => p.ttft)), maxW = Math.max(...sweep.map((p) => p.worst));
    const gx = (b) => cx + ((b - 2) / 46) * gw;
    const line = (get, mx, col) => {
      ctx.save(); ctx.strokeStyle = col; ctx.lineWidth = 1.5; ctx.beginPath();
      sweep.forEach((p, i) => { const X = gx(p.b), Y = gy + gh - 6 - (get(p) / mx) * (gh - 14); if (i === 0) ctx.moveTo(X, Y); else ctx.lineTo(X, Y); });
      ctx.stroke(); ctx.restore();
    };
    line((p) => p.ttft, maxT, T.accent);
    line((p) => p.worst, maxW, alphaOf(categorical(1), 1));
    ctx.save(); ctx.strokeStyle = T.warn; ctx.lineWidth = 1; ctx.setLineDash([3, 3]);
    ctx.beginPath(); ctx.moveTo(gx(budget), gy + 2); ctx.lineTo(gx(budget), gy + gh - 2); ctx.stroke(); ctx.setLineDash([]); ctx.restore();
    r.label(`${SEQ[0]} TTFT`, cx + 4, gy + 12, { color: T.accent, font: '9px ui-monospace, monospace' });
    r.label('worst gap', cx + 4, gy + 23, { color: alphaOf(categorical(1), 1), font: '9px ui-monospace, monospace' });
    r.label('small budget', cx + 2, gy + gh + 10, { color: T.n10, font: '9px ui-monospace, monospace' });
    r.label('large', cx + gw - 26, gy + gh + 10, { color: T.n10, font: '9px ui-monospace, monospace' });

    // ---- hover-to-inspect --------------------------------------------------
    geom = { x0, xR, plotTop, plotBot, unit, yOf, xOf, tOf, rows, rowH };
    if (page.pointer.over && !dragBudget) {
      const p = page.pointer;
      let tip = null;
      if (p.x >= x0 - 6 && p.x <= xR + 6 && p.y >= plotTop - 8 && p.y <= plotBot + 4) {
        const t = tOf(p.x);
        const s = steps.find((q) => t >= q.t0 && t <= q.t1);
        if (s) {
          const who = s.decIds.length ? s.decIds.map((i) => SEQ[i]).join(', ') : 'none';
          tip = `step ${s.idx + 1} of ${steps.length}   (${ms(s.msDur)})\n`
            + (s.chunk ? `prefill: ${SEQ[0]} chunk ${s.chunkNo}/${s.chunksTotal} — ${s.chunk} token${s.chunk === 1 ? '' : 's'}\n` : 'prefill: none this step\n')
            + `decode: ${s.dec} token${s.dec === 1 ? '' : 's'} (${who})\n`
            + `tokens used ${s.tokens} / budget ${budget}\n`
            + `cost = ${OVERHEAD_MS} ms overhead + ${s.tokens}×${TOK_MS} ms/token`;
        }
      }
      if (!tip) {
        for (const row of rows) {
          if (p.y < row.y - 2 || p.y > row.y + row.h + 2 || p.x < x0 || p.x > xR) continue;
          const t = tOf(p.x);
          const g = (row.gaps || []).find((q) => t >= q.a && t <= q.b);
          if (g) {
            const blockers = steps.filter((s) => s.t1 > g.a && s.t0 < g.b && s.chunk > 0);
            const first = blockers[0], last = blockers[blockers.length - 1];
            tip = `${SEQ[row.i]} waited ${ms(g.g)} between tokens\n`
              + (blockers.length
                ? `waiting: ${SEQ[0]}'s prefill occupied ${blockers.length === 1 ? `step ${first.idx + 1}` : `steps ${first.idx + 1}–${last.idx + 1}`} (${blockers.reduce((n, s) => n + s.chunk, 0)} prompt tokens)`
                : `waiting: no decode slot for ${SEQ[row.i]} in those steps`)
              + (row.i === 0 ? `\n(this is ${SEQ[0]}'s own wait for its first token)` : '');
          } else {
            const a = arrivals[row.i];
            let bestT = null, bd = Infinity;
            for (const q of a) { const d = Math.abs(q - t); if (d < bd) { bd = d; bestT = q; } }
            if (bestT != null) tip = `${SEQ[row.i]}: token at ${ms(bestT)}\n${row.i === 0 ? 'first token = end of the last prefill chunk' : 'one decode token per scheduler step it was packed into'}`;
          }
          break;
        }
      }
      if (!tip && Math.abs(p.y - by) <= 14 && p.x >= x0 - 10 && p.x <= xR + 26) {
        tip = `token budget = ${budget} tokens per step\n↕ drag to re-pack the timeline\nsmaller → smoother for ${SEQ.slice(1, S + 1).join('/')}, slower first token for ${SEQ[0]}`;
      }
      if (tip) page.setTip(tip);
    }

    // ---- readout -----------------------------------------------------------
    const s = page.step();
    let o = !chunked
      ? `one big prefill: all ${P} prompt tokens in a single step; ${SEQ.slice(1, S + 1).join('/')} get no decode slot until it finishes.`
      : overrun
        ? `chunked, but the ${budget}-token budget is below the ${S} decode${S === 1 ? '' : 's'} it must schedule: each step still forces 1 prefill token, overrunning to ${S + 1} tokens → ${nChunks} chunks.`
        : `chunked: each step carries ${S} decode token${S === 1 ? '' : 's'} + a prefill chunk of ≤ ${cap}, up to the ${budget}-token budget → ${nChunks} chunk${nChunks === 1 ? '' : 's'}.`;
    o += `   tier:${r.name}\n`;
    o += `${SEQ[0]} time-to-first-token ${ms(cur.ttft)} = ${pctOf(cur.ttft, base.ttft)}% of one-big-prefill · ${SEQ.slice(1, S + 1).join('/')} worst token gap ${ms(cur.worstITL)} = ${pctOf(cur.worstITL, base.worstITL)}% of one-big-prefill (both lower-is-better; 100% = parity). `;
    o += s ? `\n${s.label}` : '';
    page.setReadout(o);
  },
}).then((page) => {
  window.__chunkedPrefillPage = page;
  const q = new URLSearchParams(location.search);
  const t = page.controls._transport;
  // Every control is a URL hook (the framework mirrors state into the query
  // string); these restore it on load. `?budget=` is the headless stand-in for
  // dragging the budget line, since --screenshot has no pointer.
  const num = (k, lo, hi) => (q.has(k) ? clamp(parseInt(q.get(k), 10) || lo, lo, hi) : null);
  const P = num('P', 8, 64); if (P != null) page.controls.set('P', P, { rebuild: true, silent: true });
  const S = num('S', 1, 4); if (S != null) page.controls.set('S', S, { rebuild: true, silent: true });
  const B = num('budget', 2, 48); if (B != null) page.controls.set('budget', B, { rebuild: true, silent: true });
  if (q.has('mode')) page.controls.set('mode', q.get('mode') === 'stall' ? 'stall' : 'chunked', { rebuild: true, silent: true });
  if (t) t.rebuild();
  // ?hover=x,y fakes the cursor (headless stand-in for a real hover) so the
  // step / gap tooltips are screenshot-verifiable. Canvas-space px.
  if (q.has('hover')) {
    const [hx, hy] = q.get('hover').split(',').map(Number);
    page.pointer.x = hx; page.pointer.y = hy; page.pointer.over = true;
  }
  // Deterministic frame for capture: pause before seeking, so autoplay does not
  // advance off the requested step.
  if (q.has('step') || q.has('hover') || q.has('budget') || q.has('mode')) { if (t) t.pause(); }
  if (q.has('step') && t) t.seek(parseInt(q.get('step'), 10));
  if (q.get('play') === '1' && t) t.play();
  page.redraw();
});
