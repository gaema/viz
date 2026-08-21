// context-editing concept page -- an agent's context window fills with stale
// tool results, so the harness CLEARS them once occupancy crosses a threshold.
// The saving is obvious. The cost is not, and the cost is what this page is for.
//
// THE MECHANISM. A long-running agent accumulates a transcript: user turns,
// tool calls, tool RESULTS (by far the largest blocks), assistant text. Most of
// that is stale within a few turns -- a file read six turns ago, a search whose
// answer has already been used. Context editing scores whole TURNS and TOOL
// RESULTS rather than individual tokens, and once the transcript crosses a
// threshold it clears the old tool results (leaving a short marker) and, in the
// stronger policy, folds the old turns into one running summary.
//
// It is the same grammar as token-level KV eviction one layer up: score, drop,
// keep the recent tail. Two differences matter. The unit is a MESSAGE, not a
// token. And it edits the visible TRANSCRIPT, not the cache -- which is exactly
// where the surprise lives.
//
// THE SUBTLE COST. Everything before the edit point is a shared PREFIX that the
// serving stack was re-reading cheaply from a prompt cache. Editing history
// invalidates that prefix from the edit point onward, so the very next turn has
// to PREFILL every token after the edit at full price. A compaction that saves
// tokens can therefore buy a large latency spike, and a too-eager threshold
// makes every turn expensive. Drag the threshold down and watch it happen.
//
// THE HONEST COUNTER-POINT. When cached prefix re-reads are cheap enough,
// KEEPING EVERYTHING is cheaper than summarising: a stable prefix is nearly free
// to re-read, while an edit is never free. The `cached prefix re-read` slider is
// that dial. Lower it and compaction loses; raise it and compaction wins. The
// page does not decide for you -- it computes both arms and reports the ratio.
//
// EVERY NUMBER ON SCREEN IS COMPUTED FROM THIS SIMULATION. Occupancy, tokens
// cleared, tokens re-prefilled after an edit, per-turn cost and cumulative cost
// all come out of `run()` below; nothing is illustrative.
import { mount } from '../framework/layout.js';
import { seededRand } from '../framework/tensor.js';
import { T, alphaOf, mixColor, inkOn } from '../framework/theme.js';

// ---------------------------------------------------------------------------
// Cost unit. 1 TE ("prefill token-equivalent") = the compute of prefilling one
// uncached token. A token that is still inside an unbroken cached prefix costs
// `cacheRate` TE instead of 1. The rate is a SLIDER, not a claim: real caches
// differ by stack and by deployment, and the interesting question is how the
// verdict moves across the range, not what one vendor charges today.
// ---------------------------------------------------------------------------

const STUB_TOKENS = 12;          // what a cleared tool result leaves behind
const SUMMARY_FRAC = 0.06;       // a summary is ~6% of what it replaces
const SUMMARY_MIN = 60, SUMMARY_MAX = 800;

const ARMS = [
  { key: 'never', name: 'never compact', short: 'never compact',
    desc: 'keep the whole transcript — the baseline every ratio here is against, and the arm that eventually overflows the window' },
  { key: 'clear', name: 'clear old tool results', short: 'clear results',
    desc: 'once occupancy crosses the threshold, replace every tool result older than the kept tail with a short marker' },
  { key: 'summary', name: 'clear results + summarise turns', short: 'clear + summarise',
    desc: 'the same clearing, plus the old turns ahead of the kept tail collapse into one running summary' },
];
const armAt = (k) => ARMS.find((a) => a.key === k) || ARMS[1];

const KINDS = {
  user:   { label: 'user turn',     hue: 'teal' },
  call:   { label: 'tool call',     hue: 'gold' },
  result: { label: 'TOOL RESULT',   hue: 'violet' },
  text:   { label: 'assistant text', hue: 'accent' },
};

// Block states inside one arm's transcript.
const NOT_YET = 0, LIVE = 1, STUB = 2, FOLDED = 3, SUMMARY = 4;

// ---------------------------------------------------------------------------
// The conversation. Deterministic in (turns, seed) alone, so all three arms see
// EXACTLY the same conversation and the comparison is a controlled one.
// ---------------------------------------------------------------------------
function buildBlocks(st) {
  const N = Math.max(4, st.turns | 0);
  const u = seededRand((st.seed | 0) + 1, N * 5, { lo: 0, hi: 1 });
  const blocks = [];
  for (let t = 0; t < N; t++) {
    const r = (k) => u[t * 5 + k];
    blocks.push({ turn: t, kind: 'user', tokens: 40 + Math.round(r(0) * 90) });
    if (r(1) > 0.22) {                                    // most turns call a tool
      blocks.push({ turn: t, kind: 'call', tokens: 26 + Math.round(r(2) * 50) });
      // Heavy tail: a directory listing and a whole-file read differ by ~20x,
      // and it is the rare huge result that drives a window over its threshold.
      const heavy = Math.exp(r(3) * 2.6);
      blocks.push({ turn: t, kind: 'result', tokens: Math.round(260 + heavy * 620) });
    }
    blocks.push({ turn: t, kind: 'text', tokens: 70 + Math.round(r(4) * 300) });
  }
  blocks.forEach((b, i) => { b.id = i; });
  return blocks;
}

const parsePins = (st) => {
  const s = new Set();
  String(st.pins == null ? '' : st.pins).split(',').forEach((x) => {
    const v = parseInt(x, 10); if (Number.isFinite(v) && v >= 0) s.add(v);
  });
  return s;
};

// ---------------------------------------------------------------------------
// One arm, turn by turn. Returns a per-turn record carrying the transcript
// state, what the edit cost, and the running totals.
// ---------------------------------------------------------------------------
function runArm(blocks, st, armKey, pins) {
  const N = Math.max(4, st.turns | 0);
  const W = Math.max(4000, (st.win | 0) * 1000);
  const thr = (st.thresh / 100) * W;
  const keep = Math.max(0, st.keep | 0);
  const cacheRate = Math.max(0, Math.min(1, st.cache));
  const nb = blocks.length;

  const state = new Uint8Array(nb);                 // NOT_YET everywhere
  const sumTok = new Float64Array(nb);              // tokens carried by a SUMMARY block
  const eff = (i) => {
    const s = state[i];
    return s === LIVE ? blocks[i].tokens : s === STUB ? STUB_TOKENS : s === SUMMARY ? sumTok[i] : 0;
  };

  const turns = [];
  let prevCount = 0;          // block slots present at the end of the previous turn
  let cum = 0, cumCleared = 0, overflowAt = -1;

  for (let t = 0; t < N; t++) {
    // --- the turn arrives: every block of turn t enters the transcript
    let count = prevCount;
    while (count < nb && blocks[count].turn === t) { state[count] = LIVE; count++; }

    const total0 = (() => { let m = 0; for (let i = 0; i < count; i++) m += eff(i); return m; })();

    // --- compaction, if this arm does that and the threshold is crossed
    let editIdx = -1, clearedTok = 0, foldedTok = 0, nCleared = 0, fired = false;
    if (armKey !== 'never' && total0 > thr) {
      fired = true;
      // Tool results, newest first. The last `keep` survive; so does anything pinned.
      const results = [];
      for (let i = 0; i < count; i++) if (blocks[i].kind === 'result' && state[i] === LIVE) results.push(i);
      const keptTail = new Set(results.slice(Math.max(0, results.length - keep)));
      for (const i of results) {
        if (keptTail.has(i) || pins.has(blocks[i].id)) continue;
        clearedTok += blocks[i].tokens - STUB_TOKENS;
        state[i] = STUB; nCleared++;
        if (editIdx < 0 || i < editIdx) editIdx = i;
      }

      if (armKey === 'summary') {
        // Everything ahead of the protected tail folds into one running summary.
        let boundary = count;
        for (const i of keptTail) boundary = Math.min(boundary, i);
        for (let i = 0; i < count; i++) if (pins.has(blocks[i].id)) { boundary = Math.min(boundary, i); break; }
        for (let i = 0; i < count; i++) if (blocks[i].turn >= t - 1) { boundary = Math.min(boundary, i); break; }

        const region = [];
        let regionTok = 0;
        for (let i = 0; i < boundary; i++) {
          if (state[i] !== LIVE && state[i] !== STUB && state[i] !== SUMMARY) continue;
          if (pins.has(blocks[i].id)) continue;
          region.push(i); regionTok += eff(i);
        }
        const sTok = Math.max(SUMMARY_MIN, Math.min(SUMMARY_MAX, Math.round(regionTok * SUMMARY_FRAC)));
        if (region.length > 1 && regionTok > sTok + 120) {
          foldedTok = regionTok - sTok;
          const head = region[0];
          for (const i of region) { state[i] = FOLDED; sumTok[i] = 0; }
          state[head] = SUMMARY; sumTok[head] = sTok;
          if (editIdx < 0 || head < editIdx) editIdx = head;
        }
      }
      if (editIdx < 0) fired = false;                 // threshold crossed, nothing left to clear
    }

    const total = (() => { let m = 0; for (let i = 0; i < count; i++) m += eff(i); return m; })();

    // --- the cost of prefilling THIS turn.
    // Tokens before the edit point are still an unbroken cached prefix and are
    // re-read at `cacheRate`. Everything from the edit point onward -- including
    // text that did not itself change, because a prefix cache is positional --
    // is prefilled at full price, together with the tokens this turn appended.
    const cutoff = editIdx >= 0 ? editIdx : prevCount;
    let cachedTok = 0; for (let i = 0; i < cutoff; i++) cachedTok += eff(i);
    const fullTok = Math.max(0, total - cachedTok);
    const cost = cacheRate * cachedTok + fullTok;
    cum += cost;
    cumCleared += clearedTok + foldedTok;

    const over = total > W;
    if (over && overflowAt < 0) overflowAt = t;

    turns.push({
      t, count, total, cost, cum, cumCleared,
      cachedTok, fullTok, editIdx, fired,
      clearedTok, foldedTok, nCleared,
      occupancy: total / W, over, overflowAt,
      state: state.slice(0, count), sumTok: sumTok.slice(0, count),
    });
    prevCount = count;
  }
  return { turns, W, thr };
}

// One simulation of all three arms over the same conversation. Everything drawn
// is read back out of here, so the picture and the numbers cannot drift apart.
let cur = null;

function simulate(st) {
  const blocks = buildBlocks(st);
  const pins = parsePins(st);
  const arms = {};
  for (const a of ARMS) arms[a.key] = runArm(blocks, st, a.key, pins);
  const N = arms.never.turns.length;
  const W = arms.never.W, thr = arms.never.thr;

  const steps = [];
  for (let t = 0; t < N; t++) {
    steps.push({
      t,
      label: `turn ${t + 1} of ${N}`,
      rec: { never: arms.never.turns[t], clear: arms.clear.turns[t], summary: arms.summary.turns[t] },
    });
  }
  cur = { blocks, pins, arms, N, W, thr, steps };
  return steps;
}

// ---------------------------------------------------------------------------
// Draw-time hit rects, captured so the pointer layer tests exactly what was
// painted. `grab` is the handle currently under the hand.
// ---------------------------------------------------------------------------
let rTape = null, rKeep = null, rCost = null, rThresh = null;
let tapeRects = [], costW = 0, costX0 = 0, resultLefts = [];
let grab = null;

const fmtK = (v) => (v >= 10000 ? (v / 1000).toFixed(0) + 'k' : v >= 1000 ? (v / 1000).toFixed(1) + 'k' : String(Math.round(v)));
const pct1 = (v) => (100 * v).toFixed(1) + '%';

// Why this block is (or is not) still in the context, in the vocabulary of the
// active policy. This sentence is the page.
function blockReason(i, rec, st, keptTailSet) {
  const b = cur.blocks[i], s = rec.state[i];
  if (s === undefined || s === NOT_YET) return `not in the conversation yet — arrives on turn ${b.turn + 1}`;
  if (cur.pins.has(b.id)) return `PINNED — you protected this block, so no compaction may touch it (click again to release)`;
  if (s === LIVE) {
    if (b.kind !== 'result') return `KEPT — ${KINDS[b.kind].label}, still verbatim in the context`;
    return keptTailSet.has(i)
      ? `KEPT — inside the recent tail: the last ${st.keep | 0} tool result${(st.keep | 0) === 1 ? '' : 's'} are never cleared`
      : `KEPT — the threshold has not been crossed yet, so nothing has been edited`;
  }
  if (s === STUB) return `CLEARED — a stale tool result outside the kept tail; ${b.tokens} tokens replaced by a ${STUB_TOKENS}-token marker`;
  if (s === FOLDED) return `FOLDED — absorbed into the running summary to the left; its ${b.tokens} tokens are gone from the context`;
  if (s === SUMMARY) return `SUMMARY — this slot now carries the running summary of everything that was folded into it`;
  return '';
}

mount({
  mount: 'body',
  title: 'context-editing — compacting an agent transcript, and what the edit costs',
  blurb: 'A long-running agent fills its context with tool results that went stale turns ago. Context editing clears them once occupancy crosses a threshold — the same idea as token-level KV eviction, but scoring whole turns and tool results, and editing the visible transcript rather than the cache. The catch is underneath: everything before the edit point was a cached prefix, and editing history invalidates it, so the next turn must re-prefill everything after the cut at full price. Drag the ▾ threshold down the tape and watch the recompute spike; drag the ◂▸ kept-tail bracket to protect more recent results; click any block to pin it. Then drop the cached-prefix re-read cost and find the setting where compacting LOSES to never compacting at all.',
  prefer: 'canvas2d',
  aspect: '16 / 10',
  autoplay: true,
  animate: false,
  compare: { key: 'policy', a: 'never', b: 'summary', labelA: 'never compact (baseline)', labelB: 'clear + summarise', rebuild: true },
  challenges: [
    {
      goal: 'Make compaction LOSE: find a setting where clearing costs MORE total prefill than never compacting.',
      hint: 'drop the cached prefix re-read cost toward 0 — a free prefix makes keeping everything nearly free, while every edit still pays full price for the tail.',
      check: (api) => ({
        solved: (api.probe.ratio ?? 0) > 1.0,
        detail: `clear+summarise cumulative cost is ${(100 * (api.probe.ratio ?? 0)).toFixed(1)}% of never-compacting (lower is better; >100% = compaction lost)`,
      }),
    },
    {
      goal: 'Now make it WIN by a clear margin: get the active policy under 80% of the never-compacting baseline.',
      hint: 'raise the cached re-read cost (an expensive prefix punishes a long transcript) and raise the threshold so edits fire rarely.',
      check: (api) => ({
        solved: (api.probe.ratio ?? 2) < 0.8,
        detail: `active policy is at ${(100 * (api.probe.ratio ?? 1)).toFixed(1)}% of never-compacting (need < 80.0%)`,
      }),
    },
    {
      goal: 'Protect something: pin at least one tool result and keep the policy under the window.',
      hint: 'click a violet TOOL RESULT block on the tape — it gains a ring and survives every later compaction.',
      check: (api) => ({
        solved: (api.probe.nPins ?? 0) >= 1 && !(api.probe.activeOver ?? true),
        detail: `${api.probe.nPins ?? 0} block(s) pinned · active policy occupancy ${pct1(api.probe.occ ?? 0)} of the window`,
      }),
    },
  ],

  controls: (c, page) => {
    c.select('policy', { label: 'policy shown on the tape', value: 'clear', options: ARMS.map((a) => ({ value: a.key, label: a.name })), rebuild: true });
    c.slider('thresh', { label: 'clear threshold (% of window) — drag ▾ too', min: 20, max: 98, step: 1, value: 70, rebuild: true });
    c.slider('keep', { label: 'recent tool results kept — drag ◂▸ too', min: 0, max: 12, step: 1, value: 3, rebuild: true });
    c.slider('cache', { label: 'cached prefix re-read cost (× a fresh token)', min: 0, max: 1, step: 0.01, value: 0.1, rebuild: true });
    c.stepper('win', { label: 'context window (k tokens)', min: 8, max: 200, step: 8, value: 64, rebuild: true });
    c.stepper('turns', { label: 'conversation length (turns)', min: 6, max: 60, step: 2, value: 30, rebuild: true });
    c.slider('seed', { label: 'seed (conversation content)', min: 0, max: 99, step: 1, value: 4, rebuild: true });
    c.text('pins', { label: 'pinned block ids (click a block)', value: '', placeholder: 'none', rebuild: true });
    c.transport({ compute: () => simulate(page.state), speed: 3, loop: true });
  },

  // Direct manipulation: the ▾ threshold marker on the tape, the ◂▸ kept-tail
  // bracket under it, and click-to-pin on any block. Each re-runs the whole
  // simulation, so the cost chart moves under your hand rather than tweening.
  onPointer: (page, ev) => {
    if (!cur || !rTape) return;
    const st = page.state;
    const nearX = (x, y, rect, tol) => rect && Math.abs(ev.x - x) <= (tol || 9) && ev.y >= rect.y - 10 && ev.y <= rect.y + rect.h + 10;
    const threshFromX = (x) => Math.max(20, Math.min(98, Math.round(100 * (x - rTape.x) / Math.max(1, rTape.w))));
    const keepFromX = (x) => {
      let best = st.keep | 0, bd = Infinity;
      for (let k = 0; k < resultLefts.length; k++) {
        if (resultLefts[k] == null) continue;
        const d = Math.abs(x - resultLefts[k]);
        if (d < bd) { bd = d; best = k; }
      }
      return Math.max(0, Math.min(12, best));
    };

    if (ev.type === 'down') {
      grab = null;
      if (nearX(rThresh, 0, rTape, 10)) grab = 'thresh';
      else if (rKeep && nearX(rKeep.x, 0, rKeep, 11)) grab = 'keep';
      if (grab) {
        const tr = page.controls._transport; if (tr) tr.pause();
        page.controls.set(grab, grab === 'thresh' ? threshFromX(ev.x) : keepFromX(ev.x), { rebuild: true });
        return;
      }
      // Click-to-pin: toggle the block under the cursor.
      for (const hit of tapeRects) {
        if (ev.x >= hit.x && ev.x <= hit.x + hit.w && ev.y >= rTape.y && ev.y <= rTape.y + rTape.h) {
          const id = cur.blocks[hit.i].id;
          const set = parsePins(st);
          if (set.has(id)) set.delete(id); else set.add(id);
          page.controls.set('pins', [...set].sort((a, b) => a - b).join(','), { rebuild: true });
          return;
        }
      }
    } else if (ev.type === 'up' || ev.type === 'leave') {
      grab = null;
    } else if (ev.type === 'move' && grab && page.pointer.down) {
      const v = grab === 'thresh' ? threshFromX(ev.x) : keepFromX(ev.x);
      if ((page.state[grab] | 0) !== v) page.controls.set(grab, v, { rebuild: true });
    }
  },

  draw: (page) => {
    const r = page.renderer, ctx = page.ctx, st = page.state;
    r.clear(T.n0);
    const sim = cur; if (!sim || !sim.steps.length) return;

    const s = page.step();
    const t = s ? Math.min(s.t, sim.N - 1) : sim.N - 1;
    const armKey = armAt(st.policy).key;
    const arm = armAt(armKey);
    const rec = sim.arms[armKey].turns[t];
    const base = sim.arms.never.turns[t];
    const W = sim.W, thr = sim.thr;

    // ---- geometry --------------------------------------------------------
    const pad = 14, gut = 118;
    const x0 = pad + gut, gridW = Math.max(120, page.W - 2 * pad - gut - 44);
    const topY = 34;
    const availH = Math.max(240, page.H - topY - 12);
    const tapeH = Math.max(38, Math.round(availH * 0.20));
    const yTape = topY + 16;
    const yKeep = yTape + tapeH + 5;
    const costH = Math.max(60, Math.round(availH * 0.30));
    const yCost = yKeep + 30;
    const yPanel = yCost + costH + 34;
    rTape = { x: x0, y: yTape, w: gridW, h: tapeH };
    rCost = { x: x0, y: yCost, w: gridW, h: costH };
    costX0 = x0; costW = gridW;

    const lab = (txt, x, y, col, font) => r.label(txt, x, y, { color: col || T.n11, font: font || '11px ui-monospace, monospace' });
    const tx = (tok) => (tok / W) * gridW;      // token -> px along the tape (window = full width)

    // ---- band 1: the context tape ---------------------------------------
    lab('context', pad, yTape + 12, T.n12);
    lab('transcript', pad, yTape + 25, T.n11, '10px ui-monospace, monospace');
    lab(`window ${fmtK(W)}`, pad, yTape + 38, T.n9, '10px ui-monospace, monospace');

    ctx.save();
    ctx.fillStyle = alphaOf('n9', 0.10);
    ctx.fillRect(x0, yTape, gridW, tapeH);
    ctx.strokeStyle = alphaOf('n14', 0.22); ctx.lineWidth = 1;
    ctx.strokeRect(x0 + 0.5, yTape + 0.5, gridW - 1, tapeH - 1);
    ctx.restore();

    // Which tool results are in the protected tail right now (for the tooltip).
    const keptTailSet = new Set();
    {
      const live = [];
      for (let i = 0; i < rec.count; i++) if (cur.blocks[i].kind === 'result' && rec.state[i] === LIVE) live.push(i);
      for (const i of live.slice(Math.max(0, live.length - (st.keep | 0)))) keptTailSet.add(i);
    }

    tapeRects = []; resultLefts = [];
    let cx = x0, over = false;
    const liveResultsRTL = [];   // left edges of live tool results, newest first
    for (let i = 0; i < rec.count; i++) {
      const b = cur.blocks[i], sState = rec.state[i];
      const tok = sState === LIVE ? b.tokens : sState === STUB ? STUB_TOKENS : sState === SUMMARY ? rec.sumTok[i] : 0;
      if (tok <= 0) continue;
      const w = Math.max(1.2, tx(tok));
      if (cx + w > x0 + gridW + 0.5) over = true;
      const drawW = Math.max(1.2, Math.min(w, x0 + gridW - cx));
      if (drawW > 0.5) {
        const pinned = cur.pins.has(b.id);
        let fill;
        if (sState === STUB) fill = alphaOf('n9', 0.55);
        else if (sState === SUMMARY) fill = T.warn;
        else fill = mixColor(T.n0, T[KINDS[b.kind].hue], b.kind === 'result' ? 0.82 : 0.55);
        ctx.save();
        ctx.fillStyle = fill;
        ctx.fillRect(cx, yTape + 1, drawW, tapeH - 2);
        if (drawW > 2.5) { ctx.strokeStyle = alphaOf('n0', 0.55); ctx.lineWidth = 1; ctx.strokeRect(cx + 0.5, yTape + 1.5, drawW - 1, tapeH - 3); }
        if (pinned) {
          ctx.strokeStyle = T.goldDeep; ctx.lineWidth = 2;
          ctx.strokeRect(cx + 1, yTape + 2, Math.max(2, drawW - 2), tapeH - 4);
        }
        if (sState === STUB) {                       // a visible scar where a result was
          ctx.strokeStyle = alphaOf('bad', 0.75); ctx.lineWidth = 1;
          ctx.beginPath(); ctx.moveTo(cx, yTape + tapeH - 2); ctx.lineTo(cx + drawW, yTape + 2); ctx.stroke();
        }
        if (b.kind === 'result' && sState === LIVE && drawW > 26) {
          ctx.fillStyle = inkOn(fill); ctx.font = '9px ui-monospace, monospace'; ctx.textAlign = 'center';
          ctx.fillText(fmtK(b.tokens), cx + drawW / 2, yTape + tapeH / 2 + 3);
        }
        ctx.restore();
      }
      tapeRects.push({ i, x: cx, w: drawW });
      if (b.kind === 'result' && sState === LIVE) liveResultsRTL.unshift(cx);
      cx += w;
    }
    // resultLefts[k] = where the tail of k kept results begins (k = 0 -> the far right)
    resultLefts[0] = x0 + Math.min(gridW, tx(rec.total));
    for (let k = 1; k <= 12; k++) resultLefts[k] = liveResultsRTL[k - 1] != null ? liveResultsRTL[k - 1] : resultLefts[k - 1];

    // the edit point: everything to its right had to be re-prefilled this turn
    if (rec.editIdx >= 0) {
      let ex = x0;
      for (let i = 0; i < rec.editIdx; i++) {
        const sS = rec.state[i];
        ex += tx(sS === LIVE ? cur.blocks[i].tokens : sS === STUB ? STUB_TOKENS : sS === SUMMARY ? rec.sumTok[i] : 0);
      }
      ex = Math.min(ex, x0 + gridW);
      ctx.save();
      ctx.fillStyle = alphaOf('bad', 0.13);
      ctx.fillRect(ex, yTape - 5, Math.min(gridW, tx(rec.total)) + x0 - ex, tapeH + 10);
      ctx.strokeStyle = T.bad; ctx.lineWidth = 2;
      ctx.beginPath(); ctx.moveTo(ex, yTape - 6); ctx.lineTo(ex, yTape + tapeH + 6); ctx.stroke();
      ctx.restore();
      lab(`edit point → ${fmtK(rec.fullTok)} tokens re-prefilled at full price`, Math.min(ex + 4, x0 + gridW - 240), yTape - 9, T.bad, '10px ui-monospace, monospace');
    } else {
      lab(`no edit this turn — the whole prefix is still cached`, x0 + 2, yTape - 9, T.n9, '10px ui-monospace, monospace');
    }

    // the draggable threshold marker
    const thX = x0 + gridW * (st.thresh / 100);
    rThresh = thX;
    ctx.save();
    ctx.strokeStyle = grab === 'thresh' ? T.warn : alphaOf('warn', 0.9);
    ctx.lineWidth = grab === 'thresh' ? 2.5 : 1.6;
    ctx.setLineDash([4, 3]);
    ctx.beginPath(); ctx.moveTo(thX, yTape - 4); ctx.lineTo(thX, yTape + tapeH + 4); ctx.stroke();
    ctx.setLineDash([]);
    ctx.fillStyle = ctx.strokeStyle; ctx.font = '11px ui-monospace, monospace'; ctx.textAlign = 'center';
    ctx.fillText('▾', thX, yTape - 5);
    ctx.restore();
    lab(`threshold ${st.thresh | 0}%`, pad, yTape + 51, T.warn, '10px ui-monospace, monospace');

    if (over) {
      ctx.save();
      ctx.fillStyle = T.bad; ctx.font = '11px ui-monospace, monospace'; ctx.textAlign = 'left';
      ctx.fillText('▸ OVERFLOW', x0 + gridW + 4, yTape + tapeH / 2 + 4);
      ctx.restore();
    }

    // ---- the draggable kept-tail bracket --------------------------------
    const kX = resultLefts[Math.max(0, Math.min(12, st.keep | 0))];
    const kRight = x0 + Math.min(gridW, tx(rec.total));
    rKeep = { x: kX, y: yKeep, w: Math.max(4, kRight - kX), h: 9 };
    ctx.save();
    ctx.fillStyle = alphaOf('ok', (st.keep | 0) === 0 ? 0.16 : (grab === 'keep' ? 0.6 : 0.34));
    ctx.fillRect(rKeep.x, rKeep.y, rKeep.w, rKeep.h);
    ctx.strokeStyle = grab === 'keep' ? T.okDeep : alphaOf('okDeep', 0.85);
    ctx.lineWidth = grab === 'keep' ? 2.4 : 1.5;
    ctx.beginPath(); ctx.moveTo(rKeep.x, rKeep.y - 4); ctx.lineTo(rKeep.x, rKeep.y + rKeep.h + 4); ctx.stroke();
    ctx.fillStyle = ctx.strokeStyle; ctx.font = '10px ui-monospace, monospace'; ctx.textAlign = 'center';
    ctx.fillText('◂▸', rKeep.x, rKeep.y + rKeep.h + 13);
    ctx.restore();
    lab(`keep last ${st.keep | 0}`, pad, yKeep + 9, T.okDeep, '10px ui-monospace, monospace');
    lab('tool results', pad, yKeep + 21, T.n9, '10px ui-monospace, monospace');

    // ---- band 2: per-turn prefill cost ----------------------------------
    lab('prefill cost', pad, yCost + 11, T.n12);
    lab('per turn', pad, yCost + 24, T.n11, '10px ui-monospace, monospace');
    lab('(token-equiv)', pad, yCost + 37, T.n9, '10px ui-monospace, monospace');

    let cMax = 1e-6;
    for (let i = 0; i < sim.N; i++) {
      cMax = Math.max(cMax, sim.arms.never.turns[i].cost, sim.arms[armKey].turns[i].cost);
    }
    const bw = gridW / sim.N;
    ctx.save();
    for (let i = 0; i < sim.N; i++) {
      const x = x0 + i * bw;
      const a = sim.arms[armKey].turns[i], n = sim.arms.never.turns[i];
      // active arm: split the bar into the cheap cached part and the full-price part
      const hAll = (a.cost / cMax) * costH;
      const hFull = (a.fullTok / cMax) * costH;
      ctx.fillStyle = alphaOf('accent', 0.38);
      ctx.fillRect(x + 0.7, yCost + costH - hAll, Math.max(1, bw - 1.4), hAll);
      ctx.fillStyle = a.fired ? T.bad : alphaOf('accent', 0.85);
      ctx.fillRect(x + 0.7, yCost + costH - hFull, Math.max(1, bw - 1.4), hFull);
      // the baseline, as a step outline over the top
      const hN = (n.cost / cMax) * costH;
      ctx.strokeStyle = alphaOf('n11', 0.85); ctx.lineWidth = 1.2;
      ctx.beginPath();
      ctx.moveTo(x, yCost + costH - hN); ctx.lineTo(x + bw, yCost + costH - hN); ctx.stroke();
      if (i === t) {
        ctx.strokeStyle = T.n14; ctx.lineWidth = 1.5;
        ctx.strokeRect(x + 0.2, yCost - 2, Math.max(1.5, bw - 0.4), costH + 4);
      }
    }
    ctx.strokeStyle = alphaOf('n14', 0.28); ctx.lineWidth = 1;
    ctx.beginPath(); ctx.moveTo(x0, yCost + costH + 0.5); ctx.lineTo(x0 + gridW, yCost + costH + 0.5); ctx.stroke();
    ctx.restore();
    lab(`bars = ${arm.short} (red = an edit turn, and the red height IS the re-prefill)   ·   grey steps = never compact`, x0 + 2, yCost + costH + 14, T.n9, '10px ui-monospace, monospace');
    lab(`peak ${fmtK(cMax)}`, x0 + gridW + 4, yCost + 10, T.n9, '10px ui-monospace, monospace');

    // ---- the cumulative panel -------------------------------------------
    const cumNever = Math.max(1e-9, base.cum);
    const rows = ARMS.map((a) => ({ a, v: sim.arms[a.key].turns[t] }));
    let vMax = 1e-9; for (const row of rows) vMax = Math.max(vMax, row.v.cum);
    const barX = pad + 136, barW = Math.max(90, Math.min(320, page.W - barX - 210));
    lab('cumulative prefill cost through this turn — lower is better; 100% = never compacting', pad, yPanel - 9, T.n11, '11px ui-monospace, monospace');
    rows.forEach((row, k) => {
      const y = yPanel + k * 20, active = row.a.key === armKey;
      const ratio = row.v.cum / cumNever;
      const col = row.a.key === 'never' ? T.n10 : ratio < 0.9 ? T.ok : ratio <= 1.0 ? T.warn : T.bad;
      lab(row.a.short, pad + 2, y + 10, active ? T.n14 : T.n10, active ? '12px ui-monospace, monospace' : '11px ui-monospace, monospace');
      ctx.save();
      ctx.fillStyle = alphaOf('n9', 0.16); ctx.fillRect(barX, y, barW, 12);
      ctx.fillStyle = col; ctx.fillRect(barX, y, barW * Math.min(1, row.v.cum / vMax), 12);
      if (active) { ctx.strokeStyle = T.n14; ctx.lineWidth = 1.5; ctx.strokeRect(barX - 1.5, y - 1.5, barW + 3, 15); }
      ctx.restore();
      const tag = row.a.key === 'never'
        ? `${fmtK(row.v.cum)} TE   (baseline${row.v.over ? ' · OVERFLOWED' : ''})`
        : `${fmtK(row.v.cum)} TE   ${(100 * ratio).toFixed(1)}% of never compacting`;
      lab(tag, barX + barW + 8, y + 10, col, '12px ui-monospace, monospace');
    });

    page.probe = {
      t, N: sim.N,
      ratio: rec.cum / cumNever,
      occ: rec.occupancy,
      activeOver: rec.over,
      baseOver: base.over,
      nPins: cur.pins.size,
      cleared: rec.cumCleared,
      recompute: rec.fullTok,
    };

    // ---- hover-to-inspect ------------------------------------------------
    if (page.pointer.over && !grab) {
      const p = page.pointer;
      if (p.y >= rTape.y - 6 && p.y <= rTape.y + rTape.h + 6) {
        for (const hit of tapeRects) {
          if (p.x >= hit.x && p.x <= hit.x + Math.max(3, hit.w)) {
            const b = cur.blocks[hit.i], sS = rec.state[hit.i];
            const eTok = sS === LIVE ? b.tokens : sS === STUB ? STUB_TOKENS : sS === SUMMARY ? rec.sumTok[hit.i] : 0;
            page.setTip([
              `block ${b.id} — ${KINDS[b.kind].label}, turn ${b.turn + 1}`,
              `age ${t - b.turn} turn${t - b.turn === 1 ? '' : 's'}   ·   written ${fmtK(b.tokens)} tokens   ·   costing ${fmtK(eTok)} now`,
              `${(100 * eTok / Math.max(1, rec.total)).toFixed(1)}% of the ${fmtK(rec.total)} tokens currently in context`,
              blockReason(hit.i, rec, st, keptTailSet),
              `click to ${cur.pins.has(b.id) ? 'un' : ''}pin`,
            ].join('\n'));
            break;
          }
        }
      } else if (p.y >= rCost.y - 4 && p.y <= rCost.y + rCost.h + 4 && p.x >= x0 && p.x <= x0 + gridW) {
        const i = Math.max(0, Math.min(sim.N - 1, Math.floor((p.x - x0) / bw)));
        const a = sim.arms[armKey].turns[i], n = sim.arms.never.turns[i];
        page.setTip([
          `turn ${i + 1} — ${arm.short}`,
          `cached prefix ${fmtK(a.cachedTok)} tokens × ${st.cache.toFixed(2)} = ${fmtK(a.cachedTok * st.cache)} TE`,
          `full-price prefill ${fmtK(a.fullTok)} tokens = ${fmtK(a.fullTok)} TE`,
          `turn cost ${fmtK(a.cost)} TE   ·   never compacting ${fmtK(n.cost)} TE`,
          a.fired ? `an edit fired here: ${a.nCleared} tool result(s) cleared, ${fmtK(a.clearedTok + a.foldedTok)} tokens freed — and the prefix was invalidated from block ${a.editIdx}` : `no edit — the cached prefix survived intact`,
        ].join('\n'));
      }
    }

    // ---- readout ---------------------------------------------------------
    const ratio = rec.cum / cumNever;
    const dir = Math.abs(ratio - 1) < 5e-4
      ? 'exactly at parity with'
      : ratio < 1 ? `${((1 - ratio) * 100).toFixed(1)}% CHEAPER than` : `${((ratio - 1) * 100).toFixed(1)}% MORE EXPENSIVE than`;
    let o = `policy: ${arm.name} — ${arm.desc}\n`;
    o += `turn ${t + 1}/${sim.N}: context ${fmtK(rec.total)} / ${fmtK(W)} tokens (${pct1(rec.occupancy)} of the window, threshold ${st.thresh | 0}%), ${fmtK(rec.cumCleared)} tokens cleared so far, keeping the last ${st.keep | 0} tool result${(st.keep | 0) === 1 ? '' : 's'}${cur.pins.size ? ` + ${cur.pins.size} pinned` : ''}    tier:${r.name}\n`;
    o += `this turn: ${fmtK(rec.cachedTok)} tokens re-read from cache at ×${st.cache.toFixed(2)} + ${fmtK(rec.fullTok)} tokens prefilled at full price = ${fmtK(rec.cost)} TE   (never compacting: ${fmtK(base.cost)} TE, context ${fmtK(base.total)} tokens${base.over ? ', OVERFLOWED' : ''})\n`;
    o += `cumulative: ${fmtK(rec.cum)} TE = ${(100 * ratio).toFixed(1)}% of the never-compacting baseline (lower is better; 100% = parity) — ${dir} keeping everything\n`;
    if (armKey === 'never') {
      o += base.over
        ? `the baseline has OVERFLOWED the window: past turn ${base.overflowAt + 1} this arm cannot run at all, so its cost beyond that point is not a real alternative — it is the reason compaction exists.`
        : `nothing is ever edited here, so the prefix is never invalidated and every turn is cheap. This arm is the honest counter-point — right up until it hits the wall.`;
    } else if (!rec.fired && rec.cumCleared === 0) {
      o += `occupancy is still under the threshold, so nothing has been edited yet and this arm is identical to the baseline. Drag the ▾ threshold left, or keep playing until the window fills.`;
    } else if (ratio > 1) {
      o += `COMPACTION IS LOSING HERE. At a cached re-read cost of ×${st.cache.toFixed(2)} a stable prefix is nearly free, while every edit re-prefills the tail at full price — so summarising costs more than remembering. Raise the cached cost, or the threshold, to flip it back.`;
      if (base.over) o += ` One caveat, in fairness to compaction: this baseline has OVERFLOWED the window, so it is only "cheaper" in the sense that a run which cannot happen is cheap. Widen the window until the grey arm stays under it to make the comparison a real one.`;
    } else {
      o += `the edits are paying for themselves at ×${st.cache.toFixed(2)}: each one costs a one-turn re-prefill spike (the red bars) and buys a permanently shorter prefix for every turn after it. Drop the cached re-read cost toward 0 and watch that trade invert.`;
    }
    page.setReadout(o);
  },
}).then((page) => {
  window.__contextEditingPage = page;
  const q = new URLSearchParams(location.search);
  const tr = page.controls._transport;
  const num = (k, lo, hi, key, opts) => { if (q.has(k)) page.controls.set(key, Math.max(lo, Math.min(hi, +q.get(k))), opts || { rebuild: true }); };
  if (q.has('policy') && ARMS.some((a) => a.key === q.get('policy'))) page.controls.set('policy', q.get('policy'), { rebuild: true });
  // Every draggable handle has a URL twin -- --screenshot has no pointer, so the
  // manipulated state has to be reachable from the address bar.
  num('thresh', 20, 98, 'thresh');
  num('keep', 0, 12, 'keep');
  num('cache', 0, 1, 'cache');
  num('win', 8, 200, 'win');
  num('turns', 6, 60, 'turns');
  num('seed', 0, 99, 'seed');
  if (q.has('pins')) page.controls.set('pins', String(q.get('pins')).replace(/[^0-9,]/g, ''), { rebuild: true });
  if (q.has('hover')) {
    const [hx, hy] = q.get('hover').split(',').map(Number);
    page.pointer.x = hx; page.pointer.y = hy; page.pointer.over = true;
  }
  const deterministic = ['step', 'hover', 'thresh', 'keep', 'cache', 'win', 'turns', 'seed', 'pins', 'policy'].some((k) => q.has(k));
  if (deterministic && tr) tr.pause();
  // A hook that changed the conversation must rebuild BEFORE the seek, or
  // ?step= is clamped against the previous (shorter) step list.
  if (tr) tr.rebuild();
  if (q.has('step') && tr) tr.seek(parseInt(q.get('step'), 10));
  if (q.get('play') === '1' && tr) tr.play();
  page.redraw();
});
