// kv-eviction concept page -- keeping the KV cache a CONSTANT size while the
// context keeps growing, by deciding what to forget.
//
// Three eviction policies run over the SAME growing sequence, simulated side by
// side against a full-cache baseline:
//
//   1. sliding window   -- keep the last B tokens. Cheap, constant, and it
//                          BREAKS: the earliest tokens go first, and those are
//                          exactly the ones the attention distribution leans on.
//   2. sinks + window   -- keep the first S tokens forever plus the last B-S
//                          (StreamingLLM, arXiv:2309.17453). A couple of tokens
//                          rescue the score IN THIS TOY; the paper's own result
//                          is that FOUR initial tokens suffice on real models and
//                          that one or two do not, so do not read the toy's S=2
//                          as the published figure.
//   3. heavy hitters    -- accumulate attention per cached token and evict the
//                          lowest when the cache fills (H2O, arXiv:2306.14048).
//                          Keeps what was actually attended to, wherever it is.
//
// THE SCORE. Every policy holds the same number of slots, so cache size cannot
// separate them. What separates them is how much of the REAL attention mass the
// surviving tokens carry: retained = sum over kept tokens of a[t][i], where
// a[t] is the full-cache attention distribution at that step (it sums to 1, so
// the number reads directly as a percent of the full-cache baseline; 100% = the
// policy threw nothing away that this step wanted).
//
// THE ATTENTION IS SYNTHETIC AND DELIBERATELY SHAPED. There is no model behind
// the page, so the distribution is generated from three additive logit terms
// that stand in for what the papers above report: a strong bias toward the first
// few positions REGARDLESS of their content (the attention-sink phenomenon --
// this is the whole reason dropping token 0 is catastrophic), a recency term,
// and a fixed per-token content salience drawn deterministically from the seed
// (which is what makes some middle token a "heavy hitter" the window cannot
// see). Everything downstream -- the eviction order, the scores, the tooltips --
// is then computed honestly from that distribution.
//
// Interactive per the shared render framework's contract: the sequence
// AUTO-PLAYS + LOOPS so the cache fills, hits its cap and starts evicting;
// DRAG the budget handle (the ◂▸ at the right end of the budget bar) to resize
// B and the sink bracket handle (◂▸ under the first slots) to set S, with all
// three policy scores moving live under your hand; HOVER any slot for its
// accumulated attention and whether it survives and WHY.
import { mount } from '../framework/layout.js';
import { softmax, seededRandn } from '../framework/tensor.js';
import { T, alphaOf, mixColor, inkOn } from '../framework/theme.js';

const POLICIES = [
  { key: 'window', name: 'sliding window', short: 'window',
    desc: 'keep the last B tokens, evict everything older — constant memory, and the first tokens are the first to go' },
  { key: 'sink', name: 'sinks + sliding window', short: 'sinks+window',
    desc: 'keep the first S tokens forever plus the last B−S (StreamingLLM) — same budget, and the sinks stop the collapse' },
  { key: 'h2o', name: 'heavy-hitter eviction', short: 'heavy hitters',
    desc: 'accumulate attention per cached token; when the cache fills, evict the lowest (H2O) — keeps what was attended to, wherever it sits' },
];
const policyAt = (k) => POLICIES.find((p) => p.key === k) || POLICIES[0];

// Logit shape for the synthetic attention (see the header note). SINK_W is the
// content-independent pull toward position 0; REC_W the recency pull.
const SINK_W = 3.2, SINK_TAU = 1.2, REC_W = 2.0, REC_TAU = 3.0;

// One simulation of all three policies over one sequence. Everything the page
// draws is read back out of here, so the picture and the score can never drift.
let cur = null;

function simulate(st) {
  const N = st.N | 0, B = Math.max(2, Math.min(st.budget | 0, N));
  const S = Math.max(0, Math.min(st.sinks | 0, B - 1));
  const sal = seededRandn(st.seed | 0, N);             // fixed per-token content salience (vector)

  const keys = POLICIES.map((p) => p.key);
  const kept = {}, acc = {}, evictedAt = {};
  for (const k of keys) { kept[k] = new Uint8Array(N); acc[k] = new Float64Array(N); evictedAt[k] = new Int32Array(N).fill(-1); }
  const accTrue = new Float64Array(N);                 // full-cache accumulated attention
  const steps = [];

  for (let t = 0; t < N; t++) {
    // --- full-cache attention for this step (the baseline every score is against)
    const logit = new Float32Array(t + 1);
    for (let i = 0; i <= t; i++) {
      logit[i] = sal[i] + SINK_W * Math.exp(-i / SINK_TAU) + REC_W * Math.exp(-(t - i) / REC_TAU);
    }
    const a = softmax(logit);
    for (let i = 0; i <= t; i++) accTrue[i] += a[i];

    const retained = {}, evictedThis = {};
    for (const k of keys) {
      const K = kept[k];
      K[t] = 1;                                        // the new token always enters the cache
      // accumulate BEFORE eviction, and only over what is cached: a token that
      // was evicted earlier stops earning score, which is what makes H2O's
      // ranking a property of the policy rather than of the baseline.
      for (let i = 0; i <= t; i++) if (K[i]) acc[k][i] += a[i];

      let live = 0; for (let i = 0; i <= t; i++) if (K[i]) live++;
      evictedThis[k] = [];
      while (live > B) {
        let victim = -1;
        if (k === 'window') {                          // oldest resident
          for (let i = 0; i <= t && victim < 0; i++) if (K[i]) victim = i;
        } else if (k === 'sink') {                     // oldest resident that is not a sink
          for (let i = S; i <= t && victim < 0; i++) if (K[i]) victim = i;
          if (victim < 0) for (let i = 0; i <= t && victim < 0; i++) if (K[i]) victim = i;
        } else {                                       // lowest accumulated attention, never the current token
          let lo = Infinity;
          for (let i = 0; i < t; i++) if (K[i] && acc[k][i] < lo) { lo = acc[k][i]; victim = i; }
          if (victim < 0) { for (let i = 0; i <= t && victim < 0; i++) if (K[i]) victim = i; }
        }
        if (victim < 0) break;
        K[victim] = 0; evictedAt[k][victim] = t; evictedThis[k].push(victim); live--;
      }

      let m = 0; for (let i = 0; i <= t; i++) if (K[i]) m += a[i];
      retained[k] = m;
    }

    steps.push({
      t,
      a,
      masks: keys.reduce((o, k) => (o[k] = kept[k].slice(), o), {}),
      accs: keys.reduce((o, k) => (o[k] = acc[k].slice(), o), {}),
      accTrue: accTrue.slice(),
      evictedThis,
      retained,
      label: `token ${t}: ${t + 1} token${t ? 's' : ''} of context, cache holds ${Math.min(t + 1, B)} / ${B}`,
    });
  }
  cur = { N, B, S, steps, keys };
  return steps;
}

// Hit rects captured in draw() so the pointer layer can test against exactly
// what was painted. `grab` is the handle currently being dragged.
let rAttn = null, rSlots = null, rAcc = null, rBudget = null, rSink = null;
let slotW = 0, x0 = 0;
let grab = null;

const pct = (v) => (100 * v).toFixed(1) + '%';
const scoreColor = (v) => (v >= 0.8 ? T.ok : v >= 0.5 ? T.warn : T.bad);

// Why a slot is (or is not) in the cache, in the vocabulary of the active
// policy -- this is the sentence the page exists to teach.
function survivalReason(key, i, t, sim, rec) {
  const { B, S } = sim;
  if (i > t) return `not in the sequence yet — arrives at step ${i}`;
  const alive = !!rec.masks[key][i];
  if (key === 'window') {
    return alive
      ? `KEPT — inside the window (the last ${B} tokens, positions ${Math.max(0, t - B + 1)}…${t})`
      : `EVICTED at step ${evictStep(key, i, sim, t)} — older than the last ${B} tokens, and a window has no way to make an exception`;
  }
  if (key === 'sink') {
    if (alive && i < S) return `KEPT — attention sink: the first ${S} token${S === 1 ? '' : 's'} stay resident forever, whatever they say`;
    if (alive) return `KEPT — inside the window (the last ${B - S} of the budget; ${S} slot${S === 1 ? ' is' : 's are'} reserved for sinks)`;
    return `EVICTED at step ${evictStep(key, i, sim, t)} — outside both the sink prefix (i < ${S}) and the window`;
  }
  if (alive) {
    // rank among the live set by accumulated attention (1 = strongest)
    const acc = rec.accs.h2o; let rank = 1;
    for (let j = 0; j <= t; j++) if (rec.masks.h2o[j] && acc[j] > acc[i]) rank++;
    return `KEPT — heavy hitter: accumulated attention ${acc[i].toFixed(2)}, rank ${rank} of ${B} in the cache`;
  }
  return `EVICTED at step ${evictStep(key, i, sim, t)} — its accumulated attention was the lowest in the cache when a new token needed the slot`;
}
function evictStep(key, i, sim, t) {
  for (let s = 0; s <= t; s++) if (sim.steps[s].evictedThis[key].indexOf(i) >= 0) return String(s);
  return '?';
}

mount({
  mount: 'body',
  title: 'kv-eviction — a constant-size cache over a growing context',
  blurb: 'The context keeps growing; the cache must not. Every policy here holds the same number of slots — what differs is which tokens it throws away. The score is how much of the REAL attention mass (full cache = 100%) the survivors still carry. Sliding window is the policy that looks most reasonable and is the one that fails: it evicts the first tokens, which absorb a large share of attention regardless of what they say. Drag the sink bracket from 0 to 4 and watch the same budget recover. Drag the budget handle to resize the cache; hover any slot for whether it survives and why.',
  prefer: 'webgl2',
  aspect: '2 / 1',
  autoplay: true,
  animate: false,
  compare: { key: 'policy', a: 'window', b: 'sink', labelA: 'sliding window (naive)', labelB: 'sinks + window (StreamingLLM)' },
  challenges: [
    {
      goal: 'Break StreamingLLM: drag the sink count to 0 and watch it degenerate into the plain window.',
      hint: 'sinks S = 0 leaves the policy with nothing but the last B−0 tokens — the same set the sliding window keeps.',
      check: (api) => ({ solved: (api.state.sinks | 0) === 0 && (api.probe.sinkScore ?? 1) <= (api.probe.winScore ?? 0) + 1e-9,
        detail: `S=${api.state.sinks | 0} · sinks+window ${pct(api.probe.sinkScore ?? 0)} vs window ${pct(api.probe.winScore ?? 0)}` }),
    },
    {
      goal: 'Rescue it: get sinks+window to keep ≥ 70% of the attention mass without raising the budget.',
      hint: 'raise S — the first token alone is worth more than most of the window.',
      check: (api) => ({ solved: (api.probe.sinkScore ?? 0) >= 0.7, detail: `sinks+window retains ${pct(api.probe.sinkScore ?? 0)} (need ≥ 70.0%)` }),
    },
    {
      goal: 'Find a budget where heavy-hitter eviction keeps ≥ 90% of the attention mass.',
      hint: 'H2O keeps the tokens that actually earned attention — it needs fewer slots than a window to reach the same score.',
      check: (api) => ({ solved: (api.probe.h2oScore ?? 0) >= 0.9, detail: `heavy hitters retain ${pct(api.probe.h2oScore ?? 0)} at B=${api.state.budget | 0}` }),
    },
  ],
  controls: (c, page) => {
    c.select('policy', {
      label: 'eviction policy',
      value: 'window',
      options: POLICIES.map((p) => ({ value: p.key, label: p.name })),
    });
    c.slider('budget', { label: 'cache budget B (slots, constant)', min: 2, max: 24, step: 1, value: 8, rebuild: true });
    // Default 2, which is what this toy needs -- NOT what StreamingLLM reports.
    // The paper says four initial tokens suffice and one or two do not; that is
    // a statement about real models, and the difference is worth seeing, so the
    // slider opens on the toy's own answer and reaches 4 in one drag.
    c.slider('sinks', { label: 'sink tokens S (drag the bracket too)', min: 0, max: 6, step: 1, value: 2, rebuild: true });
    c.stepper('N', { label: 'context length (tokens)', min: 8, max: 48, value: 32, rebuild: true });
    c.slider('seed', { label: 'seed (token content)', min: 0, max: 99, step: 1, value: 7, rebuild: true });
    c.transport({ compute: () => simulate(page.state), speed: 4, loop: true });
  },

  // Direct manipulation: grab the budget handle to resize B, or the sink
  // bracket handle to set S. Both map a pointer x onto a slot count, so the
  // handle lands exactly where the cursor is and every score moves with it.
  onPointer: (page, ev) => {
    if (!cur || !rSlots) return;
    const slotAt = (x) => Math.round((x - x0) / Math.max(1, slotW));
    if (ev.type === 'down') {
      grab = null;
      const near = (rect) => rect && ev.x >= rect.x - 10 && ev.x <= rect.x + rect.w + 10 && ev.y >= rect.y - 6 && ev.y <= rect.y + rect.h + 6;
      if (near(rBudget)) grab = 'budget';
      else if (near(rSink)) grab = 'sinks';
      if (grab) {
        const t = page.controls._transport; if (t) t.pause();
        page.controls.set(grab, grab === 'budget'
          ? Math.max(2, Math.min(24, slotAt(ev.x)))
          : Math.max(0, Math.min(6, slotAt(ev.x))), { rebuild: true });
      }
    } else if (ev.type === 'up' || ev.type === 'leave') {
      grab = null;
    } else if (ev.type === 'move' && grab && page.pointer.down) {
      const v = grab === 'budget' ? Math.max(2, Math.min(24, slotAt(ev.x))) : Math.max(0, Math.min(6, slotAt(ev.x)));
      if ((page.state[grab] | 0) !== v) page.controls.set(grab, v, { rebuild: true });
    }
  },

  draw: (page) => {
    const r = page.renderer, ctx = page.ctx, st = page.state;
    r.clear(T.n0);
    const sim = cur; if (!sim || !sim.steps.length) return;
    const N = sim.N, B = sim.B, S = sim.S;
    const s = page.step();
    const t = s ? Math.min(s.t, N - 1) : N - 1;
    const rec = sim.steps[t];
    const key = policyAt(st.policy).key;
    const mask = rec.masks[key];

    // ---- geometry -------------------------------------------------------
    const pad = 14, gut = 108;
    slotW = Math.max(7, Math.min(30, (page.W - 2 * pad - gut - 8) / N));
    x0 = pad + gut;
    const gridW = slotW * N;
    const topY = 44;
    const availH = Math.max(180, page.H - topY - 14);
    const attnH = Math.round(availH * 0.28);
    const slotH = Math.max(18, Math.min(34, Math.round(availH * 0.13)));
    const accH = Math.round(availH * 0.15);
    const yAttnTop = topY + 12, yAttnBase = yAttnTop + attnH;
    const ySlot = yAttnBase + 15;
    const yBudget = ySlot + slotH + 5;
    const yAcc = yBudget + 26;
    const yPanel = yAcc + accH + 24;
    rAttn = { x: x0, y: yAttnTop, w: gridW, h: attnH };
    rSlots = { x: x0, y: ySlot, w: gridW, h: slotH };
    rAcc = { x: x0, y: yAcc, w: gridW, h: accH };

    const lab = (txt, x, y, col, font) => r.label(txt, x, y, { color: col || T.n11, font: font || '11px ui-monospace, monospace' });

    // ---- row 1: this step's full-cache attention -------------------------
    let aMax = 1e-6; for (let i = 0; i <= t; i++) aMax = Math.max(aMax, rec.a[i]);
    lab('attention', pad, yAttnTop + 12, T.n12);
    lab('this step', pad, yAttnTop + 25, T.n11, '10px ui-monospace, monospace');
    lab('(full cache', pad, yAttnTop + 37, T.n9, '10px ui-monospace, monospace');
    lab('= the truth)', pad, yAttnTop + 49, T.n9, '10px ui-monospace, monospace');
    ctx.save();
    for (let i = 0; i < N; i++) {
      const x = x0 + i * slotW;
      if (i > t) { ctx.fillStyle = alphaOf('n6', 0.18); ctx.fillRect(x + 1, yAttnBase - 2, Math.max(1, slotW - 2), 2); continue; }
      const h = Math.max(1.5, (rec.a[i] / aMax) * attnH);
      const isSink = key === 'sink' && i < S;
      ctx.fillStyle = mask[i] ? (isSink ? T.warn : T.accent) : alphaOf('n9', 0.45);
      ctx.fillRect(x + 1, yAttnBase - h, Math.max(1, slotW - 2), h);
    }
    ctx.strokeStyle = alphaOf('n14', 0.25); ctx.lineWidth = 1;
    ctx.beginPath(); ctx.moveTo(x0, yAttnBase + 0.5); ctx.lineTo(x0 + gridW, yAttnBase + 0.5); ctx.stroke();
    ctx.restore();
    lab('grey bar = mass this policy already threw away', x0 + 2, yAttnTop - 2, T.n9, '10px ui-monospace, monospace');

    // ---- row 2: the cache slots -----------------------------------------
    lab('KV cache', pad, ySlot + slotH * 0.45, T.n12);
    lab(`${B} slots`, pad, ySlot + slotH * 0.45 + 13, T.accent, '10px ui-monospace, monospace');
    let accMax = 1e-6; for (let i = 0; i < N; i++) accMax = Math.max(accMax, rec.accTrue[i]);
    ctx.save();
    for (let i = 0; i < N; i++) {
      const x = x0 + i * slotW, w = Math.max(2, slotW - 2);
      if (i > t) {                                     // not in the sequence yet
        ctx.fillStyle = T.n1; ctx.fillRect(x + 1, ySlot, w, slotH);
        ctx.strokeStyle = alphaOf('n6', 0.5); ctx.setLineDash([2, 3]); ctx.lineWidth = 1;
        ctx.strokeRect(x + 1.5, ySlot + 0.5, w - 1, slotH - 1); ctx.setLineDash([]);
        continue;
      }
      if (mask[i]) {
        const heat = Math.min(1, rec.accTrue[i] / accMax);
        const hue = key === 'sink' && i < S ? T.warn : key === 'h2o' ? T.violet : T.accent;
        ctx.fillStyle = mixColor(T.n0, hue, 0.20 + 0.70 * heat);
        ctx.fillRect(x + 1, ySlot, w, slotH);
        ctx.strokeStyle = alphaOf('n14', 0.18); ctx.lineWidth = 1;
        ctx.strokeRect(x + 1.5, ySlot + 0.5, w - 1, slotH - 1);
      } else {                                          // evicted: visibly gone
        ctx.fillStyle = alphaOf('n6', 0.20); ctx.fillRect(x + 1, ySlot, w, slotH);
        ctx.strokeStyle = alphaOf('n9', 0.55); ctx.lineWidth = 1;
        ctx.beginPath();
        ctx.moveTo(x + 2, ySlot + slotH - 2); ctx.lineTo(x + w, ySlot + 2);
        ctx.stroke();
      }
    }
    // the token being generated right now
    ctx.strokeStyle = T.n14; ctx.lineWidth = 2;
    ctx.strokeRect(x0 + t * slotW + 0.5, ySlot - 1.5, Math.max(2, slotW - 1), slotH + 3);
    ctx.restore();
    if (slotW >= 18) {
      ctx.save(); ctx.font = '9px ui-monospace, monospace'; ctx.textAlign = 'center';
      for (let i = 0; i < N; i++) {
        if (i > t || !mask[i]) continue;
        const fill = mixColor(T.n0, key === 'sink' && i < S ? T.warn : key === 'h2o' ? T.violet : T.accent, 0.20 + 0.70 * Math.min(1, rec.accTrue[i] / accMax));
        ctx.fillStyle = inkOn(fill);
        ctx.fillText(String(i), x0 + i * slotW + slotW / 2, ySlot + slotH / 2 + 3);
      }
      ctx.restore();
    }

    // ---- the two draggable handles: budget B and sink prefix S -----------
    const bw = Math.min(gridW, B * slotW);
    rBudget = { x: x0, y: yBudget, w: bw, h: 8 };
    ctx.save();
    ctx.fillStyle = alphaOf('accent', grab === 'budget' ? 0.55 : 0.30);
    ctx.fillRect(rBudget.x, rBudget.y, rBudget.w, rBudget.h);
    ctx.strokeStyle = grab === 'budget' ? T.accent : alphaOf('accent', 0.8); ctx.lineWidth = grab === 'budget' ? 2 : 1.4;
    ctx.beginPath(); ctx.moveTo(rBudget.x + rBudget.w, rBudget.y - 4); ctx.lineTo(rBudget.x + rBudget.w, rBudget.y + rBudget.h + 4); ctx.stroke();
    ctx.fillStyle = ctx.strokeStyle; ctx.font = '10px ui-monospace, monospace'; ctx.textAlign = 'center';
    ctx.fillText('◂▸', rBudget.x + rBudget.w, rBudget.y + rBudget.h + 13);
    ctx.restore();
    lab(`budget B=${B}`, pad, yBudget + 8, T.accent, '10px ui-monospace, monospace');

    const sw = Math.max(6, S * slotW);
    rSink = { x: x0, y: yBudget + 12, w: sw, h: 7 };
    ctx.save();
    ctx.fillStyle = alphaOf('warn', S === 0 ? 0.18 : (grab === 'sinks' ? 0.6 : 0.35));
    ctx.fillRect(rSink.x, rSink.y, rSink.w, rSink.h);
    ctx.strokeStyle = grab === 'sinks' ? T.warn : alphaOf('warn', 0.85); ctx.lineWidth = grab === 'sinks' ? 2 : 1.4;
    ctx.beginPath(); ctx.moveTo(rSink.x + rSink.w, rSink.y - 4); ctx.lineTo(rSink.x + rSink.w, rSink.y + rSink.h + 4); ctx.stroke();
    ctx.fillStyle = ctx.strokeStyle; ctx.font = '10px ui-monospace, monospace'; ctx.textAlign = 'center';
    ctx.fillText('◂▸', rSink.x + rSink.w, rSink.y + rSink.h + 12);
    ctx.restore();
    lab(`sinks S=${S}`, pad, yBudget + 20, T.warn, '10px ui-monospace, monospace');

    // ---- row 3: accumulated attention (who the heavy hitters are) --------
    lab('accumulated', pad, yAcc + 11, T.n12);
    lab('attention', pad, yAcc + 23, T.n11, '10px ui-monospace, monospace');
    ctx.save();
    for (let i = 0; i <= t; i++) {
      const x = x0 + i * slotW, h = Math.max(1, (rec.accTrue[i] / accMax) * accH);
      ctx.fillStyle = mask[i] ? alphaOf('violet', 0.75) : alphaOf('n9', 0.30);
      ctx.fillRect(x + 1, yAcc, Math.max(1, slotW - 2), h);
    }
    ctx.restore();
    lab('tall bar + grey = a heavy hitter this policy lost', x0 + 2, yAcc + accH + 12, T.n9, '10px ui-monospace, monospace');

    // ---- the score panel: all three policies, same budget ----------------
    const rows = POLICIES.map((p) => ({ p, v: rec.retained[p.key] }));
    const barX = pad + 132, barW = Math.max(80, Math.min(300, page.W - barX - 150));
    lab('kept attention mass (100% = full cache, same budget for all three)', pad, yPanel - 8, T.n11, '11px ui-monospace, monospace');
    rows.forEach((row, k) => {
      const y = yPanel + k * 20, active = row.p.key === key;
      lab(row.p.short, pad + 2, y + 10, active ? T.n14 : T.n10, active ? '12px ui-monospace, monospace' : '11px ui-monospace, monospace');
      ctx.save();
      ctx.fillStyle = alphaOf('n9', 0.16); ctx.fillRect(barX, y, barW, 12);
      ctx.fillStyle = scoreColor(row.v); ctx.fillRect(barX, y, barW * Math.min(1, row.v), 12);
      if (active) { ctx.strokeStyle = T.n14; ctx.lineWidth = 1.5; ctx.strokeRect(barX - 1.5, y - 1.5, barW + 3, 15); }
      ctx.restore();
      lab(pct(row.v), barX + barW + 8, y + 10, scoreColor(row.v), '12px ui-monospace, monospace');
    });

    page.probe = {
      t, N, B, S,
      winScore: rec.retained.window,
      sinkScore: rec.retained.sink,
      h2oScore: rec.retained.h2o,
      active: rec.retained[key],
    };

    // ---- hover-to-inspect ------------------------------------------------
    if (page.pointer.over && !grab) {
      const p = page.pointer;
      const inRow = (rect) => rect && p.x >= rect.x && p.x <= rect.x + rect.w && p.y >= rect.y - 4 && p.y <= rect.y + rect.h + 4;
      if (inRow(rAttn) || inRow(rSlots) || inRow(rAcc)) {
        const i = Math.max(0, Math.min(N - 1, Math.floor((p.x - x0) / slotW)));
        const lines = [`token ${i}${i === t ? '  (being generated now)' : ''}`];
        if (i <= t) {
          lines.push(`attention this step  a[${t}][${i}] = ${rec.a[i].toFixed(4)}  (${pct(rec.a[i])} of the step)`);
          lines.push(`accumulated attention  Σ = ${rec.accTrue[i].toFixed(2)}`);
        }
        lines.push(survivalReason(key, i, t, sim, rec));
        page.setTip(lines.join('\n'));
      }
    }

    // ---- readout ---------------------------------------------------------
    const pol = policyAt(key);
    const live = Math.min(t + 1, B);
    const dWin = rec.retained.sink - rec.retained.window;
    let o = `policy: ${pol.name} — ${pol.desc}\n`;
    o += `step ${t + 1}/${N}: context ${t + 1} tokens, cache ${live}/${B} slots (constant), ${Math.max(0, t + 1 - live)} token${t + 1 - live === 1 ? '' : 's'} evicted    B=${B} S=${S}    tier:${r.name}\n`;
    o += `kept attention mass — window ${pct(rec.retained.window)} · sinks+window ${pct(rec.retained.sink)} · heavy hitters ${pct(rec.retained.h2o)}  (100% = full cache)\n`;
    o += t + 1 <= B
      ? `nothing has been evicted yet — the context is still smaller than the budget, so every policy scores 100%. Keep playing (or scrub right) until the cache fills.`
      : S === 0
      ? `S=0, so sinks+window IS the sliding window — drag the ◂▸ sink handle right: the first token alone is worth more than most of the window.`
      : `${S} sink slot${S === 1 ? '' : 's'} out of ${B} buy ${pct(dWin)} more attention mass than the plain window at the same budget — that is the whole StreamingLLM fix.`;
    page.setReadout(o);
  },
}).then((page) => {
  window.__kvEvictPage = page;
  const q = new URLSearchParams(location.search);
  const t = page.controls._transport;
  if (q.has('policy') && POLICIES.some((p) => p.key === q.get('policy'))) page.controls.set('policy', q.get('policy'));
  if (q.has('n')) page.controls.set('N', Math.max(8, Math.min(48, +q.get('n') | 0)), { rebuild: true });
  if (q.has('seed')) page.controls.set('seed', Math.max(0, Math.min(99, +q.get('seed') | 0)), { rebuild: true });
  // ?w= (budget) and ?sinks= are the headless stand-ins for dragging the two
  // handles -- --screenshot has no pointer, so the manipulated state needs a URL.
  if (q.has('w')) page.controls.set('budget', Math.max(2, Math.min(24, +q.get('w') | 0)), { rebuild: true });
  if (q.has('sinks')) page.controls.set('sinks', Math.max(0, Math.min(6, +q.get('sinks') | 0)), { rebuild: true });
  // ?hover=x,y fakes the cursor so the survival-reason tooltip is verifiable.
  if (q.has('hover')) {
    const [hx, hy] = q.get('hover').split(',').map(Number);
    page.pointer.x = hx; page.pointer.y = hy; page.pointer.over = true;
  }
  // Any deterministic-frame hook pauses the transport.
  if (q.has('step') || q.has('hover') || q.has('w') || q.has('sinks')) { if (t) t.pause(); }
  // A hook that changed the sequence length must rebuild BEFORE the seek below,
  // or ?step= is clamped against the previous (shorter) step list.
  if (t && (q.has('n') || q.has('seed') || q.has('w') || q.has('sinks'))) t.rebuild();
  if (q.has('step') && t) t.seek(parseInt(q.get('step'), 10));
  if (q.get('play') === '1' && t) t.play();
  page.redraw();
});
