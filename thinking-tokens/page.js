// thinking-tokens concept page — there is no reasoning ARCHITECTURE.
//
// A "reasoning model" emits its chain of thought between two LITERAL TOKENS,
// <think> and </think>. Everything between them is ordinary autoregressive
// generation — the same forward pass, the same softmax, the same sampler as any
// other token. What makes the trace happen is TRAINING, not architecture: a
// FORMAT reward (put the thinking inside the tags) alongside an ACCURACY reward
// (DeepSeek-R1, arXiv 2501.12948).
//
// Because the boundary is just a token, test-time compute is controllable from
// the DECODER, with no retraining and no architectural change (s1 / budget
// forcing, arXiv 2501.19393):
//   • SUPPRESS the end-of-thinking token (clamp its logit toward −∞) and append
//     a continuation like "Wait" — generation carries on, and often
//     self-corrects.
//   • FORCE the token early — the model must answer now, from a truncated
//     thought.
//
// Both levers are in this page and both act on the same quantity: one logit,
// one comparison, per step.
//
// The trade is NOT a straight line. Cost is: every forced token is a token
// generated and a step of latency, exactly linear. The benefit turns over —
// past the natural stopping point you buy redundant confirmation, then doubt
// without evidence, then repetition, and the answer can get WORSE. The
// accuracy-vs-length curve here is SCHEMATIC (it shows the reported SHAPE, not
// measured numbers). The refined, trained form of the same lever is a length
// penalty tied to problem difficulty (arXiv 2506.05256) — a panel at the bottom.
//
// HONESTY NOTE — what is real here and what is not.
// This page can run a REAL GPT-2 (124M) in your browser, on an explicit
// size-labelled click. GPT-2 is NOT a reasoning model: it was never trained
// with a format reward and it has NO <think> / </think> tokens. So the real
// mode does not pretend to show a reasoning trace. It shows the DECODER
// MECHANISM on a boundary token GPT-2 actually has (".", "\n", or
// <|endoftext|>): the page clamps that token's real logit on real logits,
// injects a real continuation, and the real continuation changes under your
// hand. That IS the mechanism reasoning models are trained to use — same
// clamp, same injection, different boundary token. The synthetic stand-in
// (default, no download) shows what the lever looks like on a model that WAS
// trained for it.
//
// Anchors: the sampling concept (how a token is chosen) + prefill-vs-decode
// (what a generated token costs). Reuses the verified forward in gpt2.js.
import { mount } from '../framework/layout.js';
import { T, alphaOf, rgbaToken } from '../framework/theme.js';
import { loadGPT2, GPT2_CONFIG } from '../real-attention/gpt2.js';

const TFJS = 'https://cdn.jsdelivr.net/npm/@huggingface/transformers@3.0.2';
const TOK_MODEL = 'Xenova/gpt2';
const WEIGHTS_URL = 'https://huggingface.co/gpt2/resolve/main/model.safetensors';
const CFG = GPT2_CONFIG.gpt2;
const REAL_PROMPT = 'Q: A shop sells pens at 3 for $2. What do 12 pens cost?\nA: Let me work it out.';
const MAX_REAL_THINK = 26;      // one full JS forward per token — keep it bounded
const REAL_ANSWER_TOKENS = 6;

// ---------------------------------------------------------------------------
// The synthetic stand-in: a scripted trace of the shape a trained reasoning
// model produces. Block 0 is what the model does on its own; every later block
// is what a suppressed boundary + an injected "Wait" buys you. The blocks are
// written to show the turnover: a real fix, then a redundant confirmation, then
// evidence-free doubt, then a repetition loop.
// ---------------------------------------------------------------------------
const PROBLEM = 'A shop sells pens at 3 for $2. What do 12 pens cost?';
const BLOCKS = [
  { toks: ['12', 'pens', '÷', '3', '=', '4', 'groups', '.', '4', '×', '2', '=', '6', '.', 'So', '$6', '.'],
    ans: '$6', ok: false, stop: { sl: 2.1, al: 0.8 },
    note: 'natural stop — the model believes it is done' },
  { toks: ['Wait', ',', '4', '×', '2', '=', '8', ',', 'not', '6', '.', 'So', '$8', '.'],
    ans: '$8', ok: true, stop: { sl: 3.4, al: 0.5 },
    note: 'self-correction — the win budget forcing is famous for' },
  { toks: ['Wait', ',', 're-check', ':', '12', '/', '3', '=', '4', ',', '4', '×', '$2', '=', '$8', '.', 'Still', '$8', '.'],
    ans: '$8', ok: true, stop: { sl: 4.6, al: 0.3 },
    note: 'redundant confirmation — costs tokens, buys nothing' },
  { toks: ['Wait', ',', 'unless', '3-for-$2', 'is', 'a', 'discount', 'off', '$1', 'each', '…', 'then', '12', '×', '$1', '=', '$12', '?'],
    ans: '$12', ok: false, stop: { sl: 5.5, al: 0.2 },
    note: 'doubt with no evidence overturns a CORRECT answer' },
  { toks: ['Wait', ',', 'unless', '…', '$12', '…', 'or', '$24', '…', 'or', '$12', '…'],
    ans: '$24', ok: false, stop: { sl: 6.4, al: 0.1 },
    note: 'repetition loop — the trace degenerates' },
];
// cumulative thinking-token count at the end of each block
const CUM = (() => { const a = []; let s = 0; for (const b of BLOCKS) { s += b.toks.length; a.push(s); } return a; })();
// SCHEMATIC accuracy at each block boundary — the reported SHAPE (rise, plateau,
// turnover), NOT a measurement of any model. Labelled as schematic on-canvas.
const ACC = [46, 74, 76, 61, 48];
const L_MAX = CUM[CUM.length - 1] + 14;

const BIAS_MIN = -24, BIAS_MAX = 6;        // BIAS_MIN is presented as "−∞ (clamped)"
// The logit axis is DATA-DRIVEN: a trained model's logits sit near 0, but a raw
// GPT-2 head's sit around -150, so a fixed axis pins every real bar off-scale.
const AXIS_SPAN = 22;                      // minimum width of the drawn axis

let M = { source: 'synthetic', status: 'idle', progress: 0, err: null, real: null, busy: false, sig: '', prompt: REAL_PROMPT };
let cur = null;                            // meta for the built trace (both modes)
let geom = null;                           // rects captured in draw for hit-testing
let grab = null;                           // 'bias' | 'force' while dragging
let tokenizer = null, gpt2 = null, loadStarted = false;

// ---------------------------------------------------------------------------
// synthetic candidate logits (deterministic in the seed) — so hover can show
// real arithmetic over a stated candidate set rather than a decorative number
// ---------------------------------------------------------------------------
const ALT = ['and', 'then', 'so', 'but', 'maybe', 'check', 'thus', 'also'];
function rnd(a) { a |= 0; a = (a + 0x6D2B79F5) | 0; let t = Math.imul(a ^ (a >>> 15), 1 | a); t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t; return ((t ^ (t >>> 14)) >>> 0) / 4294967296; }
function candFor(seed, b, k, tok, isBoundary, sp) {
  const h = Math.abs((seed * 7919 + b * 131 + k * 17) | 0);
  const r1 = rnd(h), r2 = rnd(h + 1), r3 = rnd(h + 2);
  if (isBoundary) {
    const alt = ALT[(h >>> 1) % ALT.length];
    return { sl: sp.sl, al: sp.al, altTok: alt,
      cands: [{ t: '</think>', l: sp.sl, stop: true }, { t: alt, l: sp.al }, { t: ALT[(h >>> 2) % ALT.length], l: sp.al - 0.9 - r2 }, { t: ALT[(h >>> 3) % ALT.length], l: sp.al - 1.8 - r3 }] };
  }
  const tl = 3.0 + 1.6 * r1, sl = tl - (4.2 + 3.4 * r2), alt = ALT[(h >>> 4) % ALT.length];
  return { sl, al: tl, altTok: tok,
    cands: [{ t: tok, l: tl }, { t: alt, l: tl - 0.8 - r3 }, { t: ALT[(h >>> 5) % ALT.length], l: tl - 1.7 - r2 }, { t: '</think>', l: sl, stop: true }] };
}

// softmax over a stated candidate set, with the bias applied to the stop row
function pOfStop(cands, bias) {
  let mx = -Infinity;
  const ls = cands.map((c) => c.l + (c.stop ? bias : 0));
  for (const l of ls) if (l > mx) mx = l;
  let Z = 0; const e = ls.map((l) => { const v = Math.exp(l - mx); Z += v; return v; });
  const i = cands.findIndex((c) => c.stop);
  return { p: i < 0 ? 0 : e[i] / Z, ps: e.map((v) => v / Z), biased: ls };
}

// ---------------------------------------------------------------------------
// build the synthetic trace under the current decoder settings
// ---------------------------------------------------------------------------
function buildSynth(st) {
  const bias = +st.bias, force = st.force | 0, inj = st.inj | 0, seed = st.seed | 0;
  const steps = [{ kind: 'open', tok: '<think>', L: 0, label: 'open the thinking span' }];
  let L = 0, completed = 0, truncated = false, stopped = false, capped = false, why = '';
  for (let b = 0; b < BLOCKS.length && !stopped; b++) {
    const B = BLOCKS[b];
    for (let k = 0; k < B.toks.length; k++) {
      if (force > 0 && L >= force) { truncated = true; stopped = true; why = `forced: the decoder emitted </think> at the ${force}-token budget, mid-thought`; break; }
      const isB = k === B.toks.length - 1;
      const c = candFor(seed, b, k, B.toks[k], isB, B.stop);
      L++;
      steps.push({ kind: 'think', tok: B.toks[k], blk: b, injected: b > 0, L, boundary: isB, label: `${b > 0 ? 'injected block ' + b : 'natural thought'} — token ${L}${isB ? ' (boundary decision)' : ''}`, ...c });
    }
    if (stopped) break;
    completed = b + 1;
    const emits = (B.stop.sl + bias) > B.stop.al;
    if (b < inj) { /* an injected continuation: the decoder suppressed this boundary by hand */ }
    else if (emits) { stopped = true; why = `natural: p(</think>) beat every alternative at the block-${b} boundary`; }
    else if (b === BLOCKS.length - 1) { stopped = true; capped = true; why = 'suppressed to the end of the scripted trace (the stand-in runs out of blocks)'; }
    else { /* suppressed → the decoder appends the next "Wait" and keeps going */ }
  }
  const A = completed > 0 ? BLOCKS[completed - 1] : null;
  steps.push({ kind: 'close', tok: '</think>', L, forced: truncated, label: truncated ? 'FORCED </think> at the budget' : 'the boundary token won — thinking ends' });
  for (const t of ['Answer', ':', A ? A.ans : '$?']) steps.push({ kind: 'answer', tok: t, L, ok: !!(A && A.ok), label: 'the answer span' });
  cur = { steps, L, completed, truncated, capped, why, ok: !!(A && A.ok), ans: A ? A.ans : '$?',
    note: A ? A.note : 'truncated before any block finished — no conclusion reached',
    boundary: '</think>', mode: 'synthetic' };
  return steps;
}

// ---------------------------------------------------------------------------
// REAL GPT-2. The download is opt-in (a size-labelled button, or ?autoload=1).
// GPT-2 has no <think>/</think>; the boundary token here is one it really has.
// ---------------------------------------------------------------------------
const withTimeout = (p, ms, label) => Promise.race([p, new Promise((_, rej) => setTimeout(() => rej(new Error(label + ' timed out — offline?')), ms))]);
const BOUNDARY = { period: { s: '.', fallback: 13 }, newline: { s: '\n', fallback: 198 }, eot: { s: '<|endoftext|>', fallback: 50256 } };

async function ensureReal(page) {
  if (gpt2 || loadStarted) return;
  loadStarted = true;
  try {
    M.status = 'loading-tok'; page.redraw();
    const { AutoTokenizer, env } = await withTimeout(import(/* @vite-ignore */ TFJS), 25000, 'transformers.js');
    env.allowLocalModels = false;
    tokenizer = await AutoTokenizer.from_pretrained(TOK_MODEL);
    M.status = 'loading-weights'; M.progress = 0; page.redraw();
    gpt2 = await loadGPT2(WEIGHTS_URL, CFG, (p) => { M.progress = p; page.redraw(); });
    M.status = 'ready'; M.source = 'real'; M.sig = '';
    decodeReal(page);
  } catch (e) { M.status = 'offline'; M.err = String((e && e.message) || e); loadStarted = false; page.redraw(); }
}

async function encode(s) { const enc = await tokenizer(s, { add_special_tokens: false }); return Array.from(enc.input_ids.data, Number); }

// full-vocab softmax stats for one decode step
function stepStats(logits, V, stopId, bias) {
  let mx = -Infinity;
  for (let v = 0; v < V; v++) { const l = logits[v] + (v === stopId ? bias : 0); if (l > mx) mx = l; }
  let Z = 0; for (let v = 0; v < V; v++) Z += Math.exp(logits[v] + (v === stopId ? bias : 0) - mx);
  // top-4 alternatives excluding the boundary token
  const top = [];
  for (let v = 0; v < V; v++) {
    if (v === stopId) continue;
    if (top.length < 4) { top.push(v); top.sort((a, b) => logits[b] - logits[a]); }
    else if (logits[v] > logits[top[3]]) { top[3] = v; top.sort((a, b) => logits[b] - logits[a]); }
  }
  const pStop = Math.exp(logits[stopId] + bias - mx) / Z;
  return { pStop, top, Z, mx, argmaxIsStop: (logits[stopId] + bias) >= logits[top[0]] };
}

async function decodeReal(page) {
  if (!gpt2 || !tokenizer || M.busy) return;
  M.busy = true; page.redraw();
  try {
    const st = page.state, bias = +st.bias, force = st.force | 0, inj = st.inj | 0;
    const bd = BOUNDARY[st.stopTok] || BOUNDARY.period;
    let stopId = bd.fallback;
    try { if (st.stopTok !== 'eot') { const e = await encode(bd.s); if (e.length === 1) stopId = e[0]; } } catch (_) {}
    const waitIds = await encode(' Wait');
    const ids = await encode(M.prompt);
    const dec = (id) => { try { return tokenizer.decode([id]); } catch (_) { return '·' + id; } };
    const steps = [{ kind: 'open', tok: '[prompt ends]', L: 0, label: 'the prompt ends — generation starts' }];
    let L = 0, used = 0, why = '', truncated = false, capped = false;
    const cap = Math.min(MAX_REAL_THINK, force > 0 ? force : MAX_REAL_THINK);
    while (true) {
      if (force > 0 && L >= force) { truncated = true; why = `forced: the decoder cut generation at the ${force}-token budget`; break; }
      if (L >= cap) { capped = true; why = `capped at ${cap} tokens (one full in-browser forward per token)`; break; }
      const { logits, V } = gpt2.logits(ids);
      const s = stepStats(logits, V, stopId, bias);
      if (s.argmaxIsStop) {
        if (used < inj) {                      // suppress + inject: the s1 lever, on real logits
          for (const t of waitIds) { L++; ids.push(t); steps.push({ kind: 'think', tok: dec(t), injected: true, L, boundary: false, label: `INJECTED continuation — token ${L}`, sl: logits[stopId], al: logits[s.top[0]], altTok: dec(s.top[0]), pStop: s.pStop, cands: realCands(logits, s, stopId, dec) }); }
          used++; continue;
        }
        why = `natural: the boundary token "${dec(stopId).replace(/\n/g, '⏎')}" was the argmax with bias ${bias.toFixed(1)}`;
        break;
      }
      const id = s.top[0]; L++; ids.push(id);
      steps.push({ kind: 'think', tok: dec(id), injected: false, L, boundary: false, label: `generated token ${L}`, sl: logits[stopId], al: logits[s.top[0]], altTok: dec(s.top[0]), pStop: s.pStop, cands: realCands(logits, s, stopId, dec) });
      if (L % 4 === 0) { M.real = { steps: steps.slice(), L, why, truncated, capped, partial: true }; page.controls._transport && (page.controls._transport._dirty = true); page.redraw(); await new Promise((r) => setTimeout(r, 0)); }
    }
    steps.push({ kind: 'close', tok: dec(stopId), L, forced: truncated, label: truncated ? 'FORCED boundary token at the budget' : 'the boundary token won' });
    ids.push(stopId);
    for (let k = 0; k < REAL_ANSWER_TOKENS; k++) {          // the "answer" span: decode on, bias off
      const { logits, V } = gpt2.logits(ids);
      let a = 0; for (let v = 1; v < V; v++) if (logits[v] > logits[a]) a = v;
      ids.push(a); steps.push({ kind: 'answer', tok: dec(a), L, ok: null, label: 'the answer span (bias off)' });
    }
    M.real = { steps, L, why, truncated, capped, partial: false, stopTok: dec(stopId), stopId };
  } catch (e) { M.err = String((e && e.message) || e); M.status = 'offline'; }
  M.busy = false;
  if (page.controls._transport) { page.controls._transport.rebuild(); }
  page.redraw();
}

function realCands(logits, s, stopId, dec) {
  const out = [{ t: dec(stopId).replace(/\n/g, '⏎'), l: logits[stopId], stop: true }];
  for (const v of s.top) out.push({ t: dec(v).replace(/\n/g, '⏎'), l: logits[v] });
  return out;
}

function buildReal(st) {
  const R = M.real;
  if (!R) { cur = { steps: [], L: 0, completed: 0, truncated: false, capped: false, why: 'decoding…', ok: null, ans: '', note: '', boundary: '.', mode: 'real' }; return []; }
  cur = { steps: R.steps, L: R.L, completed: 0, truncated: R.truncated, capped: R.capped, why: R.why,
    ok: null, ans: R.steps.filter((s) => s.kind === 'answer').map((s) => s.tok).join(''),
    note: 'GPT-2 is not a reasoning model — this is the decoder lever on a boundary token it really has',
    boundary: R.stopTok || '.', mode: 'real' };
  return R.steps;
}

function build(page) {
  const st = page.state;
  return M.source === 'real' ? buildReal(st) : buildSynth(st);
}

// SCHEMATIC accuracy as a function of thinking length (see the header note)
function accAt(L) {
  const xs = [0, ...CUM], ys = [21, ...ACC];
  if (L <= 0) return ys[0];
  for (let i = 1; i < xs.length; i++) if (L <= xs[i]) { const t = (L - xs[i - 1]) / (xs[i] - xs[i - 1]); return ys[i - 1] + t * (ys[i] - ys[i - 1]); }
  return Math.max(41, ys[ys.length - 1] - (L - xs[xs.length - 1]) * 0.45);
}

// ---------------------------------------------------------------------------
mount({
  mount: 'body',
  slug: 'thinking-tokens',
  title: 'thinking tokens — the reasoning trace is a decoder setting',
  blurb: 'There is no reasoning ARCHITECTURE. A reasoning model emits its chain of thought between two literal tokens, <think> and </think>, and everything between them is ordinary autoregressive generation — same forward, same softmax, same sampler. The trace exists because of TRAINING (a format reward beside an accuracy reward, DeepSeek-R1 2501.12948), not because of a new block. So the boundary is just a token, and one comparison per step decides it: clamp that token\'s logit toward −∞ and append "Wait" and the model keeps thinking, often self-correcting; force the token early and it must answer now (budget forcing, s1 2501.19393). Drag the stop-token logit, inject a continuation, or set a hard budget, and watch the trace and the answer move under your hand. The cost of forcing is exactly linear; the benefit is not — push past the natural stop and you buy redundant confirmation, then evidence-free doubt, then repetition, and the answer can get worse.',
  prefer: 'canvas2d',
  aspect: '16 / 9',
  autoplay: true,
  compare: { key: 'bias', a: 0, b: BIAS_MIN, labelA: 'bias 0 — the model stops where it wants', labelB: 'clamped to −∞ — the boundary token can never win', rebuild: true },
  controls: (c, page) => {
    c.slider('bias', { label: 'stop-token logit bias (drag the bar too)', min: BIAS_MIN, max: BIAS_MAX, step: 0.5, value: 0, rebuild: true });
    c.stepper('inj', { label: 'injected "Wait" continuations', min: 0, max: BLOCKS.length - 1, value: 0, rebuild: true });
    c.stepper('force', { label: 'force </think> at N tokens (0 = off)', min: 0, max: L_MAX, value: 0, rebuild: true });
    c.slider('seed', { label: 'seed (candidate logits)', min: 0, max: 99, step: 1, value: 7, rebuild: true });
    c.select('stopTok', { label: 'real-model boundary token', options: [{ value: 'period', label: '"." (sentence end)' }, { value: 'newline', label: '"\\n" (newline)' }, { value: 'eot', label: '<|endoftext|>' }], value: 'period', rebuild: true });
    c.button('⤵ suppress this boundary + inject "Wait"', () => {
      // One injection = suppress the boundary the model just reached and append
      // "Wait" there. That is per-boundary; the bias slider is the global clamp.
      const n = Math.min(BLOCKS.length - 1, (page.state.inj | 0) + 1);
      page.controls.set('inj', n, { rebuild: true, silent: true });
      page.controls.set('force', 0, { rebuild: true });
    });
    c.button('↺ reset the decoder', () => {
      page.controls.set('inj', 0, { rebuild: true, silent: true });
      page.controls.set('force', 0, { rebuild: true, silent: true });
      page.controls.set('bias', 0, { rebuild: true });
    });
    c.transport({ compute: () => build(page), speed: 9, loop: true });
    c.button('load real GPT-2 (~548 MB)', () => ensureReal(page));
  },
  onPointer: (page, ev) => {
    if (ev.type === 'up' || ev.type === 'leave') { grab = null; return; }
    if (!geom) return;
    const { logit, curve } = geom;
    if (ev.type === 'down') {
      grab = null;
      if (ev.x >= logit.x && ev.x <= logit.x + logit.w && ev.y >= logit.y && ev.y <= logit.y + logit.h) grab = 'bias';
      else if (ev.x >= curve.x && ev.x <= curve.x + curve.w && ev.y >= curve.y && ev.y <= curve.y + curve.h) grab = 'force';
    }
    if (!page.pointer.down) return;
    if (grab === 'bias') {
      const A = geom.axis || { lo: -14, hi: 8 };
      const v = A.lo + ((ev.x - logit.x) / logit.w) * (A.hi - A.lo);
      const sl = geom.sl == null ? 0 : geom.sl;
      const b = Math.max(BIAS_MIN, Math.min(BIAS_MAX, Math.round((v - sl) * 2) / 2));
      page.controls.set('bias', b, { rebuild: true });
    } else if (grab === 'force') {
      const L = Math.round(((ev.x - curve.x) / curve.w) * L_MAX);
      page.controls.set('force', Math.max(0, Math.min(L_MAX, L)), { rebuild: true });
    }
  },
  challenges: [
    { goal: 'Suppress the boundary token and make the model SELF-CORRECT — reach a right answer it did not reach on its own.',
      hint: 'Press "⤵ suppress + inject \'Wait\'" once, or drag the stop-token bar left until the boundary can no longer win.',
      check: (api) => ({ solved: api.probe.completed >= 2 && api.probe.ok === true, detail: `${api.probe.completed} block(s) of thinking · answer ${api.probe.ans}` }) },
    { goal: 'Force it to answer NOW — cut the thought off before the first block finishes.',
      hint: 'Set "force </think> at N tokens" to something below the natural stopping point, or click in the left half of the curve panel.',
      check: (api) => ({ solved: !!api.probe.truncated, detail: api.probe.truncated ? `truncated at ${api.probe.L} thinking tokens` : 'not truncated yet' }) },
    { goal: 'Push PAST the turnover — force so much thinking that the answer gets WORSE than the self-corrected one.',
      hint: 'Keep injecting continuations (or clamp the bias to −∞) until the trace runs into doubt and repetition.',
      check: (api) => ({ solved: api.probe.completed >= 4 && api.probe.ok === false, detail: `${api.probe.completed} block(s) · answer ${api.probe.ans} (the correct one is $8)` }) },
  ],
  draw: (page) => {
    const ctx = page.ctx, st = page.state, r = page.renderer, W = page.W, H = page.H;
    r.clear(T.n0);
    const bias = +st.bias, clamped = bias <= BIAS_MIN + 0.01;

    // in real mode, a decoder change means a genuine re-decode
    if (M.source === 'real' && !M.busy) {
      const sig = `${bias}|${st.force | 0}|${st.inj | 0}|${st.stopTok}`;
      if (sig !== M.sig) { M.sig = sig; decodeReal(page); }
    }

    const steps = (cur && cur.steps) || [];
    const idx = page.controls && page.controls._transport ? page.controls._transport.index : -1;
    const rec = idx >= 0 && idx < steps.length ? steps[idx] : null;
    page.probe = { completed: cur ? cur.completed : 0, ok: cur ? cur.ok : null, ans: cur ? cur.ans : '', L: cur ? cur.L : 0, truncated: !!(cur && cur.truncated), source: M.source, bias };

    // ---- banner ----------------------------------------------------------
    const ban = (() => {
      if (M.status === 'loading-tok') return { t: '↓ loading tokenizer…', c: T.goldDeep };
      if (M.status === 'loading-weights') return { t: `↓ downloading GPT-2 weights… ${(M.progress * 100) | 0}% (~548 MB)`, c: T.goldDeep };
      if (M.busy) return { t: '⟳ re-decoding on the real model with your new decoder settings…', c: T.goldDeep };
      if (M.source === 'real') return { t: `● real GPT-2 (124M) · boundary token "${(cur && cur.boundary || '.').replace(/\n/g, '⏎')}" — GPT-2 has NO <think>: same lever, real token, real logits`, c: T.okDeep };
      if (M.status === 'offline') return { t: '○ offline — synthetic stand-in (the real-model load failed)', c: T.goldDeep };
      return { t: '○ synthetic stand-in — the shape a TRAINED reasoning model produces. "load real GPT-2" runs the same lever on real logits.', c: T.n11 };
    })();
    ctx.save(); ctx.font = '11px ui-monospace, monospace'; ctx.textAlign = 'left'; ctx.textBaseline = 'top';
    ctx.fillStyle = ban.c; ctx.fillText(fit(ctx, ban.t, W - 28), 14, 8); ctx.restore();

    // ---- region geometry -------------------------------------------------
    const PAD = 14;
    const traceTop = 28, traceH = Math.round(H * 0.36);
    const panelTop = traceTop + traceH + 8, panelH = Math.round(H * 0.36);
    const stripTop = panelTop + panelH + 8, stripH = Math.max(48, H - stripTop - 6);
    const leftW = Math.round(W * 0.44);

    // ---- panel 1: the trace ---------------------------------------------
    frame(ctx, PAD, traceTop, W - 2 * PAD, traceH);
    lab(ctx, M.source === 'real' ? 'the generated sequence — one boundary token, one comparison per step' : 'the reasoning trace — everything between the two tokens is ordinary generation', PAD + 8, traceTop + 6, T.n11, '10px ui-monospace, monospace', W - 2 * PAD - 16);
    {
      const x0 = PAD + 10; let x = x0, y = traceTop + 30;
      const dense = steps.length > 55;                 // long traces get a smaller leading
      const maxX = W - PAD - 10, lh = dense ? 17 : 22;
      ctx.save(); ctx.textBaseline = 'middle';
      // the question
      ctx.font = '12px Georgia, "Times New Roman", serif'; ctx.fillStyle = T.n10;
      const q = M.source === 'real' ? M.prompt.replace(/\n/g, ' ⏎ ') : PROBLEM;
      ctx.fillText(q, x, y); y += lh + 2; x = x0;
      const chipRects = [];
      for (let i = 0; i < steps.length; i++) {
        const s = steps[i];
        const isTok = s.kind === 'think';
        const fs = dense ? 10 : 12;
        ctx.font = (s.kind === 'open' || s.kind === 'close') ? `bold ${fs - 1}px ui-monospace, monospace` : isTok ? `${fs}px ui-monospace, monospace` : `bold ${fs}px ui-monospace, monospace`;
        const label = String(s.tok).replace(/\n/g, '⏎');
        const w = ctx.measureText(label).width + 10;
        if (x + w > maxX) { x = x0; y += lh; }
        if (y > traceTop + traceH - 14) {            // ran out of panel: say so, never drop silently
          ctx.fillStyle = T.n9; ctx.font = '10px ui-monospace, monospace';
          ctx.fillText(`… +${steps.length - i} more tokens (scrub the transport to walk them)`, x0, y);
          break;
        }
        let bg = null, fg = T.n13;
        if (s.kind === 'open') { bg = T.accentBg; fg = T.accent; }
        else if (s.kind === 'close') { bg = s.forced ? alphaOf(T.bad, 0.16) : rgbaToken('n14', 0.06); fg = s.forced ? T.bad : T.violetDeep; }
        else if (s.kind === 'answer') { fg = cur && cur.ok === true ? T.okDeep : cur && cur.ok === false ? T.bad : T.n13; }
        else if (s.injected) { bg = T.goldBg; fg = T.goldDeep; }
        else if (s.boundary) { fg = T.n12; bg = rgbaToken('n14', 0.05); }
        if (bg) { ctx.fillStyle = bg; roundRect(ctx, x - 2, y - 9, w, 18, 4); ctx.fill(); }
        if (i === idx) { ctx.strokeStyle = T.accent; ctx.lineWidth = 1.5; roundRect(ctx, x - 3, y - 10, w + 2, 20, 5); ctx.stroke(); }
        ctx.fillStyle = fg; ctx.fillText(label, x + 3, y);
        chipRects.push({ x: x - 2, y: y - 9, w, h: 18, s });
        x += w + 3;
      }
      ctx.restore();
      geom = Object.assign(geom || {}, { chips: chipRects });
    }

    // ---- panel 2: the stop-token logit (draggable) -----------------------
    const lx = PAD, ly = panelTop, lw = leftW - PAD - 4, lh2 = panelH;
    frame(ctx, lx, ly, lw, lh2);
    lab(ctx, 'the boundary decision AT THIS STEP — drag the bar', lx + 8, ly + 6, T.n11, '10px ui-monospace, monospace', lw - 16);
    {
      const ax = lx + 14, aw = lw - 28, ay = ly + 30, ah = 44;
      const cands = (rec && rec.cands) || (steps.find((s) => s.kind === 'think' && s.cands) || {}).cands || [{ t: '</think>', l: 2.1, stop: true }, { t: 'Wait', l: 0.8 }];
      const stopRow = cands.find((c) => c.stop) || cands[0];
      const altRow = cands.find((c) => !c.stop) || cands[0];
      const sl = stopRow.l, al = altRow.l;
      const raw = cands.map((c) => c.l);
      const hi = Math.max(...raw) + 2;
      const lo = Math.min(Math.min(...raw), hi - AXIS_SPAN) - 2;
      const px = (v) => ax + ((Math.max(lo, Math.min(hi, v)) - lo) / (hi - lo)) * aw;
      const base = ax;                                      // bars grow from the axis floor
      const zero = base;
      const offScale = (sl + bias) < lo;
      // axis
      ctx.save(); ctx.strokeStyle = T.n5; ctx.lineWidth = 1;
      ctx.beginPath(); ctx.moveTo(ax, ay + ah + 12); ctx.lineTo(ax + aw, ay + ah + 12); ctx.stroke();
      if (lo < 0 && hi > 0) { ctx.setLineDash([3, 3]); ctx.beginPath(); ctx.moveTo(px(0), ay - 4); ctx.lineTo(px(0), ay + ah + 12); ctx.stroke(); ctx.setLineDash([]); }
      ctx.restore();
      // ghost = unbiased stop logit; solid = biased
      const yb = ay + 4, hb = 15;
      ctx.save();
      ctx.fillStyle = rgbaToken('n14', 0.10);
      ctx.fillRect(Math.min(zero, px(sl)), yb, Math.abs(px(sl) - zero), hb);
      const bx = px(sl + bias);
      ctx.fillStyle = clamped ? alphaOf(T.bad, 0.75) : alphaOf(T.violet, 0.75);
      ctx.fillRect(Math.min(zero, bx), yb, Math.abs(bx - zero), hb);
      // best alternative
      ctx.fillStyle = alphaOf(T.accent, 0.55);
      ctx.fillRect(Math.min(zero, px(al)), yb + hb + 6, Math.abs(px(al) - zero), hb);
      // drag handle
      ctx.fillStyle = clamped ? T.bad : T.violetDeep;
      ctx.beginPath(); ctx.moveTo(bx, yb - 6); ctx.lineTo(bx - 5, yb - 14); ctx.lineTo(bx + 5, yb - 14); ctx.closePath(); ctx.fill();
      if (offScale) { ctx.font = '9px ui-monospace, monospace'; ctx.textAlign = 'left'; ctx.textBaseline = 'bottom'; ctx.fillText('◀ off-scale', ax + 2, yb - 6); }
      ctx.font = '10px ui-monospace, monospace'; ctx.textBaseline = 'middle'; ctx.textAlign = 'left';
      ctx.fillStyle = clamped ? T.bad : T.violetDeep;
      ctx.fillText(`${cur && cur.mode === 'real' ? 'boundary "' + String(stopRow.t) + '"' : '</think>'}  ${(sl + bias).toFixed(2)}`, ax + 2, yb + hb / 2);
      ctx.fillStyle = T.accent; ctx.fillText(`best alternative "${String(altRow.t)}"  ${al.toFixed(2)}`, ax + 2, yb + hb + 6 + hb / 2);
      ctx.textAlign = 'center'; ctx.fillStyle = T.n9;
      ctx.textAlign = 'left'; ctx.fillText('logit ' + lo.toFixed(0), ax, ay + ah + 22);
      if (lo < 0 && hi > 0 && px(0) > ax + 30 && px(0) < ax + aw - 30) { ctx.textAlign = 'center'; ctx.fillText('0', px(0), ay + ah + 22); }
      ctx.textAlign = 'right'; ctx.fillText(hi.toFixed(0), ax + aw, ay + ah + 22);
      ctx.restore();
      geom = Object.assign(geom || {}, { logit: { x: ax, y: ay - 16, w: aw, h: ah + 20 }, sl, axis: { lo, hi } });

      // the arithmetic, spelled out
      const S = pOfStop(cands, bias);
      const isReal = !!(cur && cur.mode === 'real');
      const pStop = isReal && rec && rec.pStop != null ? rec.pStop : S.p;
      const lines = [
        `logit(stop) ${sl.toFixed(2)}  ${bias >= 0 ? '+' : '−'} ${Math.abs(bias).toFixed(2)} (bias) = ${(sl + bias).toFixed(2)}`,
        `p(stop) = exp(${(sl + bias).toFixed(2)}) / Σ = ${(pStop * 100).toFixed(pStop < 0.001 ? 4 : 2)} %   ${isReal ? 'over vocab' : 'over 4 cands'}`,
        (sl + bias) > al ? '→ the boundary token WINS: thinking ends here.' : '→ the boundary token LOSES: generation carries on.',
        clamped ? 'clamped to −∞ — it can never win, at any step.' : '',
      ];
      ctx.save(); ctx.font = '10.5px ui-monospace, monospace'; ctx.textAlign = 'left'; ctx.textBaseline = 'top';
      let ty = ay + ah + 32;
      for (const L2 of lines) { if (!L2) continue; ctx.fillStyle = L2.startsWith('→') ? ((sl + bias) > al ? T.violetDeep : T.okDeep) : L2.startsWith('clamped') ? T.bad : T.n11; ctx.fillText(fit(ctx, L2, aw), ax, ty); ty += 14; }
      ctx.restore();
    }

    // ---- panel 3: cost is a line, benefit turns over ---------------------
    const cx = leftW + 6, cy = panelTop, cw = W - PAD - cx, ch = panelH;
    frame(ctx, cx, cy, cw, ch);
    lab(ctx, 'cost is linear, benefit turns over — click here to set a budget', cx + 8, cy + 6, T.n11, '10px ui-monospace, monospace', cw - 16);
    {
      const gx = cx + 34, gy = cy + 24, gw = cw - 48, gh = ch - 56;
      const X = (L) => gx + (L / L_MAX) * gw, Y = (v) => gy + gh - (v / 100) * gh;
      ctx.save();
      ctx.strokeStyle = T.n5; ctx.lineWidth = 1;
      ctx.beginPath(); ctx.moveTo(gx, gy); ctx.lineTo(gx, gy + gh); ctx.lineTo(gx + gw, gy + gh); ctx.stroke();
      // cost: exactly linear
      ctx.strokeStyle = alphaOf(T.n10, 0.9); ctx.setLineDash([4, 3]); ctx.lineWidth = 1.4;
      ctx.beginPath(); ctx.moveTo(X(0), Y(0)); ctx.lineTo(X(L_MAX), Y(100)); ctx.stroke(); ctx.setLineDash([]);
      // benefit: schematic, and it turns over
      ctx.strokeStyle = T.violet; ctx.lineWidth = 2; ctx.beginPath();
      for (let L = 0; L <= L_MAX; L++) { const x2 = X(L), y2 = Y(accAt(L)); if (L === 0) ctx.moveTo(x2, y2); else ctx.lineTo(x2, y2); }
      ctx.stroke();
      // the peak
      let bestL = 0; for (let L = 0; L <= L_MAX; L++) if (accAt(L) > accAt(bestL)) bestL = L;
      ctx.fillStyle = alphaOf(T.violet, 0.25); ctx.beginPath(); ctx.arc(X(bestL), Y(accAt(bestL)), 5, 0, 7); ctx.fill();
      ctx.fillStyle = T.violetDeep; ctx.font = '9.5px ui-monospace, monospace'; ctx.textAlign = 'center'; ctx.textBaseline = 'bottom';
      ctx.fillText('turnover', X(bestL), Y(accAt(bestL)) - 7);
      // where we actually are
      const Lnow = cur ? cur.L : 0;
      ctx.strokeStyle = T.okDeep; ctx.lineWidth = 1.5; ctx.beginPath(); ctx.moveTo(X(Lnow), gy); ctx.lineTo(X(Lnow), gy + gh); ctx.stroke();
      ctx.fillStyle = T.okDeep; ctx.beginPath(); ctx.arc(X(Lnow), Y(accAt(Lnow)), 3.5, 0, 7); ctx.fill();
      ctx.textAlign = Lnow > L_MAX * 0.7 ? 'right' : 'left'; ctx.textBaseline = 'top';
      ctx.fillText(` ${Lnow} thinking tokens `, X(Lnow), gy + 2);
      // the forced budget
      const F = st.force | 0;
      if (F > 0) { ctx.strokeStyle = T.bad; ctx.setLineDash([2, 3]); ctx.beginPath(); ctx.moveTo(X(F), gy); ctx.lineTo(X(F), gy + gh); ctx.stroke(); ctx.setLineDash([]); ctx.fillStyle = T.bad; ctx.fillText(` budget ${F}`, X(F), gy + gh - 12); }
      // legend + axis labels
      ctx.textAlign = 'left'; ctx.fillStyle = T.violetDeep; ctx.fillText(fit(ctx, 'answer quality — SCHEMATIC shape, not a measurement', gw), gx + 2, gy + gh + 6);
      ctx.fillStyle = T.n10; ctx.fillText(fit(ctx, 'tokens generated ∝ latency — exactly linear', gw), gx + 2, gy + gh + 19);
      ctx.save(); ctx.translate(gx - 22, gy + gh / 2); ctx.rotate(-Math.PI / 2); ctx.textAlign = 'center'; ctx.fillStyle = T.n9; ctx.fillText('→ better / more', 0, 0); ctx.restore();
      ctx.restore();
      geom = Object.assign(geom || {}, { curve: { x: gx, y: gy, w: gw, h: gh } });
    }

    // ---- panel 4: the trained form of the same lever ---------------------
    frame(ctx, PAD, stripTop, W - 2 * PAD, stripH);
    {
      ctx.save(); ctx.textBaseline = 'top'; ctx.textAlign = 'left';
      ctx.font = 'bold 10px ui-monospace, monospace'; ctx.fillStyle = T.tealDeep;
      ctx.fillText('the trained version of the same lever', PAD + 10, stripTop + 6);
      ctx.font = '10.5px ui-monospace, monospace'; ctx.fillStyle = T.n11;
      const wl = wrap(ctx, 'Instead of clamping at decode time, train the length in: reward = accuracy − λ(difficulty) · length — easy problems get a short leash, hard ones a long one (arXiv 2506.05256).', W - 2 * PAD - 20, 2);
      for (let i = 0; i < wl.length; i++) ctx.fillText(wl[i], PAD + 10, stripTop + 20 + i * 13);
      // two leashes
      const bx0 = PAD + 10, bw0 = Math.min(260, W - 2 * PAD - 240), byy = stripTop + 48;
      const rows = [{ n: 'easy', f: 0.22, c: T.ok }, { n: 'hard', f: 0.78, c: T.warn }];
      for (let i = 0; i < rows.length; i++) {
        if (byy + i * 13 + 10 > stripTop + stripH) break;
        const y2 = byy + i * 13;
        ctx.fillStyle = T.n10; ctx.font = '9.5px ui-monospace, monospace'; ctx.fillText(rows[i].n, bx0, y2);
        ctx.fillStyle = rgbaToken('n14', 0.08); ctx.fillRect(bx0 + 32, y2 + 1, bw0, 8);
        ctx.fillStyle = alphaOf(rows[i].c, 0.75); ctx.fillRect(bx0 + 32, y2 + 1, bw0 * rows[i].f, 8);
        ctx.fillStyle = T.n9; ctx.fillText(`target thinking length under λ(${rows[i].n})`, bx0 + 40 + bw0, y2);
      }
      ctx.restore();
    }

    // ---- hover-to-inspect ------------------------------------------------
    if (page.pointer.over && !grab && geom && geom.chips) {
      const p = page.pointer;
      for (const c of geom.chips) {
        if (p.x >= c.x && p.x <= c.x + c.w && p.y >= c.y && p.y <= c.y + c.h) {
          page.setTip(tipFor(c.s, bias, !!(cur && cur.mode === 'real')));
          break;
        }
      }
    }

    // ---- readout ---------------------------------------------------------
    const Lnow = cur ? cur.L : 0;
    let o = `${M.source === 'real' ? 'REAL GPT-2 (124M)' : 'synthetic stand-in'} · bias ${clamped ? '−∞ (clamped)' : bias.toFixed(1)} · injected ${st.inj | 0} · budget ${(st.force | 0) || 'off'} · ${Lnow} thinking tokens${cur && cur.completed ? ' · ' + cur.completed + ' block(s)' : ''}    tier:${r.name}\n`;
    o += cur ? `${cur.why}  →  answer ${cur.ans}${cur.ok === true ? ' ✓' : cur.ok === false ? ' ✗' : ''}. ${cur.note}` : 'building…';
    page.setReadout(o);
  },
}).then((page) => {
  window.__thinkPage = page;
  const q = new URLSearchParams(location.search);
  const tr = page.controls._transport;
  const num = (k, lo, hi) => { if (q.has(k)) page.controls.set(k, Math.max(lo, Math.min(hi, +q.get(k))), { rebuild: true, silent: true }); };
  num('bias', BIAS_MIN, BIAS_MAX); num('inj', 0, BLOCKS.length - 1); num('force', 0, L_MAX); num('seed', 0, 99);
  if (q.has('stopTok') && BOUNDARY[q.get('stopTok')]) page.controls.set('stopTok', q.get('stopTok'), { rebuild: true, silent: true });
  if (q.has('prompt')) M.prompt = q.get('prompt').slice(0, 200);
  // ?hover=x,y fakes the cursor so the per-step tooltip is screenshot-verifiable
  if (q.has('hover')) { const [hx, hy] = q.get('hover').split(',').map(Number); page.pointer.x = hx; page.pointer.y = hy; page.pointer.over = true; }
  if (tr) tr.rebuild();
  // any deterministic-frame hook stops the autoplay so one URL replays one frame
  if (q.has('step') || q.has('hover')) { if (tr) tr.pause(); }
  if (q.has('step') && tr) tr.seek(parseInt(q.get('step'), 10));
  if (q.get('play') === '1' && tr) tr.play();
  page.redraw();
  // Large download: opt-in ONLY. The page opens on the synthetic stand-in.
  if (q.get('real') === '1' || q.get('autoload') === '1') ensureReal(page);
});

// --- small helpers ---------------------------------------------------------
function frame(ctx, x, y, w, h) {
  ctx.save(); ctx.fillStyle = T.n1; roundRect(ctx, x, y, w, h, 6); ctx.fill();
  ctx.strokeStyle = T.n4; ctx.lineWidth = 1; roundRect(ctx, x + 0.5, y + 0.5, w - 1, h - 1, 6); ctx.stroke(); ctx.restore();
}
function roundRect(ctx, x, y, w, h, r) {
  const rr = Math.min(r, w / 2, h / 2);
  ctx.beginPath();
  ctx.moveTo(x + rr, y); ctx.lineTo(x + w - rr, y); ctx.quadraticCurveTo(x + w, y, x + w, y + rr);
  ctx.lineTo(x + w, y + h - rr); ctx.quadraticCurveTo(x + w, y + h, x + w - rr, y + h);
  ctx.lineTo(x + rr, y + h); ctx.quadraticCurveTo(x, y + h, x, y + h - rr);
  ctx.lineTo(x, y + rr); ctx.quadraticCurveTo(x, y, x + rr, y);
}
function lab(ctx, text, x, y, color, font, maxW) {
  ctx.save(); ctx.font = font || '10px ui-monospace, monospace'; ctx.fillStyle = color; ctx.textAlign = 'left'; ctx.textBaseline = 'top';
  ctx.fillText(maxW ? fit(ctx, text, maxW) : text, x, y); ctx.restore();
}
// Truncate to a pixel width with an ellipsis -- a canvas has no overflow rule,
// so a label longer than its panel silently runs off the edge.
function fit(ctx, text, maxW) {
  let t = String(text);
  if (ctx.measureText(t).width <= maxW) return t;
  while (t.length > 1 && ctx.measureText(t + '…').width > maxW) t = t.slice(0, -1);
  return t + '…';
}
// Word-wrap into at most `rows` lines of `maxW` px; the last line is ellipsised
// rather than dropped, so nothing vanishes without a mark.
function wrap(ctx, text, maxW, rows) {
  const words = String(text).split(' '), out = []; let line = '';
  for (let i = 0; i < words.length; i++) {
    const t = line ? line + ' ' + words[i] : words[i];
    if (ctx.measureText(t).width > maxW && line) {
      out.push(line);
      if (out.length === rows - 1) { line = words.slice(i).join(' '); break; }
      line = words[i];
    } else line = t;
  }
  out.push(fit(ctx, line, maxW));
  return out.slice(0, rows);
}
function tipFor(s, bias, isReal) {
  const nm = String(s.tok).replace(/\n/g, '⏎');
  if (s.kind === 'open') return `<think>\nEverything after this token is ordinary generation — no new block, no new attention, no new anything.`;
  if (s.kind === 'close') return `${nm}\n${s.forced ? 'FORCED by the decoder at the budget — the model did not choose to stop here.' : 'the boundary token won this step, so thinking ended here.'}`;
  if (s.kind === 'answer') return `${nm}\nthe answer span — decoded after the boundary, with the bias off.`;
  const cands = s.cands || [];
  const S = pOfStop(cands, bias);
  const lines = [`step ${s.L}  "${nm}"${s.injected ? '   ← INJECTED by the decoder' : ''}`, ''];
  for (let i = 0; i < cands.length; i++) {
    const c = cands[i], l = c.l + (c.stop ? bias : 0);
    lines.push(`${c.stop ? '»' : ' '} "${String(c.t)}"  logit ${c.l.toFixed(2)}${c.stop && bias ? ` ${bias < 0 ? '−' : '+'} ${Math.abs(bias).toFixed(2)}` : ''} → ${l.toFixed(2)}   p ${(S.ps[i] * 100).toFixed(2)} %`);
  }
  const pS = isReal && s.pStop != null ? s.pStop : S.p;
  lines.push('', `p(stop) = exp(l_stop) / Σ exp(l_i) = ${(pS * 100).toFixed(pS < 0.001 ? 4 : 2)} %   ${isReal ? '(Σ over all 50 257 tokens)' : '(Σ over the 4 candidates above)'}`);
  lines.push((cands.find((c) => c.stop) || {}).l + bias > ((cands.find((c) => !c.stop) || {}).l || 0) ? 'the boundary wins here' : 'the boundary loses here — generation continues');
  return lines.join('\n');
}
