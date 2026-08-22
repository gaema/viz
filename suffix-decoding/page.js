// suffix-decoding concept page -- speculation with NO draft model at all.
//
// The proposal does not come from a second neural network. It comes from the
// text the system has ALREADY SEEN: the prompt, the conversation so far, the
// tokens this same generation has already emitted. Index all of it in a
// depth-bounded SUFFIX tree (every position of the corpus inserts one suffix,
// so a path from the root is an n-gram that really occurred and its count is
// how many positions produced it). To speculate, take the most recent tokens,
// find the LONGEST SUFFIX of them that the tree knows, and walk the most
// frequent continuation from there. The big model then verifies the whole
// proposal in ONE forward pass exactly as it would a draft model's, keeping the
// longest agreeing prefix.
//
// So the trade against a draft model (see the speculative-decoding page, which
// owns the verify-and-accept mechanism this one reuses) is: no draft weights,
// no draft forward passes -- but the proposal is only ever as good as what has
// been seen before. On an agent loop re-emitting a similar tool call, on a
// document being quoted back, on structured output with a fixed skeleton, that
// is very good. On genuinely novel prose there is nothing to copy and the
// scheme degrades to ordinary one-token-per-forward decoding.
//
// WHAT IS REAL HERE AND WHAT IS MODELLED -- the page says this on screen too:
//   REAL   the corpus, the suffix tree and its counts, the longest-suffix
//          match, the occurrence positions, the proposal, and the accepted
//          length (a literal longest-common-prefix against the reference).
//   MODEL  that the reference continuation is what the big model would emit.
//          Both the corpus and the reference come from ONE seeded process with
//          a repetitiveness parameter, which is exactly the knob the page is
//          about; the draft-model baseline curve is an analytic geometric
//          accept model, drawn dashed and labelled "modelled".
//
// Interactive: the transport steps match / propose / verify / commit and loops;
// drag the repetitiveness and proposal-length handles on the canvas; type into
// "extra corpus text" to create a repeat by hand and watch acceptance jump;
// hover a tree child for its continuation and how often it was seen, or a
// proposed token for why it was proposed and whether it survived.
import { mount } from '../framework/layout.js';
import { rng } from '../framework/tensor.js';
import { T, alphaOf, inkOn, rgbaToken } from '../framework/theme.js';

const PHASES = ['match', 'propose', 'verify', 'commit'];
const PHASE_TEXT = [
  'MATCH — take the most recent tokens and find the LONGEST SUFFIX of them that occurs in the tree.',
  'PROPOSE — walk on from the match, taking the most frequent continuation at each step. No model runs.',
  'VERIFY — one forward of the big model covers all proposed positions at once, exactly as for a draft.',
  'COMMIT — keep the agreeing prefix plus the model’s own next token — and index the new tokens too.',
];

// --- the corpus generator ---------------------------------------------------
// A stream of short "lines". With probability rho the next line is an exact
// REPEAT of a line already used; otherwise it is fresh. Fresh lines are either
// structured (a template with two fresh arguments — so even novel text shares a
// skeleton, which is why the tree still lands a token or two at rho = 0) or
// prose (words with no reusable shape at all).
const TEMPLATES = [
  ['read', 'the', 'file'],
  ['run', 'the', 'tests', 'for'],
  ['search', 'the', 'tree', 'for', 'symbol'],
  ['edit', 'the', 'file'],
  ['open', 'the', 'report', 'and', 'quote'],
  ['list', 'all', 'files', 'under'],
  ['check', 'the', 'log', 'of'],
];
const ARGS = ['parser', 'loader', 'index', 'router', 'cache', 'buffer', 'writer',
  'reader', 'engine', 'planner', 'mapper', 'walker', 'binder', 'shaper'];
const PROSE = ['the', 'a', 'and', 'of', 'light', 'river', 'glass', 'memory',
  'slowly', 'turned', 'toward', 'morning', 'without', 'name', 'quiet', 'field',
  'under', 'wide', 'sky', 'wrote', 'again', 'never', 'same', 'word', 'twice',
  'salt', 'hollow', 'blue', 'listening', 'far'];

const pick = (next, arr) => arr[Math.min(arr.length - 1, Math.floor(next() * arr.length))];

function freshLine(next) {
  if (next() < 0.55) {
    const t = pick(next, TEMPLATES);
    return [...t, pick(next, ARGS), pick(next, ARGS), '.'];
  }
  const n = 6 + Math.floor(next() * 3);
  const out = [];
  for (let i = 0; i < n; i++) out.push(pick(next, PROSE));
  out.push('.');
  return out;
}

/**
 * One token stream, split into the history and the reference continuation.
 *
 * Repetition is modelled the way it actually arises: as a verbatim SPAN copied
 * from earlier in the same stream — a quoted paragraph, a re-emitted tool call,
 * the body of a loop coming round again. Copying whole spans rather than
 * shuffling independent lines matters, and the difference is measurable: with
 * lines drawn independently, every line BOUNDARY is unpredictable even when the
 * lines themselves repeat, and the accepted length stays near zero at every
 * repetitiveness. A copied span makes the joins predictable too, which is
 * exactly the structure a suffix index exists to exploit.
 */
function genStream(seed, rho, corpusTok, contTok) {
  const next = rng((seed | 0) * 2654435761 + 12345);
  const toks = [];
  const total = corpusTok + contTok;
  while (toks.length < total) {
    if (toks.length > 24 && next() < rho) {
      // Copy from the RECENT past, not from anywhere: a loop repeats what it
      // just did, and a document is quoted near where it was read. Sourcing
      // spans uniformly over the whole stream instead turns high
      // repetitiveness into a scrambled mosaic of copies-of-copies, where a
      // context recurs constantly with a different continuation every time --
      // which measured WORSE than moderate repetitiveness and inverted the
      // very trend this page is about.
      const len = 10 + Math.floor(next() * 20);
      const back = 8 + Math.floor(next() * 56);
      const start = Math.max(0, toks.length - back);
      for (let i = 0; i < len && start + i < toks.length; i++) toks.push(toks[start + i]);
    } else {
      toks.push(...freshLine(next));
    }
  }
  return { corpus: toks.slice(0, corpusTok), cont: toks.slice(corpusTok) };
}

const tokenize = (s) => String(s || '').toLowerCase().replace(/[^a-z0-9.\-_ ]+/g, ' ')
  .split(/\s+/).filter(Boolean).slice(0, 40);

// --- the suffix tree --------------------------------------------------------
// Depth-bounded suffix trie: for EVERY position i of the corpus, insert the
// tokens at i, i+1, … i+D-1. A path from the root is therefore an n-gram that
// occurred, and node.count is the number of positions that produced it. This is
// the whole difference from a prefix tree (radix-attention): there, one path is
// inserted per request from its start; here, one path is inserted per POSITION.
function buildTree(tokens, D) {
  const root = { tok: null, count: 0, kids: new Map(), depth: 0 };
  let nodes = 1;
  for (let i = 0; i < tokens.length; i++) {
    let node = root; root.count++;
    for (let d = 0; d < D && i + d < tokens.length; d++) {
      const t = tokens[i + d];
      let c = node.kids.get(t);
      if (!c) { c = { tok: t, count: 0, kids: new Map(), depth: d + 1 }; node.kids.set(t, c); nodes++; }
      c.count++; node = c;
    }
  }
  return { root, nodes, size: tokens.length };
}

/** Walk `pat` from the root; null if the whole pattern is not a path. */
function walk(tree, pat) {
  let n = tree.root;
  for (const t of pat) { const c = n.kids.get(t); if (!c) return null; n = c; }
  return n;
}

// A context seen exactly ONCE carries no evidence: it has one continuation,
// with count 1, and following it blindly is how a naive longest-match ends up
// proposing whatever happened to come next that single time. (Measured while
// building this page: matching purely on length anchored almost every round on
// a once-seen span straddling two unrelated lines, and the accepted length
// collapsed to 0 across the whole repetitive regime.) So the match takes the
// longest suffix that has RECURRED — count >= MIN_COUNT — and backs off toward
// shorter, better-attested contexts otherwise. Frequency, not just length.
const MIN_COUNT = 2;

/** Longest RECURRING suffix of ctx (length <= D-1). Falls back to the root,
 *  whose children are the corpus unigrams, when nothing has recurred. */
function longestSuffixMatch(tree, ctx, D) {
  const cap = Math.min(D - 1, ctx.length);
  let longest = null;                       // longest match at any count, for the tip
  for (let m = cap; m >= 1; m--) {
    const pat = ctx.slice(ctx.length - m);
    const n = walk(tree, pat);
    if (!n) continue;
    if (!longest) longest = m;
    if (n.count >= MIN_COUNT) return { node: n, len: m, pat, longest };
  }
  return { node: tree.root, len: 0, pat: [], longest };
}

const bestChild = (node) => {
  let best = null;
  for (const c of node.kids.values()) if (!best || c.count > best.count) best = c;
  return best;
};

/** Propose L tokens. Re-anchors on the longest suffix at every step, which is
 *  what keeps it going past the depth cap. Records why each token was picked. */
function propose(tree, ctx0, L, D) {
  const ctx = ctx0.slice();
  const out = [];
  for (let i = 0; i < L; i++) {
    const m = longestSuffixMatch(tree, ctx, D);
    const c = bestChild(m.node);
    if (!c) break;
    const sibs = [...m.node.kids.values()].sort((a, b) => b.count - a.count).slice(0, 4)
      .map((s) => ({ tok: s.tok, count: s.count }));
    out.push({ tok: c.tok, count: c.count, ctxCount: m.node.count, matchLen: m.len, pat: m.pat, sibs });
    ctx.push(c.tok);
  }
  return out;
}

// --- one whole run ----------------------------------------------------------
// Everything below is REAL: the tree, the match, the proposal, and the accepted
// length as a literal longest-common-prefix against the reference continuation.
function simulate(st, rho) {
  const D = st.depth | 0, L = st.plen | 0, R = st.rounds | 0;
  // ~7 tokens to a line; enough reference text to outlast every round even if
  // every proposal lands in full.
  const g = genStream(st.seed, rho, (st.corpus | 0) * 7, 24 + R * (L + 1));
  const extra = tokenize(st.extra);
  // Typed text goes at the HEAD of the corpus — it is a document pasted into
  // the prompt. Putting it there leaves the recent context (the tail) alone, so
  // typing a line changes only what the tree KNOWS, never where the match
  // starts from. That separation is what makes the edit a clean experiment.
  const base = extra.concat(g.corpus);
  const seen = base.slice();          // grows with every committed token
  const truth = g.cont;
  const rounds = [];
  let ti = 0, accSum = 0, emitted = 0;
  for (let r = 0; r < R && ti < truth.length; r++) {
    const tree = buildTree(seen, D);
    const ctx = seen.slice(Math.max(0, seen.length - (D - 1)));
    const m = longestSuffixMatch(tree, ctx, D);
    const prop = propose(tree, ctx, L, D);
    let acc = 0;
    while (acc < prop.length && ti + acc < truth.length && prop[acc].tok === truth[ti + acc]) acc++;
    const commit = truth.slice(ti, Math.min(truth.length, ti + acc + 1));
    for (const t of commit) seen.push(t);
    rounds.push({
      r, tree, ctx, match: m, prop, acc, commit,
      truth: truth.slice(ti, ti + Math.max(prop.length, acc + 1)),
      corpusLen: seen.length - commit.length, seenAt: seen.slice(),
    });
    accSum += acc; emitted += commit.length; ti += commit.length;
  }
  const n = rounds.length || 1;
  return { rounds, base, extraLen: extra.length, truth, mean: accSum / n, emitted, tpf: emitted / n, D, L, R: rounds.length };
}

// A draft model's mean accepted length, MODELLED (drawn dashed, labelled).
// Per-token agreement a is a property of the model PAIR, not of the text, so it
// barely moves with repetitiveness — that flatness is the whole comparison.
const draftAccept = (a0, rho) => Math.min(0.985, a0 + (1 - a0) * 0.18 * rho);
function draftMean(a0, rho, L) {
  const a = draftAccept(a0, rho);
  return a >= 0.999 ? L : (a - Math.pow(a, L + 1)) / (1 - a);
}

let run = null, runSig = '';
let sweep = null, sweepSig = '';
let geom = null;
let grab = null;                       // 'rho' | 'plen' while dragging

const sigOf = (st) => `${st.seed | 0}|${st.plen | 0}|${st.rounds | 0}|${st.depth | 0}|${st.corpus | 0}|${st.extra || ''}`;

function ensureRun(st) {
  const sig = sigOf(st) + '|' + (+st.rho).toFixed(3);
  if (sig !== runSig) { run = simulate(st, +st.rho); runSig = sig; }
  return run;
}
// The curve averages SEEDS, the readout reports the single run on screen.
// One run of a handful of rounds is a small sample: measured while building
// this page, a single-seed curve was non-monotone by more than its own trend
// (0.71 at repetitiveness 0.95 against 1.43 at 0.30), which is noise wearing
// the shape of a finding. Averaging four seeds is what makes the curve a claim
// rather than an anecdote — and the two numbers are labelled separately on
// screen precisely because they are not the same measurement.
const SWEEP_SEEDS = 4;

function ensureSweep(st) {
  const sig = sigOf(st) + '|' + (+st.draftA).toFixed(3);
  if (sig === sweepSig) return sweep;
  const pts = [];
  for (let i = 0; i <= 20; i++) {
    const rho = i / 20;
    let s = 0;
    for (let k = 0; k < SWEEP_SEEDS; k++) s += simulate({ ...st, seed: (st.seed | 0) + k * 17 }, rho).mean;
    pts.push({ rho, suffix: s / SWEEP_SEEDS, draft: draftMean(+st.draftA, rho, st.plen | 0) });
  }
  sweep = pts; sweepSig = sig;
  // Headless-verification hook, the same role as the ?step= / ?hover= URL hooks:
  // a screenshot cannot read a polyline, so the curve is also readable as data.
  if (typeof window !== 'undefined') window.__suffixCurve = pts;
  return sweep;
}

// --- small drawing helpers --------------------------------------------------
function chip(ctx, x, y, w, h, text, fill, ink, opts = {}) {
  ctx.save();
  ctx.fillStyle = fill;
  if (ctx.roundRect) { ctx.beginPath(); ctx.roundRect(x, y, w, h, 4); ctx.fill(); } else ctx.fillRect(x, y, w, h);
  if (opts.dash) { ctx.setLineDash([3, 3]); ctx.strokeStyle = opts.stroke || T.n8; ctx.lineWidth = 1; ctx.strokeRect(x + 0.5, y + 0.5, w - 1, h - 1); ctx.setLineDash([]); }
  else if (opts.stroke) { ctx.strokeStyle = opts.stroke; ctx.lineWidth = 1.6; ctx.strokeRect(x + 0.8, y + 0.8, w - 1.6, h - 1.6); }
  ctx.fillStyle = ink; ctx.font = opts.font || '10.5px ui-monospace, monospace';
  ctx.textAlign = 'center'; ctx.textBaseline = 'middle';
  ctx.fillText(text, x + w / 2, y + h / 2 + 0.5);
  if (opts.strike) { ctx.strokeStyle = ink; ctx.lineWidth = 1.1; ctx.beginPath(); ctx.moveTo(x + 4, y + h / 2); ctx.lineTo(x + w - 4, y + h / 2); ctx.stroke(); }
  ctx.restore();
  return { x, y, w, h };
}

function track(ctx, rect, frac, label, hue, active) {
  const { x, y, w, h } = rect;
  ctx.save();
  ctx.fillStyle = T.n3;
  if (ctx.roundRect) { ctx.beginPath(); ctx.roundRect(x, y, w, h, h / 2); ctx.fill(); } else ctx.fillRect(x, y, w, h);
  ctx.fillStyle = alphaOf(hue, 0.5);
  if (ctx.roundRect) { ctx.beginPath(); ctx.roundRect(x, y, Math.max(h, w * frac), h, h / 2); ctx.fill(); } else ctx.fillRect(x, y, w * frac, h);
  ctx.fillStyle = hue; ctx.strokeStyle = T.n0; ctx.lineWidth = 2;
  ctx.beginPath(); ctx.arc(x + w * frac, y + h / 2, active ? 9 : 7, 0, Math.PI * 2); ctx.fill(); ctx.stroke();
  ctx.fillStyle = T.n11; ctx.font = '10px ui-monospace, monospace'; ctx.textAlign = 'left'; ctx.textBaseline = 'alphabetic';
  ctx.fillText(label, x, y - 6);
  ctx.restore();
  return rect;
}

function fit(ctx, s, maxW) {
  s = String(s);
  if (ctx.measureText(s).width <= maxW) return s;
  while (s.length > 1 && ctx.measureText(s + '…').width > maxW) s = s.slice(0, -1);
  return s + '…';
}

/** Every start position in `toks` where `pat` occurs. This is literally what a
 *  node's count counts, so the ticks under the corpus ARE the count. */
function occurrences(toks, pat) {
  const out = [];
  if (!pat.length) return out;
  for (let i = 0; i + pat.length <= toks.length; i++) {
    let ok = true;
    for (let j = 0; j < pat.length; j++) if (toks[i + j] !== pat[j]) { ok = false; break; }
    if (ok) out.push(i);
  }
  return out;
}

mount({
  mount: 'body',
  title: 'suffix-decoding — speculate from text you have already seen, with no draft model',
  blurb: 'Speculative decoding needs a second, smaller model to write the proposal. Suffix decoding needs no model at all: index everything already seen — the prompt, the conversation, this generation’s own output so far — in a depth-bounded SUFFIX tree, where every position of the text inserts one suffix and each node counts how many positions produced it. To speculate, match the most recent tokens against the tree, take the longest suffix it knows, and walk the most frequent continuation. The big model verifies the proposal in one forward pass exactly as it would a draft’s (that mechanism belongs to the speculative-decoding page — this one does not re-teach it). The consequence is the interesting part: this gets BETTER with repetition, precisely where a draft model gains nothing, and collapses to ordinary decoding on genuinely novel prose. Drag the repetitiveness handle across both regimes, or type a line into “extra corpus text” to create a repeat by hand and watch the accepted length jump.',
  prefer: 'canvas2d',
  aspect: '3 / 2',
  animate: true,
  autoplay: true,
  compare: { key: 'rho', a: 0.05, b: 0.9, labelA: 'novel text — repetitiveness 0.05', labelB: 'agent loop — repetitiveness 0.90' },
  challenges: [
    {
      goal: 'Get the mean accepted length above 3.0 tokens.',
      hint: 'drag the repetitiveness handle right — the more the text repeats itself, the further the tree can copy before the model disagrees.',
      check: (api) => ({ solved: (api.probe.mean ?? 0) > 3, detail: `mean accepted ${(api.probe.mean ?? 0).toFixed(2)} tokens (need > 3.00)` }),
    },
    {
      goal: 'Make the draft model win — drive the suffix proposal below the modelled draft baseline.',
      hint: 'drag repetitiveness toward 0. On text with nothing to copy there is no proposal worth verifying, while a draft model’s agreement barely moves.',
      check: (api) => ({ solved: (api.probe.mean ?? 9) < (api.probe.draft ?? 0), detail: `suffix ${(api.probe.mean ?? 0).toFixed(2)} vs draft ${(api.probe.draft ?? 0).toFixed(2)} mean accepted` }),
    },
  ],
  controls: (c, page) => {
    c.slider('rho', { label: 'repetitiveness of the text', min: 0, max: 1, step: 0.01, value: 0.75 });
    c.stepper('plen', { label: 'proposal length', min: 1, max: 10, value: 6, rebuild: false });
    c.stepper('depth', { label: 'suffix tree depth', min: 2, max: 10, value: 6, rebuild: false });
    c.stepper('corpus', { label: 'lines already seen', min: 4, max: 24, value: 12, rebuild: false });
    c.stepper('rounds', { label: 'rounds', min: 3, max: 20, value: 12, rebuild: true });
    c.slider('draftA', { label: 'draft model per-token agreement (baseline)', min: 0.3, max: 0.95, step: 0.01, value: 0.72 });
    c.slider('seed', { label: 'seed', min: 0, max: 99, step: 1, value: 5 });
    c.text('extra', { label: 'extra corpus text', placeholder: 'type a line to create a repeat', maxlength: 90 });
    c.transport({
      compute: () => Array.from({ length: (page.state.rounds | 0) * 4 }, (_, i) => ({
        round: (i / 4) | 0, phase: i % 4, label: `round ${((i / 4) | 0) + 1} · ${PHASES[i % 4]}`,
      })),
      speed: 1.5, loop: true,
    });
  },
  onPointer: (page, ev) => {
    if (ev.type === 'up' || ev.type === 'leave') {
      if (grab) page.controls.set(grab, page.state[grab]);
      grab = null; return;
    }
    if (!geom) return;
    const near = (rect) => rect && ev.y >= rect.y - 14 && ev.y <= rect.y + rect.h + 14 && ev.x >= rect.x - 14 && ev.x <= rect.x + rect.w + 14;
    if (ev.type === 'down') grab = near(geom.rhoTrack) ? 'rho' : near(geom.lenTrack) ? 'plen' : null;
    if (!grab || !page.pointer.down) return;
    const rect = grab === 'rho' ? geom.rhoTrack : geom.lenTrack;
    const f = Math.max(0, Math.min(1, (ev.x - rect.x) / rect.w));
    if (grab === 'rho') page.controls.set('rho', Math.round(f * 100) / 100, { silent: true });
    else page.controls.set('plen', Math.max(1, Math.round(f * 10)), { silent: true });
  },
  draw: (page) => {
    const ctx = page.ctx, r = page.renderer, st = page.state, W = page.W, H = page.H;
    r.clear(T.n0);
    const data = ensureRun(st);
    const pts = ensureSweep(st);
    const L = st.plen | 0, D = st.depth | 0;
    const R = Math.max(1, data.R);

    const tp = page.controls._transport;
    const idx = tp ? tp.index : -1;
    const roundIdx = idx < 0 ? 0 : Math.min(R - 1, (idx / 4) | 0);
    const phase = idx < 0 ? -1 : idx % 4;
    const done = idx < 0 ? 0 : Math.min(R, ((idx / 4) | 0) + (idx % 4 === 3 ? 1 : 0));
    const rd = data.rounds[roundIdx] || null;

    const dm = draftMean(+st.draftA, +st.rho, L);
    const curveAt = pts[Math.max(0, Math.min(20, Math.round((+st.rho) * 20)))].suffix;
    page.probe = { mean: curveAt, run: data.mean, draft: dm, tpf: curveAt + 1, rho: +st.rho };

    const pad = 16;
    const hits = { kids: [], props: [], tokens: [] };

    // ---------- 1. the corpus: everything already seen ----------------------
    const seenNow = rd ? rd.seenAt.slice(0, rd.corpusLen) : data.base;
    r.label(`corpus — everything already seen: ${seenNow.length} tokens (prompt + conversation + this generation’s own output)`,
      pad, 15, { color: T.n11, font: '11px ui-monospace, monospace' });
    ctx.save();
    ctx.font = '10px ui-monospace, monospace';
    const tailN = 34;
    const tail = seenNow.slice(Math.max(0, seenNow.length - tailN));
    const tailOff = Math.max(0, seenNow.length - tailN);
    let cx = pad;
    if (tailOff) { r.label('…', cx, 34, { color: T.n9, font: '10px ui-monospace, monospace' }); cx += 12; }
    const matchPat = rd ? rd.match.pat : [];
    const occ = occurrences(seenNow, matchPat);
    const occSet = new Set();
    for (const o of occ) for (let j = 0; j < matchPat.length; j++) occSet.add(o + j);
    for (let i = 0; i < tail.length; i++) {
      const gi = tailOff + i;
      const w = ctx.measureText(tail[i]).width + 9;
      if (cx + w > W - pad) break;
      const isExtra = gi < data.extraLen;
      const inCtx = rd && gi >= seenNow.length - matchPat.length && matchPat.length > 0;
      let fill = isExtra ? alphaOf(T.gold, 0.35) : T.n2;
      if (occSet.has(gi)) fill = alphaOf(T.violet, inCtx ? 0.55 : 0.28);
      hits.tokens.push(chip(ctx, cx, 22, w, 17, tail[i], fill, T.n13, { font: '10px ui-monospace, monospace' }));
      cx += w + 2;
    }
    ctx.restore();

    // occurrence map: one tick per position where the matched context occurs.
    // The tick count IS the node count -- that is what a suffix index stores.
    const mapY = 44, mapW = W - 2 * pad;
    ctx.save();
    ctx.fillStyle = T.n2; ctx.fillRect(pad, mapY, mapW, 7);
    ctx.fillStyle = T.violet;
    for (const o of occ) ctx.fillRect(pad + (o / Math.max(1, seenNow.length)) * mapW, mapY, 2, 7);
    ctx.restore();
    r.label(`every position inserts one suffix (depth ${D}) · ▮ = a position where the matched context occurs — ${occ.length} of them, which is the node’s count`,
      pad, mapY + 19, { color: T.violet, font: '9.5px ui-monospace, monospace' });

    // ---------- 2. match + tree ---------------------------------------------
    const treeTop = 76;
    if (rd) {
      const swept = phase === 0 ? (page.t * 0.8) % 1 : 1;
      const backed = rd.match.longest != null && rd.match.longest > rd.match.len;
      r.label(`MATCH — longest RECURRING suffix: ${rd.match.len} token${rd.match.len === 1 ? '' : 's'}`,
        pad, treeTop, { color: T.n14, font: '11.5px ui-monospace, monospace' });
      ctx.save(); ctx.font = '10.5px ui-monospace, monospace';
      let mx = pad;
      const shownPat = rd.match.pat.length ? rd.match.pat : ['(no context matched — root)'];
      for (let i = 0; i < shownPat.length; i++) {
        const w = ctx.measureText(shownPat[i]).width + 12;
        const lit = phase !== 0 || i / shownPat.length < swept;
        chip(ctx, mx, treeTop + 8, w, 20, shownPat[i], lit ? alphaOf(T.violet, 0.45) : T.n2, T.n14, { stroke: lit ? T.violet : null });
        mx += w + 3;
      }
      ctx.restore();
      // the arrow into the node
      const nodeX = Math.min(W - 220, mx + 16), nodeY = treeTop + 18;
      ctx.save();
      ctx.strokeStyle = T.n8; ctx.lineWidth = 1.4;
      ctx.beginPath(); ctx.moveTo(mx + 3, nodeY); ctx.lineTo(nodeX - 6, nodeY); ctx.stroke();
      ctx.fillStyle = T.n8; ctx.beginPath(); ctx.moveTo(nodeX - 6, nodeY - 4); ctx.lineTo(nodeX, nodeY); ctx.lineTo(nodeX - 6, nodeY + 4); ctx.fill();
      ctx.restore();
      const nodeCount = rd.match.node.count;
      chip(ctx, nodeX, nodeY - 11, 92, 22, `node ×${nodeCount}`, alphaOf(T.accent, 0.3), T.n14, { stroke: T.accent, font: '10.5px ui-monospace, monospace' });

      // children of the match node = the continuations that have followed it
      if (backed) {
        r.label(`${rd.match.longest} tokens would match, but only once — one sighting is not evidence, so the match backs off to a span seen ≥ ${MIN_COUNT} times`,
          pad, treeTop + 44, { color: T.n10, font: '9.5px ui-monospace, monospace' });
      }
      const kids = [...rd.match.node.kids.values()].sort((a, b) => b.count - a.count).slice(0, 4);
      const kx = nodeX + 106, kw = Math.min(210, W - kx - pad - 6);
      r.label('continuations seen after it', kx, treeTop + 4, { color: T.n11, font: '9.5px ui-monospace, monospace' });
      const maxC = kids.length ? kids[0].count : 1;
      for (let i = 0; i < kids.length; i++) {
        const y = treeTop + 10 + i * 15;
        const chosen = i === 0 && phase >= 1;
        ctx.save();
        ctx.fillStyle = alphaOf(chosen ? T.ok : T.accent, 0.18 + 0.4 * (kids[i].count / maxC));
        ctx.fillRect(kx, y, kw * (kids[i].count / maxC), 13);
        ctx.strokeStyle = chosen ? T.ok : T.n5; ctx.lineWidth = chosen ? 1.6 : 1;
        ctx.strokeRect(kx + 0.5, y + 0.5, kw - 1, 12);
        ctx.fillStyle = T.n13; ctx.font = '9.5px ui-monospace, monospace';
        ctx.textAlign = 'left'; ctx.textBaseline = 'middle';
        ctx.fillText(fit(ctx, kids[i].tok, kw - 40), kx + 5, y + 7);
        ctx.textAlign = 'right'; ctx.fillStyle = T.n10;
        ctx.fillText('×' + kids[i].count, kx + kw - 4, y + 7);
        ctx.restore();
        hits.kids.push({ x: kx, y, w: kw, h: 13, kid: kids[i], total: nodeCount, rank: i });
      }
      if (!kids.length) r.label('(nothing has ever followed this context — no proposal)', kx, treeTop + 20, { color: T.n9, font: '9.5px ui-monospace, monospace' });
    }

    // ---------- 3. the round: propose / verify / commit ----------------------
    const roundTop = 158;
    r.label(rd ? `round ${roundIdx + 1} / ${R}   ·   ${phase < 0 ? 'press ▶ to run a round' : PHASE_TEXT[phase]}`
      : 'no rounds — raise “rounds”', pad, roundTop, { color: T.n14, font: '11.5px ui-monospace, monospace' });

    if (rd) {
      const cellW = Math.max(46, Math.min(96, (W - 2 * pad - 90) / Math.max(1, Math.max(rd.prop.length, rd.acc + 1))));
      const x0 = pad + 88;
      ctx.save(); ctx.font = '10px ui-monospace, monospace'; ctx.textAlign = 'right'; ctx.textBaseline = 'middle';
      ctx.fillStyle = T.n11;
      ctx.fillText('proposal', x0 - 8, roundTop + 26);
      if (phase >= 2) ctx.fillText('big model', x0 - 8, roundTop + 63);
      ctx.restore();

      for (let j = 0; j < rd.prop.length; j++) {
        const p = rd.prop[j];
        const x = x0 + j * cellW;
        const survived = j < rd.acc;
        const rejected = j === rd.acc;
        let fill = alphaOf(T.accent, 0.32), stroke = null, strike = false;
        if (phase >= 2) {
          if (survived) { fill = alphaOf(T.ok, 0.7); stroke = T.ok; }
          else if (rejected) { fill = alphaOf(T.bad, 0.72); stroke = T.bad; }
          else { fill = T.n3; strike = true; }
        }
        const w = cellW - 6;
        const rect = chip(ctx, x, roundTop + 14, w, 24, fit(ctx, p.tok, w - 8), fill, phase >= 2 && !survived && !rejected ? T.n10 : inkOn(fill), { stroke, strike });
        hits.props.push({ ...rect, j, p, survived, rejected });
        r.label(`m${p.matchLen}·×${p.count}`, x + w / 2, roundTop + 46, { color: T.n10, font: '8.5px ui-monospace, monospace', align: 'center' });
        if (phase >= 2) {
          const tv = rd.truth[j];
          const same = tv === p.tok;
          if (tv != null) {
            const f2 = alphaOf(same ? T.ok : T.violet, 0.3);
            chip(ctx, x, roundTop + 52, w, 22, fit(ctx, tv, w - 8), f2, T.n13, { dash: !same, stroke: same ? T.ok : T.violet });
          }
        }
      }
      // the single forward that covers all of them
      if (phase >= 2 && rd.prop.length) {
        const bw = rd.prop.length * cellW - 6;
        ctx.save();
        ctx.strokeStyle = T.accent; ctx.lineWidth = phase === 2 ? 2.2 : 1;
        if (phase === 2) { ctx.fillStyle = alphaOf(T.accent, 0.1); ctx.fillRect(x0 - 3, roundTop + 10, (bw + 6) * ((page.t * 0.9) % 1), 68); }
        ctx.strokeRect(x0 - 3, roundTop + 10, bw + 6, 68);
        ctx.restore();
        r.label('one forward of the big model — verified exactly as a draft model’s proposal would be (see speculative-decoding)',
          pad, roundTop + 92, { color: T.accent, font: '9.5px ui-monospace, monospace' });
      }
      if (phase >= 3) {
        r.label(`COMMIT — ${rd.acc} accepted + 1 the model wrote itself = ${rd.commit.length} tokens this round, all of them indexed into the tree for the next round`,
          pad, roundTop + 106, { color: T.okDeep, font: '10px ui-monospace, monospace' });
      }
    }

    // ---------- 4. committed output -----------------------------------------
    const outY = roundTop + 124;
    let emitted = 0;
    for (let i = 0; i < done; i++) emitted += data.rounds[i].commit.length;
    r.label(`committed output — ${emitted} token${emitted === 1 ? '' : 's'} from ${done} forward${done === 1 ? '' : 's'} of the big model`,
      pad, outY, { color: T.n11, font: '10.5px ui-monospace, monospace' });
    ctx.save(); ctx.font = '10px ui-monospace, monospace';
    let ox = pad;
    for (let i = 0; i < done; i++) {
      const ro = data.rounds[i];
      for (let j = 0; j < ro.commit.length; j++) {
        const t = ro.commit[j], w = ctx.measureText(t).width + 9;
        if (ox + w > W - pad) { i = done; break; }
        const fill = alphaOf(j < ro.acc ? T.ok : T.violet, 0.6);
        chip(ctx, ox, outY + 6, w, 17, t, fill, inkOn(fill), { font: '10px ui-monospace, monospace' });
        ox += w + 2;
      }
    }
    ctx.restore();

    // ---------- 5. accepted length per round + the rho curve -----------------
    const ctlY = H - 80;
    const panBot = ctlY - 26, panTop = Math.min(outY + 34, panBot - 96);
    const panH = panBot - panTop;
    const leftW = Math.min(220, W * 0.3);

    r.label('accepted length per round', pad, panTop - 4, { color: T.n11, font: '10px ui-monospace, monospace' });
    const bw2 = Math.max(6, Math.min(24, leftW / R));
    // Bars live BELOW the label with room for their own value on top: a full-
    // height bar otherwise puts its digit exactly where the section label is.
    const bandTop = panTop + 14;
    ctx.save(); ctx.font = '9px ui-monospace, monospace'; ctx.textAlign = 'center'; ctx.textBaseline = 'bottom';
    for (let i = 0; i < R; i++) {
      const ro = data.rounds[i], f = ro.acc / Math.max(1, L);
      const bx = pad + i * bw2, bh = Math.max(2, (panBot - bandTop) * f);
      ctx.fillStyle = i < done ? alphaOf(T.ok, 0.35 + 0.5 * f) : i === roundIdx ? alphaOf(T.accent, 0.35) : T.n3;
      ctx.fillRect(bx, panBot - bh, bw2 - 3, bh);
      ctx.fillStyle = T.n11;
      ctx.fillText(String(ro.acc), bx + (bw2 - 3) / 2, panBot - bh - 2);
    }
    ctx.strokeStyle = T.n6; ctx.lineWidth = 1;
    ctx.beginPath(); ctx.moveTo(pad, panBot + 0.5); ctx.lineTo(pad + R * bw2, panBot + 0.5); ctx.stroke();
    ctx.restore();

    // the trade, on one axis: suffix vs a draft model as the text changes
    const gx = pad + leftW + 46, gw = Math.max(150, W - gx - pad - 8), gy = panTop, gh = panH;
    const ymax = Math.max(1, L, ...pts.map((p) => Math.max(p.suffix, p.draft))) * 1.05;
    ctx.save();
    ctx.fillStyle = rgbaToken('n9', 0.06); ctx.fillRect(gx, gy, gw, gh);
    ctx.strokeStyle = T.n6; ctx.lineWidth = 1; ctx.strokeRect(gx + 0.5, gy + 0.5, gw - 1, gh - 1);
    const X = (rho) => gx + rho * gw, Y = (v) => gy + gh - (v / ymax) * gh;
    // draft baseline -- MODELLED, so it is dashed
    ctx.setLineDash([5, 4]); ctx.strokeStyle = T.warn; ctx.lineWidth = 2;
    ctx.beginPath(); pts.forEach((p, i) => (i ? ctx.lineTo(X(p.rho), Y(p.draft)) : ctx.moveTo(X(p.rho), Y(p.draft)))); ctx.stroke();
    ctx.setLineDash([]);
    ctx.strokeStyle = T.accent; ctx.lineWidth = 2.2;
    ctx.beginPath(); pts.forEach((p, i) => (i ? ctx.lineTo(X(p.rho), Y(p.suffix)) : ctx.moveTo(X(p.rho), Y(p.suffix)))); ctx.stroke();
    ctx.fillStyle = T.accent;
    ctx.beginPath(); ctx.arc(X(+st.rho), Y(curveAt), 4.5, 0, Math.PI * 2); ctx.fill();
    ctx.strokeStyle = T.n0; ctx.lineWidth = 1.4; ctx.stroke();
    // the single run on screen, beside the averaged curve it is one sample of
    ctx.fillStyle = alphaOf(T.n10, 0.9);
    ctx.beginPath(); ctx.arc(X(+st.rho), Y(data.mean), 2.4, 0, Math.PI * 2); ctx.fill();
    ctx.strokeStyle = alphaOf(T.n9, 0.8); ctx.lineWidth = 1; ctx.setLineDash([2, 3]);
    ctx.beginPath(); ctx.moveTo(X(+st.rho), gy); ctx.lineTo(X(+st.rho), gy + gh); ctx.stroke(); ctx.setLineDash([]);
    ctx.restore();
    r.label(`mean accepted length vs how repetitive the text is (${SWEEP_SEEDS} seeds × ${R} rounds per point)`, gx, gy - 4, { color: T.n11, font: '10px ui-monospace, monospace' });
    r.label('suffix tree (measured here)', gx + 4, gy + 12, { color: T.accent, font: '9.5px ui-monospace, monospace' });
    r.label('draft model (modelled)', gx + 4, gy + 24, { color: T.warn, font: '9.5px ui-monospace, monospace' });
    r.label('· = the single run on screen', gx + 4, gy + 36, { color: T.n10, font: '9px ui-monospace, monospace' });
    r.label('novel prose', gx, gy + gh + 11, { color: T.n10, font: '9px ui-monospace, monospace' });
    r.label('repeated text', gx + gw, gy + gh + 11, { color: T.n10, font: '9px ui-monospace, monospace', align: 'right' });

    // ---------- 6. handles + verdict ----------------------------------------
    const tw = Math.min(200, W * 0.24);
    geom = {
      rhoTrack: { x: pad, y: ctlY + 12, w: tw, h: 10 },
      lenTrack: { x: pad, y: ctlY + 46, w: tw, h: 10 },
    };
    track(ctx, geom.rhoTrack, +st.rho, `repetitiveness ${(+st.rho).toFixed(2)}   (drag ↔)`, T.violet, grab === 'rho');
    track(ctx, geom.lenTrack, L / 10, `proposal length ${L}   (drag ↔)`, T.accent, grab === 'plen');

    // Right column. Every line is CLAMPED to the space it actually has: at a
    // narrow canvas these strings are longer than the panel, and an unclamped
    // label does not wrap, it runs off the edge and is simply lost.
    const vx = pad + tw + 40, vw = W - vx - pad;
    const win = curveAt >= dm;
    const vline = (text, dy, font, color) => {
      ctx.save(); ctx.font = font;
      r.label(fit(ctx, text, vw), vx, ctlY + dy, { color, font });
      ctx.restore();
    };
    vline(`${curveAt.toFixed(2)} accepted per proposal → ${(curveAt + 1).toFixed(2)} tokens per forward (plain decode = 1.00)`,
      6, '11.5px ui-monospace, monospace', T.n14);
    vline(`this run on screen: ${data.mean.toFixed(2)} over ${R} rounds · draft-model baseline here: ${dm.toFixed(2)} (modelled)`,
      22, '9.5px ui-monospace, monospace', T.n11);
    vline(`the draft would also cost ${L} draft forwards every round, plus the memory its weights sit in; the tree costs a lookup`,
      35, '9.5px ui-monospace, monospace', T.n11);
    vline(win ? 'the suffix tree is ahead here — and it paid for no second model at all'
      : 'the suffix tree is behind here — nothing to copy, so it falls back toward plain decode',
      49, '10.5px ui-monospace, monospace', win ? T.okDeep : T.bad);
    if (rd) {
      // The paste has to carry the CONTEXT as well as the continuation. Pasting
      // the continuation alone adds one sighting hanging off nothing, and it
      // loses to better-attested continuations of the same context: measured,
      // round 1 stayed at 1 accepted. Pasting context + continuation is what
      // actually constitutes a repeat, and it took round 1 from 1 to 6.
      vline(`try it: paste “${rd.match.pat.join(' ')} ${rd.truth.slice(0, 8).join(' ')}” into “extra corpus text” — a repeat of this exact context`,
        63, '9px ui-monospace, monospace', T.gold);
    }

    // ---------- hover --------------------------------------------------------
    if (page.pointer.over && !grab) {
      const px = page.pointer.x, py = page.pointer.y;
      const inside = (b) => px >= b.x && px <= b.x + b.w && py >= b.y && py <= b.y + b.h;
      let tip = null;
      for (const k of hits.kids) {
        if (!inside(k)) continue;
        const share = (100 * k.kid.count) / Math.max(1, k.total);
        tip = `continuation "${k.kid.tok}"\nseen ${k.kid.count} time${k.kid.count === 1 ? '' : 's'} after this ${rd.match.len}-token context`
          + `\nthe context itself occurs ${k.total} time${k.total === 1 ? '' : 's'} → ${share.toFixed(0)}% of the time it is followed by this token`
          + (k.rank === 0 ? '\nmost frequent → this is what gets proposed' : '\nnot the most frequent → not proposed this round');
        break;
      }
      for (const h of hits.props) {
        if (tip || !inside(h)) continue;
        const p = h.p;
        const verdict = phase < 2 ? 'not verified yet' : h.survived ? 'ACCEPTED — the model agreed' : h.rejected ? 'REJECTED — the model wrote something else here' : 'DISCARDED — everything after the first disagreement is thrown away untested';
        const comp = p.sibs.map((s) => `${s.tok}×${s.count}`).join('  ');
        tip = `proposed "${p.tok}"\nwhy: the last ${p.matchLen} token${p.matchLen === 1 ? '' : 's'} matched a context seen ${p.ctxCount} time${p.ctxCount === 1 ? '' : 's'},`
          + `\nand "${p.tok}" followed it ${p.count} of those times — the most frequent continuation`
          + `\ncandidates were: ${comp}`
          + `\n${verdict}`
          + (phase >= 2 && rd.truth[h.j] != null && rd.truth[h.j] !== p.tok ? `\nthe model’s token here: "${rd.truth[h.j]}"` : '');
        break;
      }
      if (!tip && py >= mapY - 6 && py <= mapY + 14 && px >= pad && px <= pad + mapW) {
        tip = `occurrence map over all ${seenNow.length} corpus tokens\nthe matched ${matchPat.length}-token context "${matchPat.join(' ') || '(root)'}" starts at ${occ.length} position${occ.length === 1 ? '' : 's'}\nthat count is exactly what the suffix-tree node stores — no model was consulted to get it`;
      }
      if (!tip) for (const b of hits.tokens) { if (inside(b)) { tip = 'corpus token — the prompt, the conversation, and this generation’s own output are all\nindexed the same way: one suffix inserted per position, up to depth ' + D; break; } }
      if (tip) page.setTip(tip);
    }

    let o = `repetitiveness ${(+st.rho).toFixed(2)} · proposal length ${L} · tree depth ${D} → mean accepted ${curveAt.toFixed(2)} tokens over ${SWEEP_SEEDS} seeds × ${R} rounds, `;
    o += `so ${(curveAt + 1).toFixed(2)} tokens per forward of the big model (plain decode = 1.00). This run on screen: ${data.mean.toFixed(2)}. Draft-model baseline, modelled: ${dm.toFixed(2)} accepted.    tier:${r.name}\n`;
    if (rd) {
      o += `round ${roundIdx + 1}: matched the last ${rd.match.len} token${rd.match.len === 1 ? '' : 's'} `;
      o += `(a context that occurs ${rd.match.node.count} time${rd.match.node.count === 1 ? '' : 's'} in ${rd.corpusLen} indexed tokens; `;
      o += `the longest suffix present at all was ${rd.match.longest == null ? 0 : rd.match.longest}, but a span seen only once is not evidence, so the match backs off to one seen at least ${MIN_COUNT} times), `;
      o += `proposed ${rd.prop.length}, the model agreed with ${rd.acc}.\n`;
      o += `matched context: "${rd.match.pat.join(' ')}". Paste "${rd.match.pat.join(' ')} ${rd.truth.slice(0, 8).join(' ')}" into “extra corpus text” to create a repeat by hand — that tells the index this context was followed by those tokens before.\n`;
    }
    if (data.extraLen) o += `${data.extraLen} tokens of your own text are indexed at the head of the corpus.\n`;
    o += `REAL here: the corpus, the suffix tree and its counts, the longest-suffix match, the proposal, and the accepted length (a literal longest-common-prefix). `;
    o += `MODELLED: that the reference continuation is what the big model would emit, and the dashed draft-model baseline. No draft weights and no draft forward passes are spent either way — the proposal costs a tree lookup.`;
    page.setReadout(o);
  },
}).then((page) => {
  const q = new URLSearchParams(location.search);
  const t = page.controls._transport;
  for (const key of ['plen', 'depth', 'corpus']) if (q.has(key)) page.controls.set(key, parseInt(q.get(key), 10));
  if (q.has('rounds')) page.controls.set('rounds', parseInt(q.get('rounds'), 10), { rebuild: true });
  for (const key of ['rho', 'draftA', 'seed']) if (q.has(key)) page.controls.set(key, parseFloat(q.get(key)));
  if (q.has('extra')) page.controls.set('extra', q.get('extra'));
  if (q.has('hover')) {
    const [hx, hy] = q.get('hover').split(',').map(Number);
    page.pointer.x = hx; page.pointer.y = hy; page.pointer.over = true;
  }
  if (t && (q.has('step') || q.has('hover'))) t.pause();
  if (t && q.has('step')) t.seek(parseInt(q.get('step'), 10));
  if (t && q.get('play') === '1') t.play();
  page.redraw();
});
