// constrained-decoding concept page -- how a grammar or JSON schema forces the
// output to be valid, at the logit level.
//
// THE MECHANISM. The schema is COMPILED to a pushdown automaton (a finite-state
// skeleton plus a stack, because a grammar with nesting is not regular). At each
// decode step the decoder needs a BITMASK over the ENTIRE vocabulary saying
// which tokens keep the string inside the grammar; every other logit gets -inf
// before the softmax, so an invalid token has probability exactly zero. Done
// naively that means rescanning a 128k-entry vocabulary once per token, which
// costs more than the model step it guards.
//
// The production trick (XGrammar, arXiv 2411.15100, MLSys 2025) is to split the
// vocabulary in two. A token is CONTEXT-INDEPENDENT when its verdict follows
// from the automaton NODE alone -- it never consults the stack -- so the verdict
// can be precomputed once per node and cached; in a real tokenizer that is the
// large majority of the vocabulary. A token is CONTEXT-DEPENDENT when accepting
// it pops or inspects the stack (the tokens that CLOSE something: `",`, `"}`,
// `]`), so it must be re-checked live against the current stack. This page
// implements exactly that split -- the cache, its hits, and the live re-check
// count printed in the readout are produced by the code below, not narrated.
//
// The second mechanism is JUMP-FORWARD decoding (SGLang's compressed FSM). When
// the automaton has a run of single-transition edges -- after `{` the schema
// forces `"name": ` -- the answer is already determined, so the decoder emits
// the characters itself and skips the model entirely. The subtlety worth the
// pixels: the mask is built over CHARACTERS and the model works over TOKENS, and
// at the end of a jump those disagree. A tokenizer would rather emit `":` as ONE
// token than `"` then `:`; a jump has already emitted them as raw characters, so
// the model resumes on a boundary its training rarely saw. The output ribbon
// draws both segmentations and marks every seam.
//
// HONEST ABOUT THE NUMBERS: there is no model here. The logits are synthetic
// (seeded, deterministic) over a small illustrative vocabulary of 200 tokens, so
// one URL replays one exact run. Everything else -- the automaton, the stack,
// the mask, the cache accounting, the jump runs and the seams -- is really
// computed in this file.
import { mount } from '../framework/layout.js';
import { seededRandn } from '../framework/tensor.js';
import { T, alphaOf, inkOn, rgbaToken } from '../framework/theme.js';

const VMAX = 200;            // illustrative vocabulary (a deployed one is 128k+)
const MAXSTEPS = 44;
const TYPES = ['string', 'int', 'bool', 'int[]'];
const CHARS = ('{}[]":,. -0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ').split('');

// ---------------------------------------------------------------------------
// 1. schema -> fields
// ---------------------------------------------------------------------------
function parseSchema(text) {
  const raw = String(text == null ? '' : text).trim();
  if (!raw) return { error: 'empty schema — there is nothing to constrain' };
  const parts = raw.split(',').map((s) => s.trim()).filter(Boolean);
  if (!parts.length) return { error: 'no fields found' };
  const fields = [];
  for (const p of parts) {
    const m = /^([A-Za-z_][A-Za-z0-9_]*)\s*:\s*(.+)$/.exec(p);
    if (!m) return { error: `"${p}" is not  key:type` };
    const type = m[2].trim();
    if (!TYPES.includes(type)) return { error: `unknown type "${type}" in "${p}" — use ${TYPES.join(' / ')}` };
    if (fields.some((f) => f.key === m[1])) return { error: `duplicate key "${m[1]}"` };
    fields.push({ key: m[1], type });
  }
  if (fields.length > 5) fields.length = 5;
  return { fields };
}

// ---------------------------------------------------------------------------
// 2. fields -> pushdown automaton
//
// Nodes carry explicit character edges. Three edge powers make it a PDA rather
// than an FSM: `push` (enter a sub-rule / open a bracket, remembering where to
// return), `popRet` (a sub-rule ends -- the return address comes off the stack),
// `popArr` / `popObj` (a bracket closes). `retOnEnd` is the epsilon-pop: a
// number ends wherever a non-digit appears, and WHICH non-digit is legal there
// is a question only the stack can answer.
// ---------------------------------------------------------------------------
function compile(fields) {
  const nodes = [];
  const add = (label, extra) => { nodes.push(Object.assign({ id: nodes.length, label, edges: [] }, extra || {})); return nodes.length - 1; };
  const isDigit = (c) => c >= '0' && c <= '9';
  const eq = (ch) => (c) => c === ch;

  // shared sub-rules -- entered with a return address pushed, which is what
  // makes their exit tokens context-dependent.
  const INT = add('int', { shared: true, retOnEnd: true });
  nodes[INT].edges.push({ desc: '0-9', test: isDigit, to: INT });
  const STR = add('string body', { shared: true });
  nodes[STR].edges.push({ desc: 'any char but "', test: (c) => c !== '"', to: STR });
  nodes[STR].edges.push({ desc: '"  → return', test: eq('"'), popRet: true });

  const START = add('{');
  const DONE = add('done', { done: true });

  const opens = [], ends = [];
  fields.forEach((f) => {
    // key chain: " k e y " :  -- every node here has exactly ONE outgoing edge,
    // which is precisely what jump-forward decoding exploits.
    const chain = ['"'].concat(f.key.split(''), '"', ':');
    const ids = chain.map((ch) => add(ch));
    opens.push(ids[0]);
    const END = add(`⟨${f.key}✓⟩`);
    ends.push(END);
    let entry;
    if (f.type === 'string') {
      entry = add('"');
      nodes[entry].edges.push({ desc: '"  → string rule', test: eq('"'), to: STR, push: { sym: 'ret', to: END } });
    } else if (f.type === 'int') {
      entry = add('0-9');
      nodes[entry].edges.push({ desc: '0-9  → int rule', test: isDigit, to: INT, push: { sym: 'ret', to: END } });
    } else if (f.type === 'bool') {
      entry = add('t / f');
      const lit = (word) => {
        const endNode = add(word + '✓', { retOnEnd: true });
        let cur = entry;
        word.split('').forEach((ch, k) => {
          const nxt = k === word.length - 1 ? endNode : add(word[k + 1]);
          nodes[cur].edges.push(Object.assign({ desc: `"${ch}"`, test: eq(ch), to: nxt }, k === 0 ? { push: { sym: 'ret', to: END } } : {}));
          cur = nxt;
        });
      };
      lit('true'); lit('false');
    } else {                                   // int[]  -- the genuine pushdown case
      entry = add('[');
      const AOPEN = add('0-9 / ]'), ASEP = add(', / ]');
      nodes[entry].edges.push({ desc: '[  (push)', test: eq('['), to: AOPEN, push: { sym: 'arr', to: END } });
      nodes[AOPEN].edges.push({ desc: '0-9  → int rule', test: isDigit, to: INT, push: { sym: 'ret', to: ASEP } });
      nodes[AOPEN].edges.push({ desc: ']  (pop, empty array)', test: eq(']'), popArr: true });
      nodes[ASEP].edges.push({ desc: ',', test: eq(','), to: AOPEN });
      nodes[ASEP].edges.push({ desc: ']  (pop)', test: eq(']'), popArr: true });
    }
    ids.forEach((id, k) => nodes[id].edges.push({ desc: `"${chain[k]}"`, test: eq(chain[k]), to: k + 1 < ids.length ? ids[k + 1] : entry }));
  });

  nodes[START].edges.push({ desc: '{  (push)', test: eq('{'), to: opens[0], push: { sym: 'obj' } });
  ends.forEach((END, i) => {
    if (i + 1 < ends.length) nodes[END].edges.push({ desc: ',', test: eq(','), to: opens[i + 1] });
    else nodes[END].edges.push({ desc: '}  (pop)', test: eq('}'), to: DONE, popObj: true });
  });
  return { nodes, START, DONE };
}

// Set while a single advance() reads the stack to reach its verdict. A REJECTION
// only counts as context-dependent when the stack was actually consulted to
// produce it (an edge matched the character but the top symbol refused it, or
// the sub-rule's epsilon-pop had to ask where it returns to) -- not merely
// because the node happens to have a push edge somewhere.
let _consulted = false;

// One character. Returns {node, stack, kind} or null; `kind` says what it did to
// the stack: 'push' / 'pop' / 'ret' / 'plain'.
function advance(G, nodeId, stack, ch, depth) {
  if (nodeId == null || nodeId < 0) return null;
  const node = G.nodes[nodeId];
  for (const e of node.edges) {
    if (!e.test(ch)) continue;
    if (e.popRet || e.popArr || e.popObj) {
      const top = stack[stack.length - 1];
      _consulted = true;                       // this verdict is the stack's to give
      if (!top) return null;
      if (e.popRet && top.sym !== 'ret') return null;
      if (e.popArr && top.sym !== 'arr') return null;
      if (e.popObj && top.sym !== 'obj') return null;
      return { node: e.to != null ? e.to : top.to, stack: stack.slice(0, -1), kind: 'pop' };
    }
    if (e.push) return { node: e.to, stack: stack.concat([e.push]), kind: 'push' };
    return { node: e.to, stack, kind: 'plain' };
  }
  // epsilon-pop: the sub-rule is over, and the return address decides what is
  // legal here. This is the transition a finite-state mask cannot answer alone.
  if (node.retOnEnd && (depth | 0) < 3) {
    const top = stack[stack.length - 1];
    _consulted = true;                         // where a sub-rule ENDS is a stack question
    if (top && top.sym === 'ret') {
      const r = advance(G, top.to, stack.slice(0, -1), ch, (depth | 0) + 1);
      if (r) return { node: r.node, stack: r.stack, kind: 'ret' };
    }
  }
  return null;
}

// Walk a whole token's characters. `used` = this verdict consulted the AMBIENT
// stack (a pop below the token's own pushes, or a stack-sensitive rejection),
// which is exactly what makes a token context-DEPENDENT, and so un-cacheable
// against the node alone.
function tokenSim(G, nodeId, stack, str) {
  let n = nodeId, s = stack, used = false, local = 0;
  for (const ch of str) {
    _consulted = false;
    const r = advance(G, n, s, ch, 0);
    if (!r) return { ok: false, used: used || (local === 0 && _consulted), at: ch, node: n };
    if (r.kind === 'push') local++;
    else if (r.kind === 'pop' || r.kind === 'ret') { if (local > 0) local--; else used = true; }
    n = r.node; s = r.stack;
  }
  return { ok: true, used, node: n, stack: s };
}

// The single-transition run the automaton is already committed to: the
// jump-forward opportunity, where no model call could change any character.
function detRun(G, nodeId, stack, cap) {
  let n = nodeId, s = stack, text = '';
  for (let i = 0; i < (cap || 32); i++) {
    let hit = null, count = 0;
    for (const ch of CHARS) { const r = advance(G, n, s, ch, 0); if (r) { count++; hit = { ch, r }; if (count > 1) break; } }
    if (count !== 1) break;
    text += hit.ch; n = hit.r.node; s = hit.r.stack;
  }
  return { text, node: n, stack: s };
}

// ---------------------------------------------------------------------------
// 3. the illustrative vocabulary (structural pieces, seam tokens, word pieces)
// ---------------------------------------------------------------------------
function buildVocab(fields) {
  const out = [];
  const push = (t) => { if (t && out.length < VMAX && out.indexOf(t) < 0) out.push(t); };
  ['{', '}', '[', ']', ':', ',', '"', ' ', '-', '.'].forEach(push);
  ['":', '",', '"}', '"]', '":"', '{"', ',"', '["', '":[', '":{', '},', '],', '}]', '": ', ', ', ': ', '"{'].forEach(push);
  for (const f of fields) { push(f.key); push('"' + f.key); push(f.key + '":'); push(f.key.slice(0, Math.ceil(f.key.length / 2))); push(f.key.slice(Math.ceil(f.key.length / 2))); }
  // the values the synthetic model wants to write (see intentFor) must be
  // reachable AS TOKENS, or its intent could never be expressed at all
  ['Ada', 'Bob', 'Cyd', 'Dee', 'Eve', 'true', 'false', '42', '25', '99', '12', '7'].forEach(push);
  for (let d = 0; d < 10; d++) push(String(d));
  ['10', '12', '17', '20', '25', '30', '42', '64', '99', '100', '128', '256', '2026'].forEach(push);
  ['true', 'false', 'tr', 'ue', 'fal', 'se', 'nul', 'null'].forEach(push);
  'abcdefghijklmnopqrstuvwxyz'.split('').forEach(push);
  ['name', 'age', 'ok', 'id', 'key', 'val', 'type', 'size', 'tag', 'tags', 'user', 'city', 'code', 'year', 'day', 'time',
    'item', 'list', 'data', 'text', 'score', 'scores', 'count', 'total', 'level', 'state', 'label', 'title', 'note',
    'flag', 'mode', 'rate', 'unit', 'path', 'line', 'word', 'char', 'byte', 'file', 'node', 'edge', 'root', 'leaf',
    'head', 'tail', 'next', 'prev', 'open', 'read', 'ing', 'ed', 'er', 'ly', 'un', 're', 'pre', 'sub', 'ob', 'ar',
    'st', 'on', 'in', 'at', 'or', 'es', 'al', 'an', 'le', 'ra', 'ti', 'ic', 'us', 'll', 'ss', 'oo', 'ee', 'th', 'ch',
    'sh', 'qu', 'Ada', 'Bob', 'Cyd', 'Dee', 'Eve', 'yes', 'no', 'up', 'down', 'left', 'red', 'blue', 'one', 'two',
    'six', 'ten', 'the', 'and', 'for', 'was', 'not', 'you', 'she', 'him'].forEach(push);
  let filler = 0;
  while (out.length < VMAX) push('w' + (filler++));
  return out;
}

// Greedy longest-match segmentation: what a tokenizer would have produced for
// the same string. Compared with what the decoder actually emitted, the
// disagreements are the retokenization seams.
function greedyTok(text, vocab) {
  const pieces = [];
  let i = 0;
  while (i < text.length) {
    let best = '';
    for (const t of vocab) if (t.length > best.length && text.startsWith(t, i)) best = t;
    if (!best) best = text[i];
    pieces.push(best); i += best.length;
  }
  return pieces;
}

// ---------------------------------------------------------------------------
// 4. the mask, and the two-class cache that makes it affordable
// ---------------------------------------------------------------------------
function maskFor(G, nodeId, stack, vocab, cache, stats) {
  const V = vocab.length;
  let entry = cache.get(nodeId);
  if (!entry) {                                    // cache MISS: scan the vocabulary once
    entry = { legal: new Uint8Array(V), cd: new Uint8Array(V) };
    for (let i = 0; i < V; i++) { const s = tokenSim(G, nodeId, stack, vocab[i]); entry.cd[i] = s.used ? 1 : 0; entry.legal[i] = s.ok ? 1 : 0; }
    cache.set(nodeId, entry); stats.fills++; stats.scanned += V;
  } else stats.hits++;                             // cache HIT: the context-independent bits stand
  const legal = Uint8Array.from(entry.legal);
  let nCd = 0;
  for (let i = 0; i < V; i++) if (entry.cd[i]) { legal[i] = tokenSim(G, nodeId, stack, vocab[i]).ok ? 1 : 0; nCd++; }
  stats.cdChecks += nCd; stats.ciSkipped += V - nCd;
  let n = 0; for (let i = 0; i < V; i++) n += legal[i];
  return { legal, cd: entry.cd, nCd, nLegal: n };
}

// The synthetic model's INTENT: the document it would write if nothing stopped
// it. Real logits are not noise -- a model has something it wants to say -- and
// the whole question of constrained decoding is what happens when that intent
// and the grammar disagree. So the "model" here is: intent pull (weight = the
// fit slider) + a pull toward ordinary prose (weight = 1 - fit) + seeded noise.
function intentFor(fields) {
  const names = ['Ada', 'Bob', 'Cyd', 'Dee', 'Eve'], ints = ['42', '7', '25', '99', '12'];
  const parts = fields.map((f, i) => `"${f.key}":` + (
    f.type === 'string' ? `"${names[i % 5]}"` : f.type === 'int' ? ints[i % 5] : f.type === 'bool' ? (i % 2 ? 'false' : 'true') : '[7,12]'));
  return '{' + parts.join(',') + '}';
}
const isProse = (t) => /^[a-z]{2,}$/.test(t) || t === ' ' || t === '.';

const softmaxOf = (z) => {
  let m = -Infinity; for (const v of z) if (v > m) m = v;
  const e = z.map((v) => Math.exp(v - m));
  const s = e.reduce((a, b) => a + b, 0);
  return e.map((v) => v / s);
};

// ---------------------------------------------------------------------------
// 5. the decode run
// ---------------------------------------------------------------------------
let RUN = null;

function buildRun(st) {
  const parsed = parseSchema(st.schema);
  const vocab = buildVocab(parsed.fields || []);
  const V = vocab.length;
  const stats = { fills: 0, hits: 0, scanned: 0, cdChecks: 0, ciSkipped: 0 };
  const seed = st.seed | 0, fit = st.fit, useMask = st.mask !== false, jump = !!st.jump;

  if (parsed.error) {
    const dead = { i: 0, kind: 'dead', legal: new Uint8Array(V), cd: new Uint8Array(V), nLegal: 0, nCd: 0, text: '', pieces: [], stack: [], node: -1, keptMass: 0, label: 'dead state — the schema does not compile' };
    RUN = { steps: [dead], vocab, G: null, error: parsed.error, stats, fields: [], jumpChars: 0, modelCalls: 0, text: '', nodeCount: 0 };
    return [{ i: 0, label: dead.label }];
  }

  const G = compile(parsed.fields);
  const cache = new Map();
  const steps = [];
  const intent = intentFor(parsed.fields);
  const prose = vocab.map((t) => (isProse(t) ? 1 : 0));
  let node = G.START, stack = [], text = '', pieces = [], broken = false, jumpChars = 0, modelCalls = 0, driftLeft = 4;

  // Three pulls, all deterministic in the seed: toward the document the model
  // meant to write, toward ordinary prose (which JSON mostly forbids), and --
  // once it has already left its own target -- toward closing what it opened,
  // so a run terminates instead of rambling.
  const closing = vocab.map((t) => (/["}\]]/.test(t) ? 1 : 0));
  const logitsAt = (s) => {
    const base = Array.from(seededRandn(seed * 7919 + s * 131 + 17, V, { std: 1.15 }));
    const onIntent = intent.startsWith(text);
    const rest = onIntent ? intent.slice(text.length) : '';
    return base.map((z, i) => {
      const tok = vocab[i];
      const pull = onIntent
        ? (rest.startsWith(tok) ? 4.5 + 1.2 * Math.min(3, tok.length) : 0)
        : 3.2 * closing[i];
      return z + fit * pull + (1 - fit) * 3.0 * prose[i];
    });
  };

  for (let s = 0; s < MAXSTEPS; s++) {
    if (node === G.DONE) break;
    if (broken) {                                   // the string already left the grammar:
      if (driftLeft-- <= 0) break;                  // show it drifting, no mask involved
      const logits = logitsAt(s);
      const order = logits.map((z, i) => i).sort((a, b) => logits[b] - logits[a]);
      const tok = vocab[order[0]];
      text += tok; pieces.push({ text: tok, kind: 'bad' });
      steps.push({ i: steps.length, kind: 'drift', node: -1, stack: [], text, pieces: pieces.slice(), skipMask: true, label: `unconstrained "${tok}" — no mask, and no grammar left to keep` });
      continue;
    }
    if (jump) {
      const run = detRun(G, node, stack, 32);
      if (run.text.length >= 2) {                 // the model is not consulted at all
        text += run.text; pieces.push({ text: run.text, kind: 'jump' }); jumpChars += run.text.length;
        steps.push({ i: steps.length, kind: 'jump', node, stack: stack.slice(), jumped: run.text, text, pieces: pieces.slice(), skipMask: true, label: `jump-forward: emit "${run.text}" with NO model call` });
        node = run.node; stack = run.stack;
        continue;
      }
    }
    const m = maskFor(G, node, stack, vocab, cache, stats);
    const logits = logitsAt(s);
    const pBefore = softmaxOf(logits);
    const order = logits.map((z, i) => i).sort((a, b) => logits[b] - logits[a]);
    let keptMass = 0; for (let i = 0; i < V; i++) if (m.legal[i]) keptMass += pBefore[i];

    let chosen = -1;
    if (useMask) { for (const i of order) if (m.legal[i]) { chosen = i; break; } }
    else chosen = order[0];
    if (chosen < 0) {
      steps.push({ i: steps.length, kind: 'stuck', node, stack: stack.slice(), legal: m.legal, cd: m.cd, nCd: m.nCd, nLegal: 0, logits, pBefore, order, keptMass, chosen: -1, text, pieces: pieces.slice(), label: 'dead state — no token keeps the string inside the grammar' });
      break;
    }
    const rank = order.indexOf(chosen);
    const tok = vocab[chosen];
    const violated = !m.legal[chosen];
    modelCalls++;
    text += tok; pieces.push({ text: tok, kind: violated ? 'bad' : 'tok' });
    steps.push({
      i: steps.length, kind: 'token', node, stack: stack.slice(), legal: m.legal, cd: m.cd, nCd: m.nCd, nLegal: m.nLegal,
      logits, pBefore, order, keptMass, chosen, rank, tok, violated, text, pieces: pieces.slice(),
      label: `${useMask ? 'masked' : 'UNMASKED'} pick "${tok}"${violated ? ' — left the grammar' : ''}  (${m.nLegal}/${V} legal)`,
    });
    if (violated) { broken = true; continue; }
    const r = tokenSim(G, node, stack, tok); node = r.node; stack = r.stack;
  }

  const doneOK = node === G.DONE && !broken;
  steps.push({ i: steps.length, kind: 'done', node: broken ? -1 : node, stack: broken ? [] : stack.slice(), text, pieces: pieces.slice(), skipMask: true, ok: doneOK, broken, label: doneOK ? `complete — "${text}" satisfies the schema` : broken ? 'stopped — the string is no longer valid under the schema' : 'stopped — the string never closed' });

  RUN = { steps, vocab, G, error: null, stats, fields: parsed.fields, jumpChars, modelCalls, text, nodeCount: G.nodes.length };
  return steps.map((s) => ({ i: s.i, label: s.label }));
}

// ---------------------------------------------------------------------------
// 6. draw
// ---------------------------------------------------------------------------
let geom = null;        // hit-test rects captured in draw()
let grab = null;

function chip(ctx, x, y, w, h, fill, stroke, lw) {
  ctx.fillStyle = fill; ctx.strokeStyle = stroke; ctx.lineWidth = lw || 1;
  if (ctx.roundRect) { ctx.beginPath(); ctx.roundRect(x, y, w, h, 4); ctx.fill(); ctx.stroke(); }
  else { ctx.fillRect(x, y, w, h); ctx.strokeRect(x, y, w, h); }
}

mount({
  mount: 'body',
  title: 'constrained decoding — a grammar as a bitmask over the vocabulary',
  blurb: 'A schema does not persuade a model to emit valid JSON; it deletes every other option. The schema is compiled to a pushdown automaton, and at each decode step the decoder builds a BITMASK over the entire vocabulary — one bit per token, "does this keep me inside the grammar" — and adds −∞ to the rest before the softmax. Done naively that is a rescan of 128k+ tokens per token, costing more than the model step it guards. The production trick (XGrammar) splits the vocabulary: tokens whose verdict depends only on the automaton node are CONTEXT-INDEPENDENT and precomputed into a per-node mask cache; the small remainder — the tokens that close a string, an array, the object — are CONTEXT-DEPENDENT and re-checked live against the stack. The vocabulary strip colours both classes, and the cache counters in the readout are really computed here. Turn on jump-forward for the other half of the trick: where the automaton has only one way out, the decoder emits the characters itself and never calls the model — at the price of a retokenization seam, marked in the top ribbon. Edit the schema and watch the legal set collapse under your hand; a schema that does not compile reaches a dead state with the mask all red. The logits are synthetic (seeded, no model runs on this page); the automaton, mask, cache and seams are computed on the page.',
  prefer: 'canvas2d',
  aspect: '8 / 5',
  animate: true,
  compare: { key: 'mask', a: true, b: false, rebuild: true, labelA: 'mask ON — every token stays inside the grammar', labelB: 'mask OFF — the raw argmax leaves the grammar' },
  challenges: [
    { goal: 'Break the schema so hard the mask goes all-red (a dead state).', hint: 'give a field a type the compiler does not know — try  name:colour  — or empty the schema box.', check: (api) => ({ solved: (api.probe.nLegal ?? 1) === 0, detail: `${api.probe.nLegal ?? 0} legal tokens of ${api.probe.V ?? VMAX}` }) },
    { goal: 'Find a step where the mask keeps under 1% of the model’s probability mass.', hint: 'drag the “model agrees” handle (or the slider) toward 0: the grammar then fights the model, and the token it forces is one the model had all but ruled out.', check: (api) => ({ solved: (api.probe.keptMass ?? 1) < 0.01, detail: `kept mass ${(100 * (api.probe.keptMass ?? 1)).toFixed(2)}% (need < 1%)` }) },
  ],
  controls: (c, page) => {
    c.text('schema', { label: 'schema — key:type, …   (string / int / bool / int[])', value: 'name:string, scores:int[], ok:bool', placeholder: 'name:string, age:int', rebuild: true });
    c.toggle('jump', { label: 'jump-forward decoding', value: false, rebuild: true });
    c.toggle('mask', { label: 'apply the mask', value: true, rebuild: true });
    c.slider('fit', { label: 'model agrees with schema', min: 0, max: 1, step: 0.05, value: 0.7, rebuild: true });
    c.slider('seed', { label: 'seed', min: 0, max: 60, step: 1, value: 3, rebuild: true });
    c.transport({ compute: () => buildRun(page.state), loop: true, speed: 1.1 });
  },
  autoplay: true,
  onPointer: (page, ev) => {
    if (ev.type === 'up' || ev.type === 'leave') { grab = null; return; }
    if (!geom || !geom.fitHandle) return;
    const h = geom.fitHandle;
    if (ev.type === 'down' && Math.abs(ev.y - h.y) < 14 && ev.x > h.x0 - 14 && ev.x < h.x1 + 14) grab = 'fit';
    if (grab === 'fit' && page.pointer.down) {
      const v = Math.max(0, Math.min(1, (ev.x - h.x0) / (h.x1 - h.x0)));
      page.controls.set('fit', Math.round(v * 20) / 20, { rebuild: true });
    }
  },
  draw: (page) => {
    const ctx = page.ctx, r = page.renderer, st = page.state, W = page.W;
    r.clear(T.n0);
    if (!RUN) { r.label('building…', 20, 30, { color: T.n11 }); return; }
    const rec = page.step();
    const S = RUN.steps[Math.max(0, Math.min(rec ? rec.i : 0, RUN.steps.length - 1))];
    const vocab = RUN.vocab, V = vocab.length, pad = 14;
    geom = { fitHandle: null, vocab: null };
    page.probe = { nLegal: S.nLegal != null ? S.nLegal : (RUN.error ? 0 : V), keptMass: S.keptMass != null ? S.keptMass : 1, V };

    // ===== A. output ribbon: what was emitted vs how a tokenizer would split it
    let y = 22;
    r.label('output so far — one chip per emitted token; an orange chip is a jump-forward run, which is not a token at all', pad, y - 8, { color: T.n14, font: '11px ui-monospace, monospace' });
    ctx.textBaseline = 'middle'; ctx.textAlign = 'left';
    const emitBounds = new Set(); { let a = 0; for (const p of (S.pieces || [])) { a += p.text.length; emitBounds.add(a); } }
    const pref = greedyTok(S.text || '', vocab);
    let seams = 0; { let a = 0; for (const p of pref) { a += p.length; if (!emitBounds.has(a) && a < (S.text || '').length) seams++; } }

    ctx.font = '12px ui-monospace, monospace';
    let x = pad, clipped = false;
    for (const p of (S.pieces || [])) {
      const w = ctx.measureText(p.text).width + 12;
      if (x + w > W - pad - 14) { clipped = true; break; }
      const fill = p.kind === 'jump' ? alphaOf(T.warn, 0.22) : p.kind === 'bad' ? alphaOf(T.bad, 0.25) : alphaOf(T.accent, 0.16);
      const line = p.kind === 'jump' ? T.warn : p.kind === 'bad' ? T.bad : T.accentLine;
      chip(ctx, x, y, w, 20, fill, line, p.kind === 'jump' ? 1.4 : 1);
      ctx.fillStyle = p.kind === 'jump' ? T.warnDeep : T.n14; ctx.fillText(p.text === ' ' ? '␣' : p.text, x + 6, y + 10);
      x += w + 3;
    }
    if (clipped) r.label('…', x + 2, y + 14, { color: T.n10, font: '12px ui-monospace, monospace' });
    if (!(S.pieces || []).length) r.label('(nothing emitted yet)', pad, y + 14, { color: T.n9, font: '11px ui-monospace, monospace' });
    y += 30;
    r.label('the tokenizer’s own segmentation of the same string — ▲ marks a boundary the model never saw', pad, y, { color: T.n11, font: '10px ui-monospace, monospace' });
    y += 12;
    x = pad; ctx.font = '11px ui-monospace, monospace';
    let acc = 0;
    for (const p of pref) {
      const w = ctx.measureText(p).width + 10;
      if (x + w > W - pad - 14) break;
      acc += p.length;
      const seam = !emitBounds.has(acc) && acc < (S.text || '').length;
      chip(ctx, x, y, w, 18, alphaOf(T.n9, 0.14), seam ? T.bad : T.n6, seam ? 1.6 : 1);
      ctx.fillStyle = T.n12; ctx.fillText(p === ' ' ? '␣' : p, x + 5, y + 9);
      if (seam) { ctx.fillStyle = T.bad; ctx.fillText('▲', x + w - 4, y + 24); }
      x += w + 3;
    }
    y += 32;

    // ===== B. the automaton + its stack
    ctx.strokeStyle = T.n4; ctx.lineWidth = 1; ctx.beginPath(); ctx.moveTo(pad, y); ctx.lineTo(W - pad, y); ctx.stroke();
    y += 16;
    if (RUN.error) {
      chip(ctx, pad, y, W - pad * 2, 28, alphaOf(T.bad, 0.14), T.bad, 1.4);
      ctx.fillStyle = T.bad; ctx.font = '12px ui-monospace, monospace'; ctx.textBaseline = 'middle';
      ctx.fillText(`schema does not compile — ${RUN.error}`, pad + 10, y + 14);
      y += 40;
    } else {
      r.label(`pushdown automaton — ${RUN.nodeCount} nodes; the lit node is where decoding stands`, pad, y - 3, { color: T.n14, font: '11px ui-monospace, monospace' });
      y += 8;
      const stackW = 126, avail = W - pad * 2 - stackW - 14;
      const main = RUN.G.nodes.filter((n) => !n.shared), shared = RUN.G.nodes.filter((n) => n.shared);
      ctx.font = '10px ui-monospace, monospace'; ctx.textBaseline = 'middle';
      const pulse = 0.5 + 0.5 * Math.sin(page.t * 3.2);
      const drawNodes = (list, y0) => {
        let cx = pad, cy = y0;
        for (const n of list) {
          const w = Math.max(16, ctx.measureText(n.label).width + 10);
          if (cx + w > pad + avail) { cx = pad; cy += 21; }
          const isCur = n.id === S.node;
          const det = n.edges.length === 1 && !n.retOnEnd;       // a jump-forward opportunity
          const fill = isCur ? alphaOf(T.accent, 0.55 + 0.35 * pulse) : (det && st.jump) ? alphaOf(T.warn, 0.16) : n.done ? alphaOf(T.ok, 0.18) : T.n2;
          chip(ctx, cx, cy, w, 18, fill, isCur ? T.accent : (det && st.jump) ? T.warnLine : T.n5, isCur ? 1.8 : 1);
          ctx.fillStyle = isCur ? inkOn(T.accent) : T.n12;
          ctx.fillText(n.label, cx + 5, cy + 9);
          cx += w + 4;
        }
        return cy + 21;
      };
      const afterMain = drawNodes(main, y);
      r.label(st.jump ? 'orange = only one way out, so jump-forward emits it  ·  shared sub-rules ↓'
        : 'shared sub-rules — entered with a return address pushed ↓', pad, afterMain + 9, { color: T.n11, font: '10px ui-monospace, monospace' });
      ctx.font = '10px ui-monospace, monospace';
      const afterShared = drawNodes(shared, afterMain + 15);
      const sx = W - pad - stackW;
      r.label('stack', sx, y + 6, { color: T.n11, font: '10px ui-monospace, monospace' });
      const stk = S.stack || [];
      if (!stk.length) r.label('(empty)', sx, y + 24, { color: T.n9, font: '10px ui-monospace, monospace' });
      stk.slice().reverse().forEach((e, i) => {
        const yy = y + 14 + i * 20;
        const c = e.sym === 'obj' ? T.violet : e.sym === 'arr' ? T.teal : T.gold;
        chip(ctx, sx, yy, stackW, 17, alphaOf(c, 0.18), c, 1);
        ctx.fillStyle = T.n13; ctx.font = '10px ui-monospace, monospace'; ctx.textBaseline = 'middle';
        ctx.fillText(e.sym === 'ret' ? `ret → ${RUN.G.nodes[e.to].label}` : e.sym === 'obj' ? '{  object' : '[  array', sx + 5, yy + 9);
      });
      y = Math.max(afterShared, y + 14 + stk.length * 20) + 8;
    }

    // ===== C. the vocabulary mask
    ctx.strokeStyle = T.n4; ctx.lineWidth = 1; ctx.beginPath(); ctx.moveTo(pad, y); ctx.lineTo(W - pad, y); ctx.stroke();
    y += 16;
    const cols = 25, rows = Math.ceil(V / cols), cw = (W - pad * 2) / cols, chh = 11;
    const skip = !!S.skipMask;
    const nLegal = S.nLegal != null ? S.nLegal : 0;
    r.label(skip
      ? (S.kind === 'jump' ? `vocabulary mask — NOT BUILT this step: jump-forward skipped the model, so there were no logits to mask` : 'vocabulary mask — sequence over')
      : `vocabulary mask — ${nLegal} of ${V} tokens keep the string inside the grammar (${S.nCd || 0} of them context-dependent)`,
    pad, y - 2, { color: skip ? T.warnDeep : T.n14, font: '11px ui-monospace, monospace' });
    y += 6;
    const legal = S.legal || new Uint8Array(V), cdArr = S.cd || new Uint8Array(V);
    geom.vocab = { x: pad, y, cw, ch: chh, cols, rows, skip };
    ctx.fillStyle = rgbaToken('n14', 0.03); ctx.fillRect(pad, y, W - pad * 2, rows * chh);
    for (let i = 0; i < V; i++) {
      const cx = pad + (i % cols) * cw, cy = y + Math.floor(i / cols) * chh;
      if (skip) { ctx.fillStyle = alphaOf(T.n9, 0.16); ctx.fillRect(cx + 0.5, cy + 0.5, cw - 1.5, chh - 1.5); continue; }
      const ok = !!legal[i], cd = !!cdArr[i];
      ctx.fillStyle = ok ? alphaOf(T.ok, cd ? 0.34 : 0.85) : alphaOf(T.bad, cd ? 0.22 : 0.6);
      ctx.fillRect(cx + 0.5, cy + 0.5, cw - 1.5, chh - 1.5);
      if (cd) { ctx.strokeStyle = T.teal; ctx.lineWidth = 1; ctx.strokeRect(cx + 1, cy + 1, cw - 2.5, chh - 2.5); }
      if (S.chosen === i) { ctx.strokeStyle = T.n14; ctx.lineWidth = 2; ctx.strokeRect(cx, cy, cw, chh); }
    }
    y += rows * chh + 13;
    { // legend
      let lx = pad;
      ctx.font = '9.5px ui-monospace, monospace'; ctx.textBaseline = 'middle';
      const sw = (fill, cd, txt) => {
        ctx.fillStyle = fill; ctx.fillRect(lx, y - 5, 10, 10);
        if (cd) { ctx.strokeStyle = T.teal; ctx.lineWidth = 1; ctx.strokeRect(lx + 0.5, y - 4.5, 9, 9); }
        ctx.fillStyle = T.n11; ctx.fillText(txt, lx + 14, y); lx += ctx.measureText(txt).width + 26;
      };
      sw(alphaOf(T.ok, 0.85), false, 'legal · cached per node');
      sw(alphaOf(T.ok, 0.34), true, 'legal · re-checked on the stack');
      sw(alphaOf(T.bad, 0.6), false, 'masked → −∞');
      sw(alphaOf(T.bad, 0.22), true, 'masked · stack-dependent');
    }
    y += 14;

    // ===== D. logits before and after the mask
    ctx.strokeStyle = T.n4; ctx.lineWidth = 1; ctx.beginPath(); ctx.moveTo(pad, y); ctx.lineTo(W - pad, y); ctx.stroke();
    y += 16;
    if ((S.kind === 'token' || S.kind === 'stuck') && S.order) {
      const K = 20, ord = S.order.slice(0, K);
      if (S.chosen >= 0 && ord.indexOf(S.chosen) < 0) ord[K - 1] = S.chosen;
      const bw = (W - pad * 2) / ord.length, barsH = 74, yb = y + barsH + 8;
      r.label(`logits → probabilities (top ${K}) · hollow = before the mask, solid = after (−∞, renormalized) · height ∝ √p`, pad, y + 4, { color: T.n14, font: '11px ui-monospace, monospace' });
      let ksum = 0; for (let i = 0; i < V; i++) if (legal[i]) ksum += S.pBefore[i];
      const pAfter = (i) => (legal[i] && ksum > 1e-12 ? S.pBefore[i] / ksum : 0);
      let scale = 1e-9;
      for (const i of ord) scale = Math.max(scale, S.pBefore[i], pAfter(i));
      ctx.textAlign = 'center';
      for (let k = 0; k < ord.length; k++) {
        const idx = ord[k], bx = pad + k * bw, ok = !!legal[idx];
        const hb = Math.sqrt(S.pBefore[idx] / scale) * barsH;
        ctx.strokeStyle = T.n8; ctx.lineWidth = 1; ctx.strokeRect(bx + 2, yb - hb, bw - 5, Math.max(1, hb));
        if (ok) {
          const ha = Math.sqrt(pAfter(idx) / scale) * barsH;
          ctx.fillStyle = idx === S.chosen ? T.ok : alphaOf(T.ok, 0.5);
          ctx.fillRect(bx + 2, yb - ha, bw - 5, Math.max(1, ha));
        } else {
          ctx.fillStyle = alphaOf(T.bad, 0.45); ctx.fillRect(bx + 2, yb - 3, bw - 5, 3);
          ctx.fillStyle = T.bad; ctx.font = '8px ui-monospace, monospace'; ctx.textBaseline = 'alphabetic'; ctx.fillText('−∞', bx + bw / 2, yb - 7);
        }
        ctx.fillStyle = idx === S.chosen ? T.n14 : T.n10;
        ctx.font = (idx === S.chosen ? 'bold ' : '') + '8.5px ui-monospace, monospace'; ctx.textBaseline = 'top';
        const lbl = vocab[idx] === ' ' ? '␣' : vocab[idx];
        ctx.fillText(lbl.length > 5 ? lbl.slice(0, 5) : lbl, bx + bw / 2, yb + 3);
      }
      ctx.textAlign = 'left';
      ctx.strokeStyle = T.n6; ctx.lineWidth = 1; ctx.beginPath(); ctx.moveTo(pad, yb); ctx.lineTo(W - pad, yb); ctx.stroke();
      y = yb + 20;
      const hx0 = pad + 152, hx1 = Math.min(W - pad - 150, hx0 + 230), hy = y + 6;
      ctx.strokeStyle = T.n6; ctx.lineWidth = 3; ctx.beginPath(); ctx.moveTo(hx0, hy); ctx.lineTo(hx1, hy); ctx.stroke();
      ctx.fillStyle = T.violet; ctx.beginPath(); ctx.arc(hx0 + (hx1 - hx0) * st.fit, hy, 6, 0, Math.PI * 2); ctx.fill();
      r.label('model agrees with schema  (drag)', pad, hy + 4, { color: T.violet, font: '10px ui-monospace, monospace' });
      r.label(`kept mass ${(S.keptMass * 100).toFixed(2)}%`, hx1 + 12, hy + 4, { color: S.keptMass < 0.01 ? T.bad : T.n11, font: '10px ui-monospace, monospace' });
      geom.fitHandle = { x0: hx0, x1: hx1, y: hy };
    } else {
      r.label(S.kind === 'jump' ? 'no logits this step — jump-forward emitted the characters itself, so the model was never called'
        : RUN.error ? 'no logits — the automaton never compiled, so every token is −∞'
          : S.ok ? 'sequence complete — every token above was grammar-legal by construction' : 'sequence stopped',
      pad, y + 14, { color: S.kind === 'jump' ? T.warnDeep : RUN.error ? T.bad : T.n11, font: '11px ui-monospace, monospace' });
    }

    // ===== hover-to-inspect over the vocabulary strip
    if (page.pointer.over && geom.vocab && !grab) {
      const g = geom.vocab, p = page.pointer;
      const c = Math.floor((p.x - g.x) / g.cw), rr = Math.floor((p.y - g.y) / g.ch), idx = rr * g.cols + c;
      if (c >= 0 && c < g.cols && rr >= 0 && rr < g.rows && idx >= 0 && idx < V) {
        const label = vocab[idx] === ' ' ? '␣ (space)' : `"${vocab[idx]}"`;
        if (RUN.error) page.setTip(`${label}   (id ${idx})\nmasked: the schema does not compile, so no token is legal\n${RUN.error}`);
        else if (g.skip) page.setTip(`${label}   (id ${idx})\nno mask was built this step — the model was not called`);
        else {
          const sim = tokenSim(RUN.G, S.node, S.stack || [], vocab[idx]);
          const atNode = RUN.G.nodes[sim.node != null ? sim.node : S.node];
          const cls = cdArr[idx] ? 'context-DEPENDENT — its verdict reads the stack, so it is re-checked every step' : 'context-independent — verdict cached against this node, never rescanned';
          const why = sim.ok
            ? `legal: walks "${RUN.G.nodes[S.node].label}" → "${atNode.label}"`
            : `masked: node "${atNode.label}" rejects '${sim.at}'\nthat node accepts: ${atNode.edges.map((e) => e.desc).join(' · ') || '(nothing — dead)'}`;
          page.setTip(`${label}   (id ${idx})\n${why}\n${cls}`);
        }
      }
    }

    // ===== readout
    const stt = RUN.stats;
    const totalChecks = stt.cdChecks + stt.ciSkipped;
    const cdPct = totalChecks ? (100 * stt.cdChecks / totalChecks) : 0;
    let o = `step ${S.i + 1}/${RUN.steps.length} · ${S.label}\n`;
    if (RUN.error) {
      o += `dead state: all ${V} tokens are −∞, so there is no legal continuation at all — the decoder cannot emit anything. Fix the schema to compile it.\n`;
    } else {
      o += `node "${RUN.G.nodes[S.node] ? RUN.G.nodes[S.node].label : '—'}" · stack [${(S.stack || []).map((e) => e.sym).join(' ') || 'empty'}] · emitted "${S.text || ''}"\n`;
      if (S.kind === 'token') o += `mask: ${S.nLegal}/${V} legal · ${S.nCd} context-dependent re-checked live · kept probability mass ${(S.keptMass * 100).toFixed(2)}% · the token it forced, "${S.tok}", was rank ${S.rank + 1}/${V} in the model's OWN preference${S.violated ? ' — and with the mask off it has just left the grammar' : ''}\n`;
      if (S.kind === 'jump') o += `jump-forward: "${S.jumped}" was the only path out of that node, so ${S.jumped.length} characters cost zero model calls — and the boundary it resumes on is not one the tokenizer would have chosen.\n`;
      o += `cache: ${stt.fills} node masks built (${stt.scanned} token scans) · ${stt.hits} node hits reused · ${cdPct.toFixed(1)}% of this run's checks were context-dependent (a real 128k vocabulary is ~1%) · the grammar compiled once, to ${RUN.nodeCount} nodes\n`;
      o += `run: ${RUN.modelCalls} model call${RUN.modelCalls === 1 ? '' : 's'} · ${RUN.jumpChars} characters emitted by jump-forward · ${seams} retokenization seam${seams === 1 ? '' : 's'} so far\n`;
    }
    o += `trades: the distribution is TRUNCATED, not corrected — low kept mass means output that parses and reads as gibberish · mask compute is per-request state, so it does not batch like a GEMM · compilation is paid per unique schema, free once cached and painful when every request brings a new one (the old "5–15% slower" figure predates the mask cache; on a repeated schema it is near zero).   logits: synthetic · tier:${r.name}`;
    page.setReadout(o);
  },
}).then((page) => {
  window.__cdPage = page;
  const q = new URLSearchParams(location.search);
  const bool = (v, dflt) => (v == null ? dflt : !(v === '0' || v === 'false'));
  if (q.has('schema')) page.controls.set('schema', q.get('schema'), { rebuild: true });
  if (q.has('jump')) page.controls.set('jump', bool(q.get('jump'), false), { rebuild: true });
  if (q.has('mask')) page.controls.set('mask', bool(q.get('mask'), true), { rebuild: true });
  if (q.has('fit')) page.controls.set('fit', parseFloat(q.get('fit')), { rebuild: true });
  if (q.has('seed')) page.controls.set('seed', parseInt(q.get('seed'), 10), { rebuild: true });
  if (q.has('step') && page.controls._transport) { const tr = page.controls._transport; tr.rebuildIfDirty(); tr.pause(); tr.seek(+q.get('step')); }
  if (q.get('play') === '1' && page.controls._transport) page.controls._transport.play();
  if (q.has('hover')) { const [hx, hy] = q.get('hover').split(',').map(Number); page.pointer.x = hx; page.pointer.y = hy; page.pointer.over = true; }
  page.redraw();
});
