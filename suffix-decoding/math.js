// Suffix-tree speculation, plus a Jacobi lookahead that does not consult the tree.
// The suffix path is the original generator, trie, and longest-common-prefix.
// Jacobi proposes a window from an explicit trajectory and keeps only the
// prefix the reference itself emits.
import { rng } from '../framework/tensor.js';

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

export function genStream(seed, rho, corpusTok, contTok) {
  const next = rng((seed | 0) * 2654435761 + 12345);
  const toks = [];
  const total = corpusTok + contTok;
  while (toks.length < total) {
    if (toks.length > 24 && next() < rho) {
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

export const tokenize = (s) => String(s || '').toLowerCase().replace(/[^a-z0-9.\-_ ]+/g, ' ')
  .split(/\s+/).filter(Boolean).slice(0, 40);

export function buildTree(tokens, D) {
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

export function walk(tree, pat) {
  let n = tree.root;
  for (const t of pat) { const c = n.kids.get(t); if (!c) return null; n = c; }
  return n;
}

export const MIN_COUNT = 2;

export function longestSuffixMatch(tree, ctx, D) {
  const cap = Math.min(D - 1, ctx.length);
  let longest = null;
  for (let m = cap; m >= 1; m--) {
    const pat = ctx.slice(ctx.length - m);
    const n = walk(tree, pat);
    if (!n) continue;
    if (!longest) longest = m;
    if (n.count >= MIN_COUNT) return { node: n, len: m, pat, longest };
  }
  return { node: tree.root, len: 0, pat: [], longest };
}

export const bestChild = (node) => {
  let best = null;
  for (const c of node.kids.values()) if (!best || c.count > best.count) best = c;
  return best;
};

export function propose(tree, ctx0, L, D) {
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

export function jacobiPropose(window, trajectory) {
  const w = Math.max(0, window | 0);
  const proposal = trajectory.slice(0, w);
  const ngrams = [];
  for (let i = 0; i < proposal.length; i++) {
    for (let j = i + 1; j <= proposal.length; j++) ngrams.push(proposal.slice(i, j));
  }
  return { proposal, ngrams };
}

export function verifyPrefix(proposal, target) {
  const toks = proposal.map((p) => (typeof p === 'string' ? p : p.tok));
  const acc = [];
  for (let i = 0; i < toks.length && i < target.length; i++) {
    if (toks[i] !== target[i]) break;
    acc.push(toks[i]);
  }
  return acc;
}

export function jacobiBill(window, trajectory, target) {
  const { proposal } = jacobiPropose(window, trajectory);
  const acc = verifyPrefix(proposal, target);
  const proposed = String(proposal.length);
  const acceptedN = String(acc.length);
  const rejected = String(Number(proposed) - Number(acceptedN));
  return {
    proposed, acceptedN, rejected, proposal, acc,
    sentence: `${proposed} - ${rejected} = ${acceptedN}`,
  };
}

export function simulate(st, rho) {
  const D = st.depth | 0, L = st.plen | 0, R = st.rounds | 0;
  const g = genStream(st.seed, rho, (st.corpus | 0) * 7, 24 + R * (L + 1));
  const extra = tokenize(st.extra);
  const base = extra.concat(g.corpus);
  const seen = base.slice();
  const truth = g.cont;
  const rounds = [];
  let ti = 0, accSum = 0, emitted = 0;
  for (let r = 0; r < R && ti < truth.length; r++) {
    const tree = buildTree(seen, D);
    const ctx = seen.slice(Math.max(0, seen.length - (D - 1)));
    const m = longestSuffixMatch(tree, ctx, D);
    let prop;
    let acc = 0;
    let bill = null;
    if (st.spec === 'jacobi') {
      const last = ctx.length ? ctx[ctx.length - 1] : '.';
      const trajectory = [];
      for (let i = 0; i < L; i++) trajectory.push(last + String(i));
      bill = jacobiBill(L, trajectory, truth.slice(ti, ti + L));
      prop = bill.proposal.map((tok) => ({
        tok, count: 1, ctxCount: 1, matchLen: 0, pat: [],
        sibs: [{ tok, count: 1 }], jacobi: true,
      }));
      acc = bill.acc.length;
    } else {
      prop = propose(tree, ctx, L, D);
      while (acc < prop.length && ti + acc < truth.length && prop[acc].tok === truth[ti + acc]) acc++;
    }
    const commit = truth.slice(ti, Math.min(truth.length, ti + acc + 1));
    for (const t of commit) seen.push(t);
    rounds.push({
      r, tree, ctx, match: m, prop, acc, commit,
      truth: truth.slice(ti, ti + Math.max(prop.length, acc + 1)),
      corpusLen: seen.length - commit.length, seenAt: seen.slice(),
      jacobi: st.spec === 'jacobi', bill,
    });
    accSum += acc; emitted += commit.length; ti += commit.length;
  }
  const n = rounds.length || 1;
  return { rounds, base, extraLen: extra.length, truth, mean: accSum / n, emitted, tpf: emitted / n, D, L, R: rounds.length, spec: st.spec === 'jacobi' ? 'jacobi' : 'suffix' };
}

export const draftAccept = (a0, rho) => Math.min(0.985, a0 + (1 - a0) * 0.18 * rho);
export function draftMean(a0, rho, L) {
  const a = draftAccept(a0, rho);
  return a >= 0.999 ? L : (a - Math.pow(a, L + 1)) / (1 - a);
}
export function shownTpf(accepted) {
  const aS = accepted.toFixed(2);
  return { aS, tpf: (Number(aS) + 1).toFixed(2) };
}

// Names the printed accept against the printed draft and against plain decode.
// Equal printed operands are a tie. Behind the draft is still above one token
// per forward until the printed accept itself rounds to zero.
export function curveVerdict(accepted, draft) {
  const shown = shownTpf(accepted);
  const dS = Number(draft).toFixed(2);
  const cmp = Number(shown.aS) - Number(dS);
  const tie = cmp === 0;
  const ahead = cmp > 0;
  const plain = '1.00';
  let text;
  if (tie) {
    text = `the suffix tree ties the modelled draft here — ${shown.aS} accepted versus ${dS}`;
  } else if (ahead) {
    text = `the suffix tree is ahead of the modelled draft here — ${shown.aS} accepted versus ${dS}, and it paid for no second model`;
  } else if (Number(shown.tpf) > Number(plain)) {
    text = `the suffix tree is behind the modelled draft here — ${shown.aS} accepted is ${shown.tpf} tokens per forward, above plain decode's ${plain}`;
  } else {
    text = `the suffix tree matches plain decode here — ${shown.tpf} tokens per forward`;
  }
  return { ahead, tie, text, accepted: shown.aS, tpf: shown.tpf, draft: dS };
}

export const cardBlurb = 'Two ways to propose tokens without a draft model. Suffix mode indexes text already seen in a depth-bounded suffix tree and walks the most frequent continuation. Averaging four seeds, accepted length is longer when the text is fully repeated than when it is novel, and that curve is not higher at every step between them. One run on screen can stay flat or get shorter. Jacobi mode writes a lookahead window and pools n-grams from that trajectory even when the tree has no match, then keeps only the prefix the reference itself would emit. The Jacobi window is the last context token plus a position index, not a copy of a repeated n-gram. Either mode verifies by the longest agreeing prefix.';
