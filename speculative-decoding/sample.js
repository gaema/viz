// One speculative round. Position j's target and draft distributions are
// salted by the tokens already kept in this round, so a different accepted
// prefix changes the next distribution. The accept test is min(1, p/q), and
// a rejection draws from the residual normalize(max(0, p - q)).

import { softmax, seededRandn, rng } from '../framework/tensor.js';

export const VOCAB = ['the', 'cat', 'sat', 'on', 'a', 'mat', 'and', 'dog', 'ran', 'far', 'then', 'slept'];
const V = VOCAB.length;

function saltOf(kept) {
  let s = 0;
  for (let i = 0; i < kept.length; i++) s = (s + (kept[i] + 1) * (i + 3) * 97) >>> 0;
  return s;
}

// The printed ratio divides the printed p and q. Rounding the true ratio
// is a different digit when those strings have already been shortened.
// A printed draft of 0 has no finite quotient.
export function ratioLabel(p, q, digits) {
  const ps = Number(p).toFixed(digits);
  const qs = Number(q).toFixed(digits);
  const pn = Number(ps);
  const qn = Number(qs);
  const ratio = qn === 0 ? (pn === 0 ? '—' : '∞') : (pn / qn).toFixed(digits);
  return { p: ps, q: qs, ratio, text: `p/q = ${ratio}` };
}

// The accept decision stays on the unrounded ratio. This caption names
// p/q only when the printed quotient makes the same comparison.
export function acceptCaption(u, p, q, ratio, digits) {
  const shown = ratioLabel(p, q, digits);
  const trueThr = Math.min(1, Number(ratio));
  const qn = Number(shown.q);
  const printed = qn === 0 ? null : Number(shown.p) / qn;
  const draw = Number(u).toFixed(digits);
  const trueLe = Number(u) <= trueThr;
  const printedLe = printed != null && Number(draw) <= Math.min(1, printed);
  const agree = printed != null && trueLe === printedLe;
  const mark = (agree ? printedLe : trueLe) ? '≤' : '>';
  if (agree) return `draw ${draw} ${mark} min(1, p/q) = ${Math.min(1, printed).toFixed(digits)}`;
  return `draw ${draw} ${mark} ${trueThr.toFixed(digits)}`;
}

// Net rate divides the printed token rate by 1 + k times the printed
// draft cost. Rounding the true rate is a different percent.
export function netLabel(tpf, k, cost) {
  const rate = Number(tpf).toFixed(2);
  const unit = Number(cost).toFixed(2);
  const denom = 1 + (k | 0) * Number(unit);
  const pct = denom === 0 ? '∞' : (100 * Number(rate) / denom).toFixed(0);
  const worth = pct !== '∞' && Number(pct) >= 100;
  const gauge = `net ${pct}% of plain decode ${worth ? '— worth it' : '— SLOWER than not speculating'}`;
  const charge = `Charging each drafted token ${unit} of a target forward: net ${pct}% of plain decode${worth ? '.' : ' — the draft is not earning its keep.'}`;
  return { rate, cost: unit, pct, worth, gauge, charge };
}

export function distributionAt(seed, round, index, kept) {
  return softmax(seededRandn(seed * 7919 + round * 97 + index * 13 + saltOf(kept) + 1, V, { std: 1.35 }));
}

function draftBias(seed, round, index, kept) {
  return softmax(seededRandn(seed * 7919 + round * 97 + index * 13 + saltOf(kept) + 500003, V, { std: 1.35 }));
}

function sampleIdx(dist, u) {
  let c = 0;
  for (let i = 0; i < dist.length; i++) { c += dist[i]; if (u <= c) return i; }
  return dist.length - 1;
}

export function buildRun(st) {
  const K = st.k | 0, R = st.rounds | 0, a = +st.quality, seed = st.seed | 0;
  const rounds = [];
  let tokens = 0, wasted = 0;
  for (let r = 0; r < R; r++) {
    const next = rng(seed * 131071 + r * 7 + 3);
    const slots = [];
    const kept = [];
    for (let j = 0; j < K; j++) {
      const p = distributionAt(seed, r, j, kept);
      const b = draftBias(seed, r, j, kept);
      const q = new Float32Array(V);
      for (let i = 0; i < V; i++) q[i] = a * p[i] + (1 - a) * b[i];
      const u1 = next(), u2 = next(), u3 = next();
      const x = sampleIdx(q, u1);
      const ratio = q[x] > 1e-12 ? p[x] / q[x] : 0;
      const acc = u2 <= Math.min(1, ratio);
      const res = new Float32Array(V);
      let s = 0;
      for (let i = 0; i < V; i++) { const d = Math.max(0, p[i] - q[i]); res[i] = d; s += d; }
      if (s > 1e-9) { for (let i = 0; i < V; i++) res[i] /= s; } else res.set(p);
      slots.push({ p, q, x, ratio, acc, u: u2, fix: sampleIdx(res, u3) });
      if (acc) kept.push(x);
    }
    let firstRej = -1;
    for (let j = 0; j < K; j++) if (!slots[j].acc) { firstRej = j; break; }
    const nAcc = firstRej < 0 ? K : firstRej;
    const pBonus = distributionAt(seed, r, K, kept);
    const u4 = next();
    const out = [];
    for (let j = 0; j < nAcc; j++) out.push({ tok: slots[j].x, kind: 'accept' });
    out.push(firstRej >= 0 ? { tok: slots[firstRej].fix, kind: 'fix' } : { tok: sampleIdx(pBonus, u4), kind: 'bonus' });
    tokens += out.length;
    wasted += K - nAcc;
    rounds.push({ slots, firstRej, nAcc, out });
  }
  return { rounds, K, R, tokens, wasted };
}
