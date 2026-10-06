// Loss mode mixes the router toward uniform by λ. Bias mode adds a per-expert
// offset used only for top-k selection. The combine weight is the softmax of
// the raw scores of the selected experts, and the auxiliary-loss weight is 0.

export function fmt2(x) {
  const n = +x;
  return (Number.isFinite(n) ? n : 0).toFixed(2);
}

export function auxMix(skew, lam, userMult) {
  const E = skew.length;
  const inv = 1 / E;
  const raw = Float32Array.from(skew, (s, e) => ((1 - lam) * s + lam * inv) * (userMult ? userMult[e] : 1));
  let rs = 0;
  for (const x of raw) rs += x;
  const eff = Float32Array.from(raw, (x) => (rs ? x / rs : inv));
  return { eff, lam };
}

export function biasRoute(logits, bias, k, step, tokens) {
  const E = logits.length;
  const T = tokens | 0;
  const kk = Math.max(1, Math.min(k | 0, E));
  const sel = logits.map((x, i) => x + bias[i]);
  const order = logits.map((_, i) => i).sort((a, b) => (sel[b] - sel[a]) || (a - b));
  const picked = order.slice(0, kk);
  const raw = picked.map((i) => logits[i]);
  const peak = Math.max(...raw);
  const ex = raw.map((x) => Math.exp(x - peak));
  const z = ex.reduce((a, b) => a + b, 0);
  const combine = picked.map((i, j) => ({ i, w: ex[j] / z }));
  const load = new Array(E).fill(0);
  for (const i of picked) load[i] = T;
  const target = (T * kk) / E;
  const next = bias.slice();
  const moves = [];
  for (let e = 0; e < E; e++) {
    if (load[e] === target) continue;
    const before = fmt2(bias[e]);
    const st = fmt2(step);
    const after = load[e] > target ? fmt2(Number(before) - Number(st)) : fmt2(Number(before) + Number(st));
    next[e] = Number(after);
    moves.push({ e, dir: load[e] > target ? 'down' : 'up', before, step: st, after });
  }
  return { picked, combine, load, target, bias: next, lam: 0, moves };
}

export function biasRun(logits, k, step, tokens, rounds) {
  let bias = logits.map(() => 0);
  let row = biasRoute(logits, bias, k, step, tokens);
  const n = Math.max(1, rounds | 0);
  for (let i = 1; i < n; i++) row = biasRoute(logits, row.bias, k, step, tokens);
  return row;
}

export function moveSentence(move) {
  return move.dir === 'down'
    ? `${move.before} - ${move.step} = ${move.after}`
    : `${move.before} + ${move.step} = ${move.after}`;
}

export function compareCaptions(balance) {
  if (balance === 'bias') return { a: 'bias route, auxiliary weight 0', b: 'bias route, auxiliary weight 0' };
  return { a: 'λ=0 — router collapse (skewed)', b: 'λ=1 — balanced (uniform)' };
}

export const cardBlurb = 'Two routes share this picture. Auxiliary-loss mode mixes the router from its skewed preference at λ=0 toward uniform at λ=1; dropped tokens are the part of a bar above the capacity line, and dragging a bar rescales that mix. Selection-bias mode adds a per-expert offset used only to choose the top-k. The combine weight is the softmax of the raw scores of the selected experts, the auxiliary-loss weight is 0, an overloaded expert bias decreases by the step, and an underloaded expert bias increases by the step. The λ compare panes follow λ only in auxiliary-loss mode; in selection-bias mode both panes are the bias route.';
