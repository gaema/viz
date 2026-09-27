// Switch-style aux loss. f is the fraction of routed assignments, and a
// capacity drop is still a routed assignment: leaving it out shrinks the
// loss exactly when the router is overloaded. With no drops and a uniform
// dispatch, E · Σ f·P is 1.

import { softmax } from '../framework/tensor.js';

const M = (r, c) => ({ data: new Float32Array(r * c), rows: r, cols: c });

export function route(logits, N, E, k, cap, upto) {
  const g = M(N, E);
  for (let i = 0; i < N; i++) g.data.set(softmax(logits.data.subarray(i * E, (i + 1) * E)), i * E);
  const sel = [];
  for (let i = 0; i < N; i++) {
    sel.push(Array.from({ length: E }, (_, e) => e).sort((a, b) => g.data[i * E + b] - g.data[i * E + a]).slice(0, k));
  }
  const load = new Int32Array(E), drop = new Int32Array(E);
  let drops = 0, assigned = 0;
  for (let i = 0; i <= upto && i < N; i++) {
    for (const e of sel[i]) {
      if (load[e] < cap) { load[e]++; assigned++; }
      else { drop[e]++; drops++; }
    }
  }
  const seen = Math.min(upto, N - 1) + 1;
  const P = new Float32Array(E);
  for (let i = 0; i < seen; i++) for (let e = 0; e < E; e++) P[e] += g.data[i * E + e] / seen;
  const dispatched = assigned + drops;
  const f = new Float32Array(E);
  for (let e = 0; e < E; e++) f[e] = dispatched > 0 ? (load[e] + drop[e]) / dispatched : 0;
  let aux = 0;
  for (let e = 0; e < E; e++) aux += f[e] * P[e];
  aux *= E;
  return { g, sel, load, drop, drops, P, aux, assigned, f };
}
