// QK-norm divides each query and key by its own RMS before the dot product.
// A long key then stops winning just by being long. The printed mass is the
// softmax of the printed scores, not a second rounding of the true softmax.

export const QUERY = [1, 0];
export const ALIGNED = [1, 0];

export function offKey(scale) {
  const s = Math.max(0, +scale);
  return [s, s];
}

export function rms(v) {
  let acc = 0;
  for (let i = 0; i < v.length; i++) acc += v[i] * v[i];
  return Math.sqrt(acc / v.length);
}

export function qkNorm(v) {
  const r = rms(v);
  if (r === 0) return v.map(() => 0);
  return v.map((x) => x / r);
}

function dot(a, b) {
  let s = 0;
  for (let i = 0; i < a.length; i++) s += a[i] * b[i];
  return s;
}

export function massLabel(scale, norm) {
  const q = norm ? qkNorm(QUERY) : QUERY;
  const keys = [offKey(scale), ALIGNED].map((k) => (norm ? qkNorm(k) : k));
  const scores = keys.map((k) => dot(q, k).toFixed(3));
  const nums = scores.map(Number);
  const top = Math.max(...nums);
  const exps = nums.map((n) => Math.exp(n - top));
  const z = exps.reduce((a, b) => a + b, 0);
  const mass = exps.map((e) => (e / z).toFixed(3));
  let win = 0;
  for (let i = 1; i < mass.length; i++) {
    if (Number(mass[i]) > Number(mass[win])) win = i;
  }
  return { scores, mass, win };
}
