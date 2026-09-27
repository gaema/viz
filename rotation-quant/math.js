// Normalized fast Walsh–Hadamard. Orthogonal, so the L2 norm is unchanged.
// A spike spreads across every coordinate and the max shrinks. The same
// rotation has to be applied to every matmul operand: rotating only the
// activation computes a different product.

export function l2(x) {
  let s = 0;
  for (const v of x) s += v * v;
  return Math.sqrt(s);
}

export function maxAbs(x) {
  let m = 0;
  for (const v of x) m = Math.max(m, Math.abs(v));
  return m;
}

export function fwht(x) {
  const n = x.length;
  if (n === 0 || (n & (n - 1)) !== 0) throw new Error('fwht length must be a power of two');
  const out = x.slice();
  for (let h = 1; h < n; h *= 2) {
    for (let i = 0; i < n; i += h * 2) {
      for (let j = i; j < i + h; j++) {
        const u = out[j], v = out[j + h];
        out[j] = u + v;
        out[j + h] = u - v;
      }
    }
  }
  const s = 1 / Math.sqrt(n);
  for (let i = 0; i < n; i++) out[i] *= s;
  return out;
}

function matvec(W, x) {
  return W.map((row) => row.reduce((s, w, j) => s + w * x[j], 0));
}

// Difference between rotating only x and rotating neither. Nonzero means the
// weight has to be rotated with the activation or the product changes.
export function activationOnlyDelta(W, x) {
  const rx = fwht(x);
  const bare = matvec(W, x);
  const onlyX = matvec(W, rx);
  let s = 0;
  for (let i = 0; i < bare.length; i++) { const d = bare[i] - onlyX[i]; s += d * d; }
  return Math.sqrt(s);
}
