// Sinkhorn–Knopp. Repeated row and column normalisation projects a positive
// matrix onto the doubly-stochastic matrices: every row and every column
// sums to 1, so a mix of residual streams neither amplifies nor shrinks
// the carried signal as depth grows.

export function sinkhorn(M, iters = 40) {
  const n = M.length;
  let A = M.map((row) => row.map((x) => Math.max(x, 1e-12)));
  for (let t = 0; t < iters; t++) {
    A = A.map((row) => {
      const s = row.reduce((a, b) => a + b, 0);
      return row.map((x) => x / s);
    });
    const col = Array(n).fill(0);
    for (let i = 0; i < n; i++) for (let j = 0; j < n; j++) col[j] += A[i][j];
    A = A.map((row) => row.map((x, j) => x / col[j]));
  }
  return A;
}

export function rowSums(M) {
  return M.map((row) => row.reduce((a, b) => a + b, 0));
}

// Effective depth divides the contribution strings on the page.
// Rounding the true ratio is a different digit once those strings are short.
export function depthLabel(cs) {
  const shown = cs.map((c) => Number(c).toFixed(3));
  const nums = shown.map(Number);
  const s1 = nums.reduce((a, v) => a + v, 0);
  const s2 = nums.reduce((a, v) => a + v * v, 0);
  const eff = s2 > 1e-12 ? ((s1 * s1) / s2).toFixed(2) : '0.00';
  return { shown, eff };
}

// The contribution in the hover divides the two magnitudes it prints.
export function contribLabel(written, readout) {
  const w = Number(written).toFixed(4);
  const r = Number(readout).toFixed(4);
  const c = (Number(w) / (Number(r) + 1e-9)).toFixed(4);
  return { w, r, c };
}

export function colSums(M) {
  const n = M.length;
  const s = Array(n).fill(0);
  for (let i = 0; i < n; i++) for (let j = 0; j < n; j++) s[j] += M[i][j];
  return s;
}
