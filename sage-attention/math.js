// SageAttention-style score quantization. QK scores are rounded to int8
// against the row max, then softmax and the value mix run on the rounded
// scores. Exact attention uses the unrounded scores. The two outputs differ
// whenever a score is not already on the int8 grid.

function dot(a, b) {
  let s = 0;
  for (let i = 0; i < a.length; i++) s += a[i] * b[i];
  return s;
}

function softmax(row) {
  let m = -Infinity;
  for (const v of row) if (v > m) m = v;
  const e = row.map((v) => Math.exp(v - m));
  const z = e.reduce((a, b) => a + b, 0);
  return e.map((v) => v / z);
}

export function scores(Q, K) {
  const scale = Math.sqrt(Q[0].length);
  return Q.map((q) => K.map((k) => dot(q, k) / scale));
}

export function quantizeScores(S) {
  return S.map((row) => {
    let m = 0;
    for (const v of row) m = Math.max(m, Math.abs(v));
    const scale = m / 127 || 1;
    return row.map((v) => Math.round(v / scale) * scale);
  });
}

function mix(weights, V) {
  const d = V[0].length;
  return weights.map((w) => {
    const o = Array(d).fill(0);
    for (let j = 0; j < V.length; j++) for (let c = 0; c < d; c++) o[c] += w[j] * V[j][c];
    return o;
  });
}

export function attend(S, V) {
  return mix(S.map(softmax), V);
}

export function maxAbsDiff(A, B) {
  let m = 0;
  for (let i = 0; i < A.length; i++) for (let c = 0; c < A[i].length; c++) m = Math.max(m, Math.abs(A[i][c] - B[i][c]));
  return m;
}

export function attentionError(Q, K, V) {
  const S = scores(Q, K);
  return maxAbsDiff(attend(S, V), attend(quantizeScores(S), V));
}
