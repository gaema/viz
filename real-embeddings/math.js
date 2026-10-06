// A vector built so the leading prefix is the coarse part. The tail-only pair
// matches on that prefix and differs after the cut. This is not a truncation
// of a loaded model embedding.

export function fmt3(x) {
  const n = +x;
  return (Number.isFinite(n) ? n : 0).toFixed(3);
}

function cosine(u, v) {
  let dot = 0, nu = 0, nv = 0;
  const n = Math.min(u.length, v.length);
  for (let i = 0; i < n; i++) {
    dot += u[i] * v[i];
    nu += u[i] * u[i];
    nv += v[i] * v[i];
  }
  return dot / Math.sqrt(nu * nv);
}

export function nestedRead(dim, cut) {
  const d = Math.max(2, dim | 0);
  const c = Math.max(1, Math.min(d - 1, cut | 0));
  const a = Array.from({ length: d }, () => 1);
  const b = Array.from({ length: d }, (_, i) => (i < c ? 1 : -1));
  const e = Array.from({ length: d }, (_, i) => (i === 0 ? -1 : 1));
  const pre = (v) => v.slice(0, c);
  const pTail = fmt3(cosine(pre(a), pre(b)));
  const fTail = fmt3(cosine(a, b));
  const pCoarse = fmt3(cosine(pre(a), pre(e)));
  const gap = fmt3(Number(pTail) - Number(fTail));
  return { pTail, fTail, pCoarse, gap, cut: c, dim: d, a, b, e };
}

export function nestedSentence(row) {
  return `${row.pTail} - ${row.fTail} = ${row.gap}`;
}

export const cardBlurb = 'The default path embeds the words you type. Load all-MiniLM-L6-v2 for vectors from that trained model, or stay on the labelled synthetic stand-in. The nested toggle draws a different set of vectors built so the leading prefix is the coarse part: the prefix ties a pair that differs only past the cut, the prefix separates a pair that differs inside the cut, and the full vector separates the tail-only pair. Those nested vectors are constructed for this demo. They are not a truncation of the loaded model.';
