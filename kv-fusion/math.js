// Non-prefix KV fusion. A prefix cache reuses a chunk only when nothing
// precedes it. Fusion keeps the chunk wherever it sits and recomputes a
// fraction of its tokens. The residual is the unrecomputed share of the
// mismatch. Every printed total is the sum of the printed header and the
// printed fraction times the printed chunk length.

function showSum(v) {
  return Number.isInteger(v) ? String(v) : v.toFixed(2);
}

export function fusionBill(tokens, header, frac, mismatch) {
  const t = Math.max(0, tokens | 0);
  const h = Math.max(0, header | 0);
  const fracText = Math.min(1, Math.max(0, +frac)).toFixed(2);
  const mismatchText = Math.min(1, Math.max(0, +mismatch)).toFixed(2);
  const f = Number(fracText);
  const m = Number(mismatchText);
  const prefixChunk = h === 0 ? 0 : t;
  const fusionChunk = f * t;
  const cold = h + t;
  const prefixTotal = h + prefixChunk;
  const fusionTotal = h + fusionChunk;
  const error = ((1 - f) * m).toFixed(2);
  return {
    tokens: t,
    header: h,
    frac: fracText,
    mismatch: mismatchText,
    prefixChunk,
    fusionChunk,
    cold,
    prefixTotal,
    fusionTotal,
    coldText: showSum(cold),
    prefixText: showSum(prefixTotal),
    fusionText: showSum(fusionTotal),
    error,
  };
}

// The sentence compares the two printed totals. It does not assume that a
// non-prefix chunk is always cheaper to fuse: a full recompute ties, and a
// true prefix with a zero fraction also ties.
export function fusionCompare(bill) {
  const pre = Number(bill.prefixText);
  const fus = Number(bill.fusionText);
  if (fus < pre) return 'Fusion pays fewer tokens than the prefix cache.';
  if (fus > pre) return 'The prefix cache pays fewer tokens than fusion.';
  return 'Fusion and the prefix cache pay the same number of tokens.';
}
