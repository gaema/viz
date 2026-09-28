// Each weight becomes -1, 0, or +1 by rounding value/scale into that set.
// The reconstruction printed for a code is that code times the printed scale.
// The two sums are the printed weights and the printed reconstructions.

export const WEIGHTS = [0.9, 0.2, -0.8, 0.1, -0.4];

export function ternaryBill(scale) {
  const scaleText = Math.max(0, +scale).toFixed(2);
  const s = Number(scaleText);
  const weights = WEIGHTS.map((w) => w.toFixed(2));
  const codes = WEIGHTS.map((w) => {
    if (s === 0) return 0;
    return Math.max(-1, Math.min(1, Math.round(w / s)));
  });
  const recon = codes.map((c) => (c * s).toFixed(2));
  const full = weights.map(Number).reduce((a, b) => a + b, 0);
  const tern = recon.map(Number).reduce((a, b) => a + b, 0);
  let abs = 0;
  for (let i = 0; i < weights.length; i++) abs += Math.abs(Number(weights[i]) - Number(recon[i]));
  const nonzero = codes.filter((c) => c !== 0).length;
  return {
    scale: scaleText,
    weights,
    codes,
    recon,
    full: full.toFixed(2),
    tern: tern.toFixed(2),
    abs: abs.toFixed(2),
    nonzero,
    count: codes.length,
  };
}

export function ternarySentence(row) {
  return `scale ${row.scale}: ${row.nonzero} of ${row.count} weights survive. `
    + `sum ${row.full} becomes ${row.tern}. absolute error ${row.abs}.`;
}
