// Group-wise affine quant. The zero-point is part of the reconstruction:
// x' = (q - z) * s, with z = round(-min / s). Storing min and adding s*q
// disagrees with that whenever -min/s is not an integer.

export function qGroup(vals, bits) {
  let lo = Infinity, hi = -Infinity;
  for (const v of vals) { if (v < lo) lo = v; if (v > hi) hi = v; }
  const levels = (1 << bits) - 1;
  const s = (hi - lo) / levels || 1e-9;
  const z = Math.round(-lo / s);
  const q = vals.map((v) => Math.max(0, Math.min(levels, Math.round(v / s) + z)));
  const deq = q.map((c) => (c - z) * s);
  return { s, lo, hi, z, levels, q, deq };
}

// One reconstruction level per code. The same (k - z) * s the bars draw,
// not min + s*k. Those two grids differ when -min/s is not an integer.
// The printed ratio is 16 divided by the bits string on the page, not a
// second rounding of the true ratio. Those two strings differ.
export function compressionLabel(bits, group) {
  const eff = bits + (16 + 8) / group;
  const shown = eff.toFixed(2);
  const times = (16 / Number(shown)).toFixed(2);
  return { shown, times, text: `16 / ${shown} = ${times}×` };
}

// The hover equation multiplies the scale it prints. Rounding the true
// reconstruction is a different digit once that scale has been shortened.
export function weightTip(x, group, index, levels) {
  const z = group.z;
  const q = group.q[index];
  const xs = Number(x).toFixed(3);
  const barXp = ((q - z) * group.s).toFixed(3);
  let s = group.s.toFixed(3);
  for (let d = 3; d <= 12; d++) {
    const shown = group.s.toFixed(d);
    if (((q - z) * Number(shown)).toFixed(3) === barXp) { s = shown; break; }
  }
  const xp = ((q - z) * Number(s)).toFixed(3);
  const err = (Number(xs) - Number(xp)).toFixed(4);
  const text = `x = ${xs} fp16\nq = ${q} (of 0..${levels})  →  x′ = (q − z)·s = ${xp}\nerr = ${err}   [s=${s} z=${z}]`;
  return { s, z, q, x: xs, xp, err, text };
}

export function reconLevels(group) {
  const { s, z, levels } = group;
  const out = [];
  for (let k = 0; k <= levels; k++) out.push((k - z) * s);
  return out;
}
