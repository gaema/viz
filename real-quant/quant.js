// Group-wise integer quantization — the math the real-quant page visualizes on
// real GPT-2 weights, and that quant.test.mjs verifies against numpy. Pure (no
// browser deps) so it's importable from Node.
//
// Symmetric int-N, per group of `group` consecutive weights:
//   scale_g = max(|w|) / (2^(bits-1) - 1)
//   q_i     = clamp(round(w_i / scale_g), -qmax, +qmax)
//   w'_i    = q_i · scale_g                    (dequantized)
// Each group stores its own scale, so a few large weights in one group don't
// blow up the precision of the rest — the whole point of group quantization.

export function groupQuant(w, n, bits, group) {
  const qmax = (1 << (bits - 1)) - 1;
  const dq = new Float32Array(n), scales = [];
  for (let g = 0; g < n; g += group) {
    const end = Math.min(g + group, n);
    let mx = 1e-12;
    for (let i = g; i < end; i++) { const a = Math.abs(w[i]); if (a > mx) mx = a; }
    const scale = mx / qmax; scales.push(scale);
    for (let i = g; i < end; i++) {
      let q = Math.round(w[i] / scale);
      if (q > qmax) q = qmax; else if (q < -qmax) q = -qmax;
      dq[i] = q * scale;
    }
  }
  let se = 0; for (let i = 0; i < n; i++) { const d = w[i] - dq[i]; se += d * d; }
  return { dq, rmse: Math.sqrt(se / n), scales: Float32Array.from(scales), bits, group };
}

export function stats(w, n) {
  let mn = Infinity, mx = -Infinity, s = 0;
  for (let i = 0; i < n; i++) { const v = w[i]; if (v < mn) mn = v; if (v > mx) mx = v; s += v; }
  const mean = s / n; let v2 = 0;
  for (let i = 0; i < n; i++) { const d = w[i] - mean; v2 += d * d; }
  return { min: mn, max: mx, mean, std: Math.sqrt(v2 / n) };
}

// histogram of values into `bins` buckets over [lo, hi]
export function histogram(w, n, bins, lo, hi) {
  const h = new Uint32Array(bins), span = (hi - lo) || 1;
  for (let i = 0; i < n; i++) { let b = Math.floor(((w[i] - lo) / span) * bins); if (b < 0) b = 0; else if (b >= bins) b = bins - 1; h[b]++; }
  return h;
}
