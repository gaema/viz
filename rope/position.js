// Adjacent-pair RoPE. A no-position mode returns the vector unchanged:
// deleting the rotation is a different choice from stretching the base.

// Three significant figures for a single value. Multiplying two of these
// and calling the product the angle is a different number.
export function roundShown(x) {
  if (x === 0) return '0';
  if (Math.abs(x) < 1e-2 || Math.abs(x) >= 1e4) return x.toExponential(1);
  return String(Number(x.toPrecision(3)));
}

export function angleEquation(delta) {
  return `Δ = p·θᵢ = ${roundShown(delta)} rad`;
}

// The spin ratio divides the two angles in the sentence. Rounding the
// true ratio is a different digit once those angles have been shortened.
export function spinRatio(fast, slow, lastPair) {
  const a = roundShown(fast);
  const b = roundShown(slow);
  const an = Number(a);
  const bn = Number(b);
  const ratio = bn === 0 ? (an === 0 ? '0' : '∞') : roundShown(an / bn);
  const pair = lastPair | 0;
  return { fast: a, slow: b, ratio, text: `pair 0: Δ=${a} rad   pair ${pair}: Δ=${b} rad   (ratio ${ratio}× faster)` };
}

// The hover rotates the components it prints by the cos and sin it prints.
export function rotateLabel(a, b, delta) {
  const as = Number(a).toFixed(2);
  const bs = Number(b).toFixed(2);
  const cs = Math.cos(delta).toFixed(3);
  const ss = Math.sin(delta).toFixed(3);
  const rx = (Number(as) * Number(cs) - Number(bs) * Number(ss)).toFixed(3);
  const ry = (Number(as) * Number(ss) + Number(bs) * Number(cs)).toFixed(3);
  return { a: as, b: bs, c: cs, s: ss, rx, ry };
}

export function applyPosition(vec, pos, base, nope) {
  const out = vec.slice();
  if (nope) return out;
  const d = vec.length;
  const b = base > 1 ? base : 10000;
  for (let i = 0; i < (d >> 1); i++) {
    const ang = pos * Math.pow(b, (-2 * i) / d);
    const c = Math.cos(ang), s = Math.sin(ang);
    const a = vec[2 * i], y = vec[2 * i + 1];
    out[2 * i] = a * c - y * s;
    out[2 * i + 1] = a * s + y * c;
  }
  return out;
}
