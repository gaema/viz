// Position correction for a cached chunk. RoPE rotates pair i by
// pos * base^(-2i/d). A key stored at `src` is rotated again by the gap
// `dest - src`, which is the same angle as a key computed at `dest`.
// A score printed next to two angles is the cosine of the printed difference.

export const ROPE_DIM = 8;
export const ROPE_BASE = 10000;

export function pairAngle(pos, pair, dim, base) {
  const d = Math.max(2, dim | 0);
  const b = base > 1 ? base : ROPE_BASE;
  return (+pos) * Math.pow(b, (-2 * (pair | 0)) / d);
}

export function rotate2(x, y, ang) {
  const c = Math.cos(ang);
  const s = Math.sin(ang);
  return [x * c - y * s, x * s + y * c];
}

export function spliceError(x, y, src, dest, pair, dim, base) {
  const aSrc = pairAngle(src, pair, dim, base);
  const aDest = pairAngle(dest, pair, dim, base);
  const gap = pairAngle(dest - src, pair, dim, base);
  const cached = rotate2(x, y, aSrc);
  const moved = rotate2(cached[0], cached[1], gap);
  const native = rotate2(x, y, aDest);
  return {
    err: Math.hypot(moved[0] - native[0], moved[1] - native[1]),
    gap,
    aSrc,
    aDest,
  };
}

export function moveErrorText(err) {
  if (err === 0) return '0';
  return err.toExponential(1);
}

export function scoreFromPrinted(qPos, kPos, pair, dim, base) {
  const q = pairAngle(qPos, pair, dim, base).toFixed(4);
  const k = pairAngle(kPos, pair, dim, base).toFixed(4);
  const diff = (Number(q) - Number(k)).toFixed(4);
  const score = Math.cos(Number(diff)).toFixed(3);
  return { q, k, diff, score };
}
