// Pair counts for one video-transformer layer. Full attention scores every
// space-time pair. A window scores N times the neighbours inside the window,
// which is fewer pairs whenever the window does not cover the whole volume.
// `win` is the window extent in tokens along each axis. This is the same
// count the video-time page draws.

// Latent width and height after a per-axis integer downsample. A side that
// is not a multiple of the factor floors; it is not the real division.
export function latentAxes(width, height, sw) {
  const d = +sw;
  return { latW: Math.floor(width / d), latH: Math.floor(height / d) };
}

const UNITS = [[1e18, 'E'], [1e15, 'P'], [1e12, 'T'], [1e9, 'G'], [1e6, 'M'], [1e3, 'k']];
const SCALE = { E: 1e18, P: 1e15, T: 1e12, G: 1e9, M: 1e6, k: 1e3 };

export function fmtN(v) {
  if (!isFinite(v)) return '∞';
  for (const [m, s] of UNITS) if (v >= m) {
    const q = v / m;
    return (q < 10 ? q.toFixed(2) : q < 100 ? q.toFixed(1) : q.toFixed(0)) + ' ' + s;
  }
  return String(Math.round(v));
}

export function parseN(s) {
  const m = String(s).trim().match(/^([0-9.]+)\s*([EPTGMk])$/);
  if (!m) return Number(s);
  return Number(m[1]) * SCALE[m[2]];
}

export function fmtPct(p) {
  const body = p >= 1e6 ? fmtN(p)
    : p >= 1000 ? p.toFixed(0)
    : p >= 10 ? p.toFixed(0)
    : p >= 1 ? p.toFixed(1)
    : p >= 0.01 ? p.toFixed(2)
    : p > 0 ? '<0.01'
    : '0';
  return body + '%';
}

// The percent divides the two count strings in the sentence, not the
// unrounded counts. 1.34 k / 128 k is 1.0%, not the true 1.05 printed as 1.1%.
export function shareOf(part, whole) {
  const a = fmtN(part);
  const b = fmtN(whole);
  const pct = whole > 0 ? 100 * parseN(a) / parseN(b) : 0;
  return { a, b, pct: fmtPct(pct), text: `${a} vs ${b} = ${fmtPct(pct)}` };
}

export function attentionPairs({ N, S, Tt, win }) {
  const w = win | 0;
  const k = Math.min(S, w * w) * Math.min(Tt, w);
  return { full: N * N, factorised: N * (S + Tt), windowed: N * k, k };
}
