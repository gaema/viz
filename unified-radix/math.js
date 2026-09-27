// One prefix match, three stored states. Full-attention KV covers the
// whole match. Sliding-window KV covers only the tail that still fits
// the window, and never more tokens than the match itself. A recurrent
// layer reuses one checkpoint, and only when the match is non-empty.

export function unifiedReuse(match, window, recurrent) {
  const m = Math.max(0, match | 0);
  const w = Math.max(0, window | 0);
  const full = m;
  const windowed = Math.min(m, w);
  const state = m > 0 && recurrent ? 1 : 0;
  return { match: m, window: w, full, windowed, state };
}
