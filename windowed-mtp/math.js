// A multi-token draft reads a sink plus a recent window. The target still
// sees the whole context, so a fact in the dropped middle is accepted only
// at the blind rate, and a fact the draft can see is accepted outright.
// Overlap is subtracted once: a short context is not counted twice.

export function draftSpan(length, sink, window) {
  const L = Math.max(1, length | 0);
  const sinkN = Math.min(Math.max(0, sink | 0), L);
  const winN = Math.min(Math.max(0, window | 0), L);
  const overlap = Math.max(0, sinkN + winN - L);
  const visible = sinkN + winN - overlap;
  return { length: L, sink: sinkN, window: winN, overlap, visible, dropped: L - visible };
}

export function acceptLabel(pos, length, sink, window, blind) {
  const span = draftSpan(length, sink, window);
  const p = Math.max(0, Math.min(span.length - 1, pos | 0));
  const seen = p < span.sink || p >= span.length - span.window;
  const blindText = Math.min(1, Math.max(0, +blind)).toFixed(2);
  return {
    ...span,
    pos: p,
    seen,
    blind: blindText,
    accept: seen ? '1.00' : blindText,
  };
}

// The readout prints sink, window, overlap, and the visible count. The
// visible count is those three integers, not sink plus window.
export function readSentence(span) {
  return `length ${span.length}, sink ${span.sink}, window ${span.window}, overlap ${span.overlap}: `
    + `draft reads ${span.visible} of ${span.length} (${span.sink} + ${span.window} - ${span.overlap})`;
}
