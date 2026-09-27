// Two different wires. Prefill/decode ships K and V once. An attention/expert
// split ships the hidden state to the experts and back, once per layer, and
// that size does not grow with the context. Every total is the product of
// the integers printed beside it. Hidden size, KV size and byte width are
// fixed so the sliders move only the two quantities that trade.

export const HIDDEN = 8;
export const KV = 4;
export const BYTES = 2;

export function shipLabel(layers, seq, batch) {
  const L = String(Math.max(1, layers | 0));
  const S = String(Math.max(1, seq | 0));
  const B = String(Math.max(1, batch | 0));
  const H = String(HIDDEN);
  const K = String(KV);
  const D = String(BYTES);
  const pd = 2 * Number(L) * Number(K) * Number(S) * Number(D);
  const afd = 2 * Number(L) * Number(B) * Number(H) * Number(D);
  return { layers: L, seq: S, batch: B, hidden: H, kv: K, bytes: D, pd: String(pd), afd: String(afd) };
}

export function shipSentence(row) {
  return {
    kv: `2 × layers ${row.layers} × kv ${row.kv} × seq ${row.seq} × bytes ${row.bytes} = ${row.pd}`,
    hidden: `2 × layers ${row.layers} × batch ${row.batch} × hidden ${row.hidden} × bytes ${row.bytes} = ${row.afd}`,
  };
}

export function shipCompare(row) {
  const pd = Number(row.pd);
  const afd = Number(row.afd);
  if (pd > afd) return 'The one-time KV ship is larger.';
  if (afd > pd) return 'The per-step hidden-state ship is larger.';
  return 'The two ships are the same size.';
}
