// One example is laid out prefix, then suffix, then the middle. Causal
// attention on that order lets the middle see the whole suffix. Two examples
// under one causal mask let the second see every token of the first. Isolating
// them drops those pairs to zero. The cross count is the square of one
// example's length, or zero.

export function fimLabel(prefix, suffix, middle, isolate) {
  const P = Math.max(1, prefix | 0);
  const S = Math.max(1, suffix | 0);
  const M = Math.max(1, middle | 0);
  const one = P + S + M;
  const blocked = !!isolate;
  const cross = blocked ? 0 : one * one;
  return { prefix: P, suffix: S, middle: M, one, cross, blocked, suffixSeen: S };
}

export function fimSentence(row) {
  const cross = row.blocked
    ? `cross pairs ${row.cross}`
    : `cross pairs ${row.cross} = ${row.one} × ${row.one}`;
  return `prefix ${row.prefix}, suffix ${row.suffix}, middle ${row.middle}. `
    + `the first middle token sees ${row.suffixSeen} of ${row.suffix} suffix tokens. ${cross}.`;
}
