// Some layers write keys and values. The next layers reuse that cache and
// only compute a query. Writers are every `share`-th layer, rounded up so
// the tail of the stack still has a cache. Writers plus readers is the
// layer count.

export function kvShare(layers, share) {
  const L = Math.max(1, layers | 0);
  const S = Math.max(1, share | 0);
  const writers = Math.ceil(L / S);
  const readers = L - writers;
  return { layers: L, share: S, writers, readers };
}

export function shareSentence(row) {
  return `layers ${row.layers}, share ${row.share}: ${row.writers} write, ${row.readers} reuse `
    + `(${row.writers} + ${row.readers} = ${row.layers}). cache ${row.writers} of ${row.layers}`;
}

export function writesAt(index, share) {
  const S = Math.max(1, share | 0);
  return (index % S) === 0;
}
