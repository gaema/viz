// Every share-th layer writes keys and values, rounded up so the tail of the
// stack still has a writer. The layers until the next writer reuse that cache
// and only compute a query. A share of 1 means every layer writes. Writers
// plus readers is the layer count.

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
