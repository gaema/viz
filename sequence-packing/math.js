// Pack variable-length documents into one row. The block-diagonal mask lets a
// token attend only inside its own document. A mask that is true everywhere
// still trains, and tokens read across the document boundary.

export function documentOf(lengths) {
  const doc = [];
  lengths.forEach((n, d) => { for (let i = 0; i < n; i++) doc.push(d); });
  return doc;
}

export function packMask(lengths, mode) {
  const doc = documentOf(lengths);
  const n = doc.length;
  const mask = Array.from({ length: n }, (_, i) => Array.from({ length: n }, (_, j) => (
    mode === 'full' ? true : doc[i] === doc[j]
  )));
  return { mask, doc };
}

export function crossDocumentPairs(mask, doc) {
  let n = 0;
  for (let i = 0; i < mask.length; i++) {
    for (let j = 0; j < mask.length; j++) if (mask[i][j] && doc[i] !== doc[j]) n++;
  }
  return n;
}
