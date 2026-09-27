// Two orders for the same dtype list. Family order is the catalogue order
// the list was written in. Bits order is widest first; equal widths keep
// the catalogue order. Block rows sort on their amortised width.

export function orderDtypes(list, mode) {
  const rows = list.map((d, i) => ({ d, i }));
  if (mode === 'bits') rows.sort((a, b) => (b.d.bits - a.d.bits) || (a.i - b.i));
  return rows.map((x) => x.d);
}
