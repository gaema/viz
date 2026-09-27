// (QK)V costs L*L*d multiply-adds for the score matrix. Q(KV) costs L*d*d
// for the key-value product. The matching second product is omitted on both
// sides, so the comparison is unchanged: they meet when the sequence length
// equals the head dimension. Shorter sequences prefer the quadratic form;
// longer ones prefer the reassociated form.

export function assocCost(form, L, d) {
  if (form === 'kv') return L * d * d;
  return L * L * d;
}

export function cheaper(L, d) {
  const qk = assocCost('qk', L, d);
  const kv = assocCost('kv', L, d);
  if (qk === kv) return 'tie';
  return qk < kv ? 'qk' : 'kv';
}
