// Every group of four contiguous weights keeps at most two nonzeros: the two
// largest magnitudes. Ties keep the earlier index. A group that is already
// that sparse stays within the same limit.

export function fmt2(x) {
  const n = +x;
  return (Number.isFinite(n) ? n : 0).toFixed(2);
}

export function pruneGroup(values) {
  const src = (values || []).slice(0, 4);
  while (src.length < 4) src.push(0);
  const printed = src.map((v) => fmt2(v));
  const nums = printed.map(Number);
  const order = nums.map((_, i) => i).sort((a, b) => {
    const d = Math.abs(nums[b]) - Math.abs(nums[a]);
    return d !== 0 ? d : a - b;
  });
  const keep = new Set(order.slice(0, 2));
  const out = nums.map((v, i) => (keep.has(i) ? v : 0));
  const a = fmt2(Math.abs(nums[order[0]]));
  const b = fmt2(Math.abs(nums[order[1]]));
  const sum = fmt2(Number(a) + Number(b));
  const nonzero = nums.filter((v) => v !== 0).length;
  const kept = out.filter((v) => v !== 0).length;
  const dropped = nonzero - kept;
  return {
    printed,
    out: out.map((v) => fmt2(v)),
    keep: [...keep].sort((i, j) => i - j),
    a, b, sum,
    nonzero: String(nonzero),
    kept: String(kept),
    dropped: String(dropped),
  };
}

export function magnitudeSentence(row) {
  return `${row.a} + ${row.b} = ${row.sum}`;
}

export function dropSentence(row) {
  return `${row.nonzero} - ${row.kept} = ${row.dropped}`;
}
