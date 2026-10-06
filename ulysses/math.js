// One all-to-all gives each device the full sequence for an equal share of heads.
// Attention on that share is the same attendHead the single-device reference uses.
// A second exchange restores each device's sequence shard with every head.
// A head count that does not divide by the device count is refused.

export function attendHead(q, k, v) {
  const n = q.length;
  const out = new Array(n);
  for (let t = 0; t < n; t++) {
    let max = -Infinity;
    const scores = new Array(n);
    for (let s = 0; s < n; s++) {
      scores[s] = q[t] * k[s];
      if (scores[s] > max) max = scores[s];
    }
    let z = 0;
    const ex = new Array(n);
    for (let s = 0; s < n; s++) { ex[s] = Math.exp(scores[s] - max); z += ex[s]; }
    let y = 0;
    for (let s = 0; s < n; s++) y += (ex[s] / z) * v[s];
    out[t] = y;
  }
  return out;
}

export function ulyssesPlan(heads, devices, qkv) {
  const H = heads | 0;
  const D = devices | 0;
  const headsText = String(H);
  const devicesText = String(Math.max(0, D));
  if (D <= 0 || H <= 0 || H % D !== 0) {
    const rem = String(D > 0 ? H % D : H);
    return { ok: false, heads: headsText, devices: devicesText, rem, per: null, ref: null, local: null, restored: null, ranges: null };
  }
  const per = H / D;
  const seq = qkv.q[0].length;
  const ref = [];
  for (let h = 0; h < H; h++) ref.push(attendHead(qkv.q[h], qkv.k[h], qkv.v[h]));
  const local = [];
  for (let d = 0; d < D; d++) {
    const slice = [];
    for (let h = d * per; h < (d + 1) * per; h++) slice.push(attendHead(qkv.q[h], qkv.k[h], qkv.v[h]));
    local.push(slice);
  }
  const base = Math.floor(seq / D);
  const extra = seq % D;
  const ranges = [];
  let cursor = 0;
  for (let d = 0; d < D; d++) {
    const n = base + (d < extra ? 1 : 0);
    ranges.push([cursor, cursor + n]);
    cursor += n;
  }
  const restored = ranges.map(([lo, hi]) => {
    const rows = [];
    for (let t = lo; t < hi; t++) {
      const at = [];
      for (let h = 0; h < H; h++) at.push(ref[h][t]);
      rows.push(at);
    }
    return rows;
  });
  return { ok: true, heads: headsText, devices: devicesText, per: String(per), ref, local, restored, ranges };
}

export function gatheredSeqLabel(row) {
  if (!row.ok || !row.ranges || !row.ranges.length) return 'refused';
  const end = row.ranges[row.ranges.length - 1][1] - 1;
  return `full sequence 0-${end}`;
}

export function ulyssesSentence(row) {
  if (!row.ok) return `${row.heads} % ${row.devices} = ${row.rem}`;
  return `${row.heads} / ${row.devices} = ${row.per}`;
}
