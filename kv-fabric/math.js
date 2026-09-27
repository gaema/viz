// A private cache hits only when the assigned machine has already seen
// the prefix. A shared pool hits when any machine has. Arrivals are one
// wave per repeat: prefix p in wave r goes to machine (p + r) mod machines.
// A printed percent is 100 times the printed hit count over the printed
// request count.

export function poolHits(nPrefix, nMachine, repeats) {
  const P = Math.max(1, nPrefix | 0);
  const M = Math.max(1, nMachine | 0);
  const R = Math.max(1, repeats | 0);
  const seenAny = new Array(P).fill(false);
  const seenHere = Array.from({ length: M }, () => new Array(P).fill(false));
  let privateHits = 0;
  let sharedHits = 0;
  const rows = [];
  for (let r = 0; r < R; r++) {
    for (let p = 0; p < P; p++) {
      const machine = (p + r) % M;
      const privateHit = seenHere[machine][p];
      const sharedHit = seenAny[p];
      if (privateHit) privateHits += 1;
      if (sharedHit) sharedHits += 1;
      rows.push({ prefix: p, machine, wave: r, privateHit, sharedHit });
      seenHere[machine][p] = true;
      seenAny[p] = true;
    }
  }
  return { privateHits, sharedHits, requests: P * R, rows };
}

export function hitPct(hits, requests) {
  const h = String(hits | 0);
  const n = String(requests | 0);
  const pct = Number(n) === 0 ? '0.0' : ((100 * Number(h)) / Number(n)).toFixed(1);
  return { hits: h, requests: n, pct };
}
