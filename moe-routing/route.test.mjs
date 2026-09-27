import { route } from './route.js';

let fail = 0;
const ok = (c, m) => { if (c) console.log('ok ' + m); else { fail++; console.error('FAIL ' + m); } };

function logits(rows) {
  const N = rows.length, E = rows[0].length;
  const data = new Float32Array(N * E);
  rows.forEach((row, i) => row.forEach((v, e) => { data[i * E + e] = v; }));
  return { data, rows: N, cols: E };
}

const even = route(logits([[8, 0, 0, 0], [0, 8, 0, 0], [0, 0, 8, 0], [0, 0, 0, 8]]), 4, 4, 1, 4, 3);
ok(Math.abs(even.aux - 1) < 1e-6, 'uniform dispatch with no drops has aux 1 (' + even.aux + ')');
ok(even.drops === 0, 'the uniform case drops nothing');

const hot = route(logits([[8, 0], [8, 0], [8, 0], [0, 8]]), 4, 2, 1, 1, 3);
ok(hot.drop[0] === 2, 'expert 0 drops two of its three assignments');
ok(Math.abs(hot.f[0] - 3 / 4) < 1e-9, 'f counts the dropped assignments (' + hot.f[0] + ')');
ok(Math.abs(hot.f[0] - hot.load[0] / hot.assigned) > 1e-9, 'kept-only f would hide the drops');

const mix = route(logits([[2, 0, 0]]), 1, 3, 2, 4, 0);
let selSum = 0;
for (const e of mix.sel[0]) selSum += mix.g.data[e];
ok(Math.abs(selSum - 1) > 1e-3, 'top-k does not renormalize the softmax (' + selSum + ')');

if (fail) { console.error(fail + ' failed'); process.exit(1); }
console.log('PASS route');
