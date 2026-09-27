import { loraParams, fullParams, loraProduct, rankOf, extraMacs } from './math.js';

let fail = 0;
const ok = (c, m) => { if (c) console.log('ok ' + m); else { fail++; console.error('FAIL ' + m); } };

const din = 8, dout = 8, r = 2;
ok(loraParams(din, dout, r) === r * (din + dout), 'rank collapses the parameter count');
ok(fullParams(din, dout) === din * dout, 'full matrix is din*dout');
ok(loraParams(din, dout, r) < fullParams(din, dout), 'r=2 stores fewer than the full matrix');

const B = Array.from({ length: dout }, (_, i) => [((i % 3) - 1) * 0.5, (i % 5) * 0.25]);
const A = [[1, 0, -1, 0.5, 0, 0, 1, -0.5], [0, 1, 0.25, -1, 0.5, 0, 0, 1]];
const W = loraProduct(B, A);
ok(rankOf(W) <= r, 'BA rank is at most r');
const eye = Array.from({ length: dout }, (_, i) => Array.from({ length: din }, (_, j) => (i === j ? 1 : 0)));
ok(rankOf(eye) > r, 'a full-rank target is unrepresentable at this rank');

ok(extraMacs(din, dout, r, 4, true) === 0, 'merged adapter adds no serving GEMM');
ok(extraMacs(din, dout, r, 4, false) === 4 * r * (din + dout), 'unmerged adapter pays r*(din+dout) per token');

if (fail) { console.error(fail + ' failed'); process.exit(1); }
console.log('PASS lora');
