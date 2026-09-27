import { fwht, l2, maxAbs, activationOnlyDelta } from './math.js';

let fail = 0;
const ok = (c, m) => { if (c) console.log('ok ' + m); else { fail++; console.error('FAIL ' + m); } };

const x = [8, 0, 0, 0];
const y = fwht(x);
ok(Math.abs(l2(y) - l2(x)) < 1e-9, 'the rotation leaves the L2 norm unchanged');
ok(maxAbs(y) < maxAbs(x), 'the same rotation shrinks the max (' + maxAbs(x) + ' -> ' + maxAbs(y) + ')');

const W = [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]];
ok(activationOnlyDelta(W, x) > 1e-6, 'rotating only the activation changes the product');

if (fail) { console.error(fail + ' failed'); process.exit(1); }
console.log('PASS rotation-quant');
