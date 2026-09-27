import { appliesTo, newtonSchulz, singularValues } from './math.js';

let fail = 0;
const ok = (c, m) => { if (c) console.log('ok ' + m); else { fail++; console.error('FAIL ' + m); } };

ok(appliesTo('matrix') === true, 'a matrix gradient takes Muon');
ok(appliesTo('embedding') === false, 'embeddings stay on the second optimizer');
ok(appliesTo('lm_head') === false, 'the output head stays on the second optimizer');
ok(appliesTo('gain') === false, 'per-channel gains stay on the second optimizer');

const G = [[2, 0], [0, 0.5]];
const before = singularValues(G);
const after = singularValues(newtonSchulz(G, 6));
ok(after.every((v, i) => Math.abs(v - 1) < Math.abs(before[i] - 1)), 'each singular value moves toward 1 (' + before.join(',') + ' -> ' + after.join(',') + ')');
ok(after.every((v) => Math.abs(v - 1) < 0.1), 'six iterations land both singular values near 1');

if (fail) { console.error(fail + ' failed'); process.exit(1); }
console.log('PASS muon');
