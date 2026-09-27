import { attentionError } from './math.js';

let fail = 0;
const ok = (c, m) => { if (c) console.log('ok ' + m); else { fail++; console.error('FAIL ' + m); } };

const Q = [[0.3, -1.7, 0.4]];
const K = [[0.2, 0.5, -0.7], [1.1, -0.4, 0.2], [-0.6, 0.9, 0.3]];
const V = [[0.6, 0.1], [-0.3, 0.9], [0.4, -0.2]];
const err = attentionError(Q, K, V);
ok(err > 1e-4, 'int8 scores disagree with exact attention (' + err + ')');

const Z = [[0, 0, 0]];
const K0 = [[0, 0, 0], [0, 0, 0], [0, 0, 0]];
ok(attentionError(Z, K0, V) === 0, 'scores already on the grid stay exact');

if (fail) { console.error(fail + ' failed'); process.exit(1); }
console.log('PASS sage-attention');
