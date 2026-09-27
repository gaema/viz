import { recomputeCost } from './math.js';

let fail = 0;
const ok = (c, m) => { if (c) console.log('ok ' + m); else { fail++; console.error('FAIL ' + m); } };

const n = 16;
const storeAll = recomputeCost(n, 1);
const storeNone = recomputeCost(n, n);
const mid = recomputeCost(n, 4);
ok(storeAll.extraPerLayer === 0, 'store-all pays no extra forward');
ok(storeNone.memory < storeAll.memory, 'store-none holds fewer activations');
ok(storeNone.extraPerLayer > storeAll.extraPerLayer, 'store-none pays the extra forward');
ok(mid.objective < storeAll.objective && mid.objective < storeNone.objective, 'the objective has a minimum between the two ends');

if (fail) { console.error(fail + ' failed'); process.exit(1); }
console.log('PASS activation-recompute');
