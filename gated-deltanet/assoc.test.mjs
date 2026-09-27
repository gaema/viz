import { assocCost, cheaper } from './assoc.js';

let fail = 0;
const ok = (c, m) => { if (c) console.log('ok ' + m); else { fail++; console.error('FAIL ' + m); } };

const d = 16;
ok(assocCost('qk', d, d) === assocCost('kv', d, d), 'the two parenthesisations cost the same at L = d');
ok(cheaper(8, d) === 'qk', 'a shorter sequence prefers (QK)V');
ok(cheaper(32, d) === 'kv', 'a longer sequence prefers Q(KV)');
ok(cheaper(d, d) === 'tie', 'they tie when the length equals the dimension');

if (fail) { console.error(fail + ' failed'); process.exit(1); }
console.log('PASS assoc');
